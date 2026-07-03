"""Dropless sparse Mixture-of-Experts via grouped GEMM (+ optional expert parallelism).

This module provides a *numerically-equivalent* replacement for the dense
"compute-every-expert" einsum path used by the ``MoEFeedForward`` blocks in the
``qwen3``, ``qwen3_5`` and ``qwen3_vl`` model families.

The dense reference computes, for every token ``t`` and *every* expert ``e``::

    y_t = sum_{e in topk(t)}  w_{t,e} * down_e( silu(gate_e(x_t)) * up_e(x_t) )

wasting ~E/k of the expert FLOPs (E experts, k active per token). The grouped
path instead *permutes* the (token, chosen-expert) pairs so that all rows routed
to a given expert are contiguous, runs a **grouped GEMM** (one dense matmul per
expert over its own token group via :func:`jax.lax.ragged_dot` /
``tokamax.ragged_dot``), then *unpermutes* and does the weighted top-k sum. Only
``k * (B*T)`` rows of expert work are performed instead of ``E * (B*T)``.

Correctness derivation (why grouped == dense, up to fp reduction-order):
  * Routing (softmax -> top_k -> optional renorm) is done identically by the
    caller and passed in as ``topk_idx`` / ``topk_weights``; we do NOT change it.
  * Expand each token into its ``k`` (token, expert) assignments. This yields
    ``N = B*T*k`` rows, row ``i`` belonging to token ``tok[i]`` and expert
    ``exp[i]`` with weight ``w[i]``.
  * ``argsort(exp)`` produces a *permutation* ``perm`` (a bijection on the N
    rows). Gathering the token hidden states by ``tok[perm]`` groups all rows of
    expert 0 first, then expert 1, ... The per-expert counts are ``group_sizes``.
  * ``ragged_dot(x_sorted, W, group_sizes)`` applies expert ``e``'s weight to
    exactly the rows in its group -> algebraically identical to indexing
    ``W[exp[i]]`` for each row ``i`` (same operands, same matmul), so grouped ==
    dense per row.
  * Scatter the results back with the inverse permutation, multiply by ``w[i]``,
    and sum the ``k`` rows belonging to each token. This is the same weighted sum
    as the dense gather path.
  * Under expert parallelism the ``ragged_all_to_all`` dispatch/combine is a
    *pure data reshuffle* (a bijection routing each row to the device that owns
    its expert and back). It moves rows between devices but never changes which
    weight multiplies which row, so the math is unchanged.

Dropless: every (token, expert) pair is processed; there is no capacity factor
and no dropped or zero-padded tokens (``group_sizes`` sum to exactly ``N``).

Roundoff: results differ from the dense path only by floating-point
reduction order (grouped GEMM accumulates over a group; the einsum accumulates
over the ``E`` axis), i.e. ~fp epsilon, identical in fp32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tokamax


def _unshard(x):
    """Reshard an array to fully replicated (no sharding along any axis).

    The permute/sort/grouped-GEMM core operates on the flattened token axis, which
    must be unsharded for ``argsort`` / ``ragged_dot``. Under an all-replicated
    (single-device) mesh this is a no-op; with data/tensor sharding it gathers.
    """
    from jax.sharding import PartitionSpec as P, get_abstract_mesh

    try:
        from jax.sharding import reshard

        if get_abstract_mesh().empty:
            return x
        return reshard(x, P(*(None,) * x.ndim))
    except Exception:
        return x


def _auto(f):
    """Run ``f`` in full auto-sharding mode when an explicit-sharding mesh is active.

    On CPU, ``jax.lax.ragged_dot_general`` has no explicit-sharding (VMA)
    propagation rule in JAX 0.9.2, so under an Explicit-axis mesh we must drop the
    grouped-GEMM core into auto mode. Inputs are already gathered to replicated; the
    output is returned replicated (``P()``) and the caller reshards it.

    On GPU/TPU we do NOT wrap: the tokamax ragged_dot Pallas kernel (and the native
    ragged_dot lowering) propagate sharding fine, and wrapping in ``auto_axes``
    triggers an Auto/Explicit mesh-type mismatch in tokamax's ``GroupSizes``
    constant creation. Empty mesh (single-device eager) is also a plain call.
    """
    from jax.sharding import PartitionSpec as P, get_abstract_mesh

    mesh = get_abstract_mesh()
    if mesh.empty or jax.default_backend() != "cpu":
        return f
    axes = tuple(mesh.axis_names)
    try:
        from jax.sharding import auto_axes

        return auto_axes(f, axes=axes, out_sharding=P())
    except Exception:
        return f


def _ragged_dot(lhs, rhs, group_sizes, primitive: str, *, group_offset=None):
    """Grouped GEMM dispatch.

    lhs: (N, K) sorted rows. rhs: (E, K, M) stacked expert weights.
    group_sizes: (E,) int32. Returns (N, M).

    ``primitive="tokamax"`` uses ``tokamax.ragged_dot`` (auto-selects the best
    backend; guaranteed to work on CPU/GPU). ``primitive="jax"`` uses
    ``jax.lax.ragged_dot`` (the CPU-correct reference).

    NOTE (Phase-2 / GPU caveat): ``group_offset`` lets a device compute only a
    slice of the experts (needed to *shard* expert compute on-device). It is
    honored on CPU, but ``jax.lax.ragged_dot``'s ``group_offset`` is currently
    *unimplemented on GPU*. Our EP path avoids relying on it: each device holds a
    contiguous block of experts as ``rhs`` of size ``E/EP`` and calls ragged_dot
    with the default offset over its local experts, so no GPU group_offset is
    required.

    fp32 accumulation is mandatory: the bf16 tokamax ``PallasTritonRaggedDot`` VJP
    overflows the down_proj expert grad to the bf16 ceiling (3.39e38). Verified by
    adversarial repro 2026-07-03 (53 NaN events across seeds guard-off, 0 guard-on)
    on tokamax@nested-mask/A100; re-test with a proper multi-seed sweep before dropping.
    """
    out_dtype = jnp.result_type(lhs, rhs)
    lhs = lhs.astype(jnp.float32)
    rhs = rhs.astype(jnp.float32)
    if primitive == "tokamax":
        out = tokamax.ragged_dot(lhs, rhs, group_sizes, group_offset=group_offset)
    else:
        out = jax.lax.ragged_dot(lhs, rhs, group_sizes, group_offset=group_offset)
    return out.astype(out_dtype)


# ``jax.lax.ragged_all_to_all`` lowers to an HLO that XLA:CPU does not implement
# (only GPU/TPU); on CPU we emulate it bit-identically via ``all_gather`` (a pure,
# bijective data reshuffle) so EP can be exercised in CPU CI.


def _ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name
):
    """Ragged all-to-all with a CPU-portable fallback.

    Semantics (per calling shard s, per destination d along ``axis_name``):
      send_sizes[d]/input_offsets[d]  : rows of ``operand`` s sends to d.
      output_offsets[d]               : offset in d's ``output`` for s's block.
      recv_sizes[d]                   : rows s receives from d.
    Rows of ``output`` not written by any sender stay at their initial (0) value.
    """
    if jax.default_backend() != "cpu":
        return jax.lax.ragged_all_to_all(
            operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
            axis_name=axis_name,
        )

    # --- CPU emulation via all_gather (bijective, numerically identical) ---
    ep = jax.lax.psum(1, axis_name)
    src_cap = operand.shape[0]
    # Gather every sender's operand and metadata; index axis 0 by source shard.
    all_operand = jax.lax.all_gather(operand, axis_name, axis=0, tiled=False)  # (ep, src_cap, ...)
    all_in_off = jax.lax.all_gather(input_offsets, axis_name, axis=0, tiled=False)  # (ep, ep)
    all_send = jax.lax.all_gather(send_sizes, axis_name, axis=0, tiled=False)  # (ep, ep)
    all_out_off = jax.lax.all_gather(output_offsets, axis_name, axis=0, tiled=False)  # (ep, ep)
    me = jax.lax.axis_index(axis_name)

    out = output
    out_cap = output.shape[0]
    out_rows = jnp.arange(out_cap)
    # For each source s, copy the block it addressed to ME into my output buffer.
    for s in range(ep):
        in_start = all_in_off[s, me]      # start row in source s's operand
        n = all_send[s, me]               # rows sent from s to me
        out_start = all_out_off[s, me]    # where they go in MY buffer
        src_rows = all_operand[s]         # (src_cap, ...)
        # Gather rows [in_start : in_start+n] (bounded), aligned to out positions.
        # Build, for each of my output rows r, the source index (in_start + (r-out_start))
        # when out_start <= r < out_start+n, else clamp+mask.
        rel = out_rows - out_start
        valid = (rel >= 0) & (rel < n)
        gather_idx = jnp.clip(in_start + rel, 0, src_cap - 1)
        gathered = jnp.take(src_rows, gather_idx, axis=0)  # (out_cap, ...)
        mask_shape = (out_cap,) + (1,) * (out.ndim - 1)
        out = jnp.where(valid.reshape(mask_shape), gathered, out)
    return out


def _sort_tokens_by_expert(topk_idx_Nk: jax.Array, num_experts: int):
    """Build the permutation that groups (token, expert) rows by expert id.

    Args:
      topk_idx_Nk: (T, k) int expert ids, T = number of tokens.
      num_experts: E.

    Returns:
      sort_perm: (T*k,) int32 permutation that sorts flattened rows by expert id
        (stable, so within an expert rows keep flattened order).
      inv_perm: (T*k,) int32 inverse permutation.
      group_sizes: (E,) int32 count of rows per expert (sums to T*k).
      expert_of_row: (T*k,) int32 expert id of each *flattened* (pre-sort) row.
    """
    flat_expert = topk_idx_Nk.reshape(-1).astype(jnp.int32)  # (T*k,)
    # Stable argsort groups rows by expert while preserving order within a group.
    sort_perm = jnp.argsort(flat_expert, stable=True).astype(jnp.int32)
    inv_perm = jnp.argsort(sort_perm).astype(jnp.int32)
    group_sizes = jnp.bincount(flat_expert, length=num_experts).astype(jnp.int32)
    return sort_perm, inv_perm, group_sizes, flat_expert


# ---------------------------------------------------------------------------
# Per-expert LoRA on the grouped path.
# ---------------------------------------------------------------------------
#
# The dense MoE path adds, per expert e and per token x, the low-rank correction
# ``scaling * (x @ A[e]) @ B[e]`` INSIDE the expert einsum (before SiLU/top-k for
# gate/up; on the activation for down). See LoRAMoEExperts.delta_shared /
# delta_per_expert in omegalax/trainers/lora.py.
#
# In the grouped path the rows are already permuted into per-expert contiguous
# groups with ``group_sizes``, so the SAME per-expert delta is exactly two extra
# grouped GEMMs: ``ragged_dot(sorted_rows, A, group_sizes)`` then
# ``ragged_dot(mid, B, group_sizes)`` (A[e] then B[e] applied to each group),
# scaled and added to the corresponding base grouped output. This is
# algebraically identical to the dense per-expert LoRA (same operands, same
# per-expert matmuls), so grouped-LoRA == dense-LoRA up to fp reduction order.
#
# ``_lora_arrays`` pulls the raw (A, B, scaling) out of a ``LoRAMoEExperts``
# module BEFORE entering the auto/shard_map transformed core, so the core only
# ever sees plain JAX arrays (never an nnx.Module).


def _lora_arrays(adapter, dtype):
    """Return ``(A, B, scaling)`` cast to ``dtype`` for a LoRAMoEExperts adapter,
    or ``None`` when no adapter is attached. A: (E, in, r); B: (E, r, out)."""
    if adapter is None:
        return None
    a = adapter.lora_A[...].astype(dtype)
    b = adapter.lora_B[...].astype(dtype)
    return a, b, float(adapter.scaling)


def _unshard_lora(lora):
    """Gather a ``(A, B, scaling)`` LoRA tuple to replicated (matching the base
    weights) so it is consistent inside the auto-sharding grouped core. No-op for
    ``None``."""
    if lora is None:
        return None
    a, b, scaling = lora
    return _unshard(a), _unshard(b), scaling


def _grouped_lora_delta(sorted_in, lora, group_sizes):
    """Per-expert LoRA delta for grouped rows: ``scaling * (X @ A[e]) @ B[e]``,
    applied per group. ``sorted_in`` is (N, in); returns (N, out) or 0.0.

    The two low-rank grouped GEMMs always use the ``jax`` primitive
    (``jax.lax.ragged_dot``, the CPU-correct reference) regardless of the base
    path's ``primitive``: tokamax's ``ragged_dot`` custom kernel does not compose
    with the auto-sharding intermediate produced by the first (A) grouped GEMM.
    The adapters are rank-r (tiny), so the perf cost of the jax path is
    negligible and the delta is numerically identical."""
    if lora is None:
        return 0.0
    a, b, scaling = lora
    mid = _ragged_dot(sorted_in, a, group_sizes, "jax")  # (N, r)
    delta = _ragged_dot(mid, b, group_sizes, "jax")  # (N, out)
    return delta * scaling


def grouped_moe(
    hidden_ND: jax.Array,
    topk_idx_Nk: jax.Array,
    topk_weights_Nk: jax.Array,
    gate_EDF: jax.Array,
    up_EDF: jax.Array,
    down_EFD: jax.Array,
    *,
    num_experts: int,
    primitive: str = "tokamax",
    gate_lora=None,
    up_lora=None,
    down_lora=None,
) -> jax.Array:
    """Grouped-GEMM dropless MoE, EP=1 (single logical expert shard).

    Args:
      hidden_ND:      (N, D) flattened token hidden states, N = B*T.
      topk_idx_Nk:    (N, k) selected expert ids per token.
      topk_weights_Nk:(N, k) combine weights per (token, slot).
      gate_EDF/up_EDF:(E, D, F) stacked expert in-projections.
      down_EFD:       (E, F, D) stacked expert out-projection.
      num_experts:    E.
      primitive:      "tokamax" (default, GPU-perf) or "jax" (reference).
      gate_lora/up_lora/down_lora: optional ``LoRAMoEExperts`` adapters whose
        per-expert low-rank delta is applied to the matching grouped output,
        matching the dense path's per-expert LoRA (no-op when None).

    Returns:
      (N, D) combined MoE output, equivalent to the dense gather path.
    """
    N, D = hidden_ND.shape
    k = topk_idx_Nk.shape[1]
    compute_dtype = hidden_ND.dtype

    # Pull raw (A, B, scaling) out of the adapter modules here, so the transformed
    # core only sees plain arrays. None when no adapter is attached. Unshard A/B to
    # replicated just like the base weights so the auto-sharding core is consistent.
    gate_lora_a = _unshard_lora(_lora_arrays(gate_lora, compute_dtype))
    up_lora_a = _unshard_lora(_lora_arrays(up_lora, compute_dtype))
    down_lora_a = _unshard_lora(_lora_arrays(down_lora, compute_dtype))

    # When LoRA is attached, run the WHOLE grouped core on the jax ragged_dot
    # reference (both base and adapter GEMMs). The base and adapter GEMMs share
    # intermediates inside one auto-sharding region, and mixing tokamax's custom
    # ragged_dot kernel with the jax reference there produces an Auto/Explicit
    # mesh-type mismatch. LoRA is a fine-tuning (not perf-critical pretraining)
    # path, so the reference primitive throughout is the correct, consistent
    # choice; numerically identical to tokamax up to fp reduction order.
    if gate_lora_a is not None:
        primitive = "jax"

    # Gather to replicated so the sort/grouped-GEMM core (auto-sharding region) sees
    # unsharded operands; the caller reshards the replicated result afterwards.
    hidden_ND = _unshard(hidden_ND)
    topk_idx_Nk = _unshard(topk_idx_Nk)
    topk_weights_Nk = _unshard(topk_weights_Nk)
    gate_EDF = _unshard(gate_EDF)
    up_EDF = _unshard(up_EDF)
    down_EFD = _unshard(down_EFD)

    # Pass the LoRA A/B *arrays* as explicit ``_core`` operands (not closure
    # captures) so ``_auto``/``auto_axes`` converts them to Auto mode along with
    # the base weights — a closure-captured Explicit-mesh array would clash with
    # the auto-sharding core's mesh type inside tokamax's ragged_dot. Scaling is a
    # static float, kept as a closure constant. ``lora_arrays`` is a flat tuple of
    # 0 or 6 arrays (gate_A, gate_B, up_A, up_B, down_A, down_B).
    if gate_lora_a is not None:
        lora_arrays = (
            gate_lora_a[0], gate_lora_a[1],
            up_lora_a[0], up_lora_a[1],
            down_lora_a[0], down_lora_a[1],
        )
        gate_scaling, up_scaling, down_scaling = (
            gate_lora_a[2], up_lora_a[2], down_lora_a[2]
        )
    else:
        lora_arrays = ()
        gate_scaling = up_scaling = down_scaling = 1.0

    def _core(hidden_ND, topk_idx_Nk, topk_weights_Nk, gate_EDF, up_EDF, down_EFD, *lora):
        sort_perm, inv_perm, group_sizes, _ = _sort_tokens_by_expert(
            topk_idx_Nk, num_experts
        )
        # Row i of the expanded (N*k) problem belongs to token (i // k). Gather the
        # token hidden states for each (token, expert) row, then sort by expert.
        token_of_row = jnp.arange(N * k, dtype=jnp.int32) // k  # (N*k,)
        rows_ND = hidden_ND[token_of_row]  # (N*k, D)
        sorted_rows_ND = rows_ND[sort_perm]  # grouped by expert

        # Grouped GEMM: gate & up (D->F), SiLU-gate, then down (F->D).
        gate_out = _ragged_dot(sorted_rows_ND, gate_EDF, group_sizes, primitive)
        up_out = _ragged_dot(sorted_rows_ND, up_EDF, group_sizes, primitive)
        act_pre = None
        if lora:
            gA, gB, uA, uB, dA, dB = lora
            # Per-expert LoRA on gate/up: same per-expert delta as the dense path.
            gate_out = gate_out + _grouped_lora_delta(
                sorted_rows_ND, (gA, gB, gate_scaling), group_sizes
            )
            up_out = up_out + _grouped_lora_delta(
                sorted_rows_ND, (uA, uB, up_scaling), group_sizes
            )
        act = jax.nn.silu(gate_out) * up_out  # (N*k, F)
        down_out = _ragged_dot(act, down_EFD, group_sizes, primitive)  # (N*k, D)
        if lora:
            down_out = down_out + _grouped_lora_delta(
                act, (dA, dB, down_scaling), group_sizes
            )

        # Unpermute back to flattened (token, slot) order, weight, sum over k.
        unsorted = down_out[inv_perm]  # (N*k, D)
        weighted = unsorted * topk_weights_Nk.reshape(N * k, 1).astype(compute_dtype)
        return weighted.reshape(N, k, D).sum(axis=1)

    return _auto(_core)(
        hidden_ND, topk_idx_Nk, topk_weights_Nk, gate_EDF, up_EDF, down_EFD, *lora_arrays
    )


# ---------------------------------------------------------------------------
# Phase 2: expert parallelism via ragged all-to-all.
# ---------------------------------------------------------------------------
#
# jax.lax.ragged_all_to_all(operand, output, input_offsets, send_sizes,
#                           output_offsets, recv_sizes, *, axis_name):
#   For the calling shard s (a device along `axis_name` of size EP), and for each
#   destination shard d in [0, EP):
#     * send_sizes[d]    : number of rows shard s sends to shard d.
#     * input_offsets[d] : start row in shard s's `operand` of that block.
#     * output_offsets[d]: start row **in shard d's `output` buffer** where s's
#                          block is written (the "transposed"/receiver-side offset).
#     * recv_sizes[d]    : number of rows shard s receives from shard d.
#   `output` is a zero-initialized capacity buffer; rows not written by any sender
#   stay zero. The op is a pure gather/scatter across devices (a bijection on the
#   dispatched rows), so it changes *where* rows live, never the per-row math.
#
# Layout: tokens are sharded across `expert_axis` (each device owns N/EP tokens
# with all k of their slots), and the stacked expert weights are sharded across
# `expert_axis` (each device owns E/EP contiguous experts). Dispatch sends each
# (token,slot) row to the device owning its expert; combine sends the result back.


def _lora_ep_delta(sorted_in, a_pad, b_pad, scaling, gs_padded):
    """Per-(local-)expert LoRA delta on the EP device-local grouped rows:
    ``scaling * (X @ A[e]) @ B[e]`` applied per padded local-expert group. Returns
    0.0 when the adapter is absent. ``a_pad``/``b_pad`` are the local expert A/B
    stacks padded with a trailing zero dummy expert (matching ``gs_padded``)."""
    if a_pad is None:
        return 0.0
    # Use the jax ragged_dot reference for the low-rank LoRA GEMMs (see
    # _grouped_lora_delta); the EP path already defaults primitive="jax" anyway.
    mid = _ragged_dot(sorted_in, a_pad, gs_padded, "jax")  # (cap, r)
    delta = _ragged_dot(mid, b_pad, gs_padded, "jax")  # (cap, out)
    return delta * scaling


def grouped_moe_ep(
    hidden_ND: jax.Array,
    topk_idx_Nk: jax.Array,
    topk_weights_Nk: jax.Array,
    gate_EDF: jax.Array,
    up_EDF: jax.Array,
    down_EFD: jax.Array,
    *,
    num_experts: int,
    expert_axis: str = "expert",
    capacity_factor: float = 4.0,
    primitive: str = "jax",
    gate_lora=None,
    up_lora=None,
    down_lora=None,
) -> jax.Array:
    """Expert-parallel dropless MoE.

    Defaults to ``primitive="jax"``: tokamax's ``ragged_dot`` registers a
    ``custom_vjp`` whose transpose does not currently compose with ``shard_map``'s
    backward (it feeds a ``float0`` integer-tangent through the sharded in_specs).
    ``jax.lax.ragged_dot`` uses standard autodiff and differentiates cleanly under
    ``shard_map``. The GPU-perf tokamax kernel for the EP path is deferred with the
    rest of the multi-GPU perf work.

    When no expert mesh axis is active (size 1), this transparently falls back to
    the single-device :func:`grouped_moe`. Otherwise it shards the stacked expert
    weights on ``expert_axis`` and dispatches tokens to the owning device with
    :func:`jax.lax.ragged_all_to_all`, runs the grouped GEMM locally, and combines
    the results back with the inverse all-to-all.

    ``capacity_factor`` sizes the fixed per-device receive buffer as
    ``ceil(capacity_factor * N * k / EP)`` rows. This padding is a *compilation*
    requirement of ragged_all_to_all (static shapes) and does NOT drop or alter
    tokens: unused rows are zeros and are never read back (the combine copies back
    exactly the rows that were dispatched). The MoE remains dropless as long as no
    single device receives more than the buffer holds; the factor is chosen large
    enough (default 4x the mean) to cover routing imbalance in tests.
    """
    from jax import shard_map
    from jax.sharding import PartitionSpec as P, get_abstract_mesh

    mesh = get_abstract_mesh()
    ep = 1
    if not mesh.empty and expert_axis in mesh.axis_names:
        ep = int(mesh.shape[expert_axis])
    if ep == 1:
        return grouped_moe(
            hidden_ND, topk_idx_Nk, topk_weights_Nk,
            gate_EDF, up_EDF, down_EFD,
            num_experts=num_experts, primitive=primitive,
            gate_lora=gate_lora, up_lora=up_lora, down_lora=down_lora,
        )

    assert num_experts % ep == 0, f"E={num_experts} not divisible by EP={ep}"
    N, D = hidden_ND.shape
    k = topk_idx_Nk.shape[1]
    assert N % ep == 0, f"N={N} tokens not divisible by EP={ep} (shard the token axis)"
    experts_per_shard = num_experts // ep
    n_local = N // ep  # tokens per device
    local_Nk = n_local * k  # expanded rows produced per device
    compute_dtype = hidden_ND.dtype

    # Pull raw (A, B, scaling) out of the adapters. The stacked A/B ride the
    # expert axis exactly like the base gate/up/down weights (P(expert_axis,...)),
    # so shard_map hands each device its own local expert block. ``None`` => no
    # adapter (identity). We pass A/B as extra shard_map operands (or a zero-sized
    # placeholder when absent, since shard_map operands must be concrete arrays).
    _has_lora = any(x is not None for x in (gate_lora, up_lora, down_lora))
    gate_lora_a = _lora_arrays(gate_lora, compute_dtype)
    up_lora_a = _lora_arrays(up_lora, compute_dtype)
    down_lora_a = _lora_arrays(down_lora, compute_dtype)
    # Per-device receive capacity (static). Padding is a compile-time requirement of
    # ragged_all_to_all; unused rows stay zero and are never combined back.
    cap = int(-(-int(capacity_factor * N * k) // ep))

    def per_device(hidden_nd, idx_nk, w_nk, gate_w, up_w, down_w, *lora_ops):
        # lora_ops (when present) = (gate_A, gate_B, up_A, up_B, down_A, down_B),
        # each the device-local expert block of the corresponding stacked adapter.
        # --- expand this device's local tokens into (token, slot) rows ---
        token_of_row = jnp.arange(local_Nk, dtype=jnp.int32) // k
        rows = hidden_nd[token_of_row]  # (local_Nk, D)
        flat_expert = idx_nk.reshape(local_Nk).astype(jnp.int32)
        flat_w = w_nk.reshape(local_Nk).astype(compute_dtype)

        # --- sort local rows by GLOBAL expert id: blocks per destination device
        #     become contiguous (destinations are contiguous expert blocks) ---
        sort_perm = jnp.argsort(flat_expert, stable=True).astype(jnp.int32)
        inv_perm = jnp.argsort(sort_perm).astype(jnp.int32)
        rows_sorted = rows[sort_perm]
        gsz_E = jnp.bincount(flat_expert, length=num_experts).astype(jnp.int32)

        # rows this device sends to destination d = sum over d's experts.
        send_sizes = gsz_E.reshape(ep, experts_per_shard).sum(axis=1).astype(jnp.int32)
        input_offsets = _excl_cumsum(send_sizes)

        my = jax.lax.axis_index(expert_axis)
        # send_matrix[src, dst] = rows src sends to dst  (gather everyone's send row)
        send_matrix = jax.lax.all_gather(send_sizes, expert_axis, axis=0, tiled=False)
        recv_sizes = send_matrix[:, my].astype(jnp.int32)  # from each src -> me
        # For destination dst, exclusive cumsum over sources gives where each source's
        # block starts in dst's buffer:  recv_starts_by_src[src, dst].
        recv_starts_by_src = (jnp.cumsum(send_matrix, axis=0) - send_matrix).astype(jnp.int32)
        # output_offsets[dst] (my view): offset in dst's buffer for MY block.
        output_offsets = recv_starts_by_src[my, :].astype(jnp.int32)  # (ep,) over dst
        # recv_starts[src] = where src's block lands in MY buffer.
        recv_starts = recv_starts_by_src[:, my].astype(jnp.int32)  # (ep,) over src
        # For the combine, each source's block must be written back at the offset
        # that source originally read it from (its own input_offsets[me]).
        all_input_offsets = jax.lax.all_gather(
            input_offsets, expert_axis, axis=0, tiled=False
        )  # (src, dst): source src's read offset for its block to dst
        combine_output_offsets = all_input_offsets[:, my].astype(jnp.int32)  # per src -> me's slot in src

        # --- dispatch rows to the device owning their expert ---
        recv_buf = jnp.zeros((cap, D), compute_dtype)
        dispatched = _ragged_all_to_all(
            rows_sorted, recv_buf, input_offsets, send_sizes, output_offsets, recv_sizes,
            axis_name=expert_axis,
        )

        # Arrivals are ordered [src0 block | src1 block | ...]; each source block is
        # internally sorted by (global==local, since we own a contiguous block)
        # expert. Re-sort arrivals to local-expert-major so one ragged_dot covers all
        # local experts. Counts of local expert j arriving from source s:
        gsz_matrix = jax.lax.all_gather(
            gsz_E.reshape(ep, experts_per_shard), expert_axis, axis=0, tiled=False
        )  # (src, dst, local_expert)
        my_local_counts = gsz_matrix[:, my, :]  # (ep, experts_per_shard) rows from each src
        arrival_le = _arrival_local_expert_ids(my_local_counts, recv_starts, cap, experts_per_shard)
        reorder = jnp.argsort(arrival_le, stable=True).astype(jnp.int32)
        inv_reorder = jnp.argsort(reorder).astype(jnp.int32)
        rows_local = dispatched[reorder]  # (cap, D), local-expert-major then padding

        local_group_sizes = my_local_counts.sum(axis=0).astype(jnp.int32)  # (experts_per_shard,)
        pad = cap - jnp.sum(local_group_sizes)
        gs_padded = jnp.concatenate([local_group_sizes, pad.reshape(1)]).astype(jnp.int32)
        # Dummy trailing group maps the padding rows to a zero weight (discarded).
        gate_pad = jnp.concatenate([gate_w, jnp.zeros_like(gate_w[:1])], axis=0)
        up_pad = jnp.concatenate([up_w, jnp.zeros_like(up_w[:1])], axis=0)
        down_pad = jnp.concatenate([down_w, jnp.zeros_like(down_w[:1])], axis=0)

        # Pad each local LoRA A/B stack with the same trailing zero dummy expert so
        # padding rows contribute a zero delta (matches gate_pad/up_pad/down_pad).
        def _pad_expert(w):
            return jnp.concatenate([w, jnp.zeros_like(w[:1])], axis=0)

        if lora_ops:
            gA, gB, uA, uB, dA, dB = lora_ops
            gate_lora_pad = (_pad_expert(gA), _pad_expert(gB), gate_scaling)
            up_lora_pad = (_pad_expert(uA), _pad_expert(uB), up_scaling)
            down_lora_pad = (_pad_expert(dA), _pad_expert(dB), down_scaling)
        else:
            gate_lora_pad = up_lora_pad = down_lora_pad = (None, None, None)

        # --- grouped GEMM over LOCAL experts ---
        g_out = _ragged_dot(rows_local, gate_pad, gs_padded, primitive)
        u_out = _ragged_dot(rows_local, up_pad, gs_padded, primitive)
        # Per-local-expert LoRA on gate/up (same per-expert delta as the dense path).
        g_out = g_out + _lora_ep_delta(rows_local, *gate_lora_pad, gs_padded)
        u_out = u_out + _lora_ep_delta(rows_local, *up_lora_pad, gs_padded)
        act = jax.nn.silu(g_out) * u_out
        d_out = _ragged_dot(act, down_pad, gs_padded, primitive)  # (cap, D)
        d_out = d_out + _lora_ep_delta(act, *down_lora_pad, gs_padded)

        # Undo local reorder, then combine (inverse all-to-all) back to the source.
        d_unreordered = d_out[inv_reorder]  # arrival order (source-major, at recv_starts)
        combine_buf = jnp.zeros((local_Nk, D), compute_dtype)
        # Combine: for each source ``s`` read its block from MY arrival buffer at
        # recv_starts[s] (size recv_sizes[s]) and write it back into s's buffer at
        # the offset s originally read it from (combine_output_offsets[s]). Rows I
        # receive back from each source d = send_sizes[d] (what I sent d).
        combined = _ragged_all_to_all(
            d_unreordered, combine_buf, recv_starts, recv_sizes,
            combine_output_offsets, send_sizes,
            axis_name=expert_axis,
        )  # back in this device's expert-sorted order
        # Unsort to (token, slot) order, weight, and sum over k.
        out_sorted = combined[inv_perm] * flat_w.reshape(local_Nk, 1)
        return out_sorted.reshape(n_local, k, D).sum(axis=1)  # (n_local, D)

    base_operands = (hidden_ND, topk_idx_Nk, topk_weights_Nk, gate_EDF, up_EDF, down_EFD)
    base_specs = (
        P(expert_axis, None),  # hidden tokens sharded on expert axis
        P(expert_axis, None),  # topk idx
        P(expert_axis, None),  # topk weights
        P(expert_axis, None, None),  # gate experts sharded
        P(expert_axis, None, None),  # up
        P(expert_axis, None, None),  # down
    )
    if _has_lora:
        # Each adapter MUST be attached when any is (the model always injects all
        # three together); assert to avoid a silent shape mismatch.
        assert (
            gate_lora_a is not None and up_lora_a is not None and down_lora_a is not None
        ), "grouped_moe_ep: LoRA must be attached to all of gate/up/down or none."
        gate_scaling, up_scaling, down_scaling = (
            gate_lora_a[2], up_lora_a[2], down_lora_a[2]
        )
        # A: (E, in, r) sharded on the expert axis; B: (E, r, out) likewise.
        lora_operands = (
            gate_lora_a[0], gate_lora_a[1],
            up_lora_a[0], up_lora_a[1],
            down_lora_a[0], down_lora_a[1],
        )
        lora_specs = (P(expert_axis, None, None),) * 6
    else:
        gate_scaling = up_scaling = down_scaling = 1.0
        lora_operands = ()
        lora_specs = ()

    out = shard_map(
        per_device,
        mesh=mesh,
        in_specs=(*base_specs, *lora_specs),
        out_specs=P(expert_axis, None),  # output tokens sharded on expert axis
        check_vma=False,
    )(*base_operands, *lora_operands)
    return out


def _excl_cumsum(x):
    """Exclusive prefix sum of a 1D int array."""
    return (jnp.cumsum(x) - x).astype(jnp.int32)


def _arrival_local_expert_ids(my_local_counts, recv_starts, cap, experts_per_shard):
    """Local-expert id for each of the ``cap`` arrival rows (unused rows -> sentinel).

    Arrivals are laid out as ``[src0 block | src1 block | ...]``; source ``s``'s
    block starts at ``recv_starts[s]`` and internally lists local experts
    ``0..experts_per_shard-1`` with row counts ``my_local_counts[s, :]``. Returns a
    per-row local-expert id (``experts_per_shard`` sentinel for the trailing padding
    rows so they sort last, into the dummy group).
    """
    ep = my_local_counts.shape[0]
    ids = jnp.full((cap,), experts_per_shard, dtype=jnp.int32)
    row = jnp.arange(cap, dtype=jnp.int32)
    for s in range(ep):
        base = recv_starts[s]
        bounds = jnp.cumsum(my_local_counts[s])  # (experts_per_shard,)
        block_len = bounds[-1]
        local_pos = row - base
        in_block = (local_pos >= 0) & (local_pos < block_len)
        le = jnp.sum(local_pos[:, None] >= bounds[None, :], axis=1).astype(jnp.int32)
        ids = jnp.where(in_block, le, ids)
    return ids
