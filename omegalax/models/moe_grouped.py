"""Dropless sparse Mixture-of-Experts via grouped GEMM (+ optional expert parallelism).

Numerically-equivalent replacement for the dense compute-every-expert einsum, at
``k*(B*T)`` expert rows instead of ``E*(B*T)``: expand tokens to ``N=B*T*k``
(token,expert) rows, a stable ``argsort(expert)`` groups them by expert with counts
``group_sizes``, ``ragged_dot`` applies each expert's weights to exactly its rows,
and the inverse permutation + weighted top-k sum reproduces the dense gather. Under
EP the ``ragged_all_to_all`` dispatch/combine is a pure bijective reshuffle. Differs
from dense only by fp reduction order (identical in fp32).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tokamax


def _unshard(x):
    """Reshard to fully replicated: the sort/grouped-GEMM core needs an unsharded
    token axis for ``argsort`` / ``ragged_dot`` (no-op on a single-device mesh)."""
    from jax.sharding import PartitionSpec as P, get_abstract_mesh

    try:
        from jax.sharding import reshard

        if get_abstract_mesh().empty:
            return x
        return reshard(x, P(*(None,) * x.ndim))
    except Exception:
        return x


def _auto(f):
    """Wrap ``f`` in auto-sharding mode, but ONLY on CPU under an Explicit mesh.

    On CPU ``jax.lax.ragged_dot_general`` has no Explicit-sharding (VMA) rule (JAX
    0.9.2), so the core must run in auto mode (inputs already replicated, output
    returned ``P()`` and resharded by the caller). On GPU/TPU we must NOT wrap:
    tokamax's ragged_dot propagates sharding fine and ``auto_axes`` would trigger an
    Auto/Explicit mesh-type mismatch in its ``GroupSizes`` constant.
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
    """Grouped GEMM: lhs (N, K) sorted rows, rhs (E, K, M) stacked weights -> (N, M).
    ``primitive="tokamax"`` (best backend) or ``"jax"`` (CPU-correct reference).

    NOTE (GPU caveat): ``jax.lax.ragged_dot``'s ``group_offset`` is unimplemented
    on GPU, so the EP path never relies on it (each device holds its own contiguous
    ``E/EP`` expert block as ``rhs`` and uses the default offset).
    """
    out_dtype = jnp.result_type(lhs, rhs)
    # tokamax's ragged_dot custom VJP stores backward grads at the OPERAND dtype (it
    # discards the forward preferred_element_type; the accumulator is already fp32). In
    # bf16 the down_proj weight-grad can land in (bf16_max 3.39e38, fp32_max 3.40e38] and
    # overflow to inf on the bf16 store while the forward stays finite. preferred_element_type
    # =f32 / precision=HIGHEST are bit-identically broken (never reach the VJP); only fp32
    # operands => fp32 grad storage fix it (~1.6-2x on expert GEMMs; real fix is upstream:
    # decouple tokamax's backward grad-storage dtype from the operand dtype).
    lhs = lhs.astype(jnp.float32)
    rhs = rhs.astype(jnp.float32)
    if primitive == "tokamax":
        out = tokamax.ragged_dot(lhs, rhs, group_sizes, group_offset=group_offset)
    else:
        out = jax.lax.ragged_dot(lhs, rhs, group_sizes, group_offset=group_offset)
    return out.astype(out_dtype)


# XLA:CPU has no ragged_all_to_all, so on CPU we emulate it via all_gather for CI.


def _ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name
):
    """Ragged all-to-all with a CPU-portable fallback. Per calling shard s and
    destination d along ``axis_name``: s sends ``send_sizes[d]`` rows at
    ``input_offsets[d]`` to ``output_offsets[d]`` in d's buffer, and receives
    ``recv_sizes[d]`` from d; unwritten output rows keep their initial value."""
    if jax.default_backend() != "cpu":
        return jax.lax.ragged_all_to_all(
            operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
            axis_name=axis_name,
        )

    ep = jax.lax.psum(1, axis_name)
    src_cap = operand.shape[0]
    all_operand = jax.lax.all_gather(operand, axis_name, axis=0, tiled=False)  # (ep, src_cap, ...)
    all_in_off = jax.lax.all_gather(input_offsets, axis_name, axis=0, tiled=False)  # (ep, ep)
    all_send = jax.lax.all_gather(send_sizes, axis_name, axis=0, tiled=False)  # (ep, ep)
    all_out_off = jax.lax.all_gather(output_offsets, axis_name, axis=0, tiled=False)  # (ep, ep)
    me = jax.lax.axis_index(axis_name)

    out = output
    out_cap = output.shape[0]
    out_rows = jnp.arange(out_cap)
    # Each source s's block lands at out_start; row r gathers source row in_start+(r-out_start),
    # masked out outside the block.
    for s in range(ep):
        in_start = all_in_off[s, me]
        n = all_send[s, me]
        out_start = all_out_off[s, me]
        src_rows = all_operand[s]         # (src_cap, ...)
        rel = out_rows - out_start
        valid = (rel >= 0) & (rel < n)
        gather_idx = jnp.clip(in_start + rel, 0, src_cap - 1)
        gathered = jnp.take(src_rows, gather_idx, axis=0)  # (out_cap, ...)
        mask_shape = (out_cap,) + (1,) * (out.ndim - 1)
        out = jnp.where(valid.reshape(mask_shape), gathered, out)
    return out


def _sort_tokens_by_expert(topk_idx_Nk: jax.Array, num_experts: int):
    """Group (token, expert) rows by expert id via a stable argsort.

    Returns ``(sort_perm, inv_perm, group_sizes, expert_of_row)``: the (T*k,) sort
    permutation (stable -> preserves flattened order within an expert), its inverse,
    the (E,) per-expert row counts (sum to T*k), and the pre-sort expert id per row.
    """
    flat_expert = topk_idx_Nk.reshape(-1).astype(jnp.int32)  # (T*k,)
    sort_perm = jnp.argsort(flat_expert, stable=True).astype(jnp.int32)
    inv_perm = jnp.argsort(sort_perm).astype(jnp.int32)
    group_sizes = jnp.bincount(flat_expert, length=num_experts).astype(jnp.int32)
    return sort_perm, inv_perm, group_sizes, flat_expert


# Per-expert LoRA on the grouped path: the dense per-expert delta
# ``scaling * (x @ A[e]) @ B[e]`` (see LoRAMoEExperts in trainers/lora.py) becomes,
# on the already-grouped rows, two extra grouped GEMMs per group, so grouped-LoRA ==
# dense-LoRA up to fp reduction order. ``_lora_arrays`` extracts the raw arrays from
# the nnx.Module before the transformed core (which sees only plain arrays).


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
    """Per-expert LoRA delta for grouped rows: ``scaling * (X @ A[e]) @ B[e]`` per
    group. ``sorted_in`` is (N, in); returns (N, out) or 0.0.

    Always uses the ``jax`` ragged_dot (not the base ``primitive``): tokamax's
    kernel does not compose with the auto-sharding intermediate from the first (A)
    GEMM. The adapters are rank-r (tiny), so the perf cost is negligible."""
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

    ``hidden_ND`` (N=B*T, D), ``topk_idx_Nk``/``topk_weights_Nk`` (N, k),
    ``gate_EDF``/``up_EDF`` (E, D, F), ``down_EFD`` (E, F, D). ``primitive`` is
    "tokamax" (GPU-perf) or "jax" (reference). ``*_lora`` are optional
    ``LoRAMoEExperts`` adapters (no-op when None). Returns (N, D), equivalent to the
    dense gather path.
    """
    N, D = hidden_ND.shape
    k = topk_idx_Nk.shape[1]
    compute_dtype = hidden_ND.dtype

    gate_lora_a = _unshard_lora(_lora_arrays(gate_lora, compute_dtype))
    up_lora_a = _unshard_lora(_lora_arrays(up_lora, compute_dtype))
    down_lora_a = _unshard_lora(_lora_arrays(down_lora, compute_dtype))

    # With LoRA, force the whole core onto the jax ragged_dot reference: base and
    # adapter GEMMs share intermediates in one auto-sharding region, and mixing
    # tokamax's kernel there produces an Auto/Explicit mesh-type mismatch. LoRA is a
    # fine-tuning path so the ref primitive's perf cost is fine (numerically equal).
    if gate_lora_a is not None:
        primitive = "jax"

    hidden_ND = _unshard(hidden_ND)
    topk_idx_Nk = _unshard(topk_idx_Nk)
    topk_weights_Nk = _unshard(topk_weights_Nk)
    gate_EDF = _unshard(gate_EDF)
    up_EDF = _unshard(up_EDF)
    down_EFD = _unshard(down_EFD)

    # Pass the LoRA A/B arrays as explicit ``_core`` operands (not closure captures)
    # so ``auto_axes`` converts them to Auto mode with the base weights; a
    # closure-captured Explicit-mesh array would clash with the core's mesh type.
    # Scaling stays a static closure float. ``lora_arrays`` is 0 or 6 arrays.
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
        # Row i of the expanded (N*k) problem belongs to token (i // k).
        token_of_row = jnp.arange(N * k, dtype=jnp.int32) // k  # (N*k,)
        rows_ND = hidden_ND[token_of_row]  # (N*k, D)
        sorted_rows_ND = rows_ND[sort_perm]  # grouped by expert

        # Grouped GEMM: gate & up (D->F), SiLU-gate, then down (F->D).
        gate_out = _ragged_dot(sorted_rows_ND, gate_EDF, group_sizes, primitive)
        up_out = _ragged_dot(sorted_rows_ND, up_EDF, group_sizes, primitive)
        act_pre = None
        if lora:
            gA, gB, uA, uB, dA, dB = lora
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


# Expert parallelism: tokens shard across ``expert_axis`` (N/EP per device, all k slots)
# and stacked expert weights likewise (E/EP contiguous experts each). Dispatch
# ragged-all-to-alls each (token,slot) row to the device owning its expert; combine
# sends it back. Both are pure bijective reshuffles, never changing per-row math.


def _lora_ep_delta(sorted_in, a_pad, b_pad, scaling, gs_padded):
    """Per-local-expert LoRA delta ``scaling * (X @ A[e]) @ B[e]`` on the grouped rows;
    0.0 when the adapter is absent. ``a_pad``/``b_pad`` carry the trailing zero dummy expert."""
    if a_pad is None:
        return 0.0
    # jax ragged_dot reference for the LoRA GEMMs (see _grouped_lora_delta).
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
    gate_lora=None,
    up_lora=None,
    down_lora=None,
) -> jax.Array:
    """Expert-parallel dropless MoE, driven by the ``expert_axis`` mesh size.

    EP == 1 falls back to the single-device :func:`grouped_moe`. Otherwise the stacked
    expert weights shard on ``expert_axis`` and each token is dispatched to the device
    owning its expert (``ragged_all_to_all``), grouped-GEMM'd locally, and combined back.

    ``capacity_factor`` sizes the fixed per-device receive buffer
    (``ceil(capacity_factor * N * k / EP)`` rows), a static-shape requirement of
    ragged_all_to_all; dropless unless a device receives more than it holds.
    """
    from jax import shard_map
    from jax.sharding import PartitionSpec as P, get_abstract_mesh

    mesh = get_abstract_mesh()
    ep = 1
    if not mesh.empty and expert_axis in mesh.axis_names:
        ep = int(mesh.shape[expert_axis])
    if ep == 1:
        # EP == 1: single-device grouped_moe (keeps the tokamax perf default).
        return grouped_moe(
            hidden_ND, topk_idx_Nk, topk_weights_Nk,
            gate_EDF, up_EDF, down_EFD,
            num_experts=num_experts,
            gate_lora=gate_lora, up_lora=up_lora, down_lora=down_lora,
        )

    # EP path forces the jax ragged_dot: tokamax's custom_vjp transpose doesn't compose
    # with shard_map's backward. (GPU-perf tokamax EP kernel deferred.)
    primitive = "jax"
    assert num_experts % ep == 0, f"E={num_experts} not divisible by EP={ep}"
    N, D = hidden_ND.shape
    k = topk_idx_Nk.shape[1]
    assert N % ep == 0, f"N={N} tokens not divisible by EP={ep} (shard the token axis)"
    experts_per_shard = num_experts // ep
    n_local = N // ep  # tokens per device
    local_Nk = n_local * k  # expanded rows produced per device
    compute_dtype = hidden_ND.dtype

    # LoRA A/B shard on the expert axis like the base weights, so each device gets its
    # own expert block; passed as extra shard_map operands (None => no adapter).
    _has_lora = any(x is not None for x in (gate_lora, up_lora, down_lora))
    gate_lora_a = _lora_arrays(gate_lora, compute_dtype)
    up_lora_a = _lora_arrays(up_lora, compute_dtype)
    down_lora_a = _lora_arrays(down_lora, compute_dtype)
    # Per-device receive capacity (static; ragged_all_to_all needs a compile-time shape).
    cap = int(-(-int(capacity_factor * N * k) // ep))

    def per_device(hidden_nd, idx_nk, w_nk, gate_w, up_w, down_w, *lora_ops):
        # lora_ops (when present) = (gate_A, gate_B, up_A, up_B, down_A, down_B).
        # Expand local tokens into (token, slot) rows.
        token_of_row = jnp.arange(local_Nk, dtype=jnp.int32) // k
        rows = hidden_nd[token_of_row]  # (local_Nk, D)
        flat_expert = idx_nk.reshape(local_Nk).astype(jnp.int32)
        flat_w = w_nk.reshape(local_Nk).astype(compute_dtype)

        # Sort local rows by global expert id so each destination's block is contiguous.
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
        # recv_starts_by_src[src, dst]: where src's block starts in dst's buffer.
        recv_starts_by_src = (jnp.cumsum(send_matrix, axis=0) - send_matrix).astype(jnp.int32)
        # output_offsets[dst] (my view): offset in dst's buffer for MY block.
        output_offsets = recv_starts_by_src[my, :].astype(jnp.int32)  # (ep,) over dst
        # recv_starts[src] = where src's block lands in MY buffer.
        recv_starts = recv_starts_by_src[:, my].astype(jnp.int32)  # (ep,) over src
        # Combine writes each source's block back at the offset it originally read from.
        all_input_offsets = jax.lax.all_gather(
            input_offsets, expert_axis, axis=0, tiled=False
        )  # (src, dst) read offsets
        combine_output_offsets = all_input_offsets[:, my].astype(jnp.int32)

        # Dispatch rows to the device owning their expert.
        recv_buf = jnp.zeros((cap, D), compute_dtype)
        dispatched = _ragged_all_to_all(
            rows_sorted, recv_buf, input_offsets, send_sizes, output_offsets, recv_sizes,
            axis_name=expert_axis,
        )

        # Arrivals are source-major; re-sort to local-expert-major so one ragged_dot
        # covers all local experts. Counts of local expert j arriving from source s:
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

        # Pad LoRA A/B with the trailing zero dummy expert too (matches gate_pad/up_pad/down_pad).
        def _pad_expert(w):
            return jnp.concatenate([w, jnp.zeros_like(w[:1])], axis=0)

        if lora_ops:
            gA, gB, uA, uB, dA, dB = lora_ops
            gate_lora_pad = (_pad_expert(gA), _pad_expert(gB), gate_scaling)
            up_lora_pad = (_pad_expert(uA), _pad_expert(uB), up_scaling)
            down_lora_pad = (_pad_expert(dA), _pad_expert(dB), down_scaling)
        else:
            gate_lora_pad = up_lora_pad = down_lora_pad = (None, None, None)

        # Grouped GEMM over local experts.
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
        # Combine (inverse all-to-all): read each source's block back from my buffer and
        # write it to the offset that source originally read from.
        combined = _ragged_all_to_all(
            d_unreordered, combine_buf, recv_starts, recv_sizes,
            combine_output_offsets, send_sizes,
            axis_name=expert_axis,
        )  # back in expert-sorted order
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
    """Local-expert id for each of the ``cap`` arrival rows (padding rows -> sentinel).

    Arrivals are source-major (``[src0 block | src1 block | ...]``); source ``s``'s block
    starts at ``recv_starts[s]`` with per-local-expert counts ``my_local_counts[s]``. The
    ``experts_per_shard`` sentinel sorts padding rows last into the dummy group.
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
