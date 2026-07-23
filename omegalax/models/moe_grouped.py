"""Dropless sparse Mixture-of-Experts via grouped GEMM.

Numerically-equivalent replacement for the dense compute-every-expert einsum, at
``k*(B*T)`` expert rows instead of ``E*(B*T)``: expand tokens to ``N=B*T*k``
(token,expert) rows, a stable ``argsort(expert)`` groups them by expert with counts
``group_sizes``, ``ragged_dot`` applies each expert's weights to exactly its rows,
and the inverse permutation + weighted top-k sum reproduces the dense gather. Differs
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
