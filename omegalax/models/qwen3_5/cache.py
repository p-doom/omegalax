"""DeltaNet state threaded through ``GatedDeltaNet`` by the rollout engine.

Tokens are flattened to ``N = B * T`` rows. Conv tap ``k`` of token ``n`` is row
``conv_src_NK[n, k]`` of ``concat(x_NC, conv pool rows)``, where pool slot ``s`` owns
rows ``N + s * (K - 1) + j``. Write ``w`` stores a state into pool slot ``write_slot_W[w]``
(distinct out-of-range slots are skipped); its conv rows are ``conv_out_src_WK[w]`` of the
same concatenation.
"""

import dataclasses

import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinearDecode:
    """One token per row; row ``n`` steps pool slot ``row_slot_N[n]`` in place."""

    conv_src_NK: jax.Array
    conv_out_src_WK: jax.Array
    write_slot_W: jax.Array
    row_slot_N: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinearExtend:
    """Packed chunks of 64 tokens; chunk ``j`` starts from pool slot ``init_slot_I[i]`` when
    ``i = init_idx_1J[0, j] >= 0``. Write ``w`` takes the state after chunk ``final_chunk_W[w]``.
    Padding tokens (``valid_N`` false) leave the state unchanged.
    """

    valid_N: jax.Array
    conv_src_NK: jax.Array
    conv_out_src_WK: jax.Array
    write_slot_W: jax.Array
    init_slot_I: jax.Array
    init_idx_1J: jax.Array
    final_chunk_W: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinearCache:
    conv_SKC: jax.Array
    state_SHAU: jax.Array
    batch: LinearDecode | LinearExtend


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Cache:
    """Per-decoder-layer caches (``LayerCache`` or ``LinearCache``); updated in place."""

    layers: list
    positions_BT: jax.Array
