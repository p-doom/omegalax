"""DeltaNet state threaded through ``GatedDeltaNet`` by the rollout engine.

A decode row continues its pool slot's conv window and state in place. An extend flattens
tokens to ``N = B * T`` rows; the conv of token ``n`` reads the ``K - 1``
preceding rows of ``x_NC``, except for tokens ``conv_fix_F[f]`` (distinct; out-of-range
entries are skipped), whose taps are rows ``conv_src_FK[f]`` of ``concat(x_NC, conv pool
rows)``, where pool slot ``s`` owns rows ``N + s * (K - 1) + j``. Write ``w`` stores a state
into pool slot ``write_slot_W[w]`` (distinct out-of-range slots are skipped); its conv rows
are ``conv_out_src_WK[w]`` of the same concatenation.
"""

import dataclasses

import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinearDecode:
    """One token per row; row ``n`` steps pool slot ``row_slot_N[n]`` in place."""

    row_slot_N: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinearExtend:
    """Packed chunks of 64 tokens; chunk ``j`` starts from pool slot ``init_slot_I[i]`` when
    ``i = init_idx_1J[0, j] >= 0``. Write ``w`` takes the state after chunk ``final_chunk_W[w]``.
    Padding tokens (``valid_N`` false) leave the state unchanged.
    """

    valid_N: jax.Array
    conv_fix_F: jax.Array
    conv_src_FK: jax.Array
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
