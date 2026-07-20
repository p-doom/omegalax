"""Isolated GDN and convolution-state evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from omegalax.evals.executor import (
    ChainBatch,
    ChainResult,
    SegmentInputs,
    assert_chunk1_consistent,
    run_chain,
)
from omegalax.text.chat import StatePassingConfig


@dataclass(frozen=True)
class ExperimentResult:
    conditions: dict[str, ChainResult]


def _validate_layer_limit(cfg, state_config: StatePassingConfig) -> None:
    linear_layers = sum(layer_type != "full_attention" for layer_type in cfg.layer_types)
    layer_limit = state_config.gdn_layer_limit
    if layer_limit is not None and not 0 <= layer_limit <= linear_layers:
        raise ValueError(f"gdn_layer_limit must be in [0, {linear_layers}], got {layer_limit}")


def _position_offsets(
    batch: ChainBatch,
    segment_idx: int,
    *,
    pass_rope_positions: bool,
) -> jax.Array:
    if pass_rope_positions:
        return batch.chunk_indices_BC[:, segment_idx] * batch.token_ids_BCT.shape[-1]
    return jnp.zeros_like(batch.chunk_indices_BC[:, segment_idx], dtype=jnp.int32)


def _select_states(
    states: tuple[jax.Array, ...],
    *,
    pass_state: bool,
    layer_limit: int | None,
) -> tuple[jax.Array, ...] | None:
    if not pass_state or layer_limit == 0:
        return None
    if layer_limit is None:
        return states
    return tuple(
        state if layer_idx < layer_limit else jnp.zeros_like(state)
        for layer_idx, state in enumerate(states)
    )


def _zeros(states: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return tuple(jnp.zeros_like(state) for state in states)


def _take_donors(
    states: tuple[jax.Array, ...], donor_indices_B: jax.Array
) -> tuple[jax.Array, ...]:
    return tuple(state.at[donor_indices_B].get(out_sharding=state.sharding) for state in states)


def run_gdn_experiment(
    model,
    cfg,
    batch: ChainBatch,
    *,
    state_config: StatePassingConfig,
    donor_indices_B: jax.Array,
    pad_id: int,
    checkpoint: str,
) -> ExperimentResult:
    """Compare true, zero, and shuffled GDN states with true Conv controls."""
    _validate_layer_limit(cfg, state_config)
    layer_limit = state_config.gdn_layer_limit

    def true_inputs(segment_idx, previous_state):
        offsets = _position_offsets(
            batch,
            segment_idx,
            pass_rope_positions=state_config.pass_rope_positions,
        )
        if previous_state is None:
            return SegmentInputs(position_offsets_B=offsets)
        return SegmentInputs(
            position_offsets_B=offsets,
            gdn_states=_select_states(
                previous_state.gdn_states,
                pass_state=True,
                layer_limit=layer_limit,
            ),
            conv_states=_select_states(
                previous_state.conv_states,
                pass_state=state_config.pass_conv_state,
                layer_limit=layer_limit,
            ),
        )

    true_result = run_chain(
        model,
        cfg,
        batch,
        pad_id=pad_id,
        condition="true_gdn",
        segment_input_fn=true_inputs,
    )

    def control_inputs(segment_idx, *, shuffled):
        offsets = _position_offsets(
            batch,
            segment_idx,
            pass_rope_positions=state_config.pass_rope_positions,
        )
        if segment_idx == 0:
            return SegmentInputs(position_offsets_B=offsets)
        true_state = true_result.segments[segment_idx - 1].state
        gdn_states = (
            _take_donors(true_state.gdn_states, donor_indices_B)
            if shuffled
            else _zeros(true_state.gdn_states)
        )
        return SegmentInputs(
            position_offsets_B=offsets,
            gdn_states=_select_states(
                gdn_states,
                pass_state=True,
                layer_limit=layer_limit,
            ),
            conv_states=_select_states(
                true_state.conv_states,
                pass_state=state_config.pass_conv_state,
                layer_limit=layer_limit,
            ),
        )

    zero_result = run_chain(
        model,
        cfg,
        batch,
        pad_id=pad_id,
        condition="zero_gdn",
        segment_input_fn=lambda idx, _previous: control_inputs(idx, shuffled=False),
    )
    shuffled_result = run_chain(
        model,
        cfg,
        batch,
        pad_id=pad_id,
        condition="shuffled_gdn",
        segment_input_fn=lambda idx, _previous: control_inputs(idx, shuffled=True),
    )
    assert_chunk1_consistent(true_result, zero_result, checkpoint=checkpoint, atol=1e-6)
    assert_chunk1_consistent(true_result, shuffled_result, checkpoint=checkpoint, atol=1e-6)
    return ExperimentResult(
        conditions={
            "true_gdn": true_result,
            "zero_gdn": zero_result,
            "shuffled_gdn": shuffled_result,
        }
    )


def run_conv_experiment(
    model,
    cfg,
    batch: ChainBatch,
    *,
    state_config: StatePassingConfig,
    donor_indices_B: jax.Array,
    pad_id: int,
    checkpoint: str,
) -> ExperimentResult:
    """Compare true, zero, and shuffled Conv states with true GDN controls."""
    _validate_layer_limit(cfg, state_config)
    if not state_config.pass_conv_state:
        raise ValueError("Conv state experiment is not applicable when pass_conv_state=False")
    layer_limit = state_config.gdn_layer_limit

    def true_inputs(segment_idx, previous_state):
        offsets = _position_offsets(
            batch,
            segment_idx,
            pass_rope_positions=state_config.pass_rope_positions,
        )
        if previous_state is None:
            return SegmentInputs(position_offsets_B=offsets)
        return SegmentInputs(
            position_offsets_B=offsets,
            gdn_states=_select_states(
                previous_state.gdn_states,
                pass_state=state_config.pass_gdn_state,
                layer_limit=layer_limit,
            ),
            conv_states=_select_states(
                previous_state.conv_states,
                pass_state=True,
                layer_limit=layer_limit,
            ),
        )

    true_result = run_chain(
        model,
        cfg,
        batch,
        pad_id=pad_id,
        condition="true_conv",
        segment_input_fn=true_inputs,
    )

    def control_inputs(segment_idx, *, shuffled):
        offsets = _position_offsets(
            batch,
            segment_idx,
            pass_rope_positions=state_config.pass_rope_positions,
        )
        if segment_idx == 0:
            return SegmentInputs(position_offsets_B=offsets)
        true_state = true_result.segments[segment_idx - 1].state
        conv_states = (
            _take_donors(true_state.conv_states, donor_indices_B)
            if shuffled
            else _zeros(true_state.conv_states)
        )
        return SegmentInputs(
            position_offsets_B=offsets,
            gdn_states=_select_states(
                true_state.gdn_states,
                pass_state=state_config.pass_gdn_state,
                layer_limit=layer_limit,
            ),
            conv_states=_select_states(
                conv_states,
                pass_state=True,
                layer_limit=layer_limit,
            ),
        )

    zero_result = run_chain(
        model,
        cfg,
        batch,
        pad_id=pad_id,
        condition="zero_conv",
        segment_input_fn=lambda idx, _previous: control_inputs(idx, shuffled=False),
    )
    shuffled_result = run_chain(
        model,
        cfg,
        batch,
        pad_id=pad_id,
        condition="shuffled_conv",
        segment_input_fn=lambda idx, _previous: control_inputs(idx, shuffled=True),
    )
    assert_chunk1_consistent(true_result, zero_result, checkpoint=checkpoint, atol=1e-6)
    assert_chunk1_consistent(true_result, shuffled_result, checkpoint=checkpoint, atol=1e-6)
    return ExperimentResult(
        conditions={
            "true_conv": true_result,
            "zero_conv": zero_result,
            "shuffled_conv": shuffled_result,
        }
    )
