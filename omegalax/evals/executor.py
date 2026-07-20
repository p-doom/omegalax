"""Shared full-document evaluation chain execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from omegalax.text import api as text_api
from omegalax.trainers.loss import chunked_cross_entropy_stats


@dataclass(frozen=True)
class ChainBatch:
    document_ids: tuple[str, ...]
    token_ids_BCT: jax.Array
    attention_mask_BCT: jax.Array
    loss_mask_BCT: jax.Array
    chunk_indices_BC: jax.Array


@dataclass(frozen=True)
class SegmentInputs:
    position_offsets_B: jax.Array
    gdn_states: tuple[jax.Array, ...] | None = None
    conv_states: tuple[jax.Array, ...] | None = None


@dataclass(frozen=True)
class SegmentState:
    gdn_states: tuple[jax.Array, ...]
    conv_states: tuple[jax.Array, ...]


@dataclass(frozen=True)
class SegmentResult:
    nll_sum_B: jax.Array
    token_count_B: jax.Array
    state: SegmentState


@dataclass(frozen=True)
class ChainResult:
    condition: str
    document_ids: tuple[str, ...]
    segments: tuple[SegmentResult, ...]

    @property
    def nll_sum_B(self) -> jax.Array:
        return jnp.sum(jnp.stack([segment.nll_sum_B for segment in self.segments]), axis=0)

    @property
    def token_count_B(self) -> jax.Array:
        return jnp.sum(jnp.stack([segment.token_count_B for segment in self.segments]), axis=0)

    @property
    def nll_B(self) -> jax.Array:
        return self.nll_sum_B / jnp.maximum(self.token_count_B, 1.0)


def _position_ids(position_offsets_B: jax.Array, seq_len: int) -> jax.Array:
    positions_BT = (
        jnp.asarray(position_offsets_B, dtype=jnp.int32)[:, None]
        + jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    )
    return jnp.stack([positions_BT] * 3, axis=0)


def run_chain(
    model,
    cfg,
    batch: ChainBatch,
    *,
    pad_id: int,
    condition: str,
    segment_input_fn: Callable[[int, SegmentState | None], SegmentInputs],
) -> ChainResult:
    """Run one independent condition over every segment in each document."""
    segments = []
    previous_state = None
    for segment_idx in range(batch.token_ids_BCT.shape[1]):
        inputs = segment_input_fn(segment_idx, previous_state)
        token_ids_BT = batch.token_ids_BCT[:, segment_idx]
        hidden_BTD, _, gdn_states, conv_states = text_api.forward_with_gdn_state(
            model,
            token_ids_BT,
            pad_id=pad_id,
            cfg=cfg,
            attention_mask_BT=batch.attention_mask_BCT[:, segment_idx],
            initial_gdn_states=inputs.gdn_states,
            initial_conv_states=inputs.conv_states,
            position_ids_ZBT=_position_ids(inputs.position_offsets_B, token_ids_BT.shape[1]),
            return_conv_states=True,
        )

        def document_stats(hidden_TD, targets_T, loss_mask_T):
            return chunked_cross_entropy_stats(
                hidden_TD[None, :, :],
                model.output_weight(),
                targets_T[None, :],
                loss_mask_T[None, :],
                num_tiles=1,
            )

        nll_sum_B, token_count_B = jax.vmap(document_stats)(
            hidden_BTD,
            token_ids_BT,
            batch.loss_mask_BCT[:, segment_idx],
        )
        previous_state = SegmentState(gdn_states=gdn_states, conv_states=conv_states)
        segments.append(
            SegmentResult(
                nll_sum_B=nll_sum_B,
                token_count_B=token_count_B,
                state=previous_state,
            )
        )

    return ChainResult(
        condition=condition,
        document_ids=tuple(batch.document_ids),
        segments=tuple(segments),
    )


def assert_chunk1_consistent(
    reference: ChainResult,
    candidate: ChainResult,
    *,
    checkpoint: str,
    atol: float,
) -> None:
    """Require identical first-chunk targets and absolute-tolerance NLL agreement."""
    reference_segment = reference.segments[0]
    candidate_segment = candidate.segments[0]
    reference_counts = jax.device_get(reference_segment.token_count_B)
    candidate_counts = jax.device_get(candidate_segment.token_count_B)
    reference_nlls = jax.device_get(reference_segment.nll_sum_B)
    candidate_nlls = jax.device_get(candidate_segment.nll_sum_B)
    for document_idx, document_id in enumerate(reference.document_ids):
        reference_count = float(reference_counts[document_idx])
        candidate_count = float(candidate_counts[document_idx])
        context = (
            f"checkpoint={checkpoint}, document={document_id}, "
            f"reference_condition={reference.condition}, "
            f"candidate_condition={candidate.condition}"
        )
        if reference_count != candidate_count:
            raise ValueError(
                f"Chunk 1 token_count mismatch ({context}): "
                f"reference_token_count={reference_count}, "
                f"candidate_token_count={candidate_count}"
            )

        reference_nll = float(reference_nlls[document_idx])
        candidate_nll = float(candidate_nlls[document_idx])
        difference = abs(candidate_nll - reference_nll)
        if not difference <= atol:
            raise ValueError(
                f"Chunk 1 nll_sum difference exceeds atol={atol} ({context}): "
                f"reference_nll_sum={reference_nll}, candidate_nll_sum={candidate_nll}, "
                f"difference={difference}"
            )
