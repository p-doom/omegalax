"""Causal evaluation conditions for the controlled symbolic memory tasks."""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY
from omegalax.text import api as text_api


class MemoryEvalMode(StrEnum):
    FULL_CONTEXT = "full_context"
    RESET = "reset"
    CORRECT_STATE = "correct_state"
    ZERO_STATE = "zero_state"
    COUNTERFACTUAL_STATE = "counterfactual_state"


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryEvalResult:
    accuracy: float
    answer_nll: float
    donor_accuracy: float
    donor_answer_nll: float
    predictions_B: np.ndarray
    answer_nll_B: np.ndarray
    donor_answer_nll_B: np.ndarray


def _counterfactual_indices(
    pair_index_B: np.ndarray,
    world_B: np.ndarray,
) -> np.ndarray:
    pair_index_B = np.asarray(pair_index_B)
    world_B = np.asarray(world_B)
    if pair_index_B.ndim != 1 or world_B.shape != pair_index_B.shape:
        raise ValueError("pair_index_B and world_B must be aligned rank-1 arrays")
    if pair_index_B.size % 2:
        raise ValueError("Counterfactual batches must contain complete pairs")
    donor_indices = np.arange(pair_index_B.size, dtype=np.int32)
    for start in range(0, pair_index_B.size, 2):
        if (
            pair_index_B[start] != pair_index_B[start + 1]
            or world_B[start] != 0
            or world_B[start + 1] != 1
        ):
            raise ValueError("Counterfactual pair members must be adjacent in world order 0, 1")
        donor_indices[start : start + 2] = (start + 1, start)
    return donor_indices


def swap_counterfactual_pairs(
    states: tuple[jax.Array, ...],
    *,
    pair_index_B: np.ndarray,
    world_B: np.ndarray,
) -> tuple[jax.Array, ...]:
    donor_indices = jnp.asarray(_counterfactual_indices(pair_index_B, world_B), dtype=jnp.int32)
    return tuple(state[donor_indices] for state in states)


def _assert_reset_only_at_segment_zero(reset_state_BC: np.ndarray) -> None:
    """Guard the invariant this evaluator relies on but never applies explicitly.

    ``carried_states`` starts as ``None`` before segment 0 for every mode, which only
    matches ``reset_state_BC`` if segment 0 is the sole reset point. A batch that
    resets elsewhere (e.g. a future variable-length/multi-document eval) would
    silently skip that reset instead of erroring, so we fail loudly here instead.
    """
    reset_state_BC = np.asarray(reset_state_BC)
    expected_BC = np.zeros_like(reset_state_BC, dtype=bool)
    expected_BC[:, 0] = True
    if not np.array_equal(reset_state_BC.astype(bool), expected_BC):
        raise ValueError(
            "_segmented_final_hidden only supports a state reset at segment 0; "
            f"got reset_state_BC={reset_state_BC.tolist()}"
        )


def _segmented_final_hidden(
    model,
    model_config,
    token_ids_BCT: jax.Array,
    attention_mask_BCT: jax.Array,
    reset_state_BC: np.ndarray,
    mode: MemoryEvalMode,
    *,
    pair_index_B: np.ndarray,
    world_B: np.ndarray,
) -> jax.Array:
    _assert_reset_only_at_segment_zero(reset_state_BC)
    carried_states = None
    final_hidden = None
    for segment_idx in range(token_ids_BCT.shape[1]):
        initial_states = carried_states
        if mode is MemoryEvalMode.RESET:
            initial_states = None
        elif mode is MemoryEvalMode.ZERO_STATE and carried_states is not None:
            initial_states = tuple(jnp.zeros_like(state) for state in carried_states)
        elif (
            mode is MemoryEvalMode.COUNTERFACTUAL_STATE
            and segment_idx == token_ids_BCT.shape[1] - 1
            and carried_states is not None
        ):
            initial_states = swap_counterfactual_pairs(
                carried_states,
                pair_index_B=pair_index_B,
                world_B=world_B,
            )
        final_hidden, _, carried_states = text_api.forward_with_gdn_state(
            model,
            token_ids_BCT[:, segment_idx],
            0,
            model_config,
            attention_mask_BT=attention_mask_BCT[:, segment_idx],
            initial_gdn_states=initial_states,
        )
    if final_hidden is None:
        raise ValueError("A memory-task batch must contain at least one segment")
    return final_hidden


def _score_answers(
    model,
    hidden_BTD: jax.Array,
    positions_B: jax.Array,
    answers_B: jax.Array,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    batch_indices = jnp.arange(hidden_BTD.shape[0], dtype=jnp.int32)
    answer_hidden_BD = hidden_BTD[batch_indices, positions_B - 1]
    logits_BV = jnp.einsum("BD,DV->BV", answer_hidden_BD, model.output_weight())
    predictions_B = jnp.argmax(logits_BV, axis=-1)
    answer_logits_B = logits_BV[batch_indices, answers_B]
    nll_B = jax.nn.logsumexp(logits_BV, axis=-1) - answer_logits_B
    accuracy = jnp.mean(predictions_B == answers_B)
    return (
        float(jax.device_get(accuracy)),
        float(jax.device_get(jnp.mean(nll_B))),
        np.asarray(jax.device_get(predictions_B)),
        np.asarray(jax.device_get(nll_B)),
    )


def evaluate_memory_batch(
    model,
    model_config,
    batch: dict[str, Any],
) -> dict[MemoryEvalMode, MemoryEvalResult]:
    token_ids_BCT = jnp.asarray(batch["token_ids_BCT"], dtype=jnp.int32)
    attention_mask_BCT = jnp.asarray(batch["attention_mask_BCT"], dtype=jnp.int32)
    reset_state_BC = np.asarray(batch["reset_state_BC"])
    metadata = dict(batch[BATCH_PRETRAIN_METADATA_KEY])
    pair_index_B = np.asarray(metadata["pair_index_B"])
    world_B = np.asarray(metadata["world_B"])
    donor_indices = _counterfactual_indices(pair_index_B, world_B)
    answers_B = jnp.asarray(metadata["answer_token_B"], dtype=jnp.int32)
    donor_answers_B = answers_B[jnp.asarray(donor_indices)]
    local_positions_B = jnp.asarray(metadata["answer_position_B"], dtype=jnp.int32)

    results: dict[MemoryEvalMode, MemoryEvalResult] = {}
    for mode in MemoryEvalMode:
        if mode is MemoryEvalMode.FULL_CONTEXT:
            batch_size, num_segments, segment_length = token_ids_BCT.shape
            token_ids_BT = token_ids_BCT.reshape(batch_size, num_segments * segment_length)
            attention_mask_BT = attention_mask_BCT.reshape(
                batch_size, num_segments * segment_length
            )
            hidden_BTD, _ = text_api.forward(
                model,
                token_ids_BT,
                0,
                model_config,
                attention_mask_BT=attention_mask_BT,
            )
            positions_B = local_positions_B + (num_segments - 1) * segment_length
        else:
            hidden_BTD = _segmented_final_hidden(
                model,
                model_config,
                token_ids_BCT,
                attention_mask_BCT,
                reset_state_BC,
                mode,
                pair_index_B=pair_index_B,
                world_B=world_B,
            )
            positions_B = local_positions_B
        accuracy, answer_nll, predictions_B, answer_nll_B = _score_answers(
            model, hidden_BTD, positions_B, answers_B
        )
        donor_accuracy, donor_answer_nll, _, donor_answer_nll_B = _score_answers(
            model, hidden_BTD, positions_B, donor_answers_B
        )
        results[mode] = MemoryEvalResult(
            accuracy=accuracy,
            answer_nll=answer_nll,
            donor_accuracy=donor_accuracy,
            donor_answer_nll=donor_answer_nll,
            predictions_B=predictions_B,
            answer_nll_B=answer_nll_B,
            donor_answer_nll_B=donor_answer_nll_B,
        )
    return results
