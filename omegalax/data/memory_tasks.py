"""Deterministic paired symbolic tasks for controlled Statepassing experiments."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
from enum import StrEnum

import grain
import numpy as np

from omegalax.data.pretrain_data_set import (
    BATCH_PRETRAIN_METADATA_KEY,
    resolve_pretrain_dp,
)


class MemoryTask(StrEnum):
    EXACT = "exact"
    DOMINANT_RULE = "dominant_rule"


class MemoryTaskSplit(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class MemoryTaskCue(StrEnum):
    """Whether the to-be-remembered slot is flagged as relevant at encoding time.

    MARKED tags the target slot so relevance is known immediately (today's
    default, single-binding behavior at ``num_exact_bindings=1``). SURPRISE
    tags every candidate slot identically, so relevance can only be resolved
    once the final-segment query reveals which key mattered.
    """

    MARKED = "marked"
    SURPRISE = "surprise"


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryTaskConfig:
    task: MemoryTask | str
    seed: int
    segment_length: int
    num_segments: int
    vocab_size: int = 1024
    num_rule_examples: int = 5
    num_exact_bindings: int = 1
    relevance_cue: MemoryTaskCue | str = MemoryTaskCue.MARKED
    split: MemoryTaskSplit | str = MemoryTaskSplit.TRAIN

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", MemoryTask(self.task))
        object.__setattr__(self, "split", MemoryTaskSplit(self.split))
        object.__setattr__(self, "relevance_cue", MemoryTaskCue(self.relevance_cue))
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.num_segments < 2:
            raise ValueError("num_segments must be >= 2")
        if self.segment_length < 8:
            raise ValueError("segment_length must be >= 8")
        if self.vocab_size <= 256:
            raise ValueError("vocab_size must be > 256")
        if self.num_rule_examples < 3 or not self.num_rule_examples % 2:
            raise ValueError("num_rule_examples must be odd and >= 3")
        if self.num_exact_bindings < 1:
            raise ValueError("num_exact_bindings must be >= 1")
        if self.task is MemoryTask.DOMINANT_RULE:
            required = 1 + 3 * self.num_rule_examples
            if self.segment_length < required:
                raise ValueError(
                    f"segment_length must be >= {required} for {self.num_rule_examples} examples"
                )
            if self.num_rule_examples + 1 > _RULE_ITEMS.size:
                raise ValueError(
                    f"num_rule_examples={self.num_rule_examples} needs "
                    f"{self.num_rule_examples + 1} pairwise-distinct rule items "
                    f"but only {_RULE_ITEMS.size} exist"
                )
            if self.num_exact_bindings != 1 or self.relevance_cue is not MemoryTaskCue.MARKED:
                raise ValueError(
                    "num_exact_bindings/relevance_cue are only defined for MemoryTask.EXACT"
                )
        if self.task is MemoryTask.EXACT:
            required = 1 + 3 * self.num_exact_bindings
            if self.segment_length < required:
                raise ValueError(
                    f"segment_length must be >= {required} for "
                    f"{self.num_exact_bindings} exact bindings"
                )
            if self.num_exact_bindings + 1 > _VALUES.size:
                raise ValueError(
                    f"num_exact_bindings={self.num_exact_bindings} needs "
                    f"{self.num_exact_bindings + 1} pairwise-distinct values "
                    f"but only {_VALUES.size} exist"
                )
        if self.relevance_cue is MemoryTaskCue.SURPRISE and self.num_exact_bindings < 3:
            raise ValueError("relevance_cue=surprise requires num_exact_bindings >= 3")


_EXACT_TASK = 1
_RULE_TASK = 2
_BIND = 3
_DEMO = 4
_QUERY = 5
_ANSWER = 6
_DECOY_BIND = 7
_KEYS = np.arange(32, 64, dtype=np.int32)
_VALUES = np.arange(64, 96, dtype=np.int32)
_RULE_ITEMS = np.arange(96, 160, dtype=np.int32)
_RULE_LABELS = np.asarray((160, 161), dtype=np.int32)
_FILLER_START = 256


def _rng(config: MemoryTaskConfig, pair_index: int) -> np.random.Generator:
    task_id = 0 if config.task is MemoryTask.EXACT else 1
    split_id = 0 if config.split is MemoryTaskSplit.TRAIN else 1
    return np.random.default_rng(
        np.random.SeedSequence((config.seed, pair_index, task_id, split_id))
    )


def _multi_slot_exact_tokens(
    rng: np.random.Generator, paired: np.ndarray, config: MemoryTaskConfig
) -> tuple[np.ndarray, int]:
    num_slots = config.num_exact_bindings
    keys = rng.choice(_KEYS, size=num_slots, replace=False)
    target_slot = int(rng.integers(0, num_slots))
    # Both counterfactual target values and every decoy value must be drawn
    # from a single without-replacement pool: drawing them from separate
    # pools (as before) lets a decoy collide with a target value, which
    # contaminates donor/shortcut scoring downstream.
    all_values = rng.choice(_VALUES, size=num_slots + 1, replace=False)
    target_values = all_values[:2]
    decoy_values = all_values[2:]
    # Surprise reuses _BIND on every slot (no differential cue); marked flags
    # decoys with a distinct tag so only the target slot carries _BIND.
    decoy_tag = _BIND if config.relevance_cue is MemoryTaskCue.SURPRISE else _DECOY_BIND

    paired[:, 0, 0] = _EXACT_TASK
    decoy_idx = 0
    for slot in range(num_slots):
        start = 1 + 3 * slot
        if slot == target_slot:
            for world in (0, 1):
                paired[world, 0, start : start + 3] = (_BIND, keys[slot], target_values[world])
        else:
            # Written identically to both world planes: worlds must differ
            # only in the target slot's value, never in a decoy.
            paired[:, 0, start : start + 3] = (decoy_tag, keys[slot], decoy_values[decoy_idx])
            decoy_idx += 1
    for world in (0, 1):
        paired[world, -1, -5:] = (
            _EXACT_TASK,
            _QUERY,
            keys[target_slot],
            _ANSWER,
            target_values[world],
        )
    return paired, target_slot


def _paired_tokens(pair_index: int, config: MemoryTaskConfig) -> tuple[np.ndarray, int]:
    if pair_index < 0:
        raise ValueError("pair_index must be >= 0")
    rng = _rng(config, pair_index)
    shared = rng.integers(
        _FILLER_START,
        config.vocab_size,
        size=(config.num_segments, config.segment_length),
        dtype=np.int32,
    )
    paired = np.stack((shared.copy(), shared.copy()))

    if config.task is MemoryTask.EXACT:
        if config.num_exact_bindings == 1:
            key = int(rng.choice(_KEYS))
            values = rng.choice(_VALUES, size=2, replace=False)
            for world in (0, 1):
                paired[world, 0, :4] = (_EXACT_TASK, _BIND, key, values[world])
                paired[world, -1, -5:] = (
                    _EXACT_TASK,
                    _QUERY,
                    key,
                    _ANSWER,
                    values[world],
                )
            return paired, 0
        return _multi_slot_exact_tokens(rng, paired, config)

    items = rng.choice(_RULE_ITEMS, size=config.num_rule_examples + 1, replace=False)
    majority_world_0 = int(rng.integers(0, 2))
    majority_count = config.num_rule_examples // 2 + 1
    pattern = np.asarray(
        [majority_world_0] * majority_count
        + [1 - majority_world_0] * (config.num_rule_examples - majority_count),
        dtype=np.int32,
    )
    rng.shuffle(pattern)
    paired[:, 0, 0] = _RULE_TASK
    for example_idx, item in enumerate(items[:-1]):
        start = 1 + 3 * example_idx
        paired[0, 0, start : start + 3] = (_DEMO, item, _RULE_LABELS[pattern[example_idx]])
        paired[1, 0, start : start + 3] = (
            _DEMO,
            item,
            _RULE_LABELS[1 - pattern[example_idx]],
        )
    answers = (_RULE_LABELS[majority_world_0], _RULE_LABELS[1 - majority_world_0])
    for world in (0, 1):
        paired[world, -1, -5:] = (
            _RULE_TASK,
            _QUERY,
            items[-1],
            _ANSWER,
            answers[world],
        )
    return paired, 0


def build_memory_task_batch(
    pair_indices: Sequence[int],
    config: MemoryTaskConfig,
    *,
    flatten: bool,
) -> dict[str, object]:
    if not pair_indices:
        raise ValueError("pair_indices must not be empty")
    pair_indices = tuple(int(pair_index) for pair_index in pair_indices)
    generated = [_paired_tokens(pair_index, config) for pair_index in pair_indices]
    token_ids_BCT = np.concatenate([paired for paired, _ in generated], axis=0)
    target_slots = [target_slot for _, target_slot in generated]
    num_windows = token_ids_BCT.shape[0]
    attention_mask_BCT = np.ones_like(token_ids_BCT, dtype=np.int32)
    loss_mask_BCT = np.zeros_like(token_ids_BCT, dtype=np.int32)
    loss_mask_BCT[:, -1, -1] = 1
    chunk_idx_BC = np.broadcast_to(
        np.arange(config.num_segments, dtype=np.int32),
        (num_windows, config.num_segments),
    ).copy()
    reset_state_BC = np.zeros((num_windows, config.num_segments), dtype=np.bool_)
    reset_state_BC[:, 0] = True
    metadata = {
        "pair_index_B": np.repeat(np.asarray(pair_indices, dtype=np.int64), 2),
        "world_B": np.tile(np.asarray((0, 1), dtype=np.int32), len(pair_indices)),
        "answer_token_B": token_ids_BCT[:, -1, -1].copy(),
        "answer_position_B": np.full(num_windows, config.segment_length - 1, dtype=np.int32),
        "target_slot_B": np.repeat(np.asarray(target_slots, dtype=np.int32), 2),
    }

    if flatten:
        return {
            "token_ids_BT": token_ids_BCT.reshape(num_windows, -1),
            "attention_mask_BT": attention_mask_BCT.reshape(num_windows, -1),
            "loss_mask_BT": loss_mask_BCT.reshape(num_windows, -1),
            "chunk_idx_B": np.zeros(num_windows, dtype=np.int32),
            BATCH_PRETRAIN_METADATA_KEY: metadata,
        }
    return {
        "token_ids_BCT": token_ids_BCT,
        "attention_mask_BCT": attention_mask_BCT,
        "loss_mask_BCT": loss_mask_BCT,
        "chunk_idx_BC": chunk_idx_BC,
        "reset_state_BC": reset_state_BC,
        BATCH_PRETRAIN_METADATA_KEY: metadata,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class _PairIndex:
    dp_size: int
    dp_index: int

    def __call__(self, local_index: int) -> int:
        return self.dp_index + int(local_index) * self.dp_size


@dataclasses.dataclass(frozen=True, slots=True)
class _MemoryBatchBuilder:
    config: MemoryTaskConfig
    flatten: bool

    def __call__(self, pair_indices: Sequence[int]) -> dict[str, object]:
        return build_memory_task_batch(pair_indices, self.config, flatten=self.flatten)


def make_memory_task_iterator(
    config: MemoryTaskConfig,
    *,
    global_batch_size: int,
    num_batches: int,
    flatten: bool,
    dp_size: int,
    fsdp_size: int,
    process_index: int | None = None,
) -> Iterator[dict[str, object]]:
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be > 0")
    if num_batches <= 0:
        raise ValueError("num_batches must be > 0")
    if flatten:
        windows_per_batch = global_batch_size
    else:
        if global_batch_size % config.num_segments:
            raise ValueError("global_batch_size must be divisible by num_segments")
        windows_per_batch = global_batch_size // config.num_segments
    if windows_per_batch % 2:
        raise ValueError("Each batch must contain an integer number of counterfactual pairs")
    pairs_per_batch = windows_per_batch // 2

    effective_dp_size, resolved_dp_index = resolve_pretrain_dp(
        dp_size=dp_size,
        fsdp_size=fsdp_size,
        process_index=process_index,
    )
    dataset = grain.MapDataset.range(num_batches * pairs_per_batch).map(
        _PairIndex(effective_dp_size, resolved_dp_index)
    )
    batched = dataset.batch(
        batch_size=pairs_per_batch,
        drop_remainder=True,
        batch_fn=_MemoryBatchBuilder(config, flatten),
    )
    return iter(batched.to_iter_dataset(grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)))
