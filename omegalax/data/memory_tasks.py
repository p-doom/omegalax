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
    ASSOCIATIVE_BANK = "associative_bank"
    RULE_BANK = "rule_bank"


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
    bank_value_alphabet: int = 256
    num_rule_examples: int = 5
    num_exact_bindings: int = 1
    exact_sequence_length: int = 1
    relevance_cue: MemoryTaskCue | str = MemoryTaskCue.MARKED
    split: MemoryTaskSplit | str = MemoryTaskSplit.TRAIN
    generator_version: int = 1

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
        if self.generator_version not in (1, 2):
            raise ValueError("generator_version must be 1 or 2")
        if self.num_rule_examples < 3 or not self.num_rule_examples % 2:
            raise ValueError("num_rule_examples must be odd and >= 3")
        if self.num_exact_bindings < 1:
            raise ValueError("num_exact_bindings must be >= 1")
        if self.exact_sequence_length < 1:
            raise ValueError("exact_sequence_length must be >= 1")
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
            if (
                self.num_exact_bindings != 1
                or self.exact_sequence_length != 1
                or self.relevance_cue is not MemoryTaskCue.MARKED
            ):
                raise ValueError(
                    "num_exact_bindings/exact_sequence_length/relevance_cue are only defined "
                    "for MemoryTask.EXACT"
                )
        if self.task is MemoryTask.EXACT:
            if self.exact_sequence_length > 1 and (
                self.num_exact_bindings != 1 or self.relevance_cue is not MemoryTaskCue.MARKED
            ):
                raise ValueError(
                    "exact_sequence_length > 1 requires num_exact_bindings=1 and "
                    "relevance_cue=marked"
                )
            required = max(
                1 + 3 * self.num_exact_bindings,
                4 + self.exact_sequence_length,
            )
            if self.segment_length < required:
                raise ValueError(
                    f"segment_length must be >= {required} for "
                    f"{self.num_exact_bindings} exact bindings and "
                    f"exact_sequence_length={self.exact_sequence_length}"
                )
            if self.num_exact_bindings + 1 > _VALUES.size:
                raise ValueError(
                    f"num_exact_bindings={self.num_exact_bindings} needs "
                    f"{self.num_exact_bindings + 1} pairwise-distinct values "
                    f"but only {_VALUES.size} exist"
                )
        if self.task is MemoryTask.ASSOCIATIVE_BANK:
            if self.segment_length != 64:
                raise ValueError("segment_length must be == 64 for MemoryTask.ASSOCIATIVE_BANK")
            if self.num_segments != 4:
                raise ValueError("num_segments must be == 4 for MemoryTask.ASSOCIATIVE_BANK")
            if self.vocab_size < 1024:
                raise ValueError("vocab_size must be >= 1024 for MemoryTask.ASSOCIATIVE_BANK")
            if (
                self.num_exact_bindings != 1
                or self.exact_sequence_length != 1
                or self.relevance_cue is not MemoryTaskCue.MARKED
            ):
                raise ValueError(
                    "num_exact_bindings/exact_sequence_length/relevance_cue are only defined "
                    "for MemoryTask.EXACT"
                )
            if (
                type(self.bank_value_alphabet) is not int
                or not 48 <= self.bank_value_alphabet <= _BANK_VALUES.size
            ):
                raise ValueError("bank_value_alphabet must be between 48 and 256")
        elif self.bank_value_alphabet != _BANK_VALUES.size:
            raise ValueError(
                "bank_value_alphabet is only configurable for MemoryTask.ASSOCIATIVE_BANK"
            )
        if self.task is MemoryTask.RULE_BANK:
            if self.segment_length != 64:
                raise ValueError("segment_length must be == 64 for MemoryTask.RULE_BANK")
            if self.num_segments != 5:
                raise ValueError("num_segments must be == 5 for MemoryTask.RULE_BANK")
            if self.vocab_size < 1024:
                raise ValueError("vocab_size must be >= 1024 for MemoryTask.RULE_BANK")
            if (
                self.num_exact_bindings != 1
                or self.exact_sequence_length != 1
                or self.relevance_cue is not MemoryTaskCue.MARKED
            ):
                raise ValueError(
                    "num_exact_bindings/exact_sequence_length/relevance_cue are only defined "
                    "for MemoryTask.EXACT"
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
_BANK_TASK = 8
_KEYS = np.arange(32, 64, dtype=np.int32)
_VALUES = np.arange(64, 96, dtype=np.int32)
_RULE_ITEMS = np.arange(96, 160, dtype=np.int32)
_RULE_LABELS = np.asarray((160, 161), dtype=np.int32)
_FILLER_START = 256
_BANK_KEYS = np.arange(32, 96, dtype=np.int32)
_BANK_VALUES = np.arange(96, 352, dtype=np.int32)
_BANK_NUM_SLOTS = 24
_BANK_SLOTS_PER_SEGMENT = 8
_BANK_FILLER_START = 492
_BANK_FILLER_END = 1024
_RULE_BANK_TASK = 9
_RULE_BANK_NAMESPACES = np.arange(352, 360, dtype=np.int32)
_RULE_BANK_ITEMS = np.arange(360, 488, dtype=np.int32)
_RULE_BANK_LABELS = np.arange(488, 492, dtype=np.int32)
_RULE_BANK_NUM_NAMESPACES = 8
_RULE_BANK_DEMOS_PER_NAMESPACE = 7
_RULE_BANK_NUM_DEMOS = _RULE_BANK_NUM_NAMESPACES * _RULE_BANK_DEMOS_PER_NAMESPACE
_RULE_BANK_DEMOS_PER_SEGMENT = 14
_RULE_BANK_NUM_EARLY_SEGMENTS = 4
_RULE_BANK_LABEL_COUNTS = (4, 1, 1, 1)


def _rng(
    config: MemoryTaskConfig, pair_index: int, *, stream: int | None = None
) -> np.random.Generator:
    if config.task is MemoryTask.EXACT:
        task_id = 0
    elif config.task is MemoryTask.DOMINANT_RULE:
        task_id = 1
    elif config.task is MemoryTask.ASSOCIATIVE_BANK:
        task_id = 2
    elif config.task is MemoryTask.RULE_BANK:
        task_id = 3
    else:
        raise ValueError(f"Unhandled MemoryTask: {config.task!r}")
    split_id = 0 if config.split is MemoryTaskSplit.TRAIN else 1
    entropy = (config.seed, pair_index, task_id, split_id)
    if config.generator_version == 2:
        if stream not in (0, 1):
            raise ValueError("generator_version=2 requires stream=0 or stream=1")
        entropy = (*entropy, stream)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _filler_and_task_rng(
    config: MemoryTaskConfig, pair_index: int
) -> tuple[np.random.Generator, np.random.Generator]:
    if config.generator_version == 1:
        rng = _rng(config, pair_index)
        return rng, rng
    return _rng(config, pair_index, stream=0), _rng(config, pair_index, stream=1)


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


def _associative_bank_tokens(pair_index: int, config: MemoryTaskConfig) -> tuple[np.ndarray, int]:
    if config.split is MemoryTaskSplit.TRAIN:
        bank_index = pair_index
        fixed_slot = None
    else:
        bank_index = pair_index // _BANK_NUM_SLOTS
        fixed_slot = pair_index % _BANK_NUM_SLOTS

    filler_rng, rng = _filler_and_task_rng(config, bank_index)
    shared = filler_rng.integers(
        _BANK_FILLER_START,
        _BANK_FILLER_END,
        size=(config.num_segments, config.segment_length),
        dtype=np.int32,
    )
    paired = np.stack((shared.copy(), shared.copy()))

    keys = rng.choice(_BANK_KEYS, size=_BANK_NUM_SLOTS, replace=False)
    # Base and per-slot alternate values are drawn from a single
    # without-replacement pool (as in _multi_slot_exact_tokens) so that no
    # base value can ever collide with an alternate value.
    all_values = rng.choice(
        _BANK_VALUES[: config.bank_value_alphabet],
        size=2 * _BANK_NUM_SLOTS,
        replace=False,
    )
    base_values = all_values[:_BANK_NUM_SLOTS]
    alt_values = all_values[_BANK_NUM_SLOTS:]
    target_slot = int(rng.integers(0, _BANK_NUM_SLOTS)) if fixed_slot is None else fixed_slot

    paired[:, 0, 0] = _BANK_TASK
    for slot in range(_BANK_NUM_SLOTS):
        segment = slot // _BANK_SLOTS_PER_SEGMENT
        pos_in_segment = slot % _BANK_SLOTS_PER_SEGMENT
        start = (1 if segment == 0 else 0) + 3 * pos_in_segment
        if slot == target_slot:
            for world, value in enumerate((base_values[slot], alt_values[slot])):
                paired[world, segment, start : start + 3] = (_BIND, keys[slot], value)
        else:
            # Written identically to both world planes: worlds must differ
            # only in the target slot's value, never in a non-target slot.
            paired[:, segment, start : start + 3] = (_BIND, keys[slot], base_values[slot])

    answer_values = (base_values[target_slot], alt_values[target_slot])
    for world in (0, 1):
        paired[world, -1, -5:] = (
            _BANK_TASK,
            _QUERY,
            keys[target_slot],
            _ANSWER,
            answer_values[world],
        )
    return paired, target_slot


def _rule_bank_tokens(pair_index: int, config: MemoryTaskConfig) -> tuple[np.ndarray, int]:
    if config.split is MemoryTaskSplit.TRAIN:
        bank_index = pair_index
        fixed_namespace = None
    else:
        bank_index = pair_index // _RULE_BANK_NUM_NAMESPACES
        fixed_namespace = pair_index % _RULE_BANK_NUM_NAMESPACES

    filler_rng, rng = _filler_and_task_rng(config, bank_index)
    shared = filler_rng.integers(
        _BANK_FILLER_START,
        _BANK_FILLER_END,
        size=(config.num_segments, config.segment_length),
        dtype=np.int32,
    )
    paired = np.stack((shared.copy(), shared.copy()))

    items = rng.choice(_RULE_BANK_ITEMS, size=_RULE_BANK_NUM_DEMOS, replace=False)
    namespaces = np.repeat(_RULE_BANK_NAMESPACES, _RULE_BANK_DEMOS_PER_NAMESPACE)
    labels_world0 = np.empty(_RULE_BANK_NUM_DEMOS, dtype=np.int32)
    base_dominant = np.empty(_RULE_BANK_NUM_NAMESPACES, dtype=np.int32)
    alt_dominant = np.empty(_RULE_BANK_NUM_NAMESPACES, dtype=np.int32)
    for namespace in range(_RULE_BANK_NUM_NAMESPACES):
        label_order = rng.permutation(_RULE_BANK_LABELS)
        base_dominant[namespace] = label_order[0]
        alt_dominant[namespace] = label_order[1]
        pos = namespace * _RULE_BANK_DEMOS_PER_NAMESPACE
        for label, count in zip(label_order, _RULE_BANK_LABEL_COUNTS):
            labels_world0[pos : pos + count] = label
            pos += count

    # All 56 demos are globally shuffled together (not shuffled per
    # namespace) so a namespace's 7 demos land at scattered positions across
    # segments 0..3, rather than as one contiguous, trivially-locatable block.
    shuffle_order = rng.permutation(_RULE_BANK_NUM_DEMOS)
    namespaces = namespaces[shuffle_order]
    items = items[shuffle_order]
    labels_world0 = labels_world0[shuffle_order]

    query_namespace = (
        int(rng.integers(0, _RULE_BANK_NUM_NAMESPACES))
        if fixed_namespace is None
        else fixed_namespace
    )

    paired[:, :_RULE_BANK_NUM_EARLY_SEGMENTS, 0] = _RULE_BANK_TASK
    for demo_idx in range(_RULE_BANK_NUM_DEMOS):
        segment, pos = divmod(demo_idx, _RULE_BANK_DEMOS_PER_SEGMENT)
        start = 1 + 4 * pos
        paired[:, segment, start : start + 4] = (
            _DEMO,
            namespaces[demo_idx],
            items[demo_idx],
            labels_world0[demo_idx],
        )

    # Worlds differ only in the queried namespace's demos: swap its base and
    # alternate label wherever they occur so the 4/1/1/1 histogram survives
    # with dominance flipped, leaving every other namespace byte-identical.
    query_namespace_id = int(_RULE_BANK_NAMESPACES[query_namespace])
    base_label = int(base_dominant[query_namespace])
    alt_label = int(alt_dominant[query_namespace])
    swap = {base_label: alt_label, alt_label: base_label}
    for demo_idx in np.flatnonzero(namespaces == query_namespace_id):
        segment, pos = divmod(int(demo_idx), _RULE_BANK_DEMOS_PER_SEGMENT)
        label_position = 1 + 4 * pos + 3
        current_label = int(paired[1, segment, label_position])
        paired[1, segment, label_position] = swap.get(current_label, current_label)

    for world, answer in enumerate((base_label, alt_label)):
        paired[world, -1, -5:] = (
            _RULE_BANK_TASK,
            _QUERY,
            query_namespace_id,
            _ANSWER,
            answer,
        )
    return paired, query_namespace


def _paired_tokens(pair_index: int, config: MemoryTaskConfig) -> tuple[np.ndarray, int]:
    if pair_index < 0:
        raise ValueError("pair_index must be >= 0")
    if config.task is MemoryTask.ASSOCIATIVE_BANK:
        return _associative_bank_tokens(pair_index, config)
    if config.task is MemoryTask.RULE_BANK:
        return _rule_bank_tokens(pair_index, config)

    filler_rng, rng = _filler_and_task_rng(config, pair_index)
    shared = filler_rng.integers(
        _FILLER_START,
        config.vocab_size,
        size=(config.num_segments, config.segment_length),
        dtype=np.int32,
    )
    paired = np.stack((shared.copy(), shared.copy()))

    if config.task is MemoryTask.EXACT:
        if config.num_exact_bindings == 1:
            key = int(rng.choice(_KEYS))
            # Keep the two worlds different at every answer position so token-level
            # donor metrics stay unambiguous, but allow realistic repetitions across
            # positions. One draw per position preserves the historical D=1 stream.
            values = np.stack(
                [
                    rng.choice(_VALUES, size=2, replace=False)
                    for _ in range(config.exact_sequence_length)
                ],
                axis=1,
            )
            for world in (0, 1):
                paired[world, 0, : 3 + config.exact_sequence_length] = (
                    _EXACT_TASK,
                    _BIND,
                    key,
                    *values[world],
                )
                paired[world, -1, -(4 + config.exact_sequence_length) :] = (
                    _EXACT_TASK,
                    _QUERY,
                    key,
                    _ANSWER,
                    *values[world],
                )
            return paired, 0
        return _multi_slot_exact_tokens(rng, paired, config)

    if config.task is MemoryTask.DOMINANT_RULE:
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

    raise ValueError(f"Unhandled MemoryTask: {config.task!r}")


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
    answer_length = config.exact_sequence_length if config.task is MemoryTask.EXACT else 1
    loss_mask_BCT[:, -1, -answer_length:] = 1
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
        "answer_tokens_BD": token_ids_BCT[:, -1, -answer_length:].copy(),
        "answer_positions_BD": np.broadcast_to(
            np.arange(config.segment_length - answer_length, config.segment_length, dtype=np.int32),
            (num_windows, answer_length),
        ).copy(),
    }
    if config.task is MemoryTask.RULE_BANK:
        # RULE_BANK's generic target_slot is the query namespace itself, so a
        # separate target_slot_B would just duplicate query_namespace_B below.
        metadata["query_namespace_B"] = np.repeat(np.asarray(target_slots, dtype=np.int32), 2)
    else:
        metadata["target_slot_B"] = np.repeat(np.asarray(target_slots, dtype=np.int32), 2)
    if config.task is MemoryTask.ASSOCIATIVE_BANK:
        target_segments = [slot // _BANK_SLOTS_PER_SEGMENT for slot in target_slots]
        metadata["target_segment_B"] = np.repeat(np.asarray(target_segments, dtype=np.int32), 2)
        metadata["target_age_B"] = np.repeat(
            np.asarray(
                [config.num_segments - 1 - segment for segment in target_segments],
                dtype=np.int32,
            ),
            2,
        )

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
