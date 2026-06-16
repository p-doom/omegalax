"""Pair-sampled pretraining iterator for state passing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from omegalax.data.pretrain_doc_chain import (
    DEFAULT_PAD_ID,
    DEFAULT_SEGMENT_LENGTH,
    BATCH_PRETRAIN_METADATA_KEY,
    DocChainReader,
    DocPairRef,
    build_pair_arrays,
    iter_document_pair_refs,
    pair_ref_to_record,
    resolve_pretrain_dp,
)

DocumentKey = tuple[int, int]


def _pair_ref_from_record(record: dict[str, Any]) -> DocPairRef:
    eos_token_idx = record.get("eos_token_idx")
    return DocPairRef(
        source_idx=int(record["source_idx"]),
        record_idx=int(record["record_idx"]),
        doc_id=str(record["doc_id"]),
        pair_idx=int(record["pair_idx"]),
        start=int(record["start"]),
        mid=int(record["mid"]),
        end=int(record["end"]),
        doc_token_count=int(record["doc_token_count"]),
        eos_token_idx=None if eos_token_idx is None else int(eos_token_idx),
    )


class StatepassingPretrainIterator:
    def __init__(
        self,
        sources: str | Path | Sequence[str | Path],
        *,
        batch_size: int,
        segment_length: int = DEFAULT_SEGMENT_LENGTH,
        pad_id: int = DEFAULT_PAD_ID,
        eos_id: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        num_epochs: int | None = None,
        dp_size: int = 1,
        fsdp_size: int = 1,
        dp_index: int | None = None,
        process_index: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size % 2:
            raise ValueError("batch_size must be even for 2-segment statepassing samples")
        if segment_length <= 0:
            raise ValueError("segment_length must be > 0")

        effective_dp_size, resolved_dp_index = resolve_pretrain_dp(
            dp_size=dp_size,
            fsdp_size=fsdp_size,
            process_index=process_index,
        )
        if dp_index is not None:
            resolved_dp_index = int(dp_index)
        if resolved_dp_index < 0 or resolved_dp_index >= effective_dp_size:
            raise ValueError(
                f"dp_index must be in [0, {effective_dp_size}), got {resolved_dp_index}"
            )

        self.reader = DocChainReader(sources)
        self.batch_size = int(batch_size)
        self.pair_batch_size = self.batch_size // 2
        self.segment_length = int(segment_length)
        self.pad_id = int(pad_id)
        self.eos_id = eos_id
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_epochs = num_epochs
        self.dp_size = int(effective_dp_size)
        self.dp_index = int(resolved_dp_index)

        self._pairs_by_record: dict[DocumentKey, list[DocPairRef]] = {}
        for source_idx, record_idx, doc in self.reader.iter_records():
            for pair in iter_document_pair_refs(
                doc,
                segment_length=self.segment_length,
                source_idx=source_idx,
                record_idx=record_idx,
                eos_id=self.eos_id,
            ):
                key = (source_idx, record_idx)
                self._pairs_by_record.setdefault(key, []).append(pair)

        self._record_keys = list(self._pairs_by_record)
        if not self._record_keys:
            raise ValueError("Statepassing iterator requires at least one retained pair")

        self._epoch = 0
        self._order: list[DocPairRef] = []
        self._order_pos = 0
        self._reset_epoch_order()

    def __iter__(self) -> "StatepassingPretrainIterator":
        return self

    def _reset_epoch_order(self) -> None:
        record_keys = list(self._record_keys)
        rng = np.random.default_rng(self.seed + self._epoch)
        if self.shuffle:
            rng.shuffle(record_keys)

        assigned_pairs = [
            pair
            for key in record_keys[self.dp_index :: self.dp_size]
            for pair in self._pairs_by_record[key]
        ]
        if self.shuffle:
            rng.shuffle(assigned_pairs)
        if not assigned_pairs:
            raise ValueError(
                f"No statepassing pairs assigned to dp_index={self.dp_index} "
                f"with dp_size={self.dp_size}"
            )

        self._order = assigned_pairs
        self._order_pos = 0

    def _advance_epoch(self) -> bool:
        if self.num_epochs is not None and self._epoch + 1 >= self.num_epochs:
            return False
        self._epoch += 1
        self._reset_epoch_order()
        return True

    def _next_pair(self) -> DocPairRef:
        if self._order_pos >= len(self._order) and not self._advance_epoch():
            raise StopIteration
        pair = self._order[self._order_pos]
        self._order_pos += 1
        return pair

    def __next__(self) -> dict[str, Any]:
        batch_pairs: list[DocPairRef] = []
        while len(batch_pairs) < self.pair_batch_size:
            batch_pairs.append(self._next_pair())

        token_ids = []
        attention_masks = []
        loss_masks = []
        chunk_indices = []
        reset_states = []
        last_chunk_flags = []
        doc_ids = []
        source_indices = []
        record_indices = []
        pair_indices = []
        doc_cache = {}

        for pair in batch_pairs:
            doc_key = (pair.source_idx, pair.record_idx)
            doc = doc_cache.get(doc_key)
            if doc is None:
                doc = self.reader.read(pair.source_idx, pair.record_idx)
                doc_cache[doc_key] = doc
            arrays = build_pair_arrays(
                doc.token_ids,
                pair,
                segment_length=self.segment_length,
                pad_id=self.pad_id,
                eos_id=self.eos_id,
            )
            token_ids.append(arrays["token_ids_ST"])
            attention_masks.append(arrays["attention_mask_ST"])
            loss_masks.append(arrays["loss_mask_ST"])
            chunk_indices.append(arrays["chunk_idx_S"])
            reset_states.append(arrays["reset_state_S"])
            last_chunk_flags.append(arrays["is_last_chunk_S"])
            doc_ids.append(pair.doc_id)
            source_indices.append(pair.source_idx)
            record_indices.append(pair.record_idx)
            pair_indices.append(pair.pair_idx)

        return {
            "token_ids_BST": np.stack(token_ids).astype(np.int32),
            "attention_mask_BST": np.stack(attention_masks).astype(np.int32),
            "loss_mask_BST": np.stack(loss_masks).astype(np.int32),
            "chunk_idx_BS": np.stack(chunk_indices).astype(np.int32),
            "reset_state_BS": np.stack(reset_states).astype(np.bool_),
            "is_last_chunk_BS": np.stack(last_chunk_flags).astype(np.bool_),
            BATCH_PRETRAIN_METADATA_KEY: {
                "doc_ids": doc_ids,
                "source_idx_B": np.asarray(source_indices, dtype=np.int32),
                "record_idx_B": np.asarray(record_indices, dtype=np.int32),
                "pair_idx_B": np.asarray(pair_indices, dtype=np.int32),
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "source_paths": [str(path) for path in self.reader.source_paths],
            "batch_size": self.batch_size,
            "segment_length": self.segment_length,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "dp_size": self.dp_size,
            "dp_index": self.dp_index,
            "epoch": self._epoch,
            "order": [pair_ref_to_record(pair) for pair in self._order],
            "order_pos": self._order_pos,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 2:
            raise ValueError(
                f"Unsupported statepassing iterator state version: {state.get('version')}"
            )
        expected = {
            "source_paths": [str(path) for path in self.reader.source_paths],
            "batch_size": self.batch_size,
            "segment_length": self.segment_length,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "dp_size": self.dp_size,
            "dp_index": self.dp_index,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Statepassing iterator state mismatch for {key}: "
                    f"state={state.get(key)!r}, iterator={value!r}"
                )

        self._epoch = int(state["epoch"])
        self._order = [_pair_ref_from_record(pair_state) for pair_state in state["order"]]
        self._order_pos = int(state["order_pos"])


def make_statepassing_iterator(
    sources: str | Path | Sequence[str | Path],
    *,
    batch_size: int,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int | None = None,
    dp_size: int = 1,
    fsdp_size: int = 1,
    dp_index: int | None = None,
    process_index: int | None = None,
) -> StatepassingPretrainIterator:
    return StatepassingPretrainIterator(
        sources,
        batch_size=batch_size,
        segment_length=segment_length,
        pad_id=pad_id,
        eos_id=eos_id,
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        dp_size=dp_size,
        fsdp_size=fsdp_size,
        dp_index=dp_index,
        process_index=process_index,
    )


make_statepassing_pretrain_iterator = make_statepassing_iterator
