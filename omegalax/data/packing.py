"""Sequence packing for the Grain SFT pipeline.

A packing step concatenates whole training records into ``max_length`` rows so a
16k window is not mostly padding. This module contains only the *grouping* logic
(how many records go into one pack); the actual concatenation into token /
segment / position arrays is done by
:class:`omegalax.data.collator_qwen3.PackedVLMSFTCollator`.

Grouping is greedy **next-fit** bin packing over the record token lengths stored
by the record builder (``_omegalax_measured_length``): a record is appended to the
current pack while it still fits in ``max_length``; the first record that does not
fit closes the current pack and opens the next. Next-fit is chosen over first-fit
because it is single-pass and streaming, which makes the iterator cheaply
checkpointable (the only carried state is the in-progress pack plus the one
look-ahead record that did not fit). No record is ever split or truncated: a
record longer than ``max_length`` is a hard error, not silently dropped — the
record builder is responsible for keeping records ``<= max_length``.
"""

from __future__ import annotations

from typing import Any

import grain
from grain._src.python.dataset import dataset as _grain_dataset

from omegalax.data.collator_qwen3 import PACK_EXAMPLES_KEY
from omegalax.data.grain_pipeline import SOURCE_ID_KEY

MEASURED_LENGTH_KEY = "_omegalax_measured_length"


class _SequencePackIterator(_grain_dataset.DatasetIterator):
    """Greedy next-fit packer over a stream of record dicts (checkpointable)."""

    def __init__(
        self,
        parent: _grain_dataset.DatasetIterator,
        *,
        max_length: int,
        length_key: str,
        source_id_key: str | None,
    ):
        super().__init__(parent)
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        self._max_length = max_length
        self._length_key = length_key
        self._source_id_key = source_id_key
        self._buffer: list[dict[str, Any]] = []
        self._buffer_len = 0
        self._pending: dict[str, Any] | None = None
        self._parent_exhausted = False

    def _record_len(self, ex: dict[str, Any]) -> int:
        if self._length_key not in ex:
            raise KeyError(
                f"Record is missing {self._length_key!r}, required for sequence "
                "packing. Rebuild the dataset with build_records_from_chat, which "
                "stamps the measured token length on every record."
            )
        length = int(ex[self._length_key])
        if length <= 0:
            raise ValueError(f"Record has non-positive measured length {length}.")
        if length > self._max_length:
            raise ValueError(
                f"Record length {length} exceeds max_length={self._max_length}; "
                "packing never truncates. Rebuild the record index so every "
                "record fits the window."
            )
        return length

    def _emit(self) -> dict[str, Any]:
        pack: dict[str, Any] = {PACK_EXAMPLES_KEY: self._buffer}
        # Preserve a source id for the mixing metrics (tag by the first record).
        if (
            self._source_id_key is not None
            and self._buffer
            and self._source_id_key in self._buffer[0]
        ):
            pack[self._source_id_key] = self._buffer[0][self._source_id_key]
        self._buffer = []
        self._buffer_len = 0
        return pack

    def __next__(self) -> dict[str, Any]:
        while True:
            if self._pending is not None:
                ex = self._pending
                self._pending = None
            elif self._parent_exhausted:
                if self._buffer:
                    return self._emit()
                raise StopIteration
            else:
                try:
                    ex = next(self._parent)
                except StopIteration:
                    self._parent_exhausted = True
                    if self._buffer:
                        return self._emit()
                    raise
            length = self._record_len(ex)
            if self._buffer and self._buffer_len + length > self._max_length:
                # Does not fit: close the current pack, carry this record over.
                self._pending = ex
                return self._emit()
            self._buffer.append(ex)
            self._buffer_len += length

    def get_state(self) -> dict[str, Any]:
        return {
            "parent": self._parent.get_state(),
            "buffer": list(self._buffer),
            "buffer_len": self._buffer_len,
            "pending": self._pending,
            "parent_exhausted": self._parent_exhausted,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._parent.set_state(state["parent"])
        self._buffer = list(state["buffer"])
        self._buffer_len = int(state["buffer_len"])
        self._pending = state["pending"]
        self._parent_exhausted = bool(state["parent_exhausted"])

    def __str__(self) -> str:
        return f"SequencePackDatasetIterator(max_length={self._max_length})"


class SequencePackIterDataset(grain.IterDataset):
    """Greedy next-fit sequence packing over an ``IterDataset`` of records.

    Each output element is a ``dict`` with :data:`PACK_EXAMPLES_KEY` set to the
    list of records in that pack (and the first record's source id copied to the
    top level for mixing metrics). Feed it into ``.batch(batch_size, PackedVLMSFTCollator)``.
    """

    def __init__(
        self,
        parent: grain.IterDataset,
        *,
        max_length: int,
        length_key: str = MEASURED_LENGTH_KEY,
        source_id_key: str | None = SOURCE_ID_KEY,
    ):
        super().__init__(parent)
        self._max_length = max_length
        self._length_key = length_key
        self._source_id_key = source_id_key

    def __iter__(self) -> _SequencePackIterator:
        return _SequencePackIterator(
            self._parent.__iter__(),
            max_length=self._max_length,
            length_key=self._length_key,
            source_id_key=self._source_id_key,
        )

    def __str__(self) -> str:
        return f"SequencePackIterDataset(max_length={self._max_length})"
