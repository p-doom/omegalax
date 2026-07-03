"""Sequence packing (a.k.a. example packing / multi-document packing) for SFT.

Bin variable-length samples into fixed-length ``max_length`` sequences so a batch
carries far fewer padding tokens than pad-to-max. Each packed sequence records,
per token: which document it belongs to (``segment_ids``; 0 = padding), a
per-document RESET position (``position_ids``; every document restarts at 0), and
a segment-aware ``loss_mask`` whose boundary predictions are removed.

Two correctness invariants make packed training == training each document alone:

  * **No cross-document attention.** The attention path turns ``segment_ids`` into
    a block-diagonal causal mask (see
    :func:`omegalax.attention.document_causal_attention` /
    :func:`omegalax.attention.context_parallel_attention`), so a query only ever
    attends to earlier tokens of ITS OWN document.

  * **No cross-document next-token target.** The next-token prediction made at a
    document's LAST position would target the NEXT document's first token. The loss
    reads ``loss_mask`` at the TARGET position, so zeroing the FIRST token of every
    document (``doc_mask[0] = 0``) drops exactly those boundary predictions (and the
    doc->padding one). Within a document the mask is untouched, so each document's
    supervised set is identical to running it standalone.

``first_fit_pack`` is greedy first-fit bin-packing (each sample kept whole; a
sample longer than ``max_length`` is its own bin and truncated). ``PackingTextSFTCollator``
is the grain ``batch_fn``: it encodes a group of chat examples, packs them, and
emits a FIXED ``(num_packed_rows, max_length)`` batch (JIT-shape-stable).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizer

from omegalax.data.collator_qwen3 import _build_assistant_loss_mask
from omegalax.data.qwen3_encoding import encode_qwen_messages as _encode_qwen_messages


def first_fit_pack(lengths: Sequence[int], max_length: int) -> list[list[int]]:
    """Greedy first-fit bin-packing of ``lengths`` into bins of size ``max_length``.

    Returns a list of bins; each bin is a list of indices into ``lengths`` (input
    order preserved). A sample longer than ``max_length`` is capped at ``max_length``
    for the fit (the caller truncates it) and, being full, lands in its own bin.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")
    bins: list[list[int]] = []
    remaining: list[int] = []
    for i, length in enumerate(lengths):
        need = min(int(length), max_length)
        placed = False
        for b in range(len(bins)):
            if remaining[b] >= need:
                bins[b].append(i)
                remaining[b] -= need
                placed = True
                break
        if not placed:
            bins.append([i])
            remaining.append(max_length - need)
    return bins


def _pack_one_row(
    docs: list[dict[str, np.ndarray]],
    max_length: int,
    pad_id: int,
) -> dict[str, np.ndarray]:
    """Concatenate ``docs`` into one length-``max_length`` row (see module docstring).

    Each doc is ``{"token_ids", "loss_mask"}`` (1-D, equal length, already truncated
    to <= ``max_length``). Emits per-token ``token_ids``/``segment_ids``/
    ``position_ids``/``loss_mask``/``attention_mask``.
    """
    token_ids = np.full(max_length, pad_id, dtype=np.int32)
    segment_ids = np.zeros(max_length, dtype=np.int32)
    position_ids = np.zeros(max_length, dtype=np.int32)
    loss_mask = np.zeros(max_length, dtype=np.int32)
    attention_mask = np.zeros(max_length, dtype=np.int32)

    cursor = 0
    for local_idx, doc in enumerate(docs):
        tok = np.asarray(doc["token_ids"], dtype=np.int32)
        msk = np.asarray(doc["loss_mask"], dtype=np.int32)
        length = tok.shape[0]
        if length == 0:
            continue
        end = cursor + length
        seg_id = local_idx + 1  # 1-based; 0 is reserved for padding
        token_ids[cursor:end] = tok
        segment_ids[cursor:end] = seg_id
        position_ids[cursor:end] = np.arange(length, dtype=np.int32)  # reset per doc
        attention_mask[cursor:end] = 1
        # Boundary rule: the FIRST token of each document is the next-token TARGET of
        # the previous document's last position -> must never be supervised.
        doc_mask = msk.copy()
        doc_mask[0] = 0
        loss_mask[cursor:end] = doc_mask
        cursor = end

    return {
        "token_ids_BT": token_ids,
        "segment_ids_BT": segment_ids,
        "position_ids_BT": position_ids,
        "loss_mask_BT": loss_mask,
        "attention_mask_BT": attention_mask,
    }


def build_packed_sequences(
    samples: Sequence[dict[str, np.ndarray]],
    max_length: int,
    pad_id: int,
    *,
    bins: list[list[int]] | None = None,
) -> dict[str, np.ndarray]:
    """Pack per-sample ``{"token_ids", "loss_mask"}`` into ``(num_bins, max_length)``.

    ``bins`` (from :func:`first_fit_pack`) may be supplied to reuse a packing; else
    it is computed. Samples longer than ``max_length`` are truncated. The number of
    output rows equals the number of bins (data-dependent) -- callers that need a
    fixed batch dim (e.g. JIT) use :class:`PackingTextSFTCollator`.
    """
    trunc = [
        {
            "token_ids": np.asarray(s["token_ids"], dtype=np.int32)[:max_length],
            "loss_mask": np.asarray(s["loss_mask"], dtype=np.int32)[:max_length],
        }
        for s in samples
    ]
    if bins is None:
        bins = first_fit_pack([s["token_ids"].shape[0] for s in trunc], max_length)

    rows = [_pack_one_row([trunc[i] for i in b], max_length, pad_id) for b in bins]
    if not rows:
        rows = [_pack_one_row([], max_length, pad_id)]
    keys = rows[0].keys()
    return {k: np.stack([r[k] for r in rows]).astype(np.int32) for k in keys}


class PackingTextSFTCollator:
    """Grain ``batch_fn`` that packs a group of chat examples into fixed rows.

    Encodes each example (ChatML + assistant-only loss mask, exactly like
    :class:`~omegalax.data.collator_qwen3.TextSFTCollator`), first-fit bin-packs them
    into ``max_length`` sequences, and emits a FIXED ``(num_packed_rows, max_length)``
    batch so the JIT train step never recompiles:

      * fewer bins than ``num_packed_rows`` -> the batch is padded with all-padding
        rows (segment id 0, loss 0);
      * more bins -> the overflow bins are dropped for this step (``num_dropped_docs``
        is tracked; with shuffling over many epochs this is a negligible, uniform
        sample), keeping the shape fixed.

    Feed the group via grain's batch size, sized as ``num_packed_rows * group_factor``
    so packing has enough documents to fill the rows densely.

    Outputs ``token_ids_BT``, ``segment_ids_BT``, ``position_ids_BT``,
    ``loss_mask_BT``, ``attention_mask_BT`` -- all ``(num_packed_rows, max_length)``
    int32.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        num_packed_rows: int,
    ) -> None:
        if num_packed_rows <= 0:
            raise ValueError(f"num_packed_rows must be > 0, got {num_packed_rows}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_packed_rows = num_packed_rows
        assert tokenizer.pad_token_id is not None, (
            "tokenizer must have pad_token_id set (e.g. Qwen3-VL, Qwen3.5)"
        )
        self._pad_id = int(tokenizer.pad_token_id)
        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
        self.num_dropped_docs = 0

    def _encode(self, example: dict[str, Any]) -> dict[str, np.ndarray]:
        encoded = _encode_qwen_messages(example["messages"], tokenizer=self.tokenizer)
        token_ids = np.asarray(encoded["input_ids"], dtype=np.int32)[: self.max_length]
        loss_mask = _build_assistant_loss_mask(
            token_ids, self._im_start_id, self._im_end_id, self._assistant_token_id
        )
        return {"token_ids": token_ids, "loss_mask": loss_mask}

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        samples = [self._encode(ex) for ex in examples]
        bins = first_fit_pack([s["token_ids"].shape[0] for s in samples], self.max_length)

        if len(bins) > self.num_packed_rows:
            dropped = bins[self.num_packed_rows :]
            self.num_dropped_docs += sum(len(b) for b in dropped)
            bins = bins[: self.num_packed_rows]

        packed = build_packed_sequences(samples, self.max_length, self._pad_id, bins=bins)

        n_rows = packed["token_ids_BT"].shape[0]
        if n_rows < self.num_packed_rows:
            pad_rows = self.num_packed_rows - n_rows
            packed = {
                k: np.concatenate(
                    [v, np.zeros((pad_rows, self.max_length), dtype=np.int32)], axis=0
                )
                if k != "token_ids_BT"
                else np.concatenate(
                    [v, np.full((pad_rows, self.max_length), self._pad_id, dtype=np.int32)],
                    axis=0,
                )
                for k, v in packed.items()
            }
        return packed
