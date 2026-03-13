"""JSONL dataset reader with process-aware splitting for multi-host training."""

from __future__ import annotations

import jax
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def _split_messages(messages: list[dict], max_turns: int) -> list[list[dict]]:
    """Split a conversation into chunks of at most *max_turns* messages.

    Splits on 2-message (assistant/user pair) boundaries so each chunk
    is a self-contained conversation.  The last chunk may be shorter.
    """
    if len(messages) <= max_turns:
        return [messages]

    step = max_turns - (max_turns % 2) if max_turns >= 2 else 1
    return [messages[i : i + step] for i in range(0, len(messages), step)]


class JSONLDataset:
    """Reads a JSONL file where each line is a JSON object with at least a
    ``"messages"`` key.

    Expected line format (multimodal)::

        {"messages": [{"role": "user", "content": [{"type": "image", "url": "/abs/path/img.jpg"}, {"type": "text", "text": "Describe"}]}, ...]}

    Or text-only::

        {"messages": [{"role": "user", "content": "Hello"}, ...]}

    Args:
        path: Path to the JSONL file.
        max_turns: If set, conversations with more messages than this are
            split into multiple examples of at most *max_turns* messages
            each (splitting on pair boundaries).
    """

    def __init__(self, path: str | Path, *, max_turns: int | None = None) -> None:
        self.path = Path(path)
        self.max_turns = max_turns
        self.examples = self._load()

    def _load(self) -> list[dict]:
        examples: list[dict] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if self.max_turns is not None and len(raw.get("messages", [])) > self.max_turns:
                    for chunk in _split_messages(raw["messages"], self.max_turns):
                        ex = dict(raw)
                        ex["messages"] = chunk
                        examples.append(ex)
                else:
                    examples.append(raw)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _split_indices(self, process_split: bool) -> list[int]:
        indices = np.arange(len(self.examples))
        if not process_split:
            return list(indices)
        pid = jax.process_index()
        nproc = jax.process_count()
        return list(np.array_split(indices, nproc)[pid])

    def iter_examples(
        self,
        *,
        shuffle: bool = True,
        seed: int = 0,
        process_split: bool = True,
        num_epochs: int | None = None,
    ) -> Iterator[dict]:
        """Yield examples, optionally shuffled and repeated.

        Args:
            shuffle: Whether to shuffle indices each epoch.
            seed: Base RNG seed for shuffling.
            process_split: If True, each JAX process sees a disjoint slice.
            num_epochs: Number of passes over the data.  ``None`` = infinite.
        """
        indices = self._split_indices(process_split)
        epoch = 0
        while num_epochs is None or epoch < num_epochs:
            order = np.array(indices)
            if shuffle:
                rng = np.random.RandomState(seed + epoch)
                rng.shuffle(order)
            for idx in order:
                yield self.examples[idx]
            epoch += 1
