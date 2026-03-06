"""JSONL dataset reader with process-aware splitting for multi-host training."""

from __future__ import annotations

import jax
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np


class JSONLDataset:
    """Reads a JSONL file where each line is a JSON object with at least a
    ``"messages"`` key.

    Expected line format (multimodal)::

        {"messages": [{"role": "user", "content": [{"type": "image", "url": "/abs/path/img.jpg"}, {"type": "text", "text": "Describe"}]}, ...]}

    Or text-only::

        {"messages": [{"role": "user", "content": "Hello"}, ...]}
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.examples = self._load()

    def _load(self) -> list[dict]:
        examples: list[dict] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
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
