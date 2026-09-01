"""Measure exact and filler-independent overlap between memory train and eval streams."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from omegalax.data import memory_tasks
from omegalax.data.memory_tasks import MemoryTask, MemoryTaskConfig, MemoryTaskSplit


_SYMBOL_POOLS = {
    MemoryTask.EXACT: {"keys": (32, 64), "values": (64, 96)},
    MemoryTask.DOMINANT_RULE: {"items": (96, 160), "labels": (160, 162)},
    MemoryTask.ASSOCIATIVE_BANK: {"keys": (32, 96), "values": (96, 352)},
    MemoryTask.RULE_BANK: {
        "namespaces": (352, 360),
        "items": (360, 488),
        "labels": (488, 492),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_contract", type=Path, required=True)
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    return parser


def _core(world: np.ndarray, filler_start: int) -> bytes:
    flat = world.reshape(-1)
    return flat[flat < filler_start].tobytes()


def _config_from_manifest(manifest: dict[str, object]) -> MemoryTaskConfig:
    fields = {field.name: manifest[field.name] for field in dataclasses.fields(MemoryTaskConfig)}
    return MemoryTaskConfig(**fields)


def main() -> None:
    args = _parser().parse_args()
    if args.checkpoint_step <= 0:
        raise ValueError("--checkpoint_step must be > 0")

    contract = json.loads(args.training_contract.read_text())
    manifest = json.loads(args.eval_manifest.read_text())
    train_config = MemoryTaskConfig(**contract["task"])
    eval_config = _config_from_manifest(manifest)
    if train_config.split is not MemoryTaskSplit.TRAIN:
        raise ValueError("The training contract must use split=train")
    if eval_config.split is not MemoryTaskSplit.EVAL:
        raise ValueError("The eval manifest must use split=eval")
    if train_config.task is not eval_config.task:
        raise ValueError("Training contract and eval manifest must use the same task")
    topology = contract["runtime"]["topology"]
    if topology != {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}:
        raise ValueError(f"Only the recorded single-device topology is supported, got {topology}")

    train_pairs = (
        args.checkpoint_step
        * int(contract["pairs_per_batch"])
        * int(contract["optimizer"]["grad_accum_steps"])
    )
    filler_start = (
        memory_tasks._BANK_FILLER_START
        if train_config.task in (MemoryTask.ASSOCIATIVE_BANK, MemoryTask.RULE_BANK)
        else memory_tasks._FILLER_START
    )
    symbol_ranges = _SYMBOL_POOLS[train_config.task]
    train_symbols = {name: set() for name in symbol_ranges}
    eval_symbols = {name: set() for name in symbol_ranges}

    eval_pairs: dict[bytes, list[int]] = {}
    eval_pair_cores: dict[tuple[bytes, bytes], list[int]] = {}
    eval_worlds: dict[bytes, list[tuple[int, int]]] = {}
    eval_world_cores: dict[bytes, list[tuple[int, int]]] = {}
    pair_indices = [int(index) for index in manifest["pair_indices"]]
    for pair_index in pair_indices:
        pair, _ = memory_tasks._paired_tokens(pair_index, eval_config)
        eval_pairs.setdefault(pair.tobytes(), []).append(pair_index)
        eval_pair_cores.setdefault(
            (_core(pair[0], filler_start), _core(pair[1], filler_start)), []
        ).append(pair_index)
        for world in (0, 1):
            eval_worlds.setdefault(pair[world].tobytes(), []).append((pair_index, world))
            eval_world_cores.setdefault(_core(pair[world], filler_start), []).append(
                (pair_index, world)
            )
        values = pair.reshape(-1)
        for name, (start, stop) in symbol_ranges.items():
            eval_symbols[name].update(
                int(value) for value in values[(values >= start) & (values < stop)]
            )

    matched_pairs: dict[int, int] = {}
    matched_pair_cores: dict[int, int] = {}
    matched_worlds: dict[tuple[int, int], tuple[int, int]] = {}
    matched_world_cores: dict[tuple[int, int], tuple[int, int]] = {}
    for train_pair_index in range(train_pairs):
        pair, _ = memory_tasks._paired_tokens(train_pair_index, train_config)
        for eval_pair_index in eval_pairs.get(pair.tobytes(), ()):
            matched_pairs.setdefault(eval_pair_index, train_pair_index)
        pair_core = (_core(pair[0], filler_start), _core(pair[1], filler_start))
        for eval_pair_index in eval_pair_cores.get(pair_core, ()):
            matched_pair_cores.setdefault(eval_pair_index, train_pair_index)
        for world in (0, 1):
            for eval_key in eval_worlds.get(pair[world].tobytes(), ()):
                matched_worlds.setdefault(eval_key, (train_pair_index, world))
            for eval_key in eval_world_cores.get(_core(pair[world], filler_start), ()):
                matched_world_cores.setdefault(eval_key, (train_pair_index, world))
        if any(
            len(train_symbols[name]) < stop - start for name, (start, stop) in symbol_ranges.items()
        ):
            values = pair.reshape(-1)
            for name, (start, stop) in symbol_ranges.items():
                train_symbols[name].update(
                    int(value) for value in values[(values >= start) & (values < stop)]
                )

    eval_world_count = 2 * len(pair_indices)
    result = {
        "training_contract": str(args.training_contract.resolve()),
        "training_revision": contract["revision"],
        "checkpoint_step": args.checkpoint_step,
        "train_pairs": train_pairs,
        "eval_manifest": str(args.eval_manifest.resolve()),
        "eval_pairs": len(pair_indices),
        "eval_worlds": eval_world_count,
        "exact_pair_overlap": len(matched_pairs),
        "exact_pair_overlap_pct": 100.0 * len(matched_pairs) / len(pair_indices),
        "exact_world_overlap": len(matched_worlds),
        "exact_world_overlap_pct": 100.0 * len(matched_worlds) / eval_world_count,
        "core_pair_overlap": len(matched_pair_cores),
        "core_pair_overlap_pct": 100.0 * len(matched_pair_cores) / len(pair_indices),
        "core_world_overlap": len(matched_world_cores),
        "core_world_overlap_pct": 100.0 * len(matched_world_cores) / eval_world_count,
        "core_world_matches": [
            {
                "eval_pair": eval_key[0],
                "eval_world": eval_key[1],
                "train_pair": train_key[0],
                "train_world": train_key[1],
            }
            for eval_key, train_key in sorted(matched_world_cores.items())
        ],
        "symbol_pools": {
            name: {
                "allowed": stop - start,
                "train_seen": len(train_symbols[name]),
                "eval_seen": len(eval_symbols[name]),
                "shared": len(train_symbols[name] & eval_symbols[name]),
            }
            for name, (start, stop) in symbol_ranges.items()
        },
        "generator_file": str(Path(memory_tasks.__file__).resolve()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
