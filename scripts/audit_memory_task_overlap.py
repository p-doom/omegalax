"""Measure exact and filler-independent overlap between memory train and eval streams."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import socket

import numpy as np

from omegalax.data import memory_tasks
from omegalax.data.memory_tasks import MemoryTask, MemoryTaskConfig, MemoryTaskSplit
from omegalax.evals.memory_pollution import (
    load_training_streams,
    pair_digest,
    table_digest,
    training_pairs_at_checkpoint,
    validate_pollution_free_manifest,
)


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


def _require_slurm_job() -> None:
    slurm_node = os.environ.get("SLURMD_NODENAME", "").split(".", 1)[0]
    current_node = socket.gethostname().split(".", 1)[0]
    if not os.environ.get("SLURM_JOB_ID") or not slurm_node or slurm_node != current_node:
        raise RuntimeError(
            "Refusing to scan memory-task overlap outside a Slurm compute node; "
            "submit it with sbatch"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_contract", type=Path, required=True)
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--allow_legacy_manifest", action="store_true")
    return parser


def _core(world: np.ndarray, filler_start: int) -> bytes:
    flat = world.reshape(-1)
    return flat[flat < filler_start].tobytes()


def _config_from_manifest(manifest: dict[str, object]) -> MemoryTaskConfig:
    fields = {
        field.name: manifest.get(field.name, field.default)
        for field in dataclasses.fields(MemoryTaskConfig)
    }
    return MemoryTaskConfig(**fields)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _require_slurm_job()
    if args.checkpoint_step <= 0:
        raise ValueError("--checkpoint_step must be > 0")

    contract = json.loads(args.training_contract.read_text())
    manifest = json.loads(args.eval_manifest.read_text())
    if manifest.get("schema_version") == 2:
        validate_pollution_free_manifest(manifest)
        loaded_streams = load_training_streams(args.training_contract.parent, args.checkpoint_step)
        missing_references = []
        for stream in loaded_streams:
            expected_reference = {
                "checkpoint_root": str(stream.checkpoint_root),
                "checkpoint_step": stream.checkpoint_step,
                "contract_sha256": stream.contract_sha256,
                "train_pairs": stream.train_pairs,
            }
            if not any(
                all(reference.get(name) == value for name, value in expected_reference.items())
                for reference in manifest["pollution_guard"]["training_streams"]
            ):
                missing_references.append(expected_reference)
        if missing_references:
            raise ValueError(
                "Pollution-guard manifest does not include the complete audited ancestry: "
                f"{missing_references}"
            )
    else:
        if not args.allow_legacy_manifest:
            raise ValueError("Legacy eval manifests require --allow_legacy_manifest")
        train_config = MemoryTaskConfig(**contract["task"])
        train_pairs = training_pairs_at_checkpoint(contract, args.checkpoint_step)
        loaded_streams = ((args.training_contract.parent.resolve(), train_config, train_pairs),)
    if manifest.get("schema_version") == 2:
        training_streams = tuple(
            (stream.checkpoint_root, stream.config, stream.train_pairs) for stream in loaded_streams
        )
        train_config = loaded_streams[0].config
    else:
        training_streams = loaded_streams
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

    filler_start = (
        memory_tasks._BANK_FILLER_START
        if train_config.task in (MemoryTask.ASSOCIATIVE_BANK, MemoryTask.RULE_BANK)
        else memory_tasks._FILLER_START
    )
    symbol_ranges = dict(_SYMBOL_POOLS[train_config.task])
    if train_config.task is MemoryTask.ASSOCIATIVE_BANK:
        symbol_ranges["values"] = (
            int(memory_tasks._BANK_VALUES[0]),
            int(memory_tasks._BANK_VALUES[0]) + train_config.bank_value_alphabet,
        )
    train_symbols = {name: set() for name in symbol_ranges}
    eval_symbols = {name: set() for name in symbol_ranges}

    eval_pairs: dict[bytes, list[int]] = {}
    eval_pair_cores: dict[tuple[bytes, bytes], list[int]] = {}
    eval_worlds: dict[bytes, list[tuple[int, int]]] = {}
    eval_world_cores: dict[bytes, list[tuple[int, int]]] = {}
    eval_pair_semantics: dict[bytes, list[int]] = {}
    eval_tables: dict[bytes, list[tuple[int, int]]] = {}
    pair_indices = [int(index) for index in manifest["pair_indices"]]
    for pair_index in pair_indices:
        pair, _ = memory_tasks._paired_tokens(pair_index, eval_config)
        eval_pairs.setdefault(pair.tobytes(), []).append(pair_index)
        eval_pair_cores.setdefault(
            (_core(pair[0], filler_start), _core(pair[1], filler_start)), []
        ).append(pair_index)
        eval_pair_semantics.setdefault(pair_digest(pair, eval_config), []).append(pair_index)
        for world in (0, 1):
            eval_worlds.setdefault(pair[world].tobytes(), []).append((pair_index, world))
            eval_world_cores.setdefault(_core(pair[world], filler_start), []).append(
                (pair_index, world)
            )
            eval_tables.setdefault(table_digest(pair[world], eval_config), []).append(
                (pair_index, world)
            )
        values = pair.reshape(-1)
        for name, (start, stop) in symbol_ranges.items():
            eval_symbols[name].update(
                int(value) for value in values[(values >= start) & (values < stop)]
            )

    matched_pairs: dict[int, tuple[str, int]] = {}
    matched_pair_cores: dict[int, tuple[str, int]] = {}
    matched_worlds: dict[tuple[int, int], tuple[str, int, int]] = {}
    matched_world_cores: dict[tuple[int, int], tuple[str, int, int]] = {}
    matched_pair_semantics: dict[int, tuple[str, int]] = {}
    matched_tables: dict[tuple[int, int], tuple[str, int, int]] = {}
    for checkpoint_root, stream_config, stream_train_pairs in training_streams:
        checkpoint_root = str(checkpoint_root)
        for train_pair_index in range(stream_train_pairs):
            pair, _ = memory_tasks._paired_tokens(train_pair_index, stream_config)
            for eval_pair_index in eval_pairs.get(pair.tobytes(), ()):
                matched_pairs.setdefault(eval_pair_index, (checkpoint_root, train_pair_index))
            pair_core = (_core(pair[0], filler_start), _core(pair[1], filler_start))
            for eval_pair_index in eval_pair_cores.get(pair_core, ()):
                matched_pair_cores.setdefault(eval_pair_index, (checkpoint_root, train_pair_index))
            for eval_pair_index in eval_pair_semantics.get(pair_digest(pair, stream_config), ()):
                matched_pair_semantics.setdefault(
                    eval_pair_index, (checkpoint_root, train_pair_index)
                )
            for world in (0, 1):
                for eval_key in eval_worlds.get(pair[world].tobytes(), ()):
                    matched_worlds.setdefault(eval_key, (checkpoint_root, train_pair_index, world))
                for eval_key in eval_world_cores.get(_core(pair[world], filler_start), ()):
                    matched_world_cores.setdefault(
                        eval_key, (checkpoint_root, train_pair_index, world)
                    )
                for eval_key in eval_tables.get(table_digest(pair[world], stream_config), ()):
                    matched_tables.setdefault(eval_key, (checkpoint_root, train_pair_index, world))
            if any(
                len(train_symbols[name]) < stop - start
                for name, (start, stop) in symbol_ranges.items()
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
        "train_pairs": sum(stream_train_pairs for _, _, stream_train_pairs in training_streams),
        "training_streams": [
            {"checkpoint_root": str(root), "train_pairs": stream_train_pairs}
            for root, _, stream_train_pairs in training_streams
        ],
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
        "semantic_pair_overlap": len(matched_pair_semantics),
        "semantic_table_overlap": len(matched_tables),
        "core_world_matches": [
            {
                "eval_pair": eval_key[0],
                "eval_world": eval_key[1],
                "train_checkpoint_root": train_key[0],
                "train_pair": train_key[1],
                "train_world": train_key[2],
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
