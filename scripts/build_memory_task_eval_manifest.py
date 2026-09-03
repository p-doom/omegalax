"""Build one immutable memory-task eval manifest against shared training streams."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import tempfile

from omegalax.data.memory_tasks import (
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskCue,
    MemoryTaskSplit,
)
from omegalax.evals.memory_pollution import (
    build_pollution_free_manifest,
    load_training_streams,
)
from omegalax.text.checkpoint import resolve_checkpoint


def _require_slurm_job() -> None:
    slurm_node = os.environ.get("SLURMD_NODENAME", "").split(".", 1)[0]
    current_node = socket.gethostname().split(".", 1)[0]
    if not os.environ.get("SLURM_JOB_ID") or not slurm_node or slurm_node != current_node:
        raise RuntimeError(
            "Refusing to build a memory eval manifest outside a Slurm compute node; "
            "submit it with sbatch"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--exclude_manifest",
        type=Path,
        action="append",
        default=[],
        help="Previously used schema-2 gate manifest to exclude; repeat for multiple gates.",
    )
    parser.add_argument(
        "--final_holdout",
        action="store_true",
        help="Mark a final holdout and require at least one excluded gate manifest.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=[task.value for task in MemoryTask], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--segment_length", type=int, default=16)
    parser.add_argument("--num_segments", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--bank_value_alphabet", type=int, default=256)
    parser.add_argument("--num_rule_examples", type=int, default=5)
    parser.add_argument("--num_exact_bindings", type=int, default=1)
    parser.add_argument("--generator_version", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--relevance_cue",
        choices=[cue.value for cue in MemoryTaskCue],
        default=MemoryTaskCue.MARKED.value,
    )
    parser.add_argument("--num_pairs", type=int, required=True)
    return parser


def _write_immutable(path: Path, manifest: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite different eval manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(content)
    temporary_path = Path(temporary.name)
    try:
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_text() != content:
                raise ValueError(f"Refusing to overwrite different eval manifest: {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _require_slurm_job()
    eval_config = MemoryTaskConfig(
        task=args.task,
        seed=args.seed,
        segment_length=args.segment_length,
        num_segments=args.num_segments,
        vocab_size=args.vocab_size,
        bank_value_alphabet=args.bank_value_alphabet,
        num_rule_examples=args.num_rule_examples,
        num_exact_bindings=args.num_exact_bindings,
        relevance_cue=args.relevance_cue,
        split=MemoryTaskSplit.EVAL,
        generator_version=args.generator_version,
    )
    resolved = [resolve_checkpoint(checkpoint) for checkpoint in args.checkpoint]
    streams_by_identity = {}
    for checkpoint in resolved:
        for stream in load_training_streams(checkpoint.root, checkpoint.step):
            streams_by_identity[(stream.checkpoint_root, stream.checkpoint_step)] = stream
    manifest = build_pollution_free_manifest(
        eval_config,
        num_pairs=args.num_pairs,
        training_streams=tuple(streams_by_identity.values()),
        exclude_manifests=args.exclude_manifest,
        final_holdout=args.final_holdout,
    )
    _write_immutable(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
