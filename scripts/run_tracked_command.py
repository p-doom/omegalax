"""Run one command as a tracked Weights & Biases job."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import re
import subprocess

import pyarrow.parquet as pq
import wandb


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--id")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--config_json")
    parser.add_argument("--result_dir", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _log_results(run, result_dirs: Sequence[str]) -> None:
    for raw_result_dir in result_dirs:
        result_dir = Path(raw_result_dir).expanduser().resolve()
        if not result_dir.is_dir():
            raise FileNotFoundError(f"result directory does not exist: {result_dir}")
        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", result_dir.name).strip("-.") or "results"
        artifact = wandb.Artifact(f"{label}-{run.id}", type="evaluation-results")
        for relative in ("eval_config.json", "status.json"):
            path = result_dir / relative
            if path.is_file():
                artifact.add_file(path, name=relative)
        for subdirectory in ("summary", "plots", "comparisons"):
            root = result_dir / subdirectory
            if root.is_dir():
                artifact.add_dir(root, name=subdirectory)
        run.log_artifact(artifact)

        summary_paths = tuple(sorted((result_dir / "summary").glob("**/metrics.parquet")))
        for index, summary_path in enumerate(summary_paths):
            table = pq.read_table(summary_path).to_pandas()
            run.log({f"summary_metrics/{label}_{index}": wandb.Table(dataframe=table)})


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = {"command": args.command}
    if args.config_json:
        config.update(json.loads(Path(args.config_json).read_text()))
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        name=args.name,
        id=args.id,
        resume="allow" if args.id else None,
        tags=args.tag,
        job_type="evaluation",
        config=config,
    )
    completed = subprocess.run(args.command, check=False)
    run.summary["exit_code"] = completed.returncode
    if completed.returncode == 0:
        _log_results(run, args.result_dir)
    run.finish(exit_code=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
