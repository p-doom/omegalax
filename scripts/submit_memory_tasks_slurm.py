"""Render or explicitly submit a single controlled-memory training job."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--python_bin", type=Path, required=True)
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--job_script", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--time", dest="time_limit", default="04:00:00")
    parser.add_argument("--expected_commit")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def _flag_value(args: list[str], name: str) -> str | None:
    flag = f"--{name}"
    values = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == flag:
            if index + 1 == len(args):
                raise ValueError(f"{flag} requires a value")
            values.append(args[index + 1])
            index += 2
            continue
        if argument.startswith(f"{flag}="):
            values.append(argument.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ValueError(f"{flag} may be specified only once")
    return values[0] if values else None


def _validate_train_args(args: list[str]) -> list[str]:
    args = list(args)
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        raise ValueError("Pass train_memory_tasks.py arguments after --")
    for name in (
        "save_dir",
        "task",
        "condition",
        "jax_cache_dir",
        "run_plan_sha256",
        "success_criterion",
    ):
        if not _flag_value(args, name):
            raise ValueError(f"--{name} is required in train_args")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", _flag_value(args, "run_plan_sha256")):
        raise ValueError("--run_plan_sha256 must be exactly 64 hexadecimal characters")
    for name in ("resume", "text_attn_backend"):
        if _flag_value(args, name) is not None:
            raise ValueError(f"--{name} is set by the renderer and must not appear in train_args")
    return args


def _command_lines(train_args: list[str]) -> str:
    forced_args = ["--resume=if_present", "--text_attn_backend=mosaic_gpu"]
    all_args = [*train_args, *forced_args]
    return shlex.join(all_args)


def render_sbatch(
    *,
    repo_root: Path,
    python_bin: Path,
    job_name: str,
    log_dir: Path,
    time_limit: str,
    train_args: list[str],
    expected_commit: str = "UNSPECIFIED",
) -> str:
    train_args = _validate_train_args(train_args)
    repo_root = repo_root.expanduser().resolve()
    python_bin = python_bin.expanduser().absolute()
    log_dir = log_dir.expanduser().resolve()
    wandb_dir = log_dir / "wandb"
    wandb_cache_dir = log_dir.parent / "wandb_cache"
    primary_command = _command_lines(train_args)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={shlex.quote(job_name)}
#SBATCH --partition=standard
#SBATCH --time={shlex.quote(time_limit)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100-SXM5:1
#SBATCH --signal=USR1@1800
#SBATCH --requeue
#SBATCH --output={shlex.quote(str(log_dir / "%x-%j.out"))}
#SBATCH --error={shlex.quote(str(log_dir / "%x-%j.err"))}

set -euo pipefail
REPO_ROOT={shlex.quote(str(repo_root))}
PYTHON_BIN={shlex.quote(str(python_bin))}
EXPECTED_COMMIT={shlex.quote(expected_commit)}
LOG_DIR={shlex.quote(str(log_dir))}
export WANDB_MODE=online
export WANDB_DIR={shlex.quote(str(wandb_dir))}
export WANDB_CACHE_DIR={shlex.quote(str(wandb_cache_dir))}
export PYTHONPATH="$REPO_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_LOCAL_DEVICE_IDS=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMEGALAX_DELTANET_KERNEL=pallas
mkdir -p "$LOG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" - "$REPO_ROOT" "$EXPECTED_COMMIT" <<'PY'
import pathlib
import subprocess
import sys

import omegalax

root = pathlib.Path(sys.argv[1]).resolve()
expected_commit = sys.argv[2]
module_path = pathlib.Path(omegalax.__file__).resolve()
if not module_path.is_relative_to(root):
    raise RuntimeError(f"omegalax imported from {{module_path}}, expected under {{root}}")
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
).stdout.strip()
if commit != expected_commit:
    raise RuntimeError(f"git commit changed: expected {{expected_commit}}, got {{commit}}")
if subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout:
    raise RuntimeError("refusing to run from a dirty worktree")
PY

TRAIN_LOG="$LOG_DIR/train-${{SLURM_JOB_ID:-local}}.log"
srun --cpu-bind=none --ntasks=1 --gpus-per-task=1 "$PYTHON_BIN" "$REPO_ROOT/scripts/train_memory_tasks.py" \\
  {primary_command} >"$TRAIN_LOG" 2>&1
"""


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    expected_commit = args.expected_commit or _git_commit(args.repo_root)
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    script = render_sbatch(
        repo_root=args.repo_root,
        python_bin=args.python_bin,
        job_name=args.job_name,
        log_dir=log_dir,
        time_limit=args.time_limit,
        train_args=args.train_args,
        expected_commit=expected_commit,
    )
    args.job_script.parent.mkdir(parents=True, exist_ok=True)
    args.job_script.write_text(script)
    print(args.job_script)
    if args.submit:
        subprocess.run(["sbatch", str(args.job_script)], check=True)


if __name__ == "__main__":
    main()
