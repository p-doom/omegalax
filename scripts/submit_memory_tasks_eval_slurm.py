"""Render or explicitly submit a single controlled-memory eval job."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--python_bin", type=Path, required=True)
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--job_script", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--time", dest="time_limit", default="02:00:00")
    parser.add_argument("--expected_commit")
    parser.add_argument(
        "--allow_zero_shot_horizon_override",
        action="store_true",
        help="Forward the evaluator's explicit C/L-only zero-shot horizon override.",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER)
    return parser


# ASSOCIATIVE_BANK's 3072 is the frozen exhaustive 128 banks x 24 slots
# design; RULE_BANK's 2048 is 256 banks x 8 namespaces (also the original G1
# canonical count reused for EXACT/DOMINANT_RULE) -- not a generic
# comparable-stratum rationale. Mirrors eval_memory_tasks._CANONICAL_NUM_PAIRS
# without importing omegalax here.
_CANONICAL_NUM_PAIRS_BY_TASK = {
    "exact": "2048",
    "dominant_rule": "2048",
    "rule_bank": "2048",
    "associative_bank": "3072",
}


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


def _validate_eval_args(args: list[str]) -> list[str]:
    args = list(args)
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        raise ValueError("Pass eval_memory_tasks.py arguments after --")
    for name in (
        "checkpoint",
        "task",
        "seed",
        "segment_length",
        "num_segments",
        "vocab_size",
        "bank_value_alphabet",
        "num_rule_examples",
        "num_exact_bindings",
        "generator_version",
        "relevance_cue",
        "eval_manifest",
        "num_pairs",
        "pairs_per_batch",
        "jax_cache_dir",
        "wiki_mirror_dir",
    ):
        if not _flag_value(args, name):
            raise ValueError(f"--{name} is required in eval_args")
    task = _flag_value(args, "task")
    try:
        expected_num_pairs = _CANONICAL_NUM_PAIRS_BY_TASK[task]
    except KeyError as exc:
        raise ValueError(f"Unknown --task for the canonical renderer: {task!r}") from exc
    if _flag_value(args, "num_pairs") != expected_num_pairs:
        raise ValueError(
            f"The canonical v1 renderer requires --num_pairs={expected_num_pairs} for task={task!r}"
        )
    horizon_override_flag = "--allow_zero_shot_horizon_override"
    if any(
        argument == horizon_override_flag or argument.startswith(f"{horizon_override_flag}=")
        for argument in args
    ):
        raise ValueError(f"{horizon_override_flag} must be set on the renderer, before --")
    for name in ("tp_size", "fsdp_size", "dp_size", "text_attn_backend"):
        if _flag_value(args, name) is not None:
            raise ValueError(f"--{name} is set by the renderer and must not appear in eval_args")
    return args


def _command_line(eval_args: list[str], *, allow_zero_shot_horizon_override: bool) -> str:
    return shlex.join(
        [
            *eval_args,
            *(["--allow_zero_shot_horizon_override"] if allow_zero_shot_horizon_override else []),
            "--tp_size=1",
            "--fsdp_size=1",
            "--dp_size=1",
            "--text_attn_backend=mosaic_gpu",
        ]
    )


def render_sbatch(
    *,
    repo_root: Path,
    python_bin: Path,
    job_name: str,
    log_dir: Path,
    time_limit: str,
    eval_args: list[str],
    expected_commit: str = "UNSPECIFIED",
    allow_zero_shot_horizon_override: bool = False,
) -> str:
    eval_args = _validate_eval_args(eval_args)
    repo_root = repo_root.expanduser().resolve()
    python_bin = python_bin.expanduser().absolute()
    log_dir = log_dir.expanduser().resolve()
    command = _command_line(
        eval_args,
        allow_zero_shot_horizon_override=allow_zero_shot_horizon_override,
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={shlex.quote(job_name)}
#SBATCH --partition=standard
#SBATCH --time={shlex.quote(time_limit)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100-SXM5:1
#SBATCH --output={shlex.quote(str(log_dir / "%x-%j.out"))}
#SBATCH --error={shlex.quote(str(log_dir / "%x-%j.err"))}

set -euo pipefail
REPO_ROOT={shlex.quote(str(repo_root))}
PYTHON_BIN={shlex.quote(str(python_bin))}
EXPECTED_COMMIT={shlex.quote(expected_commit)}
LOG_DIR={shlex.quote(str(log_dir))}
export PYTHONPATH="$REPO_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_LOCAL_DEVICE_IDS=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMEGALAX_DELTANET_KERNEL=pallas
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" - "$REPO_ROOT" "$EXPECTED_COMMIT" <<'PY'
import pathlib
import subprocess
import sys

import jax
import omegalax
from omegalax.models.qwen3_5.kernels import _resolve_backend

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
if jax.default_backend() != "gpu":
    raise RuntimeError(f"expected JAX GPU backend, got {{jax.default_backend()!r}}")
devices = jax.local_devices()
if len(devices) != 1 or devices[0].platform != "gpu":
    raise RuntimeError(f"expected exactly one visible GPU device, got {{devices!r}}")
if _resolve_backend() != "pallas":
    raise RuntimeError(f"expected DeltaNet backend 'pallas', got {{_resolve_backend()!r}}")
PY

EVAL_LOG="$LOG_DIR/eval-${{SLURM_JOB_ID:-local}}.log"
srun --cpu-bind=none --ntasks=1 --gpus-per-task=1 "$PYTHON_BIN" "$REPO_ROOT/scripts/eval_memory_tasks.py" \\
  {command} >"$EVAL_LOG" 2>&1
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
        eval_args=args.eval_args,
        expected_commit=expected_commit,
        allow_zero_shot_horizon_override=args.allow_zero_shot_horizon_override,
    )
    args.job_script.parent.mkdir(parents=True, exist_ok=True)
    args.job_script.write_text(script)
    print(args.job_script)
    if args.submit:
        subprocess.run(["sbatch", str(args.job_script)], check=True)


if __name__ == "__main__":
    main()
