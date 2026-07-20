"""Render and submit checkpoint evaluation jobs to Slurm."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import re
import shlex
import subprocess

from omegalax.evals import runner
from omegalax.evals.runner import CheckpointEvalRequest, CheckpointEvalSpec
from omegalax.training_contract import ManualEvalConfig


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODES = ("all", "subset", "plot", "compare")
_EXPERIMENTS = ("gdn", "conv")
_PLOT_TYPES = ("in_horizon", "beyond_horizon", "exact_length", "heatmap")
_SPEC_FIELDS = (
    "c_train",
    "pass_gdn_state",
    "gdn_layer_limit",
    "pass_conv_state",
    "pass_rope_positions",
    "pad_id",
    "eos_id",
)


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def _parse_layer_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected an integer or 'all', got {value!r}") from error


def _parse_plot_types(value: str) -> tuple[str, ...]:
    plot_types = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = tuple(item for item in plot_types if item not in _PLOT_TYPES)
    if not plot_types or unknown:
        raise argparse.ArgumentTypeError(
            f"plot_types must contain only {', '.join(_PLOT_TYPES)}; got {value!r}"
        )
    return plot_types


def _parse_experiments(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    experiments = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = tuple(item for item in experiments if item not in _EXPERIMENTS)
    if not experiments or unknown:
        raise ValueError(f"experiments must contain only {', '.join(_EXPERIMENTS)}; got {value!r}")
    return experiments


def _as_requests(
    request_or_requests: (
        CheckpointEvalRequest
        | CheckpointEvalSpec
        | Sequence[CheckpointEvalRequest | CheckpointEvalSpec]
    ),
) -> tuple[CheckpointEvalRequest | CheckpointEvalSpec, ...]:
    if isinstance(request_or_requests, (CheckpointEvalRequest, CheckpointEvalSpec)):
        return (request_or_requests,)
    return tuple(request_or_requests)


def _command_argument(name: str, value: object) -> str:
    return f"--{name}={value}"


def render_eval_sbatch(
    request_or_requests: (
        CheckpointEvalRequest
        | CheckpointEvalSpec
        | Sequence[CheckpointEvalRequest | CheckpointEvalSpec]
    ),
    *,
    manifest_path: str | Path | None,
    repo_root: str | Path,
    log_dir: str | Path,
    partition: str,
    qos: str | None,
    time_limit: str,
    mode: str,
    experiments: Sequence[str] | None,
    plot_types: Sequence[str],
    plot_output_roots: Sequence[str | Path],
    comparison_name: str | None,
    pilot_cap: int | None,
) -> str:
    """Render one fixed-topology evaluation batch script."""

    requests = _as_requests(request_or_requests)
    command = ["python", "scripts/run_checkpoint_evals.py"]
    for request in requests:
        command.append(_command_argument("checkpoint", request.checkpoint))
        manual_config = (
            request.manual_config
            if isinstance(request, CheckpointEvalRequest)
            else ManualEvalConfig(
                c_train=request.c_train,
                pass_gdn_state=request.pass_gdn_state,
                gdn_layer_limit=request.gdn_layer_limit,
                pass_conv_state=request.pass_conv_state,
                pass_rope_positions=request.pass_rope_positions,
                pad_id=request.pad_id,
                eos_id=request.eos_id,
            )
        )
        if manual_config is not None:
            command.extend(
                (
                    _command_argument("c_train", manual_config.c_train),
                    _command_argument("pass_gdn_state", str(manual_config.pass_gdn_state).lower()),
                    _command_argument(
                        "gdn_layer_limit",
                        (
                            "all"
                            if manual_config.gdn_layer_limit is None
                            else manual_config.gdn_layer_limit
                        ),
                    ),
                    _command_argument(
                        "pass_conv_state", str(manual_config.pass_conv_state).lower()
                    ),
                    _command_argument(
                        "pass_rope_positions",
                        str(manual_config.pass_rope_positions).lower(),
                    ),
                    _command_argument("pad_id", manual_config.pad_id),
                    _command_argument("eos_id", manual_config.eos_id),
                )
            )

    command.append(_command_argument("mode", mode))
    if manifest_path is not None:
        command.append(_command_argument("manifest_path", manifest_path))
    if experiments is not None:
        command.append(_command_argument("experiments", ",".join(experiments)))
    command.append(_command_argument("plot_types", ",".join(plot_types)))
    command.extend(_command_argument("plot_output_root", root) for root in plot_output_roots)
    if pilot_cap is not None and mode in ("all", "subset"):
        command.append(_command_argument("document_cap", pilot_cap))
    if comparison_name is not None:
        command.append(_command_argument("comparison_name", comparison_name))
    command.extend(
        (
            "--tp_size=1",
            "--fsdp_size=1",
            "--dp_size=4",
            "--batch_size=16",
        )
    )

    log_dir = Path(log_dir)
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=checkpoint-eval",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:4",
        "#SBATCH --cpus-per-task=24",
        "#SBATCH --mem=120G",
        f"#SBATCH --partition={shlex.quote(str(partition))}",
        f"#SBATCH --time={shlex.quote(str(time_limit))}",
        f"#SBATCH --output={shlex.quote(str(log_dir / 'checkpoint_eval_%j.out'))}",
        f"#SBATCH --error={shlex.quote(str(log_dir / 'checkpoint_eval_%j.err'))}",
    ]
    if qos is not None:
        lines.append(f"#SBATCH --qos={shlex.quote(str(qos))}")
    lines.extend(
        (
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            "export JAX_LOCAL_DEVICE_IDS=0,1,2,3",
            "",
            " \\\n  ".join(shlex.quote(argument) for argument in command),
        )
    )
    return "\n".join(lines) + "\n"


def _validate_request(
    requests: Sequence[CheckpointEvalRequest | CheckpointEvalSpec],
    *,
    manifest_path: str | Path | None,
    mode: str,
    experiments: Sequence[str] | None,
    plot_types: Sequence[str],
    comparison_name: str | None,
) -> tuple[tuple[CheckpointEvalSpec, ...], tuple[str, ...] | None]:
    if not requests:
        raise ValueError("At least one checkpoint spec is required")
    if mode not in _MODES:
        raise ValueError(f"Unsupported mode: {mode!r}")
    if experiments is not None:
        unknown_experiments = tuple(
            experiment for experiment in experiments if experiment not in _EXPERIMENTS
        )
        if not experiments or unknown_experiments:
            raise ValueError(
                f"experiments must contain only {', '.join(_EXPERIMENTS)}; "
                f"got {tuple(experiments)!r}"
            )
        if len(set(experiments)) != len(experiments):
            raise ValueError(f"Duplicate experiments are not allowed: {tuple(experiments)!r}")
    if mode in ("subset", "compare") and not experiments:
        raise ValueError(f"mode={mode} requires explicit experiments")
    unknown_plot_types = tuple(
        plot_type for plot_type in plot_types if plot_type not in _PLOT_TYPES
    )
    if not plot_types or unknown_plot_types:
        raise ValueError(
            f"plot_types must contain only {', '.join(_PLOT_TYPES)}; got {tuple(plot_types)!r}"
        )
    if len(set(plot_types)) != len(plot_types):
        raise ValueError(f"Duplicate plot types are not allowed: {tuple(plot_types)!r}")
    if mode == "compare" and len(requests) < 2:
        raise ValueError("Compare mode requires at least two checkpoint specs")
    if mode == "compare" and not comparison_name:
        raise ValueError("Compare mode requires a comparison name")
    if mode == "compare" and Path(comparison_name).name != comparison_name:
        raise ValueError(f"Invalid comparison name: {comparison_name!r}")
    if mode in ("all", "subset", "compare") and manifest_path is None:
        raise ValueError(f"mode={mode} requires a manifest path")

    specs = runner.resolve_checkpoint_eval_requests(requests)
    if experiments is not None:
        for spec in specs:
            unavailable = set(experiments) - set(runner.applicable_experiments(spec))
            if unavailable:
                raise ValueError(
                    f"Experiment(s) {sorted(unavailable)} are not applicable to "
                    f"checkpoint {spec.checkpoint}"
                )

    if mode == "compare":
        comparison_plot_types = tuple(
            plot_type for plot_type in plot_types if plot_type != "heatmap"
        )
        if not comparison_plot_types:
            raise ValueError("Heatmap-only requests cannot produce a checkpoint comparison")
        return specs, comparison_plot_types
    return specs, None


def _submit_script(path: Path, *, dependency: str | None = None) -> str:
    command = ["sbatch"]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(path))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    match = re.search(r"Submitted batch job (\d+)", completed.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse Slurm job id from: {completed.stdout!r}")
    return match.group(1)


def submit_checkpoint_evals(
    requests: Sequence[CheckpointEvalRequest | CheckpointEvalSpec],
    *,
    manifest_path: str | Path | None,
    repo_root: str | Path,
    job_dir: str | Path,
    log_dir: str | Path,
    partition: str,
    qos: str | None,
    time_limit: str,
    mode: str,
    experiments: Sequence[str] | None,
    plot_types: Sequence[str],
    plot_output_roots: Sequence[str | Path],
    comparison_name: str | None,
    pilot_cap: int | None,
    dry_run: bool,
) -> tuple[Path, ...]:
    """Write evaluation scripts and submit them in the required dependency order."""

    requests = tuple(requests)
    specs, comparison_plot_types = _validate_request(
        requests,
        manifest_path=manifest_path,
        mode=mode,
        experiments=experiments,
        plot_types=plot_types,
        comparison_name=comparison_name,
    )

    if mode == "plot" and not dry_run:
        for spec in specs:
            run_dir = runner._result_dir_from_checkpoint_request(spec.checkpoint)
            runner.validate_checkpoint_results(run_dir, experiments=experiments)

    job_dir = Path(job_dir)
    log_dir = Path(log_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    script_paths = []
    evaluation_mode = "subset" if mode == "compare" else mode
    for index, request in enumerate(requests):
        script_path = job_dir / f"eval_{index:03d}.sbatch"
        script_path.write_text(
            render_eval_sbatch(
                request,
                manifest_path=manifest_path,
                repo_root=repo_root,
                log_dir=log_dir,
                partition=partition,
                qos=qos,
                time_limit=time_limit,
                mode=evaluation_mode,
                experiments=experiments,
                plot_types=plot_types,
                plot_output_roots=plot_output_roots,
                comparison_name=None,
                pilot_cap=pilot_cap,
            ),
            encoding="utf-8",
        )
        script_paths.append(script_path)

    comparison_path = None
    if mode == "compare":
        comparison_path = job_dir / "compare.sbatch"
        comparison_path.write_text(
            render_eval_sbatch(
                requests,
                manifest_path=None,
                repo_root=repo_root,
                log_dir=log_dir,
                partition=partition,
                qos=qos,
                time_limit=time_limit,
                mode="compare",
                experiments=experiments,
                plot_types=comparison_plot_types or (),
                plot_output_roots=plot_output_roots,
                comparison_name=comparison_name,
                pilot_cap=None,
            ),
            encoding="utf-8",
        )
        script_paths.append(comparison_path)

    if not dry_run:
        job_ids = [_submit_script(path) for path in script_paths[: len(requests)]]
        if comparison_path is not None:
            _submit_script(comparison_path, dependency=":".join(job_ids))
    return tuple(script_paths)


def parse_submit_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--c_train", action="append", type=int)
    parser.add_argument("--pass_gdn_state", action="append", type=_parse_bool)
    parser.add_argument("--gdn_layer_limit", action="append", type=_parse_layer_limit)
    parser.add_argument("--pass_conv_state", action="append", type=_parse_bool)
    parser.add_argument("--pass_rope_positions", action="append", type=_parse_bool)
    parser.add_argument("--pad_id", action="append", type=int)
    parser.add_argument("--eos_id", action="append", type=int)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--manifest_path")
    parser.add_argument("--experiments")
    parser.add_argument("--plot_types", type=_parse_plot_types, default=_PLOT_TYPES)
    parser.add_argument("--plot_output_root", action="append", default=[])
    parser.add_argument("--comparison_name")
    parser.add_argument("--repo_root", default=str(_REPO_ROOT))
    parser.add_argument("--job_dir", default=str(_REPO_ROOT / "eval_jobs"))
    parser.add_argument("--log_dir", default=str(_REPO_ROOT / "eval_logs"))
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--qos")
    parser.add_argument("--time_limit", default="04:00:00")
    parser.add_argument("--pilot_cap", type=int)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def _build_requests(args: argparse.Namespace) -> tuple[CheckpointEvalRequest, ...]:
    num_requests = len(args.checkpoint)
    provided_fields = tuple(field for field in _SPEC_FIELDS if getattr(args, field) is not None)
    if provided_fields and len(provided_fields) != len(_SPEC_FIELDS):
        missing_fields = tuple(field for field in _SPEC_FIELDS if field not in provided_fields)
        raise ValueError(
            "If any legacy evaluation flag is provided, all seven are required; missing: "
            + ", ".join(missing_fields)
        )
    if not provided_fields:
        return tuple(CheckpointEvalRequest(checkpoint) for checkpoint in args.checkpoint)
    for field in _SPEC_FIELDS:
        field_length = len(getattr(args, field))
        if field_length != num_requests:
            raise ValueError(
                "Repeated checkpoint spec fields must have aligned lengths: "
                f"checkpoint={num_requests}, {field}={field_length}"
            )

    return tuple(
        CheckpointEvalRequest(
            checkpoint,
            manual_config=ManualEvalConfig(
                c_train=args.c_train[index],
                pass_gdn_state=args.pass_gdn_state[index],
                gdn_layer_limit=args.gdn_layer_limit[index],
                pass_conv_state=args.pass_conv_state[index],
                pass_rope_positions=args.pass_rope_positions[index],
                pad_id=args.pad_id[index],
                eos_id=args.eos_id[index],
            ),
        )
        for index, checkpoint in enumerate(args.checkpoint)
    )


def submit_from_arguments(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    args = parse_submit_args(argv)
    requests = _build_requests(args)
    experiments = _parse_experiments(args.experiments)
    return submit_checkpoint_evals(
        requests,
        manifest_path=args.manifest_path,
        repo_root=args.repo_root,
        job_dir=args.job_dir,
        log_dir=args.log_dir,
        partition=args.partition,
        qos=args.qos,
        time_limit=args.time_limit,
        mode=args.mode,
        experiments=experiments,
        plot_types=args.plot_types,
        plot_output_roots=tuple(args.plot_output_root),
        comparison_name=args.comparison_name,
        pilot_cap=args.pilot_cap,
        dry_run=args.dry_run,
    )


def main() -> None:
    submit_from_arguments()


if __name__ == "__main__":
    main()
