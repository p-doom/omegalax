"""Local CLI for checkpoint state-usage evaluations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from absl import flags

from omegalax.evals.runner import CheckpointEvalRequest, run_evals
from omegalax.training_contract import ManualEvalConfig


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


def parse_worker_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--plot_types", type=_parse_plot_types, required=True)
    parser.add_argument("--plot_output_root", action="append", default=[])
    parser.add_argument("--document_cap", type=int)
    parser.add_argument("--comparison_name")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--fsdp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
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
                c_train=args.c_train[idx],
                pass_gdn_state=args.pass_gdn_state[idx],
                gdn_layer_limit=args.gdn_layer_limit[idx],
                pass_conv_state=args.pass_conv_state[idx],
                pass_rope_positions=args.pass_rope_positions[idx],
                pad_id=args.pad_id[idx],
                eos_id=args.eos_id[idx],
            ),
        )
        for idx, checkpoint in enumerate(args.checkpoint)
    )


def _parse_experiments(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    experiments = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = tuple(item for item in experiments if item not in _EXPERIMENTS)
    if not experiments or unknown:
        raise ValueError(f"experiments must contain only {', '.join(_EXPERIMENTS)}; got {value!r}")
    return experiments


def run_from_arguments(argv: Sequence[str] | None = None):
    args = parse_worker_args(argv)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["run_checkpoint_evals.py"])
    requests = _build_requests(args)
    experiments = _parse_experiments(args.experiments)

    if args.mode in ("all", "subset") and args.manifest_path is None:
        raise ValueError(f"mode={args.mode} requires manifest_path")
    if args.mode == "compare" and len(requests) < 2:
        raise ValueError("Compare mode requires at least two checkpoint specs")
    if args.mode == "compare" and not args.comparison_name:
        raise ValueError("Compare mode requires a comparison name")

    return run_evals(
        requests,
        mode=args.mode,
        manifest_path=args.manifest_path,
        experiments=experiments,
        plot_types=args.plot_types,
        plot_output_roots=tuple(args.plot_output_root),
        document_cap=args.document_cap,
        comparison_name=args.comparison_name,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
        dp_size=args.dp_size,
        batch_size=args.batch_size,
    )


def main() -> None:
    run_from_arguments()


if __name__ == "__main__":
    main()
