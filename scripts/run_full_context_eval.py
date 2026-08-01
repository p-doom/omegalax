"""Run or compare the off-the-shelf Qwen3.5 full-context evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from absl import flags

from omegalax.evals.full_context import compare_relative_nll, run_full_context_eval


def _candidate(value: str) -> tuple[str, str]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=RESULT_PATH")
    return label, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--model_path", required=True)
    run.add_argument("--manifest_path", required=True)
    run.add_argument("--output_dir", required=True)
    run.add_argument("--c_train", type=int, default=4)
    run.add_argument("--pad_id", type=int, default=248046)
    run.add_argument("--document_cap", type=int)
    run.add_argument("--min_doc_chunks", type=int, default=2)
    run.add_argument("--max_doc_chunks", type=int, default=16)
    run.add_argument("--tp_size", type=int, default=1)
    run.add_argument("--fsdp_size", type=int, default=1)
    run.add_argument("--dp_size", type=int, default=1)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline_result", required=True)
    compare.add_argument("--candidate", action="append", type=_candidate, required=True)
    compare.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["run_full_context_eval.py"])
    if args.mode == "compare":
        return compare_relative_nll(
            baseline_result=args.baseline_result,
            candidate_results=args.candidate,
            output_dir=args.output_dir,
        )

    return run_full_context_eval(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        c_train=args.c_train,
        pad_id=args.pad_id,
        document_cap=args.document_cap,
        min_doc_chunks=args.min_doc_chunks,
        max_doc_chunks=args.max_doc_chunks,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
        dp_size=args.dp_size,
    )


if __name__ == "__main__":
    main()
