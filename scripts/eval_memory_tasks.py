"""Evaluate a checkpoint on fixed paired symbolic memory examples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from absl import flags
import jax
import numpy as np

from omegalax.data.memory_tasks import (
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskSplit,
    build_memory_task_batch,
)
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY
from omegalax.evals.memory_tasks import MemoryEvalMode, evaluate_memory_batch
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.checkpoint import resolve_checkpoint, restore_model_params


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    return {
        mode: {
            "num_examples": len(mode_rows),
            "accuracy": float(np.mean([row["correct"] for row in mode_rows])),
            "answer_nll": float(np.mean([row["answer_nll"] for row in mode_rows])),
            "donor_accuracy": float(np.mean([row["donor_correct"] for row in mode_rows])),
            "donor_answer_nll": float(np.mean([row["donor_answer_nll"] for row in mode_rows])),
        }
        for mode, mode_rows in grouped.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=[task.value for task in MemoryTask], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--segment_length", type=int, default=16)
    parser.add_argument("--num_segments", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--num_rule_examples", type=int, default=5)
    parser.add_argument("--num_pairs", type=int, default=256)
    parser.add_argument("--pairs_per_batch", type=int, default=8)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--fsdp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--text_attn_backend", default="xla")
    parser.add_argument("--jax_cache_dir", type=Path, default=Path("/tmp/jax_cache"))
    return parser


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _manifest(config: MemoryTaskConfig, num_pairs: int) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "task": str(config.task),
        "split": str(config.split),
        "seed": config.seed,
        "segment_length": config.segment_length,
        "num_segments": config.num_segments,
        "vocab_size": config.vocab_size,
        "num_rule_examples": config.num_rule_examples,
        "pair_indices": list(range(num_pairs)),
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest | {"sha256": hashlib.sha256(payload).hexdigest()}


def _validate_training_contract(checkpoint_root: Path, task: MemoryTask) -> None:
    contract_path = checkpoint_root / "memory_task_contract.json"
    if not contract_path.is_file():
        return
    contract = json.loads(contract_path.read_text())
    trained_split = contract["task"]["split"]
    if trained_split != MemoryTaskSplit.TRAIN.value:
        raise ValueError(
            f"Refusing to evaluate {checkpoint_root}: memory_task_contract.json records "
            f"split={trained_split!r}, expected {MemoryTaskSplit.TRAIN.value!r}"
        )
    trained_task = contract["task"]["task"]
    if trained_task != task.value:
        raise ValueError(
            f"Refusing to evaluate {checkpoint_root} on task={task.value!r}: "
            f"memory_task_contract.json was trained on task={trained_task!r}"
        )


def _revision() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _plot(summary: dict[str, dict[str, float]], plots_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [mode.value for mode in MemoryEvalMode]
    x = np.arange(len(modes))
    labels = [mode.replace("_", "\n") for mode in modes]
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    width = 0.38
    axis.bar(
        x - width / 2, [summary[mode]["accuracy"] for mode in modes], width, label="own answer"
    )
    axis.bar(
        x + width / 2,
        [summary[mode]["donor_accuracy"] for mode in modes],
        width,
        label="paired-world answer",
    )
    axis.set_ylabel("Exact one-token accuracy")
    axis.set_ylim(0, 1)
    axis.set_xticks(x, labels)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "accuracy.svg", format="svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(
        x - width / 2, [summary[mode]["answer_nll"] for mode in modes], width, label="own answer"
    )
    axis.bar(
        x + width / 2,
        [summary[mode]["donor_answer_nll"] for mode in modes],
        width,
        label="paired-world answer",
    )
    axis.set_ylabel("Answer-token NLL")
    axis.set_xticks(x, labels)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "answer_nll.svg", format="svg")
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["eval_memory_tasks.py"])
    if args.num_pairs <= 0 or args.num_pairs % args.pairs_per_batch:
        raise ValueError("num_pairs must be positive and divisible by pairs_per_batch")
    jax.config.update("jax_compilation_cache_dir", str(args.jax_cache_dir))
    task_config = MemoryTaskConfig(
        task=args.task,
        seed=args.seed,
        segment_length=args.segment_length,
        num_segments=args.num_segments,
        vocab_size=args.vocab_size,
        num_rule_examples=args.num_rule_examples,
        split=MemoryTaskSplit.EVAL,
    )
    resolved = resolve_checkpoint(args.checkpoint)
    _validate_training_contract(resolved.root, task_config.task)
    model_config = text_api.resolve_config(str(resolved.config_path))
    if task_config.vocab_size > model_config.vocab_size:
        raise ValueError(
            f"Task vocab_size={task_config.vocab_size} exceeds model vocab_size={model_config.vocab_size}"
        )
    model, model_config = text_api.init_model(
        model_config,
        jax.random.key(0),
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
        dp_size=args.dp_size,
    )
    restore_model_params(model, resolved)
    set_attn_backend(model, text_backend=args.text_attn_backend)

    eval_name = f"memory_{task_config.task.value}_v1"
    run_dir = resolved.root / "evals" / eval_name / f"step_{resolved.step}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(task_config, args.num_pairs)
    config = {
        "schema_version": 1,
        "checkpoint_root": str(resolved.root),
        "checkpoint_step": resolved.step,
        "manifest_sha256": manifest["sha256"],
        "pairs_per_batch": args.pairs_per_batch,
        "conditions": [mode.value for mode in MemoryEvalMode],
        "revision": _revision(),
    }
    for path, value in (
        (run_dir / "manifest.json", manifest),
        (run_dir / "eval_config.json", config),
    ):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to change immutable eval input: {path}")
        _write_json(path, value)
    _write_json(run_dir / "status.json", {"status": "running"})

    rows: list[dict[str, Any]] = []
    for start in range(0, args.num_pairs, args.pairs_per_batch):
        batch = build_memory_task_batch(
            tuple(range(start, start + args.pairs_per_batch)),
            task_config,
            flatten=False,
        )
        metadata = dict(batch[BATCH_PRETRAIN_METADATA_KEY])
        answers_B = np.asarray(metadata["answer_token_B"])
        donor_answers_B = answers_B[np.arange(answers_B.size) ^ 1]
        results = evaluate_memory_batch(model, model_config, batch)
        for mode, result in results.items():
            for idx in range(answers_B.size):
                prediction = int(result.predictions_B[idx])
                rows.append(
                    {
                        "mode": mode.value,
                        "pair_index": int(metadata["pair_index_B"][idx]),
                        "world": int(metadata["world_B"][idx]),
                        "answer_token": int(answers_B[idx]),
                        "donor_answer_token": int(donor_answers_B[idx]),
                        "prediction": prediction,
                        "correct": prediction == int(answers_B[idx]),
                        "donor_correct": prediction == int(donor_answers_B[idx]),
                        "answer_nll": float(result.answer_nll_B[idx]),
                        "donor_answer_nll": float(result.donor_answer_nll_B[idx]),
                    }
                )

    raw_path = run_dir / "raw.jsonl"
    temporary = raw_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(raw_path)
    mode_summary = summarize_rows(rows)
    effects = {
        "correct_minus_counterfactual_accuracy": mode_summary["correct_state"]["accuracy"]
        - mode_summary["counterfactual_state"]["accuracy"],
        "counterfactual_donor_minus_own_accuracy": mode_summary["counterfactual_state"][
            "donor_accuracy"
        ]
        - mode_summary["counterfactual_state"]["accuracy"],
        "correct_minus_counterfactual_answer_nll": mode_summary["counterfactual_state"][
            "answer_nll"
        ]
        - mode_summary["correct_state"]["answer_nll"],
    }
    summary = {"modes": mode_summary, "causal_effects": effects}
    _write_json(run_dir / "summary.json", summary)
    _plot(mode_summary, run_dir / "plots")
    _write_json(run_dir / "status.json", {"status": "complete"})
    print(json.dumps({"run_dir": str(run_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
