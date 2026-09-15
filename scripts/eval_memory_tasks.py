"""Evaluate a checkpoint on fixed paired symbolic memory examples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


def _require_slurm_job() -> None:
    slurm_node = os.environ.get("SLURMD_NODENAME", "").split(".", 1)[0]
    current_node = socket.gethostname().split(".", 1)[0]
    if not os.environ.get("SLURM_JOB_ID") or not slurm_node or slurm_node != current_node:
        raise RuntimeError(
            "Refusing to run a memory-task evaluation outside a Slurm compute node; "
            "submit it with sbatch"
        )


if __name__ == "__main__" and not {"-h", "--help"}.intersection(sys.argv[1:]):
    _require_slurm_job()


from absl import flags  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from omegalax.data.memory_tasks import (  # noqa: E402
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskCue,
    MemoryTaskSplit,
    build_memory_task_batch,
)
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY  # noqa: E402
from omegalax.evals.memory_tasks import MemoryEvalMode, make_memory_evaluator  # noqa: E402
from omegalax.evals.memory_pollution import (  # noqa: E402
    load_training_streams,
    validate_pollution_free_manifest,
)
from omegalax.models.qwen3_5.kernels import _resolve_backend  # noqa: E402
from omegalax.models.sharding_runtime import set_attn_backend  # noqa: E402
from omegalax.text import api as text_api  # noqa: E402
from omegalax.text.checkpoint import (  # noqa: E402
    resolve_checkpoint,
    restore_model_params,
)


# target_slot/target_segment/target_age double as recency rank (segment 0 writes
# slots in temporal order), separating "the state can't hold K items" (flat across
# strata) from "the model defaults to a recency policy" (accuracy skewed toward the
# most recent slot/segment/age). query_namespace has no such ordering and just
# isolates per-namespace rule-bank performance.
_STRATA_FIELDS = ("target_slot", "target_segment", "target_age", "query_namespace")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for mode, mode_rows in grouped.items():
        mode_summary: dict[str, Any] = {
            "num_examples": len(mode_rows),
            "accuracy": float(np.mean([row["correct"] for row in mode_rows])),
            "answer_nll": float(np.mean([row["answer_nll"] for row in mode_rows])),
            "donor_accuracy": float(np.mean([row["donor_correct"] for row in mode_rows])),
            "donor_answer_nll": float(np.mean([row["donor_answer_nll"] for row in mode_rows])),
        }
        for field in _STRATA_FIELDS:
            by_value: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in mode_rows:
                if field in row:
                    by_value[int(row[field])].append(row)
            if by_value:
                mode_summary[f"accuracy_by_{field}"] = {
                    str(value): float(np.mean([row["correct"] for row in value_rows]))
                    for value, value_rows in sorted(by_value.items())
                }
                mode_summary[f"answer_nll_by_{field}"] = {
                    str(value): float(np.mean([row["answer_nll"] for row in value_rows]))
                    for value, value_rows in sorted(by_value.items())
                }
        summary[mode] = mode_summary
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument(
        "--allow_zero_shot_horizon_override",
        action="store_true",
        help=(
            "Allow only segment_length and/or num_segments to differ from the "
            "checkpoint's training task contract."
        ),
    )
    parser.add_argument("--eval_role", choices=("gate", "final"))
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--stage1_receipt", type=Path)
    parser.add_argument("--num_pairs", type=int, required=True)
    parser.add_argument("--pairs_per_batch", type=int, default=8)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--fsdp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--text_attn_backend", default="mosaic_gpu")
    parser.add_argument("--jax_cache_dir", type=Path, required=True)
    parser.add_argument("--wiki_mirror_dir", type=Path)
    return parser


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_stage1_json(path: Path, label: str) -> dict[str, Any]:
    try:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} is not a regular file: {path}")
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable Stage-1 {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Stage-1 {label} must be a JSON object: {path}")
    return value


def _require_stage1_self_hash(value: dict[str, Any], label: str) -> None:
    if value.get("sha256") != _canonical_json_sha256(
        {name: item for name, item in value.items() if name != "sha256"}
    ):
        raise ValueError(f"Stage-1 {label} self-hash mismatch")


def _checkpoint_tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Stage-1 checkpoint is not a directory: {root}")
    rows: list[tuple[str, str]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            child = current_path / name
            if not child.is_dir() or child.is_symlink():
                raise ValueError(f"Stage-1 checkpoint has invalid directory: {child}")
            rows.append((child.relative_to(root).as_posix(), "D"))
        for name in files:
            child = current_path / name
            if not child.is_file() or child.is_symlink():
                raise ValueError(f"Stage-1 checkpoint has invalid file: {child}")
            rows.append((child.relative_to(root).as_posix(), "F:" + _file_sha256(child)))
    digest = hashlib.sha256()
    for name, value in sorted(rows):
        digest.update(name.encode() + b"\0" + value.encode() + b"\n")
    return digest.hexdigest()


# seed/split are expected to differ between a training contract and the
# requested eval config (eval always uses split=EVAL, and often a distinct
# seed); every other field is part of the frozen task semantics.
_MEMORY_TASK_CONTRACT_ALLOWED_DIFFERENCES = frozenset({"seed", "split"})
_ZERO_SHOT_HORIZON_OVERRIDE_FIELDS = frozenset({"segment_length", "num_segments"})

# ASSOCIATIVE_BANK's 3072 is the frozen exhaustive 128 banks x 24 slots
# design; RULE_BANK's 2048 is 256 banks x 8 namespaces (also the original G1
# canonical count reused for EXACT/DOMINANT_RULE) -- not a generic
# comparable-stratum rationale.
_CANONICAL_NUM_PAIRS = {
    MemoryTask.EXACT: 2048,
    MemoryTask.DOMINANT_RULE: 2048,
    MemoryTask.RULE_BANK: 2048,
    MemoryTask.ASSOCIATIVE_BANK: 3072,
}


def _require_canonical_num_pairs(task: str, num_pairs: int) -> None:
    expected = _CANONICAL_NUM_PAIRS[MemoryTask(task)]
    if num_pairs != expected:
        raise ValueError(
            f"--num_pairs must be exactly {expected} for task={task!r}, got {num_pairs}"
        )


def _read_training_contract(checkpoint_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = checkpoint_root / "memory_task_contract.json"
    if not contract_path.is_file():
        raise ValueError(f"Refusing to evaluate without memory_task_contract.json: {contract_path}")
    try:
        contract = json.loads(contract_path.read_text())
        trained_task = dict(contract["task"])
        trained_task.setdefault("generator_version", 1)
        trained_task.setdefault("bank_value_alphabet", 256)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Unreadable memory-task contract: {contract_path}") from exc
    if not isinstance(contract, dict) or not isinstance(trained_task, dict):
        raise ValueError(f"Unreadable memory-task contract: {contract_path}")
    return contract, trained_task


def _validate_training_contract(
    checkpoint_root: Path,
    task_config: MemoryTaskConfig,
    *,
    allow_zero_shot_horizon_override: bool = False,
) -> str:
    contract, trained_task = _read_training_contract(checkpoint_root)
    requested_task = dataclasses.asdict(task_config)
    if set(trained_task) != set(requested_task):
        raise ValueError(
            f"Refusing to evaluate {checkpoint_root}: memory-task contract has a different "
            f"task field set (trained={sorted(trained_task)!r}, requested={sorted(requested_task)!r})"
        )
    try:
        trained_split = trained_task["split"]
    except KeyError as exc:
        raise ValueError(
            f"Unreadable memory-task contract: {checkpoint_root / 'memory_task_contract.json'}"
        ) from exc
    if trained_split != MemoryTaskSplit.TRAIN.value:
        raise ValueError(
            f"Refusing to evaluate {checkpoint_root}: memory_task_contract.json records "
            f"split={trained_split!r}, expected {MemoryTaskSplit.TRAIN.value!r}"
        )
    allowed_differences = _MEMORY_TASK_CONTRACT_ALLOWED_DIFFERENCES
    if allow_zero_shot_horizon_override:
        allowed_differences = allowed_differences | _ZERO_SHOT_HORIZON_OVERRIDE_FIELDS
    mismatches = {
        name: (trained_task.get(name), value)
        for name, value in requested_task.items()
        if name not in allowed_differences and trained_task.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"Refusing to evaluate {checkpoint_root} on a mismatched memory-task config: "
            f"{mismatches!r}"
        )
    if allow_zero_shot_horizon_override and not any(
        trained_task.get(name) != requested_task[name]
        for name in _ZERO_SHOT_HORIZON_OVERRIDE_FIELDS
    ):
        raise ValueError(
            "--allow_zero_shot_horizon_override requires segment_length and/or "
            "num_segments to differ from the training task contract"
        )
    return _canonical_json_sha256(contract)


def _eval_name(
    task_config: MemoryTaskConfig,
    *,
    allow_zero_shot_horizon_override: bool,
    eval_role: str | None,
) -> str:
    if allow_zero_shot_horizon_override:
        name = (
            f"memory_{task_config.task.value}_zero_shot_horizon_"
            f"c{task_config.num_segments}_l{task_config.segment_length}"
        )
    else:
        name = f"memory_{task_config.task.value}"
    role = f"_{eval_role}" if eval_role is not None else ""
    return f"{name}{role}_pollution_guard_v1"


def _validate_stage1_common_c_holdout(
    manifest: dict[str, Any],
    path: Path,
    task_config: MemoryTaskConfig,
    *,
    num_pairs: int,
    checkpoint_root: Path,
    checkpoint_step: int,
    training_contract_sha256: str,
    eval_role: str | None,
    receipt_path: Path | None,
) -> None:
    if receipt_path is None:
        raise ValueError("STAGE1_COMMON_C_HOLDOUT requires --stage1_receipt")
    try:
        manifest_task = MemoryTaskConfig(**manifest["task"])
        protection = manifest["protection"]
        selection = manifest["selection"]
        pair_indices = manifest["pair_indices"]
        family = manifest["family"]
        c = manifest["c"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unreadable STAGE1_COMMON_C_HOLDOUT manifest: {path}") from exc
    _require_stage1_self_hash(manifest, "manifest")
    if (
        manifest_task != task_config
        or family != f"{task_config.task.value}_{task_config.relevance_cue.value}"
        or c != task_config.num_segments
        or manifest_task.num_segments != c
    ):
        raise ValueError("STAGE1_COMMON_C_HOLDOUT task, family, or C does not match request")
    if len(pair_indices) != num_pairs or len(set(pair_indices)) != num_pairs:
        raise ValueError("STAGE1_COMMON_C_HOLDOUT pair indices do not match --num_pairs")
    if not all(isinstance(index, int) for index in pair_indices):
        raise ValueError("STAGE1_COMMON_C_HOLDOUT pair indices must be integers")
    if (
        not isinstance(protection, dict)
        or protection.get("policy") != "filler_free_tables_and_pairs_v1"
        or not isinstance(selection, dict)
        or selection.get("selection_coordinate_only") is not True
        or selection.get("cross_c_paired_task_claim") is not False
        or selection.get("candidate_cap") != 204800
    ):
        raise ValueError("STAGE1_COMMON_C_HOLDOUT protection or selection is invalid")
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _read_stage1_json(receipt_path, "endpoint receipt")
    _require_stage1_self_hash(receipt, "endpoint receipt")
    stage1_root = receipt_path.parent.parent
    if receipt_path.parent != stage1_root / "receipts":
        raise ValueError(f"Stage-1 receipt is outside the receipts directory: {receipt_path}")
    expected_manifest_path = stage1_root / "manifests" / family / f"c{c}.json"
    manifest_binding = receipt.get("manifest")
    if (
        receipt.get("kind") != "STAGE1_ENDPOINT_C_OVERLAP_RECEIPT"
        or receipt.get("family") != family
        or receipt.get("c") != c
        or not isinstance(manifest_binding, dict)
        or manifest_binding.get("path") != str(expected_manifest_path.relative_to(stage1_root))
        or expected_manifest_path != path
        or manifest_binding.get("file_sha256") != _file_sha256(path)
        or manifest_binding.get("self_sha256") != manifest.get("sha256")
    ):
        raise ValueError("Stage-1 receipt does not bind the requested endpoint and manifest")
    inventory = receipt.get("common_inventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("source_projections_path") != "source_projections.json"
        or inventory.get("ancestry_path") != "ancestry_inventory.json"
    ):
        raise ValueError("Stage-1 receipt has invalid common inventory paths")
    source_path = stage1_root / inventory["source_projections_path"]
    ancestry_path = stage1_root / inventory["ancestry_path"]
    source_projections = _read_stage1_json(source_path, "source projections")
    ancestry = _read_stage1_json(ancestry_path, "ancestry inventory")
    if (
        protection.get("source_projection_sha256") != _file_sha256(source_path)
        or protection.get("ancestry_inventory_sha256") != _file_sha256(ancestry_path)
        or inventory.get("source_projections_sha256") != _file_sha256(source_path)
        or inventory.get("ancestry_sha256") != _file_sha256(ancestry_path)
        or source_projections.get("kind") != "STAGE1_EXCLUSION_INVENTORY"
        or ancestry.get("kind") != "STAGE1_ANCESTRY_INVENTORY"
    ):
        raise ValueError("Stage-1 source projection or ancestry binding mismatch")
    source_keys = protection.get("compatible_source_keys")
    stream_ids = protection.get("training_stream_ids")
    if (
        not isinstance(source_keys, list)
        or len(source_keys) != len(set(source_keys))
        or receipt.get("applicable_source_keys") != source_keys
        or not isinstance(stream_ids, list)
        or len(stream_ids) != len(set(stream_ids))
        or receipt.get("applicable_training_stream_ids") != stream_ids
    ):
        raise ValueError("Stage-1 receipt protection bindings mismatch")
    projection_by_key = {
        item.get("source_key"): item
        for item in source_projections.get("sources", [])
        if isinstance(item, dict)
    }
    if len(projection_by_key) != len(source_projections.get("sources", [])) or any(
        projection_by_key.get(key, {}).get("family_assignment") != family
        or projection_by_key[key].get("applicability", {}).get(family) != "APPLICABLE"
        for key in source_keys
    ):
        raise ValueError("Stage-1 source projection does not cover the manifest family")
    overlap = receipt.get("overlap")
    if (
        not isinstance(overlap, dict)
        or overlap.get("pair_core_intersection_count") != 0
        or overlap.get("world_table_intersection_count") != 0
        or overlap.get("candidate_pair_digest_set_sha256")
        != manifest.get("concrete_pair_digest_set_sha256")
        or overlap.get("candidate_table_digest_set_sha256")
        != manifest.get("concrete_table_digest_set_sha256")
        or overlap.get("pair_protection_digest_set_sha256")
        != protection.get("pair_protection_digest_set_sha256")
        or overlap.get("table_protection_digest_set_sha256")
        != protection.get("table_protection_digest_set_sha256")
    ):
        raise ValueError("Stage-1 receipt does not prove null overlap")
    endpoint_records = [
        item
        for item in ancestry.get("endpoints", [])
        if isinstance(item, dict) and item.get("endpoint") == receipt["endpoint"]
    ]
    if len(endpoint_records) != 1:
        raise ValueError("Stage-1 ancestry has no unique endpoint record")
    endpoint_record = endpoint_records[0]
    expected_checkpoint = checkpoint_root / f"{checkpoint_step:06d}"
    if (
        Path(endpoint_record.get("checkpoint", "")).expanduser().resolve() != expected_checkpoint
        or inventory.get("endpoint_ancestry_record_sha256")
        != _canonical_json_sha256(endpoint_record)
        or endpoint_record.get("checkpoint_tree_sha256")
        != _checkpoint_tree_sha256(expected_checkpoint)
    ):
        raise ValueError("Stage-1 ancestry checkpoint binding mismatch")
    try:
        actual_streams = load_training_streams(checkpoint_root, checkpoint_step)
        recorded_streams = endpoint_record["streams"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Stage-1 ancestry has unreadable training streams") from exc
    if len(recorded_streams) != len(actual_streams):
        raise ValueError("Stage-1 ancestry length does not match checkpoint ancestry")
    for recorded, actual in zip(recorded_streams, actual_streams, strict=True):
        contract_path = actual.checkpoint_root / "memory_task_contract.json"
        try:
            recorded_task = MemoryTaskConfig(**recorded["task"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stage-1 ancestry has invalid training task") from exc
        if (
            recorded.get("checkpoint_root") != str(actual.checkpoint_root)
            or recorded.get("checkpoint_step") != actual.checkpoint_step
            or recorded.get("contract_path") != str(contract_path)
            or recorded.get("contract_file_sha256") != _file_sha256(contract_path)
            or recorded.get("contract_canonical_sha256") != actual.contract_sha256
            or recorded.get("train_pairs") != actual.train_pairs
            or recorded_task != actual.config
        ):
            raise ValueError("Stage-1 ancestry contract binding mismatch")
    if actual_streams[0].contract_sha256 != training_contract_sha256:
        raise ValueError("Evaluated checkpoint contract changed while loading Stage-1 manifest")
    inventory_streams = {
        item.get("stream_id"): item
        for item in ancestry.get("deduplicated_training_streams", [])
        if isinstance(item, dict)
    }
    if any(
        stream_id not in inventory_streams
        or inventory_streams[stream_id].get("family") != family
        or receipt["endpoint"]
        not in inventory_streams[stream_id].get("originating_endpoint_keys", [])
        for stream_id in stream_ids
    ):
        raise ValueError("Stage-1 ancestry does not bind the protected training streams")
    if eval_role != "final":
        raise ValueError("STAGE1_COMMON_C_HOLDOUT requires eval_role=final")


def _load_manifest(
    path: Path,
    task_config: MemoryTaskConfig,
    *,
    num_pairs: int,
    checkpoint_root: Path,
    checkpoint_step: int,
    training_contract_sha256: str,
    eval_role: str | None,
    stage1_receipt: Path | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        manifest = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable memory-task eval manifest: {path}") from exc
    if isinstance(manifest, dict) and manifest.get("kind") == "STAGE1_COMMON_C_HOLDOUT":
        manifest = _read_stage1_json(path, "eval manifest")
        _validate_stage1_common_c_holdout(
            manifest,
            path,
            task_config,
            num_pairs=num_pairs,
            checkpoint_root=checkpoint_root,
            checkpoint_step=checkpoint_step,
            training_contract_sha256=training_contract_sha256,
            eval_role=eval_role,
            receipt_path=stage1_receipt,
        )
        return manifest
    if stage1_receipt is not None:
        raise ValueError("--stage1_receipt is only valid with STAGE1_COMMON_C_HOLDOUT")
    try:
        manifest_config = MemoryTaskConfig(
            **{
                field.name: manifest.get(field.name, field.default)
                for field in dataclasses.fields(MemoryTaskConfig)
            }
        )
        pair_indices = manifest["pair_indices"]
        training_streams = manifest["pollution_guard"]["training_streams"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unreadable memory-task eval manifest: {path}") from exc
    validate_pollution_free_manifest(manifest)
    final_holdout = manifest["pollution_guard"].get("final_holdout")
    if eval_role == "final" and final_holdout is not True:
        raise ValueError("eval_role=final requires a final-holdout eval manifest")
    if eval_role == "gate" and final_holdout is True:
        raise ValueError("eval_role=gate rejects a final-holdout eval manifest")
    if manifest_config != task_config:
        raise ValueError(
            f"Eval manifest task config does not match request: "
            f"manifest={manifest_config!r}, requested={task_config!r}"
        )
    if len(pair_indices) != num_pairs or len(set(pair_indices)) != num_pairs:
        raise ValueError(
            f"Eval manifest must contain exactly {num_pairs} unique pair indices, "
            f"got {len(pair_indices)} entries and {len(set(pair_indices))} unique values"
        )
    required_streams = load_training_streams(checkpoint_root, checkpoint_step)
    expected_references = [
        {
            "checkpoint_root": str(stream.checkpoint_root),
            "checkpoint_step": stream.checkpoint_step,
            "contract_sha256": stream.contract_sha256,
            "train_pairs": stream.train_pairs,
        }
        for stream in required_streams
    ]
    if expected_references[0]["contract_sha256"] != training_contract_sha256:
        raise ValueError("Evaluated checkpoint contract changed while loading its eval manifest")
    missing_references = [
        expected
        for expected in expected_references
        if not any(
            all(reference.get(name) == value for name, value in expected.items())
            for reference in training_streams
        )
    ]
    if missing_references:
        raise ValueError(
            "Eval manifest pollution guard does not include the complete checkpoint ancestry: "
            f"{missing_references}"
        )
    return manifest


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
    relevant_paths = (
        Path(__file__),
        repo / "omegalax/data/memory_tasks.py",
        repo / "omegalax/evals/memory_tasks.py",
        repo / "omegalax/evals/memory_pollution.py",
        repo / "scripts/build_memory_task_eval_manifest.py",
    )
    digest = hashlib.sha256()
    for path in relevant_paths:
        resolved = path.resolve()
        digest.update(str(resolved.relative_to(repo)).encode() + b"\0")
        digest.update(resolved.read_bytes())
    return {"commit": commit, "dirty": dirty, "content_sha256": digest.hexdigest()}


def _ensure_not_complete(run_dir: Path, manifest: dict[str, Any], config: dict[str, Any]) -> None:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return
    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unreadable eval status at {status_path}") from exc
    if status.get("status") != "complete":
        return
    for path, value in (
        (run_dir / "manifest.json", manifest),
        (run_dir / "eval_config.json", config),
    ):
        if not path.is_file() or json.loads(path.read_text()) != value:
            raise ValueError(f"Completed eval has inconsistent immutable input: {path}")
    _verify_artifact_manifest(run_dir)
    raise ValueError(f"Eval is already complete; refusing to overwrite {run_dir}")


def _verify_artifact_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Completed eval is missing artifact manifest: {manifest_path}")
    try:
        artifact_manifest = json.loads(manifest_path.read_text())
        artifacts = artifact_manifest["artifacts"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Completed eval has an unreadable artifact manifest: {manifest_path}"
        ) from exc
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"Completed eval has an empty artifact manifest: {manifest_path}")
    resolved_run_dir = run_dir.resolve()
    for relative_path, expected_sha256 in artifacts.items():
        path = (run_dir / relative_path).resolve()
        if not path.is_relative_to(resolved_run_dir):
            raise ValueError(f"Artifact manifest escapes eval directory: {relative_path!r}")
        if not path.is_file():
            raise ValueError(f"Completed eval is missing artifact: {relative_path}")
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Completed eval artifact hash mismatch for {relative_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )


def _write_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    artifact_paths = (
        run_dir / "raw.jsonl",
        run_dir / "summary.json",
        run_dir / "plots" / "accuracy.svg",
        run_dir / "plots" / "answer_nll.svg",
    )
    artifacts = {str(path.relative_to(run_dir)): _file_sha256(path) for path in artifact_paths}
    manifest = {"schema_version": 1, "artifacts": artifacts}
    _write_json(run_dir / "artifact_manifest.json", manifest)
    return manifest


def _write_mirror_file(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different Wiki mirror file: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _mirror_svgs(run_dir: Path, mirror_dir: Path, artifact_manifest: dict[str, Any]) -> None:
    mirror_dir = mirror_dir.expanduser().resolve()
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for name in ("accuracy.svg", "answer_nll.svg"):
        _write_mirror_file(mirror_dir / name, (run_dir / "plots" / name).read_bytes())
    readme = (
        "# Controlled memory eval\n\n"
        f"Source: {run_dir}\n\n"
        f"Artifact manifest SHA256: {_canonical_json_sha256(artifact_manifest)}\n"
    )
    _write_mirror_file(mirror_dir / "README.md", readme.encode())


def _plot(summary: dict[str, dict[str, float]], plots_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "omegalax-memory-eval-v1"
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
    figure.savefig(plots_dir / "accuracy.svg", format="svg", metadata={"Date": None})
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
    figure.savefig(plots_dir / "answer_nll.svg", format="svg", metadata={"Date": None})
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _require_slurm_job()
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["eval_memory_tasks.py"])
    if args.num_pairs <= 0 or args.num_pairs % args.pairs_per_batch:
        raise ValueError("num_pairs must be positive and divisible by pairs_per_batch")
    if (args.tp_size, args.fsdp_size, args.dp_size) != (1, 1, 1):
        raise ValueError(
            "Memory evaluation currently supports only one device with tp_size=fsdp_size=dp_size=1"
        )
    jax.config.update("jax_compilation_cache_dir", str(args.jax_cache_dir))
    if jax.device_count() != 1:
        raise ValueError(
            f"Memory evaluation currently requires exactly one visible JAX device, got {jax.device_count()}"
        )
    visible_devices = jax.local_devices()
    runtime = {
        "jax_backend": jax.default_backend(),
        "visible_device_platforms": sorted({device.platform for device in visible_devices}),
        "visible_device_count": len(visible_devices),
        "deltanet_kernel": _resolve_backend(),
    }
    task_config = MemoryTaskConfig(
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
    resolved = resolve_checkpoint(args.checkpoint)
    training_contract_sha256 = _validate_training_contract(
        resolved.root,
        task_config,
        allow_zero_shot_horizon_override=args.allow_zero_shot_horizon_override,
    )
    _, source_training_task = _read_training_contract(resolved.root)
    _require_canonical_num_pairs(source_training_task["task"], args.num_pairs)
    eval_name = _eval_name(
        task_config,
        allow_zero_shot_horizon_override=args.allow_zero_shot_horizon_override,
        eval_role=args.eval_role,
    )
    run_dir = resolved.root / "evals" / eval_name / f"step_{resolved.step}"
    manifest = _load_manifest(
        args.eval_manifest,
        task_config,
        num_pairs=args.num_pairs,
        checkpoint_root=resolved.root,
        checkpoint_step=resolved.step,
        training_contract_sha256=training_contract_sha256,
        eval_role=args.eval_role,
        stage1_receipt=args.stage1_receipt,
    )
    config = {
        "schema_version": 4,
        "checkpoint_root": str(resolved.root),
        "checkpoint_step": resolved.step,
        "manifest_path": str(args.eval_manifest.expanduser().resolve()),
        "manifest_sha256": manifest["sha256"],
        "training_contract_sha256": training_contract_sha256,
        "source_training_task": source_training_task,
        "requested_eval_task": dataclasses.asdict(task_config),
        "zero_shot_horizon_override": args.allow_zero_shot_horizon_override,
        "allowed_training_task_differences": sorted(
            _MEMORY_TASK_CONTRACT_ALLOWED_DIFFERENCES
            | (
                _ZERO_SHOT_HORIZON_OVERRIDE_FIELDS
                if args.allow_zero_shot_horizon_override
                else frozenset()
            )
        ),
        "pairs_per_batch": args.pairs_per_batch,
        "topology": {"tp_size": args.tp_size, "fsdp_size": args.fsdp_size, "dp_size": args.dp_size},
        "text_attn_backend": args.text_attn_backend,
        "runtime": runtime,
        "conditions": [mode.value for mode in MemoryEvalMode],
        "revision": _revision(),
    }
    if args.eval_role is not None:
        config["eval_role"] = args.eval_role
    if args.stage1_receipt is not None:
        stage1_receipt_path = args.stage1_receipt.expanduser().resolve()
        config["stage1_receipt"] = {
            "path": str(stage1_receipt_path),
            "file_sha256": _file_sha256(stage1_receipt_path),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_not_complete(run_dir, manifest, config)
    for path, value in (
        (run_dir / "manifest.json", manifest),
        (run_dir / "eval_config.json", config),
    ):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to change immutable eval input: {path}")
        _write_json(path, value)
    _write_json(run_dir / "status.json", {"status": "running"})
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
    evaluate_batch = make_memory_evaluator(model_config)

    rows: list[dict[str, Any]] = []
    pair_indices = [int(index) for index in manifest["pair_indices"]]
    for start in range(0, args.num_pairs, args.pairs_per_batch):
        batch = build_memory_task_batch(
            tuple(pair_indices[start : start + args.pairs_per_batch]),
            task_config,
            flatten=False,
        )
        metadata = dict(batch[BATCH_PRETRAIN_METADATA_KEY])
        answers_B = np.asarray(metadata["answer_token_B"])
        donor_answers_B = answers_B[np.arange(answers_B.size) ^ 1]
        results = evaluate_batch(model, batch)
        for mode, result in results.items():
            for idx in range(answers_B.size):
                prediction = int(result.predictions_B[idx])
                row = {
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
                for row_key, metadata_key in (
                    ("target_slot", "target_slot_B"),
                    ("target_segment", "target_segment_B"),
                    ("target_age", "target_age_B"),
                    ("query_namespace", "query_namespace_B"),
                ):
                    if metadata_key in metadata:
                        row[row_key] = int(metadata[metadata_key][idx])
                rows.append(row)

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
    artifact_manifest = _write_artifact_manifest(run_dir)
    if args.wiki_mirror_dir is not None:
        _mirror_svgs(run_dir, args.wiki_mirror_dir, artifact_manifest)
    _write_json(run_dir / "status.json", {"status": "complete"})
    print(json.dumps({"run_dir": str(run_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
