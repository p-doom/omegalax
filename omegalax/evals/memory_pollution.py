"""Build memory-task eval manifests disjoint from recorded training streams."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from omegalax.data import memory_tasks
from omegalax.data.memory_tasks import MemoryTask, MemoryTaskConfig, MemoryTaskSplit


_POLICY = "filler_free_tables_and_pairs_v1"
_ALLOWED_TRAIN_EVAL_DIFFERENCES = frozenset({"seed", "split", "segment_length", "num_segments"})
_LEGACY_V1_CONTENT_SHA256 = {
    MemoryTask.EXACT: "51089c1ecf697b43266ae81844a30b2fbb67b3de0f2eb04d7a5bb92ea0a6d018",
    MemoryTask.DOMINANT_RULE: "51089c1ecf697b43266ae81844a30b2fbb67b3de0f2eb04d7a5bb92ea0a6d018",
    MemoryTask.ASSOCIATIVE_BANK: "9263e91aa2e3088c37946266d0af87433376b795adbdba8c3a7640306cd942c3",
    MemoryTask.RULE_BANK: "9263e91aa2e3088c37946266d0af87433376b795adbdba8c3a7640306cd942c3",
}
_COMPATIBLE_V2_CONTENT_SHA256 = frozenset(
    {
        "32f307dc07996ab71d09f6a8aa660238447a817e9be31d27464a97d386478478",
        "fe8d88245437d92c8c096624f2aa112ef46b506fe41e1fc6aa17f0c611c4c03e",
        # Unique 230179f v2 content hash (currently only the clean A48 run): later
        # HF-init/provenance changes leave deterministic pair consumption unchanged.
        "971f564351b87e962092de318fcb260fa1c1552b6b1870ebc1c90834a6693610",
        # 1a926ed only omits the scheduled-save force flag added by 402e59f; Orbax
        # save-policy behavior does not affect generator semantics or pair consumption.
        "e8b1a5abf304cc8c9163b717254fe54568e242b8d12c9ad3db26b50f4c0e8ad2",
        # 402e59f only forces scheduled checkpoint saves through Orbax's save policy;
        # generator semantics and deterministic pair consumption are unchanged.
        "c382c3dfbd0e9f4113e12320cab0761d3ed4377a961c096e7f0e7ac85b5bdd9f",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryTrainingStream:
    checkpoint_root: Path
    checkpoint_step: int
    contract_sha256: str
    config: MemoryTaskConfig
    train_pairs: int


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _current_training_content_sha256() -> str:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        repo / "scripts/train_memory_tasks.py",
        repo / "omegalax/data/memory_tasks.py",
        repo / "omegalax/trainers/pretrain.py",
        repo / "omegalax/trainers/tail_pretrain.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repo)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest(*parts: bytes) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def _table_records(world: np.ndarray, config: MemoryTaskConfig) -> tuple[tuple[int, ...], ...]:
    early = world[:-1].reshape(-1)
    if config.task is MemoryTask.EXACT:
        tags = (memory_tasks._BIND, memory_tasks._DECOY_BIND)
        width = 3
    elif config.task is MemoryTask.DOMINANT_RULE:
        tags = (memory_tasks._DEMO,)
        width = 3
    elif config.task is MemoryTask.ASSOCIATIVE_BANK:
        tags = (memory_tasks._BIND,)
        width = 3
    elif config.task is MemoryTask.RULE_BANK:
        tags = (memory_tasks._DEMO,)
        width = 4
    else:
        raise ValueError(f"Unhandled MemoryTask: {config.task!r}")
    records = [
        tuple(int(value) for value in early[index : index + width])
        for index in range(early.size - width + 1)
        if int(early[index]) in tags
    ]
    if not records:
        raise ValueError(f"No task-table records found for task={config.task.value!r}")
    return tuple(sorted(records))


def table_digest(world: np.ndarray, config: MemoryTaskConfig) -> bytes:
    records = np.asarray(_table_records(world, config), dtype=np.int32)
    return _digest(config.task.value.encode(), records.tobytes())


def pair_digest(pair: np.ndarray, config: MemoryTaskConfig) -> bytes:
    filler_start = (
        memory_tasks._BANK_FILLER_START
        if config.task in (MemoryTask.ASSOCIATIVE_BANK, MemoryTask.RULE_BANK)
        else memory_tasks._FILLER_START
    )
    world_cores = []
    for world in pair:
        flat = world.reshape(-1)
        world_cores.append(_digest(flat[flat < filler_start].astype(np.int32).tobytes()))
    return _digest(config.task.value.encode(), *sorted(world_cores))


def training_pairs_at_checkpoint(contract: dict[str, Any], checkpoint_step: int) -> int:
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be > 0")
    source = contract["source"]
    if source.get("restore") == "huggingface_safetensors":
        if source.get("global_step_offset") != 0 or "checkpoint_step" in source:
            raise ValueError("HuggingFace memory initialization must start at global step 0")
        source_step = 0
    else:
        source_step = int(source.get("checkpoint_step", 0))
    updates = checkpoint_step - source_step
    planned_updates = int(contract["optimizer"]["num_steps"])
    if updates <= 0 or updates > planned_updates:
        raise ValueError(
            f"checkpoint step {checkpoint_step} implies {updates} updates for a run planned "
            f"for {planned_updates} updates from source step {source_step}"
        )
    return (
        updates * int(contract["pairs_per_batch"]) * int(contract["optimizer"]["grad_accum_steps"])
    )


def load_training_stream(checkpoint_root: Path, checkpoint_step: int) -> MemoryTrainingStream:
    checkpoint_root = checkpoint_root.expanduser().resolve()
    contract_path = checkpoint_root / "memory_task_contract.json"
    try:
        contract = json.loads(contract_path.read_text())
        config = MemoryTaskConfig(**contract["task"])
        topology = contract["runtime"]["topology"]
        training_content_sha256 = contract["revision"]["content_sha256"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Unreadable memory-task contract: {contract_path}") from exc
    if config.split is not MemoryTaskSplit.TRAIN:
        raise ValueError(f"Training contract must use split=train: {contract_path}")
    if topology != {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}:
        raise ValueError(f"Only recorded single-device streams are supported, got {topology}")
    current_content_sha256 = _current_training_content_sha256()
    compatible_legacy_hash = (
        config.generator_version == 1
        and training_content_sha256 == _LEGACY_V1_CONTENT_SHA256.get(config.task)
    )
    compatible_v2_hash = (
        config.generator_version == 2 and training_content_sha256 in _COMPATIBLE_V2_CONTENT_SHA256
    )
    if (
        training_content_sha256 != current_content_sha256
        and not compatible_legacy_hash
        and not compatible_v2_hash
    ):
        raise ValueError(
            f"Cannot reconstruct {checkpoint_root}: training content hash changed "
            f"from {training_content_sha256} to {current_content_sha256}"
        )
    return MemoryTrainingStream(
        checkpoint_root=checkpoint_root,
        checkpoint_step=int(checkpoint_step),
        contract_sha256=canonical_json_sha256(contract),
        config=config,
        train_pairs=training_pairs_at_checkpoint(contract, int(checkpoint_step)),
    )


def load_training_streams(
    checkpoint_root: Path, checkpoint_step: int
) -> tuple[MemoryTrainingStream, ...]:
    streams = []
    seen = set()
    current_root = checkpoint_root.expanduser().resolve()
    current_step = int(checkpoint_step)
    while True:
        identity = (current_root, current_step)
        if identity in seen:
            raise ValueError(f"Cyclic memory-task checkpoint ancestry at {identity}")
        seen.add(identity)
        streams.append(load_training_stream(current_root, current_step))

        contract = json.loads((current_root / "memory_task_contract.json").read_text())
        source = contract["source"]
        if "checkpoint_root" not in source:
            break
        if "memory_task_contract_sha256" not in source:
            raise ValueError(
                f"Cannot reconstruct {current_root}: source contract provenance is missing"
            )
        expected_source_contract_sha256 = source["memory_task_contract_sha256"]
        if expected_source_contract_sha256 is None:
            break
        source_root = Path(source["checkpoint_root"]).expanduser().resolve()
        source_contract_path = source_root / "memory_task_contract.json"
        try:
            source_contract = json.loads(source_contract_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Unreadable source memory-task contract: {source_contract_path}"
            ) from exc
        actual_source_contract_sha256 = canonical_json_sha256(source_contract)
        if actual_source_contract_sha256 != expected_source_contract_sha256:
            raise ValueError(
                f"Source memory-task contract hash mismatch at {source_contract_path}: "
                f"expected {expected_source_contract_sha256}, got {actual_source_contract_sha256}"
            )
        current_root = source_root
        current_step = int(source["checkpoint_step"])
    return tuple(streams)


def _validate_stream_config(stream: MemoryTrainingStream, eval_config: MemoryTaskConfig) -> None:
    trained = dataclasses.asdict(stream.config)
    requested = dataclasses.asdict(eval_config)
    mismatches = {
        name: (trained[name], value)
        for name, value in requested.items()
        if name not in _ALLOWED_TRAIN_EVAL_DIFFERENCES and trained[name] != value
    }
    if mismatches:
        raise ValueError(
            f"Training stream {stream.checkpoint_root} has incompatible task fields: {mismatches}"
        )


def _eval_group_size(task: MemoryTask) -> int:
    if task is MemoryTask.ASSOCIATIVE_BANK:
        return memory_tasks._BANK_NUM_SLOTS
    if task is MemoryTask.RULE_BANK:
        return memory_tasks._RULE_BANK_NUM_NAMESPACES
    return 1


def _manifest_config(manifest: dict[str, Any], path: Path | None = None) -> MemoryTaskConfig:
    try:
        fields = {
            field.name: manifest.get(field.name, field.default)
            for field in dataclasses.fields(MemoryTaskConfig)
        }
        config = MemoryTaskConfig(**fields)
    except (TypeError, ValueError) as exc:
        location = f" {path}" if path is not None else ""
        raise ValueError(f"Invalid memory-task eval manifest config{location}") from exc
    if config.split is not MemoryTaskSplit.EVAL:
        location = f" {path}" if path is not None else ""
        raise ValueError(f"Memory-task eval manifest{location} must use split=eval")
    return config


def _validate_exclusion_config(
    exclusion_config: MemoryTaskConfig,
    eval_config: MemoryTaskConfig,
    path: Path,
) -> None:
    excluded = dataclasses.asdict(exclusion_config)
    requested = dataclasses.asdict(eval_config)
    mismatches = {
        name: (excluded[name], value)
        for name, value in requested.items()
        if name not in _ALLOWED_TRAIN_EVAL_DIFFERENCES and excluded[name] != value
    }
    if mismatches:
        raise ValueError(f"Exclusion manifest {path} has incompatible task fields: {mismatches}")


def _manifest_digests(
    manifest: dict[str, Any], config: MemoryTaskConfig
) -> tuple[set[bytes], set[bytes]]:
    pair_digests: set[bytes] = set()
    table_digests: set[bytes] = set()
    for pair_index in manifest["pair_indices"]:
        pair, _ = memory_tasks._paired_tokens(pair_index, config)
        pair_digests.add(pair_digest(pair, config))
        table_digests.update(table_digest(pair[world], config) for world in (0, 1))
    return pair_digests, table_digests


def _load_exclusion_manifest(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = path.expanduser().resolve()
    try:
        content = path.read_bytes()
        manifest = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable exclusion manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Exclusion manifest must contain a JSON object: {path}")
    pollution_guard = manifest.get("pollution_guard")
    if isinstance(pollution_guard, dict) and pollution_guard.get("exclusion_manifests"):
        raise ValueError(f"Exclusion source must be a gate manifest: {path}")
    validate_pollution_free_manifest(manifest)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}, manifest


def build_pollution_free_manifest(
    eval_config: MemoryTaskConfig,
    *,
    num_pairs: int,
    training_streams: Sequence[MemoryTrainingStream],
    exclude_manifests: Sequence[Path] = (),
    final_holdout: bool = False,
) -> dict[str, Any]:
    if eval_config.split is not MemoryTaskSplit.EVAL:
        raise ValueError("Eval config must use split=eval")
    if final_holdout and not exclude_manifests:
        raise ValueError("A final holdout requires at least one exclusion manifest")
    if not training_streams:
        raise ValueError("At least one training stream is required")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    for stream in training_streams:
        _validate_stream_config(stream, eval_config)

    normalized_streams = sorted(
        training_streams,
        key=lambda stream: (str(stream.checkpoint_root), stream.checkpoint_step),
    )
    identities = [
        (str(stream.checkpoint_root), stream.checkpoint_step) for stream in normalized_streams
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Training streams must not contain duplicate checkpoint references")

    train_pair_digests: set[bytes] = set()
    train_table_digests: set[bytes] = set()
    scanned_streams: set[tuple[MemoryTaskConfig, int]] = set()
    for stream in normalized_streams:
        stream_key = (stream.config, stream.train_pairs)
        if stream_key in scanned_streams:
            continue
        scanned_streams.add(stream_key)
        for pair_index in range(stream.train_pairs):
            pair, _ = memory_tasks._paired_tokens(pair_index, stream.config)
            train_pair_digests.add(pair_digest(pair, stream.config))
            train_table_digests.update(table_digest(pair[world], stream.config) for world in (0, 1))

    exclusion_references = []
    exclusion_paths = sorted(path.expanduser().resolve() for path in exclude_manifests)
    if len(set(exclusion_paths)) != len(exclusion_paths):
        raise ValueError("Exclusion manifests must not contain duplicate paths")
    for path in exclusion_paths:
        reference, exclusion_manifest = _load_exclusion_manifest(path)
        exclusion_config = _manifest_config(exclusion_manifest, path)
        _validate_exclusion_config(exclusion_config, eval_config, path)
        excluded_pair_digests, excluded_table_digests = _manifest_digests(
            exclusion_manifest, exclusion_config
        )
        train_pair_digests.update(excluded_pair_digests)
        train_table_digests.update(excluded_table_digests)
        exclusion_references.append(reference)

    group_size = _eval_group_size(eval_config.task)
    if num_pairs % group_size:
        raise ValueError(f"num_pairs={num_pairs} must be divisible by eval group size {group_size}")
    target_groups = num_pairs // group_size
    candidate_group_limit = target_groups * 100
    selected: list[int] = []
    pair_collision_groups = 0
    table_collision_groups = 0
    considered_groups = 0
    for group_index in range(candidate_group_limit):
        considered_groups += 1
        group_indices = tuple(range(group_index * group_size, (group_index + 1) * group_size))
        pair_collision = False
        table_collision = False
        for pair_index in group_indices:
            pair, _ = memory_tasks._paired_tokens(pair_index, eval_config)
            pair_collision |= pair_digest(pair, eval_config) in train_pair_digests
            table_collision |= any(
                table_digest(pair[world], eval_config) in train_table_digests for world in (0, 1)
            )
        if pair_collision:
            pair_collision_groups += 1
        if table_collision:
            table_collision_groups += 1
        if pair_collision or table_collision:
            continue
        selected.extend(group_indices)
        if len(selected) == num_pairs:
            break
    if len(selected) != num_pairs:
        raise ValueError(
            f"Could not select {num_pairs} disjoint eval pairs after checking "
            f"{considered_groups * group_size} candidates"
        )

    pollution_guard = {
        "policy": _POLICY,
        "digest": "sha256",
        "ignored_content": "filler_tokens",
        "table_order": "ignored",
        "collision_action": "reject",
        "eval_group_size": group_size,
        "candidate_groups_considered": considered_groups,
        "pair_collision_groups": pair_collision_groups,
        "table_collision_groups": table_collision_groups,
        "training_streams": [
            {
                "checkpoint_root": str(stream.checkpoint_root),
                "checkpoint_step": stream.checkpoint_step,
                "contract_sha256": stream.contract_sha256,
                "train_pairs": stream.train_pairs,
            }
            for stream in normalized_streams
        ],
    }
    if exclusion_references:
        pollution_guard["exclusion_manifests"] = exclusion_references
    if final_holdout:
        pollution_guard["final_holdout"] = True

    manifest = {
        "schema_version": 2,
        **dataclasses.asdict(eval_config),
        "pair_indices": selected,
        "pollution_guard": pollution_guard,
    }
    return manifest | {"sha256": canonical_json_sha256(manifest)}


def validate_pollution_free_manifest(manifest: dict[str, Any]) -> None:
    try:
        expected = manifest["sha256"]
        policy = manifest["pollution_guard"]["policy"]
        pair_indices = manifest["pair_indices"]
        task = MemoryTask(manifest["task"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Eval manifest is missing pollution-guard metadata") from exc
    payload = {name: value for name, value in manifest.items() if name != "sha256"}
    actual = canonical_json_sha256(payload)
    if expected != actual:
        raise ValueError(f"Eval manifest hash mismatch: expected {expected}, got {actual}")
    if manifest.get("schema_version") != 2 or policy != _POLICY:
        raise ValueError("Unsupported memory-task pollution-guard manifest")
    if not isinstance(pair_indices, list) or any(
        type(pair_index) is not int or pair_index < 0 for pair_index in pair_indices
    ):
        raise ValueError("Eval manifest pair_indices must be non-negative JSON integers")
    if len(set(pair_indices)) != len(pair_indices):
        raise ValueError("Eval manifest pair_indices must be unique")
    group_size = _eval_group_size(task)
    if len(pair_indices) % group_size:
        raise ValueError(f"Eval manifest pair_indices must contain complete groups of {group_size}")
    for start in range(0, len(pair_indices), group_size):
        group = pair_indices[start : start + group_size]
        if group != list(range(group[0], group[0] + group_size)) or group[0] % group_size:
            raise ValueError(f"Invalid {task.value} eval group: {group}")

    eval_config = _manifest_config(manifest)
    exclusion_references = manifest["pollution_guard"].get("exclusion_manifests", [])
    if not isinstance(exclusion_references, list):
        raise ValueError("Eval manifest exclusion_manifests must be a JSON list")
    final_holdout = manifest["pollution_guard"].get("final_holdout", False)
    if not isinstance(final_holdout, bool):
        raise ValueError("Eval manifest final_holdout must be boolean")
    if final_holdout and not exclusion_references:
        raise ValueError("A final holdout requires at least one exclusion manifest")
    seen_paths = set()
    excluded_pair_digests: set[bytes] = set()
    excluded_table_digests: set[bytes] = set()
    for reference in exclusion_references:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("Invalid exclusion manifest reference")
        path_value = reference["path"]
        expected_file_sha256 = reference["sha256"]
        if not isinstance(path_value, str) or not isinstance(expected_file_sha256, str):
            raise ValueError("Invalid exclusion manifest reference")
        path = Path(path_value)
        if not path.is_absolute() or str(path.resolve()) != path_value:
            raise ValueError(f"Exclusion manifest path must be absolute and resolved: {path_value}")
        if path in seen_paths:
            raise ValueError("Exclusion manifests must not contain duplicate paths")
        seen_paths.add(path)
        actual_reference, exclusion_manifest = _load_exclusion_manifest(path)
        if actual_reference["sha256"] != expected_file_sha256:
            raise ValueError(
                f"Exclusion manifest file hash mismatch at {path}: "
                f"expected {expected_file_sha256}, got {actual_reference['sha256']}"
            )
        exclusion_config = _manifest_config(exclusion_manifest, path)
        _validate_exclusion_config(exclusion_config, eval_config, path)
        pair_digests, table_digests = _manifest_digests(exclusion_manifest, exclusion_config)
        excluded_pair_digests.update(pair_digests)
        excluded_table_digests.update(table_digests)

    manifest_pair_digests, manifest_table_digests = _manifest_digests(manifest, eval_config)
    if manifest_pair_digests & excluded_pair_digests:
        raise ValueError("Eval manifest overlaps an exclusion manifest pair digest")
    if manifest_table_digests & excluded_table_digests:
        raise ValueError("Eval manifest overlaps an exclusion manifest table digest")
