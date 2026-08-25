"""Canonical content identities for compiled-data producers and consumers."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

TOKENIZER_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)
PROCESSOR_ASSET_NAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
)
COMPILED_DATASET_SCHEMA_VERSION = 3
COMPILED_ARTIFACT_CONTRACT_VERSION = 2
EXTERNAL_ARTIFACT_INVENTORY_VERSION = 1


def validate_sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be an exact SHA-256 digest")


def _validate_asset_contract(value: Any, field: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    required = {"source", "revision", "files", "behavior_sha256"}
    if set(value) != required:
        raise ValueError(f"{field} fields must be exactly {sorted(required)}")
    if not isinstance(value["source"], str) or not value["source"]:
        raise ValueError(f"{field}.source must be non-empty")
    if value["revision"] is not None:
        revision = value["revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or revision != revision.lower()
            or any(char not in "0123456789abcdef" for char in revision)
        ):
            raise ValueError(f"{field}.revision must be an exact lowercase commit digest or null")
    validate_sha256(value["behavior_sha256"], f"{field}.behavior_sha256")
    if not isinstance(value["files"], list) or not value["files"]:
        raise ValueError(f"{field}.files must be a non-empty list")
    paths: list[str] = []
    for index, item in enumerate(value["files"]):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"{field}.files[{index}] has an invalid schema")
        if not isinstance(item["path"], str) or not item["path"]:
            raise ValueError(f"{field}.files[{index}].path must be non-empty")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 0:
            raise ValueError(f"{field}.files[{index}].size_bytes must be non-negative")
        validate_sha256(item["sha256"], f"{field}.files[{index}].sha256")
        paths.append(item["path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{field}.files must be sorted and unique by path")


def validate_measurement_contract(contract: Any) -> None:
    if not isinstance(contract, dict):
        raise TypeError("measurement contract must be an object")
    required = {"producer_sha", "tokenizer", "processor", "preprocessor"}
    if set(contract) != required:
        raise ValueError(f"measurement contract fields must be exactly {sorted(required)}")
    producer_sha = contract["producer_sha"]
    if (
        not isinstance(producer_sha, str)
        or len(producer_sha) != 40
        or producer_sha != producer_sha.lower()
        or any(char not in "0123456789abcdef" for char in producer_sha)
    ):
        raise ValueError("measurement contract producer_sha must be an exact Git SHA")
    _validate_asset_contract(contract["tokenizer"], "measurement contract tokenizer")
    if contract["processor"] is not None:
        _validate_asset_contract(contract["processor"], "measurement contract processor")
    preprocessor = contract["preprocessor"]
    if preprocessor is not None:
        if not isinstance(preprocessor, dict) or set(preprocessor) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("measurement contract preprocessor has an invalid schema")
        if not isinstance(preprocessor["path"], str) or not preprocessor["path"]:
            raise ValueError("measurement contract preprocessor.path must be non-empty")
        if not isinstance(preprocessor["size_bytes"], int) or preprocessor["size_bytes"] < 0:
            raise ValueError("measurement contract preprocessor.size_bytes must be non-negative")
        validate_sha256(preprocessor["sha256"], "measurement contract preprocessor.sha256")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _open_regular_file_nofollow(path: str | Path) -> tuple[int, Path]:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parent.parts[1:]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(f"artifact path contains a symlink: {absolute}") from exc
        raise
    finally:
        os.close(directory_fd)
    return fd, absolute


def _consume_stable_regular_file(
    path: str | Path, *, collect: bool
) -> tuple[bytes | None, os.stat_result, str]:
    fd, absolute = _open_regular_file_nofollow(path)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact path is not a regular file: {absolute}")
        chunks: list[bytes] | None = [] if collect else None
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"artifact changed while hashing: {absolute}")
        payload = b"".join(chunks) if chunks is not None else None
        return payload, after, digest.hexdigest()
    finally:
        os.close(fd)


def read_stable_regular_file(path: str | Path, *, require_sealed: bool = False) -> bytes:
    payload, file_stat, _digest = _consume_stable_regular_file(path, collect=True)
    if payload is None:
        raise RuntimeError("stable file reader did not collect the requested payload")
    if require_sealed and file_stat.st_mode & 0o222:
        raise ValueError(f"artifact file is writable and therefore not sealed: {path}")
    return payload


def file_identity(path: str | Path) -> dict[str, int | str]:
    _payload, file_stat, digest = _consume_stable_regular_file(path, collect=False)
    return {"size_bytes": file_stat.st_size, "sha256": digest}


def sealed_file_identity(path: str | Path) -> dict[str, int | str]:
    _payload, file_stat, digest = _consume_stable_regular_file(path, collect=False)
    if file_stat.st_mode & 0o222:
        raise ValueError(f"artifact file is writable and therefore not sealed: {path}")
    return {"size_bytes": file_stat.st_size, "sha256": digest}


def _validate_compiled_shards(shards: Any) -> None:
    if not isinstance(shards, list) or not shards:
        raise ValueError("compiled artifact shards must be a non-empty list")
    paths: list[str] = []
    required = {"path", "size_bytes", "sha256", "num_records", "max_record_index"}
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != required:
            raise ValueError(f"compiled artifact shards[{index}] has an invalid schema")
        path = shard["path"]
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"compiled artifact shards[{index}].path must be relative")
        if not isinstance(shard["num_records"], int) or shard["num_records"] <= 0:
            raise ValueError(f"compiled artifact shards[{index}].num_records must be positive")
        if shard["max_record_index"] != shard["num_records"] - 1:
            raise ValueError(f"compiled artifact shards[{index}] has an invalid record bound")
        if not isinstance(shard["size_bytes"], int) or shard["size_bytes"] <= 0:
            raise ValueError(f"compiled artifact shards[{index}].size_bytes must be positive")
        validate_sha256(shard["sha256"], f"compiled artifact shards[{index}].sha256")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("compiled artifact shards must be sorted and unique by path")


def _verify_regular_file(path: Path, expected: dict, field: str) -> None:
    actual = sealed_file_identity(path)
    expected_identity = {
        "size_bytes": expected["size_bytes"],
        "sha256": expected["sha256"],
    }
    if actual != expected_identity:
        raise ValueError(f"{field} content identity does not match: {path}")


def validate_external_artifact_inventory(
    inventory: Any,
    *,
    verify_files: bool,
) -> dict[str, dict]:
    if not isinstance(inventory, dict):
        raise TypeError("external artifact inventory must be an object")
    required = {
        "schema_version",
        "root",
        "retention_pin_sha256",
        "shards",
        "inventory_sha256",
    }
    if set(inventory) != required:
        raise ValueError(f"external artifact inventory fields must be exactly {sorted(required)}")
    if inventory["schema_version"] != EXTERNAL_ARTIFACT_INVENTORY_VERSION:
        raise ValueError("unsupported external artifact inventory schema")
    root = Path(inventory["root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("external artifact inventory root must be an existing absolute directory")
    root = Path(os.path.abspath(root))
    if root.stat().st_mode & 0o222:
        raise ValueError("external artifact inventory root is writable and therefore not sealed")
    validate_sha256(inventory["retention_pin_sha256"], "retention_pin_sha256")
    shards = inventory["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("external artifact inventory shards must be non-empty")
    required_shard_fields = {"path", "size_bytes", "sha256", "max_record_index"}
    paths: list[str] = []
    by_absolute_path: dict[str, dict] = {}
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != required_shard_fields:
            raise ValueError(f"external artifact inventory shards[{index}] has an invalid schema")
        relative = Path(shard["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == ".":
            raise ValueError(f"external artifact inventory shards[{index}].path must be relative")
        if not isinstance(shard["size_bytes"], int) or shard["size_bytes"] <= 0:
            raise ValueError(
                f"external artifact inventory shards[{index}].size_bytes must be positive"
            )
        if not isinstance(shard["max_record_index"], int) or shard["max_record_index"] < 0:
            raise ValueError(
                f"external artifact inventory shards[{index}].max_record_index must be non-negative"
            )
        validate_sha256(shard["sha256"], f"external artifact inventory shards[{index}].sha256")
        paths.append(relative.as_posix())
        absolute = root / relative
        if verify_files:
            _verify_regular_file(absolute, shard, "external artifact shard")
        by_absolute_path[os.path.abspath(absolute)] = shard
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("external artifact inventory shards must be sorted and unique by path")
    identity = {
        "schema_version": inventory["schema_version"],
        "retention_pin_sha256": inventory["retention_pin_sha256"],
        "shards": shards,
    }
    if inventory["inventory_sha256"] != canonical_sha256(identity):
        raise ValueError("external artifact inventory digest does not match its contents")
    return by_absolute_path


def verify_compiled_dataset(path: str | Path) -> dict:
    root = Path(os.path.abspath(Path(path).expanduser()))
    metadata_path = root / "metadata.json"
    metadata = json.loads(read_stable_regular_file(metadata_path, require_sealed=True))
    if metadata.get("version") != COMPILED_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported compiled dataset schema: expected {COMPILED_DATASET_SCHEMA_VERSION}"
        )
    contract = metadata.get("artifact_contract")
    if not isinstance(contract, dict):
        raise TypeError("compiled dataset is missing artifact_contract")
    required = {
        "schema_version",
        "producer_sha",
        "source_chat",
        "measurement_contract",
        "measurement_contract_sha256",
        "shards",
        "shard_inventory_sha256",
        "record_profile",
        "lineage",
        "external_dependencies",
    }
    if set(contract) != required:
        raise ValueError(f"compiled artifact contract fields must be exactly {sorted(required)}")
    if contract["schema_version"] != COMPILED_ARTIFACT_CONTRACT_VERSION:
        raise ValueError("unsupported compiled artifact contract schema")
    validate_measurement_contract(contract["measurement_contract"])
    if contract["producer_sha"] != contract["measurement_contract"]["producer_sha"]:
        raise ValueError("compiled artifact producer SHA does not match measurement contract")
    if contract["measurement_contract_sha256"] != canonical_sha256(
        contract["measurement_contract"]
    ):
        raise ValueError("compiled artifact measurement contract digest does not match")
    source_chat = contract["source_chat"]
    if not isinstance(source_chat, dict) or set(source_chat) != {"size_bytes", "sha256"}:
        raise ValueError("compiled artifact source_chat has an invalid schema")
    validate_sha256(source_chat["sha256"], "compiled artifact source_chat.sha256")
    _validate_compiled_shards(contract["shards"])
    if contract["shard_inventory_sha256"] != canonical_sha256(contract["shards"]):
        raise ValueError("compiled artifact shard inventory digest does not match")
    if sum(shard["num_records"] for shard in contract["shards"]) != metadata.get("num_records"):
        raise ValueError("compiled artifact shard record counts do not match metadata")
    for shard in contract["shards"]:
        _verify_regular_file(root / shard["path"], shard, "compiled shard")
    profile = contract["record_profile"]
    profile_fields = {
        "num_records",
        "max_measured_length",
        "max_vision_patches",
        "max_images",
    }
    if not isinstance(profile, dict) or set(profile) != profile_fields:
        raise ValueError("compiled artifact record_profile has an invalid schema")
    if profile["num_records"] != metadata["num_records"]:
        raise ValueError("compiled artifact record_profile count does not match metadata")
    for field in profile_fields - {"num_records"}:
        if not isinstance(profile[field], int) or profile[field] < 0:
            raise ValueError(f"compiled artifact record_profile.{field} must be non-negative")
    lineage = contract["lineage"]
    lineage_fields = {"split", "split_key", "val_fraction", "ids", "ids_sha256"}
    if not isinstance(lineage, dict) or set(lineage) != lineage_fields:
        raise ValueError("compiled artifact lineage has an invalid schema")
    if lineage["split"] not in {"train", "val", "unpartitioned"}:
        raise ValueError("compiled artifact lineage split is invalid")
    if not isinstance(lineage["split_key"], str) or not lineage["split_key"]:
        raise ValueError("compiled artifact lineage split_key must be non-empty")
    if (
        not isinstance(lineage["val_fraction"], (int, float))
        or isinstance(lineage["val_fraction"], bool)
        or not 0.0 <= lineage["val_fraction"] <= 1.0
    ):
        raise ValueError("compiled artifact lineage val_fraction must be in [0, 1]")
    ids = lineage["ids"]
    if (
        not isinstance(ids, list)
        or ids != sorted(set(ids))
        or not all(isinstance(item, str) and item for item in ids)
    ):
        raise ValueError("compiled artifact lineage ids must be sorted, unique, and non-empty")
    if lineage["ids_sha256"] != canonical_sha256(ids):
        raise ValueError("compiled artifact lineage digest does not match")
    external = contract["external_dependencies"]
    if external is not None:
        validate_external_artifact_inventory(external, verify_files=True)
    return metadata


def validate_train_val_lineage(train_metadata: list[dict], val_metadata: dict) -> None:
    train_lineages = [metadata["artifact_contract"]["lineage"] for metadata in train_metadata]
    val_lineage = val_metadata["artifact_contract"]["lineage"]
    if val_lineage["split"] != "val":
        raise ValueError("validation artifact must declare lineage split='val'")
    for lineage in train_lineages:
        if lineage["split"] != "train":
            raise ValueError("training artifact paired with validation must declare split='train'")
        if lineage["split_key"] != val_lineage["split_key"]:
            raise ValueError("train/validation lineage split keys do not match")
        if lineage["val_fraction"] != val_lineage["val_fraction"]:
            raise ValueError("train/validation lineage fractions do not match")
    train_ids = {lineage_id for lineage in train_lineages for lineage_id in lineage["ids"]}
    val_ids = set(val_lineage["ids"])
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"train/validation lineage overlap: {overlap[:5]}")


def validate_training_dataset_contract(
    train_paths: list[str | Path],
    *,
    val_path: str | Path | None,
    measurement_contract: dict,
    max_length: int,
    max_vision_patches_per_sample: int | None,
    max_vision_images_per_sample: int | None,
) -> tuple[list[dict], dict | None]:
    validate_measurement_contract(measurement_contract)
    if not train_paths:
        raise ValueError("at least one training dataset is required")
    train_metadata = [verify_compiled_dataset(path) for path in train_paths]
    val_metadata = verify_compiled_dataset(val_path) if val_path is not None else None
    all_metadata = train_metadata + ([val_metadata] if val_metadata is not None else [])
    for path, metadata in zip(
        [*train_paths, *([val_path] if val_path is not None else [])], all_metadata
    ):
        contract = metadata["artifact_contract"]
        if contract["measurement_contract"] != measurement_contract:
            raise ValueError(f"dataset measurement contract does not match runtime: {path}")
        if metadata.get("max_length") != max_length:
            raise ValueError(
                f"dataset max_length={metadata.get('max_length')} does not match runtime "
                f"max_length={max_length}: {path}"
            )
        profile = contract["record_profile"]
        if (
            max_vision_patches_per_sample is not None
            and profile["max_vision_patches"] > max_vision_patches_per_sample
        ):
            raise ValueError(
                f"dataset observed max_vision_patches={profile['max_vision_patches']} exceeds "
                f"runtime max_vision_patches_per_sample={max_vision_patches_per_sample}: {path}"
            )
        if (
            max_vision_images_per_sample is not None
            and profile["max_images"] > max_vision_images_per_sample
        ):
            raise ValueError(
                f"dataset observed max_images={profile['max_images']} exceeds runtime "
                f"max_vision_images_per_sample={max_vision_images_per_sample}: {path}"
            )
    if val_metadata is not None:
        validate_train_val_lineage(train_metadata, val_metadata)
    return train_metadata, val_metadata


def _resolved_asset_inventory(identifier: str, names: tuple[str, ...]) -> dict:
    root = Path(identifier).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("processor/tokenizer source must be an explicit local snapshot directory")
    root = Path(os.path.abspath(root))
    files: list[dict[str, int | str]] = []
    for name in names:
        path = root / name
        if not path.exists():
            continue
        files.append({"path": name, **sealed_file_identity(path)})
    if not files:
        raise ValueError(f"no pinned local assets found under {root}")
    return {
        "source": str(root),
        "revision": None,
        "files": sorted(files, key=lambda item: str(item["path"])),
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def make_measurement_contract(
    *,
    producer_sha: str,
    tokenizer,
    tokenizer_source: str,
    image_processor,
    processor_source: str | None,
    preprocessor_config_path: str | Path | None,
) -> dict:
    if (
        len(producer_sha) != 40
        or producer_sha != producer_sha.lower()
        or any(c not in "0123456789abcdef" for c in producer_sha)
    ):
        raise ValueError("producer_sha must be an exact 40-character Git SHA")
    tokenizer_assets = _resolved_asset_inventory(tokenizer_source, TOKENIZER_ASSET_NAMES)
    tokenizer_behavior = {
        "backend": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": _jsonable(tokenizer.special_tokens_map),
    }
    tokenizer_assets["behavior_sha256"] = canonical_sha256(tokenizer_behavior)

    if image_processor is None:
        processor = None
    else:
        if not processor_source:
            raise ValueError("processor_source is required when an image processor is configured")
        processor = _resolved_asset_inventory(processor_source, PROCESSOR_ASSET_NAMES)
        processor["behavior_sha256"] = canonical_sha256(_jsonable(image_processor.to_dict()))

    preprocessor = (
        None
        if preprocessor_config_path is None
        else {
            "path": Path(preprocessor_config_path).name,
            **file_identity(preprocessor_config_path),
        }
    )
    contract = {
        "producer_sha": producer_sha.lower(),
        "tokenizer": tokenizer_assets,
        "processor": processor,
        "preprocessor": preprocessor,
    }
    validate_measurement_contract(contract)
    return contract
