"""Canonical content identities for compiled-data producers and consumers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers.utils.hub import cached_file

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


def _validate_sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value.lower())
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
    if value["revision"] is not None and not isinstance(value["revision"], str):
        raise ValueError(f"{field}.revision must be a string or null")
    _validate_sha256(value["behavior_sha256"], f"{field}.behavior_sha256")
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
        _validate_sha256(item["sha256"], f"{field}.files[{index}].sha256")
        paths.append(item["path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{field}.files must be sorted and unique by path")


def validate_measurement_contract(contract: Any) -> None:
    if not isinstance(contract, dict):
        raise TypeError("measurement contract must be an object")
    required = {"producer_sha", "tokenizer", "processor", "renderer", "preprocessor"}
    if set(contract) != required:
        raise ValueError(f"measurement contract fields must be exactly {sorted(required)}")
    producer_sha = contract["producer_sha"]
    if (
        not isinstance(producer_sha, str)
        or len(producer_sha) != 40
        or any(char not in "0123456789abcdef" for char in producer_sha.lower())
    ):
        raise ValueError("measurement contract producer_sha must be an exact Git SHA")
    _validate_asset_contract(contract["tokenizer"], "measurement contract tokenizer")
    if contract["processor"] is not None:
        _validate_asset_contract(contract["processor"], "measurement contract processor")
    renderer = contract["renderer"]
    if not isinstance(renderer, dict) or set(renderer) != {"class", "config_sha256"}:
        raise ValueError("measurement contract renderer has an invalid schema")
    if not isinstance(renderer["class"], str) or not renderer["class"]:
        raise ValueError("measurement contract renderer.class must be non-empty")
    _validate_sha256(renderer["config_sha256"], "measurement contract renderer.config_sha256")
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
        _validate_sha256(preprocessor["sha256"], "measurement contract preprocessor.sha256")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_identity(path: str | Path) -> dict[str, int | str]:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _resolved_asset_inventory(identifier: str, names: tuple[str, ...]) -> dict:
    files: list[dict[str, int | str]] = []
    revisions: set[str] = set()
    for name in names:
        resolved = cached_file(
            identifier,
            name,
            local_files_only=True,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if resolved is None:
            continue
        path = Path(resolved).resolve()
        parts = path.parts
        if "snapshots" in parts:
            snapshot_index = parts.index("snapshots")
            revisions.add(parts[snapshot_index + 1])
            relative_path = Path(*parts[snapshot_index + 2 :]).as_posix()
        else:
            relative_path = path.name
        files.append({"path": relative_path, **file_identity(path)})
    if not files:
        raise ValueError(f"no pinned local assets found for {identifier!r}")
    if len(revisions) > 1:
        raise ValueError(f"assets for {identifier!r} resolve to multiple revisions: {revisions}")
    return {
        "source": identifier,
        "revision": next(iter(revisions), None),
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
    renderer_config,
    preprocessor_config_path: str | Path | None,
) -> dict:
    if len(producer_sha) != 40 or any(c not in "0123456789abcdef" for c in producer_sha.lower()):
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
    renderer = {
        "class": f"{type(renderer_config).__module__}.{type(renderer_config).__qualname__}",
        "config_sha256": canonical_sha256(_jsonable(renderer_config)),
    }
    contract = {
        "producer_sha": producer_sha.lower(),
        "tokenizer": tokenizer_assets,
        "processor": processor,
        "renderer": renderer,
        "preprocessor": preprocessor,
    }
    validate_measurement_contract(contract)
    return contract
