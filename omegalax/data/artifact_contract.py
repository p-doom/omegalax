"""Content identities shared by compiled-data producers and consumers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(payload.encode())


def file_identity(path: str | Path) -> dict[str, int | str]:
    path = Path(path).expanduser()
    return {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


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


def validate_measurement_contract(contract: Any) -> None:
    required = {
        "tokenizer_sha256",
        "processor_sha256",
        "preprocessor_sha256",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise TypeError(f"measurement contract fields must be exactly {sorted(required)}")
    for name in required:
        value = contract[name]
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"measurement contract {name} must be a SHA-256 digest or null")
    if contract["tokenizer_sha256"] is None:
        raise ValueError("measurement contract tokenizer_sha256 is required")


def make_measurement_contract(
    *,
    tokenizer,
    image_processor,
    preprocessor_config_path: str | Path | None,
) -> dict[str, Any]:
    tokenizer_behavior = {
        "backend": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    contract = {
        "tokenizer_sha256": canonical_sha256(_jsonable(tokenizer_behavior)),
        "processor_sha256": (
            None
            if image_processor is None
            else canonical_sha256(_jsonable(image_processor.to_dict()))
        ),
        "preprocessor_sha256": (
            None if preprocessor_config_path is None else file_sha256(preprocessor_config_path)
        ),
    }
    validate_measurement_contract(contract)
    return contract
