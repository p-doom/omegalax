"""Immutable training metadata used to resolve checkpoint evaluation settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from omegalax.data.pretrain_data_set import COMPILED_METADATA_FILENAME
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_CURRICULUM_INDEX_FORMAT,
    STATEPASSING_FIXED_C_INDEX_FORMAT,
    STATEPASSING_WINDOW_INDEX_FORMAT,
)


TRAINING_CONTRACT_FILENAME = "training_contract.json"
TRAINING_CONTRACT_SCHEMA_VERSION = 1
EVAL_CONFIG_FIELDS = (
    "c_train",
    "pass_gdn_state",
    "gdn_layer_limit",
    "pass_conv_state",
    "pass_rope_positions",
    "pad_id",
    "eos_id",
)


@dataclass(frozen=True)
class ManualEvalConfig:
    c_train: int
    pass_gdn_state: bool
    gdn_layer_limit: int | None
    pass_conv_state: bool
    pass_rope_positions: bool
    pad_id: int
    eos_id: int


@dataclass(frozen=True)
class ResolvedEvalConfig:
    c_train: int
    pass_gdn_state: bool
    gdn_layer_limit: int | None
    pass_conv_state: bool
    pass_rope_positions: bool
    pad_id: int
    eos_id: int
    resolution_source: str
    training_contract_hash: str | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def training_contract_hash(contract: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(contract)).hexdigest()}"


def _metadata_hash(metadata: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(metadata)).hexdigest()}"


def _curriculum_horizons(metadata: Mapping[str, Any]) -> list[dict[str, int | None]]:
    train_order = [int(value) for value in metadata["train_order"]]
    train_split = dict(dict(metadata["splits"])["train"])
    phases = dict(train_split["phases"])
    horizons = []
    last_step = 0
    cumulative_max = 0
    for num_segments in train_order:
        phase_steps = int(dict(phases[str(num_segments)])["phase_steps"])
        if phase_steps < 0:
            raise ValueError(f"Curriculum phase_steps must be non-negative, got {phase_steps}")
        if phase_steps == 0:
            continue
        cumulative_max = max(cumulative_max, num_segments)
        horizons.append(
            {
                "start_step": last_step + 1,
                "end_step": last_step + phase_steps,
                "c_train": cumulative_max,
            }
        )
        last_step += phase_steps
    if not horizons:
        raise ValueError("Curriculum metadata has no training updates")
    return horizons


def build_training_contract(
    train_index_path: str | Path,
    *,
    pass_gdn_state: bool,
    gdn_layer_limit: int | None,
    pass_conv_state: bool,
    pass_rope_positions: bool,
    pad_id: int,
    eos_id: int,
) -> dict[str, Any]:
    index_path = Path(train_index_path).expanduser().resolve()
    metadata_path = index_path / COMPILED_METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(f"Training index has no metadata file: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    index_format = metadata.get("format")
    if index_format == STATEPASSING_CURRICULUM_INDEX_FORMAT:
        horizons = _curriculum_horizons(metadata)
    elif index_format in (
        STATEPASSING_FIXED_C_INDEX_FORMAT,
        STATEPASSING_WINDOW_INDEX_FORMAT,
    ):
        num_segments = int(metadata["num_segments"])
        if num_segments <= 0:
            raise ValueError(f"Training index num_segments must be positive, got {num_segments}")
        horizons = [{"start_step": 1, "end_step": None, "c_train": num_segments}]
    else:
        raise ValueError(
            "Cannot create a Statepassing training contract from unsupported index "
            f"format={index_format!r}"
        )

    return {
        "schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
        "training_index": {
            "path": str(index_path),
            "metadata_hash": _metadata_hash(metadata),
        },
        "eval_statepassing_config": {
            "pass_gdn_state": bool(pass_gdn_state),
            "gdn_layer_limit": gdn_layer_limit,
            "pass_conv_state": bool(pass_conv_state),
            "pass_rope_positions": bool(pass_rope_positions),
            "pad_id": int(pad_id),
            "eos_id": int(eos_id),
        },
        "horizon_by_step": horizons,
    }


def _checkpoint_directories(checkpoint_root: Path) -> tuple[Path, ...]:
    if not checkpoint_root.is_dir():
        return ()
    return tuple(
        path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()
    )


def _write_contract(path: Path, contract: Mapping[str, Any]) -> None:
    payload = json.dumps(contract, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, raw_temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temp_path = Path(raw_temp_path)
    try:
        temp_path.write_bytes(payload)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def ensure_training_contract(
    checkpoint_root: str | Path,
    contract: Mapping[str, Any],
) -> str:
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    contract_path = checkpoint_root / TRAINING_CONTRACT_FILENAME
    if contract_path.is_file():
        stored = json.loads(contract_path.read_text())
        if _canonical_json(stored) != _canonical_json(contract):
            raise ValueError(f"Training contract conflicts with the requested run: {contract_path}")
        return training_contract_hash(stored)
    if _checkpoint_directories(checkpoint_root):
        raise ValueError(
            "Checkpoint directory contains checkpoints but no training_contract.json; "
            "a safe resume is not possible."
        )
    _write_contract(contract_path, contract)
    return training_contract_hash(contract)


def load_training_contract(
    checkpoint_root: str | Path,
) -> tuple[dict[str, Any], str] | None:
    contract_path = Path(checkpoint_root).expanduser().resolve() / TRAINING_CONTRACT_FILENAME
    if not contract_path.is_file():
        return None
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Training contract is not readable: {contract_path}") from error
    if contract.get("schema_version") != TRAINING_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported training contract schema_version={contract.get('schema_version')!r}"
        )
    return contract, training_contract_hash(contract)


def _contract_eval_config(contract: Mapping[str, Any], checkpoint_step: int) -> ManualEvalConfig:
    c_train = None
    for raw_horizon in contract.get("horizon_by_step", ()):
        horizon = dict(raw_horizon)
        start_step = int(horizon["start_step"])
        raw_end_step = horizon.get("end_step")
        end_step = None if raw_end_step is None else int(raw_end_step)
        if checkpoint_step >= start_step and (end_step is None or checkpoint_step <= end_step):
            c_train = int(horizon["c_train"])
            break
    if c_train is None:
        raise ValueError(
            f"Training contract has no C_train horizon for checkpoint step {checkpoint_step}"
        )
    state_config = dict(contract.get("eval_statepassing_config", {}))
    try:
        resolved = ManualEvalConfig(c_train=c_train, **state_config)
    except TypeError as error:
        raise ValueError("Training contract has an incomplete eval_statepassing_config") from error
    _validate_manual_config(resolved)
    return resolved


def _validate_manual_config(config: ManualEvalConfig) -> None:
    if config.c_train <= 0:
        raise ValueError(f"c_train must be > 0, got {config.c_train}")
    if config.gdn_layer_limit is not None and config.gdn_layer_limit < 0:
        raise ValueError("gdn_layer_limit must be non-negative or None")
    for name in ("pass_gdn_state", "pass_conv_state", "pass_rope_positions"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be a boolean")


def resolve_eval_config(
    checkpoint_root: str | Path,
    checkpoint_step: int,
    manual_config: ManualEvalConfig | None,
) -> ResolvedEvalConfig:
    loaded = load_training_contract(checkpoint_root)
    if loaded is None:
        if manual_config is None:
            raise ValueError(
                "Checkpoint has no training_contract.json; all seven legacy evaluation "
                "flags must be provided."
            )
        _validate_manual_config(manual_config)
        return ResolvedEvalConfig(
            **asdict(manual_config),
            resolution_source="manual_flags",
            training_contract_hash=None,
        )

    contract, contract_hash = loaded
    contract_config = _contract_eval_config(contract, int(checkpoint_step))
    if manual_config is not None:
        _validate_manual_config(manual_config)
        conflicts = [
            name
            for name in EVAL_CONFIG_FIELDS
            if getattr(manual_config, name) != getattr(contract_config, name)
        ]
        if conflicts:
            raise ValueError(
                "Manual evaluation flags conflict with training_contract.json for: "
                + ", ".join(conflicts)
            )
    return ResolvedEvalConfig(
        **asdict(contract_config),
        resolution_source="training_contract",
        training_contract_hash=contract_hash,
    )
