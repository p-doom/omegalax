"""Shared helpers for weight mapping and assignment across model families."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

import jax.numpy as jnp
from etils import epath


def stoi(token: str) -> int | str:
    return int(token) if token.isdigit() else token


def map_to_bonsai_key(mapping: dict[str, tuple[str, Enum]], torch_key: str):
    for pattern, (jax_key, transform) in mapping.items():
        match = re.match(pattern, torch_key)
        if match:
            return re.sub(pattern, jax_key, torch_key), transform
    return None, None


def assign_weights_from_eval_shape(
    keys: list[str | int],
    tensor: Any,
    state_dict: dict[str, Any],
    torch_key: str,
    transform_rule: tuple[tuple[int, ...] | None, tuple[int, ...] | None, bool] | None,
):
    value = jnp.asarray(tensor)
    if transform_rule is not None:
        permute_rule, reshape_rule, transpose_last = transform_rule
        if permute_rule is not None:
            value = value.transpose(permute_rule)
        if reshape_rule is not None:
            value = value.reshape(reshape_rule)
        if transpose_last:
            value = value.T

    node: Any = state_dict
    for k in keys[:-1]:
        node = node[k]
    leaf_key = keys[-1]
    target = node[leaf_key]

    if hasattr(target, "shape") and target.shape != value.shape:
        raise ValueError(f"Shape mismatch for '{torch_key}': expected {target.shape}, got {value.shape}")

    target_dtype = getattr(target, "dtype", None)
    if target_dtype is not None:
        value = value.astype(target_dtype)

    node[leaf_key] = value


def load_hf_config(path: str | epath.Path) -> dict[str, Any]:
    cfg_path = epath.Path(path) / "config.json"
    if not cfg_path.exists():
        raise ValueError(f"Expected HuggingFace config.json under {path}")
    with cfg_path.open() as f:
        return json.load(f)
