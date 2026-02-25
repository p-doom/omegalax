"""Shared helpers for weight mapping and assignment across model families."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from etils import epath

TransformRule = tuple[tuple[int, ...] | None, tuple[int, ...] | None, bool] | None


class Transform(Enum):
    """Canonical transform for HF -> JAX weight mapping."""
    LINEAR = ((1, 0), None, False)
    EMBED = None
    SCALE = None
    BIAS = None
    CONV3D = "conv3d"  # sentinel; handled specially per model


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
    transform_rule: TransformRule,
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


def assign_to_state_dict(
    state_dict: dict[str, Any],
    dotted_key: str,
    value: Any,
    label: str,
) -> None:
    """Navigate state_dict by dotted key and set the value. Raises on shape mismatch or bad path."""
    keys = [stoi(k) for k in dotted_key.split(".")]
    node: Any = state_dict
    for k in keys[:-1]:
        node = node[k]
    leaf_key = keys[-1]
    target = node[leaf_key]
    if hasattr(target, "shape") and target.shape != value.shape:
        raise ValueError(
            f"Shape mismatch for '{label}': expected {target.shape}, got {value.shape}"
        )
    target_dtype = getattr(target, "dtype", None)
    if target_dtype is not None:
        value = value.astype(target_dtype)
    node[leaf_key] = value


# MoE expert loading helpers
def init_expert_buffers(
    num_layers: int,
    num_experts: int,
    emb_dim: int,
    moe_dim: int,
    is_moe_layer: Callable[[int], bool],
) -> tuple[dict[tuple[int, str], np.ndarray], dict[tuple[int, str], int]]:
    """Pre-allocate expert weight buffers: gate/up are (E, D, F), down is (E, F, D)."""
    expert_arrays: dict[tuple[int, str], np.ndarray] = {}
    expert_fill: dict[tuple[int, str], int] = {}
    for layer_idx in range(num_layers):
        if is_moe_layer(layer_idx):
            expert_arrays[(layer_idx, "gate_proj")] = np.empty((num_experts, emb_dim, moe_dim), dtype=np.float32)  # EDF
            expert_arrays[(layer_idx, "up_proj")] = np.empty((num_experts, emb_dim, moe_dim), dtype=np.float32)  # EDF
            expert_arrays[(layer_idx, "down_proj")] = np.empty((num_experts, moe_dim, emb_dim), dtype=np.float32)  # EFD
            for proj in ("gate_proj", "up_proj", "down_proj"):
                expert_fill[(layer_idx, proj)] = 0
    return expert_arrays, expert_fill


def handle_moe_key(
    torch_key: str,
    get_tensor: Callable[[str], Any],
    expert_arrays: dict[tuple[int, str], np.ndarray],
    expert_fill: dict[tuple[int, str], int],
    router_buf: dict[int, Any],
    unmatched: list[str],
    *,
    num_experts: int,
    hf_prefix: str = "model.layers",
) -> bool:
    """Try to match expert/router HF keys and fill buffers. Returns True if handled."""
    esc = re.escape(hf_prefix)
    gate_up_m = re.match(rf"{esc}\.(\d+)\.mlp\.experts\.gate_up_proj(?:\.weight)?", torch_key)
    if gate_up_m:
        layer_idx = int(gate_up_m.group(1))
        if (layer_idx, "gate_proj") in expert_arrays:
            fused_E2FD = np.asarray(get_tensor(torch_key))
            gate_EFD, up_EFD = np.split(fused_E2FD, 2, axis=1)
            expert_arrays[(layer_idx, "gate_proj")] = np.swapaxes(gate_EFD.astype(np.float32), 1, 2)  # -> EDF
            expert_arrays[(layer_idx, "up_proj")] = np.swapaxes(up_EFD.astype(np.float32), 1, 2)  # -> EDF
            expert_fill[(layer_idx, "gate_proj")] = num_experts
            expert_fill[(layer_idx, "up_proj")] = num_experts
        else:
            unmatched.append(torch_key)
        return True

    down_m = re.match(rf"{esc}\.(\d+)\.mlp\.experts\.down_proj(?:\.weight)?", torch_key)
    if down_m:
        layer_idx = int(down_m.group(1))
        if (layer_idx, "down_proj") in expert_arrays:
            down_EDF = np.asarray(get_tensor(torch_key))
            expert_arrays[(layer_idx, "down_proj")] = np.swapaxes(down_EDF.astype(np.float32), 1, 2)  # -> EFD
            expert_fill[(layer_idx, "down_proj")] = num_experts
        else:
            unmatched.append(torch_key)
        return True

    expert_m = re.match(
        rf"{esc}\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
        torch_key,
    )
    if expert_m:
        layer_idx = int(expert_m.group(1))
        expert_idx = int(expert_m.group(2))
        proj_name = expert_m.group(3)
        key = (layer_idx, proj_name)
        if key in expert_arrays:
            tensor = np.asarray(get_tensor(torch_key))
            expert_arrays[key][expert_idx] = tensor.T.astype(np.float32)
            expert_fill[key] += 1
        else:
            unmatched.append(torch_key)
        return True

    router_m = re.match(rf"{esc}\.(\d+)\.mlp\.gate\.weight", torch_key)
    if router_m:
        layer_idx = int(router_m.group(1))
        router_buf[layer_idx] = np.asarray(get_tensor(torch_key))
        return True

    return False


def finalize_experts(
    expert_arrays: dict[tuple[int, str], np.ndarray],
    expert_fill: dict[tuple[int, str], int],
    router_buf: dict[int, Any],
    state_dict: dict[str, Any],
    *,
    num_experts: int,
    jax_layer_prefix: str = "layers",
) -> None:
    """Verify expert fill counts, convert to JAX, and assign into state_dict."""
    for key in list(expert_arrays.keys()):
        layer_idx, proj_name = key
        if expert_fill[key] != num_experts:
            raise RuntimeError(
                f"Layer {layer_idx} {proj_name}: expected {num_experts} experts, "
                f"got {expert_fill[key]}"
            )
        arr = expert_arrays.pop(key)
        value = jnp.asarray(arr)
        assign_to_state_dict(
            state_dict,
            f"{jax_layer_prefix}.{layer_idx}.mlp.{proj_name}",
            value,
            f"expert layer {layer_idx} {proj_name}",
        )
    for layer_idx, router_tensor in router_buf.items():
        value = jnp.asarray(router_tensor.T)
        assign_to_state_dict(
            state_dict,
            f"{jax_layer_prefix}.{layer_idx}.mlp.router.kernel",
            value,
            f"router layer {layer_idx}",
        )


def write_moe_experts_to_hf(
    expert_params: dict[int, dict[str, np.ndarray]],
    router_params: dict[int, np.ndarray],
    hf_tensors: dict[str, np.ndarray],
    *,
    num_layers: int,
    is_moe_layer: Callable[[int], bool],
    hf_prefix: str,
) -> None:
    """Assemble gate/up/down and router from JAX state and write HF expert keys."""
    for layer_idx in range(num_layers):
        if not is_moe_layer(layer_idx):
            continue
        params = expert_params.get(layer_idx, {})
        gate_EDF = params.get("gate_proj")
        up_EDF = params.get("up_proj")
        down_EFD = params.get("down_proj")
        if gate_EDF is None or up_EDF is None or down_EFD is None:
            raise RuntimeError(
                f"Missing expert weights for layer {layer_idx}: "
                f"gate={gate_EDF is not None}, up={up_EDF is not None}, down={down_EFD is not None}"
            )
        gate_EFD = np.swapaxes(gate_EDF, 1, 2)
        up_EFD = np.swapaxes(up_EDF, 1, 2)
        gate_up_E2FD = np.concatenate([gate_EFD, up_EFD], axis=1)
        hf_tensors[f"{hf_prefix}.{layer_idx}.mlp.experts.gate_up_proj"] = gate_up_E2FD.astype(np.float32)
        down_EDF = np.swapaxes(down_EFD, 1, 2)
        hf_tensors[f"{hf_prefix}.{layer_idx}.mlp.experts.down_proj"] = down_EDF.astype(np.float32)
        router_DE = router_params.get(layer_idx)
        if router_DE is None:
            raise RuntimeError(f"Missing router weights for MoE layer {layer_idx}")
        hf_tensors[f"{hf_prefix}.{layer_idx}.mlp.gate.weight"] = router_DE.T.astype(np.float32)


def find_safetensors(file_dir: str | epath.Path) -> list[epath.Path]:
    """Return list of *.safetensors under file_dir. Raises ValueError if none found."""
    path = epath.Path(file_dir).expanduser()
    files = list(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")
    return files


def check_conversion_errors(unmatched: list[str]) -> None:
    """Raise RuntimeError if there are unmatched HuggingFace keys."""
    if unmatched:
        raise RuntimeError(
            f"Unmapped HuggingFace parameters:\n" + "\n".join(sorted(unmatched))
        )


def load_hf_config(path: str | epath.Path) -> dict[str, Any]:
    cfg_path = epath.Path(path) / "config.json"
    if not cfg_path.exists():
        raise ValueError(f"Expected HuggingFace config.json under {path}")
    with cfg_path.open() as f:
        return json.load(f)


def inverse_transform(value, transform_rule: TransformRule):
    """Apply the inverse of a HF->JAX transform to produce HF layout."""
    rule = transform_rule
    if rule is None:
        return value
    permute_rule, reshape_rule, transpose_last = rule
    inv = value
    if transpose_last:
        inv = inv.T
    if reshape_rule is not None:
        inv = inv.reshape(reshape_rule)
    if permute_rule is not None:
        inv_perm = [0] * len(permute_rule)
        for i, p in enumerate(permute_rule):
            inv_perm[p] = i
        inv = inv.transpose(inv_perm)
    return inv


def _to_regex(pattern: str) -> str:
    escaped = re.escape(pattern)
    escaped = re.sub(r"\\\\([0-9]+)", r"([0-9]+)", escaped)
    return escaped


def _hf_template(pattern: str) -> str:
    """Convert an HF regex to a template usable with str.format."""
    template = pattern.replace(r"\.", ".")
    template = re.sub(r"\([^\)]+\)", "{}", template)
    return template


def build_inverse_mapping(mapping: dict[str, tuple[str, Enum | Any]]):
    """Create a list of (jax_regex, hf_template, transform) for export."""
    inverse = []
    for hf_pattern, (jax_pattern, transform) in mapping.items():
        inverse.append((_to_regex(jax_pattern), _hf_template(hf_pattern), transform))
    return inverse


def flatten_pure_state(tree: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested dict of params into dotted keys -> leaf values."""
    flat: dict[str, Any] = {}

    def _recurse(node: Any, path: list[str]):
        if isinstance(node, dict):
            for k, v in node.items():
                _recurse(v, path + [str(k)])
        else:
            flat[".".join(path)] = node

    _recurse(tree, [])
    return flat


def save_hf_config(cfg: dict[str, Any], path: str | epath.Path):
    cfg_path = epath.Path(path) / "config.json"
    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)
