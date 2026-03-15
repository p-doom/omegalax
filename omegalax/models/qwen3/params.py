"""Unified loader/export entrypoints for Qwen3 (dense + MoE)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from safetensors import numpy as stnp
from etils import epath

from omegalax.models.params_utils import (
    build_inverse_mapping,
    flatten_pure_state,
    inverse_transform,
    save_hf_config,
    write_moe_experts_to_hf,
)
from .config import Qwen3Config
from .loader import (
    _get_key_mapping,
    create_qwen3_from_safetensors,
)
from .model import Qwen3

__all__ = [
    "create_qwen3_from_safetensors",
    "export_qwen3_to_safetensors",
    "qwen3_to_hf_config_dict",
]


def _jnp_dtype_to_hf(dtype: Any) -> str:
    kind = str(dtype).lower()
    if "bfloat16" in kind:
        return "bfloat16"
    if "float16" in kind:
        return "float16"
    if "float32" in kind:
        return "float32"
    raise ValueError(f"Unsupported dtype for HF config export: {dtype!r}")


def qwen3_to_hf_config_dict(cfg: Qwen3Config) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dtype": _jnp_dtype_to_hf(cfg.dtype),
        "model_type": "qwen3_moe" if cfg.is_moe else "qwen3",
        "vocab_size": cfg.vocab_size,
        "num_hidden_layers": cfg.num_layers,
        "hidden_size": cfg.emb_dim,
        "num_attention_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.mlp_dim,
        "rms_norm_eps": cfg.norm_eps,
        "rope_parameters": {
            "rope_theta": cfg.rope_theta,
            "rope_type": "default",
        },
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_word_embeddings,
    }
    if cfg.rope_scaling_factor is not None:
        result["rope_parameters"]["factor"] = cfg.rope_scaling_factor
    if cfg.local_rope_theta is not None:
        result["rope_parameters"]["local_rope_theta"] = cfg.local_rope_theta
    if cfg.is_moe:
        result.update(
            num_experts=cfg.num_experts,
            num_local_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            mlp_only_layers=list(cfg.mlp_only_layers),
            decoder_sparse_step=cfg.decoder_sparse_step,
            norm_topk_prob=cfg.norm_topk_prob,
            router_aux_loss_coef=cfg.aux_loss_coef,
        )
    return result


def export_qwen3_to_safetensors(
    model: Qwen3, cfg: Qwen3Config, out_dir: str | Path | epath.Path,
) -> Path | epath.Path:
    """Export a Qwen3 nnx model (dense or MoE) to HuggingFace-style safetensors."""
    out_dir = epath.Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / "model.safetensors"

    _, abs_state = nnx.split(model)
    pure_state = nnx.to_pure_dict(abs_state)
    flat_state = flatten_pure_state(pure_state)
    inverse_mapping = build_inverse_mapping(_get_key_mapping())

    hf_tensors: dict[str, np.ndarray] = {}
    unmatched: list[str] = []

    expert_params: dict[int, dict[str, np.ndarray]] = {}
    router_params: dict[int, np.ndarray] = {}

    def _handle_moe_special(jax_key: str, value) -> bool:
        if not cfg.is_moe:
            return False

        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.gate_proj", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            expert_params.setdefault(int(m.group(1)), {})["gate_proj"] = np.asarray(jax.device_get(value))
            return True

        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.up_proj", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            expert_params.setdefault(int(m.group(1)), {})["up_proj"] = np.asarray(jax.device_get(value))
            return True

        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.down_proj", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            expert_params.setdefault(int(m.group(1)), {})["down_proj"] = np.asarray(jax.device_get(value))
            return True

        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.router\.kernel", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            router_params[int(m.group(1))] = np.asarray(jax.device_get(value))
            return True

        return False

    for jax_key, value in flat_state.items():
        if _handle_moe_special(jax_key, value):
            continue

        matched = False
        for jax_regex, hf_template, transform in inverse_mapping:
            m = re.fullmatch(jax_regex, jax_key)
            if not m:
                continue
            hf_key = hf_template.format(*m.groups())
            arr = np.asarray(jax.device_get(value))
            transform_rule = transform.value if hasattr(transform, "value") else transform
            arr = inverse_transform(arr, transform_rule)
            hf_tensors[hf_key] = arr
            matched = True
            break
        if not matched:
            unmatched.append(jax_key)

    if cfg.is_moe:
        write_moe_experts_to_hf(
            expert_params, router_params, hf_tensors,
            num_layers=cfg.num_layers, is_moe_layer=cfg.is_moe_layer,
            hf_prefix="model.layers",
        )

    if unmatched:
        missing = "\n".join(sorted(unmatched))
        raise RuntimeError(f"Unmapped JAX parameters during export:\n{missing}")

    stnp.save_file(hf_tensors, str(tensor_path))
    save_hf_config(qwen3_to_hf_config_dict(cfg), out_dir)

    return tensor_path
