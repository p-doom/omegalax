"""Public loader/export entrypoints for Qwen3-VL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from safetensors import numpy as stnp

from omegalax.models.params_utils import (
    Transform,
    build_inverse_mapping,
    flatten_pure_state,
    inverse_transform,
    save_hf_config,
    write_moe_experts_to_hf,
)
from .config import Qwen3VLConfig
from .loader import _get_non_expert_mapping, create_qwen3_vl_from_safetensors
from .model import Qwen3VL

__all__ = ["create_qwen3_vl_from_safetensors", "_get_key_and_transform_mapping", "export_qwen3_vl_to_safetensors"]


def _make_hf_config_dict(cfg: Qwen3VLConfig) -> dict[str, Any]:
    model_type = "qwen3_vl_moe" if cfg.num_experts > 0 else "qwen3_vl"
    text_model_type = "qwen3_vl_moe_text" if cfg.num_experts > 0 else "qwen3_vl_text"
    return {
        "model_type": model_type,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "image_token_id": cfg.image_token_id,
        "video_token_id": cfg.video_token_id,
        "vision_start_token_id": cfg.vision_start_token_id,
        "vision_config": {
            "hidden_size": cfg.vision.hidden_size,
            "intermediate_size": cfg.vision.intermediate_size,
            "num_heads": cfg.vision.num_heads,
            "patch_size": cfg.vision.patch_size,
            "temporal_patch_size": cfg.vision.temporal_patch_size,
            "in_channels": cfg.vision.in_channels,
            "spatial_merge_size": cfg.vision.spatial_merge_size,
            "out_hidden_size": cfg.vision.out_hidden_size,
            "depth": cfg.vision.depth,
            "hidden_act": cfg.vision.hidden_act,
            "num_position_embeddings": cfg.vision.num_position_embeddings,
            "deepstack_visual_indexes": list(cfg.vision.deepstack_visual_indexes),
            "model_type": "qwen3_vl",
        },
        "text_config": {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.emb_dim,
            "intermediate_size": cfg.mlp_dim,
            "num_hidden_layers": cfg.num_layers,
            "num_attention_heads": cfg.num_heads,
            "num_key_value_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "rms_norm_eps": cfg.norm_eps,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "rope_parameters": {
                "rope_theta": cfg.rope_theta,
                "mrope_section": list(cfg.mrope_section),
            },
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "num_experts": cfg.num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "mlp_only_layers": list(cfg.mlp_only_layers),
            "decoder_sparse_step": cfg.decoder_sparse_step,
            "norm_topk_prob": cfg.norm_topk_prob,
            "model_type": text_model_type,
        },
    }


def export_qwen3_vl_to_safetensors(model: Qwen3VL, cfg: Qwen3VLConfig, out_dir: str | Path) -> Path:
    """Export a Qwen3-VL nnx model to HuggingFace-style safetensors."""
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    tensor_path = out_path / "model.safetensors"

    _, abs_state = nnx.split(model)
    pure_state = nnx.to_pure_dict(abs_state)
    flat_state = flatten_pure_state(pure_state)
    inverse_mapping = build_inverse_mapping(_get_non_expert_mapping())

    hf_tensors: dict[str, np.ndarray] = {}
    unmatched: list[str] = []

    expert_params: dict[int, dict[str, np.ndarray]] = {}
    router_params: dict[int, np.ndarray] = {}

    def _handle_special(jax_key: str, value) -> bool:
        if jax_key == "vision.patch_embed.proj.kernel":
            arr = np.asarray(jax.device_get(value))
            out_channels = cfg.vision.hidden_size
            in_channels = cfg.vision.in_channels
            temporal = cfg.vision.temporal_patch_size
            patch = cfg.vision.patch_size
            reshaped = arr.T.reshape(out_channels, in_channels, temporal, patch, patch)
            hf_tensors["model.visual.patch_embed.proj.weight"] = reshaped.astype(np.float32)
            return True
        if cfg.num_experts > 0:
            m = re.fullmatch(r"text.layers\.([0-9]+)\.mlp\.(.+)", jax_key)
            if m:
                layer_idx = int(m.group(1))
                if not cfg.is_moe_layer(layer_idx):
                    return False
                suffix = m.group(2)
                if suffix == "gate_proj":
                    expert_params.setdefault(layer_idx, {})["gate_proj"] = np.asarray(jax.device_get(value))
                    return True
                if suffix == "up_proj":
                    expert_params.setdefault(layer_idx, {})["up_proj"] = np.asarray(jax.device_get(value))
                    return True
                if suffix == "down_proj":
                    expert_params.setdefault(layer_idx, {})["down_proj"] = np.asarray(jax.device_get(value))
                    return True
                if suffix == "router.kernel":
                    router_params[layer_idx] = np.asarray(jax.device_get(value))
                    return True
        return False

    for jax_key, value in flat_state.items():
        if _handle_special(jax_key, value):
            continue

        matched = False
        for jax_regex, hf_template, transform in inverse_mapping:
            m = re.fullmatch(jax_regex, jax_key)
            if not m:
                continue
            hf_key = hf_template.format(*m.groups())
            arr = np.asarray(jax.device_get(value))
            if transform == Transform.CONV3D:
                raise RuntimeError(f"Unexpected CONV3D transform in generic export path for {jax_key}")
            transform_rule = transform.value if hasattr(transform, "value") else transform
            arr = inverse_transform(arr, transform_rule)
            hf_tensors[hf_key] = arr
            matched = True
            break
        if not matched:
            unmatched.append(jax_key)

    if cfg.num_experts > 0:
        write_moe_experts_to_hf(
            expert_params,
            router_params,
            hf_tensors,
            num_layers=cfg.num_layers,
            is_moe_layer=cfg.is_moe_layer,
            hf_prefix="model.language_model.layers",
        )

    if unmatched:
        raise RuntimeError(f"Unmapped JAX parameters during export:\n" + "\n".join(sorted(unmatched)))

    stnp.save_file(hf_tensors, tensor_path)
    save_hf_config(_make_hf_config_dict(cfg), out_path)

    return tensor_path
