"""Public loader/export entrypoints for Qwen3.5."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from safetensors import numpy as stnp

from omegalax.models.params_utils import (
    build_inverse_mapping,
    flatten_pure_state,
    inverse_transform,
    save_hf_config,
)
from .config import Qwen3_5Config
from .loader import _get_non_expert_mapping, create_qwen3_5_from_safetensors
from .model import Qwen3_5ForConditionalGeneration

__all__ = ["create_qwen3_5_from_safetensors", "export_qwen3_5_to_safetensors"]


def _make_hf_config_dict(cfg: Qwen3_5Config) -> dict[str, Any]:
    txt = cfg.text_config
    vis = cfg.vision_config
    return {
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": txt.tie_word_embeddings,
        "image_token_id": cfg.image_token_id,
        "video_token_id": cfg.video_token_id,
        "vision_start_token_id": cfg.vision_start_token_id,
        "vision_end_token_id": cfg.vision_end_token_id,
        "vision_config": {
            "depth": vis.depth,
            "hidden_size": vis.hidden_size,
            "intermediate_size": vis.intermediate_size,
            "num_heads": vis.num_heads,
            "patch_size": vis.patch_size,
            "temporal_patch_size": vis.temporal_patch_size,
            "spatial_merge_size": vis.spatial_merge_size,
            "in_channels": vis.in_channels,
            "out_hidden_size": vis.out_hidden_size,
            "num_position_embeddings": vis.num_position_embeddings,
            "hidden_act": "gelu_pytorch_tanh",
            "model_type": "qwen3_5_moe",
        },
        "text_config": {
            "vocab_size": txt.vocab_size,
            "hidden_size": txt.hidden_size,
            "num_hidden_layers": txt.num_hidden_layers,
            "num_attention_heads": txt.num_attention_heads,
            "num_key_value_heads": txt.num_key_value_heads,
            "head_dim": txt.head_dim,
            "hidden_act": "silu",
            "rms_norm_eps": txt.rms_norm_eps,
            "layer_types": list(txt.layer_types),
            "rope_parameters": {
                "rope_theta": txt.rope_theta,
                "partial_rotary_factor": txt.partial_rotary_factor,
                "mrope_section": list(txt.mrope_section),
                "mrope_interleaved": txt.mrope_interleaved,
                "rope_type": "default",
            },
            "attention_bias": txt.attention_bias,
            "tie_word_embeddings": txt.tie_word_embeddings,
            "linear_conv_kernel_dim": txt.linear_conv_kernel_dim,
            "linear_key_head_dim": txt.linear_key_head_dim,
            "linear_num_key_heads": txt.linear_num_key_heads,
            "linear_num_value_heads": txt.linear_num_value_heads,
            "linear_value_head_dim": txt.linear_value_head_dim,
            "moe_intermediate_size": txt.moe_intermediate_size,
            "shared_expert_intermediate_size": txt.shared_expert_intermediate_size,
            "num_experts": txt.num_experts,
            "num_experts_per_tok": txt.num_experts_per_tok,
            "router_aux_loss_coef": txt.router_aux_loss_coef,
            # FIXME (f.srambical) check whether we need the _text-suffix
            "model_type": "qwen3_5_moe_text",
        },
    }


def export_qwen3_5_to_safetensors(
    model: Qwen3_5ForConditionalGeneration, cfg: Qwen3_5Config, out_dir: str | Path
) -> Path:
    """Export a Qwen3.5 nnx model to HuggingFace-style safetensors."""
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    tensor_path = out_path / "model.safetensors"

    _, abs_state = nnx.split(model)
    pure_state = nnx.to_pure_dict(abs_state)
    flat_state = flatten_pure_state(pure_state)

    inverse_mapping = build_inverse_mapping(_get_non_expert_mapping())

    hf_tensors: dict[str, np.ndarray] = {}
    unmatched: list[str] = []

    def _handle_special(jax_key: str, value) -> bool:
        if jax_key == "vision.patch_embed.proj.kernel":
            arr = np.asarray(jax.device_get(value))
            hf_arr = arr.transpose(4, 3, 0, 1, 2)
            hf_tensors["model.visual.patch_embed.proj.weight"] = hf_arr.astype(np.float32)
            return True

        # FIXME (f.srambical)
        # Linear attention norm.weight (SCALE, no transform)
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.linear_attn\.norm\.weight", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.linear_attn.norm.weight"] = arr.astype(np.float32)
            return True

        # Vision merger norm.weight (SCALE, no transform)
        if jax_key == "vision.merger.norm.weight":
            arr = np.asarray(jax.device_get(value))
            hf_tensors["model.visual.merger.norm.weight"] = arr.astype(np.float32)
            return True

        # Linear attention conv1d weight: (C, K) -> (C, 1, K)
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.linear_attn\.conv_weight", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.linear_attn.conv1d.weight"] = arr[:, None, :].astype(
                np.float32
            )
            return True

        # dt_bias
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.linear_attn\.dt_bias", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.linear_attn.dt_bias"] = arr.astype(np.float32)
            return True

        # A_log
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.linear_attn\.A_log", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.linear_attn.A_log"] = arr.astype(np.float32)
            return True

        # MoE expert fused gate/up: stored as-is (E, 2F, D)
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.mlp\.gate_up_proj", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"] = arr.astype(np.float32)
            return True

        # MoE expert down_proj: stored as-is (E, D, F)
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.mlp\.down_proj", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"] = arr.astype(np.float32)
            return True

        # MoE router: (D, E) -> (E, D)
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.mlp\.router\.kernel", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.mlp.gate.weight"] = arr.T.astype(np.float32)
            return True

        # Shared expert gate: (D, 1) -> (1, D)
        m = re.fullmatch(r"text\.layers\.([0-9]+)\.mlp\.shared_expert_gate\.kernel", jax_key)
        if m:
            layer_idx = m.group(1)
            arr = np.asarray(jax.device_get(value))
            hf_tensors[f"model.language_model.layers.{layer_idx}.mlp.shared_expert_gate.weight"] = (
                arr.T.astype(np.float32)
            )
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
            transform_rule = transform.value if hasattr(transform, "value") else transform
            arr = inverse_transform(arr, transform_rule)
            hf_tensors[hf_key] = arr
            matched = True
            break
        if not matched:
            unmatched.append(jax_key)

    if unmatched:
        raise RuntimeError(f"Unmapped JAX parameters during export:\n" + "\n".join(sorted(unmatched)))

    stnp.save_file(hf_tensors, tensor_path)
    save_hf_config(_make_hf_config_dict(cfg), out_path)

    return tensor_path
