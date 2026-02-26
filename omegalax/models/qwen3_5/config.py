"""Configuration for Qwen3.5 vision-language model."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp

from omegalax.models.shard_config import ShardConfig


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3_5VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    dtype: Any = jnp.float32


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3_5TextConfig:
    vocab_size: int = 248_320
    hidden_size: int = 4096
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    layer_types: tuple[str, ...] = ()
    rope_theta: float = 10_000_000
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, ...] = (11, 11, 10)
    # Stored for config fidelity; ignored in the forward pass (see HF source).
    mrope_interleaved: bool = True
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    # Linear attention (Gated Delta Net) config
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 64
    linear_value_head_dim: int = 128

    # MoE config
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    num_experts: int = 512
    num_experts_per_tok: int = 10
    router_aux_loss_coef: float = 0.001
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.default)
    dtype: Any = jnp.bfloat16


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3_5Config:
    vision_config: Qwen3_5VisionConfig = dataclasses.field(default_factory=Qwen3_5VisionConfig)
    text_config: Qwen3_5TextConfig = dataclasses.field(default_factory=Qwen3_5TextConfig)
    image_token_id: int = 248_056
    video_token_id: int = 248_057
    vision_start_token_id: int = 248_053
    vision_end_token_id: int = 248_054


# Model specs registry
_QWEN3_5_SPECS: dict[str, dict] = {
    "qwen3.5-smoke": {
        "hf_repo_id": None,
        "vision_config": {
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
            "out_hidden_size": 128,
            "num_position_embeddings": 100,
        },
        "text_config": {
            "vocab_size": 1024,
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rms_norm_eps": 1e-6,
            "layer_types": ("linear_attention", "linear_attention", "linear_attention", "full_attention"),
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": (2, 1, 1),
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 32,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        },
    },
    "qwen3.5-397b-a17b": {
        "hf_repo_id": "Qwen/Qwen3.5-397B-A17B",
        "vision_config": {},
        "text_config": {},
    },
}

_MODEL_ID_TO_SPEC: dict[str, str] = {}
for _spec_key, _spec in _QWEN3_5_SPECS.items():
    _MODEL_ID_TO_SPEC[_spec_key] = _spec_key
    _hf_id = _spec.get("hf_repo_id")
    if _hf_id:
        _MODEL_ID_TO_SPEC[_hf_id] = _spec_key


def list_qwen3_5_model_ids() -> list[str]:
    return [s["hf_repo_id"] for s in _QWEN3_5_SPECS.values() if s.get("hf_repo_id")]


def get_qwen3_5_spec(model_id: str) -> dict:
    spec_key = _MODEL_ID_TO_SPEC.get(model_id)
    if spec_key:
        return dict(_QWEN3_5_SPECS[spec_key])
    supported = sorted(_MODEL_ID_TO_SPEC.keys())
    raise ValueError(f"Unsupported Qwen3.5 model_id '{model_id}'. Supported ids: {supported}")


def is_supported_qwen3_5_model_id(model_id: str) -> bool:
    return model_id in _MODEL_ID_TO_SPEC


def list_supported_qwen3_5_model_ids() -> list[str]:
    return sorted(_MODEL_ID_TO_SPEC.keys())


def make_config(model_id: str) -> Qwen3_5Config:
    spec = get_qwen3_5_spec(model_id)
    vis_kw = spec["vision_config"]
    txt_kw = spec["text_config"]
    return Qwen3_5Config(
        vision_config=Qwen3_5VisionConfig(**vis_kw),
        text_config=Qwen3_5TextConfig(**txt_kw),
    )


def _hf_torch_dtype_to_jnp(torch_dtype: str | None) -> Any:
    # FIXME (f.srambical): find the ground-truth hf dtype strings and raise an error on others
    """Map HuggingFace torch_dtype string to jnp.dtype. Defaults to bfloat16 for text."""
    if torch_dtype is None:
        return jnp.bfloat16
    kind = (torch_dtype if isinstance(torch_dtype, str) else str(torch_dtype)).lower()
    if "bfloat16" in kind or "bf16" in kind:
        return jnp.bfloat16
    if "float32" in kind or "fp32" in kind:
        return jnp.float32
    if "float16" in kind or "fp16" in kind:
        return jnp.float16
    return jnp.bfloat16


def make_config_from_hf(hf_cfg: dict[str, Any]) -> Qwen3_5Config:
    """Build a Qwen3_5Config from a HuggingFace config.json dict."""
    vis = hf_cfg["vision_config"]
    txt = hf_cfg["text_config"]
    rope_params = txt.get("rope_parameters") or txt.get("rope_scaling") or {}
    torch_dtype = hf_cfg.get("torch_dtype")
    text_dtype = _hf_torch_dtype_to_jnp(torch_dtype)
    vision_dtype = _hf_torch_dtype_to_jnp(vis.get("torch_dtype")) if vis.get("torch_dtype") is not None else (text_dtype if torch_dtype is not None else jnp.float32)

    return Qwen3_5Config(
        vision_config=Qwen3_5VisionConfig(
            depth=vis["depth"],
            hidden_size=vis["hidden_size"],
            intermediate_size=vis["intermediate_size"],
            num_heads=vis["num_heads"],
            patch_size=vis["patch_size"],
            temporal_patch_size=vis["temporal_patch_size"],
            spatial_merge_size=vis["spatial_merge_size"],
            in_channels=vis["in_channels"],
            out_hidden_size=vis["out_hidden_size"],
            num_position_embeddings=vis["num_position_embeddings"],
            dtype=vision_dtype,
        ),
        text_config=Qwen3_5TextConfig(
            vocab_size=txt["vocab_size"],
            hidden_size=txt["hidden_size"],
            num_hidden_layers=txt["num_hidden_layers"],
            num_attention_heads=txt["num_attention_heads"],
            num_key_value_heads=txt["num_key_value_heads"],
            head_dim=txt["head_dim"],
            rms_norm_eps=txt["rms_norm_eps"],
            layer_types=tuple(txt["layer_types"]),
            rope_theta=rope_params.get("rope_theta") or txt["rope_theta"],
            partial_rotary_factor=rope_params["partial_rotary_factor"],
            mrope_section=tuple(rope_params["mrope_section"]),
            mrope_interleaved=rope_params["mrope_interleaved"],
            attention_bias=txt["attention_bias"],
            tie_word_embeddings=hf_cfg["tie_word_embeddings"],
            linear_conv_kernel_dim=txt["linear_conv_kernel_dim"],
            linear_key_head_dim=txt["linear_key_head_dim"],
            linear_num_key_heads=txt["linear_num_key_heads"],
            linear_num_value_heads=txt["linear_num_value_heads"],
            linear_value_head_dim=txt["linear_value_head_dim"],
            moe_intermediate_size=txt["moe_intermediate_size"],
            shared_expert_intermediate_size=txt["shared_expert_intermediate_size"],
            num_experts=txt.get("num_experts") or txt.get("num_local_experts"),
            num_experts_per_tok=txt["num_experts_per_tok"],
            router_aux_loss_coef=txt["router_aux_loss_coef"],
            dtype=text_dtype,
        ),
        image_token_id=hf_cfg["image_token_id"],
        video_token_id=hf_cfg["video_token_id"],
        vision_start_token_id=hf_cfg["vision_start_token_id"],
        vision_end_token_id=hf_cfg["vision_end_token_id"],
    )
