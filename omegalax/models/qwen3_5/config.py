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

    # Dense FFN config (used when num_experts == 0)
    intermediate_size: int = 0

    # MoE config (used when num_experts > 0)
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    router_aux_loss_coef: float = 0.001
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.default)
    dtype: Any = jnp.bfloat16

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3_5Config:
    vision_config: Qwen3_5VisionConfig = dataclasses.field(default_factory=Qwen3_5VisionConfig)
    text_config: Qwen3_5TextConfig = dataclasses.field(default_factory=Qwen3_5TextConfig)
    image_token_id: int = 248_056
    video_token_id: int = 248_057
    vision_start_token_id: int = 248_053
    vision_end_token_id: int = 248_054


_SMOKE_VISION = {
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
}

_SMOKE_LINEAR_ATTN = {
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 32,
}

# Model specs registry
_QWEN3_5_SPECS: dict[str, dict] = {
    "qwen3.5-smoke": {
        "hf_repo_id": None,
        "vision_config": _SMOKE_VISION,
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
            **_SMOKE_LINEAR_ATTN,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        },
    },
    "qwen3.5-smoke-dense": {
        "hf_repo_id": None,
        "vision_config": _SMOKE_VISION,
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
            **_SMOKE_LINEAR_ATTN,
            "intermediate_size": 256,
        },
    },
    "qwen3.5-0.8b": {
        "hf_repo_id": "Qwen/Qwen3.5-0.8B",
        "vision_config": {
            "depth": 12,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_heads": 12,
            "out_hidden_size": 1024,
        },
        "text_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 3584,
            "linear_num_value_heads": 16,
            "tie_word_embeddings": True,
        },
    },
    "qwen3.5-2b": {
        "hf_repo_id": "Qwen/Qwen3.5-2B",
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "out_hidden_size": 2048,
        },
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 6144,
            "linear_num_value_heads": 16,
            "tie_word_embeddings": True,
        },
    },
    "qwen3.5-4b": {
        "hf_repo_id": "Qwen/Qwen3.5-4B",
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "out_hidden_size": 2560,
        },
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 9216,
            "linear_num_value_heads": 32,
            "tie_word_embeddings": True,
        },
    },
    "qwen3.5-9b": {
        "hf_repo_id": "Qwen/Qwen3.5-9B",
        "vision_config": {},
        "text_config": {
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 12288,
            "linear_num_value_heads": 32,
        },
    },
    "qwen3.5-27b": {
        "hf_repo_id": "Qwen/Qwen3.5-27B",
        "vision_config": {
            "out_hidden_size": 5120,
        },
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "intermediate_size": 17408,
            "linear_num_value_heads": 48,
        },
    },
    "qwen3.5-35b-a3b": {
        "hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "vision_config": {
            "out_hidden_size": 2048,
        },
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "linear_num_value_heads": 32,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
    },
    "qwen3.5-122b-a10b": {
        "hf_repo_id": "Qwen/Qwen3.5-122B-A10B",
        "vision_config": {
            "out_hidden_size": 3072,
        },
        "text_config": {
            "hidden_size": 3072,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
    },
    "qwen3.5-397b-a17b": {
        "hf_repo_id": "Qwen/Qwen3.5-397B-A17B",
        "vision_config": {},
        "text_config": {
            "num_hidden_layers": 60,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "num_experts": 512,
            "num_experts_per_tok": 10,
        },
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


_FULL_ATTENTION_INTERVAL = 4


def _generate_layer_types(num_layers: int) -> tuple[str, ...]:
    """Generate layer_types with full_attention every 4th layer (1-indexed)."""
    return tuple(
        "full_attention" if (i + 1) % _FULL_ATTENTION_INTERVAL == 0 else "linear_attention"
        for i in range(num_layers)
    )


def make_config(model_id: str) -> Qwen3_5Config:
    spec = get_qwen3_5_spec(model_id)
    vis_kw = spec["vision_config"]
    txt_kw = dict(spec["text_config"])
    if not txt_kw.get("layer_types"):
        txt_kw["layer_types"] = _generate_layer_types(txt_kw["num_hidden_layers"])
    return Qwen3_5Config(
        vision_config=Qwen3_5VisionConfig(**vis_kw),
        text_config=Qwen3_5TextConfig(**txt_kw),
    )


def _required(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {where}.")
    return mapping[key]


def _hf_dtype_to_jnp(hf_dtype: str) -> Any:
    """Map HuggingFace dtype string to jnp.dtype."""
    kind = (hf_dtype if isinstance(hf_dtype, str) else str(hf_dtype)).lower()
    if "bfloat16" in kind or "bf16" in kind:
        return jnp.bfloat16
    if "float32" in kind or "fp32" in kind:
        return jnp.float32
    if "float16" in kind or "fp16" in kind:
        return jnp.float16
    raise ValueError(f"Unsupported dtype '{hf_dtype}'.")


def make_config_from_hf(hf_cfg: dict[str, Any]) -> Qwen3_5Config:
    """Build a Qwen3_5Config from a HuggingFace config.json dict."""
    vis = _required(hf_cfg, "vision_config", "hf_cfg")
    txt = _required(hf_cfg, "text_config", "hf_cfg")
    rope_params = _required(txt, "rope_parameters", "hf_cfg['text_config']")
    if not isinstance(rope_params, dict):
        raise ValueError("Expected rope_parameters to be a dict in hf_cfg['text_config'].")
    text_dtype = _hf_dtype_to_jnp(_required(txt, "dtype", "hf_cfg['text_config']"))
    vision_dtype = _hf_dtype_to_jnp(vis["dtype"]) if vis.get("dtype") is not None else jnp.float32

    has_moe = "num_experts" in txt and txt["num_experts"] > 0

    text_kw: dict[str, Any] = {
        "vocab_size": txt["vocab_size"],
        "hidden_size": txt["hidden_size"],
        "num_hidden_layers": txt["num_hidden_layers"],
        "num_attention_heads": txt["num_attention_heads"],
        "num_key_value_heads": txt["num_key_value_heads"],
        "head_dim": txt["head_dim"],
        "rms_norm_eps": txt["rms_norm_eps"],
        "layer_types": tuple(txt["layer_types"]),
        "rope_theta": _required(rope_params, "rope_theta", "rope_parameters"),
        "partial_rotary_factor": rope_params["partial_rotary_factor"],
        "mrope_section": tuple(rope_params["mrope_section"]),
        "mrope_interleaved": rope_params["mrope_interleaved"],
        "attention_bias": txt["attention_bias"],
        "tie_word_embeddings": hf_cfg["tie_word_embeddings"],
        "linear_conv_kernel_dim": txt["linear_conv_kernel_dim"],
        "linear_key_head_dim": txt["linear_key_head_dim"],
        "linear_num_key_heads": txt["linear_num_key_heads"],
        "linear_num_value_heads": txt["linear_num_value_heads"],
        "linear_value_head_dim": txt["linear_value_head_dim"],
        "dtype": text_dtype,
    }

    if has_moe:
        text_kw.update(
            moe_intermediate_size=txt["moe_intermediate_size"],
            shared_expert_intermediate_size=txt["shared_expert_intermediate_size"],
            num_experts=txt["num_experts"],
            num_experts_per_tok=txt["num_experts_per_tok"],
            router_aux_loss_coef=txt["router_aux_loss_coef"],
        )
    else:
        text_kw["intermediate_size"] = txt["intermediate_size"]

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
        text_config=Qwen3_5TextConfig(**text_kw),
        image_token_id=hf_cfg["image_token_id"],
        video_token_id=hf_cfg["video_token_id"],
        vision_start_token_id=hf_cfg["vision_start_token_id"],
        vision_end_token_id=hf_cfg["vision_end_token_id"],
    )
