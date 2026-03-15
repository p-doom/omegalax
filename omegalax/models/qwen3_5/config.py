"""Configuration for Qwen3.5 vision-language model."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from etils import epath

from omegalax.models.params_utils import load_hf_config_from_source
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
    dtype: Any = jnp.bfloat16


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

_QWEN3_5_SMOKE_SPECS: dict[str, dict[str, Any]] = {
    "qwen3.5-smoke": {
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
}

_QWEN3_5_REPOS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
)

_SUPPORTED_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}
_SUPPORTED_MODEL_IDS = sorted((*_QWEN3_5_SMOKE_SPECS.keys(), *_QWEN3_5_REPOS))
_FULL_ATTENTION_INTERVAL = 4


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


def list_qwen3_5_model_ids() -> list[str]:
    return list(_QWEN3_5_REPOS)


def resolve_qwen3_5_repo_id(model_id: str) -> str:
    return model_id


def get_qwen3_5_spec(model_id: str) -> dict[str, Any]:
    if model_id in _QWEN3_5_SMOKE_SPECS:
        return dict(_QWEN3_5_SMOKE_SPECS[model_id])
    if model_id in _QWEN3_5_REPOS:
        return {"hf_repo_id": model_id}
    raise ValueError(f"Unsupported Qwen3.5 model_id '{model_id}'. Supported ids: {_SUPPORTED_MODEL_IDS}")


def is_supported_qwen3_5_model_id(model_id: str) -> bool:
    return model_id in _QWEN3_5_SMOKE_SPECS or model_id in _QWEN3_5_REPOS


def list_supported_qwen3_5_model_ids() -> list[str]:
    return list(_SUPPORTED_MODEL_IDS)


def _generate_layer_types(num_layers: int) -> tuple[str, ...]:
    """Generate layer_types with full_attention every 4th layer (1-indexed)."""
    return tuple(
        "full_attention" if (i + 1) % _FULL_ATTENTION_INTERVAL == 0 else "linear_attention"
        for i in range(num_layers)
    )


def make_config(model_id: str) -> Qwen3_5Config:
    if model_id in _QWEN3_5_SMOKE_SPECS:
        spec = _QWEN3_5_SMOKE_SPECS[model_id]
        return Qwen3_5Config(
            vision_config=Qwen3_5VisionConfig(**spec["vision_config"]),
            text_config=Qwen3_5TextConfig(**spec["text_config"]),
        )

    if "/" not in model_id and not epath.Path(model_id).expanduser().exists():
        raise ValueError(f"Unsupported Qwen3.5 model_id '{model_id}'. Supported ids: {_SUPPORTED_MODEL_IDS}")

    hf_cfg = load_hf_config_from_source(resolve_qwen3_5_repo_id(model_id))
    return make_config_from_hf(hf_cfg)


def make_config_from_hf(hf_cfg: dict[str, Any]) -> Qwen3_5Config:
    """Build a Qwen3_5Config from a HuggingFace config.json dict."""
    model_type = _required(hf_cfg, "model_type", "hf_cfg")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported Qwen3.5 model_type '{model_type}'. Expected one of {sorted(_SUPPORTED_MODEL_TYPES)}."
        )

    vis = _required(hf_cfg, "vision_config", "hf_cfg")
    txt = _required(hf_cfg, "text_config", "hf_cfg")
    if not isinstance(vis, dict):
        raise ValueError("Expected vision_config to be a dict in hf_cfg.")
    if not isinstance(txt, dict):
        raise ValueError("Expected text_config to be a dict in hf_cfg.")

    rope_params = _required(txt, "rope_parameters", "hf_cfg['text_config']")
    if not isinstance(rope_params, dict):
        raise ValueError("Expected rope_parameters to be a dict in hf_cfg['text_config'].")
    rope_type = rope_params.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"Unsupported rope_parameters.rope_type '{rope_type}' for Qwen3.5.")

    text_dtype = _hf_dtype_to_jnp(_required(txt, "dtype", "hf_cfg['text_config']"))
    vision_dtype = _hf_dtype_to_jnp(vis["dtype"]) if vis.get("dtype") is not None else text_dtype

    has_moe = model_type == "qwen3_5_moe"
    if has_moe == ("intermediate_size" in txt):
        raise ValueError(
            "Qwen3.5 config must map cleanly to exactly one of dense or MoE text settings."
        )

    text_kw: dict[str, Any] = {
        "vocab_size": _required(txt, "vocab_size", "hf_cfg['text_config']"),
        "hidden_size": _required(txt, "hidden_size", "hf_cfg['text_config']"),
        "num_hidden_layers": _required(txt, "num_hidden_layers", "hf_cfg['text_config']"),
        "num_attention_heads": _required(txt, "num_attention_heads", "hf_cfg['text_config']"),
        "num_key_value_heads": _required(txt, "num_key_value_heads", "hf_cfg['text_config']"),
        "head_dim": _required(txt, "head_dim", "hf_cfg['text_config']"),
        "rms_norm_eps": _required(txt, "rms_norm_eps", "hf_cfg['text_config']"),
        "layer_types": tuple(_required(txt, "layer_types", "hf_cfg['text_config']")),
        "rope_theta": _required(rope_params, "rope_theta", "rope_parameters"),
        "partial_rotary_factor": _required(rope_params, "partial_rotary_factor", "rope_parameters"),
        "mrope_section": tuple(_required(rope_params, "mrope_section", "rope_parameters")),
        "mrope_interleaved": _required(rope_params, "mrope_interleaved", "rope_parameters"),
        "attention_bias": _required(txt, "attention_bias", "hf_cfg['text_config']"),
        "tie_word_embeddings": _required(hf_cfg, "tie_word_embeddings", "hf_cfg"),
        "linear_conv_kernel_dim": _required(txt, "linear_conv_kernel_dim", "hf_cfg['text_config']"),
        "linear_key_head_dim": _required(txt, "linear_key_head_dim", "hf_cfg['text_config']"),
        "linear_num_key_heads": _required(txt, "linear_num_key_heads", "hf_cfg['text_config']"),
        "linear_num_value_heads": _required(txt, "linear_num_value_heads", "hf_cfg['text_config']"),
        "linear_value_head_dim": _required(txt, "linear_value_head_dim", "hf_cfg['text_config']"),
        "dtype": text_dtype,
    }

    if has_moe:
        text_kw.update(
            moe_intermediate_size=_required(txt, "moe_intermediate_size", "hf_cfg['text_config']"),
            shared_expert_intermediate_size=_required(
                txt, "shared_expert_intermediate_size", "hf_cfg['text_config']"
            ),
            num_experts=_required(txt, "num_experts", "hf_cfg['text_config']"),
            num_experts_per_tok=_required(txt, "num_experts_per_tok", "hf_cfg['text_config']"),
            router_aux_loss_coef=_required(txt, "router_aux_loss_coef", "hf_cfg['text_config']"),
        )
    else:
        text_kw["intermediate_size"] = _required(txt, "intermediate_size", "hf_cfg['text_config']")

    return Qwen3_5Config(
        vision_config=Qwen3_5VisionConfig(
            depth=_required(vis, "depth", "hf_cfg['vision_config']"),
            hidden_size=_required(vis, "hidden_size", "hf_cfg['vision_config']"),
            intermediate_size=_required(vis, "intermediate_size", "hf_cfg['vision_config']"),
            num_heads=_required(vis, "num_heads", "hf_cfg['vision_config']"),
            patch_size=_required(vis, "patch_size", "hf_cfg['vision_config']"),
            temporal_patch_size=_required(vis, "temporal_patch_size", "hf_cfg['vision_config']"),
            spatial_merge_size=_required(vis, "spatial_merge_size", "hf_cfg['vision_config']"),
            in_channels=_required(vis, "in_channels", "hf_cfg['vision_config']"),
            out_hidden_size=_required(vis, "out_hidden_size", "hf_cfg['vision_config']"),
            num_position_embeddings=_required(vis, "num_position_embeddings", "hf_cfg['vision_config']"),
            dtype=vision_dtype,
        ),
        text_config=Qwen3_5TextConfig(**text_kw),
        image_token_id=_required(hf_cfg, "image_token_id", "hf_cfg"),
        video_token_id=_required(hf_cfg, "video_token_id", "hf_cfg"),
        vision_start_token_id=_required(hf_cfg, "vision_start_token_id", "hf_cfg"),
        vision_end_token_id=_required(hf_cfg, "vision_end_token_id", "hf_cfg"),
    )
