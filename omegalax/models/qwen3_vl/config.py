"""Configuration for Qwen3-VL models."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from etils import epath

from omegalax.models.params_utils import load_hf_config_from_source
from omegalax.models.shard_config import ShardConfig


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3VLVisionConfig:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    patch_size: int
    temporal_patch_size: int
    in_channels: int
    spatial_merge_size: int
    out_hidden_size: int
    depth: int
    hidden_act: str
    num_position_embeddings: int
    deepstack_visual_indexes: tuple[int, ...]
    dtype: Any = jnp.bfloat16


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3VLConfig:
    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: float
    norm_eps: float
    tie_word_embeddings: bool
    mrope_section: tuple[int, ...]
    vision: Qwen3VLVisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    # MoE settings; zero/empty means dense.
    moe_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    mlp_only_layers: tuple[int, ...] = dataclasses.field(default_factory=tuple)
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = True
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.default)
    dtype: Any = jnp.bfloat16

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts > 0
            and layer_idx not in self.mlp_only_layers
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )


_QWEN3_VL_SMOKE_SPECS: dict[str, dict[str, Any]] = {
    "qwen3-vl-smoke": {
        "vocab_size": 1024,
        "emb_dim": 128,
        "mlp_dim": 512,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 32,
        "num_kv_heads": 4,
        "rope_theta": 1_000_000,
        "norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "mrope_section": (8, 4, 4),
        "image_token_id": 2,
        "video_token_id": 3,
        "vision_start_token_id": 4,
        "vision": {
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_heads": 4,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "out_hidden_size": 128,
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": (0,),
        },
        "moe_intermediate_size": 0,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "mlp_only_layers": (),
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
    },
    "qwen3-vl-smoke-moe": {
        "vocab_size": 1024,
        "emb_dim": 128,
        "mlp_dim": 512,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 32,
        "num_kv_heads": 2,
        "rope_theta": 1_000_000,
        "norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "mrope_section": (8, 4, 4),
        "image_token_id": 2,
        "video_token_id": 3,
        "vision_start_token_id": 4,
        "vision": {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "out_hidden_size": 128,
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": (0,),
        },
        "moe_intermediate_size": 128,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "mlp_only_layers": (),
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
    },
}

_QWEN3_VL_REPOS = (
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
)

_SUPPORTED_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe"}
_SUPPORTED_MODEL_IDS = sorted((*_QWEN3_VL_SMOKE_SPECS.keys(), *_QWEN3_VL_REPOS))


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


def list_qwen3_vl_model_ids() -> list[str]:
    return list(_QWEN3_VL_REPOS)


def get_vl_spec(model_id: str) -> dict[str, Any]:
    if model_id in _QWEN3_VL_SMOKE_SPECS:
        return dict(_QWEN3_VL_SMOKE_SPECS[model_id])
    if model_id in _QWEN3_VL_REPOS:
        return {"hf_repo_id": model_id}
    raise ValueError(f"Unsupported Qwen3-VL model_id '{model_id}'. Supported ids: {_SUPPORTED_MODEL_IDS}")


def resolve_qwen3_vl_repo_id(model_id: str) -> str:
    return model_id


def is_supported_qwen3_vl_model_id(model_id: str) -> bool:
    return model_id in _QWEN3_VL_SMOKE_SPECS or model_id in _QWEN3_VL_REPOS


def list_supported_qwen3_vl_model_ids() -> list[str]:
    return list(_SUPPORTED_MODEL_IDS)


def make_vl_config(model_id: str) -> Qwen3VLConfig:
    if model_id in _QWEN3_VL_SMOKE_SPECS:
        spec = _QWEN3_VL_SMOKE_SPECS[model_id]
        vis = spec["vision"]
        return Qwen3VLConfig(
            num_layers=spec["num_layers"],
            vocab_size=spec["vocab_size"],
            emb_dim=spec["emb_dim"],
            mlp_dim=spec["mlp_dim"],
            num_heads=spec["num_heads"],
            head_dim=spec["head_dim"],
            num_kv_heads=spec["num_kv_heads"],
            rope_theta=spec["rope_theta"],
            norm_eps=spec["norm_eps"],
            tie_word_embeddings=spec["tie_word_embeddings"],
            mrope_section=tuple(spec["mrope_section"]),
            moe_intermediate_size=spec["moe_intermediate_size"],
            num_experts=spec["num_experts"],
            num_experts_per_tok=spec["num_experts_per_tok"],
            mlp_only_layers=tuple(spec["mlp_only_layers"]),
            decoder_sparse_step=spec["decoder_sparse_step"],
            norm_topk_prob=spec["norm_topk_prob"],
            image_token_id=spec["image_token_id"],
            video_token_id=spec["video_token_id"],
            vision_start_token_id=spec["vision_start_token_id"],
            vision=Qwen3VLVisionConfig(
                hidden_size=vis["hidden_size"],
                intermediate_size=vis["intermediate_size"],
                num_heads=vis["num_heads"],
                patch_size=vis["patch_size"],
                temporal_patch_size=vis["temporal_patch_size"],
                in_channels=vis["in_channels"],
                spatial_merge_size=vis["spatial_merge_size"],
                out_hidden_size=vis["out_hidden_size"],
                depth=vis["depth"],
                hidden_act=vis["hidden_act"],
                num_position_embeddings=vis["num_position_embeddings"],
                deepstack_visual_indexes=tuple(vis["deepstack_visual_indexes"]),
            ),
        )

    if "/" not in model_id and not epath.Path(model_id).expanduser().exists():
        raise ValueError(f"Unsupported Qwen3-VL model_id '{model_id}'. Supported ids: {_SUPPORTED_MODEL_IDS}")

    hf_cfg = load_hf_config_from_source(resolve_qwen3_vl_repo_id(model_id))
    return make_vl_config_from_hf(hf_cfg)


def make_vl_config_from_hf(hf_cfg: dict[str, Any]) -> Qwen3VLConfig:
    """Build a Qwen3VLConfig from a HuggingFace config.json dict."""
    model_type = _required(hf_cfg, "model_type", "hf_cfg")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported Qwen3-VL model_type '{model_type}'. Expected one of {sorted(_SUPPORTED_MODEL_TYPES)}."
        )

    vis = _required(hf_cfg, "vision_config", "hf_cfg")
    txt = _required(hf_cfg, "text_config", "hf_cfg")
    if not isinstance(vis, dict):
        raise ValueError("Expected vision_config to be a dict in hf_cfg.")
    if not isinstance(txt, dict):
        raise ValueError("Expected text_config to be a dict in hf_cfg.")

    rope_theta = _required(txt, "rope_theta", "hf_cfg['text_config']")
    rope_scaling = _required(txt, "rope_scaling", "hf_cfg['text_config']")
    if not isinstance(rope_scaling, dict):
        raise ValueError("Expected rope_scaling to be a dict in hf_cfg['text_config'].")
    rope_type = rope_scaling.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"Unsupported rope_scaling.rope_type '{rope_type}' for Qwen3-VL.")

    mrope_section = _required(rope_scaling, "mrope_section", "hf_cfg['text_config'].rope_scaling")
    text_dtype = _hf_dtype_to_jnp(_required(txt, "dtype", "hf_cfg['text_config']"))
    vision_dtype = _hf_dtype_to_jnp(vis["dtype"]) if vis.get("dtype") is not None else text_dtype
    is_moe = model_type == "qwen3_vl_moe"

    return Qwen3VLConfig(
        num_layers=_required(txt, "num_hidden_layers", "hf_cfg['text_config']"),
        vocab_size=_required(txt, "vocab_size", "hf_cfg['text_config']"),
        emb_dim=_required(txt, "hidden_size", "hf_cfg['text_config']"),
        mlp_dim=_required(txt, "intermediate_size", "hf_cfg['text_config']"),
        num_heads=_required(txt, "num_attention_heads", "hf_cfg['text_config']"),
        head_dim=_required(txt, "head_dim", "hf_cfg['text_config']"),
        num_kv_heads=_required(txt, "num_key_value_heads", "hf_cfg['text_config']"),
        rope_theta=rope_theta,
        norm_eps=_required(txt, "rms_norm_eps", "hf_cfg['text_config']"),
        tie_word_embeddings=_required(hf_cfg, "tie_word_embeddings", "hf_cfg"),
        mrope_section=tuple(mrope_section),
        moe_intermediate_size=_required(txt, "moe_intermediate_size", "hf_cfg['text_config']") if is_moe else 0,
        num_experts=_required(txt, "num_experts", "hf_cfg['text_config']") if is_moe else 0,
        num_experts_per_tok=_required(txt, "num_experts_per_tok", "hf_cfg['text_config']") if is_moe else 0,
        mlp_only_layers=tuple(_required(txt, "mlp_only_layers", "hf_cfg['text_config']")) if is_moe else (),
        decoder_sparse_step=_required(txt, "decoder_sparse_step", "hf_cfg['text_config']") if is_moe else 1,
        norm_topk_prob=_required(txt, "norm_topk_prob", "hf_cfg['text_config']") if is_moe else True,
        image_token_id=_required(hf_cfg, "image_token_id", "hf_cfg"),
        video_token_id=_required(hf_cfg, "video_token_id", "hf_cfg"),
        vision_start_token_id=_required(hf_cfg, "vision_start_token_id", "hf_cfg"),
        dtype=text_dtype,
        vision=Qwen3VLVisionConfig(
            hidden_size=_required(vis, "hidden_size", "hf_cfg['vision_config']"),
            intermediate_size=_required(vis, "intermediate_size", "hf_cfg['vision_config']"),
            num_heads=_required(vis, "num_heads", "hf_cfg['vision_config']"),
            patch_size=_required(vis, "patch_size", "hf_cfg['vision_config']"),
            temporal_patch_size=_required(vis, "temporal_patch_size", "hf_cfg['vision_config']"),
            in_channels=_required(vis, "in_channels", "hf_cfg['vision_config']"),
            spatial_merge_size=_required(vis, "spatial_merge_size", "hf_cfg['vision_config']"),
            out_hidden_size=_required(vis, "out_hidden_size", "hf_cfg['vision_config']"),
            depth=_required(vis, "depth", "hf_cfg['vision_config']"),
            hidden_act=_required(vis, "hidden_act", "hf_cfg['vision_config']"),
            num_position_embeddings=_required(vis, "num_position_embeddings", "hf_cfg['vision_config']"),
            deepstack_visual_indexes=tuple(_required(vis, "deepstack_visual_indexes", "hf_cfg['vision_config']")),
            dtype=vision_dtype,
        ),
    )
