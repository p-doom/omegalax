"""Configuration for Qwen3-VL models."""

from __future__ import annotations

import dataclasses
from typing import Any


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


_QWEN3_VL_SPECS: dict[str, dict[str, Any]] = {
    "qwen3-vl-smoke": {
        "hf_repo_id": None,
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
    },
}


_MODEL_ID_TO_SPEC: dict[str, str] = {}
for _spec_key, _spec in _QWEN3_VL_SPECS.items():
    _MODEL_ID_TO_SPEC[_spec_key] = _spec_key
    _hf_id = _spec.get("hf_repo_id")
    if _hf_id:
        _MODEL_ID_TO_SPEC[_hf_id] = _spec_key


def list_qwen3_vl_model_ids() -> list[str]:
    return [spec["hf_repo_id"] for spec in _QWEN3_VL_SPECS.values() if spec.get("hf_repo_id")]


def get_vl_spec(model_id: str) -> dict[str, Any]:
    spec_key = _MODEL_ID_TO_SPEC.get(model_id)
    if spec_key:
        return dict(_QWEN3_VL_SPECS[spec_key])
    supported = sorted(_MODEL_ID_TO_SPEC.keys())
    raise ValueError(f"Unsupported Qwen3-VL model_id '{model_id}'. Supported ids: {supported}")


def make_vl_config(model_id: str) -> Qwen3VLConfig:
    spec = get_vl_spec(model_id)
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


def make_vl_config_from_hf(hf_cfg: dict[str, Any]) -> Qwen3VLConfig:
    """Build a Qwen3VLConfig from a HuggingFace config.json dict."""
    vis = hf_cfg.get("vision_config", {})
    txt = hf_cfg.get("text_config", {})

    rope_params = txt.get("rope_parameters", txt.get("rope_scaling", {})) or {}
    rope_theta = rope_params.get("rope_theta", txt.get("rope_theta", 1_000_000))
    mrope_section = tuple(rope_params.get("mrope_section", [24, 20, 20]))

    hf_head_dim = txt.get("head_dim")
    if hf_head_dim is None:
        hf_head_dim = txt["hidden_size"] // txt["num_attention_heads"]

    return Qwen3VLConfig(
        num_layers=txt["num_hidden_layers"],
        vocab_size=txt["vocab_size"],
        emb_dim=txt["hidden_size"],
        mlp_dim=txt["intermediate_size"],
        num_heads=txt["num_attention_heads"],
        head_dim=hf_head_dim,
        num_kv_heads=txt.get("num_key_value_heads", txt["num_attention_heads"]),
        rope_theta=rope_theta,
        norm_eps=txt.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=txt.get("tie_word_embeddings", False),
        mrope_section=mrope_section,
        image_token_id=hf_cfg.get("image_token_id", 151655),
        video_token_id=hf_cfg.get("video_token_id", 151656),
        vision_start_token_id=hf_cfg.get("vision_start_token_id", 151652),
        vision=Qwen3VLVisionConfig(
            hidden_size=vis["hidden_size"],
            intermediate_size=vis["intermediate_size"],
            num_heads=vis.get("num_heads", vis.get("num_attention_heads", 16)),
            patch_size=vis.get("patch_size", 14),
            temporal_patch_size=vis.get("temporal_patch_size", 2),
            in_channels=vis.get("in_channels", 3),
            spatial_merge_size=vis.get("spatial_merge_size", 2),
            out_hidden_size=vis.get("out_hidden_size", txt["hidden_size"]),
            depth=vis.get("depth", 32),
            hidden_act=vis.get("hidden_act", "gelu_pytorch_tanh"),
            num_position_embeddings=vis.get("num_position_embeddings", 4096),
            deepstack_visual_indexes=tuple(vis.get("deepstack_visual_indexes", [])),
        ),
    )
