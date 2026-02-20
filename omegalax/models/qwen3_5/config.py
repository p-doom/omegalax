"""Configuration for Qwen3.5 vision-language model."""

import dataclasses


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
    hidden_act: str = "gelu"
    norm_eps: float = 1e-6


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3_5TextConfig:
    vocab_size: int = 248_320
    hidden_size: int = 4096
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    layer_types: tuple[str, ...] = ()
    rope_theta: float = 10_000_000
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, ...] = (11, 11, 10)
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


def _canonicalize_key(model_id: str) -> str:
    key = model_id.lower().split("/")[-1].replace("_", "-")
    # FIXME (f.srambical)
    for suffix in ("-instruct", "-chat", "-base", "-thinking"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def list_qwen3_5_model_ids() -> list[str]:
    return [s["hf_repo_id"] for s in _QWEN3_5_SPECS.values() if s.get("hf_repo_id")]


def get_qwen3_5_spec(model_id: str) -> dict:
    key = _canonicalize_key(model_id)
    spec = _QWEN3_5_SPECS.get(key)
    if spec is None:
        for s in _QWEN3_5_SPECS.values():
            if s.get("hf_repo_id") and s["hf_repo_id"].lower() == model_id.lower():
                spec = s
                break
    if spec is None:
        raise ValueError(f"Unsupported Qwen3.5 model_id '{model_id}'")
    return dict(spec)


def make_config(model_id: str) -> Qwen3_5Config:
    spec = get_qwen3_5_spec(model_id)
    vis_kw = spec.get("vision_config", {})
    txt_kw = spec.get("text_config", {})
    return Qwen3_5Config(
        vision_config=Qwen3_5VisionConfig(**vis_kw),
        text_config=Qwen3_5TextConfig(**txt_kw),
    )
