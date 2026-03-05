"""Unified Qwen3 configuration (dense + MoE)."""

import dataclasses
from typing import Any

import jax.numpy as jnp
from omegalax.models.shard_config import ShardConfig


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Configuration for Qwen3 models (dense and MoE variants)."""

    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    rope_scaling_factor: float | None
    local_rope_theta: float | None
    norm_eps: float
    tie_word_embeddings: bool

    # MoE fields (num_experts == 0 means dense)
    moe_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    mlp_only_layers: tuple[int, ...] = dataclasses.field(default_factory=tuple)
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = True
    aux_loss_coef: float = 0.0

    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.default)
    dtype: Any = jnp.bfloat16

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def variant(self) -> str:
        return "moe" if self.is_moe else "dense"

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.is_moe
            and layer_idx not in self.mlp_only_layers
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )

    @classmethod
    def with_sharding(cls, **kwargs):
        kwargs.pop("variant", None)
        kwargs["shd_cfg"] = ShardConfig.default()
        return cls(**kwargs)


_QWEN3_DENSE_SPECS: dict[str, dict] = {
    "qwen3-smoke": {
        "hf_repo_id": None,
        "vocab_size": 1024,
        "emb_dim": 128,
        "mlp_dim": 512,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 32,
        "num_kv_heads": 4,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": False,
    },
    "qwen3-0.6b": {
        "hf_repo_id": "Qwen/Qwen3-0.6B",
        "vocab_size": 151_936,
        "emb_dim": 1024,
        "mlp_dim": 3_072,
        "num_layers": 28,
        "num_heads": 16,
        "head_dim": 128,
        "num_kv_heads": 8,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": True,
    },
    "qwen3-1.7b": {
        "hf_repo_id": "Qwen/Qwen3-1.7B",
        "vocab_size": 151_936,
        "emb_dim": 2_048,
        "mlp_dim": 6_144,
        "num_layers": 28,
        "num_heads": 16,
        "head_dim": 128,
        "num_kv_heads": 8,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": True,
    },
    "qwen3-4b": {
        "hf_repo_id": "Qwen/Qwen3-4B",
        "vocab_size": 151_936,
        "emb_dim": 2_560,
        "mlp_dim": 9_728,
        "num_layers": 36,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 8,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": True,
    },
    "qwen3-8b": {
        "hf_repo_id": "Qwen/Qwen3-8B",
        "vocab_size": 151_936,
        "emb_dim": 4_096,
        "mlp_dim": 12_288,
        "num_layers": 36,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 8,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": False,
    },
    "qwen3-14b": {
        "hf_repo_id": "Qwen/Qwen3-14B",
        "vocab_size": 151_936,
        "emb_dim": 5_120,
        "mlp_dim": 17_408,
        "num_layers": 40,
        "num_heads": 40,
        "head_dim": 128,
        "num_kv_heads": 8,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": False,
    },
    "qwen3-32b": {
        "hf_repo_id": "Qwen/Qwen3-32B",
        "vocab_size": 151_936,
        "emb_dim": 5_120,
        "mlp_dim": 25_600,
        "num_layers": 64,
        "num_heads": 64,
        "head_dim": 128,
        "num_kv_heads": 8,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": False,
    },
}

_QWEN3_MOE_SPECS: dict[str, dict] = {
    "qwen3-smoke-moe": {
        "hf_repo_id": None,
        "vocab_size": 512,
        "emb_dim": 128,
        "mlp_dim": 256,
        "moe_intermediate_size": 256,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 32,
        "num_kv_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "mlp_only_layers": (),
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": False,
    },
    "qwen3-30b-a3b": {
        "hf_repo_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "vocab_size": 151_936,
        "emb_dim": 2_048,
        "mlp_dim": 6_144,
        "moe_intermediate_size": 768,
        "num_layers": 48,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 4,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "mlp_only_layers": (),
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "rope_theta": 1_000_000,
        "rope_scaling_factor": None,
        "local_rope_theta": None,
        "tie_word_embeddings": False,
    },
}

_ALL_SPECS: dict[str, dict] = {**_QWEN3_DENSE_SPECS, **_QWEN3_MOE_SPECS}

_MODEL_ID_TO_SPEC: dict[str, str] = {}
for _spec_key, _spec in _ALL_SPECS.items():
    _MODEL_ID_TO_SPEC[_spec_key] = _spec_key
    _hf_id = _spec.get("hf_repo_id")
    if _hf_id:
        _MODEL_ID_TO_SPEC[_hf_id] = _spec_key


def list_qwen3_dense_model_ids() -> list[str]:
    return [s["hf_repo_id"] for s in _QWEN3_DENSE_SPECS.values() if s.get("hf_repo_id")]


def list_qwen3_moe_model_ids() -> list[str]:
    return [s["hf_repo_id"] for s in _QWEN3_MOE_SPECS.values() if s.get("hf_repo_id")]


def get_spec(model_id: str) -> dict:
    spec_key = _MODEL_ID_TO_SPEC.get(model_id)
    if spec_key:
        return dict(_ALL_SPECS[spec_key])
    supported = sorted(_MODEL_ID_TO_SPEC.keys())
    raise ValueError(f"Unsupported Qwen3 model_id '{model_id}'. Supported ids: {supported}")


def is_supported_model_id(model_id: str) -> bool:
    return model_id in _MODEL_ID_TO_SPEC


def make_config(model_id: str) -> Qwen3Config:
    """Build a Qwen3Config from a spec key or HF repo id."""
    spec = get_spec(model_id)
    kw = {k: v for k, v in spec.items() if k != "hf_repo_id"}
    if "mlp_only_layers" in kw:
        kw["mlp_only_layers"] = tuple(kw["mlp_only_layers"])
    kw["norm_eps"] = kw.pop("norm_eps", 1e-6)
    return Qwen3Config.with_sharding(**kw)
