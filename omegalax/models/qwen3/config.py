"""Unified Qwen3 configuration (dense + MoE)."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from etils import epath

from omegalax.models.params_utils import load_hf_config_from_source
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


_QWEN3_SMOKE_SPECS: dict[str, dict[str, Any]] = {
    "qwen3-smoke": {
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
        "norm_eps": 1e-6,
        "tie_word_embeddings": False,
    },
    "qwen3-smoke-moe": {
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
        "norm_eps": 1e-6,
        "tie_word_embeddings": False,
    },
}

_QWEN3_DENSE_REPOS = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
)

_QWEN3_MOE_REPOS = (
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
)

_REAL_MODEL_IDS = (*_QWEN3_DENSE_REPOS, *_QWEN3_MOE_REPOS)
_SUPPORTED_MODEL_IDS = sorted((*_QWEN3_SMOKE_SPECS.keys(), *_REAL_MODEL_IDS))
_SUPPORTED_MODEL_TYPES = {"qwen3", "qwen3_moe"}


def _required(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {where}.")
    return mapping[key]


def _required_any(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    names = " or ".join(repr(key) for key in keys)
    raise ValueError(f"Missing required key {names} in {where}.")


def _hf_dtype_to_jnp(hf_dtype: str | None) -> Any:
    if hf_dtype is None:
        return jnp.bfloat16
    kind = str(hf_dtype).lower()
    if "bfloat16" in kind or "bf16" in kind:
        return jnp.bfloat16
    if "float32" in kind or "fp32" in kind:
        return jnp.float32
    if "float16" in kind or "fp16" in kind:
        return jnp.float16
    raise ValueError(f"Unsupported dtype '{hf_dtype}'.")


def list_qwen3_dense_model_ids() -> list[str]:
    return list(_QWEN3_DENSE_REPOS)


def list_qwen3_moe_model_ids() -> list[str]:
    return list(_QWEN3_MOE_REPOS)


def resolve_qwen3_repo_id(model_id: str) -> str:
    return model_id


def get_spec(model_id: str) -> dict[str, Any]:
    if model_id in _QWEN3_SMOKE_SPECS:
        return dict(_QWEN3_SMOKE_SPECS[model_id])
    if model_id in _REAL_MODEL_IDS:
        return {"hf_repo_id": model_id}
    raise ValueError(f"Unsupported Qwen3 model_id '{model_id}'. Supported ids: {_SUPPORTED_MODEL_IDS}")


def is_supported_model_id(model_id: str) -> bool:
    return model_id in _QWEN3_SMOKE_SPECS or model_id in _REAL_MODEL_IDS


def make_config_from_hf(hf_cfg: dict[str, Any]) -> Qwen3Config:
    """Build a Qwen3Config from a HuggingFace config.json dict."""
    model_type = _required(hf_cfg, "model_type", "hf_cfg")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported Qwen3 model_type '{model_type}'. Expected one of {sorted(_SUPPORTED_MODEL_TYPES)}."
        )

    rope_params = hf_cfg.get("rope_parameters")
    if rope_params is None:
        rope_theta = _required(hf_cfg, "rope_theta", "hf_cfg")
        rope_scaling_factor = None
        local_rope_theta = None
    else:
        if not isinstance(rope_params, dict):
            raise ValueError("Expected rope_parameters to be a dict in hf_cfg.")
        rope_theta = _required(rope_params, "rope_theta", "hf_cfg['rope_parameters']")
        rope_type = rope_params.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"Unsupported rope_parameters.rope_type '{rope_type}' for Qwen3.")
        rope_scaling_factor = rope_params.get("factor")
        local_rope_theta = rope_params.get("local_rope_theta")

    dtype = _hf_dtype_to_jnp(hf_cfg.get("dtype"))
    is_moe = model_type == "qwen3_moe"

    cfg = Qwen3Config(
        num_layers=_required(hf_cfg, "num_hidden_layers", "hf_cfg"),
        vocab_size=_required(hf_cfg, "vocab_size", "hf_cfg"),
        emb_dim=_required(hf_cfg, "hidden_size", "hf_cfg"),
        mlp_dim=_required(hf_cfg, "intermediate_size", "hf_cfg"),
        num_heads=_required(hf_cfg, "num_attention_heads", "hf_cfg"),
        head_dim=_required(hf_cfg, "head_dim", "hf_cfg"),
        num_kv_heads=_required(hf_cfg, "num_key_value_heads", "hf_cfg"),
        rope_theta=rope_theta,
        rope_scaling_factor=rope_scaling_factor,
        local_rope_theta=local_rope_theta,
        norm_eps=_required(hf_cfg, "rms_norm_eps", "hf_cfg"),
        tie_word_embeddings=_required(hf_cfg, "tie_word_embeddings", "hf_cfg"),
        moe_intermediate_size=_required(hf_cfg, "moe_intermediate_size", "hf_cfg") if is_moe else 0,
        num_experts=_required_any(hf_cfg, ("num_experts", "num_local_experts"), "hf_cfg")
        if is_moe
        else int(hf_cfg.get("num_experts", hf_cfg.get("num_local_experts", 0))),
        num_experts_per_tok=_required(hf_cfg, "num_experts_per_tok", "hf_cfg") if is_moe else 0,
        mlp_only_layers=tuple(_required(hf_cfg, "mlp_only_layers", "hf_cfg")) if is_moe else (),
        decoder_sparse_step=_required(hf_cfg, "decoder_sparse_step", "hf_cfg") if is_moe else 1,
        norm_topk_prob=_required(hf_cfg, "norm_topk_prob", "hf_cfg") if is_moe else True,
        aux_loss_coef=float(hf_cfg.get("router_aux_loss_coef", 0.0)),
        dtype=dtype,
    )
    return dataclasses.replace(cfg, shd_cfg=ShardConfig.default())


def make_config(model_id: str) -> Qwen3Config:
    """Build a Qwen3Config from a smoke preset, local model dir, or HF repo id."""
    if model_id in _QWEN3_SMOKE_SPECS:
        kw = dict(_QWEN3_SMOKE_SPECS[model_id])
        if "mlp_only_layers" in kw:
            kw["mlp_only_layers"] = tuple(kw["mlp_only_layers"])
        return Qwen3Config.with_sharding(**kw)

    if "/" not in model_id and not epath.Path(model_id).expanduser().exists():
        raise ValueError(f"Unsupported Qwen3 model_id '{model_id}'. Supported ids: {_SUPPORTED_MODEL_IDS}")

    hf_cfg = load_hf_config_from_source(resolve_qwen3_repo_id(model_id))
    return make_config_from_hf(hf_cfg)
