import dataclasses

import jax.numpy as jnp

from ..config import Qwen3Config, ShardConfig


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3MoeConfig(Qwen3Config):
    moe_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    mlp_only_layers: tuple[int, ...] = dataclasses.field(default_factory=tuple)
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = True
    aux_loss_coef: float = 0.0

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            layer_idx not in self.mlp_only_layers
            and self.num_experts > 0
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )


_QWEN3_MOE_SPECS: dict[str, dict[str, int | float | bool | str | None]] = {
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


def list_qwen3_moe_model_ids() -> list[str]:
    return [spec["hf_repo_id"] for spec in _QWEN3_MOE_SPECS.values() if spec.get("hf_repo_id")]


def get_moe_spec(model_id: str) -> dict[str, int | float | bool | str | None]:
    key = model_id.lower().split("/")[-1].replace("_", "-")
    spec = _QWEN3_MOE_SPECS.get(key)
    if spec is None:
        for s in _QWEN3_MOE_SPECS.values():
            if s.get("hf_repo_id") and s["hf_repo_id"].lower() == model_id.lower():
                spec = s
                break
    if spec is None:
        raise ValueError(f"Unsupported Qwen3 MoE model_id '{model_id}'")
    return dict(spec)


def make_moe_config(model_id: str, use_sharding: bool = False) -> Qwen3MoeConfig:
    spec = get_moe_spec(model_id)
    return Qwen3MoeConfig.with_sharding(
        use_sharding,
        variant="moe",
        num_layers=int(spec["num_layers"]),
        vocab_size=int(spec["vocab_size"]),
        emb_dim=int(spec["emb_dim"]),
        mlp_dim=int(spec["mlp_dim"]),
        num_heads=int(spec["num_heads"]),
        head_dim=int(spec["head_dim"]),
        num_kv_heads=int(spec["num_kv_heads"]),
        rope_theta=int(spec["rope_theta"]),
        rope_scaling_factor=spec["rope_scaling_factor"],
        local_rope_theta=spec["local_rope_theta"],
        norm_eps=1e-6,
        tie_word_embeddings=bool(spec["tie_word_embeddings"]),
        moe_intermediate_size=int(spec["moe_intermediate_size"]),
        num_experts=int(spec["num_experts"]),
        num_experts_per_tok=int(spec["num_experts_per_tok"]),
        mlp_only_layers=tuple(spec.get("mlp_only_layers", ())),
        decoder_sparse_step=int(spec.get("decoder_sparse_step", 1)),
        norm_topk_prob=bool(spec.get("norm_topk_prob", True)),
    )
