import dataclasses

from ..config import Qwen3Config

_QWEN3_DENSE_SPECS: dict[str, dict[str, int | float | bool | str | None]] = {
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


_MODEL_ID_TO_SPEC: dict[str, str] = {}
for _spec_key, _spec in _QWEN3_DENSE_SPECS.items():
    _MODEL_ID_TO_SPEC[_spec_key] = _spec_key
    _hf_id = _spec.get("hf_repo_id")
    if _hf_id:
        _MODEL_ID_TO_SPEC[_hf_id] = _spec_key


def list_qwen3_dense_model_ids() -> list[str]:
    return [spec["hf_repo_id"] for spec in _QWEN3_DENSE_SPECS.values() if spec.get("hf_repo_id")]


def get_dense_spec(model_id: str) -> dict[str, int | float | bool | str | None]:
    spec_key = _MODEL_ID_TO_SPEC.get(model_id)
    if spec_key:
        return dict(_QWEN3_DENSE_SPECS[spec_key])
    supported = sorted(_MODEL_ID_TO_SPEC.keys())
    raise ValueError(f"Unsupported Qwen3 dense model_id '{model_id}'. Supported ids: {supported}")


def is_supported_dense_model_id(model_id: str) -> bool:
    return model_id in _MODEL_ID_TO_SPEC


def make_dense_config(model_id: str, use_sharding: bool = False) -> Qwen3Config:
    spec = get_dense_spec(model_id)
    return Qwen3Config.with_sharding(
        use_sharding,
        variant="dense",
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
    )
