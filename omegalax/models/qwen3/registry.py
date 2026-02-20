"""Registry for Qwen3 variants (dense + MoE)."""

from typing import Callable, Type

from flax import nnx

from .config import Qwen3Config
from .dense.config import get_dense_spec, list_qwen3_dense_model_ids, make_dense_config
from .dense.model import Qwen3Dense
from .moe.config import Qwen3MoeConfig, get_moe_spec, list_qwen3_moe_model_ids, make_moe_config
from .moe.model import Qwen3Moe


_VARIANTS: dict[str, tuple[Callable[..., Qwen3Config], Callable[..., nnx.Module]]] = {
    "dense": (make_dense_config, Qwen3Dense),
    "moe": (make_moe_config, Qwen3Moe),
}


def build_config(model_id: str, *, variant: str | None = None, use_sharding: bool = False) -> Qwen3Config:
    if variant is None:
        variant = infer_variant(model_id)
    builder, _ = _VARIANTS.get(variant, (None, None))
    if builder is None:
        raise ValueError(f"Unknown Qwen3 variant '{variant}'")
    return builder(model_id, use_sharding=use_sharding)


def get_model_cls(variant: str) -> Type[nnx.Module]:
    _, cls = _VARIANTS.get(variant, (None, None))
    if cls is None:
        raise ValueError(f"Unknown Qwen3 variant '{variant}'")
    return cls


def infer_variant(model_id: str) -> str:
    key = model_id.lower()
    if "moe" in key:
        return "moe"
    # Qwen3 MoE models use the AXB naming convention (e.g. 30B-A3B)
    import re
    if re.search(r"\d+b-a\d+b", key):
        return "moe"
    return "dense"


__all__ = [
    "Qwen3Config",
    "Qwen3MoeConfig",
    "build_config",
    "get_model_cls",
    "infer_variant",
    "list_qwen3_dense_model_ids",
    "list_qwen3_moe_model_ids",
    "Qwen3Dense",
    "Qwen3Moe",
]
