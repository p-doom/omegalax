"""Lightweight registry for routing model ids to architectures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Arch(str, Enum):
    TEXT = "text"
    VLM = "vlm"


from omegalax.models.qwen3.dense.config import get_dense_spec as _get_dense_spec
from omegalax.models.qwen3.moe.config import get_moe_spec as _get_moe_spec
from omegalax.models.qwen3_vl.config import get_vl_spec as _get_vl_spec
from omegalax.models.qwen3_5.config import get_qwen3_5_spec as _get_qwen3_5_spec


_ARCH_RESOLVERS: tuple[tuple[Arch, tuple], ...] = (
    (Arch.TEXT, (_get_dense_spec, _get_moe_spec)),
    (Arch.VLM, (_get_vl_spec, _get_qwen3_5_spec)),
)


def _matches_any(resolvers, model_id: str) -> bool:
    for resolver in resolvers:
        try:
            resolver(model_id)
            return True
        except ValueError:
            continue
    return False


def infer_arch(model_id: str) -> Arch:
    """Determine the architecture via exact resolution against known registries."""
    for arch, resolvers in _ARCH_RESOLVERS:
        if _matches_any(resolvers, model_id):
            return arch
    raise ValueError(f"Cannot infer architecture for model id '{model_id}'")


@dataclass(frozen=True)
class ModelArch:
    arch: Arch


def resolve(model_id: str) -> ModelArch:
    """Return the architecture classification for a given model id."""
    return ModelArch(arch=infer_arch(model_id))
