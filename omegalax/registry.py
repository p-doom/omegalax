"""Lightweight registry for routing model ids to architectures."""

from __future__ import annotations

from enum import Enum


class Arch(str, Enum):
    TEXT = "text"
    VLM = "vlm"


from omegalax.models.qwen3.config import get_spec as _get_qwen3_spec
from omegalax.models.qwen3_vl.config import get_vl_spec as _get_vl_spec
from omegalax.models.qwen3_5.config import get_qwen3_5_spec as _get_qwen3_5_spec


_ARCH_RESOLVERS: tuple[tuple[Arch, tuple], ...] = (
    (Arch.TEXT, (_get_qwen3_spec,)),
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


def resolve_hf_repo_id(model_id: str) -> str:
    """Map a short spec key or HF repo id to the canonical HuggingFace repo id.

    Returns the input unchanged when it already contains '/' (i.e. is already
    a full HF repo id).  For short keys like ``qwen3-vl-2b`` or ``qwen3-0-6b``
    this looks up the ``hf_repo_id`` in the corresponding spec registry.
    """
    if "/" in model_id:
        return model_id
    for _arch, resolvers in _ARCH_RESOLVERS:
        for resolver in resolvers:
            try:
                spec = resolver(model_id)
                hf_id = spec.get("hf_repo_id") if isinstance(spec, dict) else None
                if hf_id:
                    return hf_id
            except ValueError:
                continue
    return model_id


def resolve(model_id: str) -> Arch:
    """Return the architecture classification for a given model id."""
    return infer_arch(model_id)
