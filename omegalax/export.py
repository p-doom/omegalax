"""Unified HuggingFace export dispatcher."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from omegalax.models.params_utils import flatten_pure_state
from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.qwen3.params import export_qwen3_to_safetensors, qwen3_to_hf_config_dict
from omegalax.models.qwen3_vl.config import Qwen3VLConfig
from omegalax.models.qwen3_vl.model import Qwen3VL
from omegalax.models.qwen3_vl.params import (
    export_qwen3_vl_to_safetensors,
    qwen3_vl_to_hf_config_dict,
)
from omegalax.models.qwen3_5.config import Qwen3_5Config
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors, qwen3_5_to_hf_config_dict
from omegalax.trainers.lora import merge_lora_into_base


def read_lora_metadata(save_dir: Path) -> dict:
    """Return the LoRA settings persisted next to an Orbax checkpoint tree."""
    import json

    path = save_dir / "lora_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no lora_metadata.json next to the checkpoint at {save_dir}. Every "
            "checkpoint written by omegalax.trainers.vlm has one; without it an "
            "adapter checkpoint exports as the base model with no error. Write the "
            "file from the training run's recipe (enable_lora, lora_rank, lora_alpha)."
        )
    metadata = json.loads(path.read_text())
    missing = {"enable_lora", "lora_rank", "lora_alpha"} - metadata.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}; refusing to guess a LoRA rank")
    return metadata


def export_model_to_hf(model, cfg, out_dir: str | Path) -> Path:
    """Route to the correct exporter based on model/config type.

    If the model contains any ``LoRALinear`` wrappers (from LoRA-enabled
    training), they are merged into the base linears in-place before
    dispatch. The post-merge model is structurally identical to a
    full-FT model with the same effective weights, so the existing
    model-specific exporters consume it unchanged. Downstream serving
    (sglang) sees plain dense kernels and needs no adapter awareness.
    """
    n_merged = merge_lora_into_base(model)
    if n_merged > 0:
        print(f"[export] merged {n_merged} LoRA adapters into base before HF export")

    if isinstance(cfg, Qwen3Config) and isinstance(model, Qwen3):
        return export_qwen3_to_safetensors(model, cfg, out_dir)

    if isinstance(cfg, Qwen3VLConfig) and isinstance(model, Qwen3VL):
        return export_qwen3_vl_to_safetensors(model, cfg, out_dir)

    if isinstance(cfg, Qwen3_5Config) and isinstance(model, Qwen3_5ForConditionalGeneration):
        return export_qwen3_5_to_safetensors(model, cfg, out_dir)

    raise ValueError(
        f"Unsupported model/config combination for export: {type(model)} / {type(cfg)}"
    )


def param_fingerprint(model) -> dict[str, int]:
    """Per-leaf checksum for comparing a model against the base it came from.

    Device-side reduction, so it costs one pass and no host transfer of the
    weights. Compared before-restore against after-merge it answers the only
    question the exporter cannot answer any other way: did the checkpoint
    actually change the weights we are about to write? ``partial_restore=True``
    drops leaves it cannot path-match without raising, and a LoRA adapter that
    never got merged leaves the base kernels untouched -- both produce a
    base-identical export.
    """

    def checksum(leaf) -> int:
        # Mixing bits before reduction prevents equal-and-opposite updates from cancelling.
        bits = jax.lax.bitcast_convert_type(leaf.astype(jnp.float32), jnp.uint32)
        bits = bits ^ (bits >> jnp.uint32(16))
        bits *= jnp.uint32(0x7FEB352D)
        bits ^= bits >> jnp.uint32(15)
        return int(jnp.sum(bits, dtype=jnp.uint32))

    _, state = nnx.split(model)
    return {
        key: checksum(leaf) for key, leaf in flatten_pure_state(nnx.to_pure_dict(state)).items()
    }


_HF_ARCHITECTURES = {
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
    "qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
}


def model_config_to_hf_dict(cfg) -> dict:
    """Serialize a runtime config to HF config.json format."""
    if isinstance(cfg, Qwen3Config):
        hf_cfg = qwen3_to_hf_config_dict(cfg)
    elif isinstance(cfg, Qwen3VLConfig):
        hf_cfg = qwen3_vl_to_hf_config_dict(cfg)
    elif isinstance(cfg, Qwen3_5Config):
        hf_cfg = qwen3_5_to_hf_config_dict(cfg)
    else:
        raise ValueError(f"Unsupported config type for HF serialization: {type(cfg)}")

    model_type = hf_cfg["model_type"]
    if model_type not in _HF_ARCHITECTURES:
        raise ValueError(
            f"No HF architecture registered for model_type {model_type!r}. Add it to "
            f"_HF_ARCHITECTURES; an export without `architectures` cannot be served."
        )
    hf_cfg["architectures"] = [_HF_ARCHITECTURES[model_type]]
    return hf_cfg
