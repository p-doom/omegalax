"""Architecture-specific API for vision-language models."""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec

from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.qwen3_5 import Qwen3_5Config
from omegalax.models.qwen3_5.config import (
    list_supported_qwen3_5_model_ids,
)
from omegalax.models.qwen3_5.config import (
    make_config_from_hf as make_qwen3_5_config_from_hf,
)
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_vl import Qwen3VL
from omegalax.models.qwen3_vl.config import (
    Qwen3VLConfig,
    list_supported_qwen3_vl_model_ids,
    make_vl_config_from_hf,
)
from omegalax.models.qwen3_vl.loader import (
    create_qwen3_vl_from_safetensor_files,
    validate_dense_qwen3_vl_safetensors,
)
from omegalax.models.shard_config import axis_rules_for_mesh, shard_config_for_mesh
from omegalax.models.sharding_runtime import (
    batch_partition_spec as runtime_batch_partition_spec,
)
from omegalax.models.sharding_runtime import (
    init_model_sharded,
)
from omegalax.models.sharding_runtime import (
    shard_batch as runtime_shard_batch,
)
from omegalax.models.sharding_runtime import (
    shard_batch_dict as runtime_shard_batch_dict,
)
from omegalax.vlm.local_snapshot import LocalVLMSnapshot

VLMConfig = Qwen3_5Config | Qwen3VLConfig


def resolve_config(model_snapshot: LocalVLMSnapshot) -> VLMConfig:
    """Resolve the exact config committed by a sealed pretrained snapshot."""
    if type(model_snapshot) is not LocalVLMSnapshot:
        raise TypeError("resolve_config requires an open LocalVLMSnapshot")
    with (
        model_snapshot.files() as files,
        open(files["config.json"], encoding="utf-8") as stream,
    ):
        hf_cfg = json.load(stream)
    return _config_from_hf(hf_cfg, str(model_snapshot.path))


def validate_pretrained(model_snapshot: LocalVLMSnapshot) -> tuple[Qwen3VLConfig, int]:
    if type(model_snapshot) is not LocalVLMSnapshot:
        raise TypeError("validate_pretrained requires an open LocalVLMSnapshot")
    with model_snapshot.files() as files:
        with open(files["config.json"], encoding="utf-8") as stream:
            hf_cfg = json.load(stream)
        config = _config_from_hf(hf_cfg, str(model_snapshot.path))
        if type(config) is not Qwen3VLConfig or config.num_experts != 0:
            raise NotImplementedError(
                "Sealed-snapshot production loading currently supports dense Qwen3-VL only; "
                "Qwen3.5 and Qwen3-VL MoE remain release blockers"
            )
        weight_files = [
            files[name] for name in model_snapshot.names if name.endswith(".safetensors")
        ]
        return validate_dense_qwen3_vl_safetensors(weight_files, hf_cfg)


def _config_from_hf(hf_cfg: dict, source: str) -> VLMConfig:
    model_type = hf_cfg.get("model_type")
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        return make_qwen3_5_config_from_hf(hf_cfg)
    if model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        return make_vl_config_from_hf(hf_cfg)

    raise ValueError(
        f"Unsupported VLM model/config source '{source}'. "
        f"Supported Qwen3.5 ids: {list_supported_qwen3_5_model_ids()}; "
        f"supported Qwen3-VL ids: {list_supported_qwen3_vl_model_ids()}."
    )


def align_config_to_mesh(cfg: VLMConfig, mesh: Mesh) -> VLMConfig:
    """Drop singleton mesh axes from sharding specs to avoid degenerate constraints."""
    if isinstance(cfg, Qwen3_5Config):
        return dataclasses.replace(
            cfg,
            text_config=dataclasses.replace(
                cfg.text_config,
                shd_cfg=shard_config_for_mesh(cfg.text_config.shd_cfg, mesh),
            ),
        )
    if isinstance(cfg, Qwen3VLConfig):
        return dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def batch_partition_spec(cfg: VLMConfig) -> PartitionSpec:
    if isinstance(cfg, Qwen3_5Config):
        return runtime_batch_partition_spec(cfg.text_config.shd_cfg)
    if isinstance(cfg, Qwen3VLConfig):
        return runtime_batch_partition_spec(cfg.shd_cfg)
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def shard_batch(token_ids_BT: jax.Array, cfg: VLMConfig, mesh: Mesh) -> jax.Array:
    if isinstance(cfg, Qwen3_5Config):
        return runtime_shard_batch(token_ids_BT, cfg.text_config.shd_cfg, mesh)
    if isinstance(cfg, Qwen3VLConfig):
        return runtime_shard_batch(token_ids_BT, cfg.shd_cfg, mesh)
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def shard_batch_dict(batch: dict, cfg: VLMConfig, mesh: Mesh) -> dict[str, jax.Array]:
    """Shard every array in a batch dict (batch dim sharded, rest replicated)."""
    if isinstance(cfg, Qwen3_5Config):
        return runtime_shard_batch_dict(batch, cfg.text_config.shd_cfg, mesh)
    if isinstance(cfg, Qwen3VLConfig):
        return runtime_shard_batch_dict(batch, cfg.shd_cfg, mesh)
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def vocab_size(cfg: VLMConfig) -> int:
    if isinstance(cfg, Qwen3_5Config):
        return int(cfg.text_config.vocab_size)
    if isinstance(cfg, Qwen3VLConfig):
        return int(cfg.vocab_size)
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def init_model(
    config: VLMConfig,
    rng: jax.Array,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> tuple[nnx.Module, VLMConfig]:
    """Initialize a vision-language model."""
    if type(config) not in {Qwen3_5Config, Qwen3VLConfig}:
        raise TypeError("init_model requires an exact VLMConfig")
    cfg = config
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    cfg = align_config_to_mesh(cfg, mesh)

    axis_rules = axis_rules_for_mesh(mesh)
    if isinstance(cfg, Qwen3_5Config):
        model = init_model_sharded(Qwen3_5ForConditionalGeneration, cfg, rng, mesh, axis_rules)
        return model, cfg
    if isinstance(cfg, Qwen3VLConfig):
        model = init_model_sharded(Qwen3VL, cfg, rng, mesh, axis_rules)
        return model, cfg
    raise ValueError(f"Unsupported VLM config type: {type(cfg)}")


def forward(
    model: nnx.Module,
    token_ids_BT: jax.Array,
    pad_id: int,
    cfg,
    *,
    attention_mask_BT: jax.Array | None = None,
    pixel_values: jax.Array | None = None,
    image_grid_thw: jax.Array | None = None,
    vision_cu_seqlens: jax.Array | None = None,
    position_ids_ZBT: jax.Array | None = None,
):
    """Forward pass returning hidden states before lm_head, plus aux loss."""
    if attention_mask_BT is None:
        attention_mask_BT = (token_ids_BT != pad_id).astype(jnp.int32)

    if isinstance(model, Qwen3_5ForConditionalGeneration):
        segment_ids_BT = attention_mask_BT.astype(jnp.int32)
        return model(
            token_ids_BT,
            segment_ids_BT,
            None,
            jnp.array(0, dtype=jnp.int32),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_cu_seqlens=vision_cu_seqlens,
            position_ids_ZBT=position_ids_ZBT,
        )

    if isinstance(model, Qwen3VL):
        return model(
            token_ids_BT,
            attention_mask_BT,
            position_ids_ZBT=position_ids_ZBT,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_cu_seqlens=vision_cu_seqlens,
        )

    raise ValueError(f"Unsupported VLM model type: {type(model)}")


def load_pretrained(
    model_snapshot: LocalVLMSnapshot,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> tuple[nnx.Module, VLMConfig]:
    """Load a pretrained VLM from a validated local snapshot."""
    if type(model_snapshot) is not LocalVLMSnapshot:
        raise TypeError("load_pretrained requires an open LocalVLMSnapshot")
    validate_pretrained(model_snapshot)
    ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    with model_snapshot.files() as files:
        with open(files["config.json"], encoding="utf-8") as stream:
            hf_cfg = json.load(stream)
        weight_files = [
            files[name] for name in model_snapshot.names if name.endswith(".safetensors")
        ]
        return create_qwen3_vl_from_safetensor_files(
            weight_files,
            hf_cfg,
            tp_size=tp_size,
            fsdp_size=fsdp_size,
            dp_size=dp_size,
        )


def make_cache(*_args, **_kwargs):
    """Placeholder for cache creation to keep the interface symmetric."""
    return None


def decode(*_args, **_kwargs):
    raise NotImplementedError("decode is not implemented for vision-language models.")
