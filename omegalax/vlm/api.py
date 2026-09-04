"""Architecture-specific API for vision-language models."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, PartitionSpec

from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.params_utils import load_hf_config_from_source
from omegalax.models.qwen3_5 import Qwen3_5Config
from omegalax.models.qwen3_5 import make_config as make_qwen3_5_config
from omegalax.models.qwen3_5.config import (
    is_supported_qwen3_5_model_id,
    list_supported_qwen3_5_model_ids,
)
from omegalax.models.qwen3_5.config import (
    make_config_from_hf as make_qwen3_5_config_from_hf,
)
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_vl import Qwen3VL, make_vl_config
from omegalax.models.qwen3_vl.config import (
    Qwen3VLConfig,
    is_supported_qwen3_vl_model_id,
    list_supported_qwen3_vl_model_ids,
    make_vl_config_from_hf,
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

VLMConfig = Qwen3_5Config | Qwen3VLConfig


def _validate_qwen3_5_process_local_batch(batch: dict, cfg: Qwen3_5Config, mesh: Mesh) -> None:
    required = {
        "token_ids_BT",
        "pixel_values",
        "vision_patch_valid",
        "image_grid_thw",
        "vision_cu_seqlens",
    }
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"Qwen3.5 vision batch is missing {sorted(missing)}")

    expected_axis = "fsdp" if mesh.shape["fsdp"] > 1 else None
    if (
        tuple(mesh.axis_names) != ("tp", "fsdp", "dp")
        or mesh.shape["tp"] != 1
        or mesh.shape["dp"] != 1
        or cfg.text_config.shd_cfg.act_btd[0] != expected_axis
        or jax.process_count() != mesh.shape["fsdp"]
        or jax.local_device_count() != 1
    ):
        raise ValueError(
            "Qwen3.5 vision batches require tp=1, dp=1, one local device per process, "
            "process_count=fsdp, and batch sharding on fsdp"
        )

    token_ids = np.asarray(batch["token_ids_BT"])
    pixel_values = np.asarray(batch["pixel_values"])
    patch_valid = np.asarray(batch["vision_patch_valid"])
    grid_thw = np.asarray(batch["image_grid_thw"])
    cu_seqlens = np.asarray(batch["vision_cu_seqlens"])
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError(
            "Qwen3.5 vision sharding requires exactly one padded sample per process; "
            f"got token_ids_BT shape {token_ids.shape}"
        )
    if pixel_values.ndim != 2 or pixel_values.shape[0] == 0:
        raise ValueError(
            "Qwen3.5 vision sharding requires a non-empty fixed pixel_values block per process"
        )
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or np.any(grid_thw <= 0):
        raise ValueError(
            f"Qwen3.5 image_grid_thw must have positive shape (M, 3); got {grid_thw.shape}"
        )
    if np.any(grid_thw[:, 0] != 1):
        raise ValueError("Qwen3.5 vision sharding supports images only; every grid t must equal 1")

    expected_cu = np.concatenate(
        [
            np.zeros(1, dtype=np.int32),
            np.cumsum(np.prod(grid_thw, axis=-1, dtype=np.int32), dtype=np.int32),
        ]
    )
    if pixel_values.shape[0] != int(expected_cu[-1]):
        raise ValueError(
            "Qwen3.5 process-local pixel_values and image_grid_thw disagree: "
            f"{pixel_values.shape[0]} patches != {int(expected_cu[-1])} grid patches"
        )
    if cu_seqlens.shape != expected_cu.shape or not np.array_equal(cu_seqlens, expected_cu):
        raise ValueError(
            "Qwen3.5 vision_cu_seqlens must match the padded image_grid_thw boundaries"
        )
    if patch_valid.shape != (pixel_values.shape[0],):
        raise ValueError(
            "Qwen3.5 vision_patch_valid must have one entry per pixel patch; "
            f"got {patch_valid.shape} for {pixel_values.shape[0]} patches"
        )
    if np.any(patch_valid[1:] > patch_valid[:-1]):
        raise ValueError("Qwen3.5 vision_patch_valid must be a valid-prefix mask")


def resolve_config(model_or_id: str | VLMConfig) -> VLMConfig:
    """Resolve VLM config from model id (Qwen3.5 or Qwen3-VL) or pass through."""
    if not isinstance(model_or_id, str):
        return model_or_id

    if is_supported_qwen3_5_model_id(model_or_id):
        return make_qwen3_5_config(model_or_id)
    if is_supported_qwen3_vl_model_id(model_or_id):
        return make_vl_config(model_or_id)

    hf_cfg = load_hf_config_from_source(model_or_id)
    model_type = hf_cfg.get("model_type")
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        return make_qwen3_5_config_from_hf(hf_cfg)
    if model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        return make_vl_config_from_hf(hf_cfg)

    raise ValueError(
        f"Unsupported VLM model/config source '{model_or_id}'. "
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
        _validate_qwen3_5_process_local_batch(batch, cfg, mesh)
        batch = {key: value for key, value in batch.items() if key != "vision_cu_seqlens"}
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
    model_or_id: str | VLMConfig,
    rng: jax.Array,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> tuple[nnx.Module, VLMConfig]:
    """Initialize a vision-language model."""
    cfg = resolve_config(model_or_id)
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
    vision_patch_valid: jax.Array,
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
        if vision_cu_seqlens is not None:
            raise ValueError("vision_cu_seqlens is not accepted for Qwen3.5")
        segment_ids_BT = attention_mask_BT.astype(jnp.int32)
        return model(
            token_ids_BT,
            segment_ids_BT,
            None,
            jnp.array(0, dtype=jnp.int32),
            vision_patch_valid=vision_patch_valid,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids_ZBT=position_ids_ZBT,
        )

    if isinstance(model, Qwen3VL):
        return model(
            token_ids_BT,
            attention_mask_BT,
            vision_patch_valid=vision_patch_valid,
            position_ids_ZBT=position_ids_ZBT,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_cu_seqlens=vision_cu_seqlens,
        )

    raise ValueError(f"Unsupported VLM model type: {type(model)}")


def load_pretrained(
    model_source: str | Path,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> tuple[nnx.Module, VLMConfig]:
    """Load a pretrained VLM from HuggingFace safetensors."""
    from omegalax.models.qwen3_5 import create_qwen3_5_from_safetensors
    from omegalax.models.qwen3_vl import create_qwen3_vl_from_safetensors

    local_dir = Path(model_source).expanduser().resolve()
    if not (local_dir / "config.json").is_file():
        raise ValueError(f"model_source must be a local HuggingFace model directory: {local_dir}")
    cfg = resolve_config(str(local_dir))
    # Validates any active mesh matches the requested (tp, fsdp, dp); the loaders
    # below build their own mesh from these sizes, so the return value is unused.
    ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    if isinstance(cfg, Qwen3VLConfig):
        model, cfg = create_qwen3_vl_from_safetensors(
            local_dir, tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size
        )
        return model, cfg
    if isinstance(cfg, Qwen3_5Config):
        model, cfg = create_qwen3_5_from_safetensors(
            local_dir, tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size
        )
        return model, cfg
    raise ValueError(f"Unsupported VLM config type for pretrained loading: {type(cfg)}")


def make_cache(*_args, **_kwargs):
    """Placeholder for cache creation to keep the interface symmetric."""
    return


def decode(*_args, **_kwargs):
    raise NotImplementedError("decode is not implemented for vision-language models.")
