"""Qwen3.5 VLM contracts exercised on a four-device CPU mesh."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from omegalax.distributed.mesh import make_mesh
from omegalax.models.qwen3_5 import vision as vision_lib
from omegalax.models.qwen3_5.config import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.vlm import make_sft_train_step
from omegalax.vlm import api as vlm_api

P = PartitionSpec


def _config() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=("linear_attention",),
            partial_rotary_factor=0.5,
            mrope_section=(2, 1, 1),
            intermediate_size=32,
            linear_conv_kernel_dim=3,
            linear_key_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_value_head_dim=4,
            dtype=jnp.float32,
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=1,
            temporal_patch_size=1,
            spatial_merge_size=1,
            in_channels=1,
            out_hidden_size=16,
            num_position_embeddings=4,
            dtype=jnp.float32,
        ),
        image_token_id=2,
        video_token_id=3,
        vision_start_token_id=4,
        vision_end_token_id=5,
    )


def _put(mesh, value, spec):
    return jax.device_put(jnp.asarray(value), NamedSharding(mesh, spec))


def _assert_embedding_and_output_weight(model, mesh):
    token_ids = _put(mesh, np.arange(16, dtype=np.int32).reshape(4, 4), P("fsdp", None))

    text_hidden, _ = nnx.jit(lambda m, t: m.text(token_ids_BT=t))(model, token_ids)
    segment_ids = _put(mesh, np.ones((4, 4), dtype=np.int32), P("fsdp", None))
    positions = _put(
        mesh,
        np.broadcast_to(np.arange(4, dtype=np.int32), (3, 4, 4)),
        P(None, "fsdp", None),
    )
    provided_hidden, _ = nnx.jit(
        lambda m, t, s, p: m.text(
            token_ids_BT=t,
            segment_ids_BT=s,
            position_ids_ZBT=p,
        )
    )(model, token_ids, segment_ids, positions)

    assert text_hidden.shape == (4, 4, 16)
    assert text_hidden.sharding.spec == P("fsdp", None, None)
    np.testing.assert_allclose(np.asarray(provided_hidden), np.asarray(text_hidden))
    np.testing.assert_array_equal(
        np.asarray(model.output_weight()), np.asarray(model.lm_head.kernel[...])
    )


def _assert_trainer_consumer(model, cfg, mesh):
    batch = {
        "token_ids_BT": np.full((4, 4), cfg.image_token_id, dtype=np.int32),
        "attention_mask_BT": np.ones((4, 4), dtype=np.int32),
        "loss_mask_BT": np.tile(np.asarray([[0, 1, 1, 1]], dtype=np.int32), (4, 1)),
        "pixel_values": np.arange(16, dtype=np.float32).reshape(16, 1),
        "vision_patch_valid": np.ones((16,), dtype=np.bool_),
        "image_grid_thw": np.tile(np.asarray([[1, 2, 2]], dtype=np.int32), (4, 1)),
        "vision_cu_seqlens": np.asarray([0, 4, 8, 12, 16], dtype=np.int32),
        "position_ids_ZBT": np.broadcast_to(np.arange(4, dtype=np.int32), (3, 4, 4)).copy(),
    }
    sharded = vlm_api.shard_batch_dict(batch, cfg, mesh)
    assert "vision_cu_seqlens" not in sharded
    optimizer = MixedPrecisionOptimizer(model, optax.sgd(1e-3))
    train_step = make_sft_train_step(cfg, num_loss_tiles=1)
    loss, metrics = train_step(optimizer, (sharded,))

    assert np.isfinite(float(loss))
    assert np.isfinite(float(metrics["grad_norm"]))
    assert bool(metrics["optimizer_healthy"])
    assert int(metrics["supervised_tokens"]) == 12
    assert int(optimizer.step[...]) == 1


def _assert_batch_ingress_drops_local_offsets(cfg, mesh):
    batch = {
        "token_ids_BT": np.ones((4, 4), dtype=np.int32),
        "attention_mask_BT": np.ones((4, 4), dtype=np.int32),
        "loss_mask_BT": np.ones((4, 4), dtype=np.int32),
        "position_ids_ZBT": np.zeros((3, 4, 4), dtype=np.int32),
        "pixel_values": np.ones((16, 1), dtype=np.float32),
        "vision_patch_valid": np.ones((16,), dtype=np.bool_),
        "image_grid_thw": np.tile(np.asarray([[1, 2, 2]], dtype=np.int32), (4, 1)),
        # The leading M+1 dimension cannot be evenly sharded on this mesh.
        "vision_cu_seqlens": np.asarray([0, 4, 8, 12, 16], dtype=np.int32),
    }
    sharded = vlm_api.shard_batch_dict(batch, cfg, mesh)
    assert "vision_cu_seqlens" not in sharded
    assert sharded["image_grid_thw"].sharding.spec == P("fsdp", None)


def _install_checked_attention():
    def checked_local(q, _k, v, cu, _scale):
        assert q.shape[0] == 16
        assert cu.shape == (5,)
        expected_cu = jnp.arange(5, dtype=jnp.int32) * 4
        valid = jnp.all(cu == expected_cu)
        return jnp.where(valid, v, jnp.full_like(v, jnp.nan))

    vision_lib._cudnn_packed_vision_attention_local = checked_local


def main():
    assert jax.device_count() == 4
    mesh = make_mesh(tp_size=1, fsdp_size=4, dp_size=1)
    model, cfg = vlm_api.init_model(_config(), jax.random.key(0), tp_size=1, fsdp_size=4, dp_size=1)
    _install_checked_attention()
    _assert_embedding_and_output_weight(model, mesh)
    _assert_batch_ingress_drops_local_offsets(cfg, mesh)
    _assert_trainer_consumer(model, cfg, mesh)


if __name__ == "__main__":
    main()
