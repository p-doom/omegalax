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


def _vision_batch():
    grids = np.tile(
        np.asarray([[1, 1, 1], [1, 1, 3]], dtype=np.int32),
        (4, 1),
    )
    return {
        "token_ids_BT": np.full((4, 4), 2, dtype=np.int32),
        "attention_mask_BT": np.ones((4, 4), dtype=np.int32),
        "loss_mask_BT": np.tile(np.asarray([[0, 1, 1, 1]], dtype=np.int32), (4, 1)),
        "pixel_values": np.arange(16, dtype=np.float32).reshape(16, 1),
        "vision_patch_valid": np.ones((16,), dtype=np.bool_),
        "image_grid_thw": grids,
        "position_ids_ZBT": np.broadcast_to(np.arange(4, dtype=np.int32), (3, 4, 4)).copy(),
    }


def _shard_global_batch(batch, mesh):
    sharded = {}
    for key, value in batch.items():
        spec = (
            P(None, "fsdp", None)
            if key == "position_ids_ZBT"
            else P("fsdp", *((None,) * (value.ndim - 1)))
        )
        sharded[key] = _put(mesh, value, spec)
    return sharded


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
    sharded = _shard_global_batch(_vision_batch(), mesh)
    optimizer = MixedPrecisionOptimizer(model, optax.sgd(1e-3))
    train_step = make_sft_train_step(cfg, num_loss_tiles=1)
    loss, metrics = train_step(optimizer, (sharded,))

    assert np.isfinite(float(loss))
    assert np.isfinite(float(metrics["grad_norm"]))
    assert bool(metrics["optimizer_healthy"])
    assert int(metrics["supervised_tokens"]) == 12
    assert int(optimizer.step[...]) == 1


def _assert_batch_ingress_rejects_process_misalignment(cfg, mesh):
    batch = _vision_batch()
    batch["vision_cu_seqlens"] = np.asarray([0, 1, 4], dtype=np.int32)
    try:
        vlm_api.shard_batch_dict(batch, cfg, mesh)
    except ValueError as error:
        assert "one local device per process" in str(error)
    else:
        raise AssertionError("single-process four-device vision ingress was accepted")


def _install_checked_attention():
    def checked_local(q, _k, v, cu, _scale):
        assert q.shape[0] == 4
        assert cu.shape == (3,)
        expected_cu = jnp.asarray([0, 1, 4], dtype=jnp.int32)
        valid = jnp.all(cu == expected_cu)
        return jnp.where(valid, v, jnp.full_like(v, jnp.nan))

    vision_lib._cudnn_packed_vision_attention_local = checked_local


def _assert_attention_does_not_all_gather(mesh):
    qkv = _put(mesh, np.arange(32, dtype=np.float32).reshape(16, 2, 1), P("fsdp", None, None))
    grid = _put(
        mesh,
        np.tile(np.asarray([[1, 1, 1], [1, 1, 3]], dtype=np.int32), (4, 1)),
        P("fsdp", None),
    )
    lowered = jax.jit(
        lambda q, k, v, g: vision_lib._cudnn_packed_vision_attention(q, k, v, g, 1.0)
    ).lower(qkv, qkv, qkv, grid)
    hlo = lowered.as_text().lower()
    assert "all_gather" not in hlo
    assert "all-gather" not in hlo


def main():
    assert jax.device_count() == 4
    mesh = make_mesh(tp_size=1, fsdp_size=4, dp_size=1)
    model, cfg = vlm_api.init_model(_config(), jax.random.key(0), tp_size=1, fsdp_size=4, dp_size=1)
    _install_checked_attention()
    _assert_attention_does_not_all_gather(mesh)
    _assert_embedding_and_output_weight(model, mesh)
    _assert_batch_ingress_rejects_process_misalignment(cfg, mesh)
    _assert_trainer_consumer(model, cfg, mesh)


if __name__ == "__main__":
    main()
