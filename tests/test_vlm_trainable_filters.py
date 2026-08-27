from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.qwen3_5.config import Qwen3_5VisionConfig
from omegalax.models.qwen3_5.vision import VisionModel
from omegalax.models.shard_config import ShardConfig
from omegalax.trainers.vlm import (
    TrainConfig,
    _trainable_non_vision,
    _trainable_non_vision_except_merger,
    _validate_train_config,
)


class _MiniVLM(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        vision_cfg = Qwen3_5VisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=8,
            num_position_embeddings=16,
            dtype=jnp.float32,
        )
        self.vision = VisionModel(
            vision_cfg,
            shd_cfg=ShardConfig.no_sharding(),
            rngs=rngs,
        )
        self.text = nnx.Linear(8, 8, rngs=rngs)


def _selected_paths(model: nnx.Module, state_filter) -> set[str]:
    return {
        ".".join(str(part) for part in path)
        for path, _ in nnx.to_flat_state(nnx.state(model, state_filter))
    }


class VLMTrainableFilterTest(absltest.TestCase):
    def test_merger_filter_selects_only_the_real_merger(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        with mesh_rules(mesh):
            model = _MiniVLM(rngs=nnx.Rngs(0))

        all_paths = _selected_paths(model, nnx.Param)
        non_vision_paths = {path for path in all_paths if not path.startswith("vision.")}
        merger_paths = {path for path in all_paths if path.startswith("vision.merger.")}

        self.assertTrue(merger_paths)
        self.assertEqual(_selected_paths(model, _trainable_non_vision), non_vision_paths)
        self.assertEqual(
            _selected_paths(model, _trainable_non_vision_except_merger),
            non_vision_paths | merger_paths,
        )

    def test_merger_filter_rejects_deepstack_and_nested_mergers(self):
        param = nnx.Param(jnp.zeros(()))
        self.assertTrue(_trainable_non_vision_except_merger(("vision", "merger", "fc1"), param))
        self.assertFalse(
            _trainable_non_vision_except_merger(
                ("vision", "deepstack_mergers", 0, "merger", "fc1"), param
            )
        )
        self.assertFalse(
            _trainable_non_vision_except_merger(("wrapper", "vision", "merger", "fc1"), param)
        )

    def test_feature_combinations_fail_at_the_config_boundary(self):
        _validate_train_config(TrainConfig(freeze_vision_tower=True, train_vision_merger=True))
        _validate_train_config(TrainConfig(enable_lora=True, lora_qwen3_5_deltanet=True))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_train_config(TrainConfig(enable_lora=True, freeze_vision_tower=True))
        with self.assertRaisesRegex(ValueError, "requires freeze_vision_tower"):
            _validate_train_config(TrainConfig(train_vision_merger=True))
        with self.assertRaisesRegex(ValueError, "requires enable_lora"):
            _validate_train_config(TrainConfig(lora_qwen3_5_deltanet=True))


if __name__ == "__main__":
    absltest.main()
