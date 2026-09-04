"""Qwen3.5 process-local vision block contracts."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from absl.testing import absltest

from omegalax.data.collator_qwen3 import _pad_vision_arrays
from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.vlm.api import _validate_qwen3_5_process_local_batch


def _config():
    return Qwen3_5Config(text_config=Qwen3_5TextConfig(shd_cfg=ShardConfig.no_sharding()))


def _batch():
    pixels = np.arange(4, dtype=np.float32).reshape(4, 1)
    grid = np.asarray([[1, 1, 1], [1, 1, 3]], dtype=np.int32)
    pixels, grid, cu = _pad_vision_arrays(
        pixels,
        grid,
        merge_size=1,
        max_patches=8,
        max_images=3,
    )
    return {
        "token_ids_BT": np.ones((1, 4), dtype=np.int32),
        "pixel_values": pixels,
        "vision_patch_valid": np.arange(8) < 4,
        "image_grid_thw": grid,
        "vision_cu_seqlens": cu,
    }


class Qwen3_5VisionShardingTest(absltest.TestCase):
    def test_collator_padding_defines_one_complete_local_block(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        batch = _batch()

        _validate_qwen3_5_process_local_batch(batch, _config(), mesh)

        np.testing.assert_array_equal(batch["vision_cu_seqlens"], [0, 1, 4, 8])
        self.assertEqual(int(np.prod(batch["image_grid_thw"], axis=-1).sum()), 8)

    def test_batchwide_block_is_rejected(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        batch = _batch()
        batch["token_ids_BT"] = np.ones((2, 4), dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "exactly one padded sample"):
            _validate_qwen3_5_process_local_batch(batch, _config(), mesh)

    def test_offsets_must_match_padded_grid(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        batch = _batch()
        batch["vision_cu_seqlens"] = np.asarray([0, 2, 4, 8], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "must match the padded image_grid_thw"):
            _validate_qwen3_5_process_local_batch(batch, _config(), mesh)


if __name__ == "__main__":
    absltest.main()
