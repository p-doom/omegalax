from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.vision_routing import _vision_token_destinations


class VisionTokenDestinationsTest(absltest.TestCase):
    def test_interleaved_padding_routes_real_embeddings_compactly(self):
        image_mask_BT = jnp.array(
            [
                [False, True, True, False],
                [True, False, False, False],
            ]
        )
        vision_patch_valid = jnp.repeat(
            jnp.array([True, False, True, True, False]),
            4,
        )

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            batch_N, seq_N = _vision_token_destinations(
                image_mask_BT,
                vision_patch_valid,
                spatial_merge_size=2,
            )

        np.testing.assert_array_equal(batch_N, [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(seq_N, [1, 4, 2, 0, 4])


if __name__ == "__main__":
    absltest.main()
