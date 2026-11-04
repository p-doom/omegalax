import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from jax.sharding import PartitionSpec as P

from omegalax.models.qwen3_5.config import Qwen3_5VisionConfig
from omegalax.models.qwen3_5.vision import VisionPatchEmbed


class Qwen3_5VisionPatchEmbedTest(absltest.TestCase):
    def test_patch_embed_uses_channel_first_flat_patch_layout(self):
        cfg = Qwen3_5VisionConfig(
            hidden_size=2,
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            dtype=jnp.float32,
        )
        with jax.set_mesh(jax.make_mesh((1,), ("hidden",))):
            patch_embed = VisionPatchEmbed(cfg, hidden_shd=P(), rngs=nnx.Rngs(0))
            kernel = np.arange(48, dtype=np.float32).reshape(2, 2, 2, 3, 2) / 48
            bias = np.array([0.25, -0.5], dtype=np.float32)
            patch_embed.proj.kernel[...] = jnp.asarray(kernel)
            patch_embed.proj.bias[...] = jnp.asarray(bias)

            pixels = np.arange(24, dtype=np.float32).reshape(1, 24) / 24
            patches = pixels.reshape(1, 3, 2, 2, 2).transpose(0, 2, 3, 4, 1)
            expected = np.einsum("nthwc,thwco->no", patches, kernel) + bias

            np.testing.assert_allclose(
                np.asarray(patch_embed(jnp.asarray(pixels))), expected
            )

    def test_channel_last_reshape_would_not_match(self):
        cfg = Qwen3_5VisionConfig(
            hidden_size=2,
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            dtype=jnp.float32,
        )
        with jax.set_mesh(jax.make_mesh((1,), ("hidden",))):
            patch_embed = VisionPatchEmbed(cfg, hidden_shd=P(), rngs=nnx.Rngs(0))
            kernel = np.arange(48, dtype=np.float32).reshape(2, 2, 2, 3, 2) / 48
            bias = np.array([0.25, -0.5], dtype=np.float32)
            patch_embed.proj.kernel[...] = jnp.asarray(kernel)
            patch_embed.proj.bias[...] = jnp.asarray(bias)

            pixels = np.arange(24, dtype=np.float32).reshape(1, 24) / 24
            scrambled = pixels.reshape(1, 2, 2, 2, 3)
            wrong = np.einsum("nthwc,thwco->no", scrambled, kernel) + bias

            got = np.asarray(patch_embed(jnp.asarray(pixels)))
            self.assertFalse(np.allclose(got, wrong))


if __name__ == "__main__":
    absltest.main()
