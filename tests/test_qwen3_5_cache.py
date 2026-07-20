"""Optional KV-cache behavior for Qwen3.5 full-attention layers."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=1"
).strip()

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.models.qwen3_5.cache import init_cache
from omegalax.models.qwen3_5.rope import generate_text_rope
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api


class Qwen3_5CacheTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test_qwen3_5_cache"])
        cls.model, cls.cfg = text_api.init_model(
            "qwen3.5-smoke-dense",
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, "xla")
        cls.full_attention_idx = cls.cfg.layer_types.index("full_attention")
        cls.attention = cls.model.text.layers[cls.full_attention_idx].attn

    def _inputs(self, length: int = 7):
        hidden = jax.random.normal(
            jax.random.key(1),
            (1, length, self.cfg.hidden_size),
            dtype=self.cfg.dtype,
        )
        positions = jnp.arange(length, dtype=jnp.int32)[None, :]
        position_ids = jnp.stack([positions] * 3, axis=0)
        cos, sin = generate_text_rope(
            position_ids,
            self.cfg.head_dim,
            self.cfg.partial_rotary_factor,
            self.cfg.rope_theta,
            self.cfg.mrope_section,
        )
        return hidden, cos.astype(self.cfg.dtype), sin.astype(self.cfg.dtype), positions

    def test_allocates_cache_only_for_full_attention_layers(self):
        cache = init_cache(self.cfg, batch_size=1, cache_size=8, dtype=self.cfg.dtype)

        self.assertLen(cache, self.cfg.num_hidden_layers)
        for layer_type, layer_cache in zip(self.cfg.layer_types, cache, strict=True):
            if layer_type == "full_attention":
                self.assertIsNotNone(layer_cache)
                self.assertEqual(
                    layer_cache.k_cache.shape,
                    (1, 8, self.cfg.num_key_value_heads, self.cfg.head_dim),
                )
                self.assertEqual(layer_cache.v_cache.shape, layer_cache.k_cache.shape)
                self.assertEqual(layer_cache.k_cache.dtype, self.cfg.dtype)
                self.assertEqual(layer_cache.v_cache.dtype, self.cfg.dtype)
                self.assertEqual(int(layer_cache.cur_ind[...]), 0)
            else:
                self.assertIsNone(layer_cache)

    def test_cached_prefill_matches_existing_attention_path(self):
        hidden, cos, sin, positions = self._inputs()
        segment_ids = jnp.ones(positions.shape, dtype=jnp.int32)

        baseline = self.attention(hidden, cos, sin, segment_ids, positions)
        explicit_none = self.attention(
            hidden,
            cos,
            sin,
            segment_ids,
            positions,
            cache=None,
        )
        cache = init_cache(self.cfg, batch_size=1, cache_size=8, dtype=self.cfg.dtype)
        layer_cache = cache[self.full_attention_idx]
        split = 4
        cached_first = self.attention(
            hidden[:, :split],
            cos[:, :split],
            sin[:, :split],
            segment_ids[:, :split],
            positions[:, :split],
            cache=layer_cache,
        )
        cached_second = self.attention(
            hidden[:, split:],
            cos[:, split:],
            sin[:, split:],
            segment_ids[:, split:],
            positions[:, split:],
            cache=layer_cache,
        )
        cached = jnp.concatenate([cached_first, cached_second], axis=1)

        np.testing.assert_array_equal(np.asarray(explicit_none), np.asarray(baseline))
        np.testing.assert_allclose(
            np.asarray(cached, dtype=np.float32),
            np.asarray(baseline, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertEqual(int(layer_cache.cur_ind[...]), hidden.shape[1])


if __name__ == "__main__":
    absltest.main()
