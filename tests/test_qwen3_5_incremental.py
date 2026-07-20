"""Full-model incremental Qwen3.5 inference with recurrent states."""

from __future__ import annotations

import dataclasses
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=1"
).strip()

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api


class Qwen3_5IncrementalTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test_qwen3_5_incremental"])
        base_cfg = text_api.resolve_config("qwen3.5-smoke-dense")
        incremental_cfg = dataclasses.replace(
            base_cfg,
            num_hidden_layers=5,
            layer_types=base_cfg.layer_types + ("linear_attention",),
        )
        cls.model, cls.cfg = text_api.init_model(
            incremental_cfg,
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, "xla")

    def test_prefill_and_token_decode_match_uninterrupted_forward(self):
        tokens = jnp.asarray([[11, 29, 7, 41, 5, 83]], dtype=jnp.int32)
        positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
        position_ids = jnp.stack([positions] * 3, axis=0)
        attention_mask = jnp.ones_like(tokens)

        full_cache = text_api.make_cache(
            self.cfg,
            batch_size=1,
            token_len=3,
            generate_steps=3,
            dtype=self.cfg.dtype,
        )
        full_hidden, _, full_gdn, full_conv = text_api.forward_with_gdn_state(
            self.model,
            tokens,
            pad_id=0,
            cfg=self.cfg,
            attention_mask_BT=attention_mask,
            position_ids_ZBT=position_ids,
            return_conv_states=True,
            cache=full_cache,
        )
        cache = text_api.make_cache(
            self.cfg,
            batch_size=1,
            token_len=3,
            generate_steps=3,
            dtype=self.cfg.dtype,
        )

        hidden_parts = []
        gdn_states = None
        conv_states = None
        for start, end in ((0, 3), (3, 4), (4, 5), (5, 6)):
            hidden, _, gdn_states, conv_states = text_api.forward_with_gdn_state(
                self.model,
                tokens[:, start:end],
                pad_id=0,
                cfg=self.cfg,
                attention_mask_BT=attention_mask[:, start:end],
                initial_gdn_states=gdn_states,
                initial_conv_states=conv_states,
                position_ids_ZBT=position_ids[:, :, start:end],
                return_conv_states=True,
                cache=cache,
            )
            hidden_parts.append(hidden)

        incremental_hidden = jnp.concatenate(hidden_parts, axis=1)
        full_logits = jnp.einsum("BTD,DV->BTV", full_hidden, self.model.output_weight())
        incremental_logits = jnp.einsum(
            "BTD,DV->BTV", incremental_hidden, self.model.output_weight()
        )

        np.testing.assert_allclose(
            np.asarray(incremental_hidden, dtype=np.float32),
            np.asarray(full_hidden, dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.asarray(incremental_logits, dtype=np.float32),
            np.asarray(full_logits, dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        for incremental, full in zip(gdn_states, full_gdn, strict=True):
            np.testing.assert_allclose(
                np.asarray(incremental),
                np.asarray(full),
                rtol=1e-5,
                atol=1e-6,
            )
        for incremental, full in zip(conv_states, full_conv, strict=True):
            np.testing.assert_array_equal(np.asarray(incremental), np.asarray(full))
        for incremental_cache, uninterrupted_cache in zip(cache, full_cache, strict=True):
            if incremental_cache is not None:
                self.assertEqual(int(incremental_cache.cur_ind[...]), tokens.shape[1])
                self.assertEqual(int(uninterrupted_cache.cur_ind[...]), tokens.shape[1])
                np.testing.assert_array_equal(
                    np.asarray(incremental_cache.k_cache[...]),
                    np.asarray(uninterrupted_cache.k_cache[...]),
                )
                np.testing.assert_array_equal(
                    np.asarray(incremental_cache.v_cache[...]),
                    np.asarray(uninterrupted_cache.v_cache[...]),
                )


if __name__ == "__main__":
    absltest.main()
