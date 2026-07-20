"""Continuous autoregressive chat without state-passing boundaries."""

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
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.chat import ChatMode, ChatRuntime, StatePassingConfig


def _make_reference_forward(cfg):
    @nnx.jit
    def reference_forward(model, cache, token_ids_BT, position_ids_ZBT):
        hidden, _, final_gdn, final_conv = text_api.forward_with_gdn_state(
            model,
            token_ids_BT,
            pad_id=0,
            cfg=cfg,
            attention_mask_BT=jnp.ones_like(token_ids_BT),
            position_ids_ZBT=position_ids_ZBT,
            return_conv_states=True,
            cache=cache,
        )
        logits = jnp.einsum("BTD,DV->BTV", hidden, model.output_weight())
        return logits, final_gdn, final_conv

    return reference_forward


def _position_ids(length: int) -> jax.Array:
    positions = jnp.arange(length, dtype=jnp.int32)[None, :]
    return jnp.stack([positions] * 3, axis=0)


class ContinuousChatTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test_chat_continuous"])
        base_cfg = text_api.resolve_config("qwen3.5-smoke-dense")
        continuous_cfg = dataclasses.replace(
            base_cfg,
            num_hidden_layers=5,
            layer_types=base_cfg.layer_types + ("linear_attention",),
        )
        cls.model, cls.cfg = text_api.init_model(
            continuous_cfg,
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, "xla")
        cls.reference_forward = staticmethod(_make_reference_forward(cls.cfg))

    def test_irregular_calls_match_one_uninterrupted_session(self):
        tokens = jnp.asarray([[11, 29, 7, 41, 5, 83]], dtype=jnp.int32)
        self.assertEqual(
            self.cfg.layer_types,
            (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ),
        )
        events = []
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode="continuous",
            state_config=StatePassingConfig(
                pass_gdn_state=False,
                gdn_layer_limit=0,
                pass_conv_state=False,
                pass_rope_positions=False,
            ),
            segment_length=2,
            max_session_tokens=6,
            event_sink=events.append,
        )
        for layer_cache in runtime.cache:
            if layer_cache is not None:
                self.assertEqual(layer_cache.k_cache.shape[1], 2)

        runtime_logits = jnp.concatenate(
            [
                runtime.consume(tokens[:, :2]),
                runtime.consume(tokens[:, 2:3]),
                runtime.consume(tokens[:, 3:]),
            ],
            axis=1,
        )
        reference_cache = text_api.make_cache(
            self.cfg,
            batch_size=1,
            token_len=6,
            generate_steps=0,
            dtype=self.cfg.dtype,
        )
        reference_logits, reference_gdn, reference_conv = self.reference_forward(
            self.model,
            reference_cache,
            tokens,
            _position_ids(6),
        )

        np.testing.assert_allclose(
            np.asarray(runtime_logits, dtype=np.float32),
            np.asarray(reference_logits, dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        for actual, expected in zip(runtime.gdn_states, reference_gdn, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6
            )
        for actual, expected in zip(runtime.conv_states, reference_conv, strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        for actual_cache, expected_cache in zip(runtime.cache, reference_cache, strict=True):
            self.assertEqual(actual_cache is None, expected_cache is None)
            if actual_cache is not None:
                self.assertEqual(actual_cache.k_cache.shape[1], 6)
                self.assertEqual(int(actual_cache.cur_ind[...]), 6)
                np.testing.assert_array_equal(
                    np.asarray(actual_cache.k_cache[...]), np.asarray(expected_cache.k_cache[...])
                )
                np.testing.assert_array_equal(
                    np.asarray(actual_cache.v_cache[...]), np.asarray(expected_cache.v_cache[...])
                )
        self.assertEqual(events, [])
        self.assertEqual(runtime.completed_segments, 0)
        self.assertEqual(runtime.segment_tokens, 0)
        self.assertEqual(runtime.total_tokens, 6)

        with self.assertRaisesRegex(ValueError, "max_session_tokens=6"):
            runtime.consume(jnp.asarray([[97]], dtype=jnp.int32))
        self.assertEqual(runtime.total_tokens, 6)
        self.assertEqual(events, [])

    def test_rejects_over_limit_block_before_mutating_state(self):
        events = []
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode=ChatMode.CONTINUOUS,
            state_config=StatePassingConfig(False, None, False, False),
            max_session_tokens=5,
            event_sink=events.append,
        )
        runtime.consume(jnp.asarray([[3, 5, 7]], dtype=jnp.int32))
        cache_snapshot = [
            None
            if layer_cache is None
            else (
                int(layer_cache.cur_ind[...]),
                np.asarray(layer_cache.k_cache[...]).copy(),
                np.asarray(layer_cache.v_cache[...]).copy(),
            )
            for layer_cache in runtime.cache
        ]
        gdn_snapshot = tuple(np.asarray(state).copy() for state in runtime.gdn_states)
        conv_snapshot = tuple(np.asarray(state).copy() for state in runtime.conv_states)
        counter_snapshot = (
            runtime.total_tokens,
            runtime.segment_tokens,
            runtime.completed_segments,
        )

        with self.assertRaisesRegex(ValueError, "would exceed max_session_tokens=5"):
            runtime.consume(jnp.asarray([[11, 13, 17]], dtype=jnp.int32))

        self.assertEqual(
            (runtime.total_tokens, runtime.segment_tokens, runtime.completed_segments),
            counter_snapshot,
        )
        self.assertEqual(events, [])
        for layer_cache, snapshot in zip(runtime.cache, cache_snapshot, strict=True):
            if snapshot is None:
                self.assertIsNone(layer_cache)
            else:
                self.assertEqual(int(layer_cache.cur_ind[...]), snapshot[0])
                np.testing.assert_array_equal(np.asarray(layer_cache.k_cache[...]), snapshot[1])
                np.testing.assert_array_equal(np.asarray(layer_cache.v_cache[...]), snapshot[2])
        for state, snapshot in zip(runtime.gdn_states, gdn_snapshot, strict=True):
            np.testing.assert_array_equal(np.asarray(state), snapshot)
        for state, snapshot in zip(runtime.conv_states, conv_snapshot, strict=True):
            np.testing.assert_array_equal(np.asarray(state), snapshot)


if __name__ == "__main__":
    absltest.main()
