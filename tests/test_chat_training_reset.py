"""Training-equivalent segmented chat inference."""

from __future__ import annotations

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


def _position_ids(start: int, length: int) -> jax.Array:
    positions = jnp.arange(start, start + length, dtype=jnp.int32)[None, :]
    return jnp.stack([positions] * 3, axis=0)


def _make_reference_forward(cfg):
    @nnx.jit
    def reference_forward(
        model,
        cache,
        token_ids_BT,
        position_ids_ZBT,
        gdn_states,
        conv_states,
    ):
        hidden, _, final_gdn, final_conv = text_api.forward_with_gdn_state(
            model,
            token_ids_BT,
            pad_id=0,
            cfg=cfg,
            attention_mask_BT=jnp.ones_like(token_ids_BT),
            initial_gdn_states=gdn_states,
            initial_conv_states=conv_states,
            position_ids_ZBT=position_ids_ZBT,
            return_conv_states=True,
            cache=cache,
        )
        logits = jnp.einsum("BTD,DV->BTV", hidden, model.output_weight())
        return logits, final_gdn, final_conv

    return reference_forward


def _independent_segment_logits(model, cfg, tokens, state_cfg, segment_length, reference_forward):
    logits_parts = []
    gdn_states = None
    conv_states = None
    for start in range(0, tokens.shape[1], segment_length):
        end = min(start + segment_length, tokens.shape[1])
        position_start = start if state_cfg.pass_rope_positions else 0
        segment_cache = text_api.make_cache(
            cfg,
            batch_size=1,
            token_len=segment_length,
            generate_steps=0,
            dtype=cfg.dtype,
        )
        logits, final_gdn, final_conv = reference_forward(
            model,
            segment_cache,
            tokens[:, start:end],
            _position_ids(position_start, end - start),
            gdn_states,
            conv_states,
        )
        logits_parts.append(logits)
        if end - start == segment_length:
            if state_cfg.pass_gdn_state:
                if state_cfg.gdn_layer_limit is None:
                    gdn_states = final_gdn
                elif state_cfg.gdn_layer_limit == 0:
                    gdn_states = None
                else:
                    gdn_states = tuple(
                        state if idx < state_cfg.gdn_layer_limit else jnp.zeros_like(state)
                        for idx, state in enumerate(final_gdn)
                    )
            else:
                gdn_states = None
            if state_cfg.pass_conv_state:
                if state_cfg.gdn_layer_limit is None:
                    conv_states = final_conv
                elif state_cfg.gdn_layer_limit == 0:
                    conv_states = None
                else:
                    conv_states = tuple(
                        state if idx < state_cfg.gdn_layer_limit else jnp.zeros_like(state)
                        for idx, state in enumerate(final_conv)
                    )
            else:
                conv_states = None
        else:
            gdn_states = final_gdn
            conv_states = final_conv
    logits = jnp.concatenate(logits_parts, axis=1)
    return logits, gdn_states, conv_states


class TrainingResetChatTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test_chat_training_reset"])
        cls.model, cls.cfg = text_api.init_model(
            "qwen3.5-smoke-dense",
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, "xla")
        cls.reference_forward = staticmethod(_make_reference_forward(cls.cfg))

    def test_matches_independent_training_style_segment_forwards(self):
        tokens = jnp.asarray([[11, 29, 7, 41, 5, 83, 13, 17, 19, 23]], dtype=jnp.int32)
        state_cfg = StatePassingConfig(
            pass_gdn_state=True,
            gdn_layer_limit=1,
            pass_conv_state=True,
            pass_rope_positions=True,
        )
        events = []
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode=ChatMode.TRAINING_RESET,
            state_config=state_cfg,
            segment_length=4,
            event_sink=events.append,
        )

        first_logits = runtime.consume(tokens[:, :4])
        self.assertEqual(events, ["[KV RESET] completed_segment=1 consumed_tokens=4"])
        for layer_cache in runtime.cache:
            if layer_cache is not None:
                self.assertEqual(int(layer_cache.cur_ind[...]), 0)
                np.testing.assert_array_equal(np.asarray(layer_cache.k_cache[...]), 0)
                np.testing.assert_array_equal(np.asarray(layer_cache.v_cache[...]), 0)
        self.assertGreater(float(jnp.linalg.norm(runtime.gdn_states[0])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(runtime.conv_states[0])), 0.0)
        for state in runtime.gdn_states[1:] + runtime.conv_states[1:]:
            np.testing.assert_array_equal(np.asarray(state), 0)

        middle_logits = runtime.consume(tokens[:, 4:7])
        self.assertEqual(runtime.segment_tokens, 3)
        self.assertEqual(events, ["[KV RESET] completed_segment=1 consumed_tokens=4"])
        final_logits = runtime.consume(tokens[:, 7:])
        runtime_logits = jnp.concatenate([first_logits, middle_logits, final_logits], axis=1)
        reference_logits, reference_gdn, reference_conv = _independent_segment_logits(
            self.model,
            self.cfg,
            tokens,
            state_cfg,
            segment_length=4,
            reference_forward=self.reference_forward,
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
        self.assertEqual(
            events,
            [
                "[KV RESET] completed_segment=1 consumed_tokens=4",
                "[KV RESET] completed_segment=2 consumed_tokens=4",
            ],
        )
        self.assertEqual(runtime.completed_segments, 2)
        self.assertEqual(runtime.segment_tokens, 2)
        self.assertEqual(runtime.total_tokens, 10)
        for layer_cache in runtime.cache:
            if layer_cache is not None:
                self.assertEqual(int(layer_cache.cur_ind[...]), 2)

    def test_disabled_passing_drops_recurrent_states_at_boundary(self):
        events = []
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode=ChatMode.TRAINING_RESET,
            state_config=StatePassingConfig(
                pass_gdn_state=False,
                gdn_layer_limit=None,
                pass_conv_state=False,
                pass_rope_positions=False,
            ),
            segment_length=4,
            event_sink=events.append,
        )

        tokens = jnp.asarray([[3, 5, 7, 11, 13, 17, 19, 23]], dtype=jnp.int32)
        runtime_logits = runtime.consume(tokens)
        reference_logits, _, _ = _independent_segment_logits(
            self.model,
            self.cfg,
            tokens,
            runtime.state_config,
            segment_length=4,
            reference_forward=self.reference_forward,
        )

        self.assertIsNone(runtime.gdn_states)
        self.assertIsNone(runtime.conv_states)
        np.testing.assert_allclose(
            np.asarray(runtime_logits, dtype=np.float32),
            np.asarray(reference_logits, dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        self.assertEqual(
            events,
            [
                "[KV RESET] completed_segment=1 consumed_tokens=4",
                "[KV RESET] completed_segment=2 consumed_tokens=4",
            ],
        )


if __name__ == "__main__":
    absltest.main()
