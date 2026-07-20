"""KV carry behavior at state-passing chat boundaries."""

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


class CarryChatTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test_chat_carry"])
        cls.model, cls.cfg = text_api.init_model(
            "qwen3.5-smoke-dense",
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, "xla")
        cls.reference_forward = staticmethod(_make_reference_forward(cls.cfg))

    def test_carries_exact_last_kv_entries_without_reprocessing(self):
        tokens = jnp.asarray([[11, 29, 7, 41, 5, 83, 13, 17, 19]], dtype=jnp.int32)
        state_config = StatePassingConfig(
            pass_gdn_state=True,
            gdn_layer_limit=None,
            pass_conv_state=True,
            pass_rope_positions=True,
        )
        events = []
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode="carry_128",
            state_config=state_config,
            segment_length=4,
            carry_length=2,
            event_sink=events.append,
        )

        first_logits = runtime.consume(tokens[:, :4])

        self.assertEqual(events, ["[KV CARRY] completed_segment=1 carried_tokens=2"])
        expected_k = []
        expected_v = []
        reference_first_cache = text_api.make_cache(
            self.cfg,
            batch_size=1,
            token_len=4,
            generate_steps=2,
            dtype=self.cfg.dtype,
        )
        reference_first_logits, reference_gdn, reference_conv = self.reference_forward(
            self.model,
            reference_first_cache,
            tokens[:, :4],
            _position_ids(0, 4),
            None,
            None,
        )
        for runtime_cache, reference_cache in zip(
            runtime.cache, reference_first_cache, strict=True
        ):
            if runtime_cache is not None:
                expected_k.append(np.asarray(reference_cache.k_cache[:, 2:4]).copy())
                expected_v.append(np.asarray(reference_cache.v_cache[:, 2:4]).copy())
                self.assertEqual(runtime_cache.k_cache.shape[1], 6)
                self.assertEqual(int(runtime_cache.cur_ind[...]), 2)
                np.testing.assert_array_equal(
                    np.asarray(runtime_cache.k_cache[:, :2]), expected_k[-1]
                )
                np.testing.assert_array_equal(
                    np.asarray(runtime_cache.v_cache[:, :2]), expected_v[-1]
                )
                np.testing.assert_array_equal(np.asarray(runtime_cache.k_cache[:, 2:]), 0)
                np.testing.assert_array_equal(np.asarray(runtime_cache.v_cache[:, 2:]), 0)

        second_logits = runtime.consume(tokens[:, 4:])
        reference_second_cache = text_api.make_cache(
            self.cfg,
            batch_size=1,
            token_len=4,
            generate_steps=2,
            dtype=self.cfg.dtype,
        )
        carried_idx = 0
        for layer_cache in reference_second_cache:
            if layer_cache is not None:
                layer_cache.k_cache[:, :2] = jnp.asarray(expected_k[carried_idx])
                layer_cache.v_cache[:, :2] = jnp.asarray(expected_v[carried_idx])
                layer_cache.cur_ind[...] = jnp.array(2, dtype=jnp.int32)
                carried_idx += 1
        reference_second_logits, reference_second_gdn, reference_second_conv = (
            self.reference_forward(
                self.model,
                reference_second_cache,
                tokens[:, 4:8],
                _position_ids(4, 4),
                reference_gdn,
                reference_conv,
            )
        )
        second_expected_k = []
        second_expected_v = []
        for layer_cache in reference_second_cache:
            if layer_cache is not None:
                second_expected_k.append(np.asarray(layer_cache.k_cache[:, 4:6]).copy())
                second_expected_v.append(np.asarray(layer_cache.v_cache[:, 4:6]).copy())

        reference_third_cache = text_api.make_cache(
            self.cfg,
            batch_size=1,
            token_len=4,
            generate_steps=2,
            dtype=self.cfg.dtype,
        )
        carried_idx = 0
        for layer_cache in reference_third_cache:
            if layer_cache is not None:
                layer_cache.k_cache[:, :2] = jnp.asarray(second_expected_k[carried_idx])
                layer_cache.v_cache[:, :2] = jnp.asarray(second_expected_v[carried_idx])
                layer_cache.cur_ind[...] = jnp.array(2, dtype=jnp.int32)
                carried_idx += 1
        reference_third_logits, reference_final_gdn, reference_final_conv = self.reference_forward(
            self.model,
            reference_third_cache,
            tokens[:, 8:],
            _position_ids(8, 1),
            reference_second_gdn,
            reference_second_conv,
        )
        reference_remaining_logits = jnp.concatenate(
            [reference_second_logits, reference_third_logits], axis=1
        )

        np.testing.assert_allclose(
            np.asarray(first_logits, dtype=np.float32),
            np.asarray(reference_first_logits, dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.asarray(second_logits, dtype=np.float32),
            np.asarray(reference_remaining_logits, dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        for actual, expected in zip(runtime.gdn_states, reference_final_gdn, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6
            )
        for actual, expected in zip(runtime.conv_states, reference_final_conv, strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

        carried_idx = 0
        for layer_cache in runtime.cache:
            if layer_cache is not None:
                self.assertEqual(int(layer_cache.cur_ind[...]), 3)
                np.testing.assert_array_equal(
                    np.asarray(layer_cache.k_cache[:, :2]), second_expected_k[carried_idx]
                )
                np.testing.assert_array_equal(
                    np.asarray(layer_cache.v_cache[:, :2]), second_expected_v[carried_idx]
                )
                carried_idx += 1
        self.assertEqual(
            events,
            [
                "[KV CARRY] completed_segment=1 carried_tokens=2",
                "[KV CARRY] completed_segment=2 carried_tokens=2",
            ],
        )
        self.assertEqual(runtime.completed_segments, 2)
        self.assertEqual(runtime.segment_tokens, 1)
        self.assertEqual(runtime.total_tokens, 9)

    def test_accepts_minimum_carry_length(self):
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode=ChatMode.CARRY_128,
            state_config=StatePassingConfig(
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_conv_state=False,
                pass_rope_positions=False,
            ),
            segment_length=4,
            carry_length=1,
        )
        for layer_cache in runtime.cache:
            if layer_cache is not None:
                self.assertEqual(layer_cache.k_cache.shape[1], 5)
        self.assertIs(runtime.mode, ChatMode.CARRY_128)

        with self.assertRaisesRegex(ValueError, "carry_length"):
            ChatRuntime(
                self.model,
                self.cfg,
                mode="carry_128",
                state_config=runtime.state_config,
                segment_length=4,
                carry_length=0,
            )


if __name__ == "__main__":
    absltest.main()
