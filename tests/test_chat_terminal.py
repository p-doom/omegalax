"""Sampling, generation, reset, and terminal-chat helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

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
from omegalax.text.chat import (
    ChatMode,
    ChatRuntime,
    GenerationStop,
    StatePassingConfig,
    generate_token_ids,
    sample_token,
)
from scripts.chat_checkpoint import (
    format_user_turn,
    load_chat,
    parse_layer_limit,
    run_repl,
)


class _FakeRuntime:
    def __init__(self, logits, *, mode=ChatMode.TRAINING_RESET, total=0, maximum=32):
        self.logits = list(logits)
        self.mode = mode
        self.total_tokens = total
        self.max_session_tokens = maximum
        self.consumed = []

    def consume(self, token_ids_BT):
        token = int(np.asarray(token_ids_BT)[0, 0])
        self.consumed.append(token)
        self.total_tokens += 1
        return jnp.asarray(self.logits.pop(0), dtype=jnp.float32)[None, None, :]


class _ReplRuntime:
    mode = ChatMode.TRAINING_RESET
    max_session_tokens = 32

    def __init__(self):
        self.total_tokens = 0
        self.reset_count = 0
        self.consumed = []

    def consume(self, token_ids_BT):
        tokens = np.asarray(token_ids_BT).reshape(-1).tolist()
        self.consumed.append(tokens)
        self.total_tokens += len(tokens)
        if len(tokens) > 1:
            logits = np.tile(np.asarray([0.0, 5.0, 0.0]), (len(tokens), 1))
        elif tokens == [1]:
            logits = np.asarray([[0.0, 0.0, 5.0]])
        else:
            logits = np.asarray([[5.0, 0.0, 0.0]])
        return jnp.asarray(logits, dtype=jnp.float32)[None, :, :]

    def reset(self):
        self.total_tokens = 0
        self.reset_count += 1


class _Tokenizer:
    eos_token_id = 2

    def __init__(self):
        self.encoded = []

    def encode(self, text, *, add_special_tokens):
        self.encoded.append((text, add_special_tokens))
        return [10, 11]


class _Streamer:
    def __init__(self):
        self.tokens = []
        self.ended = False

    def put(self, token):
        token = np.asarray(token)
        if token.shape != (1,):
            raise AssertionError(f"Expected streamed token shape (1,), got {token.shape}")
        self.tokens.extend(token.tolist())

    def end(self):
        self.ended = True


class ChatTerminalTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(
                [
                    "test_chat_terminal",
                    "--checkpoint=/tmp/checkpoint",
                    "--mode=training_reset",
                    "--pass_gdn_state=true",
                    "--gdn_layer_limit=all",
                    "--pass_conv_state=false",
                    "--pass_rope_positions=false",
                ]
            )
        cls.model, cls.cfg = text_api.init_model(
            "qwen3.5-smoke-dense",
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, "xla")

    def test_greedy_and_seeded_sampling_are_reproducible(self):
        logits = jnp.asarray([0.0, 1.0, 3.0, 2.0], dtype=jnp.float32)
        for seed in range(4):
            token, _ = sample_token(logits, jax.random.key(seed), temperature=0.0)
            self.assertEqual(token, 2)

        def draw_sequence(seed):
            key = jax.random.key(seed)
            result = []
            for _ in range(8):
                token, key = sample_token(
                    logits,
                    key,
                    temperature=0.7,
                    top_k=3,
                    top_p=0.9,
                )
                result.append(token)
            return result

        self.assertEqual(draw_sequence(7), draw_sequence(7))
        uniform_logits = jnp.zeros((4,), dtype=jnp.float32)

        def draw_uniform(seed):
            key = jax.random.key(seed)
            draws = []
            for _ in range(24):
                token, key = sample_token(uniform_logits, key, temperature=1.0)
                draws.append(token)
            return draws

        self.assertNotEqual(draw_uniform(7), draw_uniform(8))
        self.assertGreater(len(set(draw_uniform(7))), 1)
        top_p_key = jax.random.key(11)
        top_p_draws = []
        for _ in range(12):
            token, top_p_key = sample_token(
                jnp.asarray([3.0, 2.0, 1.0, 0.0]),
                top_p_key,
                temperature=1.0,
                top_p=0.5,
            )
            top_p_draws.append(token)
        self.assertEqual(top_p_draws, [0] * 12)
        top_one, _ = sample_token(
            logits,
            jax.random.key(3),
            temperature=1.0,
            top_k=1,
        )
        self.assertEqual(top_one, 2)
        with self.assertRaisesRegex(ValueError, "temperature"):
            sample_token(logits, jax.random.key(0), temperature=-0.1)

    def test_generation_consumes_eos_but_does_not_return_it(self):
        runtime = _FakeRuntime(
            logits=(
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 0.0],
            )
        )
        result = generate_token_ids(
            runtime,
            initial_logits_V=jnp.asarray([0.0, 5.0, 0.0]),
            eos_id=2,
            max_new_tokens=8,
            key=jax.random.key(0),
            temperature=0.0,
        )

        self.assertEqual(result.token_ids, (1,))
        self.assertEqual(runtime.consumed, [1, 2])
        self.assertIs(result.stop, GenerationStop.EOS)

    def test_generation_stops_before_sampling_beyond_continuous_limit(self):
        runtime = _FakeRuntime(
            logits=(),
            mode=ChatMode.CONTINUOUS,
            total=5,
            maximum=5,
        )

        with mock.patch(
            "omegalax.text.chat.sample_token",
            side_effect=AssertionError("sampling must not run"),
        ):
            result = generate_token_ids(
                runtime,
                initial_logits_V=jnp.asarray([0.0, 1.0]),
                eos_id=0,
                max_new_tokens=2,
                key=jax.random.key(0),
                temperature=0.0,
            )

        self.assertEqual(result.token_ids, ())
        self.assertEqual(runtime.consumed, [])
        self.assertIs(result.stop, GenerationStop.SESSION_LIMIT)

    def test_generation_consumes_exact_max_new_tokens(self):
        runtime = _FakeRuntime(logits=([0.0, 5.0, 0.0],) * 3)
        result = generate_token_ids(
            runtime,
            initial_logits_V=jnp.asarray([0.0, 5.0, 0.0]),
            eos_id=2,
            max_new_tokens=3,
            key=jax.random.key(0),
            temperature=0.0,
        )

        self.assertEqual(result.token_ids, (1, 1, 1))
        self.assertEqual(runtime.consumed, [1, 1, 1])
        self.assertIs(result.stop, GenerationStop.MAX_NEW_TOKENS)

    def test_last_allowed_token_reports_continuous_session_limit(self):
        runtime = _FakeRuntime(
            logits=([0.0, 5.0, 0.0],),
            mode=ChatMode.CONTINUOUS,
            total=4,
            maximum=5,
        )
        result = generate_token_ids(
            runtime,
            initial_logits_V=jnp.asarray([0.0, 5.0, 0.0]),
            eos_id=2,
            max_new_tokens=1,
            key=jax.random.key(0),
            temperature=0.0,
        )

        self.assertEqual(result.token_ids, (1,))
        self.assertEqual(runtime.consumed, [1])
        self.assertIs(result.stop, GenerationStop.SESSION_LIMIT)

    def test_runtime_reset_clears_all_session_state(self):
        events = []
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode=ChatMode.TRAINING_RESET,
            state_config=StatePassingConfig(True, None, True, True),
            segment_length=4,
            event_sink=events.append,
        )
        runtime.consume(jnp.asarray([[11, 29, 7, 41, 5]], dtype=jnp.int32))
        self.assertIsNotNone(runtime.gdn_states)
        self.assertIsNotNone(runtime.conv_states)
        self.assertEqual(runtime.completed_segments, 1)
        self.assertEqual(runtime.segment_tokens, 1)
        self.assertLen(events, 1)
        events.clear()

        runtime.reset()

        self.assertIsNone(runtime.gdn_states)
        self.assertIsNone(runtime.conv_states)
        self.assertEqual(runtime.total_tokens, 0)
        self.assertEqual(runtime.segment_tokens, 0)
        self.assertEqual(runtime.completed_segments, 0)
        self.assertEqual(events, [])
        for layer_cache in runtime.cache:
            if layer_cache is not None:
                self.assertEqual(int(layer_cache.cur_ind[...]), 0)
                np.testing.assert_array_equal(np.asarray(layer_cache.k_cache[...]), 0)
                np.testing.assert_array_equal(np.asarray(layer_cache.v_cache[...]), 0)

    def test_load_chat_wires_checkpoint_tokenizer_and_runtime(self):
        resolved = SimpleNamespace(config_path="/run/config.json", step=17)
        state_config = StatePassingConfig(True, 1, True, False)
        model = object()
        cfg = object()
        tokenizer = object()
        runtime = object()
        with (
            mock.patch(
                "scripts.chat_checkpoint.resolve_checkpoint", return_value=resolved
            ) as resolve,
            mock.patch(
                "scripts.chat_checkpoint.text_api.init_model",
                return_value=(model, cfg),
            ) as init_model,
            mock.patch(
                "scripts.chat_checkpoint.restore_model_params", return_value=model
            ) as restore,
            mock.patch(
                "scripts.chat_checkpoint.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
            mock.patch("scripts.chat_checkpoint.ChatRuntime", return_value=runtime) as runtime_cls,
        ):
            actual = load_chat(
                "/run",
                tokenizer_name="Qwen/tokenizer",
                mode="carry_128",
                state_config=state_config,
                max_session_tokens=123,
                seed=9,
            )

        self.assertEqual(actual, (runtime, tokenizer, resolved))
        resolve.assert_called_once_with("/run")
        self.assertEqual(init_model.call_args.args[0], "/run/config.json")
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(init_model.call_args.args[1])),
            np.asarray(jax.random.key_data(jax.random.key(9))),
        )
        self.assertEqual(init_model.call_args.kwargs, {"tp_size": 1, "fsdp_size": 1, "dp_size": 1})
        restore.assert_called_once_with(model, resolved)
        load_tokenizer.assert_called_once_with("Qwen/tokenizer")
        runtime_cls.assert_called_once_with(
            model,
            cfg,
            mode="carry_128",
            state_config=state_config,
            segment_length=2048,
            carry_length=128,
            max_session_tokens=123,
        )

    def test_repl_encodes_incremental_turns_resets_and_exits_on_eof(self):
        runtime = _ReplRuntime()
        tokenizer = _Tokenizer()
        streamers = []
        inputs = iter(["hello", "again", "/reset", "fresh"])

        def input_fn(_prompt):
            try:
                return next(inputs)
            except StopIteration as exc:
                raise EOFError from exc

        def streamer_factory(_tokenizer):
            streamer = _Streamer()
            streamers.append(streamer)
            return streamer

        generation_calls = []

        def generation_spy(*args, **kwargs):
            generation_calls.append(kwargs.copy())
            return generate_token_ids(*args, **kwargs)

        with mock.patch("scripts.chat_checkpoint.generate_token_ids", side_effect=generation_spy):
            run_repl(
                runtime,
                tokenizer,
                seed=0,
                max_new_tokens=4,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                input_fn=input_fn,
                output_fn=lambda *args, **kwargs: None,
                streamer_factory=streamer_factory,
            )

        self.assertEqual(
            tokenizer.encoded,
            [
                ("User: hello\nAssistant:", False),
                ("\nUser: again\nAssistant:", False),
                ("User: fresh\nAssistant:", False),
            ],
        )
        self.assertEqual(runtime.reset_count, 1)
        self.assertEqual(
            runtime.consumed,
            [
                [10, 11],
                [1],
                [2],
                [10, 11],
                [1],
                [2],
                [10, 11],
                [1],
                [2],
            ],
        )
        self.assertLen(streamers, 3)
        for streamer in streamers:
            self.assertEqual(streamer.tokens, [1])
            self.assertTrue(streamer.ended)
        self.assertLen(generation_calls, 3)
        for call in generation_calls:
            self.assertEqual(call["max_new_tokens"], 4)
            self.assertEqual(call["temperature"], 0.0)
            self.assertEqual(call["top_k"], 0)
            self.assertEqual(call["top_p"], 1.0)
        self.assertFalse(
            np.array_equal(
                np.asarray(jax.random.key_data(generation_calls[0]["key"])),
                np.asarray(jax.random.key_data(generation_calls[1]["key"])),
            )
        )
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(generation_calls[0]["key"])),
            np.asarray(jax.random.key_data(generation_calls[2]["key"])),
        )

    def test_terminal_parsing_and_turn_format(self):
        self.assertEqual(format_user_turn("hello", first_turn=True), "User: hello\nAssistant:")
        self.assertEqual(
            format_user_turn("again", first_turn=False),
            "\nUser: again\nAssistant:",
        )
        self.assertEqual(
            format_user_turn("  first\nsecond  ", first_turn=True),
            "User:   first\nsecond  \nAssistant:",
        )
        self.assertIsNone(parse_layer_limit("all"))
        self.assertEqual(parse_layer_limit("0"), 0)
        self.assertEqual(parse_layer_limit("3"), 3)
        with self.assertRaisesRegex(ValueError, "gdn_layer_limit"):
            parse_layer_limit("-1")
        with self.assertRaisesRegex(ValueError, "gdn_layer_limit"):
            parse_layer_limit("many")


if __name__ == "__main__":
    absltest.main()
