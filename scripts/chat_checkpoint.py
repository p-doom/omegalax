"""Interactive terminal chat for a Qwen3.5 training checkpoint."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from collections.abc import Callable

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, TextStreamer

from omegalax.text import api as text_api
from omegalax.text.chat import (
    ChatMode,
    ChatRuntime,
    GenerationStop,
    StatePassingConfig,
    generate_token_ids,
)
from omegalax.text.checkpoint import resolve_checkpoint, restore_model_params


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "Checkpoint root or numeric Orbax step.")
flags.DEFINE_enum(
    "mode",
    None,
    [mode.value for mode in ChatMode],
    "Chat memory mode.",
)
flags.DEFINE_string("tokenizer", "Qwen/Qwen3.5-0.8B", "Tokenizer name or local path.")
flags.DEFINE_enum(
    "pass_gdn_state",
    None,
    ["true", "false"],
    "Checkpoint GDN state-passing setting.",
)
flags.DEFINE_string(
    "gdn_layer_limit",
    None,
    "Checkpoint GDN layer limit: 'all' or a non-negative integer.",
)
flags.DEFINE_enum(
    "pass_conv_state",
    None,
    ["true", "false"],
    "Checkpoint Conv state-passing setting.",
)
flags.DEFINE_enum(
    "pass_rope_positions",
    None,
    ["true", "false"],
    "Checkpoint RoPE position-passing setting.",
)
flags.DEFINE_integer("max_new_tokens", 512, "Maximum generated tokens per assistant turn.")
flags.DEFINE_float("temperature", 0.0, "Sampling temperature; zero selects greedy decode.")
flags.DEFINE_integer("top_k", 0, "Top-k sampling cutoff; zero disables it.")
flags.DEFINE_float("top_p", 1.0, "Nucleus sampling probability.")
flags.DEFINE_integer("seed", 0, "Sampling and model-initialization seed.")
flags.DEFINE_integer("max_session_tokens", 32768, "Hard token limit for continuous mode.")
flags.DEFINE_string("jax_cache_dir", "/tmp/jax_cache", "JAX compilation cache directory.")


def parse_layer_limit(value: str) -> int | None:
    if value == "all":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError(
            f"gdn_layer_limit must be 'all' or a non-negative integer: {value}"
        ) from exc
    if limit < 0:
        raise ValueError(f"gdn_layer_limit must be non-negative, got {limit}")
    return limit


def format_user_turn(text: str, *, first_turn: bool) -> str:
    prefix = "" if first_turn else "\n"
    return f"{prefix}User: {text}\nAssistant:"


def load_chat(
    checkpoint: str,
    *,
    tokenizer_name: str,
    mode: str | ChatMode,
    state_config: StatePassingConfig,
    max_session_tokens: int,
    seed: int,
):
    resolved = resolve_checkpoint(checkpoint)
    model, cfg = text_api.init_model(
        str(resolved.config_path),
        jax.random.key(seed),
        tp_size=1,
        fsdp_size=1,
        dp_size=1,
    )
    model = restore_model_params(model, resolved)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    runtime = ChatRuntime(
        model,
        cfg,
        mode=mode,
        state_config=state_config,
        segment_length=2048,
        carry_length=128,
        max_session_tokens=max_session_tokens,
    )
    return runtime, tokenizer, resolved


def _default_streamer(tokenizer):
    return TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)


def run_repl(
    runtime: ChatRuntime,
    tokenizer,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = print,
    streamer_factory: Callable = _default_streamer,
) -> None:
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    key = jax.random.key(seed)
    first_turn = True
    while True:
        try:
            user_text = input_fn("User: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return
        if user_text == "/quit":
            return
        if user_text == "/reset":
            runtime.reset()
            key = jax.random.key(seed)
            first_turn = True
            output_fn("[SESSION RESET]")
            continue

        prompt = format_user_turn(user_text, first_turn=first_turn)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        try:
            prompt_logits = runtime.consume(jnp.asarray([prompt_ids], dtype=jnp.int32))
        except ValueError as exc:
            output_fn(f"[ERROR] {exc}")
            continue

        output_fn("Assistant: ", end="", flush=True)
        streamer = streamer_factory(tokenizer)
        result = generate_token_ids(
            runtime,
            initial_logits_V=prompt_logits[0, -1],
            eos_id=int(tokenizer.eos_token_id),
            max_new_tokens=max_new_tokens,
            key=key,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            on_token=lambda token: streamer.put(np.asarray([token], dtype=np.int64)),
        )
        streamer.end()
        key = result.key
        first_turn = False
        if result.stop is GenerationStop.SESSION_LIMIT:
            output_fn(
                f"[SESSION LIMIT] max_session_tokens={runtime.max_session_tokens}; "
                "use /reset to start a new session."
            )


def _as_bool(value: str) -> bool:
    return value == "true"


def main(_) -> None:
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    required_flags = {
        "checkpoint": FLAGS.checkpoint,
        "mode": FLAGS.mode,
        "pass_gdn_state": FLAGS.pass_gdn_state,
        "gdn_layer_limit": FLAGS.gdn_layer_limit,
        "pass_conv_state": FLAGS.pass_conv_state,
        "pass_rope_positions": FLAGS.pass_rope_positions,
    }
    missing = [name for name, value in required_flags.items() if value is None]
    if missing:
        raise ValueError(f"Missing required chat flags: {', '.join(missing)}")
    if jax.device_count() != 1:
        raise ValueError(
            f"Chat requires exactly one visible device, got {jax.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES to one GPU."
        )
    state_config = StatePassingConfig(
        pass_gdn_state=_as_bool(FLAGS.pass_gdn_state),
        gdn_layer_limit=parse_layer_limit(FLAGS.gdn_layer_limit),
        pass_conv_state=_as_bool(FLAGS.pass_conv_state),
        pass_rope_positions=_as_bool(FLAGS.pass_rope_positions),
    )
    runtime, tokenizer, resolved = load_chat(
        FLAGS.checkpoint,
        tokenizer_name=FLAGS.tokenizer,
        mode=FLAGS.mode,
        state_config=state_config,
        max_session_tokens=FLAGS.max_session_tokens,
        seed=FLAGS.seed,
    )
    print(f"Loaded checkpoint step {resolved.step} in {FLAGS.mode} mode.")
    run_repl(
        runtime,
        tokenizer,
        seed=FLAGS.seed,
        max_new_tokens=FLAGS.max_new_tokens,
        temperature=FLAGS.temperature,
        top_k=FLAGS.top_k,
        top_p=FLAGS.top_p,
    )


if __name__ == "__main__":
    app.run(main)
