"""Stateful incremental runtime for terminal chat."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from flax import nnx
import jax
import jax.numpy as jnp

from omegalax.models.qwen3_5.cache import carry_cache, reset_cache
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.text import api as text_api


class ChatMode(StrEnum):
    TRAINING_RESET = "training_reset"
    CARRY_128 = "carry_128"
    CONTINUOUS = "continuous"


class GenerationStop(StrEnum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    SESSION_LIMIT = "session_limit"


@dataclass(frozen=True)
class StatePassingConfig:
    pass_gdn_state: bool
    gdn_layer_limit: int | None
    pass_conv_state: bool
    pass_rope_positions: bool


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    key: jax.Array
    stop: GenerationStop


def sample_token(
    logits_V: jax.Array,
    key: jax.Array,
    *,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[int, jax.Array]:
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    logits_V = jnp.asarray(logits_V, dtype=jnp.float32)
    if logits_V.ndim != 1:
        raise ValueError(f"Expected one-dimensional logits, got {logits_V.shape}")

    sample_key, next_key = jax.random.split(key)
    if temperature == 0:
        return int(jax.device_get(jnp.argmax(logits_V))), next_key

    filtered = logits_V / temperature
    if top_k > 0:
        k = min(top_k, filtered.shape[0])
        threshold = jax.lax.top_k(filtered, k)[0][-1]
        filtered = jnp.where(filtered >= threshold, filtered, -jnp.inf)
    if top_p < 1:
        sorted_indices = jnp.argsort(filtered)[::-1]
        sorted_logits = filtered[sorted_indices]
        sorted_probs = jax.nn.softmax(sorted_logits)
        keep = jnp.cumsum(sorted_probs) - sorted_probs < top_p
        sorted_logits = jnp.where(keep, sorted_logits, -jnp.inf)
        sorted_token = jax.random.categorical(sample_key, sorted_logits)
        token = sorted_indices[sorted_token]
    else:
        token = jax.random.categorical(sample_key, filtered)
    return int(jax.device_get(token)), next_key


def generate_token_ids(
    runtime,
    *,
    initial_logits_V: jax.Array,
    eos_id: int,
    max_new_tokens: int,
    key: jax.Array,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
    on_token: Callable[[int], None] | None = None,
) -> GenerationResult:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    logits_V = initial_logits_V
    generated = []
    for _ in range(max_new_tokens):
        if (
            runtime.mode is ChatMode.CONTINUOUS
            and runtime.total_tokens >= runtime.max_session_tokens
        ):
            return GenerationResult(tuple(generated), key, GenerationStop.SESSION_LIMIT)
        token, key = sample_token(
            logits_V,
            key,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        next_logits = runtime.consume(jnp.asarray([[token]], dtype=jnp.int32))
        session_exhausted = (
            runtime.mode is ChatMode.CONTINUOUS
            and runtime.total_tokens >= runtime.max_session_tokens
        )
        if token == eos_id:
            stop = GenerationStop.SESSION_LIMIT if session_exhausted else GenerationStop.EOS
            return GenerationResult(tuple(generated), key, stop)
        generated.append(token)
        if on_token is not None:
            on_token(token)
        if session_exhausted:
            return GenerationResult(tuple(generated), key, GenerationStop.SESSION_LIMIT)
        logits_V = next_logits[0, -1]
    return GenerationResult(tuple(generated), key, GenerationStop.MAX_NEW_TOKENS)


@nnx.jit
def _forward_cached(
    model: Qwen3_5ForCausalLM,
    cache,
    token_ids_BT: jax.Array,
    position_ids_ZBT: jax.Array,
    gdn_states,
    conv_states,
):
    segment_ids_BT = jnp.ones_like(token_ids_BT, dtype=jnp.int32)
    hidden_BTD, _, final_gdn, final_conv = model(
        token_ids_BT,
        segment_ids_BT,
        cache,
        jnp.array(0, dtype=jnp.int32),
        gdn_initial_states=gdn_states,
        conv_initial_states=conv_states,
        position_ids_ZBT=position_ids_ZBT,
        return_gdn_states=True,
        return_conv_states=True,
    )
    logits_BTV = jnp.einsum("BTD,DV->BTV", hidden_BTD, model.output_weight())
    return logits_BTV, final_gdn, final_conv


def _print_event(message: str) -> None:
    print(message, flush=True)


def _select_boundary_states(states, *, pass_state: bool, layer_limit: int | None):
    if not pass_state or layer_limit == 0:
        return None
    if layer_limit is None:
        return states
    return tuple(
        state if idx < layer_limit else jnp.zeros_like(state) for idx, state in enumerate(states)
    )


class ChatRuntime:
    def __init__(
        self,
        model: Qwen3_5ForCausalLM,
        cfg: Qwen3_5TextConfig,
        *,
        mode: ChatMode,
        state_config: StatePassingConfig,
        segment_length: int = 2048,
        carry_length: int = 128,
        max_session_tokens: int = 32768,
        event_sink: Callable[[str], None] = _print_event,
    ):
        mode = ChatMode(mode)
        if mode not in (
            ChatMode.TRAINING_RESET,
            ChatMode.CARRY_128,
            ChatMode.CONTINUOUS,
        ):
            raise ValueError(f"Unsupported chat mode: {mode}")
        if segment_length <= 0:
            raise ValueError(f"segment_length must be > 0, got {segment_length}")
        if mode is ChatMode.CARRY_128 and carry_length <= 0:
            raise ValueError(f"carry_length must be > 0, got {carry_length}")
        if mode is ChatMode.CONTINUOUS and max_session_tokens <= 0:
            raise ValueError(f"max_session_tokens must be > 0, got {max_session_tokens}")
        linear_layers = sum(layer_type != "full_attention" for layer_type in cfg.layer_types)
        if state_config.gdn_layer_limit is not None and not (
            0 <= state_config.gdn_layer_limit <= linear_layers
        ):
            raise ValueError(
                f"gdn_layer_limit must be in [0, {linear_layers}], "
                f"got {state_config.gdn_layer_limit}"
            )

        self.model = model
        self.cfg = cfg
        self.mode = mode
        self.state_config = state_config
        self.segment_length = segment_length
        self.carry_length = carry_length
        self.max_session_tokens = max_session_tokens
        self.event_sink = event_sink
        cache_length = (
            min(segment_length, max_session_tokens)
            if mode is ChatMode.CONTINUOUS
            else segment_length + (carry_length if mode is ChatMode.CARRY_128 else 0)
        )
        self.initial_cache_length = cache_length
        self.cache = text_api.make_cache(
            cfg,
            batch_size=1,
            token_len=cache_length,
            generate_steps=0,
            dtype=cfg.dtype,
        )
        self.gdn_states = None
        self.conv_states = None
        self.segment_tokens = 0
        self.completed_segments = 0
        self.total_tokens = 0

    def consume(self, token_ids_BT: jax.Array) -> jax.Array:
        token_ids_BT = jnp.asarray(token_ids_BT, dtype=jnp.int32)
        if token_ids_BT.ndim != 2 or token_ids_BT.shape[0] != 1:
            raise ValueError(
                f"ChatRuntime expects token ids with shape (1, T), got {token_ids_BT.shape}"
            )
        if token_ids_BT.shape[1] == 0:
            return jnp.empty((1, 0, self.cfg.vocab_size), dtype=self.cfg.dtype)
        if self.mode is ChatMode.CONTINUOUS:
            requested_total = self.total_tokens + token_ids_BT.shape[1]
            if requested_total > self.max_session_tokens:
                raise ValueError(
                    f"Consuming {token_ids_BT.shape[1]} tokens would exceed "
                    f"max_session_tokens={self.max_session_tokens}; "
                    f"current total is {self.total_tokens}."
                )
            self._grow_continuous_cache(requested_total)
            positions_BT = jnp.arange(
                self.total_tokens,
                requested_total,
                dtype=jnp.int32,
            )[None, :]
            position_ids_ZBT = jnp.stack([positions_BT] * 3, axis=0)
            logits, self.gdn_states, self.conv_states = _forward_cached(
                self.model,
                self.cache,
                token_ids_BT,
                position_ids_ZBT,
                self.gdn_states,
                self.conv_states,
            )
            self.total_tokens = requested_total
            return logits

        logits = []
        offset = 0
        while offset < token_ids_BT.shape[1]:
            block_length = min(
                self.segment_length - self.segment_tokens,
                token_ids_BT.shape[1] - offset,
            )
            block = token_ids_BT[:, offset : offset + block_length]
            position_start = (
                self.total_tokens if self.state_config.pass_rope_positions else self.segment_tokens
            )
            positions_BT = jnp.arange(
                position_start,
                position_start + block_length,
                dtype=jnp.int32,
            )[None, :]
            position_ids_ZBT = jnp.stack([positions_BT] * 3, axis=0)
            block_logits, self.gdn_states, self.conv_states = _forward_cached(
                self.model,
                self.cache,
                block,
                position_ids_ZBT,
                self.gdn_states,
                self.conv_states,
            )
            logits.append(block_logits)
            offset += block_length
            self.segment_tokens += block_length
            self.total_tokens += block_length

            if self.segment_tokens == self.segment_length:
                self._finish_segment()

        return jnp.concatenate(logits, axis=1)

    def _grow_continuous_cache(self, required_tokens: int) -> None:
        current_size = next(
            layer_cache.size for layer_cache in self.cache if layer_cache is not None
        )
        if required_tokens <= current_size:
            return
        new_size = current_size
        while new_size < required_tokens:
            new_size = min(self.max_session_tokens, new_size * 2)
        new_cache = text_api.make_cache(
            self.cfg,
            batch_size=1,
            token_len=new_size,
            generate_steps=0,
            dtype=self.cfg.dtype,
        )
        for old_layer, new_layer in zip(self.cache, new_cache, strict=True):
            if old_layer is None:
                if new_layer is not None:
                    raise ValueError("Cache layer structure diverged")
                continue
            if new_layer is None:
                raise ValueError("Cache layer structure diverged")
            valid_tokens = int(old_layer.cur_ind[...])
            new_layer.k_cache[:, :valid_tokens] = old_layer.k_cache[:, :valid_tokens]
            new_layer.v_cache[:, :valid_tokens] = old_layer.v_cache[:, :valid_tokens]
            new_layer.cur_ind[...] = jnp.array(valid_tokens, dtype=jnp.int32)
        self.cache = new_cache

    def _finish_segment(self) -> None:
        self.gdn_states = _select_boundary_states(
            self.gdn_states,
            pass_state=self.state_config.pass_gdn_state,
            layer_limit=self.state_config.gdn_layer_limit,
        )
        self.conv_states = _select_boundary_states(
            self.conv_states,
            pass_state=self.state_config.pass_conv_state,
            layer_limit=self.state_config.gdn_layer_limit,
        )
        self.segment_tokens = 0
        self.completed_segments += 1
        if self.mode is ChatMode.TRAINING_RESET:
            reset_cache(self.cache)
            self.event_sink(
                f"[KV RESET] completed_segment={self.completed_segments} "
                f"consumed_tokens={self.segment_length}"
            )
        else:
            carried_tokens = carry_cache(self.cache, self.carry_length)
            self.event_sink(
                f"[KV CARRY] completed_segment={self.completed_segments} "
                f"carried_tokens={carried_tokens}"
            )

    def reset(self) -> None:
        if self.mode is ChatMode.CONTINUOUS:
            self.cache = text_api.make_cache(
                self.cfg,
                batch_size=1,
                token_len=self.initial_cache_length,
                generate_steps=0,
                dtype=self.cfg.dtype,
            )
        else:
            reset_cache(self.cache)
        self.gdn_states = None
        self.conv_states = None
        self.segment_tokens = 0
        self.completed_segments = 0
        self.total_tokens = 0
