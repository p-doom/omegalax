from __future__ import annotations

import gc
from enum import Enum
from typing import Any

import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx

from omegalax.models.params_utils import assign_weights_from_eval_shape, load_hf_config, map_to_bonsai_key, stoi

from .config import Qwen3Config, make_dense_config
from .model import Qwen3Dense


def _get_key_and_transform_mapping(cfg):
    class Transform(Enum):
        BIAS = None
        # For Linear (2D) weights, HF stores (out_dim, in_dim); we transpose to (in_dim, out_dim).
        LINEAR = ((1, 0), None, False)
        EMBED = None
        SCALE = None

    return {
        r"model\.embed_tokens\.weight": ("embedder.embedding", Transform.EMBED),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (r"layers.\1.attn.q_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (r"layers.\1.attn.k_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale",
            Transform.SCALE,
        ),
        r"lm_head\.weight": ("lm_head.kernel", Transform.LINEAR),
    }


def _require(name: str, lhs: Any, rhs: Any):
    if lhs != rhs:
        raise ValueError(f"Config mismatch for {name}: expected {lhs}, found {rhs} in HF config")


def assert_dense_config(cfg: Qwen3Config, hf_cfg: dict[str, Any]):
    _require("vocab_size", cfg.vocab_size, hf_cfg.get("vocab_size"))
    _require("num_layers", cfg.num_layers, hf_cfg.get("num_hidden_layers"))
    _require("tie_word_embeddings", cfg.tie_word_embeddings, hf_cfg.get("tie_word_embeddings"))

    hf_emb = hf_cfg.get("hidden_size")
    if hf_emb is not None:
        _require("emb_dim", cfg.emb_dim, hf_emb)

    hf_heads = hf_cfg.get("num_attention_heads")
    hf_kv_heads = hf_cfg.get("num_key_value_heads", hf_heads)
    hf_head_dim = hf_cfg.get("head_dim")
    if hf_head_dim is None and hf_emb is not None and hf_heads is not None and hf_heads > 0:
        hf_head_dim = hf_emb // hf_heads
    if hf_head_dim is not None:
        _require("head_dim", cfg.head_dim, hf_head_dim)
    if hf_heads is not None:
        _require("num_heads", cfg.num_heads, hf_heads)
    if hf_kv_heads is not None:
        _require("num_kv_heads", cfg.num_kv_heads, hf_kv_heads)

    hf_mlp = hf_cfg.get("intermediate_size")
    if hf_mlp is not None:
        _require("mlp_dim", cfg.mlp_dim, hf_mlp)

    hf_rope_theta = hf_cfg.get("rope_theta") or hf_cfg.get("rotary_emb_base")
    if hf_rope_theta is not None:
        _require("rope_theta", cfg.rope_theta, hf_rope_theta)


def create_qwen3_dense_from_safe_tensors(file_dir: str, model_id: str, use_sharding: bool = False) -> Qwen3Dense:
    cfg = make_dense_config(model_id, use_sharding=use_sharding)
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    hf_cfg = load_hf_config(epath.Path(file_dir))
    assert_dense_config(cfg, hf_cfg)

    qwen3 = nnx.eval_shape(lambda: Qwen3Dense(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    unmatched_hf_keys: list[str] = []

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)
                jax_key, transform = map_to_bonsai_key(key_mapping, torch_key)
                if jax_key is None:
                    unmatched_hf_keys.append(torch_key)
                    continue
                keys = [stoi(k) for k in jax_key.split(".")]
                try:
                    assign_weights_from_eval_shape(keys, tensor, state_dict, torch_key, transform.value)
                except Exception as e:  # pylint: disable=broad-except
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    if unmatched_hf_keys:
        unmatched = "\n".join(sorted(unmatched_hf_keys))
        raise RuntimeError(f"Unmapped HuggingFace parameters:\n{unmatched}")

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["embedder"]["embedding"].T

    gc.collect()
    return nnx.merge(graph_def, state_dict)
