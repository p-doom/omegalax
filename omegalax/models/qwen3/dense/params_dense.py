from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import safetensors
from safetensors import numpy as stnp
from etils import epath
from flax import nnx

from omegalax.models.params_utils import (
    Transform,
    assign_weights_from_eval_shape,
    build_inverse_mapping,
    check_conversion_errors,
    find_safetensors,
    flatten_pure_state,
    inverse_transform,
    load_hf_config,
    map_to_bonsai_key,
    save_hf_config,
    stoi,
)

from .config import Qwen3Config, make_dense_config
from .model import Qwen3Dense


def assert_dense_config(cfg: Qwen3Config, hf_cfg: dict[str, Any]):
    _require("vocab_size", cfg.vocab_size, hf_cfg["vocab_size"])
    _require("num_layers", cfg.num_layers, hf_cfg["num_hidden_layers"])
    _require("tie_word_embeddings", cfg.tie_word_embeddings, hf_cfg["tie_word_embeddings"])
    _require("emb_dim", cfg.emb_dim, hf_cfg["hidden_size"])
    _require("num_heads", cfg.num_heads, hf_cfg["num_attention_heads"])
    _require("num_kv_heads", cfg.num_kv_heads, hf_cfg["num_key_value_heads"])
    _require("head_dim", cfg.head_dim, hf_cfg["head_dim"])
    _require("mlp_dim", cfg.mlp_dim, hf_cfg["intermediate_size"])
    rope_params = hf_cfg.get("rope_parameters") or hf_cfg.get("rope_scaling") or {}
    _require("rope_theta", cfg.rope_theta, rope_params.get("rope_theta") or hf_cfg.get("rope_theta"))


def _get_key_and_transform_mapping(cfg):
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


def create_qwen3_dense_from_safetensors(file_dir: str, model_id: str, use_sharding: bool = False) -> Qwen3Dense:
    cfg = make_dense_config(model_id, use_sharding=use_sharding)
    files = find_safetensors(file_dir)

    hf_cfg = load_hf_config(epath.Path(file_dir))
    assert_dense_config(cfg, hf_cfg)

    qwen3 = nnx.eval_shape(lambda: Qwen3Dense(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_key_and_transform_mapping(cfg)
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
                assign_weights_from_eval_shape(keys, tensor, state_dict, torch_key, transform.value)
        gc.collect()

    check_conversion_errors(unmatched_hf_keys)

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["embedder"]["embedding"].T

    gc.collect()
    return nnx.merge(graph_def, state_dict)


def _make_hf_config_dict(cfg: Qwen3Config) -> dict[str, Any]:
    return {
        "vocab_size": cfg.vocab_size,
        "num_hidden_layers": cfg.num_layers,
        "hidden_size": cfg.emb_dim,
        "num_attention_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.mlp_dim,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "model_type": "qwen3",
    }


def export_qwen3_dense_to_safetensors(model: Qwen3Dense, cfg: Qwen3Config, out_dir: str | Path) -> Path:
    """Export a dense Qwen3 nnx model to HuggingFace-style safetensors."""
    if cfg.variant != "dense":
        raise ValueError("export_qwen3_dense_to_safetensors only supports dense Qwen3 configs.")

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / "model.safetensors"

    graph_def, abs_state = nnx.split(model)
    pure_state = nnx.to_pure_dict(abs_state)
    flat_state = flatten_pure_state(pure_state)
    inverse_mapping = build_inverse_mapping(_get_key_and_transform_mapping(cfg))

    hf_tensors: dict[str, np.ndarray] = {}
    unmapped: list[str] = []

    for jax_key, value in flat_state.items():
        matched = False
        for jax_regex, hf_template, transform in inverse_mapping:
            m = re.fullmatch(jax_regex, jax_key)
            if not m:
                continue
            hf_key = hf_template.format(*m.groups())
            arr = np.asarray(jax.device_get(value))
            transform_rule = transform.value if hasattr(transform, "value") else transform
            arr = inverse_transform(arr, transform_rule)
            hf_tensors[hf_key] = arr
            matched = True
            break
        if not matched:
            unmapped.append(jax_key)

    if unmapped:
        missing = "\n".join(sorted(unmapped))
        raise RuntimeError(f"Unmapped JAX parameters during export:\n{missing}")

    stnp.save_file(hf_tensors, tensor_path)
    save_hf_config(_make_hf_config_dict(cfg), out_dir)

    return tensor_path
