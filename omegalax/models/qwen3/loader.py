"""Unified HF loader for Qwen3 checkpoints (dense + MoE)."""

from __future__ import annotations

import gc
from typing import Any

import jax.numpy as jnp
import numpy as np
import safetensors
from etils import epath
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.params_utils import (
    Transform,
    assign_weights_from_eval_shape,
    check_conversion_errors,
    finalize_experts,
    find_safetensors,
    handle_moe_key,
    init_expert_buffers,
    load_hf_config,
    map_to_bonsai_key,
    stoi,
)
from .config import Qwen3Config, make_config_from_hf
from .model import Qwen3


def _assert_config(cfg: Qwen3Config, hf_cfg: dict[str, Any]):
    def _require(name, lhs, rhs):
        if lhs != rhs:
            raise ValueError(f"Config mismatch for {name}: expected {lhs}, found {rhs} in HF config")

    _require("vocab_size", cfg.vocab_size, hf_cfg["vocab_size"])
    _require("num_layers", cfg.num_layers, hf_cfg["num_hidden_layers"])
    _require("tie_word_embeddings", cfg.tie_word_embeddings, hf_cfg["tie_word_embeddings"])
    _require("emb_dim", cfg.emb_dim, hf_cfg["hidden_size"])
    _require("num_heads", cfg.num_heads, hf_cfg["num_attention_heads"])
    _require("num_kv_heads", cfg.num_kv_heads, hf_cfg["num_key_value_heads"])
    _require("head_dim", cfg.head_dim, hf_cfg["head_dim"])
    _require("mlp_dim", cfg.mlp_dim, hf_cfg["intermediate_size"])
    _require("norm_eps", cfg.norm_eps, hf_cfg["rms_norm_eps"])

    rope_params = hf_cfg.get("rope_parameters")
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        rope_theta = rope_params["rope_theta"]
    else:
        rope_theta = hf_cfg.get("rope_theta")
    if rope_theta is None:
        raise ValueError("Missing 'rope_parameters.rope_theta' (or legacy top-level 'rope_theta') in HF config")
    _require("rope_theta", cfg.rope_theta, rope_theta)

    if cfg.is_moe:
        num_experts = hf_cfg.get("num_experts", hf_cfg.get("num_local_experts"))
        if num_experts is None:
            raise ValueError("Missing 'num_experts' (or alias 'num_local_experts') in HF config")
        _require("num_experts", cfg.num_experts, num_experts)
        _require("num_experts_per_tok", cfg.num_experts_per_tok, hf_cfg["num_experts_per_tok"])
        _require("moe_intermediate_size", cfg.moe_intermediate_size, hf_cfg["moe_intermediate_size"])


def _get_key_mapping():
    """HF -> JAX key mapping for all non-expert parameters."""
    return {
        r"model\.embed_tokens\.weight": ("embedder.embedding", Transform.EMBED),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (r"layers.\1.attn.q_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (r"layers.\1.attn.k_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale", Transform.SCALE,
        ),
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
        r"lm_head\.weight": ("lm_head.kernel", Transform.LINEAR),
    }


def create_qwen3_from_safetensors(
    file_dir: str,
    model_id: str = "",
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> Qwen3:
    """Load HuggingFace Qwen3 safetensors (dense or MoE) into a JAX model."""
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    files = find_safetensors(file_dir)

    hf_cfg = load_hf_config(epath.Path(file_dir))
    cfg = make_config_from_hf(hf_cfg)
    _assert_config(cfg, hf_cfg)

    with mesh_rules(mesh):
        model = nnx.eval_shape(lambda: Qwen3(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_key_mapping()
    unmatched_hf_keys: list[str] = []

    if cfg.is_moe:
        E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size
        expert_arrays, expert_fill = init_expert_buffers(
            cfg.num_layers, E, D, F, cfg.is_moe_layer
        )
        router_buf: dict[int, np.ndarray] = {}
    else:
        expert_arrays = expert_fill = router_buf = None

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                if cfg.is_moe and handle_moe_key(
                    torch_key, sf.get_tensor, expert_arrays, expert_fill,
                    router_buf, unmatched_hf_keys, num_experts=cfg.num_experts,
                    hf_prefix="model.layers",
                ):
                    continue

                jax_key, transform = map_to_bonsai_key(key_mapping, torch_key)
                if jax_key is None:
                    unmatched_hf_keys.append(torch_key)
                    continue
                keys = [stoi(k) for k in jax_key.split(".")]
                assign_weights_from_eval_shape(
                    keys, sf.get_tensor(torch_key), state_dict, torch_key, transform.value,
                )
        gc.collect()

    if cfg.is_moe:
        finalize_experts(
            expert_arrays, expert_fill, router_buf, state_dict,
            num_experts=cfg.num_experts, jax_layer_prefix="layers",
        )
    gc.collect()

    check_conversion_errors(unmatched_hf_keys)

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["embedder"]["embedding"].T

    gc.collect()
    model = nnx.merge(graph_def, state_dict)
    return model
