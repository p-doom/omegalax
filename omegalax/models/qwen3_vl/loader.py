"""Weight conversion from HuggingFace Qwen3-VL safetensors to JAX."""

from __future__ import annotations

import gc
import re
from typing import Any

import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx

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
from .config import Qwen3VLConfig, make_vl_config, make_vl_config_from_hf
from .model import Qwen3VL


def _assert_vl_config(cfg: Qwen3VLConfig, hf_cfg: dict):
    txt = hf_cfg["text_config"]
    vis = hf_cfg["vision_config"]
    rope_params = txt.get("rope_parameters") or txt.get("rope_scaling") or {}

    def _require(name, lhs, rhs):
        if lhs != rhs:
            raise ValueError(f"Config mismatch for {name}: expected {lhs}, found {rhs} in HF config")

    _require("vocab_size", cfg.vocab_size, txt["vocab_size"])
    _require("num_layers", cfg.num_layers, txt["num_hidden_layers"])
    _require("emb_dim", cfg.emb_dim, txt["hidden_size"])
    _require("num_heads", cfg.num_heads, txt["num_attention_heads"])
    _require("num_kv_heads", cfg.num_kv_heads, txt["num_key_value_heads"])
    _require("head_dim", cfg.head_dim, txt["head_dim"])
    _require("mlp_dim", cfg.mlp_dim, txt["intermediate_size"])
    _require("rope_theta", cfg.rope_theta, rope_params.get("rope_theta") or txt["rope_theta"])
    _require("mrope_section", tuple(cfg.mrope_section), tuple(rope_params["mrope_section"]))

    # MoE fields are absent in dense HF configs.
    _require("num_experts", cfg.num_experts, txt.get("num_experts") or txt.get("num_local_experts", 0))
    _require("num_experts_per_tok", cfg.num_experts_per_tok, txt.get("num_experts_per_tok", 0))
    _require("moe_intermediate_size", cfg.moe_intermediate_size, txt.get("moe_intermediate_size", 0))

    _require("vision.hidden_size", cfg.vision.hidden_size, vis["hidden_size"])
    _require("vision.intermediate_size", cfg.vision.intermediate_size, vis["intermediate_size"])
    _require("vision.num_heads", cfg.vision.num_heads, vis["num_heads"])
    _require("vision.depth", cfg.vision.depth, vis["depth"])
    _require("vision.patch_size", cfg.vision.patch_size, vis["patch_size"])
    _require("vision.temporal_patch_size", cfg.vision.temporal_patch_size, vis["temporal_patch_size"])
    _require("vision.spatial_merge_size", cfg.vision.spatial_merge_size, vis["spatial_merge_size"])
    _require("vision.out_hidden_size", cfg.vision.out_hidden_size, vis["out_hidden_size"])
    _require("vision.num_position_embeddings", cfg.vision.num_position_embeddings, vis["num_position_embeddings"])


def _get_vision_key_mapping():
    T = Transform
    m: dict[str, tuple[str, Transform]] = {}
    # Patch embedding
    m[r"model\.visual\.patch_embed\.proj\.weight"] = ("vision.patch_embed.proj.kernel", T.CONV3D)
    m[r"model\.visual\.patch_embed\.proj\.bias"] = ("vision.patch_embed.proj.bias", T.BIAS)
    # Position embedding
    m[r"model\.visual\.pos_embed\.weight"] = ("vision.pos_embed.embedding", T.EMBED)
    # Vision: blocks
    b = r"model\.visual\.blocks\.([0-9]+)"
    m[b + r"\.norm1\.weight"] = (r"vision.blocks.\1.norm1.scale", T.SCALE)
    m[b + r"\.norm1\.bias"] = (r"vision.blocks.\1.norm1.bias", T.BIAS)
    m[b + r"\.attn\.qkv\.weight"] = (r"vision.blocks.\1.attn.qkv.kernel", T.LINEAR)
    m[b + r"\.attn\.qkv\.bias"] = (r"vision.blocks.\1.attn.qkv.bias", T.BIAS)
    m[b + r"\.attn\.proj\.weight"] = (r"vision.blocks.\1.attn.proj.kernel", T.LINEAR)
    m[b + r"\.attn\.proj\.bias"] = (r"vision.blocks.\1.attn.proj.bias", T.BIAS)
    m[b + r"\.norm2\.weight"] = (r"vision.blocks.\1.norm2.scale", T.SCALE)
    m[b + r"\.norm2\.bias"] = (r"vision.blocks.\1.norm2.bias", T.BIAS)
    m[b + r"\.mlp\.linear_fc1\.weight"] = (r"vision.blocks.\1.mlp.fc1.kernel", T.LINEAR)
    m[b + r"\.mlp\.linear_fc1\.bias"] = (r"vision.blocks.\1.mlp.fc1.bias", T.BIAS)
    m[b + r"\.mlp\.linear_fc2\.weight"] = (r"vision.blocks.\1.mlp.fc2.kernel", T.LINEAR)
    m[b + r"\.mlp\.linear_fc2\.bias"] = (r"vision.blocks.\1.mlp.fc2.bias", T.BIAS)
    # Merger
    m[r"model\.visual\.merger\.norm\.weight"] = ("vision.merger.norm.scale", T.SCALE)
    m[r"model\.visual\.merger\.norm\.bias"] = ("vision.merger.norm.bias", T.BIAS)
    m[r"model\.visual\.merger\.linear_fc1\.weight"] = ("vision.merger.fc1.kernel", T.LINEAR)
    m[r"model\.visual\.merger\.linear_fc1\.bias"] = ("vision.merger.fc1.bias", T.BIAS)
    m[r"model\.visual\.merger\.linear_fc2\.weight"] = ("vision.merger.fc2.kernel", T.LINEAR)
    m[r"model\.visual\.merger\.linear_fc2\.bias"] = ("vision.merger.fc2.bias", T.BIAS)
    # Deepstack mergers
    d = r"model\.visual\.deepstack_merger_list\.([0-9]+)"
    m[d + r"\.norm\.weight"] = (r"vision.deepstack_mergers.\1.norm.scale", T.SCALE)
    m[d + r"\.norm\.bias"] = (r"vision.deepstack_mergers.\1.norm.bias", T.BIAS)
    m[d + r"\.linear_fc1\.weight"] = (r"vision.deepstack_mergers.\1.fc1.kernel", T.LINEAR)
    m[d + r"\.linear_fc1\.bias"] = (r"vision.deepstack_mergers.\1.fc1.bias", T.BIAS)
    m[d + r"\.linear_fc2\.weight"] = (r"vision.deepstack_mergers.\1.fc2.kernel", T.LINEAR)
    m[d + r"\.linear_fc2\.bias"] = (r"vision.deepstack_mergers.\1.fc2.bias", T.BIAS)
    return m


def _get_text_key_mapping():
    T = Transform
    m: dict[str, tuple[str, Transform]] = {}
    m[r"model\.language_model\.embed_tokens\.weight"] = ("text.embedder.embedding", T.EMBED)
    l = r"model\.language_model\.layers\.([0-9]+)"
    m[l + r"\.self_attn\.q_proj\.weight"] = (r"text.layers.\1.attn.q_proj.kernel", T.LINEAR)
    m[l + r"\.self_attn\.k_proj\.weight"] = (r"text.layers.\1.attn.k_proj.kernel", T.LINEAR)
    m[l + r"\.self_attn\.v_proj\.weight"] = (r"text.layers.\1.attn.v_proj.kernel", T.LINEAR)
    m[l + r"\.self_attn\.o_proj\.weight"] = (r"text.layers.\1.attn.o_proj.kernel", T.LINEAR)
    m[l + r"\.self_attn\.q_norm\.weight"] = (r"text.layers.\1.attn.q_norm.scale", T.SCALE)
    m[l + r"\.self_attn\.k_norm\.weight"] = (r"text.layers.\1.attn.k_norm.scale", T.SCALE)
    m[l + r"\.mlp\.gate_proj\.weight"] = (r"text.layers.\1.mlp.gate_proj.kernel", T.LINEAR)
    m[l + r"\.mlp\.up_proj\.weight"] = (r"text.layers.\1.mlp.up_proj.kernel", T.LINEAR)
    m[l + r"\.mlp\.down_proj\.weight"] = (r"text.layers.\1.mlp.down_proj.kernel", T.LINEAR)
    m[l + r"\.input_layernorm\.weight"] = (r"text.layers.\1.input_layernorm.scale", T.SCALE)
    m[l + r"\.post_attention_layernorm\.weight"] = (r"text.layers.\1.post_attention_layernorm.scale", T.SCALE)
    m[r"model\.language_model\.norm\.weight"] = ("text.final_norm.scale", T.SCALE)
    m[r"lm_head\.weight"] = ("lm_head.kernel", T.LINEAR)
    return m


def _get_non_expert_mapping():
    """Mapping for all non-expert parameters (vision + dense text path)."""
    mapping = {}
    mapping.update(_get_vision_key_mapping())
    mapping.update(_get_text_key_mapping())
    return mapping


def create_qwen3_vl_from_safetensors(file_dir: str, model_id: str = "", use_sharding: bool = False) -> tuple[Qwen3VL, Qwen3VLConfig]:
    """Load HuggingFace Qwen3-VL safetensors into a JAX Qwen3-VL model."""
    path = epath.Path(file_dir).expanduser()
    files = find_safetensors(file_dir)

    hf_cfg = load_hf_config(path)
    if model_id:
        cfg = make_vl_config(model_id)
        _assert_vl_config(cfg, hf_cfg)
    else:
        cfg = make_vl_config_from_hf(hf_cfg)

    model = nnx.eval_shape(lambda: Qwen3VL(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)

    non_expert_mapping = _get_non_expert_mapping()
    unmatched_hf_keys: list[str] = []

    is_moe = cfg.num_experts > 0
    if is_moe:
        E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size
        expert_arrays, expert_fill = init_expert_buffers(
            cfg.num_layers, E, D, F, cfg.is_moe_layer
        )
        router_buf: dict[int, Any] = {}
    else:
        expert_arrays = {}
        expert_fill = {}
        router_buf = {}

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                if is_moe and handle_moe_key(
                    torch_key,
                    sf.get_tensor,
                    expert_arrays,
                    expert_fill,
                    router_buf,
                    unmatched_hf_keys,
                    num_experts=cfg.num_experts,
                    hf_prefix="model.language_model.layers",
                ):
                    continue

                tensor = sf.get_tensor(torch_key)
                jax_key, transform = map_to_bonsai_key(non_expert_mapping, torch_key)
                if jax_key is None:
                    unmatched_hf_keys.append(torch_key)
                    continue

                keys = [stoi(k) for k in jax_key.split(".")]
                if transform == Transform.CONV3D:
                    tensor = tensor.reshape(tensor.shape[0], -1).T
                    assign_weights_from_eval_shape(keys, tensor, state_dict, torch_key, None)
                else:
                    transform_value = transform.value if transform not in (Transform.BIAS, Transform.EMBED, Transform.SCALE) else None
                    assign_weights_from_eval_shape(keys, tensor, state_dict, torch_key, transform_value)
        gc.collect()

    if is_moe:
        finalize_experts(
            expert_arrays,
            expert_fill,
            router_buf,
            state_dict,
            num_experts=cfg.num_experts,
            jax_layer_prefix="text.layers",
        )

    check_conversion_errors(unmatched_hf_keys)

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["text"]["embedder"]["embedding"].T

    gc.collect()
    return nnx.merge(graph_def, state_dict), cfg
