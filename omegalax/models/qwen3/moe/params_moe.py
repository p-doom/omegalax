"""HF loader for Qwen3 MoE checkpoints."""

from __future__ import annotations

import gc
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import safetensors
from safetensors import numpy as stnp
from etils import epath
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.params_utils import (
    Transform,
    assign_weights_from_eval_shape,
    build_inverse_mapping,
    check_conversion_errors,
    finalize_experts,
    find_safetensors,
    flatten_pure_state,
    handle_moe_key,
    init_expert_buffers,
    inverse_transform,
    load_hf_config,
    map_to_bonsai_key,
    save_hf_config,
    stoi,
    write_moe_experts_to_hf,
)
from omegalax.models.sharding_runtime import apply_sharding_to_model_state as apply_sharding_to_model_state_runtime
from ..sharding import model_state_sharding
from .config import Qwen3MoeConfig, make_moe_config
from .model import Qwen3Moe


def _assert_moe_config(cfg: Qwen3MoeConfig, hf_cfg: dict[str, Any]):
    def _require(name, lhs, rhs):
        if lhs != rhs:
            raise ValueError(f"Config mismatch for {name}: expected {lhs}, found {rhs} in HF config")

    _require("vocab_size", cfg.vocab_size, hf_cfg["vocab_size"])
    _require("num_layers", cfg.num_layers, hf_cfg["num_hidden_layers"])
    _require("emb_dim", cfg.emb_dim, hf_cfg["hidden_size"])
    _require("num_heads", cfg.num_heads, hf_cfg["num_attention_heads"])
    _require("num_kv_heads", cfg.num_kv_heads, hf_cfg["num_key_value_heads"])
    num_experts = hf_cfg["num_experts"] if "num_experts" in hf_cfg else hf_cfg["num_local_experts"]
    _require("num_experts", cfg.num_experts, num_experts)
    _require("num_experts_per_tok", cfg.num_experts_per_tok, hf_cfg["num_experts_per_tok"])
    _require("moe_intermediate_size", cfg.moe_intermediate_size, hf_cfg["moe_intermediate_size"])


def _get_non_expert_mapping():
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
            r"layers.\1.post_attention_layernorm.scale",
            Transform.SCALE,
        ),
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        r"lm_head\.weight": ("lm_head.kernel", Transform.LINEAR),
        # Dense MLP layers (for mlp_only_layers)
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
    }


def create_qwen3_moe_from_safetensors(
    file_dir: str,
    model_id: str,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
) -> Qwen3Moe:
    cfg = make_moe_config(model_id)
    files = find_safetensors(file_dir)

    hf_cfg = load_hf_config(epath.Path(file_dir))
    _assert_moe_config(cfg, hf_cfg)

    qwen3 = nnx.eval_shape(lambda: Qwen3Moe(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_non_expert_mapping()
    unmatched_hf_keys: list[str] = []

    E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size
    expert_arrays, expert_fill = init_expert_buffers(
        cfg.num_layers, E, D, F, cfg.is_moe_layer
    )
    router_buf: dict[int, np.ndarray] = {}

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                if handle_moe_key(
                    torch_key,
                    sf.get_tensor,
                    expert_arrays,
                    expert_fill,
                    router_buf,
                    unmatched_hf_keys,
                    num_experts=E,
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

    finalize_experts(
        expert_arrays,
        expert_fill,
        router_buf,
        state_dict,
        num_experts=E,
        jax_layer_prefix="layers",
    )
    gc.collect()

    check_conversion_errors(unmatched_hf_keys)

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["embedder"]["embedding"].T

    gc.collect()
    model = nnx.merge(graph_def, state_dict)
    return apply_sharding_to_model_state_runtime(
        model,
        cfg.shd_cfg,
        ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size),
        model_state_sharding,
    )


def _make_hf_config_dict(cfg: Qwen3MoeConfig) -> dict[str, Any]:
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
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "norm_topk_prob": cfg.norm_topk_prob,
        "model_type": "qwen3_moe",
    }


def export_qwen3_moe_to_safetensors(model: Qwen3Moe, cfg: Qwen3MoeConfig, out_dir: str | epath.Path) -> epath.Path:
    """Export a Qwen3 MoE nnx model to HuggingFace-style safetensors."""
    if cfg.variant != "moe":
        raise ValueError("export_qwen3_moe_to_safetensors only supports MoE Qwen3 configs.")

    out_dir = epath.Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / "model.safetensors"

    graph_def, abs_state = nnx.split(model)
    pure_state = nnx.to_pure_dict(abs_state)
    flat_state = flatten_pure_state(pure_state)
    inverse_mapping = build_inverse_mapping(_get_non_expert_mapping())

    hf_tensors: dict[str, np.ndarray] = {}
    unmatched: list[str] = []

    expert_params: dict[int, dict[str, np.ndarray]] = {}
    router_params: dict[int, np.ndarray] = {}

    def _handle_special(jax_key: str, value) -> bool:
        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.gate_proj", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            layer_idx = int(m.group(1))
            expert_params.setdefault(layer_idx, {})["gate_proj"] = np.asarray(jax.device_get(value))
            return True

        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.up_proj", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            layer_idx = int(m.group(1))
            expert_params.setdefault(layer_idx, {})["up_proj"] = np.asarray(jax.device_get(value))
            return True

        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.down_proj", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            layer_idx = int(m.group(1))
            expert_params.setdefault(layer_idx, {})["down_proj"] = np.asarray(jax.device_get(value))
            return True

        m = re.fullmatch(r"layers\.([0-9]+)\.mlp\.router\.kernel", jax_key)
        if m and cfg.is_moe_layer(int(m.group(1))):
            layer_idx = int(m.group(1))
            router_params[layer_idx] = np.asarray(jax.device_get(value))
            return True

        return False

    for jax_key, value in flat_state.items():
        if _handle_special(jax_key, value):
            continue

        matched = False
        for jax_regex, hf_template, transform in inverse_mapping:
            m2 = re.fullmatch(jax_regex, jax_key)
            if not m2:
                continue
            hf_key = hf_template.format(*m2.groups())
            arr = np.asarray(jax.device_get(value))
            transform_rule = transform.value if hasattr(transform, "value") else transform
            arr = inverse_transform(arr, transform_rule)
            hf_tensors[hf_key] = arr
            matched = True
            break
        if not matched:
            unmatched.append(jax_key)

    write_moe_experts_to_hf(
        expert_params,
        router_params,
        hf_tensors,
        num_layers=cfg.num_layers,
        is_moe_layer=cfg.is_moe_layer,
        hf_prefix="model.layers",
    )

    if unmatched:
        missing = "\n".join(sorted(unmatched))
        raise RuntimeError(f"Unmapped JAX parameters during export:\n{missing}")

    stnp.save_file(hf_tensors, str(tensor_path))
    save_hf_config(_make_hf_config_dict(cfg), out_dir)

    return tensor_path
