"""HF loader for Qwen3 MoE checkpoints."""

from __future__ import annotations

import gc
import re
from enum import Enum
from typing import Any

import jax.numpy as jnp
import numpy as np
import safetensors
from etils import epath
from flax import nnx

from omegalax.models.params_utils import assign_weights_from_eval_shape, load_hf_config, map_to_bonsai_key, stoi
from .config import Qwen3MoeConfig, make_moe_config
from .model import Qwen3Moe


def _assert_moe_config(cfg: Qwen3MoeConfig, hf_cfg: dict[str, Any]):
    def _require(name, lhs, rhs):
        if lhs != rhs:
            raise ValueError(f"Config mismatch for {name}: expected {lhs}, found {rhs} in HF config")

    _require("vocab_size", cfg.vocab_size, hf_cfg.get("vocab_size"))
    _require("num_layers", cfg.num_layers, hf_cfg.get("num_hidden_layers"))
    _require("emb_dim", cfg.emb_dim, hf_cfg.get("hidden_size"))
    _require("num_heads", cfg.num_heads, hf_cfg.get("num_attention_heads"))
    _require("num_kv_heads", cfg.num_kv_heads, hf_cfg.get("num_key_value_heads", cfg.num_heads))
    hf_num_experts = hf_cfg.get("num_experts") or hf_cfg.get("num_local_experts")
    _require("num_experts", cfg.num_experts, hf_num_experts)
    _require("num_experts_per_tok", cfg.num_experts_per_tok, hf_cfg.get("num_experts_per_tok"))
    _require("moe_intermediate_size", cfg.moe_intermediate_size, hf_cfg.get("moe_intermediate_size"))


_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
_EXPERT_GATE_UP_BATCHED_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj(?:\.weight)?"
)
_EXPERT_DOWN_BATCHED_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.down_proj(?:\.weight)?"
)

_ROUTER_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.weight")


def _get_non_expert_mapping():
    class Transform(Enum):
        LINEAR = ((1, 0), None, False)
        EMBED = None
        SCALE = None

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


def _assign_to_state_dict(state_dict, dotted_key, value, errors, label):
    keys = [stoi(k) for k in dotted_key.split(".")]
    try:
        node: Any = state_dict
        for k in keys[:-1]:
            node = node[k]
        leaf_key = keys[-1]
        target = node[leaf_key]
        if hasattr(target, "shape") and target.shape != value.shape:
            raise ValueError(f"Shape mismatch for '{label}': expected {target.shape}, got {value.shape}")
        target_dtype = getattr(target, "dtype", None)
        if target_dtype is not None:
            value = value.astype(target_dtype)
        node[leaf_key] = value
    except Exception as e:
        errors.append(f"Failed to assign '{label}': {type(e).__name__}: {e}")


def create_qwen3_moe_from_safe_tensors(
    file_dir: str, model_id: str, use_sharding: bool = False,
) -> Qwen3Moe:
    cfg = make_moe_config(model_id, use_sharding=use_sharding)
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    hf_cfg = load_hf_config(epath.Path(file_dir))
    _assert_moe_config(cfg, hf_cfg)

    qwen3 = nnx.eval_shape(lambda: Qwen3Moe(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_non_expert_mapping()
    conversion_errors: list[str] = []
    unmatched_hf_keys: list[str] = []

    E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size

    # Pre-allocate stacked expert weight arrays so that individual expert
    # tensors are written directly into the final buffer, avoiding a second
    # copy during np.stack.
    expert_arrays: dict[tuple[int, str], np.ndarray] = {}
    expert_fill: dict[tuple[int, str], int] = {}
    for layer_idx in range(cfg.num_layers):
        if cfg.is_moe_layer(layer_idx):
            expert_arrays[(layer_idx, "gate_proj")] = np.empty((E, D, F), dtype=np.float32)
            expert_arrays[(layer_idx, "up_proj")] = np.empty((E, D, F), dtype=np.float32)
            expert_arrays[(layer_idx, "down_proj")] = np.empty((E, F, D), dtype=np.float32)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                expert_fill[(layer_idx, proj)] = 0

    router_buf: dict[int, np.ndarray] = {}

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                gate_up_batch_m = _EXPERT_GATE_UP_BATCHED_RE.match(torch_key)
                if gate_up_batch_m:
                    layer_idx = int(gate_up_batch_m.group(1))
                    if (layer_idx, "gate_proj") in expert_arrays:
                        tensor = sf.get_tensor(torch_key)
                        gate, up = np.split(tensor, 2, axis=1)
                        expert_arrays[(layer_idx, "gate_proj")] = np.swapaxes(gate.astype(np.float32), 1, 2)
                        expert_arrays[(layer_idx, "up_proj")] = np.swapaxes(up.astype(np.float32), 1, 2)
                        expert_fill[(layer_idx, "gate_proj")] = cfg.num_experts
                        expert_fill[(layer_idx, "up_proj")] = cfg.num_experts
                        del tensor
                    else:
                        unmatched_hf_keys.append(torch_key)
                    continue

                down_batch_m = _EXPERT_DOWN_BATCHED_RE.match(torch_key)
                if down_batch_m:
                    layer_idx = int(down_batch_m.group(1))
                    if (layer_idx, "down_proj") in expert_arrays:
                        tensor = sf.get_tensor(torch_key)
                        expert_arrays[(layer_idx, "down_proj")] = np.swapaxes(tensor.astype(np.float32), 1, 2)
                        expert_fill[(layer_idx, "down_proj")] = cfg.num_experts
                        del tensor
                    else:
                        unmatched_hf_keys.append(torch_key)
                    continue

                expert_m = _EXPERT_RE.match(torch_key)
                if expert_m:
                    layer_idx = int(expert_m.group(1))
                    expert_idx = int(expert_m.group(2))
                    proj_name = expert_m.group(3)
                    tensor = sf.get_tensor(torch_key)  # [out, in]
                    key = (layer_idx, proj_name)
                    expert_arrays[key][expert_idx] = tensor.T.astype(np.float32)
                    expert_fill[key] += 1
                    del tensor
                    continue

                router_m = _ROUTER_RE.match(torch_key)
                if router_m:
                    layer_idx = int(router_m.group(1))
                    router_buf[layer_idx] = sf.get_tensor(torch_key)
                    continue

                jax_key, transform = map_to_bonsai_key(key_mapping, torch_key)
                if jax_key is None:
                    unmatched_hf_keys.append(torch_key)
                    continue
                keys = [stoi(k) for k in jax_key.split(".")]
                try:
                    assign_weights_from_eval_shape(
                        keys, sf.get_tensor(torch_key), state_dict, torch_key, transform.value,
                    )
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    # Verify all expert slots are filled, convert to JAX, and free the numpy buffer
    keys_to_process = list(expert_arrays.keys())
    for key in keys_to_process:
        layer_idx, proj_name = key
        if expert_fill[key] != E:
            conversion_errors.append(
                f"Layer {layer_idx} {proj_name}: expected {E} experts, got {expert_fill[key]}"
            )
            continue
        stacked_np = expert_arrays.pop(key)
        value = jnp.asarray(stacked_np)
        del stacked_np
        _assign_to_state_dict(
            state_dict, f"layers.{layer_idx}.mlp.{proj_name}",
            value, conversion_errors, f"expert layer {layer_idx} {proj_name}",
        )
        del value
    del expert_arrays
    gc.collect()

    for layer_idx, router_tensor in router_buf.items():
        value = jnp.asarray(router_tensor.T)
        _assign_to_state_dict(
            state_dict, f"layers.{layer_idx}.mlp.router.kernel",
            value, conversion_errors, f"router layer {layer_idx}",
        )
    del router_buf
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
