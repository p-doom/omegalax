"""Weight conversion from HuggingFace safetensors for Qwen3.5."""

from __future__ import annotations

import gc
import re
from collections import defaultdict
from enum import Enum
from typing import Any

import jax.numpy as jnp
import numpy as np
import safetensors
from etils import epath
from flax import nnx

from omegalax.models.params_utils import assign_weights_from_eval_shape, load_hf_config, map_to_bonsai_key, stoi
from .config import Qwen3_5Config, make_config
from .model import Qwen3_5ForConditionalGeneration


class _VT(Enum):
    """Vision transforms."""
    LINEAR = ((1, 0), None, False)
    LINEAR_BIAS = None
    EMBED = None
    SCALE = None
    CONV3D = None  # handled specially


class _TT(Enum):
    """Text transforms."""
    LINEAR = ((1, 0), None, False)
    EMBED = None
    SCALE = None


# Key mappings
def _get_vision_key_mapping():
    """HF → JAX mapping for vision encoder weights."""
    p = r"model\.visual\."
    return {
        # Patch embedding (Conv3D handled separately)
        p + r"patch_embed\.proj\.bias": (
            "visual.patch_embed.proj.bias", _VT.LINEAR_BIAS
        ),
        # Position embedding
        p + r"pos_embed\.weight": (
            "visual.pos_embed.embedding", _VT.EMBED
        ),
        # Blocks
        p + r"blocks\.([0-9]+)\.norm1\.weight": (
            r"visual.blocks.\1.norm1.weight", _VT.SCALE
        ),
        p + r"blocks\.([0-9]+)\.norm1\.bias": (
            r"visual.blocks.\1.norm1.bias", _VT.LINEAR_BIAS
        ),
        p + r"blocks\.([0-9]+)\.attn\.qkv\.weight": (
            r"visual.blocks.\1.attn.qkv.kernel", _VT.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.attn\.qkv\.bias": (
            r"visual.blocks.\1.attn.qkv.bias", _VT.LINEAR_BIAS
        ),
        p + r"blocks\.([0-9]+)\.attn\.proj\.weight": (
            r"visual.blocks.\1.attn.proj.kernel", _VT.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.attn\.proj\.bias": (
            r"visual.blocks.\1.attn.proj.bias", _VT.LINEAR_BIAS
        ),
        p + r"blocks\.([0-9]+)\.norm2\.weight": (
            r"visual.blocks.\1.norm2.weight", _VT.SCALE
        ),
        p + r"blocks\.([0-9]+)\.norm2\.bias": (
            r"visual.blocks.\1.norm2.bias", _VT.LINEAR_BIAS
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc1\.weight": (
            r"visual.blocks.\1.mlp.fc1.kernel", _VT.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc1\.bias": (
            r"visual.blocks.\1.mlp.fc1.bias", _VT.LINEAR_BIAS
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc2\.weight": (
            r"visual.blocks.\1.mlp.fc2.kernel", _VT.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc2\.bias": (
            r"visual.blocks.\1.mlp.fc2.bias", _VT.LINEAR_BIAS
        ),
        # Merger
        p + r"merger\.norm\.weight": (
            "visual.merger.norm.weight", _VT.SCALE
        ),
        p + r"merger\.norm\.bias": (
            "visual.merger.norm.bias", _VT.LINEAR_BIAS
        ),
        p + r"merger\.linear_fc1\.weight": (
            "visual.merger.fc1.kernel", _VT.LINEAR
        ),
        p + r"merger\.linear_fc1\.bias": (
            "visual.merger.fc1.bias", _VT.LINEAR_BIAS
        ),
        p + r"merger\.linear_fc2\.weight": (
            "visual.merger.fc2.kernel", _VT.LINEAR
        ),
        p + r"merger\.linear_fc2\.bias": (
            "visual.merger.fc2.bias", _VT.LINEAR_BIAS
        ),
    }


def _get_text_key_mapping():
    """HF → JAX mapping for text decoder weights."""
    p = r"model\.language_model\."
    L = r"([0-9]+)"
    return {
        # Embedding
        p + r"embed_tokens\.weight": (
            "text_model.embed_tokens.embedding", _TT.EMBED
        ),
        # Final norm
        p + r"norm\.weight": (
            "text_model.norm.weight", _TT.SCALE
        ),
        # Per-layer norms
        p + r"layers\." + L + r"\.input_layernorm\.weight": (
            r"text_model.layers.\1.input_layernorm.weight", _TT.SCALE
        ),
        p + r"layers\." + L + r"\.post_attention_layernorm\.weight": (
            r"text_model.layers.\1.post_attention_layernorm.weight", _TT.SCALE
        ),
        # Full attention
        p + r"layers\." + L + r"\.self_attn\.q_proj\.weight": (
            r"text_model.layers.\1.attn.q_proj.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.k_proj\.weight": (
            r"text_model.layers.\1.attn.k_proj.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.v_proj\.weight": (
            r"text_model.layers.\1.attn.v_proj.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.o_proj\.weight": (
            r"text_model.layers.\1.attn.o_proj.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.q_norm\.weight": (
            r"text_model.layers.\1.attn.q_norm.weight", _TT.SCALE
        ),
        p + r"layers\." + L + r"\.self_attn\.k_norm\.weight": (
            r"text_model.layers.\1.attn.k_norm.weight", _TT.SCALE
        ),
        # Linear attention (Gated Delta Net)
        p + r"layers\." + L + r"\.linear_attn\.in_proj_qkv\.weight": (
            r"text_model.layers.\1.linear_attn.in_proj_qkv.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_z\.weight": (
            r"text_model.layers.\1.linear_attn.in_proj_z.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_b\.weight": (
            r"text_model.layers.\1.linear_attn.in_proj_b.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_a\.weight": (
            r"text_model.layers.\1.linear_attn.in_proj_a.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.norm\.weight": (
            r"text_model.layers.\1.linear_attn.norm.weight", _TT.SCALE
        ),
        p + r"layers\." + L + r"\.linear_attn\.out_proj\.weight": (
            r"text_model.layers.\1.linear_attn.out_proj.kernel", _TT.LINEAR
        ),
        # MLP — shared expert
        p + r"layers\." + L + r"\.mlp\.shared_expert\.gate_proj\.weight": (
            r"text_model.layers.\1.mlp.shared_expert.gate_proj.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.mlp\.shared_expert\.up_proj\.weight": (
            r"text_model.layers.\1.mlp.shared_expert.up_proj.kernel", _TT.LINEAR
        ),
        p + r"layers\." + L + r"\.mlp\.shared_expert\.down_proj\.weight": (
            r"text_model.layers.\1.mlp.shared_expert.down_proj.kernel", _TT.LINEAR
        ),
        # LM head
        r"lm_head\.weight": (
            "lm_head.kernel", _TT.LINEAR
        ),
    }


# Regex patterns for special keys
_CONV1D_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.linear_attn\.conv1d\.weight"
)
_DT_BIAS_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.linear_attn\.dt_bias"
)
_A_LOG_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.linear_attn\.A_log"
)
_EXPERT_GATE_UP_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$"
)
_EXPERT_DOWN_BATCHED_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.down_proj$"
)
_EXPERT_PER_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
_ROUTER_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight"
)
_SHARED_EXPERT_GATE_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.shared_expert_gate\.weight"
)
_CONV3D_RE = re.compile(
    r"model\.visual\.patch_embed\.proj\.weight"
)


# Main loader
def create_qwen3_5_from_safetensors(
    file_dir: str, model_id: str
) -> Qwen3_5ForConditionalGeneration:
    """Load HuggingFace safetensors and return a Qwen3_5ForConditionalGeneration model."""
    cfg = make_config(model_id)
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    model = nnx.eval_shape(
        lambda: Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(params=0))
    )
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)

    vision_mapping = _get_vision_key_mapping()
    text_mapping = _get_text_key_mapping()
    combined_mapping = {**vision_mapping, **text_mapping}

    conversion_errors: list[str] = []
    unmatched_hf_keys: list[str] = []

    # Buffers for per-expert weights (save_pretrained format)
    expert_buf: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                # --- Special keys ---

                # Conv3D patch embedding
                if _CONV3D_RE.match(torch_key):
                    # HF: (out, in, D, H, W) → Flax: (D, H, W, in, out)
                    value = jnp.asarray(tensor.transpose(2, 3, 4, 1, 0))
                    _assign(state_dict, "visual.patch_embed.proj.kernel", value, torch_key, conversion_errors)
                    continue

                # Conv1D weights
                m = _CONV1D_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    # HF: (conv_dim, 1, K) → JAX: (conv_dim, K)
                    value = jnp.asarray(tensor.squeeze(1))
                    target = f"text_model.layers.{layer_idx}.linear_attn.conv_weight"
                    _assign(state_dict, target, value, torch_key, conversion_errors)
                    continue

                # dt_bias
                m = _DT_BIAS_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    target = f"text_model.layers.{layer_idx}.linear_attn.dt_bias"
                    _assign(state_dict, target, jnp.asarray(tensor), torch_key, conversion_errors)
                    continue

                # A_log
                m = _A_LOG_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    target = f"text_model.layers.{layer_idx}.linear_attn.A_log"
                    _assign(state_dict, target, jnp.asarray(tensor), torch_key, conversion_errors)
                    continue

                # MoE experts — batched gate_up_proj: HF (E, 2F, D), stored as-is
                m = _EXPERT_GATE_UP_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    target = f"text_model.layers.{layer_idx}.mlp.gate_up_proj"
                    _assign(state_dict, target, jnp.asarray(tensor), torch_key, conversion_errors)
                    continue

                # MoE experts — batched down_proj: HF (E, D, F), stored as-is
                m = _EXPERT_DOWN_BATCHED_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    target = f"text_model.layers.{layer_idx}.mlp.down_proj"
                    _assign(state_dict, target, jnp.asarray(tensor), torch_key, conversion_errors)
                    continue

                # MoE experts — per-expert format (from save_pretrained)
                m = _EXPERT_PER_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    expert_idx = int(m.group(2))
                    proj_name = m.group(3)
                    expert_buf[(layer_idx, proj_name)][expert_idx] = tensor
                    continue

                # MoE router: HF (E, D) → JAX Linear kernel (D, E)
                m = _ROUTER_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    value = jnp.asarray(tensor.T)
                    target = f"text_model.layers.{layer_idx}.mlp.router.kernel"
                    _assign(state_dict, target, value, torch_key, conversion_errors)
                    continue

                # Shared expert gate: HF (1, D) → JAX Linear kernel (D, 1)
                m = _SHARED_EXPERT_GATE_RE.match(torch_key)
                if m:
                    layer_idx = int(m.group(1))
                    value = jnp.asarray(tensor.T)
                    target = f"text_model.layers.{layer_idx}.mlp.shared_expert_gate.kernel"
                    _assign(state_dict, target, value, torch_key, conversion_errors)
                    continue

                # --- Regular key mapping ---
                jax_key, transform = map_to_bonsai_key(combined_mapping, torch_key)
                if jax_key is None:
                    unmatched_hf_keys.append(torch_key)
                    continue

                keys = [stoi(k) for k in jax_key.split(".")]
                try:
                    assign_weights_from_eval_shape(
                        keys, tensor, state_dict, torch_key, transform.value
                    )
                except Exception as e:
                    full_jax_key = ".".join(str(k) for k in keys)
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' → '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    # Assemble per-expert weights into batched format
    num_experts = cfg.text_config.num_experts
    layer_projs: dict[int, dict[str, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for (layer_idx, proj_name), expert_tensors in expert_buf.items():
        layer_projs[layer_idx][proj_name] = expert_tensors

    for layer_idx, projs in layer_projs.items():
        # Fuse gate_proj + up_proj → gate_up_proj
        if "gate_proj" in projs and "up_proj" in projs:
            gates = [projs["gate_proj"][i] for i in range(num_experts)]
            ups = [projs["up_proj"][i] for i in range(num_experts)]
            # Each is (F, D) in HF (out, in); concat → (2F, D); stack → (E, 2F, D)
            fused = np.stack(
                [np.concatenate([g, u], axis=0) for g, u in zip(gates, ups)], axis=0
            )
            target = f"text_model.layers.{layer_idx}.mlp.gate_up_proj"
            _assign(state_dict, target, jnp.asarray(fused), "experts.*.gate/up_proj", conversion_errors)

        # Stack down_proj: each (D, F) in HF; stack → (E, D, F)
        if "down_proj" in projs:
            downs = [projs["down_proj"][i] for i in range(num_experts)]
            stacked = np.stack(downs, axis=0)
            target = f"text_model.layers.{layer_idx}.mlp.down_proj"
            _assign(state_dict, target, jnp.asarray(stacked), "experts.*.down_proj", conversion_errors)

    if conversion_errors:
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors:\n"
            + "\n".join(conversion_errors)
        )

    if unmatched_hf_keys:
        raise RuntimeError(
            f"Unmapped HuggingFace parameters:\n" + "\n".join(sorted(unmatched_hf_keys))
        )

    if cfg.text_config.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["text_model"]["embed_tokens"]["embedding"].T

    gc.collect()
    return nnx.merge(graph_def, state_dict)


# Helpers
def _assign(
    state_dict: dict,
    dotted_key: str,
    value: jnp.ndarray,
    torch_key: str,
    errors: list[str],
):
    """Navigate state_dict by dotted key and set the value."""
    keys = [stoi(k) for k in dotted_key.split(".")]
    try:
        node: Any = state_dict
        for k in keys[:-1]:
            node = node[k]
        leaf = keys[-1]
        target = node[leaf]
        if hasattr(target, "shape") and target.shape != value.shape:
            raise ValueError(
                f"Shape mismatch: expected {target.shape}, got {value.shape}"
            )
        target_dtype = getattr(target, "dtype", None)
        if target_dtype is not None:
            value = value.astype(target_dtype)
        node[leaf] = value
    except Exception as e:
        errors.append(f"Failed to assign '{torch_key}' → '{dotted_key}': {type(e).__name__}: {e}")


def get_all_key_mappings():
    """Return the combined key mapping (useful for tests)."""
    return {**_get_vision_key_mapping(), **_get_text_key_mapping()}


SPECIAL_KEY_PATTERNS = [
    _CONV1D_RE,
    _DT_BIAS_RE,
    _A_LOG_RE,
    _EXPERT_GATE_UP_RE,
    _EXPERT_DOWN_BATCHED_RE,
    _EXPERT_PER_RE,
    _ROUTER_RE,
    _SHARED_EXPERT_GATE_RE,
    _CONV3D_RE,
]
