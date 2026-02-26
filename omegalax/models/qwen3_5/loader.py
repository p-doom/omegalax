"""Weight conversion from HuggingFace Qwen3.5 safetensors to JAX."""

from __future__ import annotations

import gc
import re
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import safetensors
from etils import epath
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.params_utils import (
    Transform,
    assign_to_state_dict,
    assign_weights_from_eval_shape,
    check_conversion_errors,
    find_safetensors,
    load_hf_config,
    map_to_bonsai_key,
    stoi,
)
from omegalax.models.sharding_runtime import apply_sharding_to_model_state as apply_sharding_to_model_state_runtime
from .config import Qwen3_5Config, make_config, make_config_from_hf
from .model import Qwen3_5ForConditionalGeneration
from .sharding import model_state_sharding


def _assert_config(cfg: Qwen3_5Config, hf_cfg: dict):
    """Validate that a spec-based config matches the HF config.json."""
    txt = hf_cfg["text_config"]
    vis = hf_cfg["vision_config"]
    rope_params = txt["rope_parameters"]

    def _require(name, lhs, rhs):
        if lhs != rhs:
            raise ValueError(
                f"Config mismatch for {name}: expected {lhs}, found {rhs} in HF config"
            )

    _require("vocab_size", cfg.text_config.vocab_size, txt["vocab_size"])
    _require("num_hidden_layers", cfg.text_config.num_hidden_layers, txt["num_hidden_layers"])
    _require("hidden_size", cfg.text_config.hidden_size, txt["hidden_size"])
    _require("num_attention_heads", cfg.text_config.num_attention_heads, txt["num_attention_heads"])
    _require("num_key_value_heads", cfg.text_config.num_key_value_heads, txt["num_key_value_heads"])
    _require("head_dim", cfg.text_config.head_dim, txt["head_dim"])
    _require("num_experts", cfg.text_config.num_experts, txt["num_experts"])
    _require("num_experts_per_tok", cfg.text_config.num_experts_per_tok, txt["num_experts_per_tok"])
    _require("moe_intermediate_size", cfg.text_config.moe_intermediate_size, txt["moe_intermediate_size"])
    _require("rope_theta", cfg.text_config.rope_theta, rope_params["rope_theta"])
    _require("mrope_section", tuple(cfg.text_config.mrope_section), tuple(rope_params["mrope_section"]))
    _require("mrope_interleaved", cfg.text_config.mrope_interleaved, rope_params["mrope_interleaved"])

    _require("vision.hidden_size", cfg.vision_config.hidden_size, vis["hidden_size"])
    _require("vision.depth", cfg.vision_config.depth, vis["depth"])
    _require("vision.num_heads", cfg.vision_config.num_heads, vis["num_heads"])
    _require("vision.patch_size", cfg.vision_config.patch_size, vis["patch_size"])
    _require("vision.out_hidden_size", cfg.vision_config.out_hidden_size, vis["out_hidden_size"])


def _get_vision_key_mapping():
    """HF → JAX mapping for vision encoder weights."""
    p = r"model\.visual\."
    return {
        # Patch embedding (Conv3D handled separately)
        p + r"patch_embed\.proj\.bias": (
            "vision.patch_embed.proj.bias", Transform.BIAS
        ),
        # Position embedding
        p + r"pos_embed\.weight": (
            "vision.pos_embed.embedding", Transform.EMBED
        ),
        # Blocks
        p + r"blocks\.([0-9]+)\.norm1\.weight": (
            r"vision.blocks.\1.norm1.weight", Transform.SCALE
        ),
        p + r"blocks\.([0-9]+)\.norm1\.bias": (
            r"vision.blocks.\1.norm1.bias", Transform.BIAS
        ),
        p + r"blocks\.([0-9]+)\.attn\.qkv\.weight": (
            r"vision.blocks.\1.attn.qkv.kernel", Transform.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.attn\.qkv\.bias": (
            r"vision.blocks.\1.attn.qkv.bias", Transform.BIAS
        ),
        p + r"blocks\.([0-9]+)\.attn\.proj\.weight": (
            r"vision.blocks.\1.attn.proj.kernel", Transform.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.attn\.proj\.bias": (
            r"vision.blocks.\1.attn.proj.bias", Transform.BIAS
        ),
        p + r"blocks\.([0-9]+)\.norm2\.weight": (
            r"vision.blocks.\1.norm2.weight", Transform.SCALE
        ),
        p + r"blocks\.([0-9]+)\.norm2\.bias": (
            r"vision.blocks.\1.norm2.bias", Transform.BIAS
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc1\.weight": (
            r"vision.blocks.\1.mlp.fc1.kernel", Transform.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc1\.bias": (
            r"vision.blocks.\1.mlp.fc1.bias", Transform.BIAS
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc2\.weight": (
            r"vision.blocks.\1.mlp.fc2.kernel", Transform.LINEAR
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc2\.bias": (
            r"vision.blocks.\1.mlp.fc2.bias", Transform.BIAS
        ),
        # Merger
        p + r"merger\.norm\.weight": (
            "vision.merger.norm.weight", Transform.SCALE
        ),
        p + r"merger\.norm\.bias": (
            "vision.merger.norm.bias", Transform.BIAS
        ),
        p + r"merger\.linear_fc1\.weight": (
            "vision.merger.fc1.kernel", Transform.LINEAR
        ),
        p + r"merger\.linear_fc1\.bias": (
            "vision.merger.fc1.bias", Transform.BIAS
        ),
        p + r"merger\.linear_fc2\.weight": (
            "vision.merger.fc2.kernel", Transform.LINEAR
        ),
        p + r"merger\.linear_fc2\.bias": (
            "vision.merger.fc2.bias", Transform.BIAS
        ),
    }


def _get_text_key_mapping():
    """HF → JAX mapping for text decoder weights (non-linear-attn, non-MoE)."""
    p = r"model\.language_model\."
    L = r"([0-9]+)"
    return {
        p + r"embed_tokens\.weight": (
            "text.embedder.embedding", Transform.EMBED
        ),
        p + r"norm\.weight": (
            "text.final_norm.weight", Transform.SCALE
        ),
        p + r"layers\." + L + r"\.input_layernorm\.weight": (
            r"text.layers.\1.input_layernorm.weight", Transform.SCALE
        ),
        p + r"layers\." + L + r"\.post_attention_layernorm\.weight": (
            r"text.layers.\1.post_attention_layernorm.weight", Transform.SCALE
        ),
        p + r"layers\." + L + r"\.self_attn\.q_proj\.weight": (
            r"text.layers.\1.attn.q_proj.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.k_proj\.weight": (
            r"text.layers.\1.attn.k_proj.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.v_proj\.weight": (
            r"text.layers.\1.attn.v_proj.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.o_proj\.weight": (
            r"text.layers.\1.attn.o_proj.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.self_attn\.q_norm\.weight": (
            r"text.layers.\1.attn.q_norm.weight", Transform.SCALE
        ),
        p + r"layers\." + L + r"\.self_attn\.k_norm\.weight": (
            r"text.layers.\1.attn.k_norm.weight", Transform.SCALE
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_qkv\.weight": (
            r"text.layers.\1.linear_attn.in_proj_qkv.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_z\.weight": (
            r"text.layers.\1.linear_attn.in_proj_z.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_b\.weight": (
            r"text.layers.\1.linear_attn.in_proj_b.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_a\.weight": (
            r"text.layers.\1.linear_attn.in_proj_a.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.linear_attn\.norm\.weight": (
            r"text.layers.\1.linear_attn.norm.weight", Transform.SCALE
        ),
        p + r"layers\." + L + r"\.linear_attn\.out_proj\.weight": (
            r"text.layers.\1.linear_attn.out_proj.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.mlp\.shared_expert\.gate_proj\.weight": (
            r"text.layers.\1.mlp.shared_expert.gate_proj.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.mlp\.shared_expert\.up_proj\.weight": (
            r"text.layers.\1.mlp.shared_expert.up_proj.kernel", Transform.LINEAR
        ),
        p + r"layers\." + L + r"\.mlp\.shared_expert\.down_proj\.weight": (
            r"text.layers.\1.mlp.shared_expert.down_proj.kernel", Transform.LINEAR
        ),
        r"lm_head\.weight": (
            "lm_head.kernel", Transform.LINEAR
        ),
    }


def _get_non_expert_mapping():
    """Mapping for all non-special parameters (vision + text core paths)."""
    mapping = {}
    mapping.update(_get_vision_key_mapping())
    mapping.update(_get_text_key_mapping())
    return mapping


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


def create_qwen3_5_from_safetensors(
    file_dir: str,
    model_id: str = "",
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
) -> tuple[Qwen3_5ForConditionalGeneration, Qwen3_5Config]:
    """Load HuggingFace Qwen3.5 safetensors into a JAX Qwen3.5 model."""
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size)

    path = epath.Path(file_dir).expanduser()
    files = find_safetensors(file_dir)

    hf_cfg = load_hf_config(path)
    if model_id:
        cfg = make_config(model_id)
        _assert_config(cfg, hf_cfg)
    else:
        cfg = make_config_from_hf(hf_cfg)

    model = nnx.eval_shape(
        lambda: Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(params=0))
    )
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)

    non_expert_mapping = _get_non_expert_mapping()
    unmatched_hf_keys: list[str] = []

    expert_buf: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)

    def _handle_linear_attn_specials(torch_key: str, tensor):
        m = _CONV1D_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            value = jnp.asarray(tensor.squeeze(1))
            target = f"text.layers.{layer_idx}.linear_attn.conv_weight"
            assign_to_state_dict(state_dict, target, value, torch_key)
            return True

        m = _DT_BIAS_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            target = f"text.layers.{layer_idx}.linear_attn.dt_bias"
            assign_to_state_dict(state_dict, target, jnp.asarray(tensor), torch_key)
            return True

        m = _A_LOG_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            target = f"text.layers.{layer_idx}.linear_attn.A_log"
            assign_to_state_dict(state_dict, target, jnp.asarray(tensor), torch_key)
            return True
        return False

    def _handle_moe_specials(torch_key: str, tensor) -> bool:
        m = _EXPERT_GATE_UP_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            target = f"text.layers.{layer_idx}.mlp.gate_up_proj"
            assign_to_state_dict(state_dict, target, jnp.asarray(tensor), torch_key)
            return True

        m = _EXPERT_DOWN_BATCHED_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            target = f"text.layers.{layer_idx}.mlp.down_proj"
            assign_to_state_dict(state_dict, target, jnp.asarray(tensor), torch_key)
            return True

        m = _EXPERT_PER_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            expert_idx = int(m.group(2))
            proj_name = m.group(3)
            expert_buf[(layer_idx, proj_name)][expert_idx] = tensor
            return True

        m = _ROUTER_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            value = jnp.asarray(tensor.T)
            target = f"text.layers.{layer_idx}.mlp.router.kernel"
            assign_to_state_dict(state_dict, target, value, torch_key)
            return True

        m = _SHARED_EXPERT_GATE_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            value = jnp.asarray(tensor.T)
            target = f"text.layers.{layer_idx}.mlp.shared_expert_gate.kernel"
            assign_to_state_dict(state_dict, target, value, torch_key)
            return True
        return False

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                # Special: Conv3D patch embedding
                if _CONV3D_RE.match(torch_key):
                    value = jnp.asarray(tensor.transpose(2, 3, 4, 1, 0))
                    assign_to_state_dict(state_dict, "vision.patch_embed.proj.kernel", value, torch_key)
                    continue

                # Linear attention specials
                if _handle_linear_attn_specials(torch_key, tensor):
                    continue

                # MoE specials
                if _handle_moe_specials(torch_key, tensor):
                    continue

                # Generic mapping
                jax_key, transform = map_to_bonsai_key(non_expert_mapping, torch_key)
                if jax_key is None:
                    unmatched_hf_keys.append(torch_key)
                    continue

                keys = [stoi(k) for k in jax_key.split(".")]
                assign_weights_from_eval_shape(
                    keys, tensor, state_dict, torch_key, transform.value
                )
        gc.collect()

    # Assemble per-expert weights into batched format (per-expert HF format)
    num_experts = cfg.text_config.num_experts
    layer_projs: dict[int, dict[str, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for (layer_idx, proj_name), expert_tensors in expert_buf.items():
        layer_projs[layer_idx][proj_name] = expert_tensors

    for layer_idx, projs in layer_projs.items():
        if "gate_proj" in projs and "up_proj" in projs:
            gates = [projs["gate_proj"][i] for i in range(num_experts)]
            ups = [projs["up_proj"][i] for i in range(num_experts)]
            fused = np.stack(
                [np.concatenate([g, u], axis=0) for g, u in zip(gates, ups)], axis=0
            )
            target = f"text.layers.{layer_idx}.mlp.gate_up_proj"
            assign_to_state_dict(state_dict, target, jnp.asarray(fused), "experts.*.gate/up_proj")

        if "down_proj" in projs:
            downs = [projs["down_proj"][i] for i in range(num_experts)]
            stacked = np.stack(downs, axis=0)
            target = f"text.layers.{layer_idx}.mlp.down_proj"
            assign_to_state_dict(state_dict, target, jnp.asarray(stacked), "experts.*.down_proj")

    check_conversion_errors(unmatched_hf_keys)

    if cfg.text_config.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["text"]["embedder"]["embedding"].T

    gc.collect()
    model = nnx.merge(graph_def, state_dict)
    model = apply_sharding_to_model_state_runtime(
        model,
        cfg.text_config.shd_cfg,
        mesh,
        model_state_sharding,
    )
    return model, cfg


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
