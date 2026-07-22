"""Weight conversion from HuggingFace Qwen3.5 safetensors to JAX.

Multi-token-prediction (MTP) head — deliberately not loaded
-----------------------------------------------------------
Real Qwen3.5 checkpoints ship a multi-token-prediction head (the ``mtp.*``
tensors: ``mtp.fc``, ``mtp.pre_fc_norm_{hidden,embedding}``, a single
full-attention decoder block ``mtp.layers.0.*`` — dense or MoE mirroring the
main model — and ``mtp.norm``). It is a DeepSeek-V3/GLM-style auxiliary head
used for (a) a multi-token-prediction training loss and (b) speculative
decoding at inference. Note (a) can improve the main model even when the head
is never used at inference — but that gain is a *pretraining*-scale result, and
its value for a short agent-policy SFT run is unestablished.

omegalax's Qwen3.5 model intentionally omits this head:

* SFT trains a plain next-token objective — ``chunked_cross_entropy_loss`` on
  the final hidden state plus the MoE router aux loss (``trainers/vlm.py``);
  there is no multi-token-prediction loss term, and the forward pass returns
  only ``(hidden_BTD, router_aux)``, so an MTP head would receive no gradient.
* Rollout is plain autoregressive generation; there is no speculative-decoding
  / draft-verify path anywhere in the repo (``vlm.api.decode`` is a stub).
* The HuggingFace reference itself does not instantiate the head either — both
  ``Qwen3_5ForConditionalGeneration`` and ``Qwen3_5MoeForConditionalGeneration``
  declare ``_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`` and define no MTP
  module. Building it here would diverge from the canonical model.

Under the current objective the head therefore receives no gradient and would
only cost memory / checkpoint bytes, so we do not build it. Rather than
blanket-skipping anything matching ``mtp.*`` (which would also hide a
genuinely-unmapped key), we enumerate the *exact* set of MTP keys the config
implies (``_expected_mtp_keys``) and drop only those; any unexpected ``mtp.*``
key, or a missing expected one, raises.

This is a reversible decision, not a dead end. Adding an MTP training objective
later is a deliberate, separately-validated feature (instantiate the head in the
model, add the multi-token loss term to the trainer, wire up LoRA/export); the
pretrained ``mtp.*`` weights are untouched on disk, and this loader's
exact-enumeration approach means the drop-set simply shrinks as those keys get
mapped — nothing here has to be undone first.
"""

from __future__ import annotations

import dataclasses
import gc
import re
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import safetensors
from etils import epath
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.shard_config import shard_config_for_mesh
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
from .config import Qwen3_5Config, make_config_from_hf
from .model import Qwen3_5ForConditionalGeneration


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

    if cfg.text_config.is_moe:
        _require("num_experts", cfg.text_config.num_experts, txt["num_experts"])
        _require(
            "num_experts_per_tok", cfg.text_config.num_experts_per_tok, txt["num_experts_per_tok"]
        )
        _require(
            "moe_intermediate_size",
            cfg.text_config.moe_intermediate_size,
            txt["moe_intermediate_size"],
        )
    else:
        _require("intermediate_size", cfg.text_config.intermediate_size, txt["intermediate_size"])

    _require("rope_theta", cfg.text_config.rope_theta, rope_params["rope_theta"])
    _require(
        "mrope_section", tuple(cfg.text_config.mrope_section), tuple(rope_params["mrope_section"])
    )
    _require(
        "mrope_interleaved", cfg.text_config.mrope_interleaved, rope_params["mrope_interleaved"]
    )

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
        p + r"patch_embed\.proj\.bias": ("vision.patch_embed.proj.bias", Transform.BIAS),
        # Position embedding
        p + r"pos_embed\.weight": ("vision.pos_embed.embedding", Transform.EMBED),
        # Blocks
        p + r"blocks\.([0-9]+)\.norm1\.weight": (r"vision.blocks.\1.norm1.weight", Transform.SCALE),
        p + r"blocks\.([0-9]+)\.norm1\.bias": (r"vision.blocks.\1.norm1.bias", Transform.BIAS),
        p + r"blocks\.([0-9]+)\.attn\.qkv\.weight": (
            r"vision.blocks.\1.attn.qkv.kernel",
            Transform.LINEAR,
        ),
        p + r"blocks\.([0-9]+)\.attn\.qkv\.bias": (
            r"vision.blocks.\1.attn.qkv.bias",
            Transform.BIAS,
        ),
        p + r"blocks\.([0-9]+)\.attn\.proj\.weight": (
            r"vision.blocks.\1.attn.proj.kernel",
            Transform.LINEAR,
        ),
        p + r"blocks\.([0-9]+)\.attn\.proj\.bias": (
            r"vision.blocks.\1.attn.proj.bias",
            Transform.BIAS,
        ),
        p + r"blocks\.([0-9]+)\.norm2\.weight": (r"vision.blocks.\1.norm2.weight", Transform.SCALE),
        p + r"blocks\.([0-9]+)\.norm2\.bias": (r"vision.blocks.\1.norm2.bias", Transform.BIAS),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc1\.weight": (
            r"vision.blocks.\1.mlp.fc1.kernel",
            Transform.LINEAR,
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc1\.bias": (
            r"vision.blocks.\1.mlp.fc1.bias",
            Transform.BIAS,
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc2\.weight": (
            r"vision.blocks.\1.mlp.fc2.kernel",
            Transform.LINEAR,
        ),
        p + r"blocks\.([0-9]+)\.mlp\.linear_fc2\.bias": (
            r"vision.blocks.\1.mlp.fc2.bias",
            Transform.BIAS,
        ),
        # Merger
        p + r"merger\.norm\.weight": ("vision.merger.norm.weight", Transform.SCALE),
        p + r"merger\.norm\.bias": ("vision.merger.norm.bias", Transform.BIAS),
        p + r"merger\.linear_fc1\.weight": ("vision.merger.fc1.kernel", Transform.LINEAR),
        p + r"merger\.linear_fc1\.bias": ("vision.merger.fc1.bias", Transform.BIAS),
        p + r"merger\.linear_fc2\.weight": ("vision.merger.fc2.kernel", Transform.LINEAR),
        p + r"merger\.linear_fc2\.bias": ("vision.merger.fc2.bias", Transform.BIAS),
    }


def _get_text_key_mapping():
    """HF → JAX mapping for text decoder weights (non-linear-attn, non-MoE)."""
    p = r"model\.language_model\."
    L = r"([0-9]+)"
    return {
        p + r"embed_tokens\.weight": ("text.embedder.embedding", Transform.EMBED),
        p + r"norm\.weight": ("text.final_norm.weight", Transform.SCALE),
        p + r"layers\." + L + r"\.input_layernorm\.weight": (
            r"text.layers.\1.input_layernorm.weight",
            Transform.SCALE,
        ),
        p + r"layers\." + L + r"\.post_attention_layernorm\.weight": (
            r"text.layers.\1.post_attention_layernorm.weight",
            Transform.SCALE,
        ),
        p + r"layers\." + L + r"\.self_attn\.q_proj\.weight": (
            r"text.layers.\1.attn.q_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.self_attn\.k_proj\.weight": (
            r"text.layers.\1.attn.k_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.self_attn\.v_proj\.weight": (
            r"text.layers.\1.attn.v_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.self_attn\.o_proj\.weight": (
            r"text.layers.\1.attn.o_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.self_attn\.q_norm\.weight": (
            r"text.layers.\1.attn.q_norm.weight",
            Transform.SCALE,
        ),
        p + r"layers\." + L + r"\.self_attn\.k_norm\.weight": (
            r"text.layers.\1.attn.k_norm.weight",
            Transform.SCALE,
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_qkv\.weight": (
            r"text.layers.\1.linear_attn.in_proj_qkv.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_z\.weight": (
            r"text.layers.\1.linear_attn.in_proj_z.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_b\.weight": (
            r"text.layers.\1.linear_attn.in_proj_b.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.linear_attn\.in_proj_a\.weight": (
            r"text.layers.\1.linear_attn.in_proj_a.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.linear_attn\.norm\.weight": (
            r"text.layers.\1.linear_attn.norm.weight",
            Transform.SCALE,
        ),
        p + r"layers\." + L + r"\.linear_attn\.out_proj\.weight": (
            r"text.layers.\1.linear_attn.out_proj.kernel",
            Transform.LINEAR,
        ),
        # MoE shared expert
        p + r"layers\." + L + r"\.mlp\.shared_expert\.gate_proj\.weight": (
            r"text.layers.\1.mlp.shared_expert.gate_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.mlp\.shared_expert\.up_proj\.weight": (
            r"text.layers.\1.mlp.shared_expert.up_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.mlp\.shared_expert\.down_proj\.weight": (
            r"text.layers.\1.mlp.shared_expert.down_proj.kernel",
            Transform.LINEAR,
        ),
        # Dense MLP (used when num_experts == 0)
        p + r"layers\." + L + r"\.mlp\.gate_proj\.weight": (
            r"text.layers.\1.mlp.gate_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.mlp\.up_proj\.weight": (
            r"text.layers.\1.mlp.up_proj.kernel",
            Transform.LINEAR,
        ),
        p + r"layers\." + L + r"\.mlp\.down_proj\.weight": (
            r"text.layers.\1.mlp.down_proj.kernel",
            Transform.LINEAR,
        ),
        r"lm_head\.weight": ("lm_head.kernel", Transform.LINEAR),
    }


def _get_non_expert_mapping():
    """Mapping for all non-special parameters (vision + text core paths)."""
    mapping = {}
    mapping.update(_get_vision_key_mapping())
    mapping.update(_get_text_key_mapping())
    return mapping


# Regex patterns for special keys
_CONV1D_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.linear_attn\.conv1d\.weight")
_DT_BIAS_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.linear_attn\.dt_bias")
_A_LOG_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.linear_attn\.A_log")
_EXPERT_GATE_UP_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$"
)
_EXPERT_DOWN_BATCHED_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.down_proj$"
)
_EXPERT_PER_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
_ROUTER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight")
_SHARED_EXPERT_GATE_RE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.shared_expert_gate\.weight"
)
_CONV3D_RE = re.compile(r"model\.visual\.patch_embed\.proj\.weight")


def _expected_mtp_keys(hf_cfg: dict, cfg: Qwen3_5Config) -> set[str]:
    """Exact set of HuggingFace ``mtp.*`` keys a Qwen3.5 checkpoint should carry.

    The multi-token-prediction head is intentionally not instantiated (see the
    module docstring). We still enumerate its keys precisely so that the drop is
    explicit and complete: only these keys are excluded, an unexpected ``mtp.*``
    key surfaces as an error, and a missing expected key surfaces too — instead
    of a wildcard that would silently swallow either.

    The structure is derived from ``text_config.mtp_num_hidden_layers`` (and, for
    MoE checkpoints, ``num_experts``). ``mtp_num_hidden_layers`` is metadata that
    only lives in ``config.json``; if it is absent/zero we expect no MTP head, and
    any ``mtp.*`` weight then present is treated as unexpected (raises) rather than
    dropped.
    """
    txt = hf_cfg["text_config"]
    n_layers = txt.get("mtp_num_hidden_layers", 0)
    if not isinstance(n_layers, int) or n_layers < 0:
        raise ValueError(
            f"Invalid text_config.mtp_num_hidden_layers={n_layers!r}; expected a non-negative int."
        )
    if n_layers == 0:
        return set()

    keys: set[str] = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    for i in range(n_layers):
        base = f"mtp.layers.{i}"
        keys.update(
            {
                f"{base}.input_layernorm.weight",
                f"{base}.post_attention_layernorm.weight",
                f"{base}.self_attn.q_proj.weight",
                f"{base}.self_attn.k_proj.weight",
                f"{base}.self_attn.v_proj.weight",
                f"{base}.self_attn.o_proj.weight",
                f"{base}.self_attn.q_norm.weight",
                f"{base}.self_attn.k_norm.weight",
            }
        )
        if cfg.text_config.is_moe:
            # The MTP block's FFN mirrors the main model's MoE FFN.
            keys.add(f"{base}.mlp.gate.weight")  # router
            keys.add(f"{base}.mlp.shared_expert_gate.weight")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys.add(f"{base}.mlp.shared_expert.{proj}.weight")
                for e in range(cfg.text_config.num_experts):
                    keys.add(f"{base}.mlp.experts.{e}.{proj}.weight")
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys.add(f"{base}.mlp.{proj}.weight")
    return keys


def create_qwen3_5_from_safetensors(
    file_dir: str,
    model_id: str = "",
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> tuple[Qwen3_5ForConditionalGeneration, Qwen3_5Config]:
    """Load HuggingFace Qwen3.5 safetensors into a JAX Qwen3.5 model."""
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)

    path = epath.Path(file_dir).expanduser()
    files = find_safetensors(file_dir)

    hf_cfg = load_hf_config(path)
    cfg = make_config_from_hf(hf_cfg)
    _assert_config(cfg, hf_cfg)
    # Align the (text) sharding config to the concrete mesh so the resolved
    # PartitionSpecs below place each tensor directly onto its FSDP/TP shard.
    cfg = dataclasses.replace(
        cfg,
        text_config=dataclasses.replace(
            cfg.text_config, shd_cfg=shard_config_for_mesh(cfg.text_config.shd_cfg, mesh)
        ),
    )

    # Build an *abstract* model (shapes/dtypes/shardings only, zero device bytes).
    # ``get_partition_spec`` resolves the logical axis annotations to concrete
    # PartitionSpecs so each loaded tensor is placed directly onto its shard;
    # ``eval_shape`` alone only attaches an AbstractMesh sharding, which
    # ``jax.device_put`` cannot consume.
    with mesh_rules(mesh):
        model = nnx.eval_shape(
            lambda: Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(params=0))
        )
        graph_def, abs_state = nnx.split(model)
        pspec_dict = nnx.to_pure_dict(nnx.get_partition_spec(abs_state))
    state_dict = nnx.to_pure_dict(abs_state)

    non_expert_mapping = _get_non_expert_mapping()
    unmatched_hf_keys: list[str] = []

    # The multi-token-prediction head is intentionally dropped (see module docstring).
    # Enumerate the exact keys we expect to skip so the exclusion is explicit and
    # complete rather than a wildcard that could hide a genuinely-unmapped key.
    expected_mtp_keys = _expected_mtp_keys(hf_cfg, cfg)
    seen_mtp_keys: set[str] = set()

    expert_buf: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)

    def _handle_linear_attn_specials(torch_key: str, tensor):
        m = _CONV1D_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            value = jnp.asarray(tensor.squeeze(1))
            target = f"text.layers.{layer_idx}.linear_attn.conv_weight"
            assign_to_state_dict(
                state_dict, target, value, torch_key, mesh=mesh, pspec_dict=pspec_dict
            )
            return True

        m = _DT_BIAS_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            target = f"text.layers.{layer_idx}.linear_attn.dt_bias"
            assign_to_state_dict(
                state_dict, target, jnp.asarray(tensor), torch_key, mesh=mesh, pspec_dict=pspec_dict
            )
            return True

        m = _A_LOG_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            target = f"text.layers.{layer_idx}.linear_attn.A_log"
            assign_to_state_dict(
                state_dict, target, jnp.asarray(tensor), torch_key, mesh=mesh, pspec_dict=pspec_dict
            )
            return True
        return False

    def _handle_moe_specials(torch_key: str, tensor) -> bool:
        # Fused gate_up_proj: (E, 2*F, D) → split into gate (E, D, F) + up (E, D, F)
        m = _EXPERT_GATE_UP_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            fused_E2FD = np.asarray(tensor)
            gate_EFD, up_EFD = np.split(fused_E2FD, 2, axis=1)
            gate_EDF = np.swapaxes(gate_EFD, 1, 2)
            up_EDF = np.swapaxes(up_EFD, 1, 2)
            assign_to_state_dict(
                state_dict,
                f"text.layers.{layer_idx}.mlp.gate_proj",
                jnp.asarray(gate_EDF),
                torch_key,
                mesh=mesh,
                pspec_dict=pspec_dict,
            )
            assign_to_state_dict(
                state_dict,
                f"text.layers.{layer_idx}.mlp.up_proj",
                jnp.asarray(up_EDF),
                torch_key,
                mesh=mesh,
                pspec_dict=pspec_dict,
            )
            return True

        # Batched down_proj: HF (E, D, F) → JAX (E, F, D)
        m = _EXPERT_DOWN_BATCHED_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            down_EDF = np.asarray(tensor)
            down_EFD = np.swapaxes(down_EDF, 1, 2)
            assign_to_state_dict(
                state_dict,
                f"text.layers.{layer_idx}.mlp.down_proj",
                jnp.asarray(down_EFD),
                torch_key,
                mesh=mesh,
                pspec_dict=pspec_dict,
            )
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
            assign_to_state_dict(
                state_dict, target, value, torch_key, mesh=mesh, pspec_dict=pspec_dict
            )
            return True

        m = _SHARED_EXPERT_GATE_RE.match(torch_key)
        if m:
            layer_idx = int(m.group(1))
            value = jnp.asarray(tensor.T)
            target = f"text.layers.{layer_idx}.mlp.shared_expert_gate.kernel"
            assign_to_state_dict(
                state_dict, target, value, torch_key, mesh=mesh, pspec_dict=pspec_dict
            )
            return True
        return False

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                # Special: Conv3D patch embedding
                if _CONV3D_RE.match(torch_key):
                    value = jnp.asarray(tensor.transpose(2, 3, 4, 1, 0))
                    assign_to_state_dict(
                        state_dict,
                        "vision.patch_embed.proj.kernel",
                        value,
                        torch_key,
                        mesh=mesh,
                        pspec_dict=pspec_dict,
                    )
                    continue

                # Linear attention specials
                if _handle_linear_attn_specials(torch_key, tensor):
                    continue

                # MoE specials
                if _handle_moe_specials(torch_key, tensor):
                    continue

                # Multi-token-prediction head: deliberately not loaded. Drop only
                # the exact expected keys; anything else under mtp.* falls through
                # to unmatched below and raises.
                if torch_key in expected_mtp_keys:
                    seen_mtp_keys.add(torch_key)
                    continue

                # Generic mapping
                jax_key, transform = map_to_bonsai_key(non_expert_mapping, torch_key)
                if jax_key is None:
                    unmatched_hf_keys.append(torch_key)
                    continue

                keys = [stoi(k) for k in jax_key.split(".")]
                assign_weights_from_eval_shape(
                    keys,
                    tensor,
                    state_dict,
                    torch_key,
                    transform.value,
                    mesh=mesh,
                    pspec_dict=pspec_dict,
                )
        gc.collect()

    # Assemble per-expert weights into batched format (per-expert HF format)
    if cfg.text_config.is_moe and expert_buf:
        num_experts = cfg.text_config.num_experts
        layer_projs: dict[int, dict[str, dict[int, np.ndarray]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for (layer_idx, proj_name), expert_tensors in expert_buf.items():
            layer_projs[layer_idx][proj_name] = expert_tensors

        for layer_idx, projs in layer_projs.items():
            # Per-expert HF weights are (F, D) each. Stack → (E, F, D), transpose → (E, D, F).
            if "gate_proj" in projs:
                gates = [projs["gate_proj"][i] for i in range(num_experts)]
                gate_EFD = np.stack(gates, axis=0)
                gate_EDF = np.swapaxes(gate_EFD, 1, 2)
                assign_to_state_dict(
                    state_dict,
                    f"text.layers.{layer_idx}.mlp.gate_proj",
                    jnp.asarray(gate_EDF),
                    "experts.*.gate_proj",
                    mesh=mesh,
                    pspec_dict=pspec_dict,
                )

            if "up_proj" in projs:
                ups = [projs["up_proj"][i] for i in range(num_experts)]
                up_EFD = np.stack(ups, axis=0)
                up_EDF = np.swapaxes(up_EFD, 1, 2)
                assign_to_state_dict(
                    state_dict,
                    f"text.layers.{layer_idx}.mlp.up_proj",
                    jnp.asarray(up_EDF),
                    "experts.*.up_proj",
                    mesh=mesh,
                    pspec_dict=pspec_dict,
                )

            if "down_proj" in projs:
                # HF per-expert down_proj.weight is (D, F). Stack → (E, D, F). Swap → (E, F, D).
                downs = [projs["down_proj"][i] for i in range(num_experts)]
                down_EDF = np.stack(downs, axis=0)
                down_EFD = np.swapaxes(down_EDF, 1, 2)
                assign_to_state_dict(
                    state_dict,
                    f"text.layers.{layer_idx}.mlp.down_proj",
                    jnp.asarray(down_EFD),
                    "experts.*.down_proj",
                    mesh=mesh,
                    pspec_dict=pspec_dict,
                )

    # The MTP head must be present in full when the config declares it: a missing
    # expected key means our understanding of the head drifted from the checkpoint,
    # which we surface rather than silently tolerate. (Unexpected mtp.* keys already
    # landed in unmatched_hf_keys and raise via check_conversion_errors below.)
    missing_mtp_keys = expected_mtp_keys - seen_mtp_keys
    if missing_mtp_keys:
        raise RuntimeError(
            "Expected multi-token-prediction (MTP) head weights were missing from the "
            "checkpoint; the deliberate MTP exclusion is out of sync with the checkpoint "
            "structure:\n" + "\n".join(sorted(missing_mtp_keys))
        )

    check_conversion_errors(unmatched_hf_keys)

    if cfg.text_config.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["text"]["embedder"]["embedding"].T

    gc.collect()
    model = nnx.merge(graph_def, state_dict)
    from omegalax.models.sharding_runtime import _finalize_q_shardings

    _finalize_q_shardings(model, mesh)
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
