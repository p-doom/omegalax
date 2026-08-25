"""Weight conversion from HuggingFace Qwen3-VL safetensors to JAX."""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Sequence
from typing import Any

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
from omegalax.models.shard_config import shard_config_for_mesh
from omegalax.models.sharding_runtime import _finalize_q_shardings

from .config import Qwen3VLConfig, make_vl_config_from_hf
from .model import Qwen3VL


def _assert_vl_config(cfg: Qwen3VLConfig, hf_cfg: dict):
    txt = hf_cfg["text_config"]
    vis = hf_cfg["vision_config"]
    rope_scaling = txt["rope_scaling"]

    def _require(name, lhs, rhs):
        if lhs != rhs:
            raise ValueError(
                f"Config mismatch for {name}: expected {lhs}, found {rhs} in HF config"
            )

    _require("vocab_size", cfg.vocab_size, txt["vocab_size"])
    _require("num_layers", cfg.num_layers, txt["num_hidden_layers"])
    _require("emb_dim", cfg.emb_dim, txt["hidden_size"])
    _require("num_heads", cfg.num_heads, txt["num_attention_heads"])
    _require("num_kv_heads", cfg.num_kv_heads, txt["num_key_value_heads"])
    _require("head_dim", cfg.head_dim, txt["head_dim"])
    _require("mlp_dim", cfg.mlp_dim, txt["intermediate_size"])
    _require("rope_theta", cfg.rope_theta, txt["rope_theta"])
    _require("mrope_section", tuple(cfg.mrope_section), tuple(rope_scaling["mrope_section"]))

    if cfg.num_experts > 0:
        _require("num_experts", cfg.num_experts, txt["num_experts"])
        _require("num_experts_per_tok", cfg.num_experts_per_tok, txt["num_experts_per_tok"])
        _require("moe_intermediate_size", cfg.moe_intermediate_size, txt["moe_intermediate_size"])
        _require("mlp_only_layers", tuple(cfg.mlp_only_layers), tuple(txt["mlp_only_layers"]))
        _require("decoder_sparse_step", cfg.decoder_sparse_step, txt["decoder_sparse_step"])
        _require("norm_topk_prob", cfg.norm_topk_prob, txt["norm_topk_prob"])

    _require("vision.hidden_size", cfg.vision.hidden_size, vis["hidden_size"])
    _require("vision.intermediate_size", cfg.vision.intermediate_size, vis["intermediate_size"])
    _require("vision.num_heads", cfg.vision.num_heads, vis["num_heads"])
    _require("vision.depth", cfg.vision.depth, vis["depth"])
    _require("vision.patch_size", cfg.vision.patch_size, vis["patch_size"])
    _require(
        "vision.temporal_patch_size", cfg.vision.temporal_patch_size, vis["temporal_patch_size"]
    )
    _require("vision.spatial_merge_size", cfg.vision.spatial_merge_size, vis["spatial_merge_size"])
    _require("vision.out_hidden_size", cfg.vision.out_hidden_size, vis["out_hidden_size"])
    _require(
        "vision.num_position_embeddings",
        cfg.vision.num_position_embeddings,
        vis["num_position_embeddings"],
    )


def _get_vision_key_mapping():
    T = Transform
    m: dict[str, tuple[str, Transform]] = {}
    m[r"model\.visual\.patch_embed\.proj\.weight"] = ("vision.patch_embed.proj.kernel", T.CONV3D)
    m[r"model\.visual\.patch_embed\.proj\.bias"] = ("vision.patch_embed.proj.bias", T.BIAS)
    m[r"model\.visual\.pos_embed\.weight"] = ("vision.pos_embed.embedding", T.EMBED)
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
    m[r"model\.visual\.merger\.norm\.weight"] = ("vision.merger.norm.scale", T.SCALE)
    m[r"model\.visual\.merger\.norm\.bias"] = ("vision.merger.norm.bias", T.BIAS)
    m[r"model\.visual\.merger\.linear_fc1\.weight"] = ("vision.merger.fc1.kernel", T.LINEAR)
    m[r"model\.visual\.merger\.linear_fc1\.bias"] = ("vision.merger.fc1.bias", T.BIAS)
    m[r"model\.visual\.merger\.linear_fc2\.weight"] = ("vision.merger.fc2.kernel", T.LINEAR)
    m[r"model\.visual\.merger\.linear_fc2\.bias"] = ("vision.merger.fc2.bias", T.BIAS)
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
    lyr = r"model\.language_model\.layers\.([0-9]+)"
    m[lyr + r"\.self_attn\.q_proj\.weight"] = (r"text.layers.\1.attn.q_proj.kernel", T.LINEAR)
    m[lyr + r"\.self_attn\.k_proj\.weight"] = (r"text.layers.\1.attn.k_proj.kernel", T.LINEAR)
    m[lyr + r"\.self_attn\.v_proj\.weight"] = (r"text.layers.\1.attn.v_proj.kernel", T.LINEAR)
    m[lyr + r"\.self_attn\.o_proj\.weight"] = (r"text.layers.\1.attn.o_proj.kernel", T.LINEAR)
    m[lyr + r"\.self_attn\.q_norm\.weight"] = (r"text.layers.\1.attn.q_norm.scale", T.SCALE)
    m[lyr + r"\.self_attn\.k_norm\.weight"] = (r"text.layers.\1.attn.k_norm.scale", T.SCALE)
    m[lyr + r"\.mlp\.gate_proj\.weight"] = (r"text.layers.\1.mlp.gate_proj.kernel", T.LINEAR)
    m[lyr + r"\.mlp\.up_proj\.weight"] = (r"text.layers.\1.mlp.up_proj.kernel", T.LINEAR)
    m[lyr + r"\.mlp\.down_proj\.weight"] = (r"text.layers.\1.mlp.down_proj.kernel", T.LINEAR)
    m[lyr + r"\.input_layernorm\.weight"] = (r"text.layers.\1.input_layernorm.scale", T.SCALE)
    m[lyr + r"\.post_attention_layernorm\.weight"] = (
        r"text.layers.\1.post_attention_layernorm.scale",
        T.SCALE,
    )
    m[r"model\.language_model\.norm\.weight"] = ("text.final_norm.scale", T.SCALE)
    m[r"lm_head\.weight"] = ("lm_head.kernel", T.LINEAR)
    return m


def _get_non_expert_mapping():
    """Mapping for all non-expert parameters (vision + dense text path)."""
    mapping = {}
    mapping.update(_get_vision_key_mapping())
    mapping.update(_get_text_key_mapping())
    return mapping


def _expected_dense_qwen3_vl_targets(
    cfg: Qwen3VLConfig,
    mapping: dict[str, tuple[str, Transform]],
) -> set[str]:
    expected: set[str] = set()
    for target, _ in mapping.values():
        if r"\1" not in target:
            if target != "lm_head.kernel" or not cfg.tie_word_embeddings:
                expected.add(target)
        elif target.startswith("text.layers."):
            expected.update(target.replace(r"\1", str(index)) for index in range(cfg.num_layers))
        elif target.startswith("vision.blocks."):
            expected.update(target.replace(r"\1", str(index)) for index in range(cfg.vision.depth))
        elif target.startswith("vision.deepstack_mergers."):
            expected.update(
                target.replace(r"\1", str(index))
                for index in range(len(cfg.vision.deepstack_visual_indexes))
            )
        else:
            raise RuntimeError(f"Unsupported dense Qwen3-VL mapping target: {target}")
    return expected


def _expected_dense_qwen3_vl_shape(key: str, cfg: Qwen3VLConfig) -> tuple[int, ...]:
    vision = cfg.vision
    merged = vision.hidden_size * vision.spatial_merge_size**2
    exact = {
        "lm_head.kernel": (cfg.emb_dim, cfg.vocab_size),
        "text.embedder.embedding": (cfg.vocab_size, cfg.emb_dim),
        "text.final_norm.scale": (cfg.emb_dim,),
        "vision.merger.norm.scale": (vision.hidden_size,),
        "vision.merger.norm.bias": (vision.hidden_size,),
        "vision.merger.fc1.kernel": (merged, merged),
        "vision.merger.fc1.bias": (merged,),
        "vision.merger.fc2.kernel": (merged, vision.out_hidden_size),
        "vision.merger.fc2.bias": (vision.out_hidden_size,),
        "vision.patch_embed.proj.kernel": (
            vision.in_channels * vision.temporal_patch_size * vision.patch_size**2,
            vision.hidden_size,
        ),
        "vision.patch_embed.proj.bias": (vision.hidden_size,),
        "vision.pos_embed.embedding": (
            vision.num_position_embeddings,
            vision.hidden_size,
        ),
    }
    if key in exact:
        return exact[key]
    suffix = key.split(".", 3)[-1]
    if key.startswith("text.layers."):
        shapes = {
            "attn.q_proj.kernel": (cfg.emb_dim, cfg.num_heads * cfg.head_dim),
            "attn.k_proj.kernel": (cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim),
            "attn.v_proj.kernel": (cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim),
            "attn.o_proj.kernel": (cfg.num_heads * cfg.head_dim, cfg.emb_dim),
            "attn.q_norm.scale": (cfg.head_dim,),
            "attn.k_norm.scale": (cfg.head_dim,),
            "mlp.gate_proj.kernel": (cfg.emb_dim, cfg.mlp_dim),
            "mlp.up_proj.kernel": (cfg.emb_dim, cfg.mlp_dim),
            "mlp.down_proj.kernel": (cfg.mlp_dim, cfg.emb_dim),
            "input_layernorm.scale": (cfg.emb_dim,),
            "post_attention_layernorm.scale": (cfg.emb_dim,),
        }
    elif key.startswith("vision.blocks."):
        shapes = {
            "norm1.scale": (vision.hidden_size,),
            "norm1.bias": (vision.hidden_size,),
            "attn.qkv.kernel": (vision.hidden_size, 3 * vision.hidden_size),
            "attn.qkv.bias": (3 * vision.hidden_size,),
            "attn.proj.kernel": (vision.hidden_size, vision.hidden_size),
            "attn.proj.bias": (vision.hidden_size,),
            "norm2.scale": (vision.hidden_size,),
            "norm2.bias": (vision.hidden_size,),
            "mlp.fc1.kernel": (vision.hidden_size, vision.intermediate_size),
            "mlp.fc1.bias": (vision.intermediate_size,),
            "mlp.fc2.kernel": (vision.intermediate_size, vision.hidden_size),
            "mlp.fc2.bias": (vision.hidden_size,),
        }
    elif key.startswith("vision.deepstack_mergers."):
        shapes = {
            "norm.scale": (merged,),
            "norm.bias": (merged,),
            "fc1.kernel": (merged, merged),
            "fc1.bias": (merged,),
            "fc2.kernel": (merged, vision.out_hidden_size),
            "fc2.bias": (vision.out_hidden_size,),
        }
    else:
        raise RuntimeError(f"Unsupported dense Qwen3-VL target shape: {key}")
    try:
        return shapes[suffix]
    except KeyError as error:
        raise RuntimeError(f"Unsupported dense Qwen3-VL target shape: {key}") from error


def _mapped_shape(source_shape: Sequence[int], transform: Transform) -> tuple[int, ...]:
    shape = tuple(source_shape)
    if transform is Transform.LINEAR:
        if len(shape) != 2:
            return shape
        return shape[::-1]
    if transform is Transform.CONV3D:
        if not shape:
            return shape
        flattened = 1
        for dimension in shape[1:]:
            flattened *= dimension
        return flattened, shape[0]
    return shape


def validate_dense_qwen3_vl_safetensors(
    files: Sequence[str | epath.Path],
    hf_cfg: dict,
) -> tuple[Qwen3VLConfig, int]:
    if not files:
        raise ValueError("Dense Qwen3-VL snapshot has no safetensors files")
    cfg = make_vl_config_from_hf(hf_cfg)
    _assert_vl_config(cfg, hf_cfg)
    if hf_cfg.get("model_type") != "qwen3_vl" or cfg.num_experts != 0:
        raise NotImplementedError(
            "Sealed-snapshot production loading currently supports dense Qwen3-VL only"
        )
    mapping = _get_non_expert_mapping()
    hf_keys: set[str] = set()
    jax_keys: set[str] = set()
    unmatched: list[str] = []
    for file in files:
        with safetensors.safe_open(file, framework="numpy") as weights:
            for hf_key in weights.keys():  # noqa: SIM118
                if hf_key in hf_keys:
                    raise ValueError(f"Duplicate Hugging Face tensor key: {hf_key}")
                hf_keys.add(hf_key)
                jax_key, transform = map_to_bonsai_key(mapping, hf_key)
                if jax_key is None:
                    unmatched.append(hf_key)
                elif jax_key in jax_keys:
                    raise ValueError(f"Multiple Hugging Face tensors map to {jax_key}")
                else:
                    actual_shape = _mapped_shape(weights.get_slice(hf_key).get_shape(), transform)
                    expected_shape = _expected_dense_qwen3_vl_shape(jax_key, cfg)
                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Dense Qwen3-VL tensor {hf_key!r} maps to {jax_key} with shape "
                            f"{actual_shape}, expected {expected_shape}"
                        )
                    jax_keys.add(jax_key)
    check_conversion_errors(unmatched)
    expected = _expected_dense_qwen3_vl_targets(cfg, mapping)
    if jax_keys != expected:
        raise ValueError(
            "Dense Qwen3-VL safetensors do not match the production loader mapping: "
            f"missing={sorted(expected - jax_keys)[:3]}, "
            f"unexpected={sorted(jax_keys - expected)[:3]}"
        )
    return cfg, len(hf_keys)


def create_qwen3_vl_from_safetensors(
    file_dir: str,
    model_id: str = "",
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> tuple[Qwen3VL, Qwen3VLConfig]:
    """Load HuggingFace Qwen3-VL safetensors into a JAX Qwen3-VL model."""
    path = epath.Path(file_dir).expanduser()
    files = find_safetensors(file_dir)
    hf_cfg = load_hf_config(path)
    return create_qwen3_vl_from_safetensor_files(
        files,
        hf_cfg,
        tp_size=tp_size,
        fsdp_size=fsdp_size,
        dp_size=dp_size,
    )


def create_qwen3_vl_from_safetensor_files(
    files: Sequence[str | epath.Path],
    hf_cfg: dict,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> tuple[Qwen3VL, Qwen3VLConfig]:
    """Load exact HuggingFace Qwen3-VL safetensor files into a JAX model."""
    if not files:
        raise ValueError("Qwen3-VL snapshot has no safetensors files")
    cfg = make_vl_config_from_hf(hf_cfg)
    _assert_vl_config(cfg, hf_cfg)
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    cfg = dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))

    # Build an *abstract* model (shapes/dtypes/shardings only, zero device bytes)
    # instead of materializing random weights we'd immediately overwrite. The throwaway
    # init was a full second copy of the params held live alongside the loaded weights,
    # doubling peak HBM during load (~70 GB for the 8B). `get_partition_spec` resolves
    # the logical axis annotations to concrete PartitionSpecs so each loaded tensor is
    # placed directly onto its FSDP/TP shard (eval_shape only attaches an AbstractMesh
    # sharding, which device_put cannot consume).
    with mesh_rules(mesh):
        model = nnx.eval_shape(lambda: Qwen3VL(cfg, rngs=nnx.Rngs(params=0)))
        graph_def, abs_state = nnx.split(model)
        pspec_dict = nnx.to_pure_dict(nnx.get_partition_spec(abs_state))
    state_dict = nnx.to_pure_dict(abs_state)

    non_expert_mapping = _get_non_expert_mapping()
    unmatched_hf_keys: list[str] = []

    is_moe = cfg.num_experts > 0
    if is_moe:
        E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size
        expert_arrays, expert_fill = init_expert_buffers(cfg.num_layers, E, D, F, cfg.is_moe_layer)
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

                if cfg.tie_word_embeddings and jax_key.startswith("lm_head"):
                    continue

                keys = [stoi(k) for k in jax_key.split(".")]
                if transform == Transform.CONV3D:
                    tensor = tensor.reshape(tensor.shape[0], -1).T
                    assign_weights_from_eval_shape(
                        keys, tensor, state_dict, torch_key, None, mesh=mesh, pspec_dict=pspec_dict
                    )
                else:
                    transform_value = (
                        transform.value
                        if transform not in (Transform.BIAS, Transform.EMBED, Transform.SCALE)
                        else None
                    )
                    assign_weights_from_eval_shape(
                        keys,
                        tensor,
                        state_dict,
                        torch_key,
                        transform_value,
                        mesh=mesh,
                        pspec_dict=pspec_dict,
                    )
        gc.collect()

    if is_moe:
        finalize_experts(
            expert_arrays,
            expert_fill,
            router_buf,
            state_dict,
            num_experts=cfg.num_experts,
            jax_layer_prefix="text.layers",
            mesh=mesh,
            pspec_dict=pspec_dict,
        )

    check_conversion_errors(unmatched_hf_keys)

    gc.collect()
    model = nnx.merge(graph_def, state_dict)
    _finalize_q_shardings(model, mesh)
    return model, cfg
