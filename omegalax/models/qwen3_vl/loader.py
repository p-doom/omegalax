"""Weight conversion from HuggingFace Qwen3-VL safetensors to JAX."""

from __future__ import annotations

import gc
from enum import Enum

import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx

from omegalax.models.params_utils import assign_weights_from_eval_shape, load_hf_config, map_to_bonsai_key, stoi
from .config import Qwen3VLConfig, make_vl_config_from_hf
from .model import Qwen3VL


class _Transform(Enum):
    LINEAR = ((1, 0), None, False)
    EMBED = None
    SCALE = None
    BIAS = None
    CONV3D = "conv3d"


def _get_key_and_transform_mapping(cfg: Qwen3VLConfig):
    T = _Transform
    m: dict[str, tuple[str, _Transform]] = {}

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

    # Vision: merger
    m[r"model\.visual\.merger\.norm\.weight"] = ("vision.merger.norm.scale", T.SCALE)
    m[r"model\.visual\.merger\.norm\.bias"] = ("vision.merger.norm.bias", T.BIAS)
    m[r"model\.visual\.merger\.linear_fc1\.weight"] = ("vision.merger.fc1.kernel", T.LINEAR)
    m[r"model\.visual\.merger\.linear_fc1\.bias"] = ("vision.merger.fc1.bias", T.BIAS)
    m[r"model\.visual\.merger\.linear_fc2\.weight"] = ("vision.merger.fc2.kernel", T.LINEAR)
    m[r"model\.visual\.merger\.linear_fc2\.bias"] = ("vision.merger.fc2.bias", T.BIAS)

    # Vision: deepstack mergers
    d = r"model\.visual\.deepstack_merger_list\.([0-9]+)"
    m[d + r"\.norm\.weight"] = (r"vision.deepstack_mergers.\1.norm.scale", T.SCALE)
    m[d + r"\.norm\.bias"] = (r"vision.deepstack_mergers.\1.norm.bias", T.BIAS)
    m[d + r"\.linear_fc1\.weight"] = (r"vision.deepstack_mergers.\1.fc1.kernel", T.LINEAR)
    m[d + r"\.linear_fc1\.bias"] = (r"vision.deepstack_mergers.\1.fc1.bias", T.BIAS)
    m[d + r"\.linear_fc2\.weight"] = (r"vision.deepstack_mergers.\1.fc2.kernel", T.LINEAR)
    m[d + r"\.linear_fc2\.bias"] = (r"vision.deepstack_mergers.\1.fc2.bias", T.BIAS)

    # Text
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


def create_qwen3_vl_from_safetensors(file_dir: str, model_id: str = "") -> Qwen3VL:
    """Load HuggingFace Qwen3-VL weights into a JAX Qwen3-VL model."""
    path = epath.Path(file_dir).expanduser()
    files = list(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    hf_cfg = load_hf_config(path)
    cfg = make_vl_config_from_hf(hf_cfg)

    qwen3_vl = nnx.eval_shape(lambda: Qwen3VL(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3_vl)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors: list[str] = []
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
                    if transform == _Transform.CONV3D:
                        tensor = tensor.reshape(tensor.shape[0], -1).T
                        assign_weights_from_eval_shape(keys, tensor, state_dict, torch_key, None)
                    else:
                        transform_value = transform.value if transform not in (_Transform.BIAS, _Transform.EMBED, _Transform.SCALE) else None
                        assign_weights_from_eval_shape(keys, tensor, state_dict, torch_key, transform_value)
                except Exception as e:
                    full_jax_key = ".".join(str(k) for k in keys)
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors:\n" + "\n".join(conversion_errors)
        )

    if unmatched_hf_keys:
        raise RuntimeError(f"Unmapped HuggingFace parameters:\n" + "\n".join(sorted(unmatched_hf_keys)))

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["text"]["embedder"]["embedding"].T

    gc.collect()
    return nnx.merge(graph_def, state_dict), cfg
