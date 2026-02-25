"""Verify that positional embeddings are generated in float32 and that RoPE is applied in the model dtype.

Tests cover all three model families: Qwen3 (dense + MoE), Qwen3.5 (text + vision), and Qwen3-VL.

Rather than testing `astype` in isolation, each test monkeypatches the rope generation and
application functions to capture the dtypes that actually flow through the model's forward pass.
"""

import dataclasses
import os
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from absl.testing import absltest

MODEL_DTYPE = jnp.bfloat16


def _make_spy(original_fn, captured: list):
    """Wrap *original_fn* so every call appends {arg_name: dtype} for all array args."""

    def spy(*args, **kwargs):
        dtypes = {}
        import inspect
        sig = inspect.signature(original_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, val in bound.arguments.items():
            if hasattr(val, "dtype"):
                dtypes[name] = val.dtype
        captured.append(dtypes)
        return original_fn(*args, **kwargs)

    return spy


class Qwen3RopeDtypeTest(absltest.TestCase):
    """Qwen3 dense Attention: generate in fp32, apply_rope receives model-dtype sin/cos."""

    def test_rope_dtype_through_attention_forward(self):
        from flax import nnx
        from omegalax.models.qwen3.dense.config import make_dense_config
        from omegalax.models.qwen3 import attention as attn_mod
        from omegalax.models.qwen3 import rope as rope_mod

        cfg = dataclasses.replace(make_dense_config("qwen3-smoke"), dtype=MODEL_DTYPE)
        attn = attn_mod.Attention(cfg, rngs=nnx.Rngs(0))

        gen_calls, apply_calls = [], []
        orig_gen = rope_mod.generate_pos_embeddings
        orig_apply = rope_mod.apply_rope

        gen_spy = _make_spy(orig_gen, gen_calls)
        apply_spy = _make_spy(orig_apply, apply_calls)

        B, T = 1, 8
        hidden = jnp.ones((B, T, cfg.emb_dim), dtype=MODEL_DTYPE)
        seg = jnp.ones((B, T), dtype=jnp.int32)

        with mock.patch.object(attn_mod, "generate_pos_embeddings", gen_spy), \
             mock.patch.object(attn_mod, "apply_rope", apply_spy):
            attn(hidden, None, seg)

        self.assertLen(gen_calls, 1)
        gen_out_sin, gen_out_cos = orig_gen(jnp.arange(T)[None, :], cfg.head_dim)
        self.assertEqual(gen_out_sin.dtype, jnp.float32, "generate_pos_embeddings must produce float32")
        self.assertEqual(gen_out_cos.dtype, jnp.float32, "generate_pos_embeddings must produce float32")

        self.assertLen(apply_calls, 2, "apply_rope should be called for q and k")
        for i, call in enumerate(apply_calls):
            self.assertEqual(call["sin_BTK"], MODEL_DTYPE,
                             f"apply_rope call {i}: sin_BTK should be {MODEL_DTYPE}")
            self.assertEqual(call["cos_BTK"], MODEL_DTYPE,
                             f"apply_rope call {i}: cos_BTK should be {MODEL_DTYPE}")
            self.assertEqual(call["x_BTHK"], MODEL_DTYPE,
                             f"apply_rope call {i}: x_BTHK should be {MODEL_DTYPE}")


class Qwen3_5TextRopeDtypeTest(absltest.TestCase):
    """Qwen3.5 TextModel: generate_text_rope in fp32, cos/sin cast to model dtype before layers."""

    def test_rope_dtype_through_text_forward(self):
        from flax import nnx
        from omegalax.models.qwen3_5.config import make_config
        from omegalax.models.qwen3_5 import model as model_mod
        from omegalax.models.qwen3_5 import rope as rope_mod

        cfg = make_config("qwen3.5-smoke")
        text_cfg = dataclasses.replace(cfg.text_config, dtype=MODEL_DTYPE)
        model = model_mod.Qwen3_5ForCausalLM(text_cfg, rngs=nnx.Rngs(0))

        gen_calls = []
        orig_gen = rope_mod.generate_text_rope
        gen_spy = _make_spy(orig_gen, gen_calls)

        layer_calls = []
        orig_layer_call = model_mod.DecoderLayer.__call__

        def layer_spy(self_layer, hidden, cos_BTK, sin_BTK, seg, pos, attn_mask=None):
            layer_calls.append({"cos": cos_BTK.dtype, "sin": sin_BTK.dtype})
            return orig_layer_call(self_layer, hidden, cos_BTK, sin_BTK, seg, pos, attn_mask)

        B, T = 1, 8
        tokens = jnp.ones((B, T), dtype=jnp.int32)
        seg = jnp.ones((B, T), dtype=jnp.int32)

        with mock.patch.object(model_mod, "generate_text_rope", gen_spy), \
             mock.patch.object(model_mod.DecoderLayer, "__call__", layer_spy):
            model(tokens, seg, None, jnp.array(0, dtype=jnp.int32))

        self.assertLen(gen_calls, 1)
        gen_dtypes = gen_calls[0]
        self.assertEqual(gen_dtypes.get("position_ids_ZBT"), jnp.int32)

        gen_out_cos, gen_out_sin = orig_gen(
            jnp.stack([jnp.arange(T)[None, :]] * 3, axis=0),
            text_cfg.head_dim, text_cfg.partial_rotary_factor,
            text_cfg.rope_theta, text_cfg.mrope_section,
        )
        self.assertEqual(gen_out_cos.dtype, jnp.float32, "generate_text_rope must produce float32")
        self.assertEqual(gen_out_sin.dtype, jnp.float32, "generate_text_rope must produce float32")

        self.assertGreater(len(layer_calls), 0)
        for i, call in enumerate(layer_calls):
            self.assertEqual(call["cos"], MODEL_DTYPE,
                             f"Layer {i}: cos_BTK should be {MODEL_DTYPE} after cast")
            self.assertEqual(call["sin"], MODEL_DTYPE,
                             f"Layer {i}: sin_BTK should be {MODEL_DTYPE} after cast")


class Qwen3_5VisionRopeDtypeTest(absltest.TestCase):
    """Qwen3.5 VisionModel: vision RoPE generated in fp32, cos/sin cast to vision dtype before blocks."""

    def test_rope_dtype_through_vision_forward(self):
        from flax import nnx
        from omegalax.models.qwen3_5.config import make_config
        from omegalax.models.qwen3_5 import vision as vis_mod

        cfg = make_config("qwen3.5-smoke")
        vis_cfg = cfg.vision_config

        vision = vis_mod.VisionModel(vis_cfg, rngs=nnx.Rngs(0))

        block_calls = []

        def block_spy(self_blk, hidden, cu_seqlens, cos_NK, sin_NK):
            block_calls.append({"cos": cos_NK.dtype, "sin": sin_NK.dtype})
            return hidden  # skip actual block to avoid JAX dynamic-slice tracing issue

        ms = vis_cfg.spatial_merge_size
        h, w = 2 * ms, 2 * ms
        n_patches = h * w
        tp = vis_cfg.temporal_patch_size
        ps = vis_cfg.patch_size
        c = vis_cfg.in_channels
        pixels = jnp.ones((n_patches, c * tp * ps * ps), dtype=jnp.float32)
        grid_thw = jnp.array([[1, h, w]], dtype=jnp.int32)

        with mock.patch.object(vis_mod.VisionBlock, "__call__", block_spy):
            vision(pixels, grid_thw)

        self.assertGreater(len(block_calls), 0)
        for i, call in enumerate(block_calls):
            self.assertEqual(call["cos"], vis_cfg.dtype,
                             f"Vision block {i}: cos_NK should be {vis_cfg.dtype}")
            self.assertEqual(call["sin"], vis_cfg.dtype,
                             f"Vision block {i}: sin_NK should be {vis_cfg.dtype}")


class Qwen3VLRopeDtypeTest(absltest.TestCase):
    """Qwen3-VL: compute_mrope_pos_embeddings in fp32, sin/cos cast to model dtype before layers."""

    def test_rope_dtype_through_text_forward(self):
        from flax import nnx
        from omegalax.models.qwen3_vl.config import make_vl_config
        from omegalax.models.qwen3_vl import model as model_mod

        base_cfg = make_vl_config("qwen3-vl-smoke")
        cfg = dataclasses.replace(base_cfg, dtype=MODEL_DTYPE)
        model = model_mod.Qwen3VL(cfg, rngs=nnx.Rngs(0))

        gen_calls = []
        orig_gen = model_mod.compute_mrope_pos_embeddings
        gen_spy = _make_spy(orig_gen, gen_calls)

        layer_calls = []
        orig_layer_call = model_mod.TextDecoderLayer.__call__

        def layer_spy(self_layer, hidden, sin_BTK, cos_BTK, mask):
            layer_calls.append({"sin": sin_BTK.dtype, "cos": cos_BTK.dtype})
            return orig_layer_call(self_layer, hidden, sin_BTK, cos_BTK, mask)

        B, T = 1, 8
        tokens = jnp.ones((B, T), dtype=jnp.int32)
        attn_mask = jnp.ones((B, T), dtype=jnp.int32)

        with mock.patch.object(model_mod, "compute_mrope_pos_embeddings", gen_spy), \
             mock.patch.object(model_mod.TextDecoderLayer, "__call__", layer_spy):
            model(tokens, attn_mask)

        self.assertLen(gen_calls, 1)
        gen_out_sin, gen_out_cos = orig_gen(
            jnp.stack([jnp.arange(T)[None, :]] * 3, axis=0),
            cfg.head_dim, cfg.rope_theta, cfg.mrope_section,
        )
        self.assertEqual(gen_out_sin.dtype, jnp.float32,
                         "compute_mrope_pos_embeddings must produce float32 sin")
        self.assertEqual(gen_out_cos.dtype, jnp.float32,
                         "compute_mrope_pos_embeddings must produce float32 cos")

        self.assertGreater(len(layer_calls), 0)
        for i, call in enumerate(layer_calls):
            self.assertEqual(call["sin"], MODEL_DTYPE,
                             f"Layer {i}: sin_BTK should be {MODEL_DTYPE} after cast")
            self.assertEqual(call["cos"], MODEL_DTYPE,
                             f"Layer {i}: cos_BTK should be {MODEL_DTYPE} after cast")


class Qwen3VLVisionRopeDtypeTest(absltest.TestCase):
    """Qwen3-VL VisionModel: vision RoPE generated in fp32, cos/sin cast to vision dtype before blocks."""

    def test_rope_dtype_through_vision_forward(self):
        from flax import nnx
        from omegalax.models.qwen3_vl.config import make_vl_config
        from omegalax.models.qwen3_vl import vision as vis_mod

        base_cfg = make_vl_config("qwen3-vl-smoke")
        vis_cfg = base_cfg.vision

        vision = vis_mod.VisionModel(vis_cfg, rngs=nnx.Rngs(0))

        block_calls = []

        def block_spy(self_blk, hidden, cu_seqlens, cos_NK, sin_NK):
            block_calls.append({"cos": cos_NK.dtype, "sin": sin_NK.dtype})
            return hidden  # skip actual block to avoid JAX dynamic-slice tracing issue

        ms = vis_cfg.spatial_merge_size
        h, w = 2 * ms, 2 * ms
        n_patches = h * w
        tp = vis_cfg.temporal_patch_size
        ps = vis_cfg.patch_size
        c = vis_cfg.in_channels
        pixels = jnp.ones((n_patches, c * tp * ps * ps), dtype=jnp.float32)
        grid_thw = jnp.array([[1, h, w]], dtype=jnp.int32)

        with mock.patch.object(vis_mod.VisionBlock, "__call__", block_spy):
            vision(pixels, grid_thw)

        self.assertGreater(len(block_calls), 0)
        for i, call in enumerate(block_calls):
            self.assertEqual(call["cos"], vis_cfg.dtype,
                             f"Vision block {i}: cos_NK should be {vis_cfg.dtype}")
            self.assertEqual(call["sin"], vis_cfg.dtype,
                             f"Vision block {i}: sin_NK should be {vis_cfg.dtype}")


if __name__ == "__main__":
    absltest.main()
