"""Verify Qwen3-VL attention honors the compute dtype (the "bf16 thing").

The GPU attention kernels tokamax dispatches to (``mosaic_gpu`` for text,
cuDNN flash for vision) only run in fp16/bf16. Historically q/k/v were
*unconditionally* cast to bf16 right before the kernel, so an fp32-configured
VL model silently ran attention in bf16 -- a precision-contract violation.

These tests assert the fixed behavior, gated on the compute dtype:

* bf16 config -> q/k/v are still cast to bf16 and the configured GPU backend is
  used (behavior unchanged / bit-identical to the old code path).
* fp32 config -> q/k/v flow into the attention op in fp32 (no downcast) and an
  fp32-capable path is used (the pure-jax "xla" backend for text; a pure-jax
  packed reference for vision), with fp32 output.

All tests run on CPU: the fp32 paths execute the real (CPU-capable) xla/pure-jax
kernels; the bf16 paths spy on / stub the GPU-only kernels to capture the dtypes
of the arguments they *would* receive (which is what proves the cast happened).
"""

import dataclasses
import os
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_vl.config import make_vl_config
from omegalax.models.qwen3_vl import model as model_mod
from omegalax.models.qwen3_vl import vision as vision_mod


class _AttnDtypeTestBase(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1))


class TextAttentionDtypeTest(_AttnDtypeTestBase):
    """TextAttention: cast/backend must be gated on cfg.dtype."""

    def _run_capture(self, dtype):
        cfg = dataclasses.replace(make_vl_config("qwen3-vl-smoke"), dtype=dtype)
        attn = model_mod.TextAttention(cfg, rngs=nnx.Rngs(0))

        B, T = 1, 8
        hidden = jnp.ones((B, T, cfg.emb_dim), dtype=dtype)
        # M-RoPE sin/cos are head_dim // 2 wide (see compute_mrope_pos_embeddings).
        sin = jnp.zeros((B, T, cfg.head_dim // 2), dtype=dtype)
        cos = jnp.ones((B, T, cfg.head_dim // 2), dtype=dtype)

        captured = {}
        orig = model_mod.dot_product_attention

        def spy(q, k, v, *args, **kwargs):
            captured["q"] = q.dtype
            captured["k"] = k.dtype
            captured["v"] = v.dtype
            captured["implementation"] = kwargs.get("implementation")
            # For the fp32 branch this is the real CPU-capable xla kernel; for
            # bf16 we stub it (mosaic_gpu is GPU-only) with a correctly-typed,
            # correctly-shaped zero so the module forward pass completes.
            if kwargs.get("implementation") == "xla":
                return orig(q, k, v, *args, **kwargs)
            return jnp.zeros(q.shape, dtype=q.dtype)

        with mock.patch.object(model_mod, "dot_product_attention", spy):
            out = attn(hidden, sin, cos)
        return captured, out

    def test_fp32_runs_attention_in_fp32_via_xla(self):
        captured, out = self._run_capture(jnp.float32)
        self.assertEqual(captured["q"], jnp.float32, "fp32 config must NOT downcast q")
        self.assertEqual(captured["k"], jnp.float32, "fp32 config must NOT downcast k")
        self.assertEqual(captured["v"], jnp.float32, "fp32 config must NOT downcast v")
        self.assertEqual(
            captured["implementation"], "xla", "fp32 must route to the fp32-capable xla backend"
        )
        self.assertEqual(out.dtype, jnp.float32, "fp32 attention output must stay fp32")

    def test_bf16_still_casts_to_bf16_and_uses_gpu_backend(self):
        attn_backend = model_mod.TextAttention(
            dataclasses.replace(make_vl_config("qwen3-vl-smoke"), dtype=jnp.bfloat16),
            rngs=nnx.Rngs(0),
        )._attn_backend
        captured, out = self._run_capture(jnp.bfloat16)
        self.assertEqual(captured["q"], jnp.bfloat16, "bf16 config must cast q to bf16 (unchanged)")
        self.assertEqual(captured["k"], jnp.bfloat16, "bf16 config must cast k to bf16 (unchanged)")
        self.assertEqual(captured["v"], jnp.bfloat16, "bf16 config must cast v to bf16 (unchanged)")
        self.assertEqual(
            captured["implementation"],
            attn_backend,
            "bf16 must use the configured GPU backend (unchanged)",
        )
        self.assertEqual(out.dtype, jnp.bfloat16, "bf16 attention output must stay bf16")


class VisionAttentionDtypeTest(_AttnDtypeTestBase):
    """Packed vision attention: cuDNN bf16 cast must be gated on the input dtype."""

    def _make_qkv(self, dtype, n=6, h=2, k=4):
        rng = jax.random.PRNGKey(0)
        q, kk, v = (
            jax.random.normal(r, (n, h, k), dtype=jnp.float32).astype(dtype)
            for r in jax.random.split(rng, 3)
        )
        # Two image segments: [0,3) and [3,6).
        cu = jnp.array([0, 3, n], dtype=jnp.int32)
        seqlens = jnp.array([3, n - 3], dtype=jnp.int32)
        return q, kk, v, cu, seqlens

    def test_fp32_uses_fp32_reference_no_downcast(self):
        q, k, v, cu, seqlens = self._make_qkv(jnp.float32)

        captured = {}
        orig_ref = vision_mod._xla_packed_vision_attention_local

        def ref_spy(q_, k_, v_, cu_, scale):
            captured["q"] = q_.dtype
            captured["k"] = k_.dtype
            captured["v"] = v_.dtype
            return orig_ref(q_, k_, v_, cu_, scale)

        with mock.patch.object(vision_mod, "_cudnn_dot_product_attention") as cudnn_stub:
            with mock.patch.object(vision_mod, "_xla_packed_vision_attention_local", ref_spy):
                out = vision_mod._cudnn_packed_vision_attention_local(
                    q, k, v, cu, seqlens, scale=0.5
                )
        cudnn_stub.assert_not_called()  # fp32 must NOT touch the cuDNN bf16 kernel
        self.assertEqual(captured["q"], jnp.float32, "fp32 vision must NOT downcast q")
        self.assertEqual(captured["k"], jnp.float32, "fp32 vision must NOT downcast k")
        self.assertEqual(captured["v"], jnp.float32, "fp32 vision must NOT downcast v")
        self.assertEqual(out.dtype, jnp.float32, "fp32 vision attention output must stay fp32")

        # Numerical sanity: block-diagonal reference matches a manual per-segment
        # softmax attention, and cross-segment tokens do not interact.
        ref = self._manual_packed_attention(q, k, v, [(0, 3), (3, 6)], scale=0.5)
        self.assertTrue(jnp.allclose(out, ref, atol=1e-5), "fp32 vision output mismatch vs manual")

    def test_bf16_casts_to_bf16_and_calls_cudnn(self):
        q, k, v, cu, seqlens = self._make_qkv(jnp.bfloat16)

        captured = {}

        def cudnn_stub(q_, k_, v_, **kwargs):
            captured["q"] = q_.dtype
            captured["k"] = k_.dtype
            captured["v"] = v_.dtype
            return jnp.zeros_like(q_)

        with mock.patch.object(vision_mod, "_cudnn_dot_product_attention", cudnn_stub):
            out = vision_mod._cudnn_packed_vision_attention_local(q, k, v, cu, seqlens, scale=0.5)

        self.assertEqual(captured["q"], jnp.bfloat16, "bf16 vision must cast q to bf16 (unchanged)")
        self.assertEqual(captured["k"], jnp.bfloat16, "bf16 vision must cast k to bf16 (unchanged)")
        self.assertEqual(captured["v"], jnp.bfloat16, "bf16 vision must cast v to bf16 (unchanged)")
        self.assertEqual(out.dtype, jnp.bfloat16, "bf16 vision attention output must stay bf16")

    @staticmethod
    def _manual_packed_attention(q, k, v, segments, scale):
        out = jnp.zeros_like(q)
        for lo, hi in segments:
            qs, ks, vs = q[lo:hi], k[lo:hi], v[lo:hi]
            logits = jnp.einsum("nhk,shk->hns", qs, ks) * scale
            w = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(logits.dtype)
            seg_out = jnp.einsum("hns,shk->nhk", w, vs)
            out = out.at[lo:hi].set(seg_out)
        return out


if __name__ == "__main__":
    absltest.main()
