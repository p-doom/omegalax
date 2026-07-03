"""CPU tests for fp8 training gating and the qwix wrapping composition.

fp8 is Hopper-only. These tests run on CPU and validate the parts that do NOT
need fp8 tensor cores:

  * ``test_no_op_when_off``: with ``fp8=False`` the model-build wrap returns the
    model UNCHANGED (same object, not qwix-quantized) -- clean pass-through.
  * ``test_requesting_fp8_on_non_hopper_raises``: with ``fp8=True`` on a non-Hopper
    host (CPU here) the wrap ASSERTS rather than silently falling back to bf16.
  * ``test_wrapping_traces_under_force``: with the Hopper gate patched to True
    (bypassing it on CPU) qwix ``quantize_model`` wraps the smoke model and a
    forward TRACES + runs without error. This is the top-risk check from the
    fp8 design pass: the codebase uses ``out_sharding=`` pervasively and we
    confirm qwix's op interception composes with it. (Numerics fall back on
    CPU -- there are no fp8 tensor cores -- which is expected; we validate the
    wrapping composes, not the fp8 math.)
  * ``test_lora_adapters_excluded``: the fp8 rules quantize only the BASE
    projection GEMM, never the LoRA low-rank delta matmuls.
  * ``test_optimizer_state_untouched``: qwix scales never appear as trainable
    ``nnx.Param``; the ``wrt=nnx.Param`` grad set (and hence the optimizer
    state) is identical with and without the fp8 wrap.

Run (login-node CPU, torch-free)::

    JAX_PLATFORMS=cpu uv run --no-sync python -m unittest tests.test_fp8_gating
"""

import contextlib
import os
import unittest.mock as mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# The tokamax mosaic_gpu attention/ragged kernels are GPU-only; the pure-JAX
# xla deltanet reference is the CPU path. (Attention backend is set per-model
# via set_attn_backend below.)
os.environ.setdefault("OMEGALAX_DELTANET_KERNEL", "xla")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import make_mesh, mesh_rules
from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.shard_config import ShardConfig
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.quant import detect
from omegalax.quant.apply import maybe_quantize_fp8
from omegalax.quant.rules import build_provider, lora_delta_paths


@contextlib.contextmanager
def _force_fp8():
    """Patch the Hopper gate True and force the unrolled layer loop for the wrapping tests."""
    prev = detect.is_hopper
    detect.is_hopper = lambda: True
    try:
        with mock.patch.object(Qwen3Config, "is_homogeneous", property(lambda self: False)):
            yield
    finally:
        detect.is_hopper = prev


def _single_mesh():
    return make_mesh(tp_size=1, fsdp_size=1, dp_size=1)


def _dense_cfg(fp8=False, fp8_recipe="e4m3_dynamic"):
    return Qwen3Config(
        num_layers=1,
        vocab_size=128,
        emb_dim=32,
        mlp_dim=64,
        num_heads=2,
        head_dim=16,
        num_kv_heads=2,
        rope_theta=1_000_000,
        rope_scaling_factor=None,
        local_rope_theta=None,
        norm_eps=1e-6,
        tie_word_embeddings=False,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.float32,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
    )


def _build_model(cfg, mesh):
    with mesh_rules(mesh):
        model = Qwen3(cfg, rngs=nnx.Rngs(0))
    set_attn_backend(model, text_backend="xla")
    return model


class Fp8GatingTest(absltest.TestCase):
    def test_is_hopper_false_on_cpu(self):
        """The Hopper gate is False on a CPU-only host (unless forced)."""
        self.assertFalse(detect.is_hopper())

    def test_no_op_when_off(self):
        """fp8=False -> maybe_quantize_fp8 returns the model UNCHANGED."""
        mesh = _single_mesh()
        cfg = _dense_cfg(fp8=False)
        model = _build_model(cfg, mesh)
        self.assertFalse(detect.fp8_active(cfg))
        wrapped = maybe_quantize_fp8(model, cfg, mesh=mesh)
        # Same object, and NOT a qwix-quantized subclass.
        self.assertIs(wrapped, model)
        self.assertFalse(hasattr(type(wrapped), "_unquantized_type"))

    def test_requesting_fp8_on_non_hopper_raises(self):
        """fp8=True on a non-Hopper host (CPU) ASSERTS -- no silent bf16 fallback."""
        mesh = _single_mesh()
        cfg = _dense_cfg(fp8=True, fp8_recipe="e4m3_dynamic")
        model = _build_model(cfg, mesh)
        # fp8_active reflects config intent only (no hardware term).
        self.assertFalse(detect.is_hopper())
        self.assertTrue(detect.fp8_active(cfg))
        with self.assertRaisesRegex(AssertionError, "fp8 requires sm_90"):
            maybe_quantize_fp8(model, cfg, mesh=mesh)

    def test_off_recipe_is_clean_pass_through(self):
        """fp8_recipe='off' is a clean pass-through on any host (no assert, no wrap)."""
        mesh = _single_mesh()
        cfg = _dense_cfg(fp8=True, fp8_recipe="off")
        model = _build_model(cfg, mesh)
        self.assertFalse(detect.fp8_active(cfg))  # recipe 'off'
        wrapped = maybe_quantize_fp8(model, cfg, mesh=mesh)
        self.assertIs(wrapped, model)

    def test_wrapping_traces_under_force(self):
        """Forced fp8: qwix wraps the model and a fwd+bwd traces+runs on CPU.

        Uses the dense (non-MoE) smoke model: it exercises the top-risk qwix +
        ``out_sharding=`` composition on the attention / MLP / lm_head GEMMs. The
        grouped-MoE experts route through ``ragged_dot`` under grouped_moe's CPU-only
        ``_auto`` auto_axes wrapper, which clashes with qwix's re-trace mesh type on
        CPU; real fp8 runs on Hopper GPU (where ``_auto`` is a no-op passthrough), so
        fp8 x grouped-MoE is validated there (deferred, Hopper-only)."""
        mesh = _single_mesh()
        cfg = _dense_cfg(fp8=True, fp8_recipe="e4m3_dynamic")
        with _force_fp8():
            self.assertTrue(detect.fp8_active(cfg))
            model = _build_model(cfg, mesh)
            wrapped = maybe_quantize_fp8(model, cfg, mesh=mesh)
            # qwix produced a quantized subclass (interception installed).
            self.assertTrue(hasattr(type(wrapped), "_unquantized_type"))
            set_attn_backend(wrapped, text_backend="xla")

            tok = jnp.asarray(
                np.random.RandomState(0).randint(1, cfg.vocab_size, size=(2, 8)).astype(np.int32)
            )
            seg = jnp.ones((2, 8), dtype=jnp.int32)

            def loss_fn(m):
                h, aux = m(tok, seg, None, jnp.array(0, jnp.int32))
                return jnp.mean(m.lm_head(h) ** 2) + aux

            with jax.set_mesh(mesh):
                loss, grads = nnx.value_and_grad(loss_fn)(wrapped)
        self.assertTrue(np.isfinite(float(loss)))
        # Gradients exist for the base params.
        self.assertGreater(len(jax.tree_util.tree_leaves(grads)), 0)

    def test_blockwise_128_recipe_carries_tile_size(self):
        """The blockwise_128 recipe sets tile_size=128 on the quant rules.

        We assert rule construction rather than a full CPU trace: 128-wide
        subchannel tiling requires contraction axes >= 128, which the smoke
        model (dims of 32/64) does not have. The 128 tiling is exercised
        numerically on the 397B flagship on Hopper (see the deferred recipe).
        """
        provider = build_provider("blockwise_128")
        quant_rules = [r for r in provider._rules if r.weight_qtype is not None]
        self.assertTrue(quant_rules)
        for r in quant_rules:
            self.assertEqual(r.tile_size, 128)
        # The default recipe must NOT tile.
        for r in build_provider("e4m3_dynamic")._rules:
            if r.weight_qtype is not None:
                self.assertIsNone(r.tile_size)

    def test_lora_adapters_excluded(self):
        """fp8 rules quantize the BASE projection GEMM, never the LoRA delta.

        With LoRA injected, the frozen base lives at ``.../q_proj/base`` (rule
        quantizes it) and the low-rank delta matmuls run at ``.../q_proj`` (the
        LoRALinear scope, matched by an exclusion rule). We assert every
        LoRALinear delta path resolves to a no-quant rule and each base path to
        a quant rule.
        """
        import re

        from omegalax.trainers.lora import inject_lora

        mesh = _single_mesh()
        cfg = _dense_cfg(fp8=True, fp8_recipe="e4m3_dynamic")
        with mesh_rules(mesh):
            model = Qwen3(cfg, rngs=nnx.Rngs(0))
            n = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        self.assertGreater(n, 0)

        delta_paths = lora_delta_paths(model)
        self.assertEqual(len(delta_paths), n)  # one per LoRALinear
        provider = build_provider("e4m3_dynamic", lora_delta_paths=delta_paths)

        def _matched_rule(path):
            for rule in provider._rules:
                if re.fullmatch(rule.module_path, path):
                    return rule
            return None

        for dp in delta_paths:
            # The LoRA delta scope must resolve to an exclusion (no weight_qtype).
            rule = _matched_rule(dp)
            self.assertIsNotNone(rule, f"no rule matched delta path {dp}")
            self.assertIsNone(
                rule.weight_qtype,
                f"LoRA delta path {dp} was quantized (rule={rule.module_path!r})",
            )
            # The frozen base under it must resolve to a quant rule.
            base_rule = _matched_rule(dp + "/base")
            self.assertIsNotNone(base_rule, f"no rule matched base path {dp}/base")
            self.assertIsNotNone(
                base_rule.weight_qtype,
                f"base path {dp}/base was NOT quantized",
            )

    def test_lora_wrap_traces_under_force(self):
        """A LoRA-injected model still wraps + traces under forced fp8 (base
        quantized, adapters excluded), fwd runs without error."""
        from omegalax.trainers.lora import inject_lora

        mesh = _single_mesh()
        cfg = _dense_cfg(fp8=True, fp8_recipe="e4m3_dynamic")
        with _force_fp8():
            with mesh_rules(mesh):
                model = Qwen3(cfg, rngs=nnx.Rngs(0))
                inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
            set_attn_backend(model, text_backend="xla")
            wrapped = maybe_quantize_fp8(model, cfg, mesh=mesh)
            self.assertTrue(hasattr(type(wrapped), "_unquantized_type"))
            set_attn_backend(wrapped, text_backend="xla")
            tok = jnp.ones((1, 8), dtype=jnp.int32)
            seg = jnp.ones((1, 8), dtype=jnp.int32)
            with jax.set_mesh(mesh):
                out, _ = nnx.jit(lambda m: m(tok, seg, None, jnp.array(0, jnp.int32)))(wrapped)
        self.assertEqual(out.shape, (1, 8, cfg.emb_dim))

    def test_optimizer_state_untouched(self):
        """The trainable-param set (wrt=nnx.Param) is identical with/without the
        fp8 wrap: qwix scales live in the quant_stats collection, never as
        nnx.Param, so MixedPrecisionOptimizer never sees them."""
        mesh = _single_mesh()
        cfg = _dense_cfg(fp8=True, fp8_recipe="e4m3_dynamic")

        # Baseline: unwrapped trainable-param paths.
        base_model = _build_model(cfg, mesh)
        base_params = nnx.state(base_model, nnx.Param)
        base_paths = {
            jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_flatten_with_path(base_params)[0]
        }

        with _force_fp8():
            wrapped = maybe_quantize_fp8(_build_model(cfg, mesh), cfg, mesh=mesh)
        wrapped_params = nnx.state(wrapped, nnx.Param)
        wrapped_paths = {
            jax.tree_util.keystr(k)
            for k, _ in jax.tree_util.tree_flatten_with_path(wrapped_params)[0]
        }
        # The fp8 wrap must not add or remove any trainable nnx.Param leaf.
        self.assertEqual(base_paths, wrapped_paths)


if __name__ == "__main__":
    absltest.main()
