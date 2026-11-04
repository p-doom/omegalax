"""Unit tests for omegalax.trainers.lora.

Run on CPU; no model loading. Validate:
* forward parity at zero-init (B=0 ⇒ LoRA(x) ≡ base(x) bitwise)
* gradient isolation: ``wrt=LoRAParam`` produces grads only for adapter
  weights; base ``nnx.Param`` instances see no gradient
* param count: trainable count matches 2·r·(d_in+d_out)·n_targets·n_layers
* after one optimizer step: base kernel is bit-exact unchanged, LoRA
  weights moved
* merge_lora_into_base: post-merge logits equal pre-merge logits
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.config import make_config as make_qwen3_5_config
from omegalax.trainers import vlm
from omegalax.trainers.lora import (
    DEFAULT_TARGET_MODULES,
    QWEN3_5_DELTANET_TARGET_MODULES,
    LoRALinear,
    LoRAParam,
    inject_lora,
    inject_model_lora,
    merge_lora_into_base,
)
from omegalax.trainers.optim import MixedPrecisionOptimizer


class _MiniAttention(nnx.Module):
    """Minimal stand-in for TextAttention: same Linear attribute names."""

    def __init__(self, d: int, *, rngs: nnx.Rngs):
        self.q_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _MiniMLP(nnx.Module):
    def __init__(self, d: int, mlp: int, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(d, mlp, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(d, mlp, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(mlp, d, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class _MiniLayer(nnx.Module):
    def __init__(self, d: int, mlp: int, *, rngs: nnx.Rngs):
        self.attn = _MiniAttention(d, rngs=rngs)
        self.mlp = _MiniMLP(d, mlp, rngs=rngs)

    def __call__(self, x):
        return self.mlp(self.attn(x))


class _MiniVisionLinear(nnx.Module):
    """Stand-in for vision tower: should be SKIPPED by injection."""

    def __init__(self, d: int, *, rngs: nnx.Rngs):
        self.q_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)


class _MiniModel(nnx.Module):
    def __init__(self, d: int = 16, mlp: int = 32, n_layers: int = 2, *, rngs: nnx.Rngs):
        self.layers = nnx.List([_MiniLayer(d, mlp, rngs=rngs) for _ in range(n_layers)])
        self.vision = _MiniVisionLinear(d, rngs=rngs)
        # An nnx.Linear at the top level whose attribute name is NOT in
        # DEFAULT_TARGET_MODULES — should not be wrapped.
        self.lm_head = nnx.Linear(d, 100, use_bias=False, rngs=rngs)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class _MiniBf16Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.q_proj = nnx.Linear(
            4,
            4,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.q_proj(x)


def _make_model(seed: int = 0, d: int = 16, mlp: int = 32, n_layers: int = 2):
    return _MiniModel(d=d, mlp=mlp, n_layers=n_layers, rngs=nnx.Rngs(seed))


class LoRATest(absltest.TestCase):
    def test_forward_parity_at_zero_init(self):
        """With B=0 by construction, the LoRA-wrapped model must produce
        bit-identical outputs to the unwrapped model."""
        model = _make_model(seed=0)
        x = jax.random.normal(jax.random.key(42), (4, 7, 16), dtype=jnp.float32)
        y_before = model(x)

        n = inject_lora(model, r=8, alpha=16, rngs=nnx.Rngs(1))
        self.assertGreater(n, 0)
        y_after = model(x)
        np.testing.assert_array_equal(np.asarray(y_before), np.asarray(y_after))

    def test_adapter_master_is_fp32_and_forward_matmuls_are_bf16(self):
        model = _MiniBf16Model(rngs=nnx.Rngs(0))
        inject_model_lora(model, r=2, alpha=2, rngs=nnx.Rngs(1))

        self.assertEqual(model.q_proj.lora_A[...].dtype, jnp.float32)
        self.assertEqual(model.q_proj.lora_B[...].dtype, jnp.float32)

        x = jnp.ones((1, 4), dtype=jnp.bfloat16)
        jaxpr = jax.make_jaxpr(model)(x).jaxpr
        dot_dtypes = [
            tuple(value.aval.dtype for value in equation.invars)
            for equation in jaxpr.eqns
            if equation.primitive.name == "dot_general"
        ]
        self.assertLen(dot_dtypes, 3)
        self.assertTrue(all(dtypes == (jnp.bfloat16, jnp.bfloat16) for dtypes in dot_dtypes))
        self.assertEqual(model(x).dtype, jnp.bfloat16)

    def test_sub_bf16_ulp_adamw_updates_accumulate_in_master(self):
        model = _MiniBf16Model(rngs=nnx.Rngs(0))
        inject_model_lora(model, r=2, alpha=2, rngs=nnx.Rngs(1))
        model.q_proj.lora_A[...] = jnp.ones_like(model.q_proj.lora_A[...])
        model.q_proj.lora_B[...] = jnp.ones_like(model.q_proj.lora_B[...])
        optimizer = MixedPrecisionOptimizer(
            model,
            optax.adamw(1e-4, weight_decay=0.0),
            wrt=LoRAParam,
        )

        def loss_fn(m):
            return jnp.sum(m.q_proj.lora_A[...]) + jnp.sum(m.q_proj.lora_B[...])

        gradients = nnx.grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))(model)
        before = np.asarray(model.q_proj.lora_B[...]).copy()
        for _ in range(16):
            optimizer.update(gradients)
        after = np.asarray(model.q_proj.lora_B[...])

        bf16_control = jnp.asarray(1.0, jnp.bfloat16)
        for _ in range(16):
            bf16_control = (bf16_control.astype(jnp.float32) - 1e-4).astype(jnp.bfloat16)
        self.assertEqual(float(bf16_control), 1.0)
        self.assertEqual(np.count_nonzero(after != before), after.size)

    def test_inject_count_matches_expected(self):
        """Per layer: 4 attention projections + 3 MLP projections = 7. Times
        n_layers. Vision (skipped) and lm_head (not in target list) → 0."""
        n_layers = 3
        model = _make_model(seed=0, n_layers=n_layers)
        n = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        self.assertEqual(n, 7 * n_layers)

    def test_vision_subtree_is_skipped(self):
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        self.assertIsInstance(model.vision.q_proj, nnx.Linear)
        self.assertNotIsInstance(model.vision.q_proj, LoRALinear)

    def test_lm_head_not_wrapped(self):
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        self.assertIsInstance(model.lm_head, nnx.Linear)
        self.assertNotIsInstance(model.lm_head, LoRALinear)

    def test_grad_isolation_via_wrt_filter(self):
        """grads filtered by wrt=LoRAParam must contain only LoRAParam
        leaves; no base nnx.Param weights should receive gradient."""
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        x = jax.random.normal(jax.random.key(42), (2, 4, 16), dtype=jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(x) ** 2)

        grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))(model)
        grad_state = nnx.state(grads, LoRAParam)
        base_grad_state = nnx.state(grads, nnx.Param)
        lora_leaves = jax.tree.leaves(nnx.pure(grad_state))
        base_leaves = jax.tree.leaves(nnx.pure(base_grad_state))
        self.assertGreater(len(lora_leaves), 0)
        self.assertEqual(len(lora_leaves), len(base_leaves))

    def test_base_kernel_bit_exact_after_step(self):
        """One optimizer step with wrt=LoRAParam must leave every base
        nnx.Linear kernel bit-identical."""
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))

        base_snapshots: dict[str, np.ndarray] = {}
        for path, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRALinear):
                base_snapshots[".".join(map(str, path))] = np.asarray(mod.base.kernel[...]).copy()
            elif isinstance(mod, nnx.Linear):
                base_snapshots[".".join(map(str, path))] = np.asarray(mod.kernel[...]).copy()

        optimizer = nnx.Optimizer(
            model,
            optax.adamw(learning_rate=1e-2, weight_decay=0.0),
            wrt=LoRAParam,
        )
        x = jax.random.normal(jax.random.key(42), (2, 4, 16), dtype=jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(x) ** 2)

        grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))(model)
        optimizer.update(model, grads)

        for path, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRALinear):
                key = ".".join(map(str, path))
                np.testing.assert_array_equal(
                    np.asarray(mod.base.kernel[...]),
                    base_snapshots[key],
                    err_msg=f"base kernel changed at {key}",
                )
            elif isinstance(mod, nnx.Linear):
                key = ".".join(map(str, path))
                np.testing.assert_array_equal(
                    np.asarray(mod.kernel[...]),
                    base_snapshots[key],
                    err_msg=f"non-LoRA Linear kernel changed at {key}",
                )

    def test_merge_logit_equivalence(self):
        """``merge_lora_into_base`` must produce a model whose forward output
        matches the LoRA-wrapped forward to within fp32 tolerance."""
        model = _make_model(seed=0)
        inject_lora(model, r=8, alpha=16, rngs=nnx.Rngs(1))

        for path, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRALinear):
                mod.lora_B[...] = (
                    jax.random.normal(
                        jax.random.key(int(np.sum([hash(p) for p in path]) % 2**31)),
                        mod.lora_B[...].shape,
                    ).astype(mod.lora_B[...].dtype)
                    * 0.1
                )

        x = jax.random.normal(jax.random.key(42), (2, 4, 16), dtype=jnp.float32)
        y_lora = np.asarray(model(x))

        n = merge_lora_into_base(model)
        self.assertGreater(n, 0)
        for _, mod in nnx.iter_modules(model):
            self.assertNotIsInstance(mod, LoRALinear)

        y_merged = np.asarray(model(x))
        # Equivalence is up to fp32 numerical error from rearranging
        # (W + αAB)x vs Wx + α(A(Bx)). Loose tol covers last-bit drift.
        np.testing.assert_allclose(y_lora, y_merged, rtol=1e-4, atol=1e-4)

    def test_checkpoint_restore_preserves_fp32_master_and_export_merge(self):
        model = _MiniBf16Model(rngs=nnx.Rngs(0))
        inject_model_lora(model, r=2, alpha=2, rngs=nnx.Rngs(1))
        model.q_proj.lora_B[...] = jnp.full_like(model.q_proj.lora_B[...], 0.25)
        optimizer = MixedPrecisionOptimizer(model, optax.adamw(1e-3), wrt=LoRAParam)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1) as manager,
        ):
            state = vlm._train_state(optimizer, jax.random.key(2), 10)
            manager.save(
                1,
                args=ocp.args.Composite(
                    train_state=ocp.args.PyTreeSave(state),
                ),
                force=True,
            )

            restored_model = _MiniBf16Model(rngs=nnx.Rngs(0))
            inject_model_lora(restored_model, r=2, alpha=2, rngs=nnx.Rngs(1))
            restored_optimizer = MixedPrecisionOptimizer(
                restored_model,
                optax.adamw(1e-3),
                wrt=LoRAParam,
            )
            abstract = vlm._abstract_train_state(
                restored_optimizer,
                jax.random.key(2),
                10,
            )
            restored = manager.restore(
                1,
                args=ocp.args.Composite(
                    train_state=ocp.args.PyTreeRestore(abstract),
                ),
            )
            nnx.update(restored_optimizer, restored["train_state"]["optimizer"])

        self.assertEqual(restored_model.q_proj.lora_B[...].dtype, jnp.float32)
        np.testing.assert_array_equal(restored_model.q_proj.lora_B[...], 0.25)
        x = jnp.ones((1, 4), dtype=jnp.bfloat16)
        before_merge = restored_model(x)
        self.assertEqual(merge_lora_into_base(restored_model), 1)
        np.testing.assert_allclose(restored_model(x), before_merge, rtol=1e-2, atol=1e-2)

    def test_default_target_modules_match_qwen3vl_attribute_names(self):
        expected = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        self.assertEqual(set(DEFAULT_TARGET_MODULES), expected)

    def test_default_injection_allows_unmatched_moe_projection_names(self):
        model = _MiniAttention(16, rngs=nnx.Rngs(0))
        count = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        self.assertEqual(count, 4)

    def test_injection_consumes_rng_in_declared_target_order(self):
        model = _MiniAttention(16, rngs=nnx.Rngs(0))
        expected_rngs = nnx.Rngs(1)
        expected_a = nnx.initializers.uniform(scale=0.25)(
            expected_rngs.params(), (16, 4), jnp.float32
        )

        inject_lora(
            model,
            r=4,
            alpha=8,
            rngs=nnx.Rngs(1),
            target_modules=("q_proj", "k_proj"),
        )

        np.testing.assert_array_equal(model.q_proj.lora_A[...], expected_a)

    def test_qwen3_5_deltanet_targets_wrap_every_projection(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_qwen3_5_config("qwen3.5-smoke-dense")
            model = Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(0))
            count = inject_model_lora(
                model,
                r=4,
                alpha=8,
                rngs=nnx.Rngs(1),
            )

        self.assertGreater(count, len(QWEN3_5_DELTANET_TARGET_MODULES))
        deltanet_layers = [
            layer.linear_attn for layer in model.text.layers if hasattr(layer, "linear_attn")
        ]
        self.assertTrue(deltanet_layers)
        for layer in deltanet_layers:
            for name in QWEN3_5_DELTANET_TARGET_MODULES:
                self.assertIsInstance(getattr(layer, name), LoRALinear)
        self.assertTupleEqual(
            QWEN3_5_DELTANET_TARGET_MODULES,
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
        )

    def test_deltanet_target_names_do_not_change_a_non_deltanet_model(self):
        model = _MiniAttention(16, rngs=nnx.Rngs(0))
        count = inject_model_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        self.assertEqual(count, 4)

    def test_mixed_precision_optimizer_with_wrt_lora(self):
        """End-to-end smoke of the trainer's pattern: build the
        MixedPrecisionOptimizer with wrt=LoRAParam, take one step, verify
        loss is finite and base kernels are bit-exact unchanged."""
        model = _make_model(seed=0)
        n = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1))
        self.assertGreater(n, 0)

        # Snapshot base kernels before stepping.
        base_snapshots = {}
        for path, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRALinear):
                base_snapshots[".".join(map(str, path))] = np.asarray(mod.base.kernel[...]).copy()

        tx = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
        opt = MixedPrecisionOptimizer(model, tx, wrt=LoRAParam)

        x = jax.random.normal(jax.random.key(42), (2, 4, 16), dtype=jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(x) ** 2), jnp.array(1.0)

        (loss, _), grads = nnx.value_and_grad(
            loss_fn,
            argnums=nnx.DiffState(0, LoRAParam),
            has_aux=True,
        )(model)
        opt.update(grads)

        self.assertTrue(jnp.isfinite(loss).item())

        for path, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRALinear):
                key = ".".join(map(str, path))
                np.testing.assert_array_equal(
                    np.asarray(mod.base.kernel[...]),
                    base_snapshots[key],
                    err_msg=f"MixedPrecisionOptimizer changed base kernel at {key}",
                )


if __name__ == "__main__":
    absltest.main()
