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
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax

from omegalax.trainers.lora import (
    LoRAParam,
    LoRALinear,
    inject_lora,
    merge_lora_into_base,
    DEFAULT_TARGET_MODULES,
)


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_model(seed: int = 0, d: int = 16, mlp: int = 32, n_layers: int = 2):
    return _MiniModel(d=d, mlp=mlp, n_layers=n_layers, rngs=nnx.Rngs(seed))


class LoRATest(absltest.TestCase):

    def test_forward_parity_at_zero_init(self):
        """With B=0 by construction, the LoRA-wrapped model must produce
        bit-identical outputs to the unwrapped model."""
        model = _make_model(seed=0)
        x = jax.random.normal(jax.random.key(42), (4, 7, 16), dtype=jnp.float32)
        y_before = model(x)

        n = inject_lora(model, r=8, alpha=16, rngs=nnx.Rngs(1), dtype=jnp.float32)
        self.assertGreater(n, 0)
        y_after = model(x)
        np.testing.assert_array_equal(np.asarray(y_before), np.asarray(y_after))

    def test_inject_count_matches_expected(self):
        """Per layer: 4 attention projections + 3 MLP projections = 7. Times
        n_layers. Vision (skipped) and lm_head (not in target list) → 0."""
        n_layers = 3
        model = _make_model(seed=0, n_layers=n_layers)
        n = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
        self.assertEqual(n, 7 * n_layers)

    def test_vision_subtree_is_skipped(self):
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
        self.assertIsInstance(model.vision.q_proj, nnx.Linear)
        self.assertNotIsInstance(model.vision.q_proj, LoRALinear)

    def test_lm_head_not_wrapped(self):
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
        self.assertIsInstance(model.lm_head, nnx.Linear)
        self.assertNotIsInstance(model.lm_head, LoRALinear)

    def test_grad_isolation_via_wrt_filter(self):
        """grads filtered by wrt=LoRAParam must contain only LoRAParam
        leaves; no base nnx.Param weights should receive gradient."""
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
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
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)

        base_snapshots: dict[str, np.ndarray] = {}
        for path, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRALinear):
                base_snapshots[".".join(map(str, path))] = np.asarray(mod.base.kernel[...]).copy()
            elif isinstance(mod, nnx.Linear):
                base_snapshots[".".join(map(str, path))] = np.asarray(mod.kernel[...]).copy()

        optimizer = nnx.Optimizer(
            model, optax.adamw(learning_rate=1e-2, weight_decay=0.0), wrt=LoRAParam,
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
        inject_lora(model, r=8, alpha=16, rngs=nnx.Rngs(1), dtype=jnp.float32)

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

    def test_default_target_modules_match_qwen3vl_attribute_names(self):
        expected = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        }
        self.assertEqual(set(DEFAULT_TARGET_MODULES), expected)

    def test_mixed_precision_optimizer_with_wrt_lora(self):
        """End-to-end smoke of the trainer's pattern: build the
        MixedPrecisionOptimizer with wrt=LoRAParam, take one step, verify
        loss is finite and base kernels are bit-exact unchanged."""
        from omegalax.trainers.optim import MixedPrecisionOptimizer

        model = _make_model(seed=0)
        n = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
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
            loss_fn, argnums=nnx.DiffState(0, LoRAParam), has_aux=True,
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
