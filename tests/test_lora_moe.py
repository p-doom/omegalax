"""Unit tests for per-expert LoRA on MoE feed-forward blocks.

Run on CPU; no model loading, no mesh. The MoE feed-forward modules store
their gate/up/down projections as rank-3 expert-stacked ``nnx.Param`` (E, D, F)
rather than ``nnx.Linear``; ``inject_lora`` therefore attaches a per-expert
stacked adapter (``LoRAMoEExperts``) at ``{name}_lora`` instead of wrapping a
Linear. Validate:
* forward parity at zero-init (B=0 => LoRA(x) === base(x) bitwise),
* inject_lora attaches expert adapters (count > 0; param count matches
  2 * r * E * (d_in + d_out) per expert projection),
* gradient isolation: wrt=LoRAParam produces grads only for adapter weights
  (incl. the new expert adapters); base expert Params see no gradient,
* after one optimizer step: base stacked expert Params are bit-exact
  unchanged, expert adapters moved and had nonzero grads,
* merge_lora_into_base: post-merge forward matches pre-merge forward (fp32).
"""

from __future__ import annotations

import contextlib
import os
import unittest.mock as mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax


@contextlib.contextmanager
def _force_unrolled_qwen3():
    """Force the qwen3 unrolled layer loop (the scan_layers opt-out was removed)."""
    from omegalax.models.qwen3.config import Qwen3Config

    with mock.patch.object(Qwen3Config, "is_homogeneous", property(lambda self: False)):
        yield

from omegalax.trainers.lora import (
    LoRAParam,
    LoRAMoEExperts,
    inject_lora,
    merge_lora_into_base,
)


class _MiniMoE(nnx.Module):
    """Minimal stand-in for the real MoE feed-forward blocks.

    Mirrors the real stacked-expert layout and forward:
      * gate_proj / up_proj: nnx.Param (E, D, F), einsum "BTD,EDF->BTEF"
      * down_proj:            nnx.Param (E, F, D), einsum "BTEF,EFD->BTED"
    plus a top-k router and gather, matching the guarded LoRA hook points
    (``{name}_lora`` slots + delta added inside the expert einsums).
    """

    _EXPERT_LORA_SHARDING = {
        "gate_proj": (None, "embed", "mlp"),
        "up_proj": (None, "embed", "mlp"),
        "down_proj": (None, "mlp", "embed"),
    }

    def __init__(self, E: int, D: int, F: int, k: int, *, rngs: nnx.Rngs):
        self.E, self.D, self.F, self.k = E, D, F, k
        init = nnx.initializers.lecun_normal()
        self.gate_proj = nnx.Param(init(rngs.params(), (E, D, F)))
        self.up_proj = nnx.Param(init(rngs.params(), (E, D, F)))
        self.down_proj = nnx.Param(init(rngs.params(), (E, F, D)))
        self.gate_proj_lora = nnx.data(None)
        self.up_proj_lora = nnx.data(None)
        self.down_proj_lora = nnx.data(None)
        self.router = nnx.Linear(D, E, use_bias=False, rngs=rngs)

    def __call__(self, hidden_BTD):
        gate_EDF = self.gate_proj[...]
        up_EDF = self.up_proj[...]
        down_EFD = self.down_proj[...]

        router_logits_BTE = self.router(hidden_BTD)
        probs_BTE = jax.nn.softmax(router_logits_BTE.astype(jnp.float32), axis=-1)
        topk_weights_BTk, topk_idx_BTk = jax.lax.top_k(probs_BTE, self.k)
        topk_weights_BTk = topk_weights_BTk / jnp.clip(
            jnp.sum(topk_weights_BTk, axis=-1, keepdims=True), min=1e-9
        )
        topk_weights_BTk = topk_weights_BTk.astype(hidden_BTD.dtype)

        gate_BTEF = jnp.einsum("BTD,EDF->BTEF", hidden_BTD, gate_EDF)
        up_BTEF = jnp.einsum("BTD,EDF->BTEF", hidden_BTD, up_EDF)
        if self.gate_proj_lora is not None:
            gate_BTEF += self.gate_proj_lora.delta_shared(hidden_BTD)
        if self.up_proj_lora is not None:
            up_BTEF += self.up_proj_lora.delta_shared(hidden_BTD)
        expert_hidden_BTEF = nnx.silu(gate_BTEF) * up_BTEF
        expert_out_BTED = jnp.einsum("BTEF,EFD->BTED", expert_hidden_BTEF, down_EFD)
        if self.down_proj_lora is not None:
            expert_out_BTED += self.down_proj_lora.delta_per_expert(expert_hidden_BTEF)

        B, T = hidden_BTD.shape[:2]
        flat_out = expert_out_BTED.reshape(B * T, self.E, self.D)
        flat_idx = topk_idx_BTk.reshape(B * T, self.k)
        gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
        gathered = gathered.reshape(B, T, self.k, self.D)
        return jnp.sum(gathered * topk_weights_BTk[..., None], axis=-2)


class _MiniMoEModel(nnx.Module):
    def __init__(self, E=4, D=16, F=32, k=2, n_layers=2, *, rngs: nnx.Rngs):
        self.layers = nnx.List([_MiniMoE(E, D, F, k, rngs=rngs) for _ in range(n_layers)])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


E, D, F, K, N_LAYERS = 4, 16, 32, 2, 2


def _make_model(seed=0):
    return _MiniMoEModel(E=E, D=D, F=F, k=K, n_layers=N_LAYERS, rngs=nnx.Rngs(seed))


class LoRAMoETest(absltest.TestCase):
    def test_forward_parity_at_zero_init(self):
        """With B=0 by construction, expert-LoRA injection must produce
        bit-identical outputs to the un-adapted MoE."""
        model = _make_model(seed=0)
        x = jax.random.normal(jax.random.key(42), (4, 7, D), dtype=jnp.float32)
        y_before = model(x)

        n = inject_lora(model, r=8, alpha=16, rngs=nnx.Rngs(1), dtype=jnp.float32)
        self.assertGreater(n, 0)
        y_after = model(x)
        np.testing.assert_array_equal(np.asarray(y_before), np.asarray(y_after))

    def test_inject_attaches_expert_adapters(self):
        """Each layer has 3 expert projections (gate/up/down); each gets a
        LoRAMoEExperts adapter. No dense Linear targets here except router,
        which is not a LoRA target, so count == 3 * n_layers."""
        model = _make_model(seed=0)
        n = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
        self.assertEqual(n, 3 * N_LAYERS)

        n_adapters = 0
        for _, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRAMoEExperts):
                n_adapters += 1
        self.assertEqual(n_adapters, 3 * N_LAYERS)

        # Slots are populated; router (not a target) is untouched.
        for layer in model.layers:
            self.assertIsInstance(layer.gate_proj_lora, LoRAMoEExperts)
            self.assertIsInstance(layer.up_proj_lora, LoRAMoEExperts)
            self.assertIsInstance(layer.down_proj_lora, LoRAMoEExperts)
            self.assertIsInstance(layer.router, nnx.Linear)

    def test_expert_adapter_param_count(self):
        """Per expert projection the adapter holds A:(E,d_in,r) + B:(E,r,d_out)
        LoRAParam leaves, i.e. r*E*(d_in+d_out) trainable elements. gate/up have
        (d_in,d_out)=(D,F); down has (F,D). Sum over 3 projections * n_layers."""
        r = 4
        model = _make_model(seed=0)
        inject_lora(model, r=r, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)

        lora_state = nnx.state(model, LoRAParam)
        total = sum(int(np.asarray(leaf).size) for leaf in jax.tree.leaves(nnx.pure(lora_state)))
        per_gate = r * E * (D + F)
        per_up = r * E * (D + F)
        per_down = r * E * (F + D)
        expected = (per_gate + per_up + per_down) * N_LAYERS
        self.assertEqual(total, expected)

        # There should be exactly 2 LoRAParam leaves (A, B) per adapter.
        n_leaves = len(jax.tree.leaves(nnx.pure(lora_state)))
        self.assertEqual(n_leaves, 2 * 3 * N_LAYERS)

    def test_grad_isolation_via_wrt_filter(self):
        """wrt=LoRAParam grads must contain only LoRAParam leaves (incl. the
        new expert adapters); base expert Params get no gradient."""
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
        x = jax.random.normal(jax.random.key(42), (2, 4, D), dtype=jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(x) ** 2)

        grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))(model)
        lora_grads = jax.tree.leaves(nnx.pure(nnx.state(grads, LoRAParam)))
        base_grads = jax.tree.leaves(nnx.pure(nnx.state(grads, nnx.Param)))
        self.assertGreater(len(lora_grads), 0)
        # Every diffable leaf is a LoRAParam; no base Param receives a grad.
        self.assertEqual(len(lora_grads), len(base_grads))
        # At least some expert-adapter grads are nonzero (B started at 0 so the
        # A-grad flows through B=0 -> 0, but the B-grad is nonzero).
        self.assertTrue(any(np.any(np.asarray(g) != 0) for g in lora_grads))

    def test_base_expert_param_bit_exact_after_step(self):
        """One optimizer step with wrt=LoRAParam must leave every base
        expert-stacked Param bit-identical, while adapters move."""
        model = _make_model(seed=0)
        inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)

        base_snap = {}
        adapter_snap = {}
        for i, layer in enumerate(model.layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                base_snap[(i, name)] = np.asarray(getattr(layer, name)[...]).copy()
                ad = getattr(layer, f"{name}_lora")
                adapter_snap[(i, name)] = (
                    np.asarray(ad.lora_A[...]).copy(),
                    np.asarray(ad.lora_B[...]).copy(),
                )

        optimizer = nnx.Optimizer(
            model, optax.adamw(learning_rate=1e-2, weight_decay=0.0), wrt=LoRAParam
        )
        x = jax.random.normal(jax.random.key(42), (2, 4, D), dtype=jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(x) ** 2)

        grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))(model)
        optimizer.update(model, grads)

        for i, layer in enumerate(model.layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(layer, name)[...]),
                    base_snap[(i, name)],
                    err_msg=f"base expert Param {name} changed at layer {i}",
                )
                ad = getattr(layer, f"{name}_lora")
                a_new = np.asarray(ad.lora_A[...])
                b_new = np.asarray(ad.lora_B[...])
                a_old, b_old = adapter_snap[(i, name)]
                # B started at 0 and must have moved (nonzero grad reached it).
                self.assertFalse(
                    np.array_equal(b_new, b_old),
                    msg=f"adapter B did not move at layer {i} {name}",
                )

    def test_merge_equivalence(self):
        """merge_lora_into_base folds expert adapters into the stacked Param;
        post-merge forward must match pre-merge forward (fp32 tol). Uses a
        nonzero B so the merge is a real (not trivial) fold."""
        model = _make_model(seed=0)
        inject_lora(model, r=8, alpha=16, rngs=nnx.Rngs(1), dtype=jnp.float32)

        # Perturb B away from zero so the adapters actually change the output.
        for _, mod in nnx.iter_modules(model):
            if isinstance(mod, LoRAMoEExperts):
                mod.lora_B[...] = (
                    jax.random.normal(jax.random.key(7), mod.lora_B[...].shape).astype(
                        mod.lora_B[...].dtype
                    )
                    * 0.1
                )

        x = jax.random.normal(jax.random.key(42), (2, 4, D), dtype=jnp.float32)
        y_lora = np.asarray(model(x))

        n = merge_lora_into_base(model)
        self.assertEqual(n, 3 * N_LAYERS)
        # Slots reset to None; adapters gone.
        for _, mod in nnx.iter_modules(model):
            self.assertNotIsInstance(mod, LoRAMoEExperts)
        for layer in model.layers:
            self.assertIsNone(layer.gate_proj_lora)
            self.assertIsNone(layer.up_proj_lora)
            self.assertIsNone(layer.down_proj_lora)

        y_merged = np.asarray(model(x))
        np.testing.assert_allclose(y_lora, y_merged, rtol=1e-4, atol=1e-4)

    def test_mixed_precision_optimizer_with_wrt_lora(self):
        """Trainer pattern: MixedPrecisionOptimizer with wrt=LoRAParam, one
        step; loss finite and base expert Params bit-exact unchanged."""
        from omegalax.trainers.optim import MixedPrecisionOptimizer

        model = _make_model(seed=0)
        n = inject_lora(model, r=4, alpha=8, rngs=nnx.Rngs(1), dtype=jnp.float32)
        self.assertGreater(n, 0)

        base_snap = {}
        for i, layer in enumerate(model.layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                base_snap[(i, name)] = np.asarray(getattr(layer, name)[...]).copy()

        tx = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
        opt = MixedPrecisionOptimizer(model, tx, wrt=LoRAParam)
        x = jax.random.normal(jax.random.key(42), (2, 4, D), dtype=jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(x) ** 2), jnp.array(1.0)

        (loss, _), grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, LoRAParam), has_aux=True
        )(model)
        opt.update(grads)

        self.assertTrue(jnp.isfinite(loss).item())
        for i, layer in enumerate(model.layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(layer, name)[...]),
                    base_snap[(i, name)],
                    err_msg=f"MixedPrecisionOptimizer changed base expert Param {name} at layer {i}",
                )


def _dense_moe_call(self, hidden_BTD):
    """Dense compute-every-expert reference for the qwen3 MoEFeedForward block
    (the deleted moe_backend='dense' path, relocated here). Patched onto
    ``MoEFeedForward.__call__`` for the reference run; includes per-expert LoRA."""
    cfg = self.cfg
    B, T = hidden_BTD.shape[:2]
    probs = jax.nn.softmax(self.router(hidden_BTD).astype(jnp.float32), axis=-1)
    w, idx = jax.lax.top_k(probs, cfg.num_experts_per_tok)
    if cfg.norm_topk_prob:
        w = w / jnp.clip(jnp.sum(w, -1, keepdims=True), min=1e-9)
    w = w.astype(probs.dtype)
    gate = self.gate_proj[...].astype(hidden_BTD.dtype)
    up = self.up_proj[...].astype(hidden_BTD.dtype)
    down = self.down_proj[...].astype(hidden_BTD.dtype)
    gate_BTEF = jnp.einsum("BTD,EDF->BTEF", hidden_BTD, gate)
    up_BTEF = jnp.einsum("BTD,EDF->BTEF", hidden_BTD, up)
    if self.gate_proj_lora is not None:
        gate_BTEF = gate_BTEF + self.gate_proj_lora.delta_shared(hidden_BTD)
    if self.up_proj_lora is not None:
        up_BTEF = up_BTEF + self.up_proj_lora.delta_shared(hidden_BTD)
    h = nnx.silu(gate_BTEF) * up_BTEF
    out_BTED = jnp.einsum("BTEF,EFD->BTED", h, down)
    if self.down_proj_lora is not None:
        out_BTED = out_BTED + self.down_proj_lora.delta_per_expert(h)
    flat = out_BTED.reshape(B * T, cfg.num_experts, cfg.emb_dim)
    fidx = idx.reshape(B * T, cfg.num_experts_per_tok)
    gathered = jnp.take_along_axis(flat, fidx[..., None], axis=1).reshape(
        B, T, cfg.num_experts_per_tok, cfg.emb_dim
    )
    merged = jnp.sum(gathered * w[..., None], axis=-2)
    mask = jax.nn.one_hot(idx, cfg.num_experts, dtype=probs.dtype)
    aux = (
        cfg.aux_loss_coef
        * jnp.sum(jnp.mean(mask, axis=(0, 1)) * jnp.mean(probs, axis=(0, 1))[None, :])
        * cfg.num_experts
    )
    return merged, aux


class LoRAGroupedVsDenseTest(absltest.TestCase):
    """LoRA-on-grouped-experts equivalence: the grouped-GEMM MoE path must produce
    the same forward output and per-expert LoRA gradients as a local dense-einsum
    reference (``_dense_moe_call`` patched onto the block), up to fp reduction
    order. Also checks the grouped adapter grads are finite/nonzero and the base
    expert weights stay frozen under wrt=LoRAParam.

    Uses the real qwen3 MoE model under a 1x1x1 mesh (the grouped path needs an
    active mesh for its ragged-GEMM sharding machinery).
    """

    def _cfg(self):
        import jax.numpy as jnp

        from omegalax.models.qwen3.config import Qwen3Config
        from omegalax.models.shard_config import ShardConfig

        return Qwen3Config(
            num_layers=2,
            vocab_size=128,
            emb_dim=32,
            mlp_dim=64,
            num_heads=4,
            head_dim=8,
            num_kv_heads=4,
            rope_theta=1_000_000,
            rope_scaling_factor=None,
            local_rope_theta=None,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            mlp_only_layers=(),
            decoder_sparse_step=1,
            norm_topk_prob=True,
            aux_loss_coef=0.01,
            shd_cfg=ShardConfig.no_sharding(),
            dtype=jnp.float32,
        )

    def _build_model(self):
        """Grouped qwen3 with injected LoRA (nonzero B so the delta actually fires)."""
        from flax import nnx

        from omegalax.models.qwen3.model import Qwen3
        from omegalax.models.sharding_runtime import set_attn_backend

        m = Qwen3(self._cfg(), rngs=nnx.Rngs(0))
        inject_lora(m, r=8, alpha=16, rngs=nnx.Rngs(1), dtype=jnp.float32)
        for _, mod in nnx.iter_modules(m):
            if isinstance(mod, LoRAMoEExperts):
                mod.lora_B[...] = (
                    jax.random.normal(jax.random.key(7), mod.lora_B[...].shape, jnp.float32) * 0.1
                )
        set_attn_backend(m, text_backend="xla")
        return m

    def test_grouped_matches_dense_fwd_bwd(self):
        from flax import nnx

        from omegalax.distributed.mesh import make_mesh, mesh_rules
        from omegalax.models.qwen3.model import MoEFeedForward

        with _force_unrolled_qwen3(), mesh_rules(make_mesh(tp_size=1, fsdp_size=1, dp_size=1)):
            m = self._build_model()

            rng = np.random.RandomState(0)
            tok = jnp.asarray(rng.randint(1, 128, size=(2, 12)).astype(np.int32))
            seg = jnp.ones((2, 12), dtype=jnp.int32)

            def loss_fn(m):
                h, aux = m(tok, seg, None, jnp.array(0, jnp.int32))
                return jnp.sum(h**2) + aux

            # Distinct jit'd fns per path so the patched (dense) trace is not cached
            # and reused for the grouped call.
            @nnx.jit
            def fwd_grouped(m):
                return m(tok, seg, None, jnp.array(0, jnp.int32))[0]

            @nnx.jit
            def fwd_dense(m):
                return m(tok, seg, None, jnp.array(0, jnp.int32))[0]

            @nnx.jit
            def grads_grouped(m):
                return nnx.grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))(m)

            @nnx.jit
            def grads_dense(m):
                return nnx.grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))(m)

            with mock.patch.object(MoEFeedForward, "__call__", _dense_moe_call):
                h_dense = np.asarray(fwd_dense(m))
                g_dense = grads_dense(m)
            h_grp = np.asarray(fwd_grouped(m))
            g_grp = grads_grouped(m)
            fwd_max = float(np.max(np.abs(h_dense - h_grp)))

        ld = jax.tree.leaves(nnx.pure(nnx.state(g_dense, LoRAParam)))
        lg = jax.tree.leaves(nnx.pure(nnx.state(g_grp, LoRAParam)))
        self.assertEqual(len(ld), len(lg))
        self.assertGreater(len(lg), 0)
        grad_max = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(ld, lg)
        )

        # Forward and per-expert LoRA gradients match the dense path (fp tol).
        self.assertLess(fwd_max, 1e-4)
        self.assertLess(grad_max, 1e-4)
        # Grouped-path expert-adapter grads are finite and (some) nonzero.
        self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in lg))
        self.assertTrue(any(np.any(np.asarray(g) != 0) for g in lg))
        # Base grouped expert weights are frozen: under wrt=LoRAParam every
        # diffable leaf is a LoRAParam (no bare base Param grad).
        bg = jax.tree.leaves(nnx.pure(nnx.state(g_grp, nnx.Param)))
        self.assertEqual(len(bg), len(lg))
        print(
            f"[lora grouped-vs-dense] fwd_max={fwd_max:.2e} grad_max={grad_max:.2e} "
            f"n_lora_grads={len(lg)}"
        )


if __name__ == "__main__":
    absltest.main()
