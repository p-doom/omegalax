"""Numerical-equivalence tests for the dropless grouped-GEMM MoE path.

The grouped MoE block (and the ``grouped_moe`` kernel) must match a local dense
compute-every-expert einsum reference (``_dense_ref`` / ``_dense_moe_block_output``)
on forward and backward, up to fp reduction-order roundoff. (The model has a single
grouped stream now; the dense einsum lives here as the slow reference used to
validate the fast kernel.)

All tests are CPU-only (faked device count set before importing jax).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Fake 4 CPU devices (some tests build small meshes). Must be set before jax import.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
).strip()

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from jax.sharding import AxisType, Mesh

from omegalax.distributed.mesh import mesh_rules
from omegalax.models.moe_grouped import grouped_moe
from omegalax.models.qwen3.config import make_config
from omegalax.models.qwen3.model import MoEFeedForward


def _single_device_mesh():
    """A 1-device Explicit (tp,fsdp,dp) mesh, independent of the faked device count.

    Explicit axis types match how the codebase builds meshes (jax.make_mesh) so the
    model's out_sharding PartitionSpecs are accepted.
    """
    return Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1),
        ("tp", "fsdp", "dp"),
        axis_types=(AxisType.Explicit,) * 3,
    )

# fp roundoff tolerances (fp32 reduction-order differences).
FWD_TOL = 1e-4
BWD_TOL = 1e-3


def _synthetic_moe(E=8, D=8, F=16, k=2, N=32, seed=0):
    rng = np.random.RandomState(seed)
    rp = lambda *s: jnp.asarray(rng.randn(*s).astype(np.float32))
    return dict(
        hidden=rp(N, D),
        gate=rp(E, D, F),
        up=rp(E, D, F),
        down=rp(E, F, D),
        router=rp(D, E),
        E=E, D=D, F=F, k=k, N=N,
    )


def _route(hidden, router, k):
    logits = hidden @ router
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    w, idx = jax.lax.top_k(probs, k)
    w = w / jnp.clip(jnp.sum(w, -1, keepdims=True), min=1e-9)
    return w.astype(hidden.dtype), idx


def _dense_ref(hidden, gate, up, down, router, k):
    w, idx = _route(hidden, router, k)
    g = jnp.einsum("ND,EDF->NEF", hidden, gate)
    u = jnp.einsum("ND,EDF->NEF", hidden, up)
    h = jax.nn.silu(g) * u
    o = jnp.einsum("NEF,EFD->NED", h, down)
    gathered = jnp.take_along_axis(o, idx[..., None], axis=1)
    return jnp.sum(gathered * w[..., None], axis=1)


def _dense_moe_block_output(mlp, hidden_BTD):
    """Dense-einsum reference for a routed-expert MoE block (relocated from the
    deleted ``moe_backend="dense"`` path) so the grouped block can be validated
    against a slow compute-every-expert reference.

    Reproduces the block forward: routing (softmax -> top_k -> optional renorm) +
    dense expert einsum + top-k gather (+ shared expert / gate when present). fp32
    reference, no LoRA / sharding; routing matches the grouped block's.
    """
    cfg = mlp.cfg
    B, T, D = hidden_BTD.shape
    probs = jax.nn.softmax(mlp.router(hidden_BTD).astype(jnp.float32), axis=-1)
    w, idx = jax.lax.top_k(probs, cfg.num_experts_per_tok)
    if getattr(cfg, "norm_topk_prob", True):
        w = w / jnp.clip(jnp.sum(w, -1, keepdims=True), min=1e-9)
    w = w.astype(hidden_BTD.dtype)
    gate = mlp.gate_proj[...].astype(hidden_BTD.dtype)
    up = mlp.up_proj[...].astype(hidden_BTD.dtype)
    down = mlp.down_proj[...].astype(hidden_BTD.dtype)
    o = jnp.einsum(
        "BTEF,EFD->BTED", jax.nn.silu(jnp.einsum("BTD,EDF->BTEF", hidden_BTD, gate))
        * jnp.einsum("BTD,EDF->BTEF", hidden_BTD, up), down,
    )
    flat = o.reshape(B * T, cfg.num_experts, D)
    fidx = idx.reshape(B * T, cfg.num_experts_per_tok)
    gathered = jnp.take_along_axis(flat, fidx[..., None], axis=1).reshape(
        B, T, cfg.num_experts_per_tok, D
    )
    merged = jnp.sum(gathered * w[..., None], axis=-2)
    if hasattr(mlp, "shared_expert"):
        merged = merged + jax.nn.sigmoid(mlp.shared_expert_gate(hidden_BTD)) * mlp.shared_expert(
            hidden_BTD
        )
    return merged


class GroupedMoEPhase1Test(absltest.TestCase):
    """Single-device grouped_moe vs dense einsum (fwd + bwd)."""

    def _check(self, primitive):
        d = _synthetic_moe()
        w, idx = _route(d["hidden"], d["router"], d["k"])

        def grouped(hidden, gate, up, down):
            ww, ii = _route(hidden, d["router"], d["k"])
            return grouped_moe(hidden, ii, ww, gate, up, down, num_experts=d["E"], primitive=primitive)

        yd = _dense_ref(d["hidden"], d["gate"], d["up"], d["down"], d["router"], d["k"])
        yg = grouped(d["hidden"], d["gate"], d["up"], d["down"])
        self.assertLess(float(jnp.max(jnp.abs(yd - yg))), FWD_TOL)

        ct = jnp.asarray(np.random.RandomState(9).randn(*yd.shape).astype(np.float32))
        gd = jax.grad(
            lambda *a: jnp.sum(_dense_ref(*a, d["router"], d["k"]) * ct), argnums=(0, 1, 2, 3)
        )(d["hidden"], d["gate"], d["up"], d["down"])
        gg = jax.grad(lambda *a: jnp.sum(grouped(*a) * ct), argnums=(0, 1, 2, 3))(
            d["hidden"], d["gate"], d["up"], d["down"]
        )
        for a, b in zip(gd, gg):
            self.assertLess(float(jnp.max(jnp.abs(a - b))), BWD_TOL)

    def test_grouped_matches_dense_jax_primitive(self):
        self._check("jax")

    def test_grouped_matches_dense_tokamax_primitive(self):
        self._check("tokamax")


class GroupedMoEModelPhase1Test(absltest.TestCase):
    """Qwen3 MoE block: moe_backend='grouped' vs 'dense' (fwd + bwd + aux_loss)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = _single_device_mesh()
        cls._set = jax.set_mesh(cls.mesh)
        cls._set.__enter__()
        cls._ctx = mesh_rules(cls.mesh)
        cls._ctx.__enter__()
        base = make_config("qwen3-smoke-moe")
        cls.cfg = dataclasses.replace(base, dtype=jnp.float32)  # fp32 to isolate algo
        cls.mlp = MoEFeedForward(cfg=cls.cfg, rngs=nnx.Rngs(params=0))
        cls.B, cls.T = 2, 7
        cls.x = jnp.asarray(np.random.RandomState(1).randn(cls.B, cls.T, cls.cfg.emb_dim).astype(np.float32))

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)
        cls._set.__exit__(None, None, None)
        super().tearDownClass()

    def test_forward_matches_dense_ref(self):
        yg, aux = self.mlp(self.x)
        yd = _dense_moe_block_output(self.mlp, self.x)
        self.assertLess(float(jnp.max(jnp.abs(yd - yg))), FWD_TOL)
        self.assertTrue(np.isfinite(float(aux)))

    def test_backward_matches_dense_ref(self):
        ct = jnp.asarray(np.random.RandomState(3).randn(self.B, self.T, self.cfg.emb_dim).astype(np.float32))
        gdef, state = nnx.split(self.mlp)

        def grouped(state, xx):
            out, _ = nnx.merge(gdef, state)(xx)
            return jnp.sum(out * ct)

        def dense(state, xx):
            return jnp.sum(_dense_moe_block_output(nnx.merge(gdef, state), xx) * ct)

        gs_g, gx_g = jax.grad(grouped, argnums=(0, 1))(state, self.x)
        gs_d, gx_d = jax.grad(dense, argnums=(0, 1))(state, self.x)
        self.assertLess(float(jnp.max(jnp.abs(gx_d - gx_g))), BWD_TOL)
        ld = jax.tree_util.tree_leaves(nnx.to_pure_dict(gs_d))
        lg = jax.tree_util.tree_leaves(nnx.to_pure_dict(gs_g))
        for a, b in zip(ld, lg):
            self.assertLess(float(jnp.max(jnp.abs(a - b))), BWD_TOL)


class GroupedMoEFlopTest(absltest.TestCase):
    """The grouped path processes k*N expert rows, not E*N (algorithmic sparsity)."""

    def test_expert_rows_reduced_by_E_over_k(self):
        E, k, N, D, F = 8, 2, 32, 8, 16
        # Algorithmic expert-matmul FLOPs are proportional to rows processed.
        per_row = 2 * (D * F + D * F + F * D)
        dense_flops = per_row * (E * N)
        grouped_flops = per_row * (k * N)
        self.assertAlmostEqual(dense_flops / grouped_flops, E / k, places=5)

    def test_grouped_jaxpr_uses_ragged_dot(self):
        mesh = _single_device_mesh()
        with jax.set_mesh(mesh), mesh_rules(mesh):
            cfg = dataclasses.replace(make_config("qwen3-smoke-moe"), dtype=jnp.float32)
            mlp = MoEFeedForward(cfg=cfg, rngs=nnx.Rngs(params=0))
            x = jnp.asarray(np.random.RandomState(2).randn(2, 8, cfg.emb_dim).astype(np.float32))
            jg = str(jax.make_jaxpr(lambda x: mlp(x)[0])(x))
            jd = str(jax.make_jaxpr(lambda x: _dense_moe_block_output(mlp, x))(x))
        # Grouped uses the ragged_dot kernel; the dense reference does not.
        self.assertIn("ragged_dot", jg)
        self.assertNotIn("ragged_dot", jd)


def _assert_backend_equiv(test, mlp, D, seed=1):
    """Grouped MoE block vs the local dense-einsum reference (fwd + bwd), any family."""
    B, T = 2, 6
    x = jnp.asarray(np.random.RandomState(seed).randn(B, T, D).astype(np.float32))
    yg, aux = mlp(x)
    yd = _dense_moe_block_output(mlp, x)
    test.assertLess(float(jnp.max(jnp.abs(yd - yg))), FWD_TOL)
    test.assertTrue(np.isfinite(float(aux)))

    ct = jnp.asarray(np.random.RandomState(seed + 1).randn(B, T, D).astype(np.float32))
    gdef, state = nnx.split(mlp)

    def grouped(state, xx):
        out, _ = nnx.merge(gdef, state)(xx)
        return jnp.sum(out * ct)

    def dense(state, xx):
        return jnp.sum(_dense_moe_block_output(nnx.merge(gdef, state), xx) * ct)

    gs_g, gx_g = jax.grad(grouped, argnums=(0, 1))(state, x)
    gs_d, gx_d = jax.grad(dense, argnums=(0, 1))(state, x)
    test.assertLess(float(jnp.max(jnp.abs(gx_d - gx_g))), BWD_TOL)
    for a, b in zip(
        jax.tree_util.tree_leaves(nnx.to_pure_dict(gs_d)),
        jax.tree_util.tree_leaves(nnx.to_pure_dict(gs_g)),
    ):
        test.assertLess(float(jnp.max(jnp.abs(a - b))), BWD_TOL)


class Qwen3_5SharedExpertPhase3Test(absltest.TestCase):
    """qwen3_5 MoE block (with shared expert + gate): grouped vs dense."""

    def test_grouped_matches_dense(self):
        from omegalax.models.qwen3_5.config import make_config as make_config_5
        from omegalax.models.qwen3_5.model import MoEFeedForward as MoE5

        mesh = _single_device_mesh()
        with jax.set_mesh(mesh), mesh_rules(mesh):
            cfg = dataclasses.replace(make_config_5("qwen3.5-smoke").text_config, dtype=jnp.float32)
            mlp = MoE5(cfg=cfg, rngs=nnx.Rngs(params=0))
            _assert_backend_equiv(self, mlp, cfg.hidden_size)


class Qwen3VLPhase3Test(absltest.TestCase):
    """qwen3_vl text MoE block: grouped vs dense."""

    def test_grouped_matches_dense(self):
        from omegalax.models.qwen3_vl.config import make_vl_config
        from omegalax.models.qwen3_vl.model import TextMoEFeedForward as MoEVL

        mesh = _single_device_mesh()
        with jax.set_mesh(mesh), mesh_rules(mesh):
            cfg = dataclasses.replace(make_vl_config("qwen3-vl-smoke-moe"), dtype=jnp.float32)
            mlp = MoEVL(cfg=cfg, rngs=nnx.Rngs(params=0))
            _assert_backend_equiv(self, mlp, cfg.emb_dim)


if __name__ == "__main__":
    absltest.main()
