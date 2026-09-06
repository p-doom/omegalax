"""Numerical + sharding tests for the Qwen3-VL sharded token-embedding gather.

Qwen3-VL used to fully replicate its 248k-vocab embedding table on every
device (``reshard(embedding_VD, P())``) before gathering token embeddings,
costing hundreds of MB/device. It now performs a *sharded gather* on the
FSDP/TP-sharded embedding param, mirroring the (non-VL) Qwen3 text model:

    self.text.embedder.embedding[...].at[(token_ids_BT,)].get(
        out_sharding=self.text.out_emb_shd
    )

These tests are torch-free and run on CPU. They cover:

  * ``test_forward_matches_replicated_baseline`` / ``test_backward_matches_
    replicated_baseline`` -- numerical equivalence (fwd + bwd) between the
    old replicate-then-gather and the new sharded gather, to roundoff.
  * ``test_embedding_param_is_sharded`` -- under a faked 4-device FSDP mesh
    the embedding param is genuinely sharded (not replicated) along the FSDP
    axis, the gather output carries ``out_emb_shd``, and we quantify the
    per-device memory saved (replicated 2*V*D vs sharded V*D/N).
"""

import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Fake multiple devices so we can exercise a real (sharded) mesh on CPU.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
)

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec, reshard

from omegalax.distributed.mesh import make_mesh, mesh_rules
from omegalax.models.qwen3_vl import Qwen3VL, make_vl_config
from omegalax.models.shard_config import shard_config_for_mesh

P = PartitionSpec

_MODEL_ID = "qwen3-vl-smoke"


def _make_sharded_model(mesh):
    """Build a Qwen3-VL smoke model with params placed on ``mesh``.

    Mirrors the loader: the ``wp(..., ("vocab", "embed"))`` annotations are
    resolved to concrete ``PartitionSpec``s via ``nnx.get_partition_spec`` and
    every leaf is ``device_put`` onto its FSDP/TP shard.
    """
    cfg = make_vl_config(_MODEL_ID)
    cfg = dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))

    with mesh_rules(mesh):
        model = Qwen3VL(cfg, rngs=nnx.Rngs(params=0))
        graph_def, state = nnx.split(model)
        pspecs = nnx.get_partition_spec(state)

        def _shard(leaf, spec):
            if not hasattr(leaf, "shape"):
                return leaf
            return jax.device_put(leaf, NamedSharding(mesh, spec))

        state = jax.tree.map(_shard, state, pspecs)
        model = nnx.merge(graph_def, state)
    return model, cfg


def _random_tokens(cfg, batch_size=4, seq_len=16, seed=0):
    # batch_size must be a multiple of the fsdp axis (out_emb_shd shards batch).
    rng = np.random.RandomState(seed)
    # Avoid special image/video/vision tokens so text-only path is exercised.
    lo = max(cfg.image_token_id, cfg.video_token_id, cfg.vision_start_token_id) + 1
    return rng.randint(lo, cfg.vocab_size, size=(batch_size, seq_len)).astype(np.int32)


class Qwen3VLShardedEmbeddingTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = make_mesh(tp_size=1, fsdp_size=4, dp_size=1)
        cls.model, cls.cfg = _make_sharded_model(cls.mesh)

    def _embedding_and_shd(self):
        emb = self.model.text.embedder.embedding[...]
        out_emb_shd = self.model.text.out_emb_shd
        return emb, out_emb_shd

    def _sharded_gather(self, token_ids_BT):
        emb, out_emb_shd = self._embedding_and_shd()
        return jnp.astype(
            emb.at[(token_ids_BT,)].get(out_sharding=out_emb_shd),
            self.model.text.embedder.dtype,
        )

    def _replicated_gather(self, token_ids_BT):
        """The old behaviour: fully replicate the table, then gather."""
        emb, out_emb_shd = self._embedding_and_shd()
        emb = reshard(emb, P())
        emb = jnp.astype(emb, self.model.text.embedder.dtype)
        return emb.at[(token_ids_BT,)].get(out_sharding=out_emb_shd)

    # -- numerical equivalence -------------------------------------------------

    def test_forward_matches_replicated_baseline(self):
        token_ids_BT = jnp.asarray(_random_tokens(self.cfg))
        with mesh_rules(self.mesh):
            sharded = jax.jit(self._sharded_gather)(token_ids_BT)
            replicated = jax.jit(self._replicated_gather)(token_ids_BT)
        diff = sharded.astype(jnp.float32) - replicated.astype(jnp.float32)
        max_abs_diff = float(jnp.max(jnp.abs(diff)))
        print(f"[fwd] embedding gather max_abs_diff = {max_abs_diff:.3e}")
        # Both index the same table with the same ids -> bit-identical.
        np.testing.assert_array_equal(np.asarray(sharded), np.asarray(replicated))
        self.assertEqual(max_abs_diff, 0.0)

    def test_backward_matches_replicated_baseline(self):
        token_ids_BT = jnp.asarray(_random_tokens(self.cfg))
        emb0, _ = self._embedding_and_shd()

        def _loss(fn, emb, token_ids_BT):
            # cotangent-weighted sum so the gradient probes every gathered row.
            gathered = fn(emb, token_ids_BT)
            weights = jnp.arange(gathered.size, dtype=jnp.float32).reshape(gathered.shape)
            return jnp.sum(gathered.astype(jnp.float32) * weights)

        out_emb_shd = self.model.text.out_emb_shd
        emb_dtype = self.model.text.embedder.dtype

        def sharded_fn(emb, token_ids_BT):
            return jnp.astype(emb.at[(token_ids_BT,)].get(out_sharding=out_emb_shd), emb_dtype)

        def replicated_fn(emb, token_ids_BT):
            emb_r = reshard(emb, P())
            return jnp.astype(emb_r.at[(token_ids_BT,)].get(out_sharding=out_emb_shd), emb_dtype)

        with mesh_rules(self.mesh):
            g_sharded = jax.jit(jax.grad(lambda e: _loss(sharded_fn, e, token_ids_BT)))(emb0)
            g_replicated = jax.jit(jax.grad(lambda e: _loss(replicated_fn, e, token_ids_BT)))(emb0)

        g_sharded = np.asarray(g_sharded, dtype=np.float32)
        g_replicated = np.asarray(g_replicated, dtype=np.float32)
        max_abs_diff = float(np.max(np.abs(g_sharded - g_replicated)))
        print(f"[bwd] d(loss)/d(embedding) max_abs_diff = {max_abs_diff:.3e}")
        np.testing.assert_array_equal(g_sharded, g_replicated)
        self.assertEqual(max_abs_diff, 0.0)

    # -- sharding + memory -----------------------------------------------------

    def test_embedding_param_is_sharded(self):
        self.assertGreater(jax.device_count(), 1, "test requires faked multi-device")
        emb, out_emb_shd = self._embedding_and_shd()
        V, D = emb.shape
        self.assertEqual(V, self.cfg.vocab_size)
        self.assertEqual(D, self.cfg.emb_dim)

        sharding = emb.sharding
        spec = sharding.spec
        print(f"embedding param sharding.spec = {spec}")
        # The embedding table is annotated ("vocab","embed") -> ("tp","fsdp").
        # With tp=1 (dropped) and fsdp=4 the D axis must be sharded on fsdp.
        self.assertEqual(spec, P(None, "fsdp"))
        self.assertFalse(sharding.is_fully_replicated, "embedding must NOT be replicated")

        n_fsdp = int(self.mesh.shape["fsdp"])
        n_shards = len(set(sharding.device_set))  # devices holding a shard
        self.assertEqual(n_shards, jax.device_count())

        # Gather output carries the activation sharding.
        token_ids_BT = jnp.asarray(_random_tokens(self.cfg))
        with mesh_rules(self.mesh):
            gathered = jax.jit(self._sharded_gather)(token_ids_BT)
        print(f"gather out sharding.spec = {gathered.sharding.spec}, out_emb_shd = {out_emb_shd}")
        self.assertEqual(gathered.sharding.spec, out_emb_shd)

        # Per-device memory. Old code replicated the table (embedder + gather
        # both saw a full [V,D] copy -> 2*V*D floats/device). New code shards
        # the single [V,D] table across N devices -> V*D/N floats/device.
        bytes_per = jnp.dtype(emb.dtype).itemsize
        replicated_bytes = 2 * V * D * bytes_per
        sharded_bytes = (V * D * bytes_per) // n_fsdp
        saved = replicated_bytes - sharded_bytes
        print(
            f"per-device embedding memory: replicated(2*V*D)={replicated_bytes} B, "
            f"sharded(V*D/{n_fsdp})={sharded_bytes} B, saved={saved} B "
            f"({saved / replicated_bytes:.1%})"
        )
        self.assertLess(sharded_bytes, replicated_bytes)


if __name__ == "__main__":
    absltest.main()
