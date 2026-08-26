"""Per-batch-index loss diagnostic in ``chunked_cross_entropy_loss``.

The diagnostic exists to expose batch-index-dependent bugs (a vision splice that
only routes batch index 0 correctly shows up as ``loss_bidx_0`` far below the
rest), so it has to be exact and it has to survive FSDP: the per-row sums stay
sharded on the batch axis while the scalar accumulators are replicated.

The FSDP case needs several devices, and the device count is fixed for the
lifetime of a process, so that test runs in a child interpreter with four host
devices rather than forcing them on every other test module in the run.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.trainers.loss import chunked_cross_entropy_loss

BATCH = 4
SEQ = 33
DIM = 8
VOCAB = 16
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _inputs(seed: int = 0):
    rng = np.random.RandomState(seed)
    hidden = jnp.asarray(rng.randn(BATCH, SEQ, DIM).astype(np.float32) * 0.5)
    kernel = jnp.asarray(rng.randn(DIM, VOCAB).astype(np.float32) * 0.5)
    targets = jnp.asarray(rng.randint(0, VOCAB, size=(BATCH, SEQ)).astype(np.int32))
    mask = np.zeros((BATCH, SEQ), dtype=np.int32)
    for row in range(BATCH):
        mask[row, 2 + row : SEQ - row] = 1
    return hidden, kernel, targets, jnp.asarray(mask)


def _reference_per_sample(hidden, kernel, targets, mask):
    """Per-row masked-mean cross-entropy, computed without any tiling."""
    logits = jnp.einsum("BTD,DV->BTV", hidden[:, :-1, :], kernel).astype(jnp.float32)
    tgt = targets[:, 1:]
    msk = mask[:, 1:].astype(jnp.float32)
    logsumexp = jax.scipy.special.logsumexp(logits, axis=-1)
    picked = jnp.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0]
    nll = (logsumexp - picked) * msk
    return np.asarray(jnp.sum(nll, axis=-1) / jnp.maximum(jnp.sum(msk, axis=-1), 1.0))


class PerSampleLossTest(absltest.TestCase):
    def test_matches_untiled_reference(self):
        hidden, kernel, targets, mask = _inputs()
        for num_tiles in (1, 4, 8):
            _, per_sample = chunked_cross_entropy_loss(
                hidden, kernel, targets, mask, num_tiles=num_tiles, return_per_sample=True
            )
            np.testing.assert_allclose(
                np.asarray(per_sample),
                _reference_per_sample(hidden, kernel, targets, mask),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"num_tiles={num_tiles}",
            )

    def test_scalar_loss_unchanged_by_the_flag(self):
        hidden, kernel, targets, mask = _inputs()
        for num_tiles in (1, 4, 8):
            plain = chunked_cross_entropy_loss(hidden, kernel, targets, mask, num_tiles=num_tiles)
            with_diag, _ = chunked_cross_entropy_loss(
                hidden, kernel, targets, mask, num_tiles=num_tiles, return_per_sample=True
            )
            np.testing.assert_array_equal(np.asarray(with_diag), np.asarray(plain))

    def test_scalar_loss_is_the_token_weighted_mean_of_the_per_sample_losses(self):
        hidden, kernel, targets, mask = _inputs()
        loss, per_sample = chunked_cross_entropy_loss(
            hidden, kernel, targets, mask, num_tiles=4, return_per_sample=True
        )
        counts = np.asarray(jnp.sum(mask[:, 1:].astype(jnp.float32), axis=-1))
        weighted = float(np.sum(np.asarray(per_sample) * counts) / np.sum(counts))
        self.assertAlmostEqual(float(loss), weighted, places=4)

    def test_distinct_rows_produce_distinct_losses(self):
        hidden, kernel, targets, mask = _inputs()
        _, per_sample = chunked_cross_entropy_loss(
            hidden, kernel, targets, mask, num_tiles=4, return_per_sample=True
        )
        values = np.asarray(per_sample)
        self.assertEqual(values.shape, (BATCH,))
        self.assertEqual(len(set(np.round(values, 5).tolist())), BATCH)


_FSDP_CHILD = textwrap.dedent(
    """
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from omegalax.distributed.mesh import mesh_rules_for
    from omegalax.trainers.loss import chunked_cross_entropy_loss
    from tests.test_per_sample_loss import BATCH, _inputs, _reference_per_sample

    assert jax.device_count() == BATCH, jax.device_count()
    hidden, kernel, targets, mask = _inputs()
    expected = _reference_per_sample(hidden, kernel, targets, mask)
    batch_axis = ("dp", "fsdp")
    with mesh_rules_for(tp_size=1, fsdp_size=BATCH, dp_size=1) as mesh:
        put = lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec))
        args = (
            put(hidden, P(batch_axis, None, None)),
            put(kernel, P()),
            put(targets, P(batch_axis, None)),
            put(mask, P(batch_axis, None)),
        )
        for num_tiles in (1, 4, 8):
            _, per_sample = jax.jit(
                lambda h, k, t, m, n=num_tiles: chunked_cross_entropy_loss(
                    h, k, t, m, num_tiles=n, return_per_sample=True
                )
            )(*args)
            np.testing.assert_allclose(np.asarray(per_sample), expected, rtol=1e-5, atol=1e-5)
            assert not per_sample.sharding.is_fully_replicated, (
                "per-sample loss is expected to come out sharded on the batch axis; "
                "the trainer reshards it so every process can read it"
            )
        replicated = jax.jit(
            lambda h, k, t, m: jax.sharding.reshard(
                chunked_cross_entropy_loss(h, k, t, m, num_tiles=4, return_per_sample=True)[1],
                P(),
            )
        )(*args)
        assert replicated.sharding.is_fully_replicated
        np.testing.assert_allclose(np.asarray(replicated), expected, rtol=1e-5, atol=1e-5)
    print("FSDP_PER_SAMPLE_OK")
    """
)


class PerSampleLossShardedTest(absltest.TestCase):
    """Regressions for the two ways FSDP broke this diagnostic.

    A scan carry has to round-trip its exact sharding. With FSDP-sharded hidden
    states the per-row sums come out sharded on the batch axis while the scalar
    accumulators are replicated, so carrying them raised ``scan body function
    carry input and carry output must have equal types ... float32[4] vs
    float32[4@(dp,fsdp)]`` at the first training step.

    That same sharding is why the trainer reshards the metric to replicated
    before reading it: fetching a batch-sharded array raises ``spans
    non-addressable (non process local) devices`` under ``process_count>1``.
    """

    def test_runs_under_fsdp_and_matches_the_replicated_result(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([REPO_ROOT, env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
        result = subprocess.run(
            [sys.executable, "-c", _FSDP_CHILD],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"FSDP child failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr[-4000:]}",
        )
        self.assertIn("FSDP_PER_SAMPLE_OK", result.stdout)


if __name__ == "__main__":
    absltest.main()
