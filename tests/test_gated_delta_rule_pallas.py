"""Forward correctness for the Pallas chunked gated delta rule.

Compares ``chunk_gated_delta_rule_pallas`` (Pallas Triton lowering) against
``chunk_gated_delta_rule_xla`` (pure-JAX reference) at multiple shapes.
Tolerances are bf16-realistic; the Pallas path runs internal accumulation in
fp32 then casts to the input dtype, matching the reference.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from omegalax.models.qwen3_5.kernels.pallas_triton import (
    chunk_gated_delta_rule_pallas,
)
from omegalax.models.qwen3_5.kernels.xla_reference import (
    chunk_gated_delta_rule_xla,
)


def _make_inputs(B, T, H, A, U, dtype=jnp.bfloat16, seed=0):
    rng = np.random.RandomState(seed)
    q = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1, dtype=dtype)
    k = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1, dtype=dtype)
    v = jnp.asarray(rng.randn(B, T, H, U).astype(np.float32) * 0.1, dtype=dtype)
    # ``g`` arrives at the kernel post-A_log/softplus from the deltanet caller;
    # values typically negative-ish. Sample similarly.
    a = jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5)
    g = -jnp.exp(a) * jax.nn.softplus(a)
    beta = jax.nn.sigmoid(jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5))
    return q, k, v, g, beta


class ForwardEquivalenceTest(parameterized.TestCase):

    @parameterized.parameters(
        # (B, T, H, A, U, name)
        dict(B=1, T=128,  H=2, A=64,  U=64,  name="tiny"),
        dict(B=2, T=512,  H=4, A=128, U=128, name="medium"),
        dict(B=1, T=2048, H=8, A=128, U=128, name="qwen3_5_2B_layer"),
    )
    def test_forward_matches_xla(self, B, T, H, A, U, name):
        q, k, v, g, beta = _make_inputs(B, T, H, A, U)
        ref = chunk_gated_delta_rule_xla(q, k, v, g, beta)
        out = chunk_gated_delta_rule_pallas(q, k, v, g, beta)
        ref_np = np.asarray(ref, dtype=np.float32)
        out_np = np.asarray(out, dtype=np.float32)
        abs_err = np.abs(out_np - ref_np)
        max_rel = abs_err.max() / max(np.abs(ref_np).max(), 1e-3)
        # bf16 round-trip on accumulators leaves ~1e-2 abs error at output
        # magnitudes in this range. Pure-JAX reference also accumulates in fp32.
        np.testing.assert_array_less(
            abs_err.max(),
            5e-2,
            err_msg=f"[{name}] max|err|={abs_err.max():.3e} max_rel={max_rel:.3e}",
        )


class TestSeqLenNotMultipleOfChunk(absltest.TestCase):

    def test_padding_path(self):
        # T=2049 forces internal padding to 2112 (chunk=64) then trim to 2049.
        q, k, v, g, beta = _make_inputs(1, 2049, 2, 64, 64)
        ref = chunk_gated_delta_rule_xla(q, k, v, g, beta)
        out = chunk_gated_delta_rule_pallas(q, k, v, g, beta)
        np.testing.assert_array_less(
            np.abs(np.asarray(out, dtype=np.float32) - np.asarray(ref, dtype=np.float32)).max(),
            5e-2,
        )


if __name__ == "__main__":
    absltest.main()
