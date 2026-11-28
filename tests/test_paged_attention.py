"""Paged attention kernel against a plain-XLA dense oracle."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from omegalax.kernels.paged_attention import paged_attention
from omegalax.kernels.paged_extend_attention import TILE, paged_extend_attention


def paged_attention_xla(
    q_NQHK, k_pages_GPSK, v_pages_GPSK, page_table_NP, q_pos_N, kv_len_N, scale
):
    N, Q, H, K = q_NQHK.shape
    G, _, S, _ = k_pages_GPSK.shape
    T = page_table_NP.shape[1] * S
    k_GNTK = k_pages_GPSK[:, page_table_NP].reshape(G, N, T, K).astype(jnp.float32)
    v_GNTK = v_pages_GPSK[:, page_table_NP].reshape(G, N, T, K).astype(jnp.float32)
    q_NQGRK = q_NQHK.reshape(N, Q, G, H // G, K).astype(jnp.float32)
    s = jnp.einsum("nqgrk,gntk->nqgrt", q_NQGRK, k_GNTK, precision="highest") * scale
    t_T = jnp.arange(T)
    pos_NQ = q_pos_N[:, None] + jnp.arange(Q)[None, :]
    mask_NQT = (t_T <= pos_NQ[..., None]) & (t_T < kv_len_N[:, None, None])
    s = jnp.where(mask_NQT[:, :, None, None, :], s, -jnp.inf)
    o = jnp.einsum("nqgrt,gntk->nqgrk", jax.nn.softmax(s, axis=-1), v_GNTK, precision="highest")
    return o.reshape(N, Q, H, K)


class PagedAttentionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name="decode_r2", N=16, Q=1, H=16, G=8, S=64, max_pages=32),
        dict(testcase_name="decode_r4", N=16, Q=1, H=32, G=8, S=64, max_pages=32),
        dict(testcase_name="decode_r6", N=16, Q=1, H=24, G=4, S=64, max_pages=32),
        dict(testcase_name="small_pages", N=16, Q=1, H=16, G=8, S=16, max_pages=128),
    )
    def test_matches_xla(self, N, Q, H, G, S, max_pages):
        rng = np.random.default_rng(0)
        K, P = 128, N * max_pages + 1
        k = jnp.asarray(rng.standard_normal((G, P, S, K)), jnp.bfloat16)
        v = jnp.asarray(rng.standard_normal((G, P, S, K)), jnp.bfloat16)
        q = jnp.asarray(rng.standard_normal((N, Q, H, K)), jnp.bfloat16)
        kv_len = rng.integers(1, max_pages * S, size=N).astype(np.int32)
        kv_len[0] = 0
        q_pos = np.maximum(kv_len - Q, 0).astype(np.int32)
        page_table = 1 + rng.permutation(N * max_pages).reshape(N, max_pages).astype(np.int32)
        args = (q, k, v, jnp.asarray(page_table), jnp.asarray(q_pos), jnp.asarray(kv_len))
        for num_work in (N, 4 * N):
            out = np.asarray(paged_attention(*args, scale=K**-0.5, num_work=num_work), np.float32)
            ref = np.asarray(paged_attention_xla(*args, scale=K**-0.5))
            self.assertTrue(np.all(out[0] == 0), "empty query block must return zeros")
            err = np.abs(out[1:] - ref[1:])
            self.assertLess(err.max(), 2e-2, f"num_work={num_work}")
            self.assertLess(err.mean(), 1e-3, f"num_work={num_work}")

    @parameterized.named_parameters(
        dict(testcase_name="r4_hd128", N=8, H=32, G=8, K=128),
        dict(testcase_name="r8_hd256", N=8, H=16, G=2, K=256),
    )
    def test_extend_matches_xla(self, N, H, G, K):
        rng = np.random.default_rng(0)
        S, max_pages = 64, 64
        P = N * max_pages + 1
        k = jnp.asarray(rng.standard_normal((G, P, S, K)), jnp.bfloat16)
        v = jnp.asarray(rng.standard_normal((G, P, S, K)), jnp.bfloat16)
        q = jnp.asarray(rng.standard_normal((N * TILE, H, K)), jnp.bfloat16)
        kv_len = rng.integers(1, max_pages * S, size=N).astype(np.int32)
        kv_len[0] = 0
        # Tiles end at kv_len, so most of them carry padding rows past it.
        q_pos = np.maximum(kv_len - rng.integers(1, TILE + 1, size=N), 0).astype(np.int32)
        page_table = 1 + rng.permutation(N * max_pages).reshape(N, max_pages).astype(np.int32)
        args = (jnp.asarray(page_table), jnp.asarray(q_pos), jnp.asarray(kv_len))
        out = np.asarray(paged_extend_attention(q, k, v, *args, scale=K**-0.5), np.float32)
        ref = np.asarray(paged_attention_xla(q.reshape(N, TILE, H, K), k, v, *args, scale=K**-0.5))
        out = out.reshape(N, TILE, H, K)
        self.assertTrue(np.all(out[0] == 0), "empty tile must return zeros")
        valid = q_pos[:, None] + np.arange(TILE)[None] < kv_len[:, None]
        err = np.abs(out - ref)[valid]
        self.assertLess(err.max(), 2e-2)
        self.assertLess(err.mean(), 1e-3)


if __name__ == "__main__":
    absltest.main()
