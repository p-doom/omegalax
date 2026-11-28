"""Gated delta rule state continuation: chunked prefill + recurrent steps == one chunked forward,
and the fused Pallas decode == its XLA reference."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from omegalax.models.qwen3_5.kernels.pallas_triton import (
    chunk_gated_delta_rule_pallas,
    gated_delta_decode_pallas,
)
from omegalax.models.qwen3_5.kernels.xla_reference import (
    chunk_gated_delta_rule_xla,
    gated_delta_decode_xla,
    recurrent_gated_delta_step_xla,
)

H, A, U, C = 4, 128, 128, 64
CHUNKED = dict(xla=chunk_gated_delta_rule_xla, pallas=chunk_gated_delta_rule_pallas)


def _inputs(T, seed):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(rng.standard_normal((1, T, H, A)) * 0.1, jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((1, T, H, U)) * 0.1, jnp.bfloat16)
    a = jnp.asarray(rng.standard_normal((1, T, H)) * 0.5, jnp.float32)
    g = -jnp.exp(a) * jax.nn.softplus(a)
    beta = jax.nn.sigmoid(jnp.asarray(rng.standard_normal((1, T, H)) * 0.5, jnp.float32))
    init = jnp.asarray(rng.standard_normal((1, H, A, U)) * 0.1, jnp.float32)
    return q, k, v, g, beta, init


def _from(init, T):
    """Row 0 starts from ``init``; returns its state after the last chunk."""
    J = -(-T // C)
    return init, jnp.full((1, J), -1, jnp.int32).at[0, 0].set(0), jnp.full((1, 1), J - 1, jnp.int32)


class StateContinuationTest(parameterized.TestCase):
    @parameterized.product(impl=["xla", "pallas"])
    def test_prefill_then_steps_matches_one_forward(self, impl):
        T, T0 = 192, 128
        q, k, v, g, beta, init = _inputs(T, 0)
        out, state = CHUNKED[impl](q, k, v, g, beta, *_from(init, T))
        out0, state0 = CHUNKED[impl](
            q[:, :T0], k[:, :T0], v[:, :T0], g[:, :T0], beta[:, :T0], *_from(init, T0)
        )
        pool = jnp.zeros((3, H, A, U), jnp.float32).at[2].set(state0[0, 0])
        slot = jnp.asarray([2], jnp.int32)
        steps = []
        for t in range(T0, T):
            o, pool = recurrent_gated_delta_step_xla(
                q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], pool, slot
            )
            steps.append(o)
        ref, got = np.asarray(out[0, T0:]), np.asarray(jnp.stack(steps, axis=1)[0])
        self.assertLess(np.abs(np.asarray(out0[0]) - np.asarray(out[0, :T0])).max(), 1e-4)
        self.assertLess(np.abs(got - ref).max(), 2e-3, f"max ref {np.abs(ref).max()}")
        self.assertLess(np.abs(np.asarray(pool[2] - state[0, 0])).max(), 2e-3)

    @parameterized.product(impl=["xla", "pallas"])
    def test_packed_resets_match_separate_rows(self, impl):
        q1, k1, v1, g1, b1, i1 = _inputs(128, 1)
        q2, k2, v2, g2, b2, i2 = _inputs(64, 2)
        cat = lambda x, y: jnp.concatenate([x, y], axis=1)
        init = jnp.concatenate([i1, i2])
        idx = jnp.asarray([[0, -1, 1]], jnp.int32)
        out, state = CHUNKED[impl](
            cat(q1, q2),
            cat(k1, k2),
            cat(v1, v2),
            cat(g1, g2),
            cat(b1, b2),
            init,
            idx,
            jnp.asarray([[1, 2]], jnp.int32),
        )
        out1, state1 = CHUNKED["xla"](q1, k1, v1, g1, b1, *_from(i1, 128))
        out2, state2 = CHUNKED["xla"](q2, k2, v2, g2, b2, *_from(i2, 64))
        self.assertLess(np.abs(np.asarray(out[0, :128] - out1[0])).max(), 2e-3)
        self.assertLess(np.abs(np.asarray(out[0, 128:] - out2[0])).max(), 2e-3)
        self.assertLess(np.abs(np.asarray(state[0, 0] - state1[0, 0])).max(), 2e-3)
        self.assertLess(np.abs(np.asarray(state[0, 1] - state2[0, 0])).max(), 2e-3)

    def test_fused_decode_matches_xla(self):
        """Four decode steps, GQA 3: two rows on their own slots, two inactive rows sharing the
        trash slot (not compared). The chain rounds to bf16 several times, so the kernel must stay
        within the XLA oracle's own precision: its distance to the oracle may not exceed the
        oracle's distance to the same math in f32 (mean of output and state, max of output).
        The conv window is only moved, so it must match exactly."""
        rng = np.random.default_rng(6)
        num_k, heads, D, K, S, N = 2, 6, 128, 4, 5, 4
        C = (2 * num_k + heads) * D
        normal = lambda shape, scale, dtype: jnp.asarray(rng.standard_normal(shape) * scale, dtype)
        f32 = lambda x: x.astype(jnp.float32)
        slot, live = jnp.asarray([2, 4, 1, 1], jnp.int32), [0, 1]
        weight = normal((C, K), 0.5, jnp.bfloat16)
        A_log, dt = normal((heads,), 0.5, jnp.float32), normal((heads,), 0.5, jnp.float32)
        norm = jnp.asarray(rng.uniform(0.3, 1.0, D), jnp.float32)
        conv = normal((S, K - 1, C), 0.5, jnp.bfloat16)
        state = normal((S, heads, D, D), 0.1, jnp.float32)
        pools = {"xla": (conv, state), "pallas": (conv, state), "f32": (f32(conv), state)}
        for _ in range(4):
            x, z = normal((N, C), 0.5, jnp.bfloat16), normal((N, heads, D), 1.0, jnp.bfloat16)
            b, a = normal((N, heads), 1.0, jnp.bfloat16), normal((N, heads), 1.0, jnp.bfloat16)
            outs = {}
            for impl, fn, cast in (
                ("xla", gated_delta_decode_xla, lambda v: v),
                ("pallas", gated_delta_decode_pallas, lambda v: v),
                ("f32", gated_delta_decode_xla, f32),
            ):
                conv, state = pools[impl]
                out, conv, state = fn(
                    cast(x), z, cast(b), a, conv, cast(weight), state, A_log, dt, norm, slot,
                    eps=1e-6,
                )
                outs[impl], pools[impl] = np.asarray(out, np.float32)[live], (conv, state)
            got, own = (np.abs(outs[i] - outs["xla"]) for i in ("pallas", "f32"))
            got_st, own_st = (
                np.abs(np.asarray(pools[i][1] - pools["xla"][1])[2::2]) for i in ("pallas", "f32")
            )
            print(
                f"fused decode out max/mean {got.max():.2e}/{got.mean():.2e}, oracle's own "
                f"{own.max():.2e}/{own.mean():.2e}; state mean {got_st.mean():.2e} vs "
                f"{own_st.mean():.2e}"
            )
            self.assertLessEqual(got.max(), own.max())
            self.assertLessEqual(got.mean(), own.mean())
            self.assertLessEqual(got_st.mean(), own_st.mean())
        np.testing.assert_array_equal(
            np.asarray(pools["pallas"][0][2::2]), np.asarray(pools["xla"][0][2::2])
        )

    def test_packed_resets_grad_matches_xla(self):
        q1, k1, v1, g1, b1, i1 = _inputs(128, 4)
        q2, k2, v2, g2, b2, i2 = _inputs(64, 5)
        cat = lambda x, y: jnp.concatenate([x, y], axis=1)
        args = (cat(q1, q2), cat(k1, k2), cat(v1, v2), cat(g1, g2), cat(b1, b2))
        init = jnp.concatenate([i1, i2])
        idx, final = jnp.asarray([[0, -1, 1]], jnp.int32), jnp.asarray([[1, 2]], jnp.int32)

        def loss(fn):
            def f(q, k, v, g, beta, init):
                out, state = fn(q, k, v, g, beta, init, idx, final)
                return jnp.sum(out**2) + jnp.sum(state**2)

            return jax.grad(f, argnums=(0, 1, 2, 3, 4, 5))

        ref = loss(chunk_gated_delta_rule_xla)(*args, init)
        got = loss(chunk_gated_delta_rule_pallas)(*args, init)
        for r, x in zip(ref, got):
            r, x = np.asarray(r, np.float32), np.asarray(x, np.float32)
            self.assertLess(np.abs(x - r).max() / np.abs(r).max(), 1e-2)

    def test_pallas_init_grad_matches_xla(self):
        q, k, v, g, beta, init = _inputs(128, 3)
        _, idx, final = _from(init, 128)

        def loss(fn):
            return lambda init: jnp.sum(fn(q, k, v, g, beta, init, idx, final)[0] ** 2)

        ref = jax.grad(loss(chunk_gated_delta_rule_xla))(init)
        got = jax.grad(loss(chunk_gated_delta_rule_pallas))(init)
        self.assertLess(np.abs(np.asarray(got - ref)).max() / np.abs(np.asarray(ref)).max(), 1e-2)


if __name__ == "__main__":
    absltest.main()
