"""Measure what cuBLAS actually achieves on this model's GEMM shapes.

    bench_gemm.py

The trace says the GEMMs reach 527 TFLOPS, 53 % of the 989 TFLOPS BF16 peak, and
that they are now 57 % of step time. Before trying to fix that, the question is
whether 53 % is a defect or simply what these shapes give: peak numbers assume
huge square matrices, and a 2048-deep contraction over a vocab-wide output is not
that. If an isolated GEMM of the same shape also lands near 53 %, there is nothing
to win here and the effort belongs elsewhere.

Shapes are Qwen3.5-2B at batch 24, seq 2048 -> M = 49152 tokens, hidden 2048,
intermediate 6144, vocab 248320, GDN qkv 6144 (2*16*128 + 16*128).

Also compares operand layouts. XLA emits A.B^T (both operands row-major, both
contracting on dim 1), which is the tensor-core friendly form; the alternative
A.B is measured alongside to show what the layout is actually worth.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

PEAK_TFLOPS = 989.0  # H100 SXM, BF16 dense

M = 49152  # tokens per step at batch 24, seq 2048
SHAPES = [
    ("MLP gate/up", M, 6144, 2048, 36),   # x[M,2048] . W[6144,2048]^T, 2 pro Layer x 18? s.u.
    ("MLP down", M, 2048, 6144, 18),
    ("GDN in_proj_qkv", M, 6144, 2048, 18),
    ("GDN out_proj", M, 2048, 2048, 18),
    ("Attention qkv", M, 4096, 2048, 6),
    ("Attention out", M, 2048, 2048, 6),
    ("LM head", M, 248320, 2048, 1),
]


def timed(fn, *args, reps: int = 5) -> float:
    out = jax.block_until_ready(fn(*args))
    del out
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}\n")
    rng = np.random.default_rng(0)

    # A.B^T -- what XLA emits (lhs_contracting=1, rhs_contracting=1)
    nt = jax.jit(lambda a, b: jax.lax.dot_general(a, b, (((1,), (1,)), ((), ()))))
    # A.B -- the other layout, for comparison
    nn = jax.jit(lambda a, b: jax.lax.dot_general(a, b, (((1,), (0,)), ((), ()))))

    print(f"{'GEMM':<18} {'M':>6} {'N':>7} {'K':>5} {'A.B^T TF/s':>11} {'%peak':>6} "
          f"{'A.B TF/s':>9} {'%peak':>6} {'x/Step':>7} {'ms/Step':>8}")
    print("-" * 96)
    total_ms = 0.0
    total_flop = 0.0
    for name, m, n, k, per_step in SHAPES:
        a = jnp.asarray(rng.standard_normal((m, k)), dtype=jnp.bfloat16)
        b_nt = jnp.asarray(rng.standard_normal((n, k)), dtype=jnp.bfloat16)
        b_nn = jnp.asarray(rng.standard_normal((k, n)), dtype=jnp.bfloat16)
        flop = 2.0 * m * n * k
        t_nt = timed(nt, a, b_nt)
        t_nn = timed(nn, a, b_nn)
        tf_nt = flop / t_nt / 1e12
        tf_nn = flop / t_nn / 1e12
        ms = t_nt * 1e3 * per_step
        total_ms += ms
        total_flop += flop * per_step
        print(f"{name:<18} {m:>6} {n:>7} {k:>5} {tf_nt:>11.1f} {tf_nt/PEAK_TFLOPS*100:>5.1f}% "
              f"{tf_nn:>9.1f} {tf_nn/PEAK_TFLOPS*100:>5.1f}% {per_step:>7} {ms:>8.1f}")
        del a, b_nt, b_nn

    # Backward. Per forward GEMM Y = X.W^T the chain rule needs two more:
    #   dgrad  dX = dY . W        -- same class as forward, contracts over N
    #   wgrad  dW = dY^T . X      -- contracts over M = 49152 tokens
    # Only the forward shapes were measured above. wgrad reduces over the token
    # axis and produces a small output, which is the classic split-K case where
    # a GEMM has plenty of work but little parallelism to spread it over.
    print()
    print(f"{'GEMM (backward)':<18} {'M':>6} {'N':>7} {'K':>5} {'dgrad TF/s':>11} {'%peak':>6} "
          f"{'wgrad TF/s':>11} {'%peak':>6}")
    print("-" * 96)
    bw_ms = 0.0
    bw_flop = 0.0
    for name, m, n, k, per_step in SHAPES:
        dy = jnp.asarray(rng.standard_normal((m, n)), dtype=jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((n, k)), dtype=jnp.bfloat16)
        x = jnp.asarray(rng.standard_normal((m, k)), dtype=jnp.bfloat16)
        flop = 2.0 * m * n * k
        t_d = timed(nn, dy, w)                       # [M,N].[N,K] -> [M,K]
        t_w = timed(nt, dy.T, x.T)                   # [N,M].[K,M]^T -> [N,K]
        print(f"{name:<18} {m:>6} {n:>7} {k:>5} {flop/t_d/1e12:>11.1f} "
              f"{flop/t_d/1e12/PEAK_TFLOPS*100:>5.1f}% {flop/t_w/1e12:>11.1f} "
              f"{flop/t_w/1e12/PEAK_TFLOPS*100:>5.1f}%")
        bw_ms += (t_d + t_w) * 1e3 * per_step
        bw_flop += 2 * flop * per_step
        del dy, w, x
    print("-" * 96)
    print(f"Backward isoliert: {bw_ms:.1f} ms/Step fuer {bw_flop/1e12:.1f} TFLOP "
          f"-> {bw_flop/(bw_ms/1e3)/1e12:.0f} TFLOPS "
          f"({bw_flop/(bw_ms/1e3)/1e12/PEAK_TFLOPS*100:.1f} % Peak)")
    print(f"Forward+Backward:  {total_ms + bw_ms:.1f} ms/Step fuer "
          f"{(total_flop + bw_flop)/1e12:.1f} TFLOP -> "
          f"{(total_flop + bw_flop)/((total_ms + bw_ms)/1e3)/1e12:.0f} TFLOPS")

    print("-" * 96)
    print(f"Summe isoliert: {total_ms:.1f} ms/Step fuer {total_flop/1e12:.1f} TFLOP "
          f"-> {total_flop/(total_ms/1e3)/1e12:.0f} TFLOPS "
          f"({total_flop/(total_ms/1e3)/1e12/PEAK_TFLOPS*100:.1f} % Peak)")
    print("\nVergleich Trace (Kaskade + Conv-BTC): GEMM-Kernels 1068 ms/Step, 527 TFLOPS, 53 % Peak.")


if __name__ == "__main__":
    main()
