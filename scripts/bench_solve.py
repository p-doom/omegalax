"""Micro-benchmark: solve_triangular vs linalg.solve for deltanet chunked solve.

Run:  python scripts/bench_solve.py
"""

import time
import jax
import jax.numpy as jnp
import jax.scipy.linalg


def build_lower_triangular(key, shape):
    """Build a unit lower-triangular lhs like deltanet does: I - (strict lower tri)."""
    B, H, J, L, _ = shape
    raw = jax.random.normal(key, shape, dtype=jnp.float32) * 0.01
    strict_lower = jnp.tril(raw, k=-1)  # zero diagonal + upper
    eye = jnp.eye(L, dtype=jnp.float32)
    lhs = eye - strict_lower  # unit lower triangular
    rhs = jnp.broadcast_to(eye, shape)
    return lhs, rhs


@jax.jit
def solve_triangular(lhs, rhs):
    return jax.scipy.linalg.solve_triangular(lhs, rhs, lower=True)


@jax.jit
def solve_general(lhs, rhs):
    return jnp.linalg.solve(lhs, rhs)


def bench(fn, lhs, rhs, *, warmup=5, iters=50, label=""):
    # warmup
    for _ in range(warmup):
        out = fn(lhs, rhs)
        out.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(lhs, rhs)
        out.block_until_ready()
    elapsed = time.perf_counter() - t0
    ms = elapsed / iters * 1000
    print(f"  {label:30s}  {ms:8.3f} ms/call  ({iters} iters)")
    return ms


def check_correctness(lhs, rhs):
    out_tri = solve_triangular(lhs, rhs)
    out_gen = solve_general(lhs, rhs)
    diff = jnp.abs(out_tri - out_gen).max()
    print(f"  max |diff| = {diff:.2e}")


def main():
    print(f"JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print()

    configs = [
        # (label, B, H, J, chunk_size)
        ("0.8B  B=2 T=2048",   2,   4,  32, 64),
        ("0.8B  B=4 T=2048",   4,   4,  32, 64),
        ("large B=1 T=4096",   1,  64,  64, 64),
        ("large B=1 T=8192",   1,  64, 128, 64),
    ]

    key = jax.random.key(42)
    for label, B, H, J, cs in configs:
        shape = (B, H, J, cs, cs)
        total_solves = B * H * J
        print(f"--- {label}  shape={shape}  ({total_solves} independent {cs}x{cs} solves) ---")
        lhs, rhs = build_lower_triangular(key, shape)
        check_correctness(lhs, rhs)
        ms_tri = bench(solve_triangular, lhs, rhs, label="solve_triangular (lower)")
        ms_gen = bench(solve_general, lhs, rhs, label="linalg.solve (general)")
        ratio = ms_gen / ms_tri if ms_tri > 0 else float("inf")
        print(f"  ratio (general/triangular) = {ratio:.2f}x")
        print()


if __name__ == "__main__":
    main()
