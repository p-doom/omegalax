"""Microbenchmark the layout changes the GDN path performs, at production shapes.

Purpose: the trace attributes 23 % of GPU time to "Reshape / copy", with one
transpose fusion at ~10 ms per call. Before rewriting any layout, check in
isolation what these transposes actually cost and what an alternative layout
would cost. The previous misattribution in this project (a kernel blamed on the
loss path) came from reasoning about a trace without an independent measurement.

Shapes match the cascade configuration: B=4 chains, C=6 segments folded into the
time axis, T=2048 per segment, H=16 value heads, A=U=128.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

B, C, T, H, A = 4, 6, 2048, 16, 128
CT = C * T


def timed(fn, *args, reps=20, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / reps


def gbps(bytes_moved, secs):
    return bytes_moved / secs / 1e9


def main() -> None:
    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}")
    print(f"shape:  B={B} CT={CT} H={H} A={A}\n")

    key = jax.random.key(0)
    # What the kernel receives today: (B, T, H, A) in bf16, cast to f32 on entry.
    x_bthA_bf16 = jax.random.normal(key, (B, CT, H, A), dtype=jnp.bfloat16)
    x_bthA_f32 = x_bthA_bf16.astype(jnp.float32)
    # The alternative layout: channels stay adjacent, (B, H, A, T).
    x_bhAT_f32 = jax.random.normal(key, (B, H, A, CT), dtype=jnp.float32)

    n = B * CT * H * A
    b_f32, b_bf16 = n * 4, n * 2

    cases = [
        ("transpose BTHA->BHTA, f32 in/out",
         jax.jit(lambda x: x.transpose(0, 2, 1, 3)), x_bthA_f32, b_f32 * 2),
        ("transpose BTHA->BHTA + cast bf16->f32 (heutiger Pfad)",
         jax.jit(lambda x: x.transpose(0, 2, 1, 3).astype(jnp.float32)), x_bthA_bf16,
         b_bf16 + b_f32),
        ("transpose BHAT->BHTA, f32 (Alternativlayout)",
         jax.jit(lambda x: x.transpose(0, 1, 3, 2)), x_bhAT_f32, b_f32 * 2),
        ("nur cast bf16->f32, kein transpose",
         jax.jit(lambda x: x.astype(jnp.float32)), x_bthA_bf16, b_bf16 + b_f32),
        ("copy f32 (Bandbreiten-Referenz)",
         jax.jit(lambda x: x + 0.0), x_bthA_f32, b_f32 * 2),
    ]

    print(f"{'Variante':<52}{'ms':>9}{'GB/s':>9}")
    print("-" * 70)
    for name, fn, arg, moved in cases:
        s = timed(fn, arg)
        print(f"{name:<52}{s * 1e3:>9.2f}{gbps(moved, s):>9.0f}")

    print(f"\nTensorgroesse f32: {b_f32 / 1e6:.0f} MB, bf16: {b_bf16 / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
