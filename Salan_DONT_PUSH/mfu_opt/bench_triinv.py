"""Is a blocked triangular inverse actually faster than cuBLAS trsm here?

Measured in isolation on the production shape before touching the model. The
doubling identity was fast but numerically unusable (E49/E50); blocked forward
substitution never forms high powers and would be stable. Whether it is also
faster is a separate question, and the cheapest way to answer it is here rather
than after a rewrite.

Shape: batch B*H*J = 15360 matrices of 64x64 fp32, right-hand side the identity
(the caller needs the full inverse). trsm costs about 100 ms/step in situ.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

C = 64
BATCH = 5 * 16 * 192  # B chains x heads x chunks at batch 30, cascade


def blocked_inverse(n_mat, block: int, precision=None):
    """Inverse of I - n_mat by block forward substitution.

    T = [[T11, 0], [T21, T22]]  ->  T^-1 = [[A, 0], [-B T21 A, B]] with A = T11^-1,
    B = T22^-1. Applied iteratively over block rows: each block row is solved
    against the already-inverted blocks above it. No matrix powers are formed, so
    the cancellation that killed the doubling form cannot arise.
    """
    eye_b = jnp.eye(block, dtype=n_mat.dtype)
    nb = C // block
    blocks = [[n_mat[..., i * block:(i + 1) * block, j * block:(j + 1) * block]
               for j in range(nb)] for i in range(nb)]
    inv = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        rhs = jnp.broadcast_to(eye_b, blocks[i][i].shape)
        inv[i][i] = jax.scipy.linalg.solve_triangular(eye_b - blocks[i][i], rhs, lower=True)
    for i in range(nb):
        for j in range(i - 1, -1, -1):
            acc = jnp.zeros_like(blocks[i][j])
            for k in range(j, i):
                acc = acc + jax.lax.dot_general(
                    blocks[i][k], inv[k][j], (((2,), (1,)), ((0,), (0,))),
                    precision=precision)
            inv[i][j] = jax.lax.dot_general(
                inv[i][i], acc, (((2,), (1,)), ((0,), (0,))), precision=precision)
    rows = [jnp.concatenate([inv[i][j] if j <= i else jnp.zeros_like(inv[i][i])
                             for j in range(nb)], axis=-1) for i in range(nb)]
    return jnp.concatenate(rows, axis=-2)


def timed(fn, arg, reps=5):
    jax.block_until_ready(fn(arg))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    rng = np.random.default_rng(0)
    n = jnp.asarray(np.tril(rng.standard_normal((BATCH, C, C)) * 0.1, -1), dtype=jnp.float32)
    eye = jnp.eye(C, dtype=jnp.float32)

    trsm = jax.jit(lambda x: jax.scipy.linalg.solve_triangular(
        eye - x, jnp.broadcast_to(eye, x.shape), lower=True))
    ref = np.asarray(trsm(n), np.float64)
    t_trsm = timed(trsm, n)
    print(f"{'Variante':<26}{'ms':>9}{'gegen trsm':>13}{'max rel. Abw.':>16}")
    print("-" * 66)
    print(f"{'trsm (cuBLAS, Referenz)':<26}{t_trsm:>9.2f}{'1.00x':>13}{'—':>16}")
    for prec, label in ((None, "TF32-Default"), (jax.lax.Precision.HIGHEST, "fp32")):
        for b in (8, 16, 32):
            fn = jax.jit(functools.partial(blocked_inverse, block=b, precision=prec))
            try:
                out = np.asarray(fn(n), np.float64)
                t = timed(fn, n)
                rel = np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-30)
                name = f"Block {b}, {label}"
                print(f"{name:<26}{t:>9.2f}{t_trsm / t:>12.2f}x{rel:>16.3e}")
            except Exception as e:  # noqa: BLE001
                print(f"{'Block ' + str(b) + ', ' + label:<26}{'FEHLER':>9}  {str(e)[:40]}")


if __name__ == "__main__":
    main()
