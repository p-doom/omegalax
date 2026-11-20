"""Regression test for the blocked chunk inverse. Needs a GPU.

Protects against the failure that cost the most here: a mathematically correct
identity that is numerically unusable at production scale. The doubling form
(``_unit_lower_inverse``) agreed with ``solve_triangular`` to 1.2e-06 on random
64x64 matrices and still produced ``loss=nan`` from the first step, because the
real matrices make its intermediate powers cancel catastrophically (E49/E50).

Two things are therefore checked at the shape the model actually runs, not at a
mock size:

1. The blocked inverse matches ``solve_triangular`` to fp32 rounding.
2. The doubling form does NOT. That assertion is deliberate -- it keeps the
   counter-example alive, so a future reader who finds the unused function and
   thinks it looks faster gets a failing test rather than NaN in a training run.

Run:  JAX_PLATFORMS=cuda python test_blocked_inverse.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from omegalax.models.qwen3_5.kernels.pallas_triton import (  # noqa: E402
    _blocked_unit_lower_inverse,
    _unit_lower_inverse,
)

C = 64          # production delta-rule chunk size
BATCH = 2048    # a slice of the production B*H*J = 15360; enough to be representative
TOL = 1e-05     # fp32 epsilon is 1.2e-07; this leaves room for accumulation


def main() -> int:
    rng = np.random.default_rng(0)
    eye = jnp.eye(C, dtype=jnp.float32)
    failures = []

    # Magnitudes matched to what the real model produces: max|A_pre| ~ 0.8 (E50).
    for scale, label in ((0.1, "klein"), (0.5, "mittel"), (1.0, "wie im Modell")):
        n = np.tril(rng.standard_normal((BATCH, C, C)).astype(np.float32) * scale, -1)
        n = jnp.asarray(n)
        ref = np.asarray(
            jax.scipy.linalg.solve_triangular(
                eye - n, jnp.broadcast_to(eye, n.shape), lower=True
            ),
            np.float64,
        )
        scl = max(np.abs(ref).max(), 1e-30)

        blocked = np.asarray(_blocked_unit_lower_inverse(n, 16), np.float64)
        err_blocked = np.abs(blocked - ref).max() / scl
        ok = err_blocked < TOL and np.isfinite(blocked).all()
        print(f"blockweise   scale={scale:<4} ({label:<13}) max|N|={float(jnp.max(jnp.abs(n))):>5.2f} "
              f"rel={err_blocked:.3e}  {'ok' if ok else 'FEHLER'}")
        if not ok:
            failures.append(f"blockweise bei scale={scale}: {err_blocked:.3e}")

        doubling = np.asarray(_unit_lower_inverse(n, eye), np.float64)
        err_doubling = np.abs(doubling - ref).max() / scl
        print(f"doubling     scale={scale:<4} ({label:<13}) {'':<12} "
              f"rel={err_doubling:.3e}  (soll schlecht sein)")
        if scale >= 1.0 and err_doubling < TOL:
            failures.append(
                f"doubling bei scale={scale} unerwartet genau ({err_doubling:.3e}) — "
                "der Gegenbeleg aus E50 traegt nicht mehr, Test pruefen"
            )

    print()
    if failures:
        for f in failures:
            print("FEHLGESCHLAGEN:", f)
        return 1
    print("alle Pruefungen bestanden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
