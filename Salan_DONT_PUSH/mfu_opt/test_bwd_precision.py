"""Measure both VJP paths against a float64 reference instead of against each other.

The Pallas path and the scan it replaces disagree by ~9e-04, far above fp32
rounding. Comparing them to each other cannot say which one is off. JAX defaults
float32 matmuls on GPU to TF32 (~10 mantissa bits), whose resolution is exactly
that order, so the suspicion is that the existing scan is itself only TF32
accurate. A float64 reference settles it: whichever path is closer to it is the
more accurate one, and "no worse than the reference" is the bar to clear.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import omegalax.models.qwen3_5.kernels.pallas_triton as pt  # noqa: E402

B, H, J, C, A, U = 2, 4, 8, 64, 128, 128
rng = np.random.default_rng(0)


def r(*s):
    return jnp.asarray(rng.standard_normal(s) * 0.1, dtype=jnp.float32)


kcd, k_dec = r(B, H, J, C, A), r(B, H, J, C, A)
u_pre, df = r(B, H, J, C, U), jnp.abs(r(B, H, J)) + 0.5
init = r(B, H, A, U)
ct = (r(B, H, J, A, U), r(B, H, J, C, U), r(B, H, A, U))

# float64 reference, forward and backward, straight from the docstring equations.
k64, kd64, u64 = [np.asarray(x, np.float64) for x in (kcd, k_dec, u_pre)]
df64, in64 = np.asarray(df, np.float64), np.asarray(init, np.float64)
dsi64, dvn64, dfi64 = [np.asarray(x, np.float64) for x in ct]
st64 = np.zeros((B, H, J, A, U)); vn64 = np.zeros((B, H, J, C, U))
for b in range(B):
    for h in range(H):
        st = in64[b, h].copy()
        for j in range(J):
            st64[b, h, j] = st
            vn64[b, h, j] = u64[b, h, j] - k64[b, h, j] @ st
            st = df64[b, h, j] * st + kd64[b, h, j].T @ vn64[b, h, j]
ref = [np.zeros((B, H, J, C, A)), np.zeros((B, H, J, C, A)),
       np.zeros((B, H, J, C, U)), np.zeros((B, H, J)), np.zeros((B, H, A, U))]
for b in range(B):
    for h in range(H):
        ds = dfi64[b, h].copy()
        for j in range(J - 1, -1, -1):
            ref[3][b, h, j] = np.sum(st64[b, h, j] * ds)
            ref[1][b, h, j] = vn64[b, h, j] @ ds.T
            dv = dvn64[b, h, j] + kd64[b, h, j] @ ds
            ref[2][b, h, j] = dv
            ref[0][b, h, j] = -(dv @ st64[b, h, j].T)
            ds = df64[b, h, j] * ds - k64[b, h, j].T @ dv + dsi64[b, h, j]
        ref[4][b, h] = ds

residuals = (kcd, k_dec, u_pre, df, init)
pt._BWD_PALLAS = False
scan = pt._state_pass_bwd(residuals, ct)
pt._BWD_PALLAS = True
kern = pt._state_pass_bwd(residuals, ct)

names = ["dkcd", "dk_dec", "du_pre", "ddf", "dinitial_state"]
print(f"{'Ausgabe':>16} {'Scan vs f64':>14} {'Kernel vs f64':>16}  Urteil")
print("-" * 68)
ws = wk = 0.0
for n, t, a, b_ in zip(names, ref, scan, kern):
    sc = max(np.abs(t).max(), 1e-30)
    es = np.abs(np.asarray(a, np.float64) - t).max() / sc
    ek = np.abs(np.asarray(b_, np.float64) - t).max() / sc
    ws, wk = max(ws, es), max(wk, ek)
    print(f"{n:>16} {es:>14.3e} {ek:>16.3e}  "
          f"{'Kernel besser' if ek < es else 'Scan besser'}")
print("-" * 68)
print(f"{'schlechteste':>16} {ws:>14.3e} {wk:>16.3e}")
print("\nURTEIL:", "Kernel nicht schlechter als der bestehende Scan"
      if wk <= ws * 1.5 else "Kernel SCHLECHTER als der Scan")
