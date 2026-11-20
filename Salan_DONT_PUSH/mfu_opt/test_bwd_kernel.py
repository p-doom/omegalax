"""Compare the Pallas state-pass VJP against the pure-JAX scan it replaces.

Runs the full custom_vjp both ways on identical inputs and reports the largest
relative deviation per output. Both paths compute the same equations in the same
order in fp32; only the scheduling differs, so agreement should be at rounding
level. Anything larger means the kernel is wrong, not merely differently rounded.
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


def rnd(*shape):
    return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.float32)


kcd, k_dec = rnd(B, H, J, C, A), rnd(B, H, J, C, A)
u_pre, df = rnd(B, H, J, C, U), jnp.abs(rnd(B, H, J)) + 0.5
init = rnd(B, H, A, U)
ct = (rnd(B, H, J, A, U), rnd(B, H, J, C, U), rnd(B, H, A, U))

state_in, v_new, _ = pt._state_pass_pallas_fwd_only(kcd, k_dec, u_pre, df, init)
residuals = (kcd, k_dec, u_pre, df, init)

pt._BWD_PALLAS = False
ref = pt._state_pass_bwd(residuals, ct)
pt._BWD_PALLAS = True
new = pt._state_pass_bwd(residuals, ct)

names = ["dkcd", "dk_dec", "du_pre", "ddf", "dinitial_state"]
worst = 0.0
for name, r, n in zip(names, ref, new):
    r, n = np.asarray(r, np.float64), np.asarray(n, np.float64)
    scale = max(np.abs(r).max(), 1e-30)
    rel = np.abs(r - n).max() / scale
    worst = max(worst, rel)
    print(f"{name:>16}  shape {str(r.shape):<24} max. rel. Abweichung {rel:.3e}")
print(f"\nschlechteste Abweichung: {worst:.3e}")
print("URTEIL:", "identisch im Rundungsrahmen" if worst < 1e-5 else "ABWEICHUNG ZU GROSS")
