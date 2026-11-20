"""Find which construct in the state-pass VJP kernel crashes Triton lowering.

The kernel runs fine in Pallas interpret mode but segfaults when lowered for the
GPU, so the fault is in code generation or resource allocation, not in the logic.
Rather than guess, build up the body in four steps and report the first that dies.
Each variant is a separate subprocess so a segfault does not take the rest down.
"""

import os
import subprocess
import sys

VARIANT = os.environ.get("BISECT_VARIANT")

if VARIANT is None:
    for v in ("carry", "carry_dupre", "carry_dupre_dkcd", "full"):
        env = dict(os.environ, BISECT_VARIANT=v, JAX_PLATFORMS="cuda")
        r = subprocess.run([sys.executable, __file__], env=env,
                           capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()
        note = tail[-1][:110] if tail else ""
        print(f"{v:<20} rc={r.returncode:<5} {'OK' if r.returncode == 0 else 'CRASH'}  {note}")
    sys.exit(0)

import functools  # noqa: E402

import jax  # noqa: E402
import jax.experimental.pallas as pl  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.pallas import triton as plt  # noqa: E402

B, H, J, C, A, U = 2, 4, 16, 64, 128, 128


def make(variant):
    def kern(kcd_ref, k_dec_ref, state_ref, v_ref, df_ref, dsin_ref, dvn_ref, dfin_ref,
             dkcd_ref, dk_dec_ref, du_ref, ddf_ref, dinit_ref):
        def body(i, dstate):
            j = J - 1 - i
            dv = dvn_ref[j] + pl.dot(k_dec_ref[j], dstate)
            if variant != "carry":
                du_ref[j] = dv
            if variant in ("carry_dupre_dkcd", "full"):
                dkcd_ref[j] = -pl.dot(dv, state_ref[j].T)
                dk_dec_ref[j] = pl.dot(v_ref[j], dstate.T)
            if variant == "full":
                ddf_ref[j] = jnp.sum(state_ref[j] * dstate)
            return df_ref[j] * dstate - pl.dot(kcd_ref[j].T, dv) + dsin_ref[j]

        dinit_ref[:, :] = jax.lax.fori_loop(0, J, body, dfin_ref[:, :])

    return kern


sa = pl.BlockSpec((None, None, J, C, A), lambda b, h: (b, h, 0, 0, 0))
su = pl.BlockSpec((None, None, J, C, U), lambda b, h: (b, h, 0, 0, 0))
ss = pl.BlockSpec((None, None, J, A, U), lambda b, h: (b, h, 0, 0, 0))
sd = pl.BlockSpec((None, None, J), lambda b, h: (b, h, 0))
si = pl.BlockSpec((None, None, A, U), lambda b, h: (b, h, 0, 0))
rng = np.random.default_rng(0)


def r(*s):
    return jnp.asarray(rng.standard_normal(s) * 0.1, dtype=jnp.float32)


out = pl.pallas_call(
    make(VARIANT),
    out_shape=(jax.ShapeDtypeStruct((B, H, J, C, A), jnp.float32),
               jax.ShapeDtypeStruct((B, H, J, C, A), jnp.float32),
               jax.ShapeDtypeStruct((B, H, J, C, U), jnp.float32),
               jax.ShapeDtypeStruct((B, H, J), jnp.float32),
               jax.ShapeDtypeStruct((B, H, A, U), jnp.float32)),
    grid=(B, H),
    in_specs=(sa, sa, ss, su, sd, ss, su, si),
    out_specs=(sa, sa, su, sd, si),
    compiler_params=plt.CompilerParams(num_warps=4, num_stages=2),
)(r(B, H, J, C, A), r(B, H, J, C, A), r(B, H, J, A, U), r(B, H, J, C, U),
  jnp.abs(r(B, H, J)) + 0.5, r(B, H, J, A, U), r(B, H, J, C, U), r(B, H, A, U))
jax.block_until_ready(out)
print("ok")
