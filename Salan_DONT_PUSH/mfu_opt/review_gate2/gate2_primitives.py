"""Gate 2: separate algebraic correctness from reduced-precision effects.

Gate 1 (E59) compared Cascade and Conv-BTC inside the real training step, but every
observation there was quantised to bfloat16, so it could not tell an algebraic error
from roundoff. This file drives the production math functions directly, on CPU,
deterministically, and reports three clearly separated tiers of evidence.

Tier A -- Conv-BTC, true float64, production against production
---------------------------------------------------------------
``deltanet._causal_depthwise_conv1d_btc`` (the optimised BTC function from the copy)
against ``deltanet._causal_depthwise_conv1d`` (the unfused BCT production function)
with transpose adaptation. Both are dtype-polymorphic, so x64 is on and every input,
output, cotangent and gradient is asserted ``float64``. Forward and gradients.

Tier B -- the Cascade *wrapper*, true float64, production wrapper against an oracle
----------------------------------------------------------------------------------
The optimised side is the real ``cascade.wrap_delta_rule`` from the copy -- the exact
function ``cascade.install()`` installs, including its ``cascade._SEGMENTS`` contract,
which production sets in ``cascade.loss_stats``. What it wraps is a deliberately
simple, independently specified, dtype-polymorphic stateful oracle (``_oracle``,
specified in its own docstring) that satisfies the inner primitive's call contract.
One fused call over ``B*C`` rows is compared against the same oracle driven segment
by segment with the state threaded, forward and VJP, in float64.

**What Tier B does and does not establish.** It exercises the wrapper's own algebra:
the row order ``b*C + c``, the ``(B*C, T) -> (B, C*T)`` fold, the ``initial_state[::C]``
selection of the chain-entry state, and the ``jnp.repeat`` / ``[::C]`` round trip of
the final state. Because the oracle is Markovian per token, any error in that layout
or state threading shows up as a large difference, and float64 leaves no roundoff
cover. It says nothing about the chunk mathematics of the real kernel. A negative
control feeds the reference the wrong row interleaving and must fail.

Tier C -- the real chunked kernel, float32, empirical composition coverage
-------------------------------------------------------------------------
``chunk_gated_delta_rule`` itself -- the real chunked kernel, dispatched to the XLA
reference on CPU -- one fused call over C*T tokens against C calls threading
``initial_state``, mirroring ``pretrain.py:457-508`` under the reachable invariant
(``bptt_chunks == C``, reset only before segment 0, so neither ``stop_gradient`` nor
``apply_state_reset`` changes anything).

This tier **cannot** run in float64. ``chunk_gated_delta_rule_xla`` hardcodes
``.astype(jnp.float32)`` on q, k, v, beta, g and the recurrent state
(``xla_reference.py:50-53,103-105``) while deriving ``tril_mask``/``eye_LM`` from
``jnp.ones``/``jnp.eye``, which become float64 the moment x64 is on. The two meet in
the scan carry and JAX rejects the function::

    TypeError: scan body function carry input and carry output must have equal types
    input carry has type float32[...] but the output carry component has float64[...]

Both arms fail identically, so there is no float64 reference to compare against, and
making one would mean editing production code. Tier C is therefore **empirical
float32 coverage of the kernel composition, not a float64 proof**: sampled agreement
over the shapes and seeds listed below. Sampled float32 agreement -- even bitwise --
is evidence about the cases actually run; it is not a proof of an algebraic identity
and says nothing about untested shapes.

Its negative control is ``_SEGMENTS = 1``, which leaves the wrapper's own C==1
short-circuit in place so no state crosses a segment boundary; the comparison must
fail by a wide margin. A non-chunk-aligned segment length is *also* run, but as extra
coverage rather than a control: measurement showed the chunked delta rule is an exact
reorganisation of the token recurrence, so moving the chunk grid changes only
rounding, and it cannot serve as a failure injection.

Determinism of inputs
---------------------
Seeds come from ``numpy.random.SeedSequence`` over fixed integers -- never Python
``hash()``, which is salted per process and would silently give the two arms
different inputs. The derivation excludes the arm, so both arms of a comparison draw
identical arrays, and the parent **asserts bitwise equality of every stored input**
(``in_*``) before it looks at any output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = ROOT.parent / "omegalax"
CHUNK = 64            # production value
SEED_ENTROPY = 20260804  # fixed; keeps the derivation stable across processes
FAMILY_ID = {"conv": 1, "delta": 2, "wrapper": 3}


def _rng(family: str, case_index: int, seed: int, misaligned: bool) -> np.random.Generator:
    """Stable across processes, independent of the arm, so both arms match."""
    return np.random.default_rng(
        np.random.SeedSequence(
            [SEED_ENTROPY, FAMILY_ID[family], case_index, int(seed), int(misaligned)]
        )
    )


# --------------------------------------------------------------------------- cases

# (label, batch, channels, seq_len, kernel_width, with_initial_state)
CONV_CASES = [
    ("K4_T64_state", 3, 8, 64, 4, True),        # production kernel width
    ("K4_T64_nostate", 3, 8, 64, 4, False),
    ("K4_T5_state", 2, 4, 5, 4, True),          # T just above K: causal boundary
    ("K8_T9_state", 2, 6, 9, 8, True),
    ("K2_T16_nostate", 2, 5, 16, 2, False),
    ("K4_T3_state", 2, 4, 3, 4, True),          # T < K: the state fills the window
]

# (label, chains, segments, seg_len, heads, a_key, a_val, with_initial_state, padtail)
DELTA_CASES = [
    ("C2_T128", 2, 2, 128, 2, 32, 32, True, False),            # J=2 per segment
    ("C3_T192_nostate", 2, 3, 192, 2, 32, 32, False, False),   # J=3, no incoming state
    ("C4_T64", 2, 4, 64, 2, 32, 32, True, False),              # J=1 per segment
    ("C2_T128_padtail", 2, 2, 128, 2, 32, 32, True, True),     # masked tail
    ("C2_T128_wide", 1, 2, 128, 1, 128, 128, True, False),     # production head dims
]

# Same shape family for the wrapper tier, plus an odd T the wrapper must still fold.
WRAPPER_CASES = [
    ("C2_T128", 2, 2, 128, 2, 32, 32, True),
    ("C3_T192_nostate", 2, 3, 192, 2, 32, 32, False),
    ("C4_T64", 2, 4, 64, 2, 32, 32, True),
    ("C6_T32", 2, 6, 32, 2, 16, 24, True),      # C=6 as in production; A != U
    ("C2_T96_odd", 3, 2, 96, 1, 128, 128, True),
]


# --------------------------------------------------------------------------- child

def _child(args: argparse.Namespace) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"
    use_x64 = args.family in ("conv", "wrapper")
    if use_x64:
        os.environ["JAX_ENABLE_X64"] = "1"

    import jax  # noqa: PLC0415

    # Must precede array creation, or weak-typed literals keep the old default.
    jax.config.update("jax_enable_x64", use_x64)
    import jax.numpy as jnp  # noqa: PLC0415

    import omegalax  # noqa: PLC0415
    from omegalax.models.qwen3_5 import deltanet as dn  # noqa: PLC0415

    tree = args.tree.resolve()
    source = Path(omegalax.__file__).resolve()
    print(f"[child] omegalax -> {source}", flush=True)
    try:
        source.relative_to(tree / "omegalax")
    except ValueError as exc:
        raise RuntimeError(
            f"PYTHONPATH selected {source}, expected a module below {tree / 'omegalax'}"
        ) from exc

    work = jnp.float64 if use_x64 else jnp.float32
    work_name = "float64" if use_x64 else "float32"
    if jnp.zeros(1).dtype != work:
        raise RuntimeError(
            f"default dtype is {jnp.zeros(1).dtype}, expected {work_name}; "
            "the dtype assertions below would be vacuous"
        )
    print(f"[child] x64={use_x64} working dtype={work_name}", flush=True)

    dtypes: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {}

    def keep(name, value):
        """Store an array and record its dtype; hard-fail on the wrong dtype."""
        host = np.asarray(jax.device_get(value))
        dtypes[name] = str(host.dtype)
        if str(host.dtype) != work_name:
            raise AssertionError(f"{name} is {host.dtype}, expected {work_name}")
        arrays[name] = host
        return value

    def draw(rng, *shape):
        out = jnp.asarray(rng.standard_normal(shape), dtype=work)
        if out.dtype != work:
            raise AssertionError(f"input is {out.dtype}, expected {work_name}")
        return out

    def _oracle(q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH, chunk_size=64,
                initial_state_BHAU=None, *, return_final_state=False):
        """Minimal stateful operator with the inner primitive's call contract.

        Deliberately NOT the delta rule and NOT derived from the wrapper. Per token,
        with state ``S`` of shape (B, H, A, U)::

            S_t   = exp(g_t) * S_{t-1} + beta_t * outer(k_t, v_t)
            out_t = q_t . S_t                       (contract over A)

        Markovian per token, so splitting the sequence at any point and threading the
        state is exact by construction; that is what makes it a clean probe of the
        wrapper's fold and state handling. Dtype-polymorphic: no ``astype``, no
        default-dtype constants. ``chunk_size`` is accepted to satisfy the positional
        signature and is otherwise unused, since the oracle is chunk-agnostic.
        """
        B, T, H, A = q_BTHA.shape
        U = v_BTHU.shape[-1]
        if initial_state_BHAU is None:
            state = jnp.zeros((B, H, A, U), dtype=q_BTHA.dtype)
        else:
            state = initial_state_BHAU

        def step(carry, t):
            decay = jnp.exp(g_BTH[:, t])[:, :, None, None]
            update = beta_BTH[:, t][:, :, None, None] * (
                k_BTHA[:, t][:, :, :, None] * v_BTHU[:, t][:, :, None, :]
            )
            new = decay * carry + update
            out = jnp.einsum("bha,bhau->bhu", q_BTHA[:, t], new)
            return new, out

        final, outs = jax.lax.scan(step, state, jnp.arange(T))
        out_BTHU = outs.transpose(1, 0, 2, 3)
        if return_final_state:
            return out_BTHU, final
        return out_BTHU

    # ------------------------------------------------------------------ Tier A: conv
    if args.family == "conv":
        for index, (label, B, Cc, T, K, with_state) in enumerate(CONV_CASES):
            rng = _rng("conv", index, args.seed, False)
            x_BTC = draw(rng, B, T, Cc)
            weight_CK = draw(rng, Cc, K)
            init_BCS = draw(rng, B, Cc, K - 1) if with_state else None
            cot_BTC = draw(rng, B, T, Cc)
            keep(f"{label}/in_x", x_BTC)
            keep(f"{label}/in_weight", weight_CK)
            keep(f"{label}/in_cotangent", cot_BTC)
            if init_BCS is not None:
                keep(f"{label}/in_init", init_BCS)

            def optimised(x, w, s):
                return dn._causal_depthwise_conv1d_btc(x, w, initial_state_BCS=s)

            def reference(x, w, s):
                out_BCT = dn._causal_depthwise_conv1d(
                    x.transpose(0, 2, 1), w, initial_state_BCS=s, return_final_state=False
                )
                return out_BCT.transpose(0, 2, 1)

            fn = optimised if args.variant == "btc" else reference
            keep(f"{label}/out", fn(x_BTC, weight_CK, init_BCS))
            argnums = (0, 1) if init_BCS is None else (0, 1, 2)
            names = ("gx", "gweight") if init_BCS is None else ("gx", "gweight", "ginit")
            grads = jax.grad(
                lambda x, w, s: jnp.sum(fn(x, w, s) * cot_BTC), argnums=argnums
            )(x_BTC, weight_CK, init_BCS)
            for gname, garray in zip(names, grads):
                keep(f"{label}/{gname}", garray)

    # --------------------------------------------------------------- Tier B: wrapper
    elif args.family == "wrapper":
        from omegalax.trainers import cascade  # noqa: PLC0415

        for index, (label, B, C, T, H, A, U, with_state) in enumerate(WRAPPER_CASES):
            rng = _rng("wrapper", index, args.seed, False)
            q = draw(rng, B, C, T, H, A)
            k = draw(rng, B, C, T, H, A)
            v = draw(rng, B, C, T, H, U)
            g = -jnp.abs(draw(rng, B, C, T, H)) * 0.05
            beta = jax.nn.sigmoid(draw(rng, B, C, T, H))
            init = draw(rng, B, H, A, U) if with_state else None
            cot = draw(rng, B, C, T, H, U)
            cot_state = draw(rng, B, H, A, U)
            for name, value in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
                                ("cotangent", cot), ("cotangent_state", cot_state)):
                keep(f"{label}/in_{name}", value)
            if init is not None:
                keep(f"{label}/in_init", init)

            def rows(x, interleave):
                """b*C+c (the wrapper's contract) or the wrong c*B+b interleaving."""
                if interleave == "bc":
                    return x.reshape((B * C,) + x.shape[2:])
                return x.transpose(1, 0, *range(2, x.ndim)).reshape((B * C,) + x.shape[2:])

            def fused(q_, k_, v_, g_, beta_, init_, interleave="bc"):
                """The real production wrapper around the oracle."""
                cascade._SEGMENTS = C   # the global cascade.loss_stats sets in production
                wrapped = cascade.wrap_delta_rule(_oracle)
                zeros = None if init_ is None else jnp.zeros_like(init_)
                packed = None if init_ is None else rows(
                    jnp.stack([init_ if c == 0 else zeros for c in range(C)], axis=1),
                    interleave,
                )
                out_rows, final_rows = wrapped(
                    rows(q_, interleave), rows(k_, interleave), rows(v_, interleave),
                    rows(g_, interleave), rows(beta_, interleave),
                    CHUNK, packed, return_final_state=True,
                )
                return out_rows.reshape((B, C) + out_rows.shape[1:]), final_rows[::C]

            def segmented(q_, k_, v_, g_, beta_, init_):
                """The oracle driven segment by segment, state threaded."""
                state = init_
                outs = []
                for c in range(C):
                    out_c, state = _oracle(
                        q_[:, c], k_[:, c], v_[:, c], g_[:, c], beta_[:, c],
                        CHUNK, state, return_final_state=True,
                    )
                    outs.append(out_c)
                return jnp.stack(outs, axis=1), state

            for tag, fn in (("fused", fused), ("segmented", segmented)):
                out, final = fn(q, k, v, g, beta, init)
                keep(f"{label}/{tag}_out", out)
                keep(f"{label}/{tag}_final", final)
                argnums = (0, 1, 2, 3, 4) if init is None else (0, 1, 2, 3, 4, 5)
                names = ["gq", "gk", "gv", "gg", "gbeta"] + ([] if init is None else ["ginit"])

                def scalar(*inputs, _fn=fn):
                    o, f = _fn(*inputs)
                    return jnp.sum(o * cot) + jnp.sum(f * cot_state)

                for gname, garray in zip(
                    names, jax.grad(scalar, argnums=argnums)(q, k, v, g, beta, init)
                ):
                    keep(f"{label}/{tag}_{gname}", garray)

            # Negative control: the wrapper fed the wrong row interleaving. Must differ.
            wrong_out, wrong_final = fused(q, k, v, g, beta, init, interleave="cb")
            keep(f"{label}/wrongorder_out", wrong_out)
            keep(f"{label}/wrongorder_final", wrong_final)

    # ----------------------------------------------------------------- Tier C: delta
    else:
        from omegalax.models.qwen3_5.kernels import chunk_gated_delta_rule  # noqa: PLC0415

        cascade = None
        if args.variant == "cascade":
            from omegalax.trainers import cascade  # noqa: PLC0415

        for index, case in enumerate(DELTA_CASES):
            label, B, C, T, H, A, U, with_state, padtail = case
            if args.misaligned:
                T = T - 32   # not a multiple of CHUNK: negative control
            rng = _rng("delta", index, args.seed, args.misaligned)
            q = draw(rng, B, C, T, H, A)
            k = draw(rng, B, C, T, H, A)
            v = draw(rng, B, C, T, H, U)
            # g is a log-decay and must be <= 0, as -exp(A_log)*softplus produces.
            g = -jnp.abs(draw(rng, B, C, T, H)) * 0.05
            beta = jax.nn.sigmoid(draw(rng, B, C, T, H))
            init = draw(rng, B, H, A, U) if with_state else None
            if padtail:
                # The only masking this primitive can see: GatedDeltaNet zeroes q, k, v,
                # beta and g on invalid positions before calling it (deltanet.py:330-337).
                valid_T = (jnp.arange(T) < T // 2).astype(work)
                mask = jnp.concatenate(
                    [jnp.ones((C - 1, T), dtype=work), valid_T[None, :]], axis=0
                )[None, :, :, None]
                q, k, v = q * mask[..., None], k * mask[..., None], v * mask[..., None]
                g, beta = g * mask, beta * mask
            cot = draw(rng, B, C, T, H, U)
            cot_state = draw(rng, B, H, A, U)
            for name, value in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
                                ("cotangent", cot), ("cotangent_state", cot_state)):
                keep(f"{label}/in_{name}", value)
            if init is not None:
                keep(f"{label}/in_init", init)

            def reference(q_, k_, v_, g_, beta_, init_):
                state = init_
                outs = []
                for c in range(C):
                    out_c, state = chunk_gated_delta_rule(
                        q_[:, c], k_[:, c], v_[:, c], g_[:, c], beta_[:, c],
                        CHUNK, state, return_final_state=True,
                    )
                    outs.append(out_c)
                return jnp.stack(outs, axis=1), state

            def optimised(q_, k_, v_, g_, beta_, init_):
                # negctl=segments1 leaves the wrapper's own C==1 short-circuit in place,
                # so no fold happens and no state crosses a segment boundary. That is the
                # failure the comparison must be able to see.
                cascade._SEGMENTS = 1 if args.negctl == "segments1" else C
                wrapped = cascade.wrap_delta_rule(chunk_gated_delta_rule)

                def rows(x):
                    return x.reshape((B * C,) + x.shape[2:])

                if init_ is None:
                    packed = None
                else:
                    zeros = jnp.zeros_like(init_)
                    packed = rows(
                        jnp.stack([init_ if c == 0 else zeros for c in range(C)], axis=1)
                    )
                out_rows, final_rows = wrapped(
                    rows(q_), rows(k_), rows(v_), rows(g_), rows(beta_),
                    CHUNK, packed, return_final_state=True,
                )
                return out_rows.reshape((B, C) + out_rows.shape[1:]), final_rows[::C]

            fn = optimised if args.variant == "cascade" else reference
            out, final = fn(q, k, v, g, beta, init)
            keep(f"{label}/out", out)
            keep(f"{label}/final", final)
            argnums = (0, 1, 2, 3, 4) if init is None else (0, 1, 2, 3, 4, 5)
            names = ["gq", "gk", "gv", "gg", "gbeta"] + ([] if init is None else ["ginit"])

            def scalar(*inputs):
                o, f = fn(*inputs)
                return jnp.sum(o * cot) + jnp.sum(f * cot_state)

            for gname, garray in zip(
                names, jax.grad(scalar, argnums=argnums)(q, k, v, g, beta, init)
            ):
                keep(f"{label}/{gname}", garray)

    meta = {
        "source": str(source),
        "tree": str(tree),
        "family": args.family,
        "variant": args.variant,
        "seed": args.seed,
        "misaligned": bool(args.misaligned),
        "negctl": args.negctl,
        "chunk": CHUNK,
        "x64": use_x64,
        "work_dtype": work_name,
        "devices": [str(d) for d in jax.devices()],
        "dtypes": dtypes,
    }
    arrays["__meta__"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(args.out, **arrays)
    print(f"[child] {args.family}/{args.variant} seed={args.seed} "
          f"arrays={len(arrays) - 1} all dtype {work_name}", flush=True)


# -------------------------------------------------------------------------- parent

def _run(args, tree: Path, family: str, variant: str, out: Path, misaligned: bool,
         negctl: str = "none") -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree)
    env["WANDB_MODE"] = "disabled"
    env["OMEGALAX_STARTUP_LOG"] = "0"
    env["PYTHONHASHSEED"] = "0"   # belt and braces; the seed derivation does not use hash()
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"), str(Path(__file__).resolve()),
        "--child", "--tree", str(tree), "--out", str(out),
        "--family", family, "--variant", variant, "--seed", str(args.seed),
        "--negctl", negctl,
    ]
    if misaligned:
        cmd.append("--misaligned")
    print(f"\n=== {family}/{variant} tree={tree.name} misaligned={misaligned} "
          f"negctl={negctl}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def _stats(left: np.ndarray, right: np.ndarray) -> dict:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    d = np.abs(a - b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {
        "max_abs": float(d.max()) if d.size else 0.0,
        "rel_l2": float(np.linalg.norm(d) / max(na, 1e-300)),
        "cosine": float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 1.0,
        "exact": bool(np.array_equal(a, b)),
        "ref_absmax": float(np.abs(a).max()) if a.size else 0.0,
        "finite": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))),
    }


def _group(per: dict, keys: list[str], tol: float) -> dict:
    if not keys:
        return {}
    worst = max(keys, key=lambda k: per[k]["rel_l2"])
    return {
        "arrays": len(keys),
        "all_exact": all(per[k]["exact"] for k in keys),
        "all_finite": all(per[k]["finite"] for k in keys),
        "max_abs": max(per[k]["max_abs"] for k in keys),
        "max_rel_l2": max(per[k]["rel_l2"] for k in keys),
        "min_cosine": min(per[k]["cosine"] for k in keys),
        "worst_array": worst,
        "within_tol": max(per[k]["rel_l2"] for k in keys) <= tol,
    }


def _report(label: str, summary: dict) -> None:
    print(f"\n--- {label}")
    print(f"  left : {summary['left']}  ({summary['left_variant']})")
    print(f"  right: {summary['right']}  ({summary['right_variant']})")
    print(f"  x64={summary['x64']} work_dtype={summary['work_dtype']} "
          f"inputs_identical={summary['inputs_identical']} "
          f"({summary['inputs_checked']} arrays)")
    for name in ("forward", "grad"):
        s = summary.get(name)
        if not s:
            continue
        print(f"  {name:<8} arrays={s['arrays']:<3} exact={str(s['all_exact']):<5} "
              f"max_abs={s['max_abs']:.3e} rel_l2={s['max_rel_l2']:.3e} "
              f"1-cos={1.0 - s['min_cosine']:.3e} worst={s['worst_array']}")


def _compare_files(label: str, left: Path, right: Path, tol: float) -> dict:
    with np.load(left, allow_pickle=False) as ld, np.load(right, allow_pickle=False) as rd:
        lmeta, rmeta = (json.loads(str(d["__meta__"])) for d in (ld, rd))
        keys = sorted(k for k in ld.files if k != "__meta__")
        if keys != sorted(k for k in rd.files if k != "__meta__"):
            raise RuntimeError(f"{label}: array sets differ between arms")
        # Inputs first: if these are not bitwise identical, nothing downstream means
        # anything. This is the guard against a per-process seed drift.
        in_keys = [k for k in keys if "/in_" in k]
        mismatched = [k for k in in_keys if not np.array_equal(ld[k], rd[k])]
        if mismatched:
            raise RuntimeError(
                f"{label}: inputs differ between arms ({len(mismatched)} arrays, "
                f"e.g. {mismatched[:3]}); the comparison would be meaningless"
            )
        per = {k: _stats(ld[k], rd[k]) for k in keys if k not in in_keys}
    forward = [k for k in per if k.endswith("/out") or k.endswith("/final")]
    grad = [k for k in per if k not in forward]
    summary = {
        "left": lmeta["source"], "right": rmeta["source"],
        "left_variant": lmeta["variant"], "right_variant": rmeta["variant"],
        "misaligned": rmeta["misaligned"],
        "x64": lmeta["x64"] and rmeta["x64"],
        "work_dtype": f"{lmeta['work_dtype']}/{rmeta['work_dtype']}",
        "inputs_identical": True, "inputs_checked": len(in_keys),
        "tol": tol, "per_array": per,
        "forward": _group(per, forward, tol), "grad": _group(per, grad, tol),
    }
    _report(label, summary)
    return summary


def _compare_within(label: str, path: Path, left_tag: str, right_tag: str, tol: float) -> dict:
    """Tier B lives in one process, so inputs are identical by construction."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        pairs = {}
        for key in data.files:
            if key == "__meta__" or f"/{left_tag}_" not in key:
                continue
            other = key.replace(f"/{left_tag}_", f"/{right_tag}_")
            if other in data.files:
                pairs[key.replace(f"/{left_tag}_", "/")] = _stats(data[key], data[other])
        n_inputs = sum(1 for k in data.files if "/in_" in k)
    if not pairs:
        raise RuntimeError(f"{label}: no {left_tag}/{right_tag} pairs found")
    forward = [k for k in pairs if k.endswith("/out") or k.endswith("/final")]
    grad = [k for k in pairs if k not in forward]
    summary = {
        "left": meta["source"], "right": meta["source"],
        "left_variant": f"{meta['variant']}:{left_tag}",
        "right_variant": f"{meta['variant']}:{right_tag}",
        "misaligned": False, "x64": meta["x64"],
        "work_dtype": f"{meta['work_dtype']}/{meta['work_dtype']}",
        "inputs_identical": True, "inputs_checked": n_inputs,
        "tol": tol, "per_array": pairs,
        "forward": _group(pairs, forward, tol), "grad": _group(pairs, grad, tol),
    }
    _report(label, summary)
    return summary


def _parent(args: argparse.Namespace) -> int:
    if not ORIGINAL.is_dir():
        raise RuntimeError(f"Original tree not found: {ORIGINAL}")
    out_dir = Path(args.results).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "roots": {"original": str(ORIGINAL), "copy": str(ROOT)},
        "seed": args.seed, "chunk": CHUNK, "seed_entropy": SEED_ENTROPY,
        "tolerances": {"f64": args.tol_f64, "f32": args.tol_f32},
        "cases": {
            "conv": [c[0] for c in CONV_CASES],
            "wrapper": [c[0] for c in WRAPPER_CASES],
            "delta": [c[0] for c in DELTA_CASES],
        },
        "comparisons": {},
    }
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="gate2_", dir=HERE) as temp_dir:
        temp = Path(temp_dir)
        jobs = {
            ("conv", "orig_ref"): (ORIGINAL, "conv", "bct", False),
            ("conv", "opt_ref"): (ROOT, "conv", "bct", False),
            ("conv", "opt_new"): (ROOT, "conv", "btc", False),
            ("wrapper", "opt"): (ROOT, "wrapper", "cascade", False),
            ("delta", "orig_ref"): (ORIGINAL, "delta", "loop", False),
            ("delta", "opt_ref"): (ROOT, "delta", "loop", False),
            ("delta", "opt_new"): (ROOT, "delta", "cascade", False),
            ("delta_odd", "orig_ref"): (ORIGINAL, "delta", "loop", True),
            ("delta_odd", "opt_new"): (ROOT, "delta", "cascade", True),
            ("delta_neg", "opt_new"): (ROOT, "delta", "cascade", False),
        }
        negctl = {("delta_neg", "opt_new"): "segments1"}
        paths = {}
        for (family, arm), (tree, fam, variant, misaligned) in jobs.items():
            path = temp / f"{family}_{arm}.npz"
            _run(args, tree, fam, variant, path, misaligned,
                 negctl.get((family, arm), "none"))
            paths[(family, arm)] = path

        cmp = report["comparisons"]
        # Tier A
        cmp["A1 conv: trees agree (orig BCT vs copy BCT), float64"] = _compare_files(
            "A1 conv: original BCT vs copy BCT -- must be exact",
            paths[("conv", "orig_ref")], paths[("conv", "opt_ref")], args.tol_f64)
        cmp["A2 conv: optimised BTC vs original BCT, float64"] = _compare_files(
            "A2 conv: ORIGINAL BCT vs OPTIMISED BTC -- Tier A, true float64",
            paths[("conv", "orig_ref")], paths[("conv", "opt_new")], args.tol_f64)
        checks.append(("A1 conv trees agree", cmp[
            "A1 conv: trees agree (orig BCT vs copy BCT), float64"]["forward"]["all_exact"]))
        a2 = cmp["A2 conv: optimised BTC vs original BCT, float64"]
        checks.append(("A2 conv BTC == BCT forward", a2["forward"]["within_tol"]))
        checks.append(("A2 conv BTC == BCT grads", a2["grad"]["within_tol"]))
        # Tier B
        cmp["B1 wrapper: fused vs segmented oracle, float64"] = _compare_within(
            "B1 wrapper: real cascade.wrap_delta_rule(oracle) fused vs segmented, float64",
            paths[("wrapper", "opt")], "fused", "segmented", args.tol_f64)
        cmp["B2 wrapper NEGATIVE CONTROL: wrong row interleaving"] = _compare_within(
            "B2 wrapper NEGATIVE CONTROL: wrong row interleaving must NOT agree",
            paths[("wrapper", "opt")], "wrongorder", "segmented", args.tol_f64)
        b1 = cmp["B1 wrapper: fused vs segmented oracle, float64"]
        checks.append(("B1 wrapper forward", b1["forward"]["within_tol"]))
        checks.append(("B1 wrapper grads", b1["grad"]["within_tol"]))
        b2 = cmp["B2 wrapper NEGATIVE CONTROL: wrong row interleaving"]
        # A control that merely fails exactness proves nothing; require a wide margin.
        checks.append(("B2 negative control detects wrong fold",
                       b2["forward"]["max_rel_l2"] > 1e-3))
        # Tier C
        cmp["C1 delta: trees agree (orig loop vs copy loop), float32"] = _compare_files(
            "C1 delta: original loop vs copy loop -- must be exact",
            paths[("delta", "orig_ref")], paths[("delta", "opt_ref")], args.tol_f32)
        cmp["C2 delta: real kernel fused vs loop, float32"] = _compare_files(
            "C2 delta: ORIGINAL loop vs OPTIMISED cascade, real kernel, float32",
            paths[("delta", "orig_ref")], paths[("delta", "opt_new")], args.tol_f32)
        # Extra coverage, NOT a negative control: segment length not a multiple of the
        # chunk size. The chunked delta rule is an exact reorganisation of the token
        # recurrence, so the chunk grid changes only rounding -- measured, see E60.
        cmp["C3 delta: segment length not a multiple of chunk (extra coverage)"] = (
            _compare_files(
                "C3 delta: non-chunk-aligned segments, extra coverage (not a control)",
                paths[("delta_odd", "orig_ref")], paths[("delta_odd", "opt_new")],
                args.tol_f32))
        # The real negative control: wrapper short-circuited so no state crosses a
        # segment boundary. The comparison must see that.
        cmp["C4 delta NEGATIVE CONTROL: no state threading (_SEGMENTS=1)"] = _compare_files(
            "C4 delta NEGATIVE CONTROL: wrapper disabled, no state hand-off, must NOT agree",
            paths[("delta", "orig_ref")], paths[("delta_neg", "opt_new")], args.tol_f32)
        checks.append(("C1 delta trees agree", cmp[
            "C1 delta: trees agree (orig loop vs copy loop), float32"]["forward"]["all_exact"]))
        c2 = cmp["C2 delta: real kernel fused vs loop, float32"]
        checks.append(("C2 delta forward", c2["forward"]["within_tol"]))
        checks.append(("C2 delta grads", c2["grad"]["within_tol"]))
        c3 = cmp["C3 delta: segment length not a multiple of chunk (extra coverage)"]
        checks.append(("C3 delta non-aligned forward", c3["forward"]["within_tol"]))
        c4 = cmp["C4 delta NEGATIVE CONTROL: no state threading (_SEGMENTS=1)"]
        # A real control must fail by a wide margin, not merely be non-exact.
        checks.append(("C4 negative control detects missing state threading",
                       c4["forward"]["max_rel_l2"] > 1e-3))

    target = out_dir / f"gate2_summary_seed{args.seed}.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nsummary -> {target}")
    print("\n=== checks")
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    print("\nNo checkpoints written; temporary arrays removed with the scope.")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--family", choices=("conv", "delta", "wrapper"), default="conv")
    p.add_argument("--variant", choices=("bct", "btc", "loop", "cascade"), default="bct")
    p.add_argument("--misaligned", action="store_true")
    p.add_argument("--negctl", choices=("none", "segments1"), default="none")
    p.add_argument("--seed", type=int, default=0)
    # float64 tiers: expect agreement at the f64 roundoff level or exact.
    p.add_argument("--tol-f64", type=float, default=1e-12)
    # float32 tier: forward expected exact, gradients only to f32 roundoff.
    p.add_argument("--tol-f32", type=float, default=1e-4)
    p.add_argument(
        "--results",
        default="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt/gate2",
    )
    args = p.parse_args()
    if args.child:
        _child(args)
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
