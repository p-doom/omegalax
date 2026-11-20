"""Gate 3: is the MFU_TRIBLOCK drift seen in E59 roundoff, or a silent bug?

E59 measured that the blocked chunk inverse is the only one of the three switches
that perturbs the forward pass (hidden rel L2 1.03e-03, GDN state 7e-08, loss
1.06e-05). E52/E56 established that the blocked inverse is accurate, but only against
*independently sampled* lower-triangular matrices -- and E50 is the recorded lesson
that synthetic matrices miss the real value range. This file closes that gap: it takes
the A_pre matrices the **production forward actually produced** and compares the two
inverse algorithms on exactly those.

Provenance -- how each A_pre is proven to come from production
-------------------------------------------------------------
``A_pre`` is a local inside ``chunk_gated_delta_rule_pallas``
(``pallas_triton.py:482-512``). It is captured by wrapping the module-global
``_blocked_unit_lower_inverse`` with a pass-through recorder, so the tensor handed to
the recorder *is* the argument production computed one line earlier. The recorder also
counts calls, which doubles as the dispatch proof: with ``MFU_TRIBLOCK=1`` the count
must equal the number of GDN layers, and the ``opt_off`` / ``orig`` arms must never
reach it.

The second seam is ``_state_pass_pallas``, whose four inputs are captured in *every*
arm. Two of them, ``k_dec_BHJCA`` and ``df_BHJ``, are computed independently of the
inverse; if those are bit-identical across arms, everything upstream of the inverse --
A_pre included -- was identical, without needing to capture A_pre twice. The other
two, ``u_pre_BHJCU`` and ``kcd_BHJCA``, are precisely ``A_inv @ vb`` and
``A_inv @ kb_decayed``, i.e. the immediate downstream of the inverse. Comparing them
across arms connects inverse error to the hidden drift E59 reported, using nothing but
production code.

The GDN calls the kernel inside ``shard_map`` (``deltanet.py:269``), so a tracer
captured there cannot be returned from the outer jit -- JAX rejects it as a leaked
value, which is how the first attempt at this harness failed. Values are therefore
moved to the host from inside with ``jax.debug.callback``.

Three capture arms, each in its own process with PYTHONPATH pinned:
``orig`` (original tree, trsm path), ``opt_off`` (copy, MFU_TRIBLOCK=0),
``opt_on`` (copy, MFU_TRIBLOCK=1).

Analysis on the captured matrices
---------------------------------
On the exact captured A_pre, at production precision (float32, GPU):

* ``blocked`` = the real ``_blocked_unit_lower_inverse(A_pre, 16)`` from the copy.
* ``trsm``    = ``solve_triangular(eye - A_pre, broadcast(eye), lower=True)`` -- the
  production ``else`` branch, verbatim.
* ``oracle``  = ``numpy.linalg.inv(I - A_pre)`` in **float64** on the host. This is a
  float64 reference on captured matrices; it is deliberately NOT called the production
  path, because the production primitive cannot run in float64 (see E60).

Reported per pair: max_abs, relative L2, cosine, percentiles, worst matrix, plus the
residual ``||(I - A_pre) @ inv - I||`` (the actual system being solved; the brief asks
for ``A_pre @ inv - I``, which is not the identity this code solves), and conditioning
indicators. Then the VJP of each float32 path with respect to A_pre under identical
deterministic cotangents, against each other and against the float64 analytic
reference ``X^T Cot X^T`` for ``X = (I - A_pre)^{-1}``.

Shapes are production-critical where they matter: chunk size 64, block width 16,
linear key/value head dim 128. Only hidden size, vocabulary and layer count are shrunk.
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
CHUNK = 64          # production delta-rule chunk size
TRI_BLOCK = 16      # production block width for the blocked inverse (E52)
CHAINS = 2
SEQ_LEN = 1024      # J = 16 chunks per row
PERCENTILES = (50.0, 90.0, 99.0, 99.9, 100.0)


# --------------------------------------------------------------------------- child

def _build_forward(jax, jnp, P_mod, text_api, ShardConfig, Qwen3_5TextConfig,
                   set_attn_backend, ensure_mesh, seed):
    """The production Qwen3.5 forward, minimal except where shapes drive branches."""
    cfg = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention", "linear_attention", "linear_attention",
                     "full_attention"),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(4, 2, 2),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,     # production
        linear_num_key_heads=1,
        linear_num_value_heads=4,
        linear_value_head_dim=128,   # production
        intermediate_size=256,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.bfloat16,
    )
    mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
    model_cfg = text_api.align_config_to_mesh(cfg, mesh)
    from jax.sharding import NamedSharding, PartitionSpec  # noqa: PLC0415

    key = jax.device_put(jax.random.key(seed), NamedSharding(mesh, PartitionSpec()))
    key, _ = jax.random.split(key)
    model, model_cfg = P_mod.init_model(model_cfg, key, tp_size=1, fsdp_size=1, dp_size=1)
    set_attn_backend(model, text_backend="cudnn")
    rng = np.random.default_rng(4000 + seed)
    tokens = jnp.asarray(
        rng.integers(1, cfg.vocab_size - 1, size=(CHAINS, SEQ_LEN), dtype=np.int32)
    )
    mask = jnp.ones((CHAINS, SEQ_LEN), dtype=jnp.int32)
    return model, model_cfg, tokens, mask


def _child_capture(args) -> None:
    # "cuda,cpu": cuda stays the default device, but jax.debug.callback needs a local
    # CPU device to stage its host transfer onto, and a cuda-only client has none.
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    os.environ["OMEGALAX_DELTANET_KERNEL"] = "pallas"

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415

    import omegalax  # noqa: PLC0415
    from omegalax.distributed.mesh import ensure_mesh  # noqa: PLC0415
    from omegalax.models.qwen3_5.config import Qwen3_5TextConfig  # noqa: PLC0415
    from omegalax.models.qwen3_5.kernels import pallas_triton as pt  # noqa: PLC0415
    from omegalax.models.shard_config import ShardConfig  # noqa: PLC0415
    from omegalax.models.sharding_runtime import set_attn_backend  # noqa: PLC0415
    from omegalax.text import api as text_api  # noqa: PLC0415
    from omegalax.trainers import pretrain as P_mod  # noqa: PLC0415
    from absl import flags as absl_flags  # noqa: PLC0415

    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS(["gate3_triblock"])

    tree = args.tree.resolve()
    source = Path(omegalax.__file__).resolve()
    print(f"[child] omegalax -> {source}", flush=True)
    source.relative_to(tree / "omegalax")   # raises if PYTHONPATH picked the wrong tree
    default_platform = jax.devices()[0].platform
    print(f"[child] devices={[str(d) for d in jax.devices()]} default={default_platform}",
          flush=True)
    if default_platform != "gpu":
        raise RuntimeError(
            f"default device platform is {default_platform!r}, expected gpu; the Pallas "
            "kernel would not be the thing under test"
        )

    counters = {"blocked_inverse": 0, "state_pass": 0}
    # Keyed by a trace-time tag: if JAX traces a layer more than once the same key is
    # overwritten with identical data rather than silently appending a duplicate, and
    # an unexpected key count fails loudly below.
    host_store: dict[str, dict[int, np.ndarray]] = {}

    def sink(name: str, tag: int):
        def callback(value):
            host_store.setdefault(name, {})[tag] = np.asarray(value)
        return callback

    real_blocked = getattr(pt, "_blocked_unit_lower_inverse", None)
    has_blocked = real_blocked is not None
    if has_blocked:
        def recording_blocked(n_mat, block):
            # n_mat IS the A_pre production computed one line earlier.
            tag = counters["blocked_inverse"]
            counters["blocked_inverse"] += 1
            jax.debug.callback(sink("a_pre", tag), n_mat)
            return real_blocked(n_mat, block)

        pt._blocked_unit_lower_inverse = recording_blocked

    real_state_pass = pt._state_pass_pallas

    def recording_state_pass(kcd, k_dec, u_pre, df, initial_state):
        tag = counters["state_pass"]
        counters["state_pass"] += 1
        for name, value in (("kcd", kcd), ("k_dec", k_dec),
                            ("u_pre", u_pre), ("df", df)):
            jax.debug.callback(sink(name, tag), value)
        return real_state_pass(kcd, k_dec, u_pre, df, initial_state)

    pt._state_pass_pallas = recording_state_pass

    model, model_cfg, tokens, mask = _build_forward(
        jax, jnp, P_mod, text_api, ShardConfig, Qwen3_5TextConfig,
        set_attn_backend, ensure_mesh, args.seed,
    )

    @nnx.jit
    def forward(model_, tokens_, mask_):
        return text_api.forward_with_gdn_state(
            model_, tokens_, 0, model_cfg, attention_mask_BT=mask_, initial_gdn_states=None
        )

    for key in counters:
        counters[key] = 0
    host_store.clear()
    hidden, aux, final = forward(model, tokens, mask)
    jax.block_until_ready((hidden, aux, final))   # the host callbacks must have fired

    n_captured = len(host_store.get("kcd", {}))
    if n_captured != counters["state_pass"]:
        raise RuntimeError(
            f"captured {n_captured} state-pass seams but traced {counters['state_pass']}; "
            "a layer was traced more than once and the capture is ambiguous"
        )

    arrays: dict[str, np.ndarray] = {
        "hidden": np.asarray(jax.device_get(hidden), dtype=np.float32),
        "aux": np.asarray(jax.device_get(aux), dtype=np.float32),
    }
    for layer_idx, state in enumerate(final):
        arrays[f"final_state/{layer_idx}"] = np.asarray(jax.device_get(state), np.float32)
    dtypes = {}
    for name, tagged in host_store.items():
        for tag in sorted(tagged):
            host = tagged[tag]
            arrays[f"{name}/{tag}"] = host
            dtypes[f"{name}/{tag}"] = str(host.dtype)

    meta = {
        "source": str(source), "tree": str(tree), "arm": args.arm, "seed": args.seed,
        "chunk": CHUNK, "tri_block": TRI_BLOCK, "chains": CHAINS, "seq_len": SEQ_LEN,
        "dispatch": dict(counters),
        "has_blocked_symbol": has_blocked,
        "switches": {
            "pallas_triton._TRI_BLOCKED": bool(getattr(pt, "_TRI_BLOCKED", False)),
            "pallas_triton._TRI_INV": bool(getattr(pt, "_TRI_INV", False)),
            "MFU_TRIBLOCK": os.environ.get("MFU_TRIBLOCK", ""),
        },
        "kernel": os.environ.get("OMEGALAX_DELTANET_KERNEL", ""),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "devices": [str(d) for d in jax.devices()],
        "dtypes": dtypes,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    arrays["__meta__"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(args.out, **arrays)
    print(f"[child] arm={args.arm} dispatch={counters} switches={meta['switches']}", flush=True)
    print(f"[child] arrays={len(arrays) - 1} -> {args.out}", flush=True)


def _child_analyse(args) -> None:
    """Both inverse algorithms plus a float64 oracle, on the captured A_pre."""
    # "cuda,cpu": cuda stays the default device, but jax.debug.callback needs a local
    # CPU device to stage its host transfer onto, and a cuda-only client has none.
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    os.environ["OMEGALAX_DELTANET_KERNEL"] = "pallas"

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    import omegalax  # noqa: PLC0415
    from omegalax.models.qwen3_5.kernels import pallas_triton as pt  # noqa: PLC0415

    tree = args.tree.resolve()
    source = Path(omegalax.__file__).resolve()
    print(f"[child] omegalax -> {source}", flush=True)
    source.relative_to(tree / "omegalax")

    out: dict[str, object] = {"source": str(source), "per_layer": {}}
    with np.load(args.apre, allow_pickle=False) as data:
        capture_meta = json.loads(str(data["__meta__"]))
        layers = sorted(k for k in data.files if k.startswith("a_pre/"))
        for key in layers:
            out["per_layer"][key] = _analyse_one(
                jax, jnp, pt, np.asarray(data[key]), args.seed)
    out["capture_dispatch"] = capture_meta["dispatch"]
    out["capture_source"] = capture_meta["source"]
    out["a_pre_dtype"] = capture_meta["dtypes"].get(layers[0], "?") if layers else "?"
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[child] analysed {len(layers)} layers -> {args.out}", flush=True)


def _pair(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> dict:
    fa = np.asarray(a, dtype=np.float64)
    fb = np.asarray(b, dtype=np.float64)
    d = np.abs(fa - fb)
    flat_a, flat_b, flat_d = fa.ravel(), fb.ravel(), d.ravel()
    na, nb = float(np.linalg.norm(flat_a)), float(np.linalg.norm(flat_b))
    mat = d.reshape(-1, d.shape[-2], d.shape[-1])
    ref = fa.reshape(mat.shape)
    per_mat = np.linalg.norm(mat.reshape(mat.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(ref.reshape(mat.shape[0], -1), axis=1), 1e-300
    )
    worst = int(np.argmax(per_mat))
    return {
        "pair": f"{name_a} vs {name_b}",
        "max_abs": float(flat_d.max()),
        "rel_l2": float(np.linalg.norm(flat_d) / max(na, 1e-300)),
        "cosine": float(np.dot(flat_a, flat_b) / (na * nb)) if na > 0 and nb > 0 else 1.0,
        "exact": bool(np.array_equal(fa, fb)),
        "abs_percentiles": {f"p{p:g}": float(np.percentile(flat_d, p)) for p in PERCENTILES},
        "worst_matrix_index": worst,
        "worst_matrix_rel_l2": float(per_mat[worst]),
        "matrices": int(mat.shape[0]),
    }


def _analyse_one(jax, jnp, pt, a_np: np.ndarray, seed: int) -> dict:
    """a_np: (..., C, C) float32, exactly as production produced it."""
    a32 = jnp.asarray(a_np)
    eye32 = jnp.eye(a_np.shape[-1], dtype=a32.dtype)

    def blocked_fn(m):
        return pt._blocked_unit_lower_inverse(m, TRI_BLOCK)

    def trsm_fn(m):
        # Verbatim the production `else` branch.
        lhs = eye32 - m
        return jax.scipy.linalg.solve_triangular(
            lhs, jnp.broadcast_to(eye32, lhs.shape), lower=True
        )

    blocked = np.asarray(jax.device_get(blocked_fn(a32)))
    trsm = np.asarray(jax.device_get(trsm_fn(a32)))

    a64 = np.asarray(a_np, dtype=np.float64)
    eye64 = np.eye(a_np.shape[-1], dtype=np.float64)
    m64 = eye64 - a64
    oracle = np.linalg.inv(m64)

    def residual(inv):
        r = np.matmul(m64, np.asarray(inv, dtype=np.float64)) - eye64
        return float(np.abs(r).max())

    result = {
        "shape": list(a_np.shape),
        "dtype_in": str(a_np.dtype),
        "forward": {
            "blocked_vs_trsm": _pair("blocked", blocked, "trsm", trsm),
            "blocked_vs_oracle64": _pair("oracle64", oracle, "blocked", blocked),
            "trsm_vs_oracle64": _pair("oracle64", oracle, "trsm", trsm),
        },
        "residual_max_abs": {
            "blocked": residual(blocked), "trsm": residual(trsm),
            "oracle64": residual(oracle),
        },
        "conditioning": {
            "max_abs_a_pre": float(np.abs(a64).max()),
            "mean_abs_a_pre": float(np.abs(a64).mean()),
            "max_abs_inverse_oracle": float(np.abs(oracle).max()),
            "max_row_sum_inverse_oracle": float(np.abs(oracle).sum(axis=-1).max()),
        },
    }

    rng = np.random.default_rng(9000 + seed)
    cot_np = rng.standard_normal(a_np.shape).astype(np.float32)
    cot32 = jnp.asarray(cot_np)

    def scalar(fn):
        return lambda m: jnp.sum(fn(m) * cot32)

    g_blocked = np.asarray(jax.device_get(jax.grad(scalar(blocked_fn))(a32)))
    g_trsm = np.asarray(jax.device_get(jax.grad(scalar(trsm_fn))(a32)))
    # float64 analytic reference: for X = (I-A)^-1 and L = <Cot, X>, dL/dA = X^T Cot X^T.
    xt = np.swapaxes(oracle, -1, -2)
    g_oracle = np.matmul(np.matmul(xt, cot_np.astype(np.float64)), xt)
    # Both algorithms only read the strictly lower part, so compare there as well as on
    # the full matrix; a difference confined to the ignored triangle is structural.
    low = np.tril(np.ones((a_np.shape[-1], a_np.shape[-1])), -1).astype(bool)
    result["vjp"] = {
        "blocked_vs_trsm_full": _pair("g_blocked", g_blocked, "g_trsm", g_trsm),
        "blocked_vs_trsm_strict_lower": _pair(
            "g_blocked", g_blocked * low, "g_trsm", g_trsm * low),
        "blocked_vs_oracle64_strict_lower": _pair(
            "g_oracle64", g_oracle * low, "g_blocked", g_blocked * low),
        "trsm_vs_oracle64_strict_lower": _pair(
            "g_oracle64", g_oracle * low, "g_trsm", g_trsm * low),
        "upper_triangle_max_abs": {
            "blocked": float(np.abs(g_blocked * ~low).max()),
            "trsm": float(np.abs(g_trsm * ~low).max()),
        },
    }
    return result


# -------------------------------------------------------------------------- parent

def _run(args, arm: str, tree: Path, mode: str, out: Path, seed: int,
         triblock: str, apre: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree)
    env["MFU_TRIBLOCK"] = triblock
    for name, value in (("MFU_TRIINV", "0"), ("MFU_BWD_PALLAS", "0"),
                        ("MFU_CASCADE", "0"), ("MFU_CONV_BTC", "0"),
                        ("MFU_MU_DTYPE", "fp32"), ("MFU_FAST_CE", "0"),
                        ("MFU_REMAT", "none"), ("MFU_CHUNK", str(CHUNK)),
                        ("MFU_TRI_BLOCK", str(TRI_BLOCK)), ("MFU_PALLAS_WARPS", "4"),
                        ("MFU_PALLAS_STAGES", "2"), ("MFU_TRIINV_DEBUG", "0"),
                        ("WANDB_MODE", "disabled"), ("OMEGALAX_STARTUP_LOG", "0")):
        env[name] = value
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"), str(Path(__file__).resolve()),
        "--child", "--mode", mode, "--arm", arm, "--tree", str(tree),
        "--out", str(out), "--seed", str(seed),
    ]
    if apre is not None:
        cmd += ["--apre", str(apre)]
    print(f"\n=== {mode} arm={arm} tree={tree.name} seed={seed} "
          f"MFU_TRIBLOCK={triblock}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def _downstream(left: Path, right: Path, label: str) -> dict:
    """Compare the two _state_pass_pallas seams and the forward output."""
    with np.load(left, allow_pickle=False) as ld, np.load(right, allow_pickle=False) as rd:
        lmeta, rmeta = (json.loads(str(d["__meta__"])) for d in (ld, rd))
        groups = {
            "k_dec (independent of inverse)": "k_dec/",
            "df (independent of inverse)": "df/",
            "u_pre (= A_inv @ vb)": "u_pre/",
            "kcd (= A_inv @ kb_decayed)": "kcd/",
            "hidden (forward output)": "hidden",
            "final_state": "final_state/",
        }
        result = {"left": lmeta["source"], "right": rmeta["source"],
                  "left_arm": lmeta["arm"], "right_arm": rmeta["arm"], "groups": {}}
        for name, prefix in groups.items():
            keys = sorted(k for k in ld.files if k.startswith(prefix) and k != "__meta__")
            if not keys or keys != sorted(k for k in rd.files if k.startswith(prefix)
                                          and k != "__meta__"):
                continue
            a = np.concatenate([np.asarray(ld[k], np.float64).ravel() for k in keys])
            b = np.concatenate([np.asarray(rd[k], np.float64).ravel() for k in keys])
            d = np.abs(a - b)
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            result["groups"][name] = {
                "arrays": len(keys),
                "exact": bool(np.array_equal(a, b)),
                "max_abs": float(d.max()),
                "rel_l2": float(np.linalg.norm(d) / max(na, 1e-300)),
                "cosine": float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 1.0,
                "abs_percentiles": {
                    f"p{p:g}": float(np.percentile(d, p)) for p in PERCENTILES
                },
            }
    print(f"\n--- downstream {label}: {result['left_arm']} vs {result['right_arm']}")
    for name, g in result["groups"].items():
        print(f"  {name:<34} n={g['arrays']:<2} exact={str(g['exact']):<5} "
              f"max_abs={g['max_abs']:.3e} rel_l2={g['rel_l2']:.3e} "
              f"1-cos={1.0 - g['cosine']:.3e}")
    return result


def _parent(args: argparse.Namespace) -> int:
    if not ORIGINAL.is_dir():
        raise RuntimeError(f"Original tree not found: {ORIGINAL}")
    out_dir = Path(args.results).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    report: dict[str, object] = {
        "roots": {"original": str(ORIGINAL), "copy": str(ROOT)},
        "config": {"chunk": CHUNK, "tri_block": TRI_BLOCK, "chains": CHAINS,
                   "seq_len": SEQ_LEN, "seeds": seeds,
                   "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                   "kernel": "pallas", "attn_backend": "cudnn", "dtype": "bf16",
                   "xla_flags": os.environ.get("XLA_FLAGS", "")},
        "seeds": {},
    }
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="tmparr_", dir=HERE) as temp_dir:
        temp = Path(temp_dir)
        for seed in seeds:
            entry: dict[str, object] = {}
            paths = {}
            for arm, tree, triblock in (("orig", ORIGINAL, "0"),
                                        ("opt_off", ROOT, "0"),
                                        ("opt_on", ROOT, "1")):
                path = temp / f"s{seed}_{arm}.npz"
                _run(args, arm, tree, "capture", path, seed, triblock)
                paths[arm] = path
            metas = {}
            for arm, path in paths.items():
                with np.load(path, allow_pickle=False) as d:
                    metas[arm] = json.loads(str(d["__meta__"]))
            entry["capture_meta"] = metas
            n_gdn = metas["opt_on"]["dispatch"]["state_pass"]
            for arm, want in (("orig", 0), ("opt_off", 0), ("opt_on", n_gdn)):
                got = metas[arm]["dispatch"]["blocked_inverse"]
                if got != want:
                    problems.append(
                        f"seed{seed} {arm}: blocked_inverse calls {got} != {want}")
            if metas["orig"]["has_blocked_symbol"]:
                problems.append(f"seed{seed}: original tree unexpectedly has the blocked symbol")
            if not metas["opt_on"]["switches"]["pallas_triton._TRI_BLOCKED"]:
                problems.append(f"seed{seed}: opt_on did not have _TRI_BLOCKED set")
            entry["downstream_trees_agree"] = _downstream(
                paths["orig"], paths["opt_off"], f"seed{seed} trees agree")
            entry["downstream_triblock"] = _downstream(
                paths["orig"], paths["opt_on"], f"seed{seed} triblock effect")
            analysis = temp / f"s{seed}_analysis.json"
            _run(args, "analyse", ROOT, "analyse", analysis, seed, "1", apre=paths["opt_on"])
            entry["inverse_analysis"] = json.loads(analysis.read_text())
            report["seeds"][f"seed{seed}"] = entry

    report["dispatch_problems"] = problems
    target = out_dir / f"gate3_summary_{args.tag}.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nsummary -> {target}")

    print("\n=== inverse accuracy on production A_pre (relative L2)")
    print("seed  layer          blk_vs_trsm   blk_vs_or64   trsm_vs_or64   "
          "res_blk    res_trsm   res_or64   max|A_pre|")
    for skey, entry in report["seeds"].items():
        for lkey, la in entry["inverse_analysis"]["per_layer"].items():
            f, r, c = la["forward"], la["residual_max_abs"], la["conditioning"]
            print(f"{skey:<6}{lkey:<15}{f['blocked_vs_trsm']['rel_l2']:.3e}     "
                  f"{f['blocked_vs_oracle64']['rel_l2']:.3e}     "
                  f"{f['trsm_vs_oracle64']['rel_l2']:.3e}      "
                  f"{r['blocked']:.2e}   {r['trsm']:.2e}   {r['oracle64']:.2e}   "
                  f"{c['max_abs_a_pre']:.3f}")
    print("\n=== VJP wrt A_pre, strictly lower part (relative L2)")
    for skey, entry in report["seeds"].items():
        for lkey, la in entry["inverse_analysis"]["per_layer"].items():
            v = la["vjp"]
            print(f"{skey:<6}{lkey:<15}"
                  f"blk_vs_trsm={v['blocked_vs_trsm_strict_lower']['rel_l2']:.3e}  "
                  f"blk_vs_or64={v['blocked_vs_oracle64_strict_lower']['rel_l2']:.3e}  "
                  f"trsm_vs_or64={v['trsm_vs_oracle64_strict_lower']['rel_l2']:.3e}  "
                  f"upper(blk/trsm)={v['upper_triangle_max_abs']['blocked']:.1e}/"
                  f"{v['upper_triangle_max_abs']['trsm']:.1e}")
    print("\n=== dispatch")
    if problems:
        for p in problems:
            print("FAIL ", p)
    else:
        print("ok    every arm showed the expected blocked-inverse dispatch")
    print("\nNo checkpoints written; temporary arrays removed with the scope.")
    return 1 if problems else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=("capture", "analyse"), default="capture")
    p.add_argument("--arm", default="opt_on", help=argparse.SUPPRESS)
    p.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--apre", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tag", default="g3")
    p.add_argument(
        "--results",
        default="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt/gate3",
    )
    args = p.parse_args()
    if args.child:
        if args.mode == "capture":
            _child_capture(args)
        else:
            _child_analyse(args)
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
