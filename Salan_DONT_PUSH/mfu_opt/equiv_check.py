"""Compare two equiv_run.py dumps and decide PASS / FAIL.

Usage:
    equiv_check.py orig.json opt.json [--tol 1e-6]

Metrics that are *supposed* to differ between the two codebases are listed in
EXPECTED_DIFF: the FLOP counter was corrected in the optimized copy, so anything
derived from it must change. Everything else — the loss, the gradient norm, the
final parameters — must agree, because a performance change may not alter the
mathematics.
"""

from __future__ import annotations

import argparse
import json
import sys

# Derived from training_flops_per_token, which was deliberately corrected.
EXPECTED_DIFF = {"mfu", "tflops_per_device", "train/mfu", "train/tflops_per_device"}
# Wall-clock and anything derived from it is not reproducible run to run.
IGNORED = {
    "step_time_s", "train/step_time_s", "global_tokens_per_sec",
    "train/global_tokens_per_sec", "tokens_per_sec_per_device",
    "train/tok/s/dev", "data_wait_frac", "train/data_wait_frac",
    "data_wait_s", "train/data_wait_s", "elapsed_s", "train/elapsed_s",
    # Hardware telemetry sampled during the run, not a result of the computation.
    # It differed between two runs of identical code and produced false FAILs.
    "gpu_util_avg", "gpu_util_max", "gpu_util_min",
    "train/gpu_util_avg", "train/gpu_util_max", "train/gpu_util_min",
}


def rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("orig")
    ap.add_argument("opt")
    ap.add_argument("--tol", type=float, default=1e-6)
    # Comparing a tree against itself is normally a PYTHONPATH mistake, but it is
    # also the only way to measure the run-to-run noise floor: XLA selects GEMM
    # algorithms by timing them, so two processes running identical code can
    # differ. Without that floor the deviations below are uninterpretable.
    ap.add_argument("--same-ok", action="store_true",
                    help="gleiche Codebase erlauben (Rauschpegel-Messung)")
    args = ap.parse_args()

    o = json.load(open(args.orig))
    n = json.load(open(args.opt))

    print(f"orig : {o['source']}")
    print(f"opt  : {n['source']}")
    if o["source"] == n["source"] and not args.same_ok:
        print("\nFEHLER: beide Dumps stammen aus derselben Codebase (PYTHONPATH falsch).")
        return 2
    print()

    failures: list[str] = []
    expected: list[str] = []

    # --- Metriken ---
    keys = sorted(set(o["metrics"]) | set(n["metrics"]))
    print(f"{'Metrik':<34} {'orig':>14} {'opt':>14} {'rel':>10}")
    print("-" * 76)
    for k in keys:
        if k in IGNORED:
            continue
        a, b = o["metrics"].get(k), n["metrics"].get(k)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        r = rel(a, b)
        if r <= args.tol:
            continue
        line = f"{k:<34} {a:>14.6g} {b:>14.6g} {r:>10.2e}"
        if k in EXPECTED_DIFF:
            expected.append(line)
        else:
            failures.append(line)
            print(line + "   <-- UNERWARTET")
    if not failures:
        print("(keine unerwarteten Metrik-Abweichungen)")

    if expected:
        print("\nErwartete Abweichungen (korrigierter FLOP-Zaehler):")
        for line in expected:
            print("  " + line)

    # --- Parameter ---
    pk = sorted(set(o["params"]) | set(n["params"]))
    missing = [k for k in pk if k not in o["params"] or k not in n["params"]]
    worst, worst_key = 0.0, ""
    for k in pk:
        if k in missing:
            continue
        for i, name in enumerate(("sum", "sumsq", "absmax")):
            r = rel(o["params"][k][i], n["params"][k][i])
            if r > worst:
                worst, worst_key = r, f"{k}[{name}]"
    print(f"\nParameter: {len(pk)} Tensoren, groesste relative Abweichung "
          f"{worst:.3e}" + (f"  ({worst_key})" if worst_key else ""))
    if missing:
        print(f"FEHLER: {len(missing)} Tensoren nur in einer Seite: {missing[:5]}")
        failures.append("param-set mismatch")
    if worst > args.tol:
        failures.append(f"params rel={worst:.3e} > tol")

    print()
    if failures:
        print(f"FAIL — {len(failures)} Abweichung(en) ausserhalb der Toleranz {args.tol:g}")
        return 1
    print(f"PASS — Mathematik identisch innerhalb {args.tol:g}; "
          f"nur der korrigierte FLOP-Zaehler weicht ab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
