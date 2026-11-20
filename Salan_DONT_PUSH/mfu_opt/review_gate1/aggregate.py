"""Consolidate the gate1 seed jobs: worst case per arm and family over all runs.

Reads the per-seed summary JSONs and reports, for each arm and tensor family, the
worst value seen across every seed and batch shape. ``orig_b`` is printed first and
is the noise floor -- every other row is only meaningful relative to it.
"""

import json
import sys

FAMILIES = ("hidden", "state_final", "state_all", "grad", "metric", "param")
ARM_ORDER = [
    "orig_b", "opt_off", "opt_conv_btc", "opt_triblock", "opt_cascade",
    "opt_cascade_conv_btc", "opt_cascade_triblock", "opt_conv_btc_triblock",
    "opt_cascade_conv_btc_triblock",
]

worst: dict[tuple[str, str, int], dict] = {}
statuses: dict[tuple[str, str], set] = {}
dispatch_problems: list[str] = []
covered: set[tuple[str, int]] = set()
steps = 0

for path in sys.argv[1:]:
    summary = json.loads(open(path).read())
    steps = max(steps, int(summary["config"]["steps"]))
    for key, run in summary["runs"].items():
        shape, seed_key, arm = key.split("/")
        covered.add((shape, int(seed_key.removeprefix("seed"))))
        if run["dispatch_problems"]:
            dispatch_problems.append(f"{path.split('/')[-1]} {key}: {run['dispatch_problems']}")
        for name, stats in (run.get("vs_orig_a") or {}).items():
            step_str, family = name.split("/")
            step = int(step_str.removeprefix("step"))
            statuses.setdefault((arm, family), set()).add(stats.get("status", "?"))
            if stats.get("status") != "compared":
                continue
            slot = worst.setdefault((arm, family, step), {"n": 0})
            slot["n"] += 1
            for field in ("max_abs", "rel_l2"):
                slot[field] = max(slot.get(field, 0.0), stats[field])
            slot["cosine"] = min(slot.get("cosine", 1.0), stats["cosine"])
            slot["worst_rel_l2"] = max(
                slot.get("worst_rel_l2", 0.0), stats["worst_tensor_rel_l2"]["rel_l2"]
            )
            for pct in ("p50", "p90", "p99", "p99.9"):
                slot[f"abs_{pct}"] = max(
                    slot.get(f"abs_{pct}", 0.0), stats["abs_percentiles"][pct]
                )
                slot[f"rms_{pct}"] = max(
                    slot.get(f"rms_{pct}", 0.0), stats["rms_normalized_percentiles"][pct]
                )
            if not stats["finite"]:
                slot["nonfinite"] = True
            slot["all_bitwise_equal"] = slot.get("all_bitwise_equal", True) and stats[
                "bitwise_equal"
            ]

print(f"covered (shape, seed): {sorted(covered)}   runs per cell: see n")
print(f"dispatch problems: {dispatch_problems if dispatch_problems else 'none'}\n")
for step in range(1, steps + 1):
    print(f"===== step {step}: worst over all seeds and shapes, vs orig_a")
    print(
        f"{'arm':<32}{'family':<12}{'n':>3}{'max_abs':>12}{'rel_l2':>12}"
        f"{'worstT_l2':>12}{'1-cosine':>12}{'p50':>11}{'p99':>11}{'p99.9':>11}{'bit=':>6}"
    )
    for arm in ARM_ORDER:
        for family in FAMILIES:
            slot = worst.get((arm, family, step))
            if slot is None:
                marks = statuses.get((arm, family))
                if marks:
                    print(f"{arm:<32}{family:<12}{'':>3}  {sorted(marks)}")
                continue
            print(
                f"{arm:<32}{family:<12}{slot['n']:>3}{slot['max_abs']:>12.3e}"
                f"{slot['rel_l2']:>12.3e}{slot['worst_rel_l2']:>12.3e}"
                f"{1.0 - slot['cosine']:>12.3e}{slot['abs_p50']:>11.3e}"
                f"{slot['abs_p99']:>11.3e}{slot['abs_p99.9']:>11.3e}"
                f"{str(slot['all_bitwise_equal']):>6}"
                + ("  NONFINITE" if slot.get("nonfinite") else "")
            )
    print()
