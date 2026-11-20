"""One-off: per-tensor gradient deviation from a gate1 summary, worst first."""

import json
import sys

path, arm = sys.argv[1], sys.argv[2]
summary = json.loads(open(path).read())
key = next(k for k in summary["runs"] if k.endswith("/" + arm))
grads = summary["runs"][key]["vs_orig_a"]["step1/grad"]
print(f"{path}\n{key}  tensors={grads['tensors']} rel_l2={grads['rel_l2']:.3e}")
rows = sorted(grads["per_tensor"].items(), key=lambda kv: -kv[1]["rel_l2"])
print(f"{'tensor':<58}{'rel_l2':>11}{'max_abs':>12}{'rms_ref':>12}{'cosine':>16}")
for name, stats in rows[:14]:
    print(
        f"{name:<58}{stats['rel_l2']:>11.3e}{stats['max_abs']:>12.3e}"
        f"{stats['rms_ref']:>12.3e}{stats['cosine']:>16.10f}"
    )
exact = sum(1 for _, s in rows if s["bitwise_equal"])
print(f"\nbitwise-equal tensors: {exact}/{len(rows)}")
