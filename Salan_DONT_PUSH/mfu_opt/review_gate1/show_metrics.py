"""One-off: per-metric deviation from a gate1 summary, worst relative first."""

import json
import sys

path, arm = sys.argv[1], sys.argv[2]
summary = json.loads(open(path).read())
key = next(k for k in summary["runs"] if k.endswith("/" + arm))
stats = summary["runs"][key]["vs_orig_a"]["step1/metric"]
print(f"{key}  metric rel_l2={stats['rel_l2']:.3e} max_abs={stats['max_abs']:.3e}")
rows = sorted(stats["per_tensor"].items(), key=lambda kv: -kv[1]["rel_l2"])
print(f"{'metric':<34}{'rel_l2':>11}{'max_abs':>12}{'ref_scale':>13}")
for name, entry in rows:
    print(f"{name:<34}{entry['rel_l2']:>11.3e}{entry['max_abs']:>12.3e}{entry['rms_ref']:>13.6g}")
