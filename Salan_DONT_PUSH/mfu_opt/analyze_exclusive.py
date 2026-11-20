"""Decompose GPU time by component using EXCLUSIVE timeline occupancy.

    analyze_exclusive.py <trace_dir_or_file>

Why this exists, and why analyze_trace.py's numbers were wrong for one bucket:
that script sums per-event ``dur``. For compute kernels on a serialised stream a
sum is fine, because they do not overlap. For asynchronous copies it is badly
wrong -- a trace of this pipeline contains 54160 MemcpyD2D events whose durations
sum to 2574 ms but which together occupy only 63 ms of the timeline, because they
are issued concurrently (``async:1``) and their intervals nest. Summing counted
the same wall-clock 40x over and inflated the Reshape/Copy bucket from ~1 % to
23 %, which pointed the whole bottleneck analysis at a non-problem.

Method: build the merged interval union per category, then subtract the union of
all other categories. What remains is time where only this category was running --
the time that would actually be recovered by removing it. Overlap is reported
separately instead of being silently assigned to one side.
"""

from __future__ import annotations

import argparse
import collections

from analyze_trace import RULES, load_events


def classify(name: str) -> str:
    for label, pat in RULES:
        if pat.search(name):
            return label
    return "unklassifiziert"


def merge(spans: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(spans) -> float:
    return sum(b - a for a, b in spans)


def subtract(a_spans, b_spans) -> list[list[float]]:
    """a_spans minus b_spans; both must already be merged and sorted."""
    res: list[list[float]] = []
    j = 0
    for a, b in a_spans:
        cur = a
        while j < len(b_spans) and b_spans[j][1] <= a:
            j += 1
        k = j
        while k < len(b_spans) and b_spans[k][0] < b:
            if b_spans[k][0] > cur:
                res.append([cur, min(b_spans[k][0], b)])
            cur = max(cur, b_spans[k][1])
            k += 1
        if cur < b:
            res.append([cur, b])
    return [s for s in res if s[1] > s[0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--steps", type=float, default=0,
                    help="Steps im Trace; 0 = automatisch aus der Zahl der "
                         "State-Pass-Kernel ableiten")
    args = ap.parse_args()

    events = load_events(args.path)

    gpu_tids: set[tuple] = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            nm = str(e.get("args", {}).get("name", ""))
            if "stream" in nm.lower():
                gpu_tids.add((e.get("pid"), e.get("tid")))

    by_cat: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    summed: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    by_name: collections.Counter = collections.Counter()
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if gpu_tids and (e.get("pid"), e.get("tid")) not in gpu_tids:
            continue
        cat = classify(e.get("name", "?"))
        ts, dur = float(e["ts"]), float(e["dur"])
        by_cat[cat].append((ts, ts + dur))
        summed[cat] += dur
        counts[cat] += 1
        by_name[e.get("name", "?")] += 1

    # Derive the step count instead of trusting a hand-passed one. The profiler
    # window does not always contain a whole number of steps: one trace here held
    # 2.78 steps while another held exactly 3, and dividing both by 3 understated
    # the first by 8 % and inverted a cross-trace comparison. _state_pass_kernel
    # fires once per GDN layer per forward, so its count is a reliable ruler as
    # long as both traces run the same model.
    if not args.steps:
        marker = sum(c for n, c in by_name.items() if "_state_pass" in n)
        per_step = 36.0  # 18 GDN layers x (forward + rematerialised forward)
        args.steps = marker / per_step if marker else 1.0
        print(f"Steps automatisch bestimmt: {args.steps:.2f} "
              f"({marker} State-Pass-Kernel / {per_step:.0f} pro Step)")

    merged = {c: merge(v) for c, v in by_cat.items()}
    all_spans = merge([s for v in by_cat.values() for s in v])
    span = all_spans[-1][1] - all_spans[0][0]
    busy = total(all_spans)

    rows = []
    for cat, m in merged.items():
        others = merge([s for c, v in by_cat.items() if c != cat for s in v])
        excl = total(subtract(m, others))
        rows.append((excl, total(m), summed[cat], counts[cat], cat))
    rows.sort(reverse=True)

    n = max(1, args.steps)
    print(f"\nTrace-Spanne {span/1e3:.1f} ms, GPU busy {busy/1e3:.1f} ms "
          f"({busy/span*100:.1f} %), {n} Steps\n")
    print(f"{'exklusiv ms':>12} {'%busy':>7} {'ms/Step':>9} | "
          f"{'belegt ms':>10} {'summiert ms':>12} {'Aufrufe':>9}  Kategorie")
    print("-" * 96)
    for excl, occ, s, c, cat in rows:
        print(f"{excl/1e3:>12.1f} {excl/busy*100:>6.1f}% {excl/1e3/n:>9.1f} | "
              f"{occ/1e3:>10.1f} {s/1e3:>12.1f} {c:>9}  {cat}")
    tot_excl = sum(r[0] for r in rows)
    print("-" * 96)
    print(f"{tot_excl/1e3:>12.1f} {tot_excl/busy*100:>6.1f}% {tot_excl/1e3/n:>9.1f} | "
          f"{'':>10} {sum(r[2] for r in rows)/1e3:>12.1f} {sum(r[3] for r in rows):>9}  Summe exklusiv")
    print(f"{(busy-tot_excl)/1e3:>12.1f} {(busy-tot_excl)/busy*100:>6.1f}% "
          f"{(busy-tot_excl)/1e3/n:>9.1f} |  ueberlappend (mehrere Kategorien gleichzeitig)")


if __name__ == "__main__":
    main()
