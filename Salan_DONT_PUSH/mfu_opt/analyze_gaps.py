"""Split GPU wall time into kernel execution versus idle gaps, and count launches.

    analyze_gaps.py <trace.json.gz>

Answers the question a component breakdown cannot: is the step slow because the
kernels are slow, or because the GPU is idle between them? A trace of the baseline
showed ~2437 ms of summed kernel time against ~2840 ms of wall clock per step,
leaving ~14 % unaccounted. That gap is either launch overhead, host stalls, or
dependencies that serialise work — and the three have completely different fixes.

Method: take every GPU-stream event, sort by start time, and walk them. Overlapping
events are merged into busy intervals, so concurrent kernels are not double counted.
What remains between busy intervals is idle time.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys


def load(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        raw = json.load(fh)
    return raw["traceEvents"] if isinstance(raw, dict) else raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gap-us", type=float, default=5.0,
                    help="Luecken ab dieser Laenge einzeln ausweisen")
    args = ap.parse_args()

    events = load(args.path)

    gpu_tids: set[tuple] = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            if re.search(r"gpu|stream", str(e.get("args", {}).get("name", "")), re.I):
                gpu_tids.add((e.get("pid"), e.get("tid")))

    spans: list[tuple[float, float, str]] = []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if gpu_tids and (e.get("pid"), e.get("tid")) not in gpu_tids:
            continue
        ts = float(e["ts"])
        spans.append((ts, ts + float(e["dur"]), e.get("name", "?")))

    if not spans:
        sys.exit("keine GPU-Events gefunden")

    spans.sort()
    kernel_time = sum(b - a for a, b, _ in spans)

    # Merge overlaps so concurrent kernels are not counted twice.
    merged: list[list[float]] = []
    for a, b, _ in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    busy = sum(b - a for a, b in merged)
    span_total = merged[-1][1] - merged[0][0]
    idle = span_total - busy

    gaps: list[tuple[float, str, str]] = []
    for i in range(1, len(merged)):
        g = merged[i][0] - merged[i - 1][1]
        if g > 0:
            gaps.append((g, "", ""))
    gaps.sort(reverse=True)
    big = [g for g, _, _ in gaps if g >= args.gap_us]

    print(f"Trace-Spanne          {span_total / 1e3:>10.1f} ms")
    print(f"GPU busy (gemergt)    {busy / 1e3:>10.1f} ms   {busy / span_total * 100:>5.1f} %")
    print(f"GPU idle (Luecken)    {idle / 1e3:>10.1f} ms   {idle / span_total * 100:>5.1f} %")
    print(f"Kernelzeit (Summe)    {kernel_time / 1e3:>10.1f} ms   "
          f"Ueberlappung {(kernel_time - busy) / 1e3:.1f} ms")
    print()
    print(f"Kernel-Starts gesamt  {len(spans):>10}")
    print(f"Luecken gesamt        {len(gaps):>10}")
    print(f"davon >= {args.gap_us:.0f} us       {len(big):>10}   "
          f"Summe {sum(big) / 1e3:.1f} ms ({sum(big) / span_total * 100:.1f} %)")
    if gaps:
        print(f"groesste Luecke       {gaps[0][0]:>10.1f} us")
        print(f"Median-Luecke         {sorted(g for g, _, _ in gaps)[len(gaps) // 2]:>10.2f} us")
        print(f"mittlere Luecke       {idle / max(1, len(gaps)):>10.2f} us")

    per = collections.Counter()
    cnt = collections.Counter()
    for a, b, n in spans:
        per[n] += b - a
        cnt[n] += 1
    print(f"\n{'ms':>9} {'Aufrufe':>8} {'us/Aufruf':>10}  Kernel")
    print("-" * 74)
    for n, d in per.most_common(12):
        print(f"{d / 1e3:>9.1f} {cnt[n]:>8} {d / cnt[n]:>10.1f}  {n[:44]}")


if __name__ == "__main__":
    main()
