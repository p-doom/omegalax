"""Rank non-GEMM kernels by how far they fall short of memory bandwidth.

    analyze_bandwidth.py <trace_dir> <hlo_dump_dir>

This is the method that found the conv transpose (E16/E20): that kernel was not
merely large, it moved ~1.2 GiB in 10.0 ms, roughly 19x slower than the same
traffic at the measured 2300 GB/s. Time alone would not have flagged it -- it was
the ratio of time to bytes that made it a defect rather than just work.

Applying that per kernel instead of by eye: join the trace's per-kernel time with
the operand and output shapes of the matching HLO fusion, and report achieved
bandwidth. Anything well under the roofline is doing something strided or
redundant; anything near it is simply moving data and can only be improved by
moving less.

The byte estimate is a lower bound. It counts each fusion's operands once plus
its outputs, which ignores re-reads inside the fusion and any operand still in
cache. A kernel that looks slow here may just be re-reading an operand, so treat
the ranking as a shortlist to inspect, not a verdict.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys

from analyze_trace import RULES, load_events

PEAK_GBS = 2300.0  # measured copy bandwidth on this H100, see bench_transpose.py

DTYPE_BYTES = {"pred": 1, "s8": 1, "u8": 1, "bf16": 2, "f16": 2, "s16": 2, "u16": 2,
               "f32": 4, "s32": 4, "u32": 4, "f64": 8, "s64": 8, "u64": 8, "c64": 8}

SHAPE = re.compile(r"\b([a-z]\d*)\[([\d,]*)\]")
# Any top-level instruction: "%name = <shape> <opcode>(<operands>)". An earlier
# version only indexed fusion/copy/transpose, so operands that were plain
# instructions or get-tuple-element resolved to zero bytes and their kernels showed
# an absurd 1 GB/s. Index everything.
INSTR = re.compile(r"^\s*%?([\w.\-]+) = (\S+) ([\w\-]+)\((.*)$", re.M)


def shape_bytes(text: str) -> int:
    """Sum the byte size of every shape literal in a fragment of HLO."""
    total = 0
    for dtype, dims in SHAPE.findall(text):
        if dtype not in DTYPE_BYTES:
            continue
        n = 1
        for d in dims.split(","):
            if d:
                n *= int(d)
        total += n * DTYPE_BYTES[dtype]
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("hlo")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--min-ms", type=float, default=3.0,
                    help="Kernels unter dieser ms/Step ignorieren")
    args = ap.parse_args()

    cands = glob.glob(os.path.join(args.hlo, "*pretrain_train_step*after_optimizations.txt"))
    if not cands:
        sys.exit(f"kein optimiertes HLO-Modul unter {args.hlo}")
    hlo = open(max(cands, key=os.path.getsize), errors="replace").read()

    # name -> (output shape text, operand text). Names in the trace use "_N" where
    # HLO uses ".N", so index both spellings.
    sizes: dict[str, tuple[str, str]] = {}
    for name, out_shape, _opcode, operands in INSTR.findall(hlo):
        sizes[name] = (out_shape, operands)
        sizes[name.replace(".", "_")] = (out_shape, operands)

    events = load_events(args.trace)
    gpu = {(e.get("pid"), e.get("tid")) for e in events
           if e.get("ph") == "M" and e.get("name") == "thread_name"
           and "stream" in str(e.get("args", {}).get("name", "")).lower()}

    def classify(n: str) -> str:
        for label, pat in RULES:
            if pat.search(n):
                return label
        return "?"

    per: collections.Counter = collections.Counter()
    cnt: collections.Counter = collections.Counter()
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if (e.get("pid"), e.get("tid")) not in gpu:
            continue
        per[e["name"]] += e["dur"]
        cnt[e["name"]] += 1

    rows = []
    unknown_ms = 0.0
    for name, dur in per.items():
        cat = classify(name)
        if cat == "GEMM / MLP":
            continue
        ms_step = dur / 1e3 / args.steps
        if ms_step < args.min_ms:
            continue
        hit = sizes.get(name)
        if not hit:
            unknown_ms += ms_step
            continue
        out_txt, op_txt = hit
        # Operands appear as %name references; resolve each to its own output shape.
        ops = 0
        for ref in re.findall(r"%([\w.\-]+)", op_txt):
            if ref in sizes:
                ops += shape_bytes(sizes[ref][0])
        gb = (shape_bytes(out_txt) + ops) / 1e9
        us_call = dur / cnt[name]
        gbs = gb * args.steps / (dur / 1e6) if dur else 0.0
        rows.append((gbs, ms_step, cnt[name] / args.steps, us_call, gb, cat, name))

    rows.sort()
    print(f"\nRoofline {PEAK_GBS:.0f} GB/s. Bytes sind eine UNTERGRENZE "
          f"(Operanden einfach gezaehlt), also ist die Bandbreite eine Untergrenze.\n")
    print(f"{'GB/s':>7} {'%roof':>6} {'ms/Step':>8} {'Aufr/St':>8} {'us/Aufr':>8} "
          f"{'GB/Aufr':>8}  {'Kategorie':<20} Kernel")
    print("-" * 116)
    for gbs, ms, calls, us, gb, cat, name in rows:
        print(f"{gbs:>7.0f} {gbs/PEAK_GBS*100:>5.0f}% {ms:>8.1f} {calls:>8.0f} {us:>8.1f} "
              f"{gb/max(calls,1)*args.steps/args.steps:>8.3f}  {cat[:20]:<20} {name[:40]}")
    print("-" * 116)
    print(f"nicht zuordenbar: {unknown_ms:.1f} ms/Step "
          f"(Kernelname ohne passende HLO-Instruktion)")


if __name__ == "__main__":
    main()
