"""Group GPU kernel time from a JAX perfetto trace into pipeline components.

    analyze_trace.py <trace_dir_or_file> [--top 25]

The trace written by ``jax.profiler.trace(..., create_perfetto_trace=True)`` is a
gzipped Chrome Trace Event JSON. Only events on GPU streams are counted, so the
result is device time, not host time.

Classification is by kernel name. It is a heuristic and deliberately reports an
"unclassified" bucket rather than forcing everything into a category — if that
bucket is large the rules below need extending, and pretending otherwise would
hide exactly the thing we are looking for.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

# Order matters: first match wins.
# Order matters: first match wins. Names verified against a real trace of this
# pipeline (H100, 2B, C=6) — do not reorder without re-checking the unclassified
# bucket, which is the honesty check on these rules.
RULES: list[tuple[str, re.Pattern]] = [
    # Pallas/Triton GDN kernels plus the cuBLAS batched triangular solve that the
    # delta rule's chunk inverse lowers to.
    ("GDN / delta rule",
     re.compile(r"triton|pallas|_state_pass|delta|chunk_gated|batch_trsm|trsm", re.I)),
    ("Full attention",
     re.compile(r"mosaic_gpu_kernel|attention|flash|fmha", re.I)),
    # The big vocab-sized multiply+reduce fusions of the chunked cross-entropy.
    # Only names that actually say what they are. An earlier version of these
    # rules mapped "input_multiply_reduce_fusion" here because a bandwidth
    # estimate suggested the loss -- the rule was built from the hypothesis and
    # then its output was read as evidence for that hypothesis. The cascade
    # experiment disproved it: that kernel fires ~18x per step (once per
    # linear-attention layer) and vanishes when the segment schedule changes,
    # neither of which is true of the loss. Generic XLA fusion names now go to
    # their own bucket instead of being guessed at.
    ("Loss / LM head",
     re.compile(r"cross_entropy|xent|log_softmax|logit|lm_head", re.I)),
    ("Optimizer",
     re.compile(r"adam|optimizer|update_moment|scale_by|clip_by|\bema\b", re.I)),
    ("Collectives",
     re.compile(r"all[-_]?reduce|all[-_]?gather|reduce[-_]?scatter|nccl|collective", re.I)),
    # nvjet_* is NVIDIA's current cuBLAS GEMM family and dominates by count; it
    # matched nothing in the first version of these rules and silently landed in
    # "unclassified", which is what made the first decomposition unreadable.
    ("GEMM / MLP",
     re.compile(r"nvjet|gemm|cutlass|cublas|dot|matmul|ampere|sm90|s16816|wgmma", re.I)),
    ("Reshape / copy",
     re.compile(r"copy|transpose|bitcast|concat|slice|pad|gather|scatter|memcpy|convert", re.I)),
    ("Norm / elementwise",
     re.compile(r"norm|rsqrt|silu|swiglu|gelu|multiply|\bmul\b|add|exp|scale|bias|loop_", re.I)),
    ("Reduction", re.compile(r"reduce|sum|mean|argmax|max_|min_", re.I)),
    # Generic XLA fusions whose name carries no semantics. Reported separately so
    # their size is visible without pretending to know what they compute.
    ("XLA-Fusion (unbenannt)", re.compile(r"^(input|loop|fusion)_", re.I)),
]


def load_events(path: str) -> list[dict]:
    if os.path.isdir(path):
        cands = sorted(
            glob.glob(os.path.join(path, "**", "*perfetto*"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.json.gz"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.trace.json*"), recursive=True)
        )
        if not cands:
            sys.exit(f"keine perfetto/json-Trace unter {path} gefunden;\n"
                     f"vorhanden: {os.listdir(path)[:10]}")
        path = cands[-1]
    print(f"Trace: {path}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        raw = json.load(fh)
    return raw["traceEvents"] if isinstance(raw, dict) else raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    events = load_events(args.path)

    # Identify GPU threads from metadata events.
    gpu_tids: set[tuple] = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") in ("thread_name", "process_name"):
            label = str(e.get("args", {}).get("name", ""))
            if re.search(r"gpu|stream|device|/device:GPU", label, re.I):
                gpu_tids.add((e.get("pid"), e.get("tid")) if e["name"] == "thread_name"
                             else (e.get("pid"), None))

    per_kernel: collections.Counter = collections.Counter()
    total = 0.0
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        key_t = (e.get("pid"), e.get("tid"))
        key_p = (e.get("pid"), None)
        if gpu_tids and key_t not in gpu_tids and key_p not in gpu_tids:
            continue
        per_kernel[e.get("name", "?")] += float(e["dur"])
        total += float(e["dur"])

    if total == 0:
        sys.exit("keine GPU-Events gefunden — Klassifikation der Threads pruefen")

    buckets: collections.Counter = collections.Counter()
    for name, dur in per_kernel.items():
        for label, pat in RULES:
            if pat.search(name):
                buckets[label] += dur
                break
        else:
            buckets["unklassifiziert"] += dur

    print(f"\nGesamte GPU-Kernel-Zeit im Trace: {total/1e3:.1f} ms\n")
    print(f"{'Komponente':<22} {'ms':>10} {'Anteil':>8}")
    print("-" * 42)
    for label, dur in buckets.most_common():
        print(f"{label:<22} {dur/1e3:>10.1f} {dur/total*100:>7.1f}%")

    print(f"\nTop {args.top} Einzelkernels:")
    print(f"{'ms':>9} {'Anteil':>7}  Name")
    print("-" * 78)
    for name, dur in per_kernel.most_common(args.top):
        print(f"{dur/1e3:>9.1f} {dur/total*100:>6.1f}%  {name[:60]}")


if __name__ == "__main__":
    main()
