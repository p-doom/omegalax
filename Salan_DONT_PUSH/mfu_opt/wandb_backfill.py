"""Publish every benchmark run in the log directory as one wandb table.

    wandb_backfill.py [--dry-run]

New benchmarks log to wandb themselves (``BENCH_WANDB=1`` in submit_bench.sh), but
everything measured before that switch existed lives only in Slurm logs. This
reads those logs and the matching sbatch files so the numbers come from what was
actually executed rather than from anyone's memory of it.

Per run it takes the median of the per-step values instead of the last one: the
first steps include compilation and warmup, and a single step is noisy enough that
picking the last one silently rewards whichever run happened to end well.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics

BENCH_ROOT = "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt"

RE_MFU = re.compile(r"train/mfu=([\d.]+)%")
RE_TOK = re.compile(r"train/tok/s/dev=(\d+)")
RE_STEP = re.compile(r"train/step_time_s=([\d.]+)s")
RE_PEAK = re.compile(r"peak=([\d.]+)GB")
RE_HDR = re.compile(
    r"MFU-BENCH tag=(\S+) gpus=(\d+) batch=(\d+) accum=(\d+) steps=(\d+) C=(\d+)"
)
RE_ENV = re.compile(r"^export (MFU_\w+)=(.*)$", re.M)
RE_XLA = re.compile(r"^=== XLA_FLAGS=(.*)$", re.M)
RE_LOSS = re.compile(r"step=\d+ loss=([^\s]+)")


def parse(path: str) -> dict | None:
    with open(path, errors="replace") as fh:
        text = fh.read()
    hdr = RE_HDR.search(text)
    if not hdr:
        return None
    tag, gpus, batch, accum, steps, chains = hdr.groups()
    mfu = [float(x) for x in RE_MFU.findall(text)]
    tok = [int(x) for x in RE_TOK.findall(text)]
    step = [float(x) for x in RE_STEP.findall(text)]
    # Drop step 1: it carries compilation and allocator warmup.
    mfu, tok, step = mfu[1:], tok[1:], step[1:]
    if not mfu:
        return None
    peak = RE_PEAK.findall(text)
    xla = RE_XLA.search(text)
    # Throughput without a loss check is worthless: NaN computes at full speed, and
    # a dozen runs were optimised against tok/s while producing loss=nan from the
    # first step (E49). Every row carries the loss and an explicit validity flag so
    # the table cannot be read the wrong way.
    losses = RE_LOSS.findall(text)
    loss_vals = [x for x in losses if x not in ("nan", "inf", "-inf")]
    loss5 = losses[4] if len(losses) > 4 else (losses[-1] if losses else None)
    has_nan = any(x in ("nan", "inf", "-inf") for x in losses)

    row = {
        "tag": tag,
        "loss_step5": loss5,
        "loss_ok": bool(loss_vals) and not has_nan,
        "job": os.path.basename(path).rsplit("_", 1)[1].split(".")[0],
        "gpus": int(gpus),
        "batch": int(batch),
        "grad_accum": int(accum),
        "steps": int(steps),
        "C": int(chains),
        "mfu_pct": round(statistics.median(mfu), 2),
        "tok_s_dev": int(statistics.median(tok)),
        "step_s": round(statistics.median(step), 4) if step else None,
        "n_steps_used": len(mfu),
        "peak_gb": float(peak[-1]) if peak else None,
        "xla_flags": (xla.group(1).strip() if xla else "") or "<none>",
        "oom": "RESOURCE_EXHAUSTED" in text or "ran out of memory" in text,
    }
    sb = os.path.join(BENCH_ROOT, "jobs", f"bench_{tag}.sbatch")
    if os.path.exists(sb):
        env = dict(RE_ENV.findall(open(sb, errors="replace").read()))
        for key in ("MFU_CASCADE", "MFU_CONV_BTC", "MFU_REMAT", "MFU_MU_DTYPE",
                    "MFU_FAST_CE", "MFU_NUM_LOSS_TILES"):
            row[key.lower()] = env.get(key, "")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [r for r in (parse(p) for p in sorted(glob.glob(f"{BENCH_ROOT}/*.log"))) if r]
    # Invalid runs sort to the bottom so the top of the table is always adoptable.
    rows.sort(key=lambda r: (r["loss_ok"], r["mfu_pct"]), reverse=True)
    cols = list(rows[0].keys())

    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" ".join(c.rjust(width[c]) for c in cols))
    for r in rows:
        print(" ".join(str(r.get(c, "")).rjust(width[c]) for c in cols))
    print(f"\n{len(rows)} Laeufe")

    if args.dry_run:
        return

    import wandb

    run = wandb.init(
        entity="pdoom",
        project="omegalax_mfu",
        group="mfu_opt",
        name="backfill_summary",
        job_type="summary",
        tags=["backfill"],
        config={"note": "alle Benchmarks aus den Slurm-Logs, Median ueber Steps 2..n; "
                        "loss_ok=False markiert Laeufe mit loss=nan (E49)"},
    )
    table = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])
    valid = [r for r in rows if r["loss_ok"]]
    best = max(valid or rows, key=lambda r: r["mfu_pct"])
    run.log({"benchmarks": table})
    run.summary.update({
        "best_mfu_pct": best["mfu_pct"],
        "best_tok_s_dev": best["tok_s_dev"],
        "best_tag": best["tag"],
        "n_runs": len(rows),
        "n_runs_valid": len(valid),
        "n_runs_nan": len(rows) - len(valid),
        "warnung": "Zeilen mit loss_ok=False erzeugen loss=nan und duerfen nicht "
                   "uebernommen werden (E49). best_* bezieht sich nur auf gueltige Laeufe.",
    })
    run.finish()
    print(f"\n-> wandb: {run.url}")


if __name__ == "__main__":
    main()
