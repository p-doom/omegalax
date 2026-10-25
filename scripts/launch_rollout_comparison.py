"""Explicit root-operated gate/production launcher with frozen source and safe fallback.

prepare does not submit. submit-gate / submit-production are explicit submission actions.
poll is idempotent; root calls it at 15/30/45 minutes (or runs watch externally).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
MODEL = "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    # Atomic metadata replacement, within project-side launch directory.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def freeze(root):
    paths = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=REPO
    ).split(b"\0")
    paths = sorted(set(p.decode() for p in paths if p))
    hashes = {
        p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
        for p in paths
        if (REPO / p).is_file()
    }
    source_digest = digest_json(hashes)
    source = root / "source" / source_digest
    if source.exists():
        raise RuntimeError(
            "Source destination already exists without completed launch metadata; inspect before retry"
        )
    source.mkdir(parents=True)
    for relative in hashes:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[relative]
        target.chmod(0o444)
    (source / "SOURCE_SHA256SUMS").write_text(
        "".join(f"{value}  {path}\n" for path, value in hashes.items())
    )
    (source / "SOURCE_SHA256SUMS").chmod(0o444)
    return {
        "source": str(source),
        "source_sha256": source_digest,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "files": hashes,
    }


def job_status(job_id):
    queue = subprocess.check_output(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"], text=True
    ).strip()
    if queue:
        return queue
    accounting = subprocess.check_output(
        ["sacct", "-n", "-X", "-j", str(job_id), "--format=State", "-P"], text=True
    )
    states = [s.split("|")[0].split()[0].rstrip("+") for s in accounting.splitlines() if s.strip()]
    return states[0] if states else "UNKNOWN"


def submit(root, state, key, script, extra):
    submissions = state.setdefault("submissions", {})
    if key in submissions:
        return submissions[key]
    unique_name = f"cmp-{state['source_sha256'][:8]}-{key}"
    # Durable intent BEFORE sbatch. An interrupted ambiguous submission never retries automatically.
    submissions[key] = {"status": "submitting", "name": unique_name, "time": time.time()}
    write_json(root / "launch.json", state)
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={unique_name}",
        f"--output={root}/{key}-%j.log",
        str(Path(state["source"]) / "scripts" / script),
        state["source"],
        state["plan"],
        str(root),
        MODEL,
        *extra,
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    submissions[key]["command"] = command
    submissions[key]["stdout"] = result.stdout
    submissions[key]["stderr"] = result.stderr
    if result.returncode:
        submissions[key]["status"] = "submission_failed"
        write_json(root / "launch.json", state)
        raise RuntimeError(result.stderr)
    submissions[key].update(
        job_id=result.stdout.strip().split(";")[0], status="submitted", time=time.time()
    )
    write_json(root / "launch.json", state)
    return submissions[key]


def verify_gate(root, plan):
    results = {}
    for T in (8192, 512):
        gradient = json.loads((root / f"gradient_T{T}.json").read_text())
        assert gradient["status"] == "PASS"
        resumed = json.loads((root / f"T{T}/resumed/result_step2.json").read_text())
        continuous = json.loads((root / f"T{T}/continuous/result_step2.json").read_text())
        assert resumed["plan_sha256"] == continuous["plan_sha256"] == plan["plan_sha256"]
        assert (
            resumed["optimizer_state_sha256"]
            and resumed["optimizer_state_sha256"] == continuous["optimizer_state_sha256"]
        )
        assert len(resumed["devices"]) == 4
        results[str(T)] = {"gradient": gradient, "resumed": resumed, "continuous": continuous}
    state = json.loads((root / "launch.json").read_text())
    write_json(
        root / "GATE_PASSED.json",
        {
            "status": "PASS",
            "parent_plan_sha256": plan["gate_parent_plan_sha256"],
            "source_sha256": state["source_sha256"],
            "results": results,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare-gate",
            "prepare-production",
            "submit-gate",
            "submit-production",
            "poll",
            "watch",
            "verify-gate",
        ],
    )
    parser.add_argument("--launch-root", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--gate-evidence")
    args = parser.parse_args()
    root = Path(args.launch_root).resolve()
    if not root.is_relative_to(Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/salan")):
        raise SystemExit("Launch artifacts must stay in project-side task storage")
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "watch":
        while True:
            state = json.loads((root / "launch.json").read_text())
            bundle = state["submissions"]["bundle"]
            if "job_id" not in bundle:
                raise RuntimeError("Ambiguous submission; inspect Slurm by unique name")
            status = job_status(bundle["job_id"])
            if status != "PENDING":
                print(f"Bundle {bundle['job_id']} state={status}; no fallback")
                return
            elapsed = time.time() - bundle["time"]
            checked = {x["minute"] for x in state["pending_checks"]}
            due = [m for m in (15, 30, 45) if elapsed >= m * 60 and m not in checked]
            if due:
                subprocess.run(
                    [
                        os.sys.executable,
                        str(Path(__file__).resolve()),
                        "poll",
                        "--launch-root",
                        str(root),
                    ],
                    check=True,
                )
                if 45 in due:
                    return
            time.sleep(
                min(
                    60,
                    max(
                        1,
                        min(
                            (m * 60 - elapsed for m in (15, 30, 45) if m not in checked), default=60
                        ),
                    ),
                )
            )
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action.startswith("prepare-"):
            if (root / "launch.json").exists():
                print((root / "launch.json").read_text())
                return
            plan = json.loads(Path(args.plan).read_text())
            assert plan["plan_sha256"] == digest_json(
                {k: v for k, v in plan.items() if k != "plan_sha256"}
            )
            parent = plan["plan_sha256"]
            if args.action == "prepare-gate":
                largest = max(
                    (i for i, r in enumerate(plan["records"]) if r["split"] == "train"),
                    key=lambda i: plan["records"][i]["token_count"],
                )
                # Use a genuine production update containing the longest selected chain.
                # This retains production MultiSteps buffers and ~500k actual tokens/update.
                gate_update = next(u for u in plan["updates"] if largest in u["members"])
                members = gate_update["members"]
                n = gate_update["context_tokens"]
                identity = [
                    {
                        k: plan["records"][i][k]
                        for k in ("sequence_id", "content_key", "token_count", "target_count")
                    }
                    for i in members
                ]
                plan.update(
                    gate_parent_plan_sha256=parent,
                    total_context_tokens=3 * n,
                    warmup_end_update=1,
                    preanneal_update=2,
                    checkpoint_updates=[1, 2, 3],
                    updates=[
                        {
                            "epoch": 0,
                            "members": members,
                            "context_tokens": n,
                            "cumulative_tokens": (i + 1) * n,
                            "identity_sha256": digest_json(identity),
                        }
                        for i in range(3)
                    ],
                )
                plan["plan_sha256"] = digest_json(
                    {k: v for k, v in plan.items() if k != "plan_sha256"}
                )
            write_json(root / "plan.json", plan)
            (root / "plan.json").chmod(0o444)
            state = {
                **freeze(root),
                "plan": str(root / "plan.json"),
                "plan_sha256": plan["plan_sha256"],
                "parent_plan_sha256": parent,
                "phase": args.action.removeprefix("prepare-"),
                "prepared_at": time.time(),
                "submissions": {},
                "pending_checks": [],
            }
            write_json(root / "launch.json", state)
            print(json.dumps({k: v for k, v in state.items() if k != "files"}, indent=2))
            return
        state = json.loads((root / "launch.json").read_text())
        if args.action == "verify-gate":
            verify_gate(root, json.loads(Path(args.plan).read_text()))
        elif args.action == "submit-gate":
            assert state["phase"] == "gate"
            submit(root, state, "gate", "rollout_comparison_single.sbatch", ["gate"])
        elif args.action == "submit-production":
            assert state["phase"] == "production"
            gate = json.loads(Path(args.gate_evidence).read_text())
            assert gate["status"] == "PASS" and gate["parent_plan_sha256"] == state["plan_sha256"]
            assert gate["source_sha256"] == state["source_sha256"], (
                "Gate source differs; rerun gate on frozen production source"
            )
            state["gate_evidence"] = str(Path(args.gate_evidence).resolve())
            submit(root, state, "bundle", "rollout_comparison_bundle.sbatch", [])
        elif args.action == "poll":
            bundle = state["submissions"]["bundle"]
            if "job_id" not in bundle:
                raise RuntimeError(
                    "Ambiguous bundle submission: inspect Slurm by unique name; no auto retry"
                )
            status = job_status(bundle["job_id"])
            elapsed = time.time() - bundle["time"]
            for minute in (15, 30, 45):
                if elapsed >= 60 * minute and not any(
                    x["minute"] == minute for x in state["pending_checks"]
                ):
                    state["pending_checks"].append(
                        {"minute": minute, "observed_at": time.time(), "state": status}
                    )
            write_json(root / "launch.json", state)
            if status == "PENDING" and elapsed >= 45 * 60:
                subprocess.run(["scancel", "--state=PENDING", bundle["job_id"]], check=True)
                # Cancellation must be confirmed in accounting, and absent from the live queue.
                for _ in range(30):
                    status = job_status(bundle["job_id"])
                    live = subprocess.check_output(
                        ["squeue", "-h", "-j", bundle["job_id"], "-o", "%T"], text=True
                    ).strip()
                    if status == "CANCELLED" and not live:
                        bundle["cancellation_confirmed"] = time.time()
                        write_json(root / "launch.json", state)
                        break
                    if status in ("RUNNING", "COMPLETED"):
                        return
                    time.sleep(2)
            if bundle.get("cancellation_confirmed"):
                assert job_status(bundle["job_id"]) == "CANCELLED"
                for T in (8192, 2048, 512):
                    submit(
                        root, state, f"T{T}", "rollout_comparison_single.sbatch", ["train", str(T)]
                    )
            print(
                json.dumps(
                    {
                        "bundle_state": status,
                        "elapsed_seconds": elapsed,
                        "submissions": state["submissions"],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
