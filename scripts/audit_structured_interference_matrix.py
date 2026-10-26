"""Verify every fixed matrix result and its Wiki plot mirror."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket


if (
    not os.environ.get("SLURM_JOB_ID")
    or not os.environ.get("SLURMD_NODENAME")
    or (os.environ["SLURMD_NODENAME"].split(".", 1)[0] != socket.gethostname().split(".", 1)[0])
):
    raise RuntimeError("Submit structured interference audit with sbatch")


ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/salan")
LOG = ROOT / "logs/structured_interference_20260930"
WIKI = Path(
    "/fast/home/salan.isaqzoi/gh_projects/salanobp/codex-setup/Wiki/documentation/evals/synthetic_memory/by_generation/generation_5/qwen08"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for index, line in enumerate((LOG / "matrix.tsv").read_text().splitlines(), 1):
        family, run_group, checkpoint, c, condition = line.split("\t")
        manifest = json.loads((LOG / "manifests" / f"{family}.json").read_text())
        task = manifest["task"]
        name = f"memory_{task}{'_d7' if family == 'exact_d7' else ''}_final_pollution_guard_v1_structured20260930_interference_c{c}_{condition}"
        run_dir = ROOT / "runs" / run_group / checkpoint / "evals" / name / "step_2048"
        row = {
            "matrix_index": index,
            "family": family,
            "checkpoint": checkpoint,
            "eval_c": int(c),
            "condition": condition,
            "run_dir": str(run_dir),
        }
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            row["status"] = "missing"
            rows.append(row)
            continue
        row["status"] = json.loads(status_path.read_text())["status"]
        if row["status"] != "complete":
            rows.append(row)
            continue
        config = json.loads((run_dir / "eval_config.json").read_text())
        copy = json.loads((run_dir / "manifest.json").read_text())
        if (
            config["manifest_sha256"] != manifest["sha256"]
            or copy != manifest
            or config["interference"]["eval_num_segments"] != int(c)
            or config["interference"]["condition"] != condition
            or config["requested_eval_task"]["seed"] != 3
        ):
            raise ValueError(f"Matrix input mismatch at row {index}")
        artifacts = json.loads((run_dir / "artifact_manifest.json").read_text())["artifacts"]
        for relative, expected in artifacts.items():
            if sha256(run_dir / relative) != expected:
                raise ValueError(f"Artifact mismatch at row {index}: {relative}")
        mirror = WIKI / family / "evals" / f"{checkpoint}_step2048" / "full_weights" / name
        for filename in ("accuracy.svg", "answer_nll.svg"):
            if sha256(mirror / filename) != sha256(run_dir / "plots" / filename):
                raise ValueError(f"Wiki plot mirror mismatch at row {index}: {filename}")
        if str(run_dir) not in (mirror / "README.md").read_text():
            raise ValueError(f"Wiki README source mismatch at row {index}")
        summary = json.loads((run_dir / "summary.json").read_text())["modes"]
        if any(
            mode["num_examples"] != 2 * len(manifest["pair_indices"]) for mode in summary.values()
        ):
            raise ValueError(f"Incomplete result count at row {index}")
        row["correct_state_accuracy"] = summary["correct_state"]["accuracy"]
        row["donor_switch_accuracy"] = summary["counterfactual_state"]["donor_accuracy"]
        row["reset_accuracy"] = summary["reset"]["accuracy"]
        row["full_context_accuracy"] = summary["full_context"]["accuracy"]
        row["accuracy_by_target_segment"] = summary["correct_state"].get(
            "accuracy_by_target_segment"
        )
        row["artifact_manifest_sha256"] = sha256(run_dir / "artifact_manifest.json")
        rows.append(row)
    report = {
        "schema_version": 1,
        "matrix_cells": len(rows),
        "complete": sum(row["status"] == "complete" for row in rows),
        "rows": rows,
    }
    output = LOG / "matrix_execution_receipt.json"
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise ValueError(f"Refusing to overwrite {output}")
    if not output.exists():
        output.write_text(content)
    print(f"Verified {report['complete']}/{len(rows)} matrix cells", flush=True)
    if len(rows) != 64 or report["complete"] != 64:
        raise RuntimeError("The fixed structured interference matrix is incomplete")


if __name__ == "__main__":
    main()
