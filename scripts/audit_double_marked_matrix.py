"""Verify every fixed double-marked result and its Wiki plot mirror."""

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
    raise RuntimeError("Submit double-marked audit with sbatch")


ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/salan")
LOG = ROOT / "logs/double_marked_20261001"
CHECKPOINT = "mem_q08_exact_k3_marked_c2_carry_b0_lr1e4_m0_d0"
EVAL_ROOT = ROOT / "runs/memory_curriculum_qwen08_20260902" / CHECKPOINT / "evals"
WIKI = Path(
    "/fast/home/salan.isaqzoi/gh_projects/salanobp/codex-setup/Wiki/documentation/evals/synthetic_memory/by_generation/generation_5/qwen08/exact_marked/evals"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((LOG / "manifest.json").read_text())
    rows = []
    for index, line in enumerate((LOG / "matrix.tsv").read_text().splitlines(), 1):
        c, variant, condition = line.split("\t")
        name = f"memory_exact_final_pollution_guard_v1_doublemarked20261001_interference_c{c}_{condition}_double_marked_{variant}"
        run_dir = EVAL_ROOT / name / "step_2048"
        status = json.loads((run_dir / "status.json").read_text())["status"]
        if status != "complete":
            raise ValueError(f"Cell {index} is {status}")
        config = json.loads((run_dir / "eval_config.json").read_text())
        if (
            json.loads((run_dir / "manifest.json").read_text()) != manifest
            or config["manifest_sha256"] != manifest["sha256"]
            or config["requested_eval_task"]["seed"] != 4
            or config["interference"]["eval_num_segments"] != int(c)
            or config["interference"]["condition"] != condition
            or config["interference"]["double_marked_variant"] != variant
            or config["interference"]["generator"] != "build_double_marked_eval_batch"
        ):
            raise ValueError(f"Input mismatch at cell {index}")
        artifacts = json.loads((run_dir / "artifact_manifest.json").read_text())["artifacts"]
        for relative, expected in artifacts.items():
            if sha256(run_dir / relative) != expected:
                raise ValueError(f"Artifact mismatch at cell {index}: {relative}")
        mirror = WIKI / f"{CHECKPOINT}_step2048" / "full_weights" / name
        for filename in ("accuracy.svg", "answer_nll.svg"):
            if sha256(mirror / filename) != sha256(run_dir / "plots" / filename):
                raise ValueError(f"Plot mirror mismatch at cell {index}: {filename}")
        if str(run_dir) not in (mirror / "README.md").read_text():
            raise ValueError(f"README mirror mismatch at cell {index}")
        modes = json.loads((run_dir / "summary.json").read_text())["modes"]
        if any(
            mode["num_examples"] != 2 * len(manifest["pair_indices"]) for mode in modes.values()
        ):
            raise ValueError(f"Incomplete result count at cell {index}")
        rows.append(
            {
                "matrix_index": index,
                "eval_c": int(c),
                "variant": variant,
                "condition": condition,
                "run_dir": str(run_dir),
                "correct_state_accuracy": modes["correct_state"]["accuracy"],
                "donor_switch_accuracy": modes["counterfactual_state"]["donor_accuracy"],
                "counterfactual_own_accuracy": modes["counterfactual_state"]["accuracy"],
                "reset_accuracy": modes["reset"]["accuracy"],
                "full_context_accuracy": modes["full_context"]["accuracy"],
                "artifact_manifest_sha256": sha256(run_dir / "artifact_manifest.json"),
            }
        )
    if len(rows) != 12:
        raise ValueError(f"Expected 12 cells, got {len(rows)}")
    output = LOG / "matrix_execution_receipt.json"
    content = json.dumps({"matrix_cells": len(rows), "rows": rows}, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise ValueError(f"Refusing to overwrite {output}")
    if not output.exists():
        output.write_text(content)
    print(f"Verified {len(rows)}/12 double-marked cells", flush=True)


if __name__ == "__main__":
    main()
