"""Build the six fixed, shared-pair interference holdouts on a compute node."""

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
    raise RuntimeError("Submit structured interference manifest construction with sbatch")


from omegalax.data.memory_tasks import MemoryTaskConfig, MemoryTaskSplit  # noqa: E402
from omegalax.evals.memory_pollution import (  # noqa: E402
    build_pollution_free_manifest,
    load_training_streams,
    validate_pollution_free_manifest,
)
from omegalax.text.checkpoint import resolve_checkpoint  # noqa: E402
from structured_interference_compat import allow_verified_d7_training_content  # noqa: E402


ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/salan")
LOG = ROOT / "logs/structured_interference_20260930"
CURRICULUM = ROOT / "runs/memory_curriculum_qwen08_20260902"
D7 = ROOT / "runs/exact_sequence_d7_20260912"
CURRICULUM_LOG = ROOT / "logs/memory_curriculum_qwen08_20260902"
D7_LOG = ROOT / "logs/exact_sequence_d7_20260912"
SPECS = {
    "exact_marked": (
        [(CURRICULUM, "mem_q08_exact_k3_marked_c2_carry_b0_lr1e4_m0_d0")],
        [
            "qwen08_exact_marked_c2_calibration",
            "qwen08_exact_marked_c4_b0",
            "qwen08_exact_marked_c6_b0",
        ],
        2048,
    ),
    "exact_surprise": (
        [(CURRICULUM, "mem_q08_exact_k3_surprise_c2_carry_b0_lr1e4_m0_d0")],
        [
            "qwen08_exact_surprise_c2_b0",
            "qwen08_exact_surprise_c4_b0",
            "qwen08_exact_surprise_c6_b0",
        ],
        2048,
    ),
    "dominant_rule": (
        [(CURRICULUM, "mem_q08_dominant_rule5_c2_carry_b0_lr1e4_m0_d0")],
        ["qwen08_dominant_rule_c2_b0", "qwen08_dominant_rule_c4_b0", "qwen08_dominant_rule_c6_b0"],
        2048,
    ),
    "exact_d7": (
        [
            (D7, "mem_q08_exact_d7_c2_carry_b0_lr1e4_m0_d0"),
            (D7, "mem_q08_exact_d7_c2_full_b0_lr1e4_m0_d0"),
        ],
        [
            D7_LOG / "qwen08_d7_full_carry_eval_v1/gate_seed1.manifest.json",
            D7_LOG / "qwen08_full_b0/gate_eval_step2048/gate_seed1.manifest.json",
        ],
        2048,
    ),
    "associative_bank": (
        [
            (CURRICULUM, "mem_q08_associative_bank_a256_carry_b0_lr1e4_m0_d0"),
            (CURRICULUM, "mem_q08_associative_bank_a256_full_b0_lr1e4_m0_d0"),
        ],
        ["qwen08_associative_bank_a256_carry_b0", "qwen08_associative_bank_a256_full_b0_v3"],
        3072,
    ),
    "rule_bank": (
        [
            (CURRICULUM, "mem_q08_rule_bank_c5_carry_b0_lr1e4_m0_d0"),
            (CURRICULUM, "mem_q08_rule_bank_c5_full_b0_lr1e4_m0_d0"),
        ],
        ["qwen08_rule_bank_carry_b0", "qwen08_rule_bank_full_b0"],
        2048,
    ),
}


def write_immutable(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(content)


def main() -> None:
    allow_verified_d7_training_content()
    report = {}
    for family, (endpoints, gates, num_pairs) in SPECS.items():
        streams = {}
        resolved = []
        for root, name in endpoints:
            checkpoint = resolve_checkpoint(root / name / "002048")
            resolved.append(str(root / name / "002048"))
            for stream in load_training_streams(checkpoint.root, checkpoint.step):
                streams[(stream.checkpoint_root, stream.checkpoint_step)] = stream
        contract = json.loads(
            (endpoints[0][0] / endpoints[0][1] / "memory_task_contract.json").read_text()
        )
        config = MemoryTaskConfig(
            **(contract["task"] | {"seed": 3, "split": MemoryTaskSplit.EVAL.value})
        )
        exclusions = [
            gate
            if isinstance(gate, Path)
            else CURRICULUM_LOG / gate / "gate_eval_step2048/manifest.json"
            for gate in gates
        ]
        manifest = build_pollution_free_manifest(
            config,
            num_pairs=num_pairs,
            training_streams=tuple(streams.values()),
            exclude_manifests=exclusions,
            final_holdout=True,
        )
        validate_pollution_free_manifest(manifest)
        output = LOG / "manifests" / f"{family}.json"
        write_immutable(output, manifest)
        report[family] = {
            "manifest": str(output),
            "manifest_sha256": manifest["sha256"],
            "num_pairs": num_pairs,
            "checkpoints": resolved,
            "ancestry": [
                {
                    "checkpoint_root": str(stream.checkpoint_root),
                    "checkpoint_step": stream.checkpoint_step,
                    "contract_sha256": stream.contract_sha256,
                    "train_pairs": stream.train_pairs,
                }
                for stream in streams.values()
            ],
            "excluded_gates": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in exclusions
            ],
        }
        print(f"{family}: selected {num_pairs} pairs; {len(streams)} ancestry streams", flush=True)
    write_immutable(LOG / "manifest_build_receipt.json", report)


if __name__ == "__main__":
    main()
