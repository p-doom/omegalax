"""Check the fixed interference matrix before any model evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket


if (
    not os.environ.get("SLURM_JOB_ID")
    or not os.environ.get("SLURMD_NODENAME")
    or (os.environ["SLURMD_NODENAME"].split(".", 1)[0] != socket.gethostname().split(".", 1)[0])
):
    raise RuntimeError("Submit structured interference preflight with sbatch")


import numpy as np  # noqa: E402

from omegalax.data.memory_tasks import (  # noqa: E402
    MemoryTaskConfig,
    build_memory_interference_batch,
    build_memory_task_batch,
)
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY  # noqa: E402
from omegalax.evals.memory_pollution import validate_pollution_free_manifest  # noqa: E402


ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/structured_interference_20260930"
)
MATRIX = {
    "exact_marked": (4, 8, 16, 32),
    "exact_surprise": (4, 8, 16, 32),
    "dominant_rule": (4, 8, 16, 32),
    "exact_d7": (4, 8, 16, 32),
    "associative_bank": (5, 6, 8),
    "rule_bank": (6, 7, 9),
}


def main() -> None:
    report = {}
    for family, cs in MATRIX.items():
        manifest = json.loads((ROOT / "manifests" / f"{family}.json").read_text())
        validate_pollution_free_manifest(manifest)
        config = MemoryTaskConfig(
            **{
                key: manifest[key]
                for key in (
                    "task",
                    "seed",
                    "segment_length",
                    "num_segments",
                    "vocab_size",
                    "bank_value_alphabet",
                    "num_rule_examples",
                    "num_exact_bindings",
                    "exact_sequence_length",
                    "relevance_cue",
                    "split",
                    "generator_version",
                )
            }
        )
        indices = tuple(manifest["pair_indices"][:2])
        base = build_memory_task_batch(indices, config, flatten=False)
        original = base["token_ids_BCT"]
        base_metadata = base[BATCH_PRETRAIN_METADATA_KEY]
        for c in cs:
            filler = build_memory_interference_batch(
                indices, config, eval_num_segments=c, structured=False, flatten=False
            )
            structured = build_memory_interference_batch(
                indices, config, eval_num_segments=c, structured=True, flatten=False
            )
            for condition, batch in (("filler", filler), ("structured", structured)):
                tokens = batch["token_ids_BCT"]
                added = tokens[:, config.num_segments - 1 : -1]
                metadata = batch[BATCH_PRETRAIN_METADATA_KEY]
                if tokens.shape != (4, c, config.segment_length):
                    raise ValueError(f"{family} C{c} {condition}: wrong token shape")
                if not np.array_equal(tokens[:, : config.num_segments - 1], original[:, :-1]):
                    raise ValueError(f"{family} C{c} {condition}: original prefix changed")
                if not np.array_equal(tokens[:, -1], original[:, -1]):
                    raise ValueError(f"{family} C{c} {condition}: target query changed")
                if not np.array_equal(
                    metadata["answer_tokens_BD"], base_metadata["answer_tokens_BD"]
                ):
                    raise ValueError(f"{family} C{c} {condition}: answer changed")
                if not np.array_equal(added[0::2], added[1::2]):
                    raise ValueError(
                        f"{family} C{c} {condition}: paired worlds have different additions"
                    )
                if not np.array_equal(
                    batch["loss_mask_BCT"][:, -1], base["loss_mask_BCT"][:, -1]
                ) or np.any(batch["loss_mask_BCT"][:, :-1]):
                    raise ValueError(f"{family} C{c} {condition}: loss mask changed")
                if not np.all(batch["reset_state_BC"][:, 0]) or np.any(
                    batch["reset_state_BC"][:, 1:]
                ):
                    raise ValueError(f"{family} C{c} {condition}: invalid state resets")
                if not np.array_equal(
                    metadata["pair_index_B"], np.repeat(indices, 2)
                ) or not np.array_equal(metadata["world_B"], [0, 1, 0, 1]):
                    raise ValueError(f"{family} C{c} {condition}: state-swap pair order changed")
                if np.any(
                    np.all(
                        metadata["answer_tokens_BD"][0::2] == metadata["answer_tokens_BD"][1::2],
                        axis=1,
                    )
                ):
                    raise ValueError(f"{family} C{c} {condition}: paired answers are identical")
                filler_start = 492 if family in ("associative_bank", "rule_bank") else 256
                if condition == "filler" and np.any(added < filler_start):
                    raise ValueError(f"{family} C{c}: filler control has task tokens")
                tag = (
                    7
                    if family == "exact_marked"
                    else 3
                    if family in ("exact_surprise", "exact_d7", "associative_bank")
                    else 4
                )
                count = (
                    1
                    if family == "exact_d7"
                    else 8
                    if family == "associative_bank"
                    else 14
                    if family == "rule_bank"
                    else 5
                )
                if condition == "structured" and np.any(np.sum(added == tag, axis=2) != count):
                    raise ValueError(f"{family} C{c}: structured record count changed")
            if not np.array_equal(
                filler["token_ids_BCT"][:, -1], structured["token_ids_BCT"][:, -1]
            ):
                raise ValueError(f"{family} C{c}: condition target mismatch")
        report[family] = {
            "manifest_sha256": manifest["sha256"],
            "checked_pair_indices": indices,
            "eval_c": cs,
        }
        print(f"{family}: {len(cs)} C values, both conditions passed", flush=True)
    output = ROOT / "preflight_receipt.json"
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise ValueError(f"Refusing to overwrite {output}")
    if not output.exists():
        output.write_text(content)


if __name__ == "__main__":
    main()
