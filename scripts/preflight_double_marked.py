"""Check the fixed double-marked eval inputs on a Slurm compute node."""

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
    raise RuntimeError("Submit double-marked preflight with sbatch")


import numpy as np  # noqa: E402

from omegalax.data.memory_tasks import (  # noqa: E402
    MemoryTaskConfig,
    build_double_marked_eval_batch,
    build_memory_task_batch,
)
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY  # noqa: E402
from omegalax.evals.memory_pollution import validate_pollution_free_manifest  # noqa: E402


ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/double_marked_20261001")


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
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
    pair_indices = tuple(manifest["pair_indices"][:2])
    base = build_memory_task_batch(pair_indices, config, flatten=False)
    for eval_c in (4, 8, 16, 32):
        batches = {
            variant: build_double_marked_eval_batch(
                pair_indices, config, eval_num_segments=eval_c, variant=variant
            )
            for variant in ("one", "first", "second")
        }
        one = batches["one"]["token_ids_BCT"]
        first = batches["first"]["token_ids_BCT"]
        second = batches["second"]["token_ids_BCT"]
        if not np.array_equal(one[:, 0], base["token_ids_BCT"][:, 0]):
            raise ValueError(f"C{eval_c}: original early segment changed")
        if not np.array_equal(first[:, 0], one[:, 0]) or not np.array_equal(
            second[:, 0], one[:, 0]
        ):
            raise ValueError(f"C{eval_c}: conditions changed the original early segment")
        if not np.array_equal(first[:, 1], second[:, 1]):
            raise ValueError(f"C{eval_c}: two-binding conditions differ before the question")
        if not np.array_equal(one[:, 2:-1], first[:, 2:-1]) or not np.array_equal(
            first[:, 2:-1], second[:, 2:-1]
        ):
            raise ValueError(f"C{eval_c}: added filler differs between conditions")
        if not np.all(one[:, 1] >= 256) or not np.all(first[:, 1, 3:] >= 256):
            raise ValueError(f"C{eval_c}: unexpected tokens in the added segment")
        if not np.all(first[:, 1, 0] == 3):
            raise ValueError(f"C{eval_c}: missing second true BIND")
        for pair_offset in range(len(pair_indices)):
            rows = slice(2 * pair_offset, 2 * pair_offset + 2)
            new_key = first[2 * pair_offset, 1, 1]
            if new_key in first[2 * pair_offset, 0, 2:10:3]:
                raise ValueError(f"C{eval_c}: new Key collides with an original Key")
            early_values = first[rows, 0, 3:10:3]
            new_values = first[rows, 1, 2]
            if len(set(new_values)) != 2 or np.isin(new_values, early_values).any():
                raise ValueError(f"C{eval_c}: new values collide")
            if not np.array_equal(first[rows, -1], one[rows, -1]):
                raise ValueError(f"C{eval_c}: original query changed")
            if not np.all(second[rows, -1, -3] == new_key) or not np.array_equal(
                second[rows, -1, -1], new_values
            ):
                raise ValueError(f"C{eval_c}: new query or answer is wrong")
        for variant, batch in batches.items():
            tokens = batch["token_ids_BCT"]
            metadata = batch[BATCH_PRETRAIN_METADATA_KEY]
            if tokens.shape != (4, eval_c, 16) or int(batch["loss_mask_BCT"].sum()) != 4:
                raise ValueError(f"C{eval_c} {variant}: shape or loss mask changed")
            if not np.array_equal(metadata["answer_tokens_BD"][:, 0], tokens[:, -1, -1]):
                raise ValueError(f"C{eval_c} {variant}: target metadata mismatch")
            if not np.all(batch["reset_state_BC"][:, 0]) or np.any(batch["reset_state_BC"][:, 1:]):
                raise ValueError(f"C{eval_c} {variant}: invalid state reset")
        print(f"C{eval_c}: one, first, and second passed", flush=True)
    receipt = {
        "manifest_sha256": manifest["sha256"],
        "checked_pair_indices": pair_indices,
        "eval_c": (4, 8, 16, 32),
        "variants": ("one", "first", "second"),
    }
    output = ROOT / "preflight_receipt.json"
    content = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise ValueError(f"Refusing to overwrite {output}")
    if not output.exists():
        output.write_text(content)


if __name__ == "__main__":
    main()
