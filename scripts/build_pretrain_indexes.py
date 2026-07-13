"""Build reusable fixed-window pretraining indexes."""

from __future__ import annotations

from pathlib import Path

from absl import app, flags

from omegalax.data.pretrain_data_set import DEFAULT_CHUNK_LENGTH, DEFAULT_EOS_ID, DataSetReader
from omegalax.data.pretrain_statepassing import (
    build_statepassing_curriculum_indexes,
    build_statepassing_fixed_c_indexes,
    build_statepassing_window_index,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("root", None, "Doc-chain dataset root.", required=True)
flags.DEFINE_string("out_dir", None, "Output root for index directories.", required=True)
flags.DEFINE_multi_string("split", ["train", "val"], "Dataset splits to index.")
flags.DEFINE_integer("chunk_length", DEFAULT_CHUNK_LENGTH, "Segment length.")
flags.DEFINE_integer("num_segments", 2, "Fixed number of chunks per statepassing window.")
flags.DEFINE_integer("eos_id", DEFAULT_EOS_ID, "EOS id used for retained-tail repair.")
flags.DEFINE_integer("records_per_shard", 100_000, "Records per output shard.")
flags.DEFINE_bool("overwrite", False, "Overwrite existing output directories.")
flags.DEFINE_integer("eos_check_records", 0, "If >0, sample final-token EOS ids before building.")
flags.DEFINE_string(
    "curriculum_allocation_order",
    None,
    "Comma-separated strictly-descending segment lengths for curriculum assignment.",
)
flags.DEFINE_string(
    "curriculum_train_order",
    None,
    "Comma-separated segment lengths for curriculum training order.",
)
flags.DEFINE_multi_string(
    "curriculum_max_tokens",
    [],
    "Optional C:max_tokens caps for curriculum segments; may be repeated or comma-separated.",
)
flags.DEFINE_integer(
    "curriculum_trim_batch_size",
    None,
    "Global segment batch size used to trim curriculum phases to full optimizer steps.",
)
flags.DEFINE_integer(
    "curriculum_trim_grad_accum_steps",
    None,
    "Gradient accumulation steps used to trim curriculum phases to full optimizer steps.",
)
flags.DEFINE_integer(
    "fixed_trim_batch_size",
    None,
    "Global segment batch size used to trim fixed-C indexes to full optimizer steps.",
)
flags.DEFINE_integer(
    "fixed_trim_grad_accum_steps",
    None,
    "Gradient accumulation steps used to trim fixed-C indexes to full optimizer steps.",
)
flags.DEFINE_float(
    "min_eos_fraction",
    None,
    "If set, require the sampled dominant final-token id to equal --eos_id with at least "
    "this fraction before building.",
)


def parse_curriculum_order(value: str | None, *, flag_name: str) -> list[int]:
    if value is None:
        return []
    order = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not order:
        raise ValueError(f"--{flag_name} must not be empty")
    return order


def parse_curriculum_max_tokens(values: list[str] | tuple[str, ...] | None) -> dict[int, int]:
    caps: dict[int, int] = {}
    for raw_value in values or []:
        for item in str(raw_value).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid --curriculum_max_tokens entry: {item!r}")
            raw_num_segments, raw_max_tokens = item.split(":", 1)
            num_segments = int(raw_num_segments)
            max_tokens = int(raw_max_tokens)
            if num_segments in caps:
                raise ValueError(f"Duplicate --curriculum_max_tokens entry for C={num_segments}")
            caps[num_segments] = max_tokens
    return caps


def eos_final_token_stats(root: str | Path, *, split: str, max_records: int) -> dict[str, object]:
    reader = DataSetReader(root, split=split)
    counts: dict[int, int] = {}
    sampled = 0
    for _, _, doc in reader.iter_records():
        if doc.token_ids.size:
            token_id = int(doc.token_ids[-1])
            counts[token_id] = counts.get(token_id, 0) + 1
            sampled += 1
        if sampled >= max_records:
            break
    dominant_id = max(counts, key=counts.get) if counts else None
    dominant_count = counts.get(dominant_id, 0) if dominant_id is not None else 0
    return {
        "sampled": sampled,
        "counts": counts,
        "dominant_id": dominant_id,
        "dominant_fraction": dominant_count / sampled if sampled else 0.0,
    }


def validate_eos_stats(
    stats: dict[str, object],
    *,
    eos_id: int | None,
    min_fraction: float | None,
    split: str,
) -> None:
    if min_fraction is None:
        return
    if eos_id is None:
        raise ValueError("--min_eos_fraction requires a non-None --eos_id")
    sampled = int(stats.get("sampled") or 0)
    dominant_id = stats.get("dominant_id")
    dominant_fraction = float(stats.get("dominant_fraction") or 0.0)
    if sampled <= 0:
        raise ValueError(f"EOS sanity check for split={split} sampled no records.")
    if dominant_id != eos_id or dominant_fraction < min_fraction:
        raise ValueError(
            f"EOS sanity check failed for split={split}: dominant_id={dominant_id}, "
            f"dominant_fraction={dominant_fraction:.4f}, expected eos_id={eos_id} "
            f"with fraction >= {min_fraction:.4f}."
        )


def build_pretrain_index(
    *,
    root: str | Path,
    out_dir: str | Path,
    split: str,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    num_segments: int = 2,
    eos_id: int | None = DEFAULT_EOS_ID,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    out_path = Path(out_dir).expanduser().resolve() / split
    return build_statepassing_window_index(
        root,
        out_path,
        chunk_length=chunk_length,
        num_segments=num_segments,
        eos_id=eos_id,
        split=split,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
    )


def build_curriculum_pretrain_indexes(
    *,
    root: str | Path,
    out_dir: str | Path,
    splits: list[str] | tuple[str, ...],
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    allocation_order: list[int],
    train_order: list[int],
    max_tokens_by_num_segments: dict[int, int] | None = None,
    trim_batch_size: int,
    trim_grad_accum_steps: int,
    eos_id: int | None = DEFAULT_EOS_ID,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    return build_statepassing_curriculum_indexes(
        root,
        out_dir,
        allocation_order=allocation_order,
        train_order=train_order,
        max_tokens_by_num_segments=max_tokens_by_num_segments,
        trim_batch_size=trim_batch_size,
        trim_grad_accum_steps=trim_grad_accum_steps,
        chunk_length=chunk_length,
        eos_id=eos_id,
        splits=splits,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
    )


def build_fixed_c_pretrain_indexes(
    *,
    root: str | Path,
    out_dir: str | Path,
    splits: list[str] | tuple[str, ...],
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    num_segments: int = 2,
    trim_batch_size: int,
    trim_grad_accum_steps: int,
    eos_id: int | None = DEFAULT_EOS_ID,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    return build_statepassing_fixed_c_indexes(
        root,
        out_dir,
        num_segments=num_segments,
        trim_batch_size=trim_batch_size,
        trim_grad_accum_steps=trim_grad_accum_steps,
        chunk_length=chunk_length,
        eos_id=eos_id,
        splits=splits,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
    )


def main(_) -> None:
    if FLAGS.eos_check_records:
        for split in FLAGS.split:
            stats = eos_final_token_stats(
                FLAGS.root, split=split, max_records=FLAGS.eos_check_records
            )
            validate_eos_stats(
                stats,
                eos_id=FLAGS.eos_id,
                min_fraction=FLAGS.min_eos_fraction,
                split=split,
            )
            print(f"eos_stats split={split}: {stats}")

    wants_curriculum = bool(FLAGS.curriculum_allocation_order or FLAGS.curriculum_train_order)
    if wants_curriculum:
        if not FLAGS.curriculum_allocation_order or not FLAGS.curriculum_train_order:
            raise ValueError(
                "--curriculum_allocation_order and --curriculum_train_order must be set together"
            )
        if FLAGS.curriculum_trim_batch_size is None:
            raise ValueError("--curriculum_trim_batch_size is required for curriculum indexes")
        if FLAGS.curriculum_trim_grad_accum_steps is None:
            raise ValueError(
                "--curriculum_trim_grad_accum_steps is required for curriculum indexes"
            )
        out = build_curriculum_pretrain_indexes(
            root=FLAGS.root,
            out_dir=FLAGS.out_dir,
            splits=FLAGS.split,
            chunk_length=FLAGS.chunk_length,
            allocation_order=parse_curriculum_order(
                FLAGS.curriculum_allocation_order,
                flag_name="curriculum_allocation_order",
            ),
            train_order=parse_curriculum_order(
                FLAGS.curriculum_train_order,
                flag_name="curriculum_train_order",
            ),
            max_tokens_by_num_segments=parse_curriculum_max_tokens(FLAGS.curriculum_max_tokens),
            trim_batch_size=FLAGS.curriculum_trim_batch_size,
            trim_grad_accum_steps=FLAGS.curriculum_trim_grad_accum_steps,
            eos_id=FLAGS.eos_id,
            records_per_shard=FLAGS.records_per_shard,
            overwrite=FLAGS.overwrite,
        )
        print(out)
        return

    if FLAGS.fixed_trim_batch_size is None:
        raise ValueError("--fixed_trim_batch_size is required for fixed-C indexes")
    if FLAGS.fixed_trim_grad_accum_steps is None:
        raise ValueError("--fixed_trim_grad_accum_steps is required for fixed-C indexes")
    out = build_fixed_c_pretrain_indexes(
        root=FLAGS.root,
        out_dir=FLAGS.out_dir,
        splits=FLAGS.split,
        chunk_length=FLAGS.chunk_length,
        num_segments=FLAGS.num_segments,
        trim_batch_size=FLAGS.fixed_trim_batch_size,
        trim_grad_accum_steps=FLAGS.fixed_trim_grad_accum_steps,
        eos_id=FLAGS.eos_id,
        records_per_shard=FLAGS.records_per_shard,
        overwrite=FLAGS.overwrite,
    )
    print(out)


if __name__ == "__main__":
    app.run(main)
