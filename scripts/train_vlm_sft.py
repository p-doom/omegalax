"""VLM SFT training from a compiled Grain dataset (text-only or multimodal)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from tensorboardX import SummaryWriter
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.collator_qwen3 import VLMSFTCollator
from omegalax.data.grain_pipeline import (
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    required_epochs_for_batches,
)
from omegalax.distributed.mesh import process_local_batch_size
from omegalax.trainers import vlm as vlm_trainer
from omegalax.registry import resolve_hf_repo_id
from omegalax.trainers.perf import resolve_peak_tflops


def _default_save_dir(model_id: str) -> Path:
    safe_name = model_id.replace("/", "_")
    return Path("runs") / "vlm_sft" / safe_name


def _grain_iter(
    data_path: str,
    collator: VLMSFTCollator,
    per_process_batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    grain_read_threads: int,
    grain_read_buffer_size: int,
    grain_workers: int,
    grain_worker_buffer_size: int,
    num_batches: int,
):
    return make_grain_iterator(
        data_path,
        batch_size=per_process_batch_size,
        batch_fn=collator,
        shuffle=shuffle,
        seed=seed,
        num_epochs=required_epochs_for_batches(
            data_path, batch_size=per_process_batch_size, num_batches=num_batches
        ),
        read_options=make_grain_read_options(
            num_threads=grain_read_threads,
            prefetch_buffer_size=grain_read_buffer_size,
        ),
        multiprocessing_options=make_grain_multiprocessing_options(
            num_workers=grain_workers,
            per_worker_buffer_size=grain_worker_buffer_size,
        ),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT a VLM from a compiled Grain chunk-index dataset.")
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to a compiled Grain chunk-index dataset directory. Build it from raw JSONL via scripts/compile_sft_dataset.py and scripts/build_sft_chunk_index.py.",
    )
    p.add_argument("--processor", type=str, default=None, help="HF repo to read tokenizer and image config from (defaults to --model-id).")
    p.add_argument("--preprocessor-config", type=str, default=None, help="Path to a JSON file whose keys override the default image processor config.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4, help="Global batch size across all JAX processes.")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp-size", type=int, default=None)
    p.add_argument("--fsdp-size", type=int, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--profile-dir", type=str, default=None)
    p.add_argument("--profile-start", type=int, default=3, help="Step to start profiling (after JIT warmup).")
    p.add_argument("--profile-end", type=int, default=8, help="Step to stop profiling.")
    p.add_argument("--log-jsonl", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pad-id", type=int, default=0)
    p.add_argument("--peak-tflops", type=str, default=None)
    p.add_argument("--log-image-sizes", action="store_true", help="Print original and resized image dimensions for the first batch.")
    p.add_argument("--tensorboard-dir", type=str, default=None, help="Directory for TensorBoard event files.")
    p.add_argument("--val-data-path", type=str, default=None, help="Path to a compiled Grain validation chunk-index dataset.")
    p.add_argument("--val-every", type=int, default=None, help="Run validation every N training steps.")
    p.add_argument("--val-steps", type=int, default=10, help="Number of batches per validation run.")
    p.add_argument("--grain-read-threads", type=int, default=16)
    p.add_argument("--grain-read-buffer-size", type=int, default=500)
    p.add_argument("--grain-workers", type=int, default=0)
    p.add_argument("--grain-worker-buffer-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.distributed.initialize()

    repo_id = args.processor or resolve_hf_repo_id(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    assert args.max_length <= tokenizer.model_max_length, f"--max-length={args.max_length} exceeds tokenizer.model_max_length={tokenizer.model_max_length}"

    ip_kwargs: dict = {}
    if args.preprocessor_config:
        with open(args.preprocessor_config) as f:
            ip_kwargs = json.load(f)
    image_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False, **ip_kwargs)
    collator = VLMSFTCollator(tokenizer, max_length=args.max_length, image_processor=image_processor)
    per_process_batch = process_local_batch_size(args.batch_size)
    if jax.process_index() == 0:
        print(
            f"global_batch_size={args.batch_size} process_count={jax.process_count()} "
            f"per_process_batch_size={per_process_batch}"
        )

    data_iter = _grain_iter(
        args.data_path,
        collator,
        per_process_batch,
        shuffle=True,
        seed=args.seed,
        grain_read_threads=args.grain_read_threads,
        grain_read_buffer_size=args.grain_read_buffer_size,
        grain_workers=args.grain_workers,
        grain_worker_buffer_size=args.grain_worker_buffer_size,
        num_batches=args.num_steps,
    )

    val_data_iter = None
    if args.val_data_path:
        val_data_iter = _grain_iter(
            args.val_data_path,
            collator,
            per_process_batch,
            shuffle=False,
            seed=args.seed,
            grain_read_threads=args.grain_read_threads,
            grain_read_buffer_size=args.grain_read_buffer_size,
            grain_workers=args.grain_workers,
            grain_worker_buffer_size=args.grain_worker_buffer_size,
            num_batches=max(1, (args.num_steps // max(args.val_every or args.num_steps, 1)) * args.val_steps),
        )

    train_cfg = vlm_trainer.TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.max_length,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        print_every=args.log_every,
    )
    save_dir = Path(args.save_dir) if args.save_dir else (
        _default_save_dir(args.model_id) if args.save_every > 0 or args.resume else None
    )
    peak_tflops = resolve_peak_tflops(args.peak_tflops)

    tb_writer = None
    if args.tensorboard_dir and jax.process_index() == 0:
        tb_dir = Path(args.tensorboard_dir)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))

    try:
        _, last_metrics = vlm_trainer.run_sft(
            args.model_id,
            train_cfg,
            data_iter,
            save_dir=save_dir,
            save_every=args.save_every,
            log_every=args.log_every,
            log_jsonl=args.log_jsonl,
            resume=args.resume,
            pad_id=args.pad_id,
            peak_tflops=peak_tflops,
            tp_size=args.tp_size,
            fsdp_size=args.fsdp_size,
            profile_dir=args.profile_dir,
            profile_steps=(args.profile_start, args.profile_end),
            tb_writer=tb_writer,
            val_data_iter=val_data_iter,
            val_every=args.val_every,
            val_steps=args.val_steps,
        )
    finally:
        if tb_writer is not None:
            tb_writer.close()

    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}")

if __name__ == "__main__":
    main()
