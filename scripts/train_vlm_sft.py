"""VLM SFT training from a compiled Grain dataset (text-only or multimodal)."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from absl import app, flags
import jax
import wandb
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
from omegalax.trainers.text import startup_log
from omegalax.trainers.perf import resolve_peak_tflops

FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", None, "HF model id.", required=True)
flags.DEFINE_string("data_path", None, "Path to compiled Grain chunk-index dataset directory.", required=True)
flags.DEFINE_string("processor", None, "HF repo to read tokenizer and image config from (defaults to --model_id).")
flags.DEFINE_string("preprocessor_config", None, "Path to JSON file whose keys override default image processor config.")
flags.DEFINE_integer("max_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("num_steps", 100, "Number of training steps.")
flags.DEFINE_integer("batch_size", 4, "Global batch size across all JAX processes.")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.01, "Weight decay.")
flags.DEFINE_integer("warmup_steps", 0, "Linear LR warmup steps.")
flags.DEFINE_enum("lr_schedule", "linear", ["linear", "cosine", "wsd"],
                  "LR schedule after warmup: 'linear' (constant), 'cosine', or 'wsd' (warmup-stable-decay).")
flags.DEFINE_float("lr_end_factor", 0.0, "Final LR as fraction of peak LR (cosine/wsd decay end value).")
flags.DEFINE_float("lr_stable_fraction", 0.8, "Fraction of post-warmup steps at peak LR (wsd only).")
flags.DEFINE_float("max_grad_norm", 1.0, "Max gradient norm for clipping (0 = no clipping).")
flags.DEFINE_integer("grad_accum_steps", 1, "Gradient accumulation steps (1 = no accumulation).")
flags.DEFINE_integer("gc_period", 0, "If >0, disable Python GC and collect every N training steps.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", None, "FSDP parallelism size.")
flags.DEFINE_integer("dp_size", None, "Data parallelism size.")
flags.DEFINE_string("save_dir", None, "Checkpoint save directory.")
flags.DEFINE_string("jax_cache_dir", "/tmp/jax_cache", "Directory for JAX persistent compilation cache.")
flags.DEFINE_integer("save_every", 50, "Save checkpoint every N steps.")
flags.DEFINE_integer("log_every", 10, "Log metrics every N steps.")
flags.DEFINE_bool("resume", False, "Resume from latest checkpoint.")
flags.DEFINE_integer("pad_id", 0, "Padding token id.")
flags.DEFINE_string("peak_tflops", None, "Peak TFLOPS for MFU calculation.")
flags.DEFINE_string("wandb_entity", None, "Weights & Biases entity (team/user).")
flags.DEFINE_string("wandb_project", None, "Weights & Biases project name.")
flags.DEFINE_string("wandb_group", None, "Weights & Biases run group.")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name.")
flags.DEFINE_list("wandb_tags", [], "Comma-separated Weights & Biases tags.")
flags.DEFINE_string("val_data_path", None, "Path to compiled Grain validation chunk-index dataset.")
flags.DEFINE_integer("val_every", None, "Run validation every N training steps.")
flags.DEFINE_integer("val_steps", 10, "Number of batches per validation run.")
flags.DEFINE_integer("grain_read_threads", 16, "Grain read threads.")
flags.DEFINE_integer("grain_read_buffer_size", 500, "Grain read buffer size.")
flags.DEFINE_integer("grain_workers", 8, "Grain multiprocessing workers.")
flags.DEFINE_integer("grain_worker_buffer_size", 4, "Grain worker buffer size.")
flags.DEFINE_integer("max_vision_patches_per_sample", 0,
                     "Max vision patches per sample for JIT stability (0 = no padding). "
                     "Multiplied by batch_size automatically.")
flags.DEFINE_integer("max_vision_images_per_sample", 0,
                     "Max images per sample for JIT stability (0 = no padding). "
                     "Multiplied by batch_size automatically.")

_ATTN_BACKENDS = [
    "mosaic_tpu", "mosaic_gpu", "cudnn", "xla", "triton",
]
flags.DEFINE_enum("text_attn_backend", "mosaic_gpu", _ATTN_BACKENDS,
                  "Attention backend for the text decoder.")

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
    num_batches: int,
    dp_size: int | None = None,
):
    return make_grain_iterator(
        data_path,
        batch_size=per_process_batch_size,
        batch_fn=collator,
        shuffle=shuffle,
        seed=seed,
        num_epochs=required_epochs_for_batches(
            data_path, batch_size=per_process_batch_size, num_batches=num_batches,
            dp_size=dp_size,
        ),
        read_options=make_grain_read_options(
            num_threads=FLAGS.grain_read_threads,
            prefetch_buffer_size=FLAGS.grain_read_buffer_size,
        ),
        multiprocessing_options=make_grain_multiprocessing_options(
            num_workers=FLAGS.grain_workers,
            per_worker_buffer_size=FLAGS.grain_worker_buffer_size,
        ),
        dp_size=dp_size,
    )


def main(_) -> None:
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    jax.distributed.initialize()
    startup_log(f"jax_compilation_cache_dir={FLAGS.jax_cache_dir}")
    startup_log("jax.distributed initialized")

    repo_id = FLAGS.processor or resolve_hf_repo_id(FLAGS.model_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    startup_log(f"loaded tokenizer from {repo_id!r}")
    assert FLAGS.max_length <= tokenizer.model_max_length, f"--max_length={FLAGS.max_length} exceeds tokenizer.model_max_length={tokenizer.model_max_length}"

    ip_kwargs: dict = {}
    if FLAGS.preprocessor_config:
        with open(FLAGS.preprocessor_config) as f:
            ip_kwargs = json.load(f)
    image_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False, **ip_kwargs)
    startup_log(f"loaded image processor from {repo_id!r}")

    if FLAGS.max_vision_patches_per_sample:
        merge_size = int(image_processor.merge_size)
        ms2 = merge_size * merge_size
        max_patches = FLAGS.max_vision_patches_per_sample * FLAGS.batch_size
        if max_patches % ms2 != 0:
            raise ValueError(
                f"max_vision_patches_per_sample * batch_size = "
                f"{FLAGS.max_vision_patches_per_sample} * {FLAGS.batch_size} "
                f"= {max_patches} must be divisible by merge_size**2={ms2} "
                f"(remainder {max_patches % ms2}). Adjust the flags so their "
                f"product is a multiple of {ms2}."
            )

    collator = VLMSFTCollator(
        tokenizer,
        max_length=FLAGS.max_length,
        image_processor=image_processor,
        max_vision_patches_per_sample=FLAGS.max_vision_patches_per_sample or None,
        max_vision_images_per_sample=FLAGS.max_vision_images_per_sample or None,
    )
    startup_log("built VLMSFTCollator")
    per_process_batch = process_local_batch_size(FLAGS.batch_size, dp_size=FLAGS.dp_size)
    startup_log(
        f"model_id={FLAGS.model_id!r} data_path={FLAGS.data_path!r} "
        f"jax_compilation_cache_dir={FLAGS.jax_cache_dir!r} "
        f"process_count={jax.process_count()} local_device_count={jax.local_device_count()}"
    )
    if jax.process_index() == 0:
        print(
            f"global_batch_size={FLAGS.batch_size} process_count={jax.process_count()} "
            f"per_process_batch_size={per_process_batch}"
        )

    total_micro_batches = FLAGS.num_steps * FLAGS.grad_accum_steps
    data_iter = _grain_iter(
        FLAGS.data_path,
        collator,
        per_process_batch,
        shuffle=True,
        seed=FLAGS.seed,
        num_batches=total_micro_batches,
        dp_size=FLAGS.dp_size,
    )
    startup_log("built train grain DataLoader iterator")

    val_data_iter = None
    if FLAGS.val_data_path:
        val_data_iter = _grain_iter(
            FLAGS.val_data_path,
            collator,
            per_process_batch,
            shuffle=False,
            seed=FLAGS.seed,
            num_batches=max(1, (FLAGS.num_steps // max(FLAGS.val_every or FLAGS.num_steps, 1)) * FLAGS.val_steps),
            dp_size=FLAGS.dp_size,
        )
        startup_log(f"built val grain DataLoader iterator from {FLAGS.val_data_path!r}")

    train_cfg = vlm_trainer.TrainConfig(
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.max_length,
        num_steps=FLAGS.num_steps,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        warmup_steps=FLAGS.warmup_steps,
        lr_schedule=FLAGS.lr_schedule,
        lr_end_factor=FLAGS.lr_end_factor,
        lr_stable_fraction=FLAGS.lr_stable_fraction,
        max_grad_norm=FLAGS.max_grad_norm,
        grad_accum_steps=FLAGS.grad_accum_steps,
        print_every=FLAGS.log_every,
    )
    save_dir = Path(FLAGS.save_dir) if FLAGS.save_dir else (
        _default_save_dir(FLAGS.model_id) if FLAGS.save_every > 0 or FLAGS.resume else None
    )
    peak_tflops = resolve_peak_tflops(FLAGS.peak_tflops)

    wandb_run = None
    if FLAGS.wandb_project and jax.process_index() == 0:
        wandb_run = wandb.init(
            entity=FLAGS.wandb_entity,
            project=FLAGS.wandb_project,
            group=FLAGS.wandb_group,
            name=FLAGS.wandb_name,
            tags=FLAGS.wandb_tags or None,
            config=flags.FLAGS.flag_values_dict(),
        )
    if FLAGS.gc_period:
        gc.disable()
        startup_log(f"gc_period={FLAGS.gc_period}: Python GC disabled, will collect every {FLAGS.gc_period} steps")

    try:
        _, last_metrics = vlm_trainer.run_sft(
            FLAGS.model_id,
            train_cfg,
            data_iter,
            save_dir=save_dir,
            save_every=FLAGS.save_every,
            log_every=FLAGS.log_every,
            resume=FLAGS.resume,
            pad_id=FLAGS.pad_id,
            peak_tflops=peak_tflops,
            tp_size=FLAGS.tp_size,
            fsdp_size=FLAGS.fsdp_size,
            dp_size=FLAGS.dp_size,
            wandb_run=wandb_run,
            val_data_iter=val_data_iter,
            val_every=FLAGS.val_every,
            val_steps=FLAGS.val_steps,
            text_attn_backend=FLAGS.text_attn_backend,
            gc_period=FLAGS.gc_period,
        )
    finally:

        if FLAGS.gc_period:
            gc.enable()
            print(f"Training completed, re-enabling Python GC")

        if wandb_run is not None:
            wandb_run.finish()

    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}")

if __name__ == "__main__":
    app.run(main)
