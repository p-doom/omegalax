"""VLM SFT training from a compiled Grain dataset (text-only or multimodal)."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from absl import app, flags
import jax
import wandb
from transformers import AutoImageProcessor, AutoTokenizer

import omegalax.compat.cudnn_ampere_packed  # noqa: F401

from omegalax.data.collator_qwen3 import VLMSFTCollator
from omegalax.data.grain_pipeline import (
    MixSource,
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    required_epochs_for_batches,
)
from omegalax.distributed.mesh import process_local_batch_size
from omegalax.trainers import vlm as vlm_trainer
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.registry import resolve_hf_repo_id
from omegalax.trainers.text import startup_log
from omegalax.trainers.perf import resolve_peak_tflops

FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", None, "HF model id.")
flags.DEFINE_string("data_path", None, "Path to compiled Grain chunk-index dataset directory.")
flags.DEFINE_string(
    "data_mix",
    None,
    'JSON list of {"path", "weight"} pairs to mix at the configured ratios, e.g. '
    '\'[{"path":"/vlm","weight":0.7},{"path":"/instruct","weight":0.3}]\'. '
    "Use this OR --data_path, not both. Mixed sources may freely combine "
    "multimodal and text-only datasets — heterogeneous batches are handled "
    "by the VLM collator and forward path.",
)
flags.DEFINE_string(
    "processor", None, "HF repo to read tokenizer and image config from (defaults to --model_id)."
)
flags.DEFINE_string(
    "preprocessor_config",
    None,
    "Path to JSON file whose keys override default image processor config.",
)
flags.DEFINE_integer("max_length", None, "Maximum sequence length.")
flags.DEFINE_integer("num_steps", None, "Number of training steps.")
flags.DEFINE_integer("batch_size", None, "Global batch size across all JAX processes.")
flags.DEFINE_float("learning_rate", None, "Learning rate.")
flags.DEFINE_float("weight_decay", None, "Weight decay.")
flags.DEFINE_integer("warmup_steps", None, "Linear LR warmup steps.")
flags.DEFINE_enum(
    "lr_schedule",
    None,
    ["linear", "cosine", "wsd"],
    "LR schedule after warmup: 'linear' (constant), 'cosine', or 'wsd' (warmup-stable-decay).",
)
flags.DEFINE_float(
    "lr_end_factor",
    None,
    "Final LR as fraction of peak LR (cosine/wsd decay end value). Required for cosine/wsd.",
)
flags.DEFINE_float(
    "lr_stable_fraction",
    None,
    "Fraction of post-warmup steps at peak LR (wsd only). Required for wsd.",
)
flags.DEFINE_float("max_grad_norm", None, "Max gradient norm for clipping (0 = no clipping).")
flags.DEFINE_integer("grad_accum_steps", None, "Gradient accumulation steps (1 = no accumulation).")
flags.DEFINE_integer(
    "gc_period", None, "If >0, disable Python GC and collect every N training steps."
)
flags.DEFINE_integer("seed", None, "RNG seed.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", None, "FSDP parallelism size.")
flags.DEFINE_integer("dp_size", None, "Data parallelism size.")
flags.DEFINE_string("save_dir", None, "Checkpoint save directory.")
flags.DEFINE_string("jax_cache_dir", None, "Directory for JAX persistent compilation cache.")
flags.DEFINE_string(
    "tokamax_cache_dir",
    None,
    "Directory for the persistent tokamax autotuning cache. If unset, autotuning runs "
    "every launch with no persistence.",
)
flags.DEFINE_integer("save_every", None, "Save checkpoint every N steps.")
flags.DEFINE_integer(
    "keep_period",
    None,
    "Permanently retain every checkpoint whose step is a multiple of this value "
    "(0 = keep all). Must be a multiple of --save_every to ever fire (the loop only "
    "saves at multiples of --save_every).",
)
flags.DEFINE_integer(
    "keep_latest",
    None,
    "Also retain the N most-recent checkpoints regardless of --keep_period.",
)
flags.DEFINE_integer("log_every", None, "Log metrics every N steps.")
flags.DEFINE_bool("log_memory", None, "Log per-process JAX/HBM memory at init and first few steps.")
flags.DEFINE_enum(
    "resume",
    None,
    [m.value for m in ResumeMode],
    "Checkpoint resume policy: 'never' (fresh start), 'if_present' "
    "(resume if a checkpoint exists at --save_dir, else start fresh — right "
    "mode for SLURM time-limit resubmits), 'required' (resume; error if no "
    "checkpoint).",
)
flags.DEFINE_integer("pad_id", None, "Padding token id.")
flags.DEFINE_string("peak_tflops", None, "Peak TFLOPS for MFU calculation.")
flags.DEFINE_string("wandb_entity", None, "Weights & Biases entity (team/user).")
flags.DEFINE_string("wandb_project", None, "Weights & Biases project name (opt-in gate for wandb).")
flags.DEFINE_string("wandb_group", None, "Weights & Biases run group.")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name.")
flags.DEFINE_list("wandb_tags", None, "Comma-separated Weights & Biases tags.")
flags.DEFINE_string("val_data_path", None, "Path to compiled Grain validation chunk-index dataset.")
flags.DEFINE_integer("val_every", None, "Run validation every N training steps.")
flags.DEFINE_integer("val_steps", None, "Number of batches per validation run.")
flags.DEFINE_integer("grain_read_threads", None, "Grain read threads.")
flags.DEFINE_integer("grain_read_buffer_size", None, "Grain read buffer size (in batches).")
flags.DEFINE_integer("grain_workers", None, "Grain multiprocessing workers.")
flags.DEFINE_integer("grain_worker_buffer_size", None, "Grain worker buffer size.")
flags.DEFINE_integer(
    "max_vision_patches_per_sample",
    None,
    "Max vision patches per sample for JIT stability (0 = no padding). "
    "Multiplied by batch_size automatically.",
)
flags.DEFINE_integer(
    "max_vision_images_per_sample",
    None,
    "Max images per sample for JIT stability (0 = no padding). "
    "Multiplied by batch_size automatically.",
)
flags.DEFINE_boolean(
    "enable_lora",
    None,
    "Enable LoRA adapters on the text decoder's q/k/v/o + "
    "gate/up/down projections. Vision tower, embedder, "
    "lm_head and layernorms remain fully frozen.",
)
flags.DEFINE_integer("lora_rank", None, "LoRA rank (required if --enable_lora).")
flags.DEFINE_float(
    "lora_alpha",
    None,
    "LoRA alpha scaling (required if --enable_lora). Effective LR multiplier is alpha/rank.",
)
flags.DEFINE_boolean(
    "freeze_vision_tower",
    None,
    "Full FT on text decoder + embedder + lm_head + "
    "layernorms while freezing the vision tower at the "
    "gradient/opt-state layer. Mutually exclusive with "
    "--enable_lora (which already freezes vision).",
)
flags.DEFINE_integer(
    "num_loss_tiles",
    None,
    "Number of tiles for chunked cross-entropy along the "
    "sequence axis. Must evenly divide (max_length - 1).",
)

_ATTN_BACKENDS = [
    "mosaic_tpu",
    "mosaic_gpu",
    "cudnn",
    "xla",
    "triton",
]
flags.DEFINE_enum(
    "text_attn_backend", None, _ATTN_BACKENDS, "Attention backend for the text decoder."
)

_REQUIRED = [
    "model_id",
    "max_length",
    "num_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "lr_schedule",
    "max_grad_norm",
    "grad_accum_steps",
    "gc_period",
    "seed",
    "tp_size",
    "fsdp_size",
    "dp_size",
    "save_dir",
    "jax_cache_dir",
    "save_every",
    "keep_period",
    "keep_latest",
    "log_every",
    "log_memory",
    "resume",
    "pad_id",
    "peak_tflops",
    "grain_read_threads",
    "grain_read_buffer_size",
    "grain_workers",
    "grain_worker_buffer_size",
    "max_vision_patches_per_sample",
    "max_vision_images_per_sample",
    "num_loss_tiles",
    "text_attn_backend",
    "enable_lora",
    "freeze_vision_tower",
]


def _validate_flags() -> None:
    """Fail loudly at startup if any required flag is unset, listing every problem at once.

    The recipe TOML is the single source of truth, so a forgotten key must error rather than
    silently fall back to a default. Hard-required flags (`_REQUIRED`) must always be present;
    feature-gated flags are required only when their feature is enabled.
    """
    problems: list[str] = []

    for name in _REQUIRED:
        if FLAGS[name].value is None:
            problems.append(name)

    # Exactly one data source.
    if (FLAGS.data_path is None) == (FLAGS.data_mix is None):
        problems.append("exactly one of {data_path, data_mix} (got neither or both)")

    # enable_lora / freeze_vision_tower are mutually exclusive (both freeze the vision tower).
    if FLAGS.enable_lora and FLAGS.freeze_vision_tower:
        problems.append("enable_lora and freeze_vision_tower are mutually exclusive")

    # LoRA hyperparameters required only when LoRA is on.
    if FLAGS.enable_lora:
        for name in ("lora_rank", "lora_alpha"):
            if FLAGS[name].value is None:
                problems.append(f"{name} (required when enable_lora=true)")

    # Validation cadence required only when a validation set is configured.
    if FLAGS.val_data_path:
        for name in ("val_every", "val_steps"):
            if FLAGS[name].value is None:
                problems.append(f"{name} (required when val_data_path is set)")

    # LR-schedule shape parameters required only for the schedules that use them.
    if FLAGS.lr_schedule in ("cosine", "wsd") and FLAGS.lr_end_factor is None:
        problems.append(f"lr_end_factor (required when lr_schedule={FLAGS.lr_schedule})")
    if FLAGS.lr_schedule == "wsd" and FLAGS.lr_stable_fraction is None:
        problems.append("lr_stable_fraction (required when lr_schedule=wsd)")

    # Weights & Biases is opt-in via wandb_project; if on, identifying fields are required.
    if FLAGS.wandb_project:
        for name in ("wandb_entity", "wandb_group", "wandb_name"):
            if FLAGS[name].value is None:
                problems.append(f"{name} (required when wandb_project is set)")

    if problems:
        raise ValueError(
            "Missing or invalid required flags (the recipe TOML must set these):\n  "
            + "\n  ".join(problems)
        )


def _parse_data_mix(spec: str) -> list[MixSource]:
    """Parse the --data_mix JSON spec into a list of MixSource."""
    raw = json.loads(spec)
    if not isinstance(raw, list) or not raw:
        raise ValueError("--data_mix must be a non-empty JSON list of {path, weight} objects")
    out: list[MixSource] = []
    for entry in raw:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"--data_mix entry must be an object with a 'path' field: {entry!r}")
        out.append(MixSource(path=str(entry["path"]), weight=float(entry.get("weight", 1.0))))
    return out


def _resolve_train_sources() -> list[MixSource]:
    if (FLAGS.data_path is None) == (FLAGS.data_mix is None):
        raise ValueError("Specify exactly one of --data_path or --data_mix.")
    if FLAGS.data_mix is not None:
        return _parse_data_mix(FLAGS.data_mix)
    return [MixSource(path=FLAGS.data_path, weight=1.0)]


def _grain_iter(
    sources: list[MixSource],
    collator: VLMSFTCollator,
    per_process_batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    num_batches: int,
    dp_size: int,
    fsdp_size: int,
):
    if len(sources) == 1:
        num_epochs: int | None = required_epochs_for_batches(
            sources[0].path,
            batch_size=per_process_batch_size,
            num_batches=num_batches,
            dp_size=dp_size,
            fsdp_size=fsdp_size,
        )
    else:
        num_epochs = None
    return make_grain_iterator(
        sources,
        batch_size=per_process_batch_size,
        batch_fn=collator,
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        read_options=make_grain_read_options(
            num_threads=FLAGS.grain_read_threads,
            prefetch_buffer_size=FLAGS.grain_read_buffer_size,
        ),
        multiprocessing_options=make_grain_multiprocessing_options(
            num_workers=FLAGS.grain_workers,
            per_worker_buffer_size=FLAGS.grain_worker_buffer_size,
        ),
        dp_size=dp_size,
        fsdp_size=fsdp_size,
    )


def main(_) -> None:
    _validate_flags()
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    jax.distributed.initialize()
    startup_log(f"jax_compilation_cache_dir={FLAGS.jax_cache_dir}")
    startup_log("jax.distributed initialized")

    repo_id = FLAGS.processor or resolve_hf_repo_id(FLAGS.model_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    startup_log(f"loaded tokenizer from {repo_id!r}")
    assert FLAGS.max_length <= tokenizer.model_max_length, (
        f"--max_length={FLAGS.max_length} exceeds tokenizer.model_max_length={tokenizer.model_max_length}"
    )

    ip_kwargs: dict = {}
    if FLAGS.preprocessor_config:
        with open(FLAGS.preprocessor_config) as f:
            ip_kwargs = json.load(f)
    image_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False, **ip_kwargs)
    startup_log(f"loaded image processor from {repo_id!r}")

    if FLAGS.max_vision_patches_per_sample:
        merge_size = int(image_processor.merge_size)
        ms2 = merge_size * merge_size
        # The budget is enforced per sample, not per batch: every batch row owns
        # an equally sized block of pixel_values, which is what lets the model
        # pair each image token with its own embedding.
        if FLAGS.max_vision_patches_per_sample % ms2 != 0:
            raise ValueError(
                f"max_vision_patches_per_sample={FLAGS.max_vision_patches_per_sample} "
                f"must be divisible by merge_size**2={ms2} (remainder "
                f"{FLAGS.max_vision_patches_per_sample % ms2}); each sample is "
                f"padded to exactly that many patches."
            )

    collator = VLMSFTCollator(
        tokenizer,
        max_length=FLAGS.max_length,
        image_processor=image_processor,
        max_vision_patches_per_sample=FLAGS.max_vision_patches_per_sample or None,
        max_vision_images_per_sample=FLAGS.max_vision_images_per_sample or None,
    )
    startup_log("built VLMSFTCollator")
    train_sources = _resolve_train_sources()
    per_process_batch = process_local_batch_size(
        FLAGS.batch_size,
        dp_size=FLAGS.dp_size,
        fsdp_size=FLAGS.fsdp_size,
    )
    sources_repr = ", ".join(f"{s.path}@{s.weight:g}" for s in train_sources)
    startup_log(
        f"model_id={FLAGS.model_id!r} data_sources=[{sources_repr}] "
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
        train_sources,
        collator,
        per_process_batch,
        shuffle=True,
        seed=FLAGS.seed,
        num_batches=total_micro_batches,
        dp_size=FLAGS.dp_size,
        fsdp_size=FLAGS.fsdp_size,
    )
    startup_log("built train grain DataLoader iterator")

    val_data_iter = None
    if FLAGS.val_data_path:
        val_data_iter = _grain_iter(
            [MixSource(path=FLAGS.val_data_path, weight=1.0)],
            collator,
            per_process_batch,
            shuffle=False,
            seed=FLAGS.seed,
            num_batches=max(
                1, (FLAGS.num_steps // max(FLAGS.val_every or FLAGS.num_steps, 1)) * FLAGS.val_steps
            ),
            dp_size=FLAGS.dp_size,
            fsdp_size=FLAGS.fsdp_size,
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
        enable_lora=FLAGS.enable_lora,
        lora_rank=FLAGS.lora_rank,
        lora_alpha=FLAGS.lora_alpha,
        freeze_vision_tower=FLAGS.freeze_vision_tower,
        num_loss_tiles=FLAGS.num_loss_tiles,
    )
    resume_mode = ResumeMode(FLAGS.resume)
    save_dir = Path(FLAGS.save_dir)
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
        startup_log(
            f"gc_period={FLAGS.gc_period}: Python GC disabled, will collect every {FLAGS.gc_period} steps"
        )

    try:
        _, last_metrics = vlm_trainer.run_sft(
            FLAGS.model_id,
            train_cfg,
            data_iter,
            save_dir=save_dir,
            save_every=FLAGS.save_every,
            keep_period=FLAGS.keep_period,
            keep_latest=FLAGS.keep_latest,
            log_every=FLAGS.log_every,
            resume=resume_mode,
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
            log_memory=FLAGS.log_memory,
            tokamax_cache_dir=FLAGS.tokamax_cache_dir,
        )
    finally:
        if FLAGS.gc_period:
            gc.enable()
            print("Training completed, re-enabling Python GC")

        if wandb_run is not None:
            wandb_run.finish()

    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}")


if __name__ == "__main__":
    app.run(main)
