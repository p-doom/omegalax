"""Text-model SFT training from a compiled Grain dataset."""

from __future__ import annotations

import gc
from pathlib import Path

import jax
import wandb
from absl import app, flags
from transformers import AutoConfig, AutoTokenizer

from omegalax.data.collator_qwen3 import TextSFTCollator
from omegalax.data.grain_pipeline import (
    MixSource,
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    parse_data_mix,
    required_epochs_for_batches,
)
from omegalax.distributed.mesh import process_local_batch_size
from omegalax.registry import resolve_hf_repo_id
from omegalax.trainers import text as text_trainer
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.trainers.perf import resolve_peak_tflops
from omegalax.trainers.text import startup_log

FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", None, "HF model id.")
flags.DEFINE_string("data_path", None, "Path to compiled Grain chunk-index dataset directory.")
flags.DEFINE_string(
    "data_mix",
    None,
    'JSON list of {"path", "weight"} pairs to mix at the configured ratios, e.g. '
    '\'[{"path":"/a","weight":0.7},{"path":"/b","weight":0.3}]\'. '
    "Use this OR --data_path, not both.",
)
flags.DEFINE_string("tokenizer", None, "HF tokenizer name/path (defaults to --model_id).")
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
flags.DEFINE_string(
    "save_dir",
    None,
    "Checkpoint save directory. Required unless --save_every=0 and --resume=never.",
)
flags.DEFINE_string("jax_cache_dir", None, "Directory for JAX persistent compilation cache.")
flags.DEFINE_integer("save_every", None, "Save checkpoint every N steps.")
flags.DEFINE_integer("log_every", None, "Log metrics every N steps.")
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
    "jax_cache_dir",
    "save_every",
    "log_every",
    "resume",
    "pad_id",
    "peak_tflops",
    "grain_read_threads",
    "grain_read_buffer_size",
    "grain_workers",
    "grain_worker_buffer_size",
    "text_attn_backend",
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

    if (FLAGS.data_path is None) == (FLAGS.data_mix is None):
        problems.append("exactly one of {data_path, data_mix} (got neither or both)")

    # Checkpointing is what needs a directory; a benchmark run with save_every=0
    # and resume=never writes nothing and must not be forced to invent a path.
    if FLAGS.save_dir is None and (
        FLAGS.save_every or ResumeMode(FLAGS.resume) is not ResumeMode.NEVER
    ):
        problems.append("save_dir (required unless save_every=0 and resume=never)")

    if FLAGS.val_data_path:
        for name in ("val_every", "val_steps"):
            if FLAGS[name].value is None:
                problems.append(f"{name} (required when val_data_path is set)")

    if FLAGS.lr_schedule in ("cosine", "wsd") and FLAGS.lr_end_factor is None:
        problems.append(f"lr_end_factor (required when lr_schedule={FLAGS.lr_schedule})")
    if FLAGS.lr_schedule == "wsd" and FLAGS.lr_stable_fraction is None:
        problems.append("lr_stable_fraction (required when lr_schedule=wsd)")

    if FLAGS.wandb_project:
        for name in ("wandb_entity", "wandb_group", "wandb_name"):
            if FLAGS[name].value is None:
                problems.append(f"{name} (required when wandb_project is set)")

    if problems:
        raise ValueError(
            "Missing or invalid required flags (the recipe TOML must set these):\n  "
            + "\n  ".join(problems)
        )


def _resolve_train_sources() -> list[MixSource]:
    if FLAGS.data_mix is not None:
        return parse_data_mix(FLAGS.data_mix)
    return [MixSource(path=FLAGS.data_path, weight=1.0)]


def _grain_iter(
    sources: list[MixSource],
    collator: TextSFTCollator,
    per_process_batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    num_batches: int,
    dp_size: int,
    fsdp_size: int,
):
    # For mixed sources, repeat each indefinitely and let the trainer's
    # step budget terminate. For a single source, keep the legacy bounded
    # epoch count so deterministic-resume tests keep working.
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

    tokenizer_name = FLAGS.tokenizer or resolve_hf_repo_id(FLAGS.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_type = AutoConfig.from_pretrained(resolve_hf_repo_id(FLAGS.model_id)).model_type
    startup_log(f"loaded tokenizer from {tokenizer_name!r}")
    assert FLAGS.max_length <= tokenizer.model_max_length, (
        f"--max_length={FLAGS.max_length} exceeds tokenizer.model_max_length={tokenizer.model_max_length}"
    )
    collator = TextSFTCollator(tokenizer, max_length=FLAGS.max_length, model_type=model_type)
    startup_log("built TextSFTCollator")
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

    train_cfg = text_trainer.TrainConfig(
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
    resume_mode = ResumeMode(FLAGS.resume)
    save_dir = Path(FLAGS.save_dir) if FLAGS.save_dir else None
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
        _, last_metrics = text_trainer.run_sft(
            FLAGS.model_id,
            train_cfg,
            data_iter,
            save_dir=save_dir,
            save_every=FLAGS.save_every,
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
