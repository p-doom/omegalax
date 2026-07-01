"""Run IID or 2-segment statepassing text pretraining."""

from __future__ import annotations

import gc
from pathlib import Path

from absl import app, flags
import jax
import jax.numpy as jnp
import wandb

from omegalax.data.pretrain_data_set import DEFAULT_CHUNK_LENGTH, DEFAULT_EOS_ID
from omegalax.data.pretrain_iid_pipeline import make_iid_iterator
from omegalax.data.pretrain_statepassing import make_statepassing_iterator
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.trainers import pretrain as pretrain_trainer
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.trainers.perf import resolve_peak_tflops
from omegalax.trainers.text import startup_log

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "pretrain_mode",
    pretrain_trainer.PretrainMode.IID_BASELINE.value,
    [mode.value for mode in pretrain_trainer.PretrainMode],
    "Pretraining experiment mode.",
)
flags.DEFINE_string("model_id", None, "Optional supported/local text model config source.")
flags.DEFINE_string("tokenizer", "Qwen/Qwen3.5-0.8B", "Fixed tokenizer name for run metadata.")
flags.DEFINE_string("train_index_path", None, "Prebuilt train index path.", required=True)
flags.DEFINE_string("val_index_path", None, "Optional prebuilt validation index path.")
flags.DEFINE_integer("seq_len", DEFAULT_CHUNK_LENGTH, "Segment length.")
flags.DEFINE_integer("batch_size", 8, "Global number of 4096-token segments per microstep.")
flags.DEFINE_integer("num_steps", None, "Optimizer steps. Overrides --max_tokens if set.")
flags.DEFINE_integer("max_tokens", 15_000_000_000, "Total training token budget.")
flags.DEFINE_integer("warmup_tokens", 150_000_000, "Warmup token budget.")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.1, "Weight decay.")
flags.DEFINE_float("adam_beta1", 0.9, "AdamW beta1.")
flags.DEFINE_float("adam_beta2", 0.95, "AdamW beta2.")
flags.DEFINE_float("adam_eps", 1e-8, "AdamW epsilon.")
flags.DEFINE_enum("lr_schedule", "cosine", ["linear", "cosine", "wsd"], "LR schedule.")
flags.DEFINE_float("lr_end_factor", 0.1, "Final LR as fraction of peak LR.")
flags.DEFINE_float("lr_stable_fraction", 0.8, "Stable fraction for WSD.")
flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm; 0 disables clipping.")
flags.DEFINE_integer("grad_accum_steps", 1, "Gradient accumulation steps.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", 1, "FSDP parallelism size.")
flags.DEFINE_integer("dp_size", 1, "Data parallelism size.")
flags.DEFINE_integer("iterator_fsdp_size", None, "Optional FSDP shard count for data iterator.")
flags.DEFINE_integer("iterator_dp_size", None, "Optional DP shard count for data iterator.")
flags.DEFINE_string("save_dir", None, "Checkpoint save directory.")
flags.DEFINE_string("jax_cache_dir", "/tmp/jax_cache", "JAX persistent compilation cache dir.")
flags.DEFINE_integer("save_every", 100, "Save checkpoint every N steps.")
flags.DEFINE_integer("log_every", 10, "Log metrics every N steps.")
flags.DEFINE_enum(
    "resume",
    ResumeMode.NEVER.value,
    [mode.value for mode in ResumeMode],
    "Checkpoint resume policy.",
)
flags.DEFINE_integer("pad_id", 0, "Padding token id.")
flags.DEFINE_integer("eos_id", DEFAULT_EOS_ID, "EOS id expected by the prebuilt index.")
flags.DEFINE_string("peak_tflops", None, "Peak TFLOPS for MFU calculation.")
flags.DEFINE_string("wandb_entity", None, "Weights & Biases entity.")
flags.DEFINE_string("wandb_project", None, "Weights & Biases project name.")
flags.DEFINE_string("wandb_group", None, "Weights & Biases run group.")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name.")
flags.DEFINE_list("wandb_tags", [], "Comma-separated Weights & Biases tags.")
flags.DEFINE_string("wandb_id", None, "Optional stable Weights & Biases run id for resume.")
flags.DEFINE_string("wandb_resume", None, "Optional Weights & Biases resume policy.")
flags.DEFINE_integer("val_every", None, "Run validation every N steps.")
flags.DEFINE_integer("val_steps", 10, "Validation batches per validation run.")
flags.DEFINE_integer("grain_read_threads", 2, "Grain read threads.")
flags.DEFINE_integer("grain_read_prefetch_buffer_size", 4, "Grain read prefetch buffer size.")
flags.DEFINE_integer("grain_workers", 8, "Grain multiprocessing workers.")
flags.DEFINE_integer("grain_worker_buffer_size", 1, "Grain worker buffer size.")
flags.DEFINE_integer("gc_period", 0, "If >0, collect Python GC every N steps.")
flags.DEFINE_enum(
    "text_attn_backend",
    "mosaic_gpu",
    ["mosaic_tpu", "mosaic_gpu", "cudnn", "xla", "triton"],
    "Attention backend for full-attention layers.",
)


def _default_pretrain_config() -> Qwen3_5TextConfig:
    layer_types = tuple(
        "full_attention" if (idx + 1) % 4 == 0 else "linear_attention" for idx in range(12)
    )
    return Qwen3_5TextConfig(
        vocab_size=248_320,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=128,
        rms_norm_eps=1e-6,
        layer_types=layer_types,
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        tie_word_embeddings=False,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=6,
        linear_num_value_heads=6,
        linear_value_head_dim=128,
        intermediate_size=3072,
        dtype=jnp.bfloat16,
    )


def _default_save_dir(pretrain_mode: pretrain_trainer.PretrainMode) -> Path:
    return Path("runs") / "text_pretrain" / pretrain_mode.value


def _validate_model_sharding(model_source) -> None:
    hidden_size = getattr(model_source, "hidden_size", None)
    if hidden_size is not None and FLAGS.fsdp_size and hidden_size % FLAGS.fsdp_size != 0:
        raise ValueError(f"fsdp_size={FLAGS.fsdp_size} must divide hidden_size={hidden_size}.")


def _num_steps_and_warmup() -> tuple[int, int]:
    tokens_per_step = FLAGS.batch_size * FLAGS.seq_len * FLAGS.grad_accum_steps
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    num_steps = FLAGS.num_steps or max(1, FLAGS.max_tokens // tokens_per_step)
    warmup_steps = max(0, FLAGS.warmup_tokens // tokens_per_step)
    return int(num_steps), int(warmup_steps)


def _make_iterator(
    index_path: str, mode: pretrain_trainer.PretrainMode, per_process_batch: int, *, shuffle: bool
):
    common = dict(
        batch_size=per_process_batch,
        chunk_length=FLAGS.seq_len,
        pad_id=FLAGS.pad_id,
        eos_id=FLAGS.eos_id,
        shuffle=shuffle,
        seed=FLAGS.seed,
        num_epochs=None,
        dp_size=FLAGS.iterator_dp_size if FLAGS.iterator_dp_size is not None else FLAGS.dp_size,
        fsdp_size=(
            FLAGS.iterator_fsdp_size if FLAGS.iterator_fsdp_size is not None else FLAGS.fsdp_size
        ),
        process_index=jax.process_index(),
        grain_workers=FLAGS.grain_workers,
        grain_worker_buffer_size=FLAGS.grain_worker_buffer_size,
        grain_read_threads=FLAGS.grain_read_threads,
        grain_read_prefetch_buffer_size=FLAGS.grain_read_prefetch_buffer_size,
    )
    if mode is pretrain_trainer.PretrainMode.IID_BASELINE:
        return make_iid_iterator(index_path, **common)
    return make_statepassing_iterator(index_path, **common)


def main(_) -> None:
    pretrain_mode = pretrain_trainer.PretrainMode(FLAGS.pretrain_mode)
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    jax.distributed.initialize()
    startup_log("jax.distributed initialized")
    startup_log(f"fixed tokenizer for metadata: {FLAGS.tokenizer!r}")

    if FLAGS.batch_size % jax.process_count():
        raise ValueError(
            f"Global batch size {FLAGS.batch_size} must be divisible by "
            f"process_count={jax.process_count()}."
        )
    per_process_batch = FLAGS.batch_size // jax.process_count()
    if pretrain_mode.is_statepassing and per_process_batch % 2:
        raise ValueError("Per-process segment batch size must be even for statepassing.")

    num_steps, warmup_steps = _num_steps_and_warmup()
    train_iter = _make_iterator(
        FLAGS.train_index_path, pretrain_mode, per_process_batch, shuffle=True
    )
    val_iter = (
        _make_iterator(FLAGS.val_index_path, pretrain_mode, per_process_batch, shuffle=False)
        if FLAGS.val_index_path
        else None
    )
    model_source = FLAGS.model_id or _default_pretrain_config()
    _validate_model_sharding(model_source)
    train_cfg = pretrain_trainer.TrainConfig(
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
        num_steps=num_steps,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        adam_beta1=FLAGS.adam_beta1,
        adam_beta2=FLAGS.adam_beta2,
        adam_eps=FLAGS.adam_eps,
        warmup_steps=warmup_steps,
        lr_schedule=FLAGS.lr_schedule,
        lr_end_factor=FLAGS.lr_end_factor,
        lr_stable_fraction=FLAGS.lr_stable_fraction,
        max_grad_norm=FLAGS.max_grad_norm,
        grad_accum_steps=FLAGS.grad_accum_steps,
        print_every=FLAGS.log_every,
    )
    resume_mode = ResumeMode(FLAGS.resume)
    save_dir = (
        Path(FLAGS.save_dir)
        if FLAGS.save_dir
        else (
            _default_save_dir(pretrain_mode)
            if FLAGS.save_every > 0 or resume_mode is not ResumeMode.NEVER
            else None
        )
    )
    peak_tflops = resolve_peak_tflops(FLAGS.peak_tflops)

    wandb_run = None
    if FLAGS.wandb_project and jax.process_index() == 0:
        wandb_kwargs = {
            "entity": FLAGS.wandb_entity,
            "project": FLAGS.wandb_project,
            "group": FLAGS.wandb_group,
            "name": FLAGS.wandb_name,
            "tags": FLAGS.wandb_tags or None,
            "config": flags.FLAGS.flag_values_dict()
            | {
                "derived_num_steps": num_steps,
                "derived_warmup_steps": warmup_steps,
                "tokenizer": FLAGS.tokenizer,
            },
        }
        if FLAGS.wandb_id:
            wandb_kwargs["id"] = FLAGS.wandb_id
        if FLAGS.wandb_resume:
            wandb_kwargs["resume"] = FLAGS.wandb_resume
        wandb_run = wandb.init(**wandb_kwargs)
    if FLAGS.gc_period:
        gc.disable()

    try:
        _, last_metrics = pretrain_trainer.run_pretrain(
            model_source,
            train_cfg,
            train_iter,
            pretrain_mode=pretrain_mode,
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
            val_data_iter=val_iter,
            val_every=FLAGS.val_every,
            val_steps=FLAGS.val_steps,
            text_attn_backend=FLAGS.text_attn_backend,
            gc_period=FLAGS.gc_period,
        )
    finally:
        if FLAGS.gc_period:
            gc.enable()
        if wandb_run is not None:
            wandb_run.finish()

    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} nll={last_metrics['nll']:.4f}")


if __name__ == "__main__":
    app.run(main)
