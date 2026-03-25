"""Text-model SFT training from a compiled Grain dataset."""

from __future__ import annotations

from pathlib import Path

from absl import app, flags
import jax
from transformers import AutoTokenizer

from omegalax.data.collator_qwen3 import TextSFTCollator
from omegalax.data.grain_pipeline import (
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    required_epochs_for_batches,
)
from omegalax.distributed.mesh import process_local_batch_size
from omegalax.registry import resolve_hf_repo_id
from omegalax.trainers import text as text_trainer
from omegalax.trainers.perf import resolve_peak_tflops
from omegalax.trainers.text import startup_log

FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", None, "HF model id.", required=True)
flags.DEFINE_string("data_path", None, "Path to compiled Grain chunk-index dataset directory.", required=True)
flags.DEFINE_string("tokenizer", None, "HF tokenizer name/path (defaults to --model_id).")
flags.DEFINE_integer("max_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("num_steps", 100, "Number of training steps.")
flags.DEFINE_integer("batch_size", 8, "Global batch size across all JAX processes.")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.01, "Weight decay.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", None, "FSDP parallelism size.")
flags.DEFINE_string("save_dir", None, "Checkpoint save directory.")
flags.DEFINE_string("jax_cache_dir", "/tmp/jax_cache", "Directory for JAX persistent compilation cache.")
flags.DEFINE_integer("save_every", 50, "Save checkpoint every N steps.")
flags.DEFINE_integer("log_every", 10, "Log metrics every N steps.")
flags.DEFINE_string("profile_dir", None, "Directory for JAX profiling output.")
flags.DEFINE_integer("profile_start", 3, "Step to start profiling (after JIT warmup).")
flags.DEFINE_integer("profile_end", 8, "Step to stop profiling.")
flags.DEFINE_bool("resume", False, "Resume from latest checkpoint.")
flags.DEFINE_integer("pad_id", 0, "Padding token id.")
flags.DEFINE_string("peak_tflops", None, "Peak TFLOPS for MFU calculation.")
flags.DEFINE_integer("grain_read_threads", 16, "Grain read threads.")
flags.DEFINE_integer("grain_read_buffer_size", 500, "Grain read buffer size.")
flags.DEFINE_integer("grain_workers", 0, "Grain multiprocessing workers.")
flags.DEFINE_integer("grain_worker_buffer_size", 1, "Grain worker buffer size.")


def _default_save_dir(model_id: str) -> Path:
    safe_name = model_id.replace("/", "_")
    return Path("runs") / "text_sft" / safe_name


def _grain_iter(
    data_path: str,
    collator: TextSFTCollator,
    per_process_batch_size: int,
    *,
    shuffle: bool,
    seed: int,
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
            num_threads=FLAGS.grain_read_threads,
            prefetch_buffer_size=FLAGS.grain_read_buffer_size,
        ),
        multiprocessing_options=make_grain_multiprocessing_options(
            num_workers=FLAGS.grain_workers,
            per_worker_buffer_size=FLAGS.grain_worker_buffer_size,
        ),
    )


def main(_) -> None:
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    jax.distributed.initialize()
    startup_log(f"jax_compilation_cache_dir={FLAGS.jax_cache_dir}")
    startup_log("jax.distributed initialized")

    tokenizer_name = FLAGS.tokenizer or resolve_hf_repo_id(FLAGS.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    startup_log(f"loaded tokenizer from {tokenizer_name!r}")
    assert FLAGS.max_length <= tokenizer.model_max_length, f"--max_length={FLAGS.max_length} exceeds tokenizer.model_max_length={tokenizer.model_max_length}"
    collator = TextSFTCollator(tokenizer, max_length=FLAGS.max_length)
    startup_log("built TextSFTCollator")
    per_process_batch = process_local_batch_size(FLAGS.batch_size)
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
    data_iter = _grain_iter(
        FLAGS.data_path,
        collator,
        per_process_batch,
        shuffle=True,
        seed=FLAGS.seed,
        num_batches=FLAGS.num_steps,
    )
    startup_log("built train grain DataLoader iterator")

    train_cfg = text_trainer.TrainConfig(
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.max_length,
        num_steps=FLAGS.num_steps,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        print_every=FLAGS.log_every,
    )
    save_dir = Path(FLAGS.save_dir) if FLAGS.save_dir else (
        _default_save_dir(FLAGS.model_id) if FLAGS.save_every > 0 or FLAGS.resume else None
    )
    peak_tflops = resolve_peak_tflops(FLAGS.peak_tflops)

    _, last_metrics = text_trainer.run_sft(
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
        profile_dir=FLAGS.profile_dir,
        profile_steps=(FLAGS.profile_start, FLAGS.profile_end),
    )
    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}")


if __name__ == "__main__":
    app.run(main)
