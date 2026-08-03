"""Run the additive fixed-C tail-training experiment path."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from absl import app, flags
import jax
import wandb

from omegalax.data.pretrain_iid_pipeline import make_iid_iterator
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_FIXED_C_INDEX_FORMAT,
    STATEPASSING_WINDOW_INDEX_FORMAT,
    make_statepassing_iterator,
)
from omegalax.training_contract import build_training_contract, ensure_training_contract
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.trainers.perf import resolve_peak_tflops
from omegalax.trainers.pretrain import GdnStateRouting, PretrainMode, validate_gdn_state_routing
from omegalax.trainers.tail_pretrain import TailTrainConfig, run_tail_pretrain
from omegalax.trainers.text import startup_log

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "pretrain_mode", PretrainMode.IID_BASELINE.value, [m.value for m in PretrainMode], "Tail mode."
)
flags.DEFINE_string(
    "train_index_path",
    None,
    "Fixed-C tail bundle root or direct Statepassing C=4 window index.",
    required=True,
)
flags.DEFINE_string("init_model_checkpoint", None, "Model-only source checkpoint.")
flags.DEFINE_string(
    "init_hf_model_path", None, "Local HuggingFace model used for fresh initialization."
)
flags.DEFINE_string("save_dir", None, "New tail checkpoint root.", required=True)
flags.DEFINE_integer("seq_len", 2048, "Segment length.")
flags.DEFINE_integer("batch_size", 192, "Global segment batch per microstep.")
flags.DEFINE_integer("grad_accum_steps", 5, "Gradient accumulation microsteps.")
flags.DEFINE_integer("global_step_offset", 40000, "Global step before the first tail update.")
flags.DEFINE_float("start_lr", None, "First local tail-update LR for cosine_bridge.")
flags.DEFINE_float("end_lr", None, "Last local tail-update LR for cosine_bridge.")
flags.DEFINE_integer("max_tokens", None, "Optional maximum number of training tokens.")
flags.DEFINE_enum(
    "lr_schedule",
    "cosine_bridge",
    ["cosine_bridge", "wsd"],
    "Tail LR schedule.",
)
flags.DEFINE_float("learning_rate", None, "Peak LR for WSD.")
flags.DEFINE_integer("warmup_tokens", 0, "Warmup token budget for WSD.")
flags.DEFINE_float("lr_end_factor", 0.0, "Final LR fraction for WSD.")
flags.DEFINE_float("lr_stable_fraction", 0.8, "Post-warmup peak-LR fraction for WSD.")
flags.DEFINE_float("weight_decay", 0.1, "AdamW weight decay.")
flags.DEFINE_float("adam_beta1", 0.9, "AdamW beta1.")
flags.DEFINE_float("adam_beta2", 0.95, "AdamW beta2.")
flags.DEFINE_float("adam_eps", 1e-8, "AdamW epsilon.")
flags.DEFINE_float("max_grad_norm", 1.0, "Global gradient clipping norm.")
flags.DEFINE_integer("seed", 0, "Training and data seed.")
flags.DEFINE_bool("enable_lora", False, "Train only the fixed mixed-rank LoRA profile.")
flags.DEFINE_integer("lora_adapter_seed", 0, "LoRA adapter initialization seed.")
flags.DEFINE_integer("bptt_chunks", 4, "Statepassing BPTT span.")
flags.DEFINE_bool("pass_gdn_state", True, "Pass GDN state.")
flags.DEFINE_integer("gdn_layer_limit", None, "Limit passed GDN states.")
flags.DEFINE_enum(
    "gdn_state_routing",
    GdnStateRouting.SAME_LAYER.value,
    [routing.value for routing in GdnStateRouting],
    "GDN state routing.",
)
flags.DEFINE_bool("pass_rope_positions", False, "Pass chunk-aware RoPE positions.")
flags.DEFINE_bool("pass_conv_state", True, "Pass causal convolution state.")
flags.DEFINE_integer("pad_id", 0, "Padding token id.")
flags.DEFINE_integer("eos_id", 248046, "Expected tail-index EOS id.")
flags.DEFINE_integer("tp_size", 1, "Tensor parallel size.")
flags.DEFINE_integer("fsdp_size", 1, "FSDP size.")
flags.DEFINE_integer("dp_size", 8, "Data parallel size.")
flags.DEFINE_integer("iterator_fsdp_size", 1, "Iterator FSDP shard count.")
flags.DEFINE_integer("iterator_dp_size", 1, "Iterator DP shard count.")
flags.DEFINE_integer(
    "iterator_excluded_prefix_stride",
    1,
    "Old iterator shard count whose consumed prefix is excluded on resume.",
)
flags.DEFINE_integer(
    "iterator_excluded_prefix_count",
    0,
    "Per-old-shard index records already consumed before a sharding migration.",
)
flags.DEFINE_bool(
    "restore_input_iter",
    True,
    "Restore the saved iterator state; disable only for an explicit iterator migration.",
)
flags.DEFINE_integer("resume_expected_step", None, "Require this exact checkpoint step on resume.")
flags.DEFINE_integer("grain_workers", 8, "Grain multiprocessing workers.")
flags.DEFINE_integer("grain_worker_buffer_size", 1, "Grain worker buffer size.")
flags.DEFINE_integer("grain_read_threads", 2, "Grain read threads.")
flags.DEFINE_integer("grain_read_prefetch_buffer_size", 4, "Grain read prefetch size.")
flags.DEFINE_integer("save_every", 500, "Checkpoint interval in local tail updates.")
flags.DEFINE_integer("keep_latest", 10, "Number of checkpoints retained.")
flags.DEFINE_integer("log_every", 10, "Metric logging interval.")
flags.DEFINE_integer(
    "stop_after_local_step",
    None,
    "Optional local stop used by production smokes; the LR contract keeps the full bundle horizon.",
)
flags.DEFINE_enum(
    "resume", ResumeMode.IF_PRESENT.value, [m.value for m in ResumeMode], "Resume policy."
)
flags.DEFINE_string("text_attn_backend", "mosaic_gpu", "Text attention backend.")
flags.DEFINE_string("jax_cache_dir", "/tmp/jax_cache", "JAX compilation cache.")
flags.DEFINE_string("peak_tflops", None, "Peak TFLOPS specification.")
flags.DEFINE_integer("gc_period", 0, "Disable automatic GC and collect every N steps if positive.")
flags.DEFINE_string("wandb_entity", None, "W&B entity.")
flags.DEFINE_string("wandb_project", None, "W&B project.")
flags.DEFINE_string("wandb_group", None, "W&B group.")
flags.DEFINE_string("wandb_name", None, "W&B run name.")
flags.DEFINE_list("wandb_tags", [], "W&B tags.")
flags.DEFINE_string("wandb_id", None, "Stable W&B run id.")
flags.DEFINE_string("wandb_resume", None, "W&B resume policy.")
flags.DEFINE_string(
    "wandb_history_source_run_id",
    None,
    "Optional read-only W&B source run whose train/loss history is replayed into this new run.",
)
flags.DEFINE_integer(
    "wandb_history_until_step",
    40000,
    "Largest source global step copied into this new W&B run.",
)


def _load_bundle() -> tuple[dict[str, object], Path, bool]:
    root = Path(FLAGS.train_index_path).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Tail bundle has no metadata.json: {root}")
    metadata = json.loads(metadata_path.read_text())
    index_format = metadata.get("format")
    if index_format not in (STATEPASSING_FIXED_C_INDEX_FORMAT, STATEPASSING_WINDOW_INDEX_FORMAT):
        raise ValueError(
            f"Expected a fixed-C bundle or Statepassing window index, got {index_format!r}"
        )
    if int(metadata["num_segments"]) != 4:
        raise ValueError(f"Tail index must use num_segments=4, got {metadata['num_segments']}")
    is_fixed_c_bundle = index_format == STATEPASSING_FIXED_C_INDEX_FORMAT
    if is_fixed_c_bundle:
        index_segments_per_update = int(metadata["trim_batch_size"]) * int(
            metadata["trim_grad_accum_steps"]
        )
        runtime_segments_per_update = int(FLAGS.batch_size) * int(FLAGS.grad_accum_steps)
        if runtime_segments_per_update != index_segments_per_update:
            raise ValueError(
                "Tail bundle segments per optimizer update does not match runtime configuration"
            )
    if int(metadata["chunk_length"]) != int(FLAGS.seq_len):
        raise ValueError("Tail bundle chunk_length does not match --seq_len")
    if metadata.get("eos_id") != FLAGS.eos_id:
        raise ValueError("Tail bundle eos_id does not match --eos_id")
    return metadata, root, is_fixed_c_bundle


def _make_tail_iterator(
    metadata: dict[str, object], root: Path, mode: PretrainMode, *, is_fixed_c_bundle: bool
):
    if FLAGS.batch_size % jax.process_count():
        raise ValueError("Global batch size must be divisible by jax.process_count()")
    per_process_batch = FLAGS.batch_size // jax.process_count()
    if is_fixed_c_bundle:
        train_split = dict(dict(metadata["splits"])["train"])
        if mode is PretrainMode.IID_BASELINE:
            path = root / str(dict(train_split["iid"])["path"])
            make_iterator = make_iid_iterator
        else:
            path = root / str(train_split["path"])
            make_iterator = make_statepassing_iterator
        num_epochs = 1
    else:
        if mode is PretrainMode.IID_BASELINE:
            raise ValueError("A direct Statepassing window index does not provide an IID split")
        path = root
        make_iterator = make_statepassing_iterator
        num_epochs = None
    iterator_kwargs = dict(
        batch_size=per_process_batch,
        chunk_length=FLAGS.seq_len,
        pad_id=FLAGS.pad_id,
        eos_id=FLAGS.eos_id,
        shuffle=True,
        seed=FLAGS.seed,
        num_epochs=num_epochs,
        dp_size=FLAGS.iterator_dp_size,
        fsdp_size=FLAGS.iterator_fsdp_size,
        process_index=jax.process_index(),
        grain_workers=FLAGS.grain_workers,
        grain_worker_buffer_size=FLAGS.grain_worker_buffer_size,
        grain_read_threads=FLAGS.grain_read_threads,
        grain_read_prefetch_buffer_size=FLAGS.grain_read_prefetch_buffer_size,
    )
    if mode is not PretrainMode.IID_BASELINE:
        iterator_kwargs["excluded_prefix_stride"] = FLAGS.iterator_excluded_prefix_stride
        iterator_kwargs["excluded_prefix_count"] = FLAGS.iterator_excluded_prefix_count
    return make_iterator(str(path), **iterator_kwargs)


def _replay_wandb_loss_history(wandb_run) -> None:
    source_run_id = FLAGS.wandb_history_source_run_id
    if not source_run_id:
        return
    if not FLAGS.wandb_entity or not FLAGS.wandb_project:
        raise ValueError("W&B history replay requires --wandb_entity and --wandb_project")

    source = wandb.Api().run(f"{FLAGS.wandb_entity}/{FLAGS.wandb_project}/{source_run_id}")
    replayed = 0
    until_step = int(FLAGS.wandb_history_until_step)
    for row in source.scan_history(keys=["_step", "train/loss"], page_size=1000):
        step = row.get("_step")
        loss = row.get("train/loss")
        if step is None or loss is None:
            continue
        step = int(step)
        if step > until_step:
            break
        wandb_run.log({"train/loss": float(loss)}, step=step)
        replayed += 1
    wandb_run.config.update(
        {
            "wandb_history_source_run_id": str(source_run_id),
            "wandb_history_replayed_metric": "train/loss",
            "wandb_history_until_step": until_step,
            "wandb_history_replayed_points": replayed,
        },
        allow_val_change=True,
    )
    startup_log(
        f"replayed {replayed} train/loss points through step {until_step} "
        f"from W&B run {source_run_id}"
    )


def main(_) -> None:
    mode = PretrainMode(FLAGS.pretrain_mode)
    if mode is PretrainMode.STATEPASSING_NO_BPTT:
        raise ValueError("The tail experiment requires statepassing_bptt or iid_baseline")
    if (FLAGS.init_model_checkpoint is None) == (FLAGS.init_hf_model_path is None):
        raise ValueError("Provide exactly one of --init_model_checkpoint or --init_hf_model_path")
    if FLAGS.lr_schedule == "cosine_bridge":
        if FLAGS.start_lr is None or FLAGS.end_lr is None:
            raise ValueError("cosine_bridge requires --start_lr and --end_lr")
    elif FLAGS.learning_rate is None:
        raise ValueError("wsd requires --learning_rate")
    validate_gdn_state_routing(
        pass_gdn_state=FLAGS.pass_gdn_state,
        gdn_layer_limit=FLAGS.gdn_layer_limit,
        gdn_state_routing=FLAGS.gdn_state_routing,
    )
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    jax.distributed.initialize()
    startup_log(
        f"iterator process_count={jax.process_count()} dp={FLAGS.iterator_dp_size} "
        f"fsdp={FLAGS.iterator_fsdp_size} excluded_prefix=(stride="
        f"{FLAGS.iterator_excluded_prefix_stride}, count={FLAGS.iterator_excluded_prefix_count}) "
        f"restore_input_iter={FLAGS.restore_input_iter}"
    )
    metadata, root, is_fixed_c_bundle = _load_bundle()
    if is_fixed_c_bundle:
        split = dict(dict(metadata["splits"])["train"])
        available_steps = int(
            dict(split["iid"])["iid_steps"]
            if mode is PretrainMode.IID_BASELINE
            else split["statepassing_steps"]
        )
    else:
        segments_per_step = int(FLAGS.batch_size) * int(FLAGS.grad_accum_steps)
        if segments_per_step % int(metadata["num_segments"]):
            raise ValueError("batch_size * grad_accum_steps must be divisible by num_segments")
        available_steps = int(metadata["num_windows"]) // (
            segments_per_step // int(metadata["num_segments"])
        )
    tokens_per_step = FLAGS.batch_size * FLAGS.seq_len * FLAGS.grad_accum_steps
    num_steps = (
        available_steps if FLAGS.max_tokens is None else int(FLAGS.max_tokens // tokens_per_step)
    )
    if num_steps <= 0:
        raise ValueError("Requested token budget does not contain one optimizer update")
    if num_steps > available_steps:
        raise ValueError(
            f"Requested {num_steps} optimizer steps, but the tail bundle contains only "
            f"{available_steps}."
        )
    config = TailTrainConfig(
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
        num_steps=num_steps,
        global_step_offset=FLAGS.global_step_offset,
        start_lr=FLAGS.start_lr,
        end_lr=FLAGS.end_lr,
        weight_decay=FLAGS.weight_decay,
        max_grad_norm=FLAGS.max_grad_norm,
        adam_beta1=FLAGS.adam_beta1,
        adam_beta2=FLAGS.adam_beta2,
        adam_eps=FLAGS.adam_eps,
        grad_accum_steps=FLAGS.grad_accum_steps,
        lr_schedule=FLAGS.lr_schedule,
        learning_rate=FLAGS.learning_rate,
        warmup_steps=FLAGS.warmup_tokens // tokens_per_step,
        lr_end_factor=FLAGS.lr_end_factor,
        lr_stable_fraction=FLAGS.lr_stable_fraction,
    )
    data_iter = _make_tail_iterator(metadata, root, mode, is_fixed_c_bundle=is_fixed_c_bundle)
    save_dir = Path(FLAGS.save_dir).expanduser().resolve()
    if mode.is_statepassing:
        contract = build_training_contract(
            root,
            pass_gdn_state=FLAGS.pass_gdn_state,
            gdn_layer_limit=FLAGS.gdn_layer_limit,
            gdn_state_routing=FLAGS.gdn_state_routing,
            pass_conv_state=FLAGS.pass_conv_state,
            pass_rope_positions=FLAGS.pass_rope_positions,
            pad_id=FLAGS.pad_id,
            eos_id=FLAGS.eos_id,
        )
        if jax.process_index() == 0:
            ensure_training_contract(save_dir, contract)

    wandb_run = None
    if FLAGS.wandb_project and jax.process_index() == 0:
        kwargs = {
            "entity": FLAGS.wandb_entity,
            "project": FLAGS.wandb_project,
            "group": FLAGS.wandb_group,
            "name": FLAGS.wandb_name,
            "tags": FLAGS.wandb_tags or None,
            "config": flags.FLAGS.flag_values_dict()
            | {
                "derived_num_steps": num_steps,
                "derived_global_stop_step": config.global_stop_step,
            },
        }
        if FLAGS.wandb_id:
            kwargs["id"] = FLAGS.wandb_id
        if FLAGS.wandb_resume:
            kwargs["resume"] = FLAGS.wandb_resume
        wandb_run = wandb.init(**kwargs)
        if wandb_run.resumed:
            startup_log("resumed existing W&B run; historical loss replay already present")
        else:
            _replay_wandb_loss_history(wandb_run)
    if FLAGS.gc_period:
        gc.disable()
    try:
        run_tail_pretrain(
            FLAGS.init_model_checkpoint,
            config,
            data_iter,
            pretrain_mode=mode,
            save_dir=save_dir,
            enable_lora=FLAGS.enable_lora,
            lora_adapter_seed=FLAGS.lora_adapter_seed,
            save_every=FLAGS.save_every,
            keep_latest=FLAGS.keep_latest,
            log_every=FLAGS.log_every,
            resume=ResumeMode(FLAGS.resume),
            pad_id=FLAGS.pad_id,
            peak_tflops=resolve_peak_tflops(FLAGS.peak_tflops),
            tp_size=FLAGS.tp_size,
            fsdp_size=FLAGS.fsdp_size,
            dp_size=FLAGS.dp_size,
            wandb_run=wandb_run,
            text_attn_backend=FLAGS.text_attn_backend,
            bptt_chunks=FLAGS.bptt_chunks,
            pass_gdn_state=FLAGS.pass_gdn_state,
            gdn_layer_limit=FLAGS.gdn_layer_limit,
            gdn_state_routing=FLAGS.gdn_state_routing,
            pass_rope_positions=FLAGS.pass_rope_positions,
            pass_conv_state=FLAGS.pass_conv_state,
            stop_after_local_step=FLAGS.stop_after_local_step,
            init_hf_model_path=FLAGS.init_hf_model_path,
            restore_input_iter=FLAGS.restore_input_iter,
            resume_expected_step=FLAGS.resume_expected_step,
        )
    finally:
        if FLAGS.gc_period:
            gc.enable()
        if wandb_run is not None:
            wandb_run.finish()
    startup_log("tail training complete")


if __name__ == "__main__":
    app.run(main)
