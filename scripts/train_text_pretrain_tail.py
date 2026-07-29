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
flags.DEFINE_string("train_index_path", None, "Fixed-C tail bundle root.", required=True)
flags.DEFINE_string("init_model_checkpoint", None, "Model-only source checkpoint.", required=True)
flags.DEFINE_string("save_dir", None, "New tail checkpoint root.", required=True)
flags.DEFINE_integer("seq_len", 2048, "Segment length.")
flags.DEFINE_integer("batch_size", 192, "Global segment batch per microstep.")
flags.DEFINE_integer("grad_accum_steps", 5, "Gradient accumulation microsteps.")
flags.DEFINE_integer("global_step_offset", 40000, "Global step before the first tail update.")
flags.DEFINE_float("start_lr", None, "First local tail-update LR.", required=True)
flags.DEFINE_float("end_lr", None, "Last local tail-update LR.", required=True)
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


def _load_bundle() -> tuple[dict[str, object], Path]:
    root = Path(FLAGS.train_index_path).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Tail bundle has no metadata.json: {root}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != STATEPASSING_FIXED_C_INDEX_FORMAT:
        raise ValueError(f"Expected a fixed-C bundle, got {metadata.get('format')!r}")
    if int(metadata["num_segments"]) != 4:
        raise ValueError(f"Tail bundle must use num_segments=4, got {metadata['num_segments']}")
    if int(metadata["trim_batch_size"]) != int(FLAGS.batch_size):
        raise ValueError("Tail bundle trim_batch_size does not match --batch_size")
    if int(metadata["trim_grad_accum_steps"]) != int(FLAGS.grad_accum_steps):
        raise ValueError("Tail bundle trim_grad_accum_steps does not match --grad_accum_steps")
    if int(metadata["chunk_length"]) != int(FLAGS.seq_len):
        raise ValueError("Tail bundle chunk_length does not match --seq_len")
    if metadata.get("eos_id") != FLAGS.eos_id:
        raise ValueError("Tail bundle eos_id does not match --eos_id")
    return metadata, root


def _make_tail_iterator(metadata: dict[str, object], root: Path, mode: PretrainMode):
    if FLAGS.batch_size % jax.process_count():
        raise ValueError("Global batch size must be divisible by jax.process_count()")
    per_process_batch = FLAGS.batch_size // jax.process_count()
    train_split = dict(dict(metadata["splits"])["train"])
    if mode is PretrainMode.IID_BASELINE:
        path = root / str(dict(train_split["iid"])["path"])
        make_iterator = make_iid_iterator
    else:
        path = root / str(train_split["path"])
        make_iterator = make_statepassing_iterator
    return make_iterator(
        str(path),
        batch_size=per_process_batch,
        chunk_length=FLAGS.seq_len,
        pad_id=FLAGS.pad_id,
        eos_id=FLAGS.eos_id,
        shuffle=True,
        seed=FLAGS.seed,
        num_epochs=1,
        dp_size=FLAGS.iterator_dp_size,
        fsdp_size=FLAGS.iterator_fsdp_size,
        process_index=jax.process_index(),
        grain_workers=FLAGS.grain_workers,
        grain_worker_buffer_size=FLAGS.grain_worker_buffer_size,
        grain_read_threads=FLAGS.grain_read_threads,
        grain_read_prefetch_buffer_size=FLAGS.grain_read_prefetch_buffer_size,
    )


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
    validate_gdn_state_routing(
        pass_gdn_state=FLAGS.pass_gdn_state,
        gdn_layer_limit=FLAGS.gdn_layer_limit,
        gdn_state_routing=FLAGS.gdn_state_routing,
    )
    if FLAGS.enable_lora and float(FLAGS.weight_decay) != 0.0:
        raise ValueError("LoRA tail training requires --weight_decay=0")
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    jax.distributed.initialize()
    metadata, root = _load_bundle()
    split = dict(dict(metadata["splits"])["train"])
    num_steps = int(
        dict(split["iid"])["iid_steps"]
        if mode is PretrainMode.IID_BASELINE
        else split["statepassing_steps"]
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
    )
    data_iter = _make_tail_iterator(metadata, root, mode)
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
        )
    finally:
        if FLAGS.gc_period:
            gc.enable()
        if wandb_run is not None:
            wandb_run.finish()
    startup_log("tail training complete")


if __name__ == "__main__":
    app.run(main)
