"""VLM SFT training from a compiled Grain dataset (text-only or multimodal)."""

from __future__ import annotations

import gc
import importlib
import json
import math
from pathlib import Path

import grain
import jax
import wandb
from absl import app, flags
from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor

from omegalax.data.collator_qwen3 import VLMSFTCollator
from omegalax.data.grain_pipeline import (
    MixSource,
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    parse_data_mix,
    required_epochs_for_batches,
)
from omegalax.distributed.mesh import process_local_batch_size
from omegalax.trainers import vlm as vlm_trainer
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.trainers.perf import resolve_peak_tflops
from omegalax.trainers.text import startup_log
from omegalax.vlm import api as vlm_api
from omegalax.vlm.local_snapshot import LocalVLMSnapshot, open_local_vlm_snapshot

FLAGS = flags.FLAGS

flags.DEFINE_string("model_snapshot", None, "Absolute sealed local Hugging Face VLM snapshot.")
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
flags.DEFINE_integer("max_length", None, "Maximum sequence length.")
flags.DEFINE_integer("schedule_horizon", None, "Immutable learning-rate schedule horizon.")
flags.DEFINE_integer(
    "invocation_end_step",
    None,
    "Required final optimizer generation for this invocation phase.",
)
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
flags.DEFINE_float("max_grad_norm", None, "Positive finite gradient clipping norm.")
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
    [ResumeMode.NEVER.value, ResumeMode.REQUIRED.value],
    "Checkpoint policy: 'never' creates a new checkpoint root; 'required' restores "
    "the exact --resume_step frontier.",
)
flags.DEFINE_integer("resume_step", None, "Exact committed step required for resume='required'.")
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
flags.DEFINE_string(
    "extra_transform",
    None,
    'A single {"class": "module:ClassName", "kwargs": {...}} object applied '
    "as a grain RandomMap augmentation to the train iterator only. The class "
    "must subclass grain.transforms.RandomMap and be importable in worker "
    "processes — set PYTHONPATH in the launch environment if the module lives "
    "outside the installed venv.",
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
    "model_snapshot",
    "max_length",
    "schedule_horizon",
    "invocation_end_step",
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

    if FLAGS.max_grad_norm is not None and (
        not math.isfinite(FLAGS.max_grad_norm) or FLAGS.max_grad_norm <= 0
    ):
        problems.append("max_grad_norm (must be positive and finite)")

    int32_max = 2_147_483_647
    if FLAGS.schedule_horizon is not None and not (0 < FLAGS.schedule_horizon <= int32_max):
        problems.append(f"schedule_horizon (must be an integer in [1, {int32_max}])")
    if (
        FLAGS.invocation_end_step is not None
        and FLAGS.schedule_horizon is not None
        and not (0 < FLAGS.invocation_end_step <= FLAGS.schedule_horizon)
    ):
        problems.append(
            "invocation_end_step (must be positive and no greater than schedule_horizon)"
        )
    if (
        FLAGS.resume_step is not None
        and FLAGS.invocation_end_step is not None
        and FLAGS.resume_step >= FLAGS.invocation_end_step
    ):
        problems.append("resume_step (must be less than invocation_end_step)")

    if (FLAGS.data_path is None) == (FLAGS.data_mix is None):
        problems.append("exactly one of {data_path, data_mix} (got neither or both)")

    # Both freeze the vision tower, so asking for both is a contradiction, not a no-op.
    if FLAGS.enable_lora and FLAGS.freeze_vision_tower:
        problems.append("enable_lora and freeze_vision_tower are mutually exclusive")

    if FLAGS.enable_lora:
        for name in ("lora_rank", "lora_alpha"):
            if FLAGS[name].value is None:
                problems.append(f"{name} (required when enable_lora=true)")

    if FLAGS.resume == ResumeMode.REQUIRED.value:
        if FLAGS.resume_step is None or FLAGS.resume_step <= 0:
            problems.append("resume_step (positive integer required when resume=required)")
    elif FLAGS.resume_step is not None:
        problems.append("resume_step (must be unset when resume=never)")

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
    if (FLAGS.data_path is None) == (FLAGS.data_mix is None):
        raise ValueError("Specify exactly one of --data_path or --data_mix.")
    if FLAGS.data_mix is not None:
        return parse_data_mix(FLAGS.data_mix)
    return [MixSource(path=FLAGS.data_path, weight=1.0)]


def _parse_extra_transform(
    spec: str | None,
) -> grain.transforms.RandomMap | None:
    """Instantiate the train-only augmentation transform from --extra_transform.

    ``spec`` is a single JSON ``{"class": "module:ClassName", "kwargs": {...}}``
    object whose class must be a ``grain.transforms.RandomMap`` subclass (our
    augmentations are stochastic). A malformed spec fails loudly here.
    """
    if spec is None:
        return None
    entry = json.loads(spec)
    module_path, _, class_name = entry["class"].partition(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    assert issubclass(cls, grain.transforms.RandomMap), (
        f"--extra_transform {entry['class']!r} is not a grain.transforms.RandomMap subclass"
    )
    return cls(**entry["kwargs"])


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
    extra_transform: grain.transforms.RandomMap | None,
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
        extra_transform=extra_transform,
    )


def _load_snapshot_assets(model_snapshot: LocalVLMSnapshot, model_cfg: vlm_api.Qwen3VLConfig):
    with model_snapshot.files() as files:
        with open(files["tokenizer_config.json"], encoding="utf-8") as stream:
            tokenizer_config = json.load(stream)
        tokenizer_class = tokenizer_config.get("tokenizer_class")
        if tokenizer_class != "Qwen2Tokenizer":
            raise ValueError(f"Unsupported sealed tokenizer class: {tokenizer_class!r}")
        with open(files["chat_template.json"], encoding="utf-8") as stream:
            chat_template = json.load(stream)
        if (
            set(chat_template) != {"chat_template"}
            or type(chat_template["chat_template"]) is not str
        ):
            raise ValueError("Sealed chat_template.json must contain exactly one string template")
        if tokenizer_config.get("chat_template") != chat_template["chat_template"]:
            raise ValueError("Sealed tokenizer config and chat template do not match")
        tokenizer = Qwen2Tokenizer._from_pretrained(
            {
                "tokenizer_file": files["tokenizer.json"],
                "tokenizer_config_file": files["tokenizer_config.json"],
            },
            str(model_snapshot.path),
            {},
            local_files_only=True,
            _is_local=True,
            trust_remote_code=False,
        )

        with open(files["preprocessor_config.json"], encoding="utf-8") as stream:
            preprocessor_config = json.load(stream)
        image_processor_type = preprocessor_config.pop("image_processor_type", None)
        if image_processor_type not in {
            "Qwen2VLImageProcessor",
            "Qwen2VLImageProcessorFast",
        }:
            raise ValueError(f"Unsupported sealed image processor class: {image_processor_type!r}")
        preprocessor_config.pop("processor_class", None)
        image_processor = Qwen2VLImageProcessor.from_dict(preprocessor_config)
    expected_token_ids = {
        "<|image_pad|>": model_cfg.image_token_id,
        "<|video_pad|>": model_cfg.video_token_id,
        "<|vision_start|>": model_cfg.vision_start_token_id,
        "<|vision_end|>": model_cfg.vision_end_token_id,
    }
    actual_token_ids = tokenizer.convert_tokens_to_ids(list(expected_token_ids))
    for token, actual in zip(expected_token_ids, actual_token_ids, strict=True):
        expected = expected_token_ids[token]
        if actual != expected:
            raise ValueError(
                f"Sealed tokenizer ID for {token} is {actual}, model config requires {expected}"
            )
    if len(tokenizer) > model_cfg.vocab_size:
        raise ValueError(
            f"Sealed tokenizer has {len(tokenizer)} tokens, model vocab has {model_cfg.vocab_size}"
        )
    processor_shape = (
        image_processor.patch_size,
        image_processor.temporal_patch_size,
        image_processor.merge_size,
    )
    model_shape = (
        model_cfg.vision.patch_size,
        model_cfg.vision.temporal_patch_size,
        model_cfg.vision.spatial_merge_size,
    )
    if processor_shape != model_shape:
        raise ValueError(
            f"Sealed image processor geometry {processor_shape} does not match model {model_shape}"
        )
    return tokenizer, image_processor


def _run(model_snapshot: LocalVLMSnapshot, tokenizer, image_processor) -> None:
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache_dir)
    jax.distributed.initialize()
    vlm_trainer._require_single_jax_process()
    vlm_trainer._require_registrar_compiled_executable_capability()
    startup_log(f"jax_compilation_cache_dir={FLAGS.jax_cache_dir}")
    startup_log("jax.distributed initialized")

    startup_log(f"loaded tokenizer and image processor from {model_snapshot.path!s}")

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
        # The dataset build budgets tokens (max_length) and records nothing about
        # patches, so it cannot refuse a sample this budget will. Below the ceiling
        # a sample can satisfy max_length and still exceed us, and the collator
        # only finds out on the batch that contains it -- thousands of steps in,
        # taking any afterok chain with it. Not an error: `_pad_vision_arrays` pads
        # to exactly max_patches, so the ceiling makes every batch pay maximum
        # vision padding. The tightest safe value is a judgement, so say what the
        # risk is and let the operator hold it.
        ceiling = ms2 * FLAGS.max_length
        if FLAGS.max_vision_patches_per_sample < ceiling:
            startup_log(
                f"WARNING max_vision_patches_per_sample="
                f"{FLAGS.max_vision_patches_per_sample} is below the "
                f"{ceiling} patches a {FLAGS.max_length}-token sample can "
                f"physically hold at merge_size={merge_size}. Nothing upstream "
                f"bounds patches, so a sample above the budget can reach the "
                f"collator and kill the run mid-loop. Raise to {ceiling} to make "
                f"that impossible, or keep it tight knowingly."
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
        f"model_snapshot={model_snapshot.path!s} data_sources=[{sources_repr}] "
        f"jax_compilation_cache_dir={FLAGS.jax_cache_dir!r} "
        f"process_count={jax.process_count()} local_device_count={jax.local_device_count()}"
    )
    if jax.process_index() == 0:
        print(
            f"global_batch_size={FLAGS.batch_size} process_count={jax.process_count()} "
            f"per_process_batch_size={per_process_batch}"
        )

    total_micro_batches = FLAGS.schedule_horizon * FLAGS.grad_accum_steps
    extra_transform = _parse_extra_transform(FLAGS.extra_transform)
    if extra_transform is not None:
        startup_log(f"loaded extra train transform: {type(extra_transform).__name__}")
    data_iter = _grain_iter(
        train_sources,
        collator,
        per_process_batch,
        shuffle=True,
        seed=FLAGS.seed,
        num_batches=total_micro_batches,
        dp_size=FLAGS.dp_size,
        fsdp_size=FLAGS.fsdp_size,
        extra_transform=extra_transform,
    )
    startup_log("built train grain DataLoader iterator")

    val_data_iter = None
    if FLAGS.val_data_path:
        # extra_transform=None — augmentation must NEVER run on validation data
        # (it would corrupt val metrics by scoring against re-scaled labels).
        val_data_iter = _grain_iter(
            [MixSource(path=FLAGS.val_data_path, weight=1.0)],
            collator,
            per_process_batch,
            shuffle=False,
            seed=FLAGS.seed,
            num_batches=max(
                1,
                (FLAGS.schedule_horizon // max(FLAGS.val_every or FLAGS.schedule_horizon, 1))
                * FLAGS.val_steps,
            ),
            dp_size=FLAGS.dp_size,
            fsdp_size=FLAGS.fsdp_size,
            extra_transform=None,
        )
        startup_log(f"built val grain DataLoader iterator from {FLAGS.val_data_path!r}")

    train_cfg = vlm_trainer.TrainConfig(
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.max_length,
        schedule_horizon=FLAGS.schedule_horizon,
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
            model_snapshot,
            train_cfg,
            data_iter,
            invocation_end_step=FLAGS.invocation_end_step,
            save_dir=save_dir,
            save_every=FLAGS.save_every,
            keep_period=FLAGS.keep_period,
            keep_latest=FLAGS.keep_latest,
            log_every=FLAGS.log_every,
            resume=resume_mode,
            resume_step=FLAGS.resume_step,
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
        )
    finally:
        if FLAGS.gc_period:
            gc.enable()
            print("Training completed, re-enabling Python GC")

        if wandb_run is not None:
            wandb_run.finish()

    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}")


def main(_) -> None:
    _validate_flags()
    with open_local_vlm_snapshot(FLAGS.model_snapshot) as model_snapshot:
        model_cfg, _ = vlm_api.validate_pretrained(model_snapshot)
        tokenizer, image_processor = _load_snapshot_assets(model_snapshot, model_cfg)
        if FLAGS.max_length > tokenizer.model_max_length:
            raise ValueError(
                f"--max_length={FLAGS.max_length} exceeds "
                f"tokenizer.model_max_length={tokenizer.model_max_length}"
            )
        _run(model_snapshot, tokenizer, image_processor)


if __name__ == "__main__":
    app.run(main)
