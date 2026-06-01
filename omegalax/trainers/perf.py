"""Throughput metrics for training: FLOP counting, step timing, and MFU.

Uses the MaxText-style ("algorithmic FLOPs") approach: attention FLOPs are
counted only over positions the kernel actually visits.
- Causal text attention is halved (2*T*H*K per token) because flash kernels
  skip masked-out future positions.
- Vision encoder attention is block-diagonal across images (sum of N_i^2
  rather than (sum N_i)^2) because the cuDNN packed/THD kernel skips
  cross-image tiles entirely via cu_seqlens.

This matches the convention used by Megatron-LM, NeMo, and most published
MFU numbers, and is consistent with the kernels in
``omegalax.models.qwen3_vl.vision``.
"""

from __future__ import annotations

import datetime
from typing import Any, Union

import jax
import jax.tree_util as jtu
import numpy as np

from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig
from omegalax.models.qwen3_vl.config import Qwen3VLConfig

# Config types that training_flops_per_token accepts (text or full VLM configs).
RunPerfConfig = Union[Qwen3Config, Qwen3_5TextConfig, Qwen3_5Config, Qwen3VLConfig]

# Training FLOPs = forward + backward; factor 3 (1 fwd + 2 bwd).
TRAINING_FLOP_MULTIPLIER = 3

# Peak bf16 TFLOPS (1e12 FLOP/s) for common GPUs. Used as denominator for MFU.
PEAK_TFLOPS: dict[str, float] = {
    "h100_sxm": 989.0,
    "h100_pcie": 756.0,
    "a100_sxm_80": 312.0,
    "a100_sxm_40": 312.0,
    "a100_pcie_80": 312.0,
    "a100_pcie_40": 312.0,
}


def resolve_peak_tflops(spec: str | float | None) -> float | None:
    """Convert a peak-TFLOPS spec to a float for MFU.

    Accepts: None; a float; or a preset name (exact key from PEAK_TFLOPS).
    Raises ValueError if a string is given that is neither a preset nor a number.
    """
    if spec is None:
        return None
    s = str(spec)
    if s in PEAK_TFLOPS:
        return PEAK_TFLOPS[s]
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(
            f"Unknown peak_tflops {spec!r}. Use a key from {list(PEAK_TFLOPS)} or a number."
        ) from e


def training_flops_per_token(cfg: RunPerfConfig, seq_len: int) -> int:
    """Theoretical training FLOPs per token (forward + backward, x3).

    Counts matmuls only. Accepts text configs (Qwen3, Qwen3.5 text) or full VLM
    configs (Qwen3_5Config → text decoder only; Qwen3VLConfig → decoder stack).
    Returns total FLOPs per token for one training step (already multiplied by 3).
    """
    if isinstance(cfg, Qwen3_5Config):
        return _training_flops_per_token_qwen3_5(cfg.text_config, seq_len)
    if isinstance(cfg, Qwen3VLConfig):
        return _training_flops_per_token_qwen3_vl(cfg, seq_len)
    if isinstance(cfg, Qwen3Config):
        if cfg.is_moe:
            return _training_flops_per_token_qwen3_moe(cfg, seq_len)
        return _training_flops_per_token_qwen3_dense(cfg, seq_len)
    if isinstance(cfg, Qwen3_5TextConfig):
        return _training_flops_per_token_qwen3_5(cfg, seq_len)
    raise TypeError(f"Unsupported config for FLOP counting: {type(cfg)}")


def _training_flops_per_token_qwen3_dense(cfg: Qwen3Config, seq_len: int) -> int:
    D = cfg.emb_dim
    H = cfg.num_heads
    G = cfg.num_kv_heads
    K = cfg.head_dim
    F = cfg.mlp_dim
    V = cfg.vocab_size
    L = cfg.num_layers
    T = seq_len

    # Per layer, per token (matmul FLOPs: 2 * M * N * K for [M,K] @ [K,N])
    qkv_flops = 2 * D * (H + 2 * G) * K
    attn_dot_flops = 2 * T * H * K  # causal attention: halved
    o_proj_flops = 2 * H * K * D
    attn_per_layer = qkv_flops + attn_dot_flops + o_proj_flops
    mlp_per_layer = 2 * 3 * D * F  # SwiGLU: gate, up, down
    embedding_flops = 2 * D * V

    forward_per_token = L * (attn_per_layer + mlp_per_layer) + embedding_flops
    return forward_per_token * TRAINING_FLOP_MULTIPLIER


def _training_flops_per_token_qwen3_vl(cfg: Qwen3VLConfig, seq_len: int) -> int:
    """Qwen3-VL decoder FLOPs (same structure as Qwen3 MoE/dense).

    This excludes the vision tower because its cost depends on the concrete
    ``image_grid_thw`` values for each batch.
    """
    D = cfg.emb_dim
    H = cfg.num_heads
    G = cfg.num_kv_heads
    K = cfg.head_dim
    F_dense = cfg.mlp_dim
    F_moe = cfg.moe_intermediate_size
    E = cfg.num_experts
    k = cfg.num_experts_per_tok
    V = cfg.vocab_size
    L = cfg.num_layers
    T = seq_len

    qkv_flops = 2 * D * (H + 2 * G) * K
    attn_dot_flops = 2 * T * H * K  # causal attention: halved
    o_proj_flops = 2 * H * K * D
    attn_per_layer = qkv_flops + attn_dot_flops + o_proj_flops

    layer_flops = 0
    for layer_idx in range(L):
        layer_flops += attn_per_layer
        if cfg.is_moe_layer(layer_idx):
            gate_flops = 2 * D * E
            expert_flops = k * (2 * 3 * D * F_moe)
            layer_flops += gate_flops + expert_flops
        else:
            layer_flops += 2 * 3 * D * F_dense

    embedding_flops = 2 * D * V
    forward_per_token = layer_flops + embedding_flops
    return forward_per_token * TRAINING_FLOP_MULTIPLIER


def qwen3_vl_vision_training_flops(
    cfg: Qwen3VLConfig, image_grid_thw: Any | None, *, vision_trainable: bool = True
) -> int:
    """Theoretical Qwen3-VL vision-tower FLOPs for one training step.

    Counts matmuls only and matches the current implementation in
    ``omegalax.models.qwen3_vl.vision``:
    - patch embed and patch mergers are linear layers (linear in token count);
    - vision attention is block-diagonal across images: the cuDNN packed/THD
      kernel uses ``vision_cu_seqlens`` to skip cross-image tiles, so per-image
      attention costs are summed (``sum_i 4 * N_i^2 * H * K``) rather than
      computed over the concatenated batch (``4 * (sum_i N_i)^2 * H * K``).

    ``vision_trainable`` controls the forward/backward multiplier. When the
    vision tower is trained, FLOPs are forward + backward (``x3``). When it is
    frozen (``--freeze_vision_tower`` or ``--enable_lora``, which take gradients
    only ``wrt`` non-vision params), no backward is built for the tower, so it
    runs forward-only (``x1``). Counting frozen vision at ``x3`` would inflate
    MFU because the vision tower dominates this VLM's FLOPs.
    """
    if image_grid_thw is None:
        return 0

    grid_N3 = np.asarray(image_grid_thw, dtype=np.int64)
    if grid_N3.size == 0:
        return 0
    if grid_N3.ndim != 2 or grid_N3.shape[1] != 3:
        raise ValueError(
            f"Expected image_grid_thw with shape (num_images, 3), got {grid_N3.shape}."
        )

    vis = cfg.vision
    merge = vis.spatial_merge_size

    per_image_tokens = grid_N3[:, 0] * grid_N3[:, 1] * grid_N3[:, 2]
    total_tokens = int(np.sum(per_image_tokens))
    sum_sq_tokens = int(np.sum(per_image_tokens * per_image_tokens))
    merged_tokens = int(
        np.sum(grid_N3[:, 0] * (grid_N3[:, 1] // merge) * (grid_N3[:, 2] // merge))
    )
    if total_tokens <= 0 or merged_tokens <= 0:
        return 0

    D = vis.hidden_size
    F = vis.intermediate_size
    H = vis.num_heads
    K = D // H
    in_features = vis.in_channels * vis.temporal_patch_size * vis.patch_size**2

    patch_embed_flops = 2 * total_tokens * in_features * D

    qkv_flops = 2 * total_tokens * D * (3 * D)
    attn_dot_flops = 4 * sum_sq_tokens * H * K  # block-diagonal: sum_i N_i^2
    o_proj_flops = 2 * total_tokens * D * D
    mlp_flops = 2 * total_tokens * D * F + 2 * total_tokens * F * D
    block_flops = vis.depth * (qkv_flops + attn_dot_flops + o_proj_flops + mlp_flops)

    merger_dim = D * (merge**2)
    merger_fc1_flops = 2 * merged_tokens * merger_dim * merger_dim
    merger_fc2_flops = 2 * merged_tokens * merger_dim * vis.out_hidden_size
    num_mergers = 1 + len(vis.deepstack_visual_indexes)
    merger_flops = num_mergers * (merger_fc1_flops + merger_fc2_flops)

    forward = patch_embed_flops + block_flops + merger_flops
    multiplier = TRAINING_FLOP_MULTIPLIER if vision_trainable else 1
    return forward * multiplier


def _training_flops_per_token_qwen3_moe(cfg: Qwen3Config, seq_len: int) -> int:
    D = cfg.emb_dim
    H = cfg.num_heads
    G = cfg.num_kv_heads
    K = cfg.head_dim
    F_dense = cfg.mlp_dim
    F_moe = cfg.moe_intermediate_size
    E = cfg.num_experts
    k = cfg.num_experts_per_tok
    V = cfg.vocab_size
    L = cfg.num_layers
    T = seq_len

    qkv_flops = 2 * D * (H + 2 * G) * K
    attn_dot_flops = 2 * T * H * K  # causal attention: halved
    o_proj_flops = 2 * H * K * D
    attn_per_layer = qkv_flops + attn_dot_flops + o_proj_flops

    layer_flops = 0
    for layer_idx in range(L):
        layer_flops += attn_per_layer
        if cfg.is_moe_layer(layer_idx):
            gate_flops = 2 * D * E
            expert_flops = k * (2 * 3 * D * F_moe)
            layer_flops += gate_flops + expert_flops
        else:
            layer_flops += 2 * 3 * D * F_dense

    embedding_flops = 2 * D * V
    forward_per_token = layer_flops + embedding_flops
    return forward_per_token * TRAINING_FLOP_MULTIPLIER


def _training_flops_per_token_qwen3_5(cfg: Qwen3_5TextConfig, seq_len: int) -> int:
    D = cfg.hidden_size
    H = cfg.num_attention_heads
    G = cfg.num_key_value_heads
    K = cfg.head_dim
    V = cfg.vocab_size
    L = cfg.num_hidden_layers
    T = seq_len

    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    nv = cfg.linear_num_value_heads
    nk = cfg.linear_num_key_heads
    ak = cfg.linear_key_head_dim
    av = cfg.linear_value_head_dim

    layer_flops = 0
    for layer_idx, layer_type in enumerate(cfg.layer_types):
        if layer_type == "full_attention":
            q_flops = 2 * D * (H * K * 2)
            kv_flops = 2 * D * (2 * G * K)
            attn_dot = 2 * T * H * K  # causal attention: halved
            o_flops = 2 * H * K * D
            layer_flops += q_flops + kv_flops + attn_dot + o_flops
        else:
            conv_dim = key_dim * 2 + value_dim
            in_proj_qkv = 2 * D * conv_dim
            in_proj_z = 2 * D * value_dim
            in_proj_b = 2 * D * nv
            in_proj_a = 2 * D * nv
            out_proj = 2 * value_dim * D
            delta_rule_per_token = 2 * nv * (ak * av)
            layer_flops += in_proj_qkv + in_proj_z + in_proj_b + in_proj_a + out_proj + delta_rule_per_token

        if cfg.is_moe:
            E = cfg.num_experts
            k = cfg.num_experts_per_tok
            F_moe = cfg.moe_intermediate_size
            F_shared = cfg.shared_expert_intermediate_size
            router_flops = 2 * D * E
            gate_up_per_expert = 2 * (2 * F_moe) * D
            down_per_expert = 2 * F_moe * D
            routed_flops = k * (gate_up_per_expert + down_per_expert)
            shared_flops = 2 * 3 * D * F_shared
            shared_gate_flops = 2 * D * 1
            layer_flops += router_flops + routed_flops + shared_flops + shared_gate_flops
        else:
            F_dense = cfg.intermediate_size
            layer_flops += 2 * 3 * D * F_dense

    embedding_flops = 2 * D * V
    forward_per_token = layer_flops + embedding_flops
    return forward_per_token * TRAINING_FLOP_MULTIPLIER


def _tree_global_bytes(tree) -> int:
    """Logical bytes for every leaf in a pytree (global, ignores sharding)."""
    total = 0
    for x in jax.tree.leaves(tree):
        dtype = getattr(x, "dtype", None)
        size = getattr(x, "size", None)
        if dtype is None or size is None:
            continue
        total += int(size) * dtype.itemsize
    return total


def _tree_local_bytes(tree) -> int:
    """Bytes that physically live on this process's devices (sum of addressable shards)."""
    total = 0
    for x in jax.tree.leaves(tree):
        shards = getattr(x, "addressable_shards", None)
        if shards is not None:
            for s in shards:
                total += s.data.nbytes
        elif hasattr(x, "nbytes"):
            total += x.nbytes
    return total


def _write_memory_log(save_dir: Any, lines: list[str]) -> None:
    """Append ``lines`` (timestamp-prefixed) to ``<save_dir>/memory.log``.

    No-op on non-primary processes or when ``save_dir`` is None.
    """
    from pathlib import Path

    log_path = Path(save_dir) / "memory.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        for line in lines:
            f.write(f"[{ts}] {line}\n")


def log_pytree_bytes(name: str, tree, save_dir: Any) -> None:
    """Log global + per-process byte counts for a pytree to ``save_dir/memory.log``."""
    if jax.process_index() != 0:
        return
    n_leaves = len(jax.tree.leaves(tree))
    gb = _tree_global_bytes(tree) / 1e9
    lb = _tree_local_bytes(tree) / 1e9
    _write_memory_log(
        save_dir,
        [f"{name}: leaves={n_leaves} global={gb:.3f} GB local(per-process)={lb:.3f} GB"],
    )


def log_device_memory(tag: str, save_dir: Any) -> None:
    """Log per-device allocator stats (in-use / peak / limit / largest-free-block)."""
    lines: list[str] = []
    for d in jax.local_devices():
        try:
            s = d.memory_stats()
        except Exception as e:
            lines.append(
                f"{tag} proc={jax.process_index()} dev={d.id}: memory_stats unavailable ({e})"
            )
            continue
        if not s:
            continue
        in_use = s.get("bytes_in_use", 0) / 1e9
        peak = s.get("peak_bytes_in_use", 0) / 1e9
        limit = s.get("bytes_limit", 0) / 1e9
        reserved = s.get("bytes_reserved", 0) / 1e9
        largest_free = s.get("largest_free_block_bytes", 0) / 1e9
        lines.append(
            f"{tag} proc={jax.process_index()} dev={d.id}: "
            f"in_use={in_use:.2f} GB peak={peak:.2f} GB "
            f"limit={limit:.2f} GB reserved={reserved:.2f} GB "
            f"largest_free_block={largest_free:.2f} GB"
        )
    if lines:
        _write_memory_log(save_dir, lines)


def log_top_leaves_with_paths(name: str, tree, save_dir: Any) -> None:
    """Log all pytree leaves by local bytes (sorted desc) to ``save_dir/memory.log``.

    Unlike ``log_live_arrays`` (which uses anonymous ``jax.live_arrays()``), this
    walks a named pytree (e.g. ``nnx.state(optimizer)``) so each entry is tied
    to the param/opt-state slot it came from. Appends to the log file so
    multiple calls within a run accumulate.
    """
    if jax.process_index() != 0:
        return
    leaves_with_paths, _ = jtu.tree_flatten_with_path(tree)
    sized = []
    for path, x in leaves_with_paths:
        shards = getattr(x, "addressable_shards", None)
        if shards is not None:
            nb = sum(s.data.nbytes for s in shards)
        else:
            nb = getattr(x, "nbytes", 0)
        sized.append((nb, jtu.keystr(path), getattr(x, "shape", None), getattr(x, "dtype", None)))
    sized.sort(reverse=True, key=lambda e: e[0])

    lines = [f"{name}: all {len(sized)} leaves by local bytes"]
    for nb, path, shape, dtype in sized:
        lines.append(f"  {nb / 1e6:12.2f} MB {path}  shape={shape} dtype={dtype}")
    _write_memory_log(save_dir, lines)


def log_live_arrays(tag: str, save_dir: Any) -> None:
    """Log a summary of all currently-alive JAX arrays on this process."""
    if jax.process_index() != 0:
        return
    try:
        arrays = jax.live_arrays()
    except Exception as e:
        _write_memory_log(save_dir, [f"{tag}: live_arrays unavailable ({e})"])
        return
    entries = []
    total_local = 0
    for a in arrays:
        shards = getattr(a, "addressable_shards", None)
        if shards is not None:
            nb = sum(s.data.nbytes for s in shards)
        else:
            nb = getattr(a, "nbytes", 0)
        total_local += nb
        entries.append((nb, getattr(a, "shape", None), getattr(a, "dtype", None)))
    entries.sort(reverse=True, key=lambda e: e[0])
    lines = [f"{tag}: live_arrays count={len(arrays)} local_total={total_local / 1e9:.3f} GB"]
    for nb, shape, dtype in entries:
        lines.append(f"  {nb / 1e6:12.2f} MB shape={shape} dtype={dtype}")
    _write_memory_log(save_dir, lines)


def log_compiled_memory_analysis(name: str, jit_fn, save_dir: Any, *args, **kwargs) -> None:
    """Best-effort: lower+compile the jit fn and log XLA's static memory analysis.

    This re-traces but should hit the persistent compile cache. Wrapped in
    try/except because (a) nnx.jit may not expose .lower for all signatures and
    (b) some backends don't implement memory_analysis.
    """
    if jax.process_index() != 0:
        return
    try:
        lowered = jit_fn.lower(*args, **kwargs)
        compiled = lowered.compile()
        ma = compiled.memory_analysis()
    except Exception as e:
        _write_memory_log(
            save_dir, [f"{name}: memory_analysis unavailable ({type(e).__name__}: {e})"]
        )
        return
    if ma is None:
        _write_memory_log(save_dir, [f"{name}: memory_analysis returned None"])
        return
    fields = [
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "host_temp_size_in_bytes",
    ]
    parts = []
    for f in fields:
        v = getattr(ma, f, None)
        if v is not None:
            parts.append(f"{f[: -len('_in_bytes')]}={v / 1e9:.3f} GB")
    _write_memory_log(save_dir, [f"{name}: " + " ".join(parts)])


class StepTimer:
    """Wall-clock timer between step dispatches (no device sync).

    First `warmup` steps return zero delta; after that returns time since last
    step() call. Relies on pipeline saturation for accuracy.
    """

    def __init__(self, warmup: int = 2):
        self._warmup = warmup
        self._step_count = 0
        self._last = datetime.datetime.now()

    def step(self) -> datetime.timedelta:
        now = datetime.datetime.now()
        delta = now - self._last
        self._last = now
        self._step_count += 1
        if self._step_count <= self._warmup:
            return datetime.timedelta(0)
        return delta


def per_device_flops_per_step(
    cfg: RunPerfConfig,
    seq_len: int,
    batch_size: int,
    image_grid_thw: Any | None = None,
    *,
    vision_trainable: bool = True,
) -> float:
    """Total training FLOPs per step, divided by device count.

    For Qwen3-VL, ``image_grid_thw`` adds the vision-tower FLOPs for the
    concrete batch. Text-decoder FLOPs are still computed from the padded
    ``seq_len`` and ``batch_size``. ``vision_trainable=False`` counts the
    frozen vision tower forward-only (``x1``) instead of forward+backward
    (``x3``); see ``qwen3_vl_vision_training_flops``.
    """
    total = training_flops_per_token(cfg, seq_len) * seq_len * batch_size
    if isinstance(cfg, Qwen3VLConfig):
        total += qwen3_vl_vision_training_flops(
            cfg, image_grid_thw, vision_trainable=vision_trainable
        )
    return total / max(1, jax.device_count())


def step_metrics(
    per_device_flops: float,
    step_delta: datetime.timedelta,
    global_tokens_per_step: int,
    peak_tflops: float | None,
) -> dict[str, float]:
    """Compute tokens/s, TFLOPS/device, and MFU from step timing."""
    sec = step_delta.total_seconds()
    if sec <= 0:
        return {
            "step_time_s": 0.0,
            "global_tokens_per_sec": 0.0,
            "tokens_per_sec_per_device": 0.0,
            "tflops_per_device": 0.0,
            "mfu": 0.0,
        }
    n_devices = jax.device_count()
    global_tokens_per_sec = global_tokens_per_step / sec
    tokens_per_sec_per_device = global_tokens_per_sec / n_devices
    flops_per_sec_per_device = per_device_flops / sec
    tflops_per_device = flops_per_sec_per_device / 1e12
    mfu = (flops_per_sec_per_device / (peak_tflops * 1e12)) if peak_tflops else 0.0
    return {
        "step_time_s": sec,
        "global_tokens_per_sec": global_tokens_per_sec,
        "tokens_per_sec_per_device": tokens_per_sec_per_device,
        "tflops_per_device": tflops_per_device,
        "mfu": mfu,
    }


def maybe_log_step_metrics(
    step_to_log: int,
    metrics_to_log: dict[str, Any],
    step_delta: datetime.timedelta,
    *,
    is_primary_process: bool,
    log_every: int,
    force: bool = False,
    per_device_flops: float,
    global_tokens_per_step: int,
    peak_tflops: float | None,
    wandb_run: Any = None,
    batch_size: int = 0,
) -> dict[str, float] | None:
    """Optionally compute and log step metrics. Returns host_metrics if logged, else None."""
    should_log = is_primary_process and log_every and step_to_log % log_every == 0
    if not (should_log or force):
        return None

    host_metrics = {k: float(v) for k, v in metrics_to_log.items()}
    required = ("loss", "grad_norm")
    missing = [k for k in required if k not in host_metrics]
    if missing:
        raise KeyError(f"Missing required metrics for logging: {missing}")
    host_metrics["step"] = step_to_log
    if batch_size > 0:
        host_metrics["total_samples"] = step_to_log * batch_size
    host_metrics.update(
        step_metrics(per_device_flops, step_delta, global_tokens_per_step, peak_tflops)
    )

    if wandb_run is not None and is_primary_process:
        _SKIP = {"step"}
        wandb_run.log(
            {f"train/{k}": v for k, v in host_metrics.items() if k not in _SKIP},
            step=step_to_log,
        )

    if is_primary_process:
        lr = host_metrics.get("lr", 0.0)
        print(
            f"time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"step={step_to_log} "
            f"loss={host_metrics['loss']:.4f} "
            f"grad_norm={host_metrics['grad_norm']:.4f} "
            f"train/total_samples={host_metrics.get('total_samples', 0)} "
            f"train/global_tokens_per_sec={host_metrics.get('global_tokens_per_sec', 0.0):.0f} "
            f"train/step_time_s={host_metrics.get('step_time_s', 0.0):.2f}s "
            f"train/lr={lr:.2e} "
            f"train/tflops_per_device={host_metrics.get('tflops_per_device', 0.0):.2f} "
            f"train/mfu={host_metrics.get('mfu', 0.0) * 100:.1f}% "
            f"train/tok/s/dev={host_metrics.get('tokens_per_sec_per_device', 0.0):.0f} ",
            flush=True,
        )

    return host_metrics
