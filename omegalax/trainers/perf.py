"""Throughput metrics: FLOP counting, step timing, MFU and HFU.

Algorithmic ("MaxText-style") FLOPs: causal attention halved; vision attention
block-diagonal across images. MFU uses model FLOPs and is LoRA-aware (a frozen
weighted matmul costs 2x forward, not 3x; weightless attention is always 3x). HFU
adds the jax.remat layer recompute, read from DECODER_LAYER_REMAT /
VISION_BLOCK_REMAT; the two are equal only for a full fine-tune without recompute.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Union

import jax
import jax.tree_util as jtu
import numpy as np

from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig
from omegalax.models.qwen3_5.kernels import resolve_backend as resolve_deltanet_backend
from omegalax.models.qwen3_vl.config import Qwen3VLConfig

# Config types that the FLOP counters accept (text or full VLM configs).
RunPerfConfig = Union[Qwen3Config, Qwen3_5TextConfig, Qwen3_5Config, Qwen3VLConfig]

# Full-FT training FLOPs = forward + backward; factor 3 (1 fwd + 2 bwd).
TRAINING_FLOP_MULTIPLIER = 3

# Peak bf16 TFLOPS (1e12 FLOP/s) for common GPUs. Used as denominator for MFU/HFU.
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


def record_deltanet_kernel(cfg: RunPerfConfig, wandb_run: Any) -> str | None:
    """Record which DeltaNet kernel this run's MFU/HFU is measuring; None if it has none.

    Resolved here rather than at first trace so an unavailable kernel fails before
    training and the recorded value is provably the one that ran.
    """
    if not isinstance(cfg, (Qwen3_5Config, Qwen3_5TextConfig)):
        return None
    backend = resolve_deltanet_backend()
    if wandb_run is not None and jax.process_index() == 0:
        wandb_run.config.update({"deltanet_kernel": backend}, allow_val_change=True)
    return backend


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardFlops:
    """Per-token forward matmul FLOPs, split by how the backward scales.

    weighted_layers: weighted matmuls in the remat layer (2x frozen / 3x trainable).
    attention: weightless matmuls, always 3x. head: lm_head, outside remat.
    """

    weighted_layers: int
    attention: int
    head: int

    @property
    def forward(self) -> int:
        return self.weighted_layers + self.attention + self.head

    def model_flops(self, *, base_weights_trainable: bool) -> int:
        """Theoretical training FLOPs/token (fwd+bwd), LoRA-aware, no recompute."""
        weighted_mult = 3 if base_weights_trainable else 2
        return weighted_mult * (self.weighted_layers + self.head) + 3 * self.attention

    def recompute_flops(self) -> int:
        """Extra FLOPs/token from recomputing the rematerialized layer forward."""
        return self.weighted_layers + self.attention

    def hardware_flops(self, *, base_weights_trainable: bool, decoder_remat: bool) -> int:
        """Actual hardware FLOPs/token = model FLOPs + activation-checkpoint recompute."""
        model = self.model_flops(base_weights_trainable=base_weights_trainable)
        return model + (self.recompute_flops() if decoder_remat else 0)


@dataclasses.dataclass(frozen=True, slots=True)
class VisionFlops:
    """Vision-tower forward matmul FLOPs for one training step (whole batch)."""

    forward: int  # patch-embed + transformer blocks + patch mergers
    block_forward: int  # only the transformer blocks (the rematerialized region)


@dataclasses.dataclass(frozen=True, slots=True)
class StepFlops:
    """Per-device FLOPs for one training step, split into model vs hardware."""

    model: float  # theoretical model FLOPs (drives MFU); LoRA-aware, no recompute
    hardware: float  # actual hardware FLOPs (drives HFU); includes remat recompute


def forward_flops_per_token(cfg: RunPerfConfig, seq_len: int) -> ForwardFlops:
    """Per-token forward matmul FLOPs (text decoder), decomposed for MFU/HFU.

    Accepts text or full VLM configs; the vision tower is counted separately by
    ``qwen3_vl_vision_flops``.
    """
    if isinstance(cfg, Qwen3_5Config):
        return _forward_flops_qwen3_5(cfg.text_config, seq_len)
    if isinstance(cfg, Qwen3VLConfig):
        return _forward_flops_qwen3_vl(cfg, seq_len)
    if isinstance(cfg, Qwen3Config):
        if cfg.is_moe:
            return _forward_flops_qwen3_moe(cfg, seq_len)
        return _forward_flops_qwen3_dense(cfg, seq_len)
    if isinstance(cfg, Qwen3_5TextConfig):
        return _forward_flops_qwen3_5(cfg, seq_len)
    raise TypeError(f"Unsupported config for FLOP counting: {type(cfg)}")


def training_flops_per_token(cfg: RunPerfConfig, seq_len: int) -> int:
    """Full fine-tune training FLOPs per token (3 * forward).

    For LoRA/frozen-aware counting use
    ``forward_flops_per_token(...).model_flops(base_weights_trainable=...)``.
    """
    return TRAINING_FLOP_MULTIPLIER * forward_flops_per_token(cfg, seq_len).forward


def _forward_flops_qwen3_dense(cfg: Qwen3Config, seq_len: int) -> ForwardFlops:
    D = cfg.emb_dim
    H = cfg.num_heads
    G = cfg.num_kv_heads
    K = cfg.head_dim
    F = cfg.mlp_dim
    V = cfg.vocab_size
    L = cfg.num_layers
    T = seq_len

    qkv_flops = 2 * D * (H + 2 * G) * K
    attn_dot_flops = 2 * T * H * K  # causal: halved (weightless)
    o_proj_flops = 2 * H * K * D
    mlp_flops = 2 * 3 * D * F  # SwiGLU gate/up/down

    weighted_layers = L * (qkv_flops + o_proj_flops + mlp_flops)
    attention = L * attn_dot_flops
    head = 2 * D * V
    return ForwardFlops(weighted_layers=weighted_layers, attention=attention, head=head)


def _forward_flops_qwen3_vl(cfg: Qwen3VLConfig, seq_len: int) -> ForwardFlops:
    """Qwen3-VL decoder forward FLOPs (excludes the vision tower)."""
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
    attn_dot_flops = 2 * T * H * K  # causal attention: halved (weightless)
    o_proj_flops = 2 * H * K * D

    weighted_layers = 0
    for layer_idx in range(L):
        weighted_layers += qkv_flops + o_proj_flops
        if cfg.is_moe_layer(layer_idx):
            gate_flops = 2 * D * E
            expert_flops = k * (2 * 3 * D * F_moe)
            weighted_layers += gate_flops + expert_flops
        else:
            weighted_layers += 2 * 3 * D * F_dense

    attention = L * attn_dot_flops
    head = 2 * D * V
    return ForwardFlops(weighted_layers=weighted_layers, attention=attention, head=head)


def _forward_flops_qwen3_moe(cfg: Qwen3Config, seq_len: int) -> ForwardFlops:
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
    attn_dot_flops = 2 * T * H * K  # causal attention: halved (weightless)
    o_proj_flops = 2 * H * K * D

    weighted_layers = 0
    for layer_idx in range(L):
        weighted_layers += qkv_flops + o_proj_flops
        if cfg.is_moe_layer(layer_idx):
            gate_flops = 2 * D * E
            expert_flops = k * (2 * 3 * D * F_moe)
            weighted_layers += gate_flops + expert_flops
        else:
            weighted_layers += 2 * 3 * D * F_dense

    attention = L * attn_dot_flops
    head = 2 * D * V
    return ForwardFlops(weighted_layers=weighted_layers, attention=attention, head=head)


def _forward_flops_qwen3_5(cfg: Qwen3_5TextConfig, seq_len: int) -> ForwardFlops:
    D = cfg.hidden_size
    H = cfg.num_attention_heads
    G = cfg.num_key_value_heads
    K = cfg.head_dim
    V = cfg.vocab_size
    T = seq_len

    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    nv = cfg.linear_num_value_heads
    ak = cfg.linear_key_head_dim
    av = cfg.linear_value_head_dim

    weighted_layers = 0
    attention = 0
    for layer_type in cfg.layer_types:
        if layer_type == "full_attention":
            q_flops = 2 * D * (H * K * 2)
            kv_flops = 2 * D * (2 * G * K)
            attn_dot = 2 * T * H * K  # causal attention: halved (weightless)
            o_flops = 2 * H * K * D
            weighted_layers += q_flops + kv_flops + o_flops
            attention += attn_dot
        else:
            conv_dim = key_dim * 2 + value_dim
            in_proj_qkv = 2 * D * conv_dim
            in_proj_z = 2 * D * value_dim
            in_proj_b = 2 * D * nv
            in_proj_a = 2 * D * nv
            out_proj = 2 * value_dim * D
            delta_rule_per_token = 2 * nv * (ak * av)  # weightless state recurrence
            weighted_layers += in_proj_qkv + in_proj_z + in_proj_b + in_proj_a + out_proj
            attention += delta_rule_per_token

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
            weighted_layers += router_flops + routed_flops + shared_flops + shared_gate_flops
        else:
            F_dense = cfg.intermediate_size
            weighted_layers += 2 * 3 * D * F_dense

    head = 2 * D * V
    return ForwardFlops(weighted_layers=weighted_layers, attention=attention, head=head)


def qwen3_vl_vision_flops(cfg: Qwen3VLConfig, image_grid_thw: Any | None) -> VisionFlops:
    """Qwen3-VL vision-tower forward matmul FLOPs for one step (no fwd/bwd multiplier).

    Vision attention is block-diagonal across images (sum_i N_i^2). ``block_forward``
    is the jax.remat'd transformer-block portion, used for HFU on a trained tower.
    """
    if image_grid_thw is None:
        return VisionFlops(forward=0, block_forward=0)

    grid_N3 = np.asarray(image_grid_thw, dtype=np.int64)
    if grid_N3.size == 0:
        return VisionFlops(forward=0, block_forward=0)
    if grid_N3.ndim != 2 or grid_N3.shape[1] != 3:
        raise ValueError(
            f"Expected image_grid_thw with shape (num_images, 3), got {grid_N3.shape}."
        )

    vis = cfg.vision
    merge = vis.spatial_merge_size

    per_image_tokens = grid_N3[:, 0] * grid_N3[:, 1] * grid_N3[:, 2]
    total_tokens = int(np.sum(per_image_tokens))
    sum_sq_tokens = int(np.sum(per_image_tokens * per_image_tokens))
    merged_tokens = int(np.sum(grid_N3[:, 0] * (grid_N3[:, 1] // merge) * (grid_N3[:, 2] // merge)))
    if total_tokens <= 0 or merged_tokens <= 0:
        return VisionFlops(forward=0, block_forward=0)

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
    return VisionFlops(forward=forward, block_forward=block_flops)


def qwen3_vl_vision_training_flops(
    cfg: Qwen3VLConfig, image_grid_thw: Any | None, *, vision_trainable: bool = True
) -> int:
    """Vision-tower model FLOPs for one step: x3 if trained, x1 if frozen (no backward built)."""
    forward = qwen3_vl_vision_flops(cfg, image_grid_thw).forward
    return forward * (TRAINING_FLOP_MULTIPLIER if vision_trainable else 1)


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


def per_device_step_flops(
    cfg: RunPerfConfig,
    seq_len: int,
    batch_size: int,
    image_grid_thw: Any | None = None,
    *,
    base_weights_trainable: bool,
    vision_trainable: bool,
    decoder_remat: bool,
    vision_remat: bool = True,
) -> StepFlops:
    """Per-device model (MFU) and hardware (HFU) FLOPs for one step.

    base_weights_trainable is False under LoRA (frozen base weights, adapters omitted).
    """
    fwd = forward_flops_per_token(cfg, seq_len)
    tokens = seq_len * batch_size
    model = fwd.model_flops(base_weights_trainable=base_weights_trainable) * tokens
    hardware = (
        fwd.hardware_flops(base_weights_trainable=base_weights_trainable, decoder_remat=decoder_remat)
        * tokens
    )

    if isinstance(cfg, Qwen3VLConfig):
        vis = qwen3_vl_vision_flops(cfg, image_grid_thw)
        if vision_trainable:
            vision_model = TRAINING_FLOP_MULTIPLIER * vis.forward
            vision_recompute = vis.block_forward if vision_remat else 0
        else:
            # Frozen tower: forward-only, no backward and therefore no recompute.
            vision_model = vis.forward
            vision_recompute = 0
        model += vision_model
        hardware += vision_model + vision_recompute

    n_devices = max(1, jax.device_count())
    return StepFlops(model=model / n_devices, hardware=hardware / n_devices)


def step_metrics(
    step_flops: StepFlops,
    step_delta: datetime.timedelta,
    global_tokens_per_step: int,
    peak_tflops: float | None,
) -> dict[str, float]:
    """Tokens/s, model/hardware TFLOPS/device, and MFU/HFU from step timing.

    tflops_per_device aliases model_tflops_per_device (mfu == tflops_per_device / peak).
    """
    sec = step_delta.total_seconds()
    if sec <= 0:
        return {
            "step_time_s": 0.0,
            "global_tokens_per_sec": 0.0,
            "tokens_per_sec_per_device": 0.0,
            "model_tflops_per_device": 0.0,
            "hardware_tflops_per_device": 0.0,
            "tflops_per_device": 0.0,
            "mfu": 0.0,
            "hfu": 0.0,
        }
    n_devices = jax.device_count()
    global_tokens_per_sec = global_tokens_per_step / sec
    tokens_per_sec_per_device = global_tokens_per_sec / n_devices
    model_tflops = step_flops.model / sec / 1e12
    hardware_tflops = step_flops.hardware / sec / 1e12
    mfu = (model_tflops / peak_tflops) if peak_tflops else 0.0
    hfu = (hardware_tflops / peak_tflops) if peak_tflops else 0.0
    return {
        "step_time_s": sec,
        "global_tokens_per_sec": global_tokens_per_sec,
        "tokens_per_sec_per_device": tokens_per_sec_per_device,
        "model_tflops_per_device": model_tflops,
        "hardware_tflops_per_device": hardware_tflops,
        "tflops_per_device": model_tflops,
        "mfu": mfu,
        "hfu": hfu,
    }


def maybe_log_step_metrics(
    step_to_log: int,
    metrics_to_log: dict[str, Any],
    step_delta: datetime.timedelta,
    *,
    is_primary_process: bool,
    log_every: int,
    force: bool = False,
    step_flops: StepFlops,
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
    host_metrics["step"] = step_to_log
    if batch_size > 0:
        host_metrics["total_samples"] = step_to_log * batch_size
    host_metrics.update(
        step_metrics(step_flops, step_delta, global_tokens_per_step, peak_tflops)
    )
    # Checked after step_metrics has filled its keys in, so a rename there is a
    # missing metric and not a logged 0.0. An MFU figure was quoted for weeks
    # before being found to be a blend of two things.
    required = (
        "loss",
        "grad_norm",
        "lr",
        "supervised_tokens",
        "total_tokens",
        "step_time_s",
        "global_tokens_per_sec",
        "tokens_per_sec_per_device",
        "model_tflops_per_device",
        "hardware_tflops_per_device",
        "mfu",
        "hfu",
    )
    missing = [k for k in required if k not in host_metrics]
    if missing:
        raise KeyError(f"Missing required metrics for logging: {missing}")

    if wandb_run is not None and is_primary_process:
        _SKIP = {"step"}
        wandb_run.log(
            {f"train/{k}": v for k, v in host_metrics.items() if k not in _SKIP},
            step=step_to_log,
        )

    if is_primary_process:
        print(
            f"time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"step={step_to_log} "
            f"loss={host_metrics['loss']:.4f} "
            f"grad_norm={host_metrics['grad_norm']:.4f} "
            f"train/total_samples={host_metrics.get('total_samples', 0)} "
            f"train/global_tokens_per_sec={host_metrics['global_tokens_per_sec']:.0f} "
            f"train/step_time_s={host_metrics['step_time_s']:.2f}s "
            f"train/supervised_tokens={host_metrics['supervised_tokens']:.0f} "
            f"train/total_tokens={host_metrics['total_tokens']:.0f} "
            f"train/lr={host_metrics['lr']:.2e} "
            f"train/model_tflops_per_device={host_metrics['model_tflops_per_device']:.2f} "
            f"train/hardware_tflops_per_device={host_metrics['hardware_tflops_per_device']:.2f} "
            f"train/mfu={host_metrics['mfu'] * 100:.1f}% "
            f"train/hfu={host_metrics['hfu'] * 100:.1f}% "
            f"train/tok/s/dev={host_metrics['tokens_per_sec_per_device']:.0f} ",
            flush=True,
        )

    return host_metrics
