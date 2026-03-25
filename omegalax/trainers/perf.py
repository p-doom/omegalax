"""Throughput metrics for training: FLOP counting, step timing, and MFU.

Uses the MaxText-style approach: attention FLOPs for causal masking are
halved (2*T*H*K per token instead of 4*T*H*K) since flash attention kernels
skip masked blocks. Vision encoder attention (bidirectional) uses full FLOPs.
"""

from __future__ import annotations

import datetime
from typing import Any, Union

import jax
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
    cfg: Qwen3VLConfig, image_grid_thw: Any | None
) -> int:
    """Theoretical Qwen3-VL vision-tower FLOPs for one training step (x3).

    Counts matmuls only and matches the current implementation in
    ``omegalax.models.qwen3_vl.vision``:
    - patch embed and patch mergers are linear layers;
    - vision attention materializes the full ``(sum N_i)^2`` attention matrix
      across all images in the batch before masking by ``vision_cu_seqlens``.
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

    total_tokens = int(np.sum(grid_N3[:, 0] * grid_N3[:, 1] * grid_N3[:, 2]))
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
    attn_dot_flops = 4 * total_tokens * total_tokens * H * K
    o_proj_flops = 2 * total_tokens * D * D
    mlp_flops = 2 * total_tokens * D * F + 2 * total_tokens * F * D
    block_flops = vis.depth * (qkv_flops + attn_dot_flops + o_proj_flops + mlp_flops)

    merger_dim = D * (merge**2)
    merger_fc1_flops = 2 * merged_tokens * merger_dim * merger_dim
    merger_fc2_flops = 2 * merged_tokens * merger_dim * vis.out_hidden_size
    num_mergers = 1 + len(vis.deepstack_visual_indexes)
    merger_flops = num_mergers * (merger_fc1_flops + merger_fc2_flops)

    forward = patch_embed_flops + block_flops + merger_flops
    return forward * TRAINING_FLOP_MULTIPLIER


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
) -> float:
    """Total training FLOPs per step, divided by device count.

    For Qwen3-VL, ``image_grid_thw`` adds the vision-tower FLOPs for the
    concrete batch. Text-decoder FLOPs are still computed from the padded
    ``seq_len`` and ``batch_size``.
    """
    total = training_flops_per_token(cfg, seq_len) * seq_len * batch_size
    if isinstance(cfg, Qwen3VLConfig):
        total += qwen3_vl_vision_training_flops(cfg, image_grid_thw)
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
    tb_writer: Any = None,
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
    host_metrics.update(
        step_metrics(per_device_flops, step_delta, global_tokens_per_step, peak_tflops)
    )

    if tb_writer is not None and is_primary_process:
        _TB_SKIP = {"step"}
        for key, val in host_metrics.items():
            if key not in _TB_SKIP:
                tb_writer.add_scalar(f"train/{key}", val, step_to_log)
        tb_writer.flush()

    return host_metrics
