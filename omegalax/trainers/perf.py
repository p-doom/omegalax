"""Throughput metrics for training: FLOP counting, step timing, and MFU.

Uses the MaxText-style approach: theoretical model FLOPs
We do NOT halve attention FLOPs for causal masking; the current implementation
materializes the full attention matrix.
"""

from __future__ import annotations

import datetime
from typing import Any, Union

import jax

from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.qwen3.moe.config import Qwen3MoeConfig
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
    if isinstance(cfg, Qwen3MoeConfig):
        return _training_flops_per_token_qwen3_moe(cfg, seq_len)
    if isinstance(cfg, Qwen3Config):
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
    attn_dot_flops = 4 * T * H * K  # full attention, no causal halving
    o_proj_flops = 2 * H * K * D
    attn_per_layer = qkv_flops + attn_dot_flops + o_proj_flops
    mlp_per_layer = 2 * 3 * D * F  # SwiGLU: gate, up, down
    embedding_flops = 2 * D * V

    forward_per_token = L * (attn_per_layer + mlp_per_layer) + embedding_flops
    return forward_per_token * TRAINING_FLOP_MULTIPLIER


def _training_flops_per_token_qwen3_vl(cfg: Qwen3VLConfig, seq_len: int) -> int:
    """Qwen3-VL decoder FLOPs (same structure as Qwen3 MoE/dense)."""
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
    attn_dot_flops = 4 * T * H * K
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


def _training_flops_per_token_qwen3_moe(cfg: Qwen3MoeConfig, seq_len: int) -> int:
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
    attn_dot_flops = 4 * T * H * K
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
    E = cfg.num_experts
    k = cfg.num_experts_per_tok
    F_moe = cfg.moe_intermediate_size
    F_shared = cfg.shared_expert_intermediate_size

    # Linear attention / GatedDeltaNet dims
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    nv = cfg.linear_num_value_heads
    nk = cfg.linear_num_key_heads
    ak = cfg.linear_key_head_dim
    av = cfg.linear_value_head_dim

    layer_flops = 0
    for layer_idx, layer_type in enumerate(cfg.layer_types):
        if layer_type == "full_attention":
            # Q has 2x width for output gate
            q_flops = 2 * D * (H * K * 2)
            kv_flops = 2 * D * (2 * G * K)
            attn_dot = 4 * T * H * K
            o_flops = 2 * H * K * D
            layer_flops += q_flops + kv_flops + attn_dot + o_flops
        else:
            # linear_attention (GatedDeltaNet)
            conv_dim = key_dim * 2 + value_dim
            in_proj_qkv = 2 * D * conv_dim
            in_proj_z = 2 * D * value_dim
            in_proj_b = 2 * D * nv
            in_proj_a = 2 * D * nv
            out_proj = 2 * value_dim * D
            delta_rule_per_token = 2 * nv * (ak * av)
            layer_flops += in_proj_qkv + in_proj_z + in_proj_b + in_proj_a + out_proj + delta_rule_per_token

        # MoE MLP (all layers in Qwen3.5)
        router_flops = 2 * D * E
        gate_up_per_expert = 2 * (2 * F_moe) * D
        down_per_expert = 2 * F_moe * D
        routed_flops = k * (gate_up_per_expert + down_per_expert)
        shared_flops = 2 * 3 * D * F_shared
        shared_gate_flops = 2 * D * 1
        layer_flops += router_flops + routed_flops + shared_flops + shared_gate_flops

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
    cfg: RunPerfConfig, seq_len: int, batch_size: int
) -> float:
    """Total training FLOPs per step, divided by device count."""
    total = training_flops_per_token(cfg, seq_len) * seq_len * batch_size
    return total / max(1, jax.device_count())


def step_metrics(
    per_device_flops: float,
    step_delta: datetime.timedelta,
    tokens_per_step: int,
    peak_tflops: float | None,
) -> dict[str, float]:
    """Compute tokens/s, TFLOPS/device, and MFU from step timing."""
    sec = step_delta.total_seconds()
    if sec <= 0:
        return {
            "step_time_s": 0.0,
            "tokens_per_sec_per_device": 0.0,
            "tflops_per_device": 0.0,
            "mfu": 0.0,
        }
    n_devices = jax.device_count()
    tokens_per_sec_total = tokens_per_step / sec
    tokens_per_sec_per_device = tokens_per_sec_total / n_devices
    flops_per_sec_per_device = per_device_flops / sec
    tflops_per_device = flops_per_sec_per_device / 1e12
    mfu = (flops_per_sec_per_device / (peak_tflops * 1e12)) if peak_tflops else 0.0
    return {
        "step_time_s": sec,
        "tokens_per_sec_per_device": tokens_per_sec_per_device,
        "tflops_per_device": tflops_per_device,
        "mfu": mfu,
    }
