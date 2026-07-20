"""Shared fixtures for the state-usage eval tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig


def tiny_hybrid_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention", "linear_attention", "full_attention"),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(1, 0, 0),
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        intermediate_size=32,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.float32,
    )


def two_document_chain_arrays() -> dict[str, np.ndarray]:
    token_ids_BCT = np.asarray(
        [
            [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 0, 0]],
            [[41, 42, 43, 44], [51, 52, 53, 54], [61, 62, 63, 0]],
        ],
        dtype=np.int32,
    )
    attention_mask_BCT = (token_ids_BCT != 0).astype(np.int32)
    return {
        "token_ids_BCT": token_ids_BCT,
        "attention_mask_BCT": attention_mask_BCT,
        "loss_mask_BCT": attention_mask_BCT.copy(),
        "chunk_indices_BC": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int32),
    }


def four_document_chain_arrays() -> dict[str, np.ndarray]:
    base = two_document_chain_arrays()
    extra_tokens = np.asarray(
        [
            [[15, 16, 17, 18], [25, 26, 27, 28], [33, 34, 35, 0]],
            [[37, 38, 39, 40], [45, 46, 47, 48], [55, 56, 0, 0]],
        ],
        dtype=np.int32,
    )
    token_ids_BCT = np.concatenate([base["token_ids_BCT"], extra_tokens], axis=0)
    attention_mask_BCT = (token_ids_BCT != 0).astype(np.int32)
    return {
        "token_ids_BCT": token_ids_BCT,
        "attention_mask_BCT": attention_mask_BCT,
        "loss_mask_BCT": attention_mask_BCT.copy(),
        "chunk_indices_BC": np.tile(
            np.arange(token_ids_BCT.shape[1], dtype=np.int32),
            (token_ids_BCT.shape[0], 1),
        ),
    }
