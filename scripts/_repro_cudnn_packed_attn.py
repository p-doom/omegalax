"""Standalone repro for cuDNN packed/THD attention on Ampere (sm80).

Isolates the failing case from the full VLM train step: calls the same cuDNN
packed kernel the vision encoder uses (q_offsets/kv_offsets, NO_MASK, BTNH) and
sweeps {bf16, fp16} x {forward-only, forward+backward} so we know exactly which
combination cuDNN rejects on this GPU. Mirrors
``omegalax/models/qwen3_vl/vision.py:_cudnn_packed_vision_attention_local``.

Run on an A100 with cuDNN >= 9.18.1 loaded (LD_LIBRARY_PATH), e.g. via the
companion sbatch. Vision encoder shape: num_heads=16, head_dim=64.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

# Apply the Hopper-gate relaxation (no-op on Hopper; needed for sm80 + cuDNN>=9.18.1).
import omegalax.compat.cudnn_ampere_packed  # noqa: F401
from jax._src.lib import cuda_versions as _cv
from jax._src.cudnn.fused_attention_stablehlo import (
    MaskType,
    dot_product_attention as cudnn_attn,
)

H, K = 16, 64          # vision: num_heads=16, head_dim=64
SEGS = [128, 96]       # two packed image segments (is_packed -> True)
N = int(sum(SEGS))
SCALE = 1.0 / np.sqrt(K)

cu = jnp.asarray([0, *np.cumsum(SEGS)], jnp.int32)      # (M+1,)
seqlens = jnp.asarray(SEGS, jnp.int32)                  # (M,)


def _attn(q_NHK, k_NHK, v_NHK):
    out = cudnn_attn(
        q_NHK[None], k_NHK[None], v_NHK[None],
        q_seqlen=seqlens[None], kv_seqlen=seqlens[None],
        q_offsets=cu[None], kv_offsets=cu[None],
        scale=SCALE, mask_type=MaskType.NO_MASK, qkv_layout="BTNH",
    )
    return out[0]


def _make(dtype, seed):
    return jax.random.normal(jax.random.PRNGKey(seed), (N, H, K), dtype)


def run(dtype, train):
    q, k, v = _make(dtype, 0), _make(dtype, 1), _make(dtype, 2)
    if train:
        fn = jax.grad(lambda q, k, v: _attn(q, k, v).sum().astype(jnp.float32),
                      argnums=(0, 1, 2))
        jax.block_until_ready(fn(q, k, v))
    else:
        jax.block_until_ready(_attn(q, k, v))


def main():
    v = _cv.cudnn_get_version()
    dev = jax.devices()[0]
    print(f"device={dev.device_kind} cc={getattr(dev, 'compute_capability', '?')} "
          f"cudnn={v} (>=91801: {v >= 91801})", flush=True)
    for dtype in (jnp.bfloat16, jnp.float16):
        for train in (False, True):
            tag = f"{np.dtype(dtype).name:9s} {'fwd+bwd' if train else 'fwd    '}"
            try:
                run(dtype, train)
                print(f"  OK   {tag}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {tag} -> {type(e).__name__}: {str(e)[:160]}", flush=True)


if __name__ == "__main__":
    main()
