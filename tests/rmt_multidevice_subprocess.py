"""Run the RMT statepassing loss and gradients on one mesh layout in a fresh process."""

from __future__ import annotations

import json
import sys


def main() -> None:
    dp_size, fsdp_size, tp_size = (int(value) for value in sys.argv[1:4])

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import numpy as np

    from omegalax.distributed.mesh import ensure_mesh, mesh_rules
    from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
    from omegalax.models.sharding_runtime import set_attn_backend
    from omegalax.text import api as text_api
    from omegalax.trainers import pretrain as pretrain_trainer

    assert jax.device_count() == dp_size * fsdp_size * tp_size, jax.devices()
    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=("linear_attention", "full_attention", "linear_attention"),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(1, 1, 0),
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=4,
        intermediate_size=32,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        num_rmt_tokens=3,
    )
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    with mesh_rules(mesh):
        model, cfg = text_api.init_model(
            cfg, jax.random.PRNGKey(0), tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size
        )
        set_attn_backend(model, text_backend="xla")
        B, C, T = 4, 2, 4
        tokens = (np.arange(B * C * T).reshape(B, C, T) * 5 + 3) % 60 + 1
        batch = text_api.shard_batch_dict(
            {
                "token_ids_BCT": tokens.astype(np.int32),
                "attention_mask_BCT": np.ones((B, C, T), np.int32),
                "loss_mask_BCT": np.ones((B, C, T), np.int32),
                "chunk_idx_BC": np.broadcast_to(np.arange(C, dtype=np.int32), (B, C)),
                "reset_state_BC": np.array([[True, False]] * B),
            },
            cfg,
            mesh,
        )

        @nnx.jit
        def loss_and_grads(model, batch):
            def loss_fn(inner_model):
                return pretrain_trainer._statepassing_loss_stats(
                    inner_model,
                    batch,
                    cfg,
                    0,
                    pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                    bptt_chunks=2,
                    pass_gdn_state=True,
                    pass_conv_state=True,
                    pass_rope_positions=True,
                )

            (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            return loss, grads

        loss, grads = loss_and_grads(model, batch)
    memory_grad = np.asarray(nnx.to_pure_dict(grads)["rmt_memory"])
    print(
        json.dumps(
            {
                "loss": float(loss),
                "memory_grad_abs_max": float(np.abs(memory_grad).max()),
                "grads_finite": all(
                    bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(grads)
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
