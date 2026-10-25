"""Four-GPU pretrained gradient through the production GDN handoff, with detach control."""

import argparse
import json
import os
from pathlib import Path
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]

import omegalax

assert Path(omegalax.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1], (
    omegalax.__file__
)
from flax import nnx
import jax
import jax.numpy as jnp
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.qwen3_5.params import create_qwen3_5_from_safetensors
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.trainers.pretrain import _statepassing_hidden_states, PretrainMode


def measure_state_gradient(model, cfg, batch):
    results = {}
    T = batch["token_ids_BCT"].shape[2]
    boundary = (batch["token_ids_BCT"].shape[1] - 1) * T
    for mode in ("full", "detached_control"):
        chunks = None if mode == "full" else 1

        @nnx.jit
        def probe(model, delta):
            def loss(model, delta):
                embedding = model.text.embedder.embedding
                original = embedding[...]
                # Token100 occurs ONLY before the handoff; later loss depends on it ONLY through GDN.
                embedding[...] = original.at[100].add(delta)
                hidden, _ = _statepassing_hidden_states(
                    model,
                    batch,
                    cfg,
                    0,
                    PretrainMode.STATEPASSING_BPTT,
                    bptt_chunks=chunks,
                    pass_gdn_state=True,
                    pass_conv_state=False,
                    pass_rope_positions=False,
                )
                embedding[...] = original
                return hidden[:, boundary : boundary + 256, 0].astype(jnp.float32).sum()

            return nnx.value_and_grad(loss, argnums=1)(model, delta)

        value, gradient = probe(model, jnp.asarray(0.0, jnp.float32))
        results[mode] = {
            "loss": float(value),
            "early_embedding_to_late_hidden_gradient": float(gradient),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-hf-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment-length", type=int, required=True, choices=[512, 8192])
    args = parser.parse_args()
    from absl import flags

    # Tokamax reads Abseil defaults lazily; our CLI belongs to argparse.
    flags.FLAGS([str(__file__)])
    assert len(jax.devices()) == 4 and all(d.platform == "gpu" for d in jax.devices())
    mesh = ensure_mesh(tp_size=1, fsdp_size=4, dp_size=1)
    model, cfg = create_qwen3_5_from_safetensors(
        args.init_hf_model_path, tp_size=1, fsdp_size=4, dp_size=1, text_only=True
    )
    cfg = text_api.align_config_to_mesh(cfg.text_config, mesh)
    set_attn_backend(model, text_backend="mosaic_gpu")
    T = args.segment_length
    # Probe the actual handoff at absolute context position16384, beyond the old16k limit.
    C = 16384 // T + 1
    tokens = jnp.full((4, C, T), 200, jnp.int32).at[:, C - 2, :].set(100)
    reset = jnp.zeros((4, C), jnp.bool_).at[:, 0].set(True)
    batch = text_api.shard_batch_dict(
        {
            "token_ids_BCT": tokens,
            "attention_mask_BCT": jnp.ones((4, C, T), jnp.int32),
            "reset_state_BC": reset,
        },
        cfg,
        mesh,
    )
    with mesh_rules(mesh):
        results = measure_state_gradient(model, cfg, batch)
    assert all(jnp.isfinite(x["early_embedding_to_late_hidden_gradient"]) for x in results.values())
    assert abs(results["full"]["early_embedding_to_late_hidden_gradient"]) > 0
    assert results["detached_control"]["early_embedding_to_late_hidden_gradient"] == 0
    with Path(args.output).open("x") as stream:
        json.dump(
            {
                "status": "PASS",
                "segment_length": T,
                "production_path": "_statepassing_hidden_states",
                "results": results,
                "handoff_context_position": 16384,
                "scope": "GDN handoff at16k through the full production chain; full55k updates checked separately",
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
