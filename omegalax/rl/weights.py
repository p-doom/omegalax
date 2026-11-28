"""Policy weights in one process: trainer copy on a device sub-mesh, engine replicas, sync."""

from __future__ import annotations

import dataclasses
import time
from typing import Sequence

import jax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from omegalax.distributed.mesh import mesh_rules
from omegalax.engine import Engine
from omegalax.engine.engine import _inference_dtypes
from omegalax.models.qwen3.loader import create_qwen3_from_safetensors
from omegalax.models.qwen3_5.loader import create_qwen3_5_from_safetensors
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.models.sharding_runtime import _finalize_q_shardings
from omegalax.models.qwen3.model import Qwen3
from omegalax.text import api as text_api

_AXES = ("tp", "fsdp", "dp")


def sub_mesh(devices: Sequence[jax.Device], tp: int, fsdp: int, dp: int) -> jax.sharding.Mesh:
    if tp * fsdp * dp != len(devices):
        raise ValueError(f"Mesh ({tp}, {fsdp}, {dp}) does not match {len(devices)} devices.")
    return jax.make_mesh((tp, fsdp, dp), _AXES, devices=list(devices))


def _model_class(cfg):
    return Qwen3_5ForCausalLM if isinstance(cfg, text_api.Qwen3_5TextConfig) else Qwen3


def load_causal_lm(path: str, mesh: jax.sharding.Mesh, dtype) -> tuple[nnx.Module, object]:
    """Text-only causal LM from a HF safetensors directory, sharded on ``mesh``, computing in ``dtype``."""
    stored = text_api.align_config_to_mesh(text_api.resolve_config(path), mesh)
    cfg = dataclasses.replace(stored, dtype=dtype)
    sizes = {a: int(mesh.shape[a]) for a in _AXES}
    with jax.set_mesh(mesh):
        if isinstance(cfg, text_api.Qwen3_5TextConfig):
            vlm, _ = create_qwen3_5_from_safetensors(
                path, tp_size=sizes["tp"], fsdp_size=sizes["fsdp"], dp_size=sizes["dp"]
            )
            with mesh_rules(mesh):
                abstract = nnx.eval_shape(lambda: Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(0)))
            state = nnx.state(abstract)
            nnx.replace_by_pure_dict(
                state,
                {
                    "text": nnx.to_pure_dict(nnx.state(vlm.text)),
                    "lm_head": nnx.to_pure_dict(nnx.state(vlm.lm_head)),
                },
            )
            model = nnx.merge(nnx.graphdef(abstract), state)
            _finalize_q_shardings(model, mesh)
        else:
            if dtype != stored.dtype:
                raise ValueError(f"Qwen3 checkpoints compute in their stored dtype {stored.dtype}, not {dtype}.")
            model = create_qwen3_from_safetensors(
                path, tp_size=sizes["tp"], fsdp_size=sizes["fsdp"], dp_size=sizes["dp"]
            )
    return model, cfg


def _abstract(cfg, mesh: jax.sharding.Mesh) -> tuple[nnx.Module, dict]:
    """Abstract model for ``cfg`` on ``mesh`` and its pure-dict ``NamedSharding`` tree."""
    with mesh_rules(mesh):
        abstract = nnx.eval_shape(lambda: _model_class(cfg)(cfg, rngs=nnx.Rngs(0)))
        specs = nnx.to_pure_dict(nnx.get_partition_spec(nnx.state(abstract)))
    return abstract, jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs)


def engine_replica(params: dict, cfg, mesh: jax.sharding.Mesh) -> tuple[nnx.Module, dict]:
    """A copy of ``params`` (``WeightSync.gather`` output) as a model on ``mesh``, with the
    sharding tree that later weight pushes to it must use."""
    cfg = text_api.align_config_to_mesh(cfg, mesh)
    abstract, shardings = _abstract(cfg, mesh)
    state = nnx.state(abstract)
    nnx.replace_by_pure_dict(state, jax.device_put(params, shardings))
    model = nnx.merge(nnx.graphdef(abstract), state)
    _finalize_q_shardings(model, mesh)
    return model, shardings


class WeightSync:
    """Trainer params -> engines. One jitted program casts every leaf to the engine's inference
    dtype and replicates it on the trainer mesh; one ``device_put`` per engine then copies it
    device to device. (An FSDP-sharded array crossing meshes goes through the host, ~30x slower.)
    Engines built by ``engine_replica`` hold the same dtypes, so their own cast is the identity.
    """

    def __init__(self, graphdef: nnx.GraphDef, params: nnx.State):
        mesh = jax.tree.leaves(params)[0].sharding.mesh
        pure = nnx.to_pure_dict(params)
        model = nnx.merge(graphdef, params)
        dtypes = jax.tree.unflatten(jax.tree.structure(pure), _inference_dtypes(model, nnx.state(model)))
        self.mesh = mesh
        self._gather = jax.jit(
            lambda p: jax.tree.map(lambda x, d: x.astype(d), p, dtypes),
            out_shardings=jax.tree.map(lambda _: NamedSharding(mesh, P()), pure),
        )
        self.engines: list[Engine] = []
        self.shardings: list[dict] = []

    def gather(self, params: nnx.State) -> dict:
        with jax.set_mesh(self.mesh):
            return self._gather(nnx.to_pure_dict(params))

    def add(self, engine: Engine, shardings: dict) -> None:
        self.engines.append(engine)
        self.shardings.append(shardings)

    def push(self, params: nnx.State, version: int) -> dict[str, float]:
        t0 = time.perf_counter()
        inference = jax.block_until_ready(self.gather(params))
        t1 = time.perf_counter()
        futures = [
            engine.update_weights(jax.device_put(inference, shardings), version)
            for engine, shardings in zip(self.engines, self.shardings)
        ]
        for f in futures:
            f.result()
        return {"sync/gather_s": t1 - t0, "sync/total_s": time.perf_counter() - t0}
