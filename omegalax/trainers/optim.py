"""Mixed-precision optimizer (T5X-style: bf16 params, fp32 state & accumulation).

Optionally offloads the fp32 Adam moments to host between steps; see
:mod:`omegalax.trainers.offload`.
"""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from omegalax.trainers.offload import (
    DEVICE_MEMORY_KIND,
    HOST_MEMORY_KIND,
    place_tree_on_memory_kind,
    sharding_on_memory_kind,
)


class MixedPrecisionOptimizer(nnx.ModelAndOptimizer):
    """AdamW-style optimizer with T5X mixed-precision semantics: fp32 optimizer
    state, gradients upcast to fp32, ``param + delta`` and weight-decay in fp32,
    then cast back to the param dtype (e.g. bf16).

    With ``offload_optimizer_state`` (see :meth:`enable_state_offload`) the fp32
    moments live on host between steps and stage to device only for the update.
    """

    def enable_state_offload(self) -> None:
        """Move the moment buffers to ``pinned_host`` and arm staging.

        Call before any checkpoint restore so restored shardings match; idempotent. The
        offload flag and captured shardings are plain attributes, invisible to checkpointing.
        """
        # Capture concrete device shardings outside jit (a tracer's .sharding is None)
        # so _stage_opt_state can re-place the moments across the jit boundary.
        opt_state_pure = nnx.pure(self.opt_state)
        device_shardings = jax.tree.map(
            lambda a: sharding_on_memory_kind(getattr(a, "sharding", None), DEVICE_MEMORY_KIND),
            opt_state_pure,
        )
        object.__setattr__(self, "_opt_state_device_shardings", device_shardings)
        # Move the whole opt_state subtree to host (self.step is separate, stays on device).
        host_state = place_tree_on_memory_kind(nnx.state(self.opt_state), HOST_MEMORY_KIND)
        nnx.update(self.opt_state, host_state)
        object.__setattr__(self, "_offload_optimizer_state", True)

    @property
    def offload_optimizer_state(self) -> bool:
        return bool(getattr(self, "_offload_optimizer_state", False))

    def _stage_opt_state(self, opt_state_arrays, memory_kind: str):
        """Place ``opt_state_arrays`` on ``memory_kind`` using the captured shardings,
        so ``device_put`` fires even on a jit tracer (whose ``.sharding`` is None)."""
        device_shardings = self._opt_state_device_shardings
        return jax.tree.map(
            lambda x, shd: (
                jax.device_put(x, sharding_on_memory_kind(shd, memory_kind))
                if shd is not None
                else x
            ),
            opt_state_arrays,
            device_shardings,
        )

    def update(self, grads, **kwargs):  # type: ignore[override]
        """Compute and apply one optimizer step with fp32 accumulation."""
        offload = self.offload_optimizer_state

        param_arrays = nnx.pure(nnx.state(self.model, self.wrt))
        grad_arrays = nnx.pure(nnx.state(grads, self.wrt))
        opt_state_arrays = nnx.pure(self.opt_state)

        if offload:
            # Stage the host-resident moments to device for the update.
            opt_state_arrays = self._stage_opt_state(opt_state_arrays, DEVICE_MEMORY_KIND)

        fp32_grads = jax.tree.map(lambda g: g.astype(jnp.float32), grad_arrays)

        updates, new_opt_state = self.tx.update(
            fp32_grads, opt_state_arrays, param_arrays, **nnx.pure(kwargs)
        )

        new_params = jax.tree.map(
            lambda p, u: (p.astype(jnp.float32) + u).astype(p.dtype),
            param_arrays,
            updates,
        )

        if offload:
            # Place the fresh moments back on host between steps.
            new_opt_state = self._stage_opt_state(new_opt_state, HOST_MEMORY_KIND)

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1
