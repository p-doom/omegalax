"""Mixed-precision optimizer (T5X-style: bf16 params, fp32 state & accumulation).

Optionally offloads the fp32 optimizer moments (Adam ``mu``/``nu``) to host
memory (``pinned_host``) between steps, staging them to ``device`` only for the
duration of the update. This frees accelerator memory (HBM) for the model /
activations. On a coherent-host platform (GH200 Grace + NVLink-C2C) the staging
is cheap and XLA can overlap it with compute; on PCIe A100/H100 it works but is
transfer-bound. Offload only changes the *memory kind* of the moment buffers —
never the arithmetic — so the update is bit-identical with offload on vs off.
See :mod:`omegalax.trainers.offload` for the platform gating and helpers.
"""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from omegalax.trainers.offload import (
    DEVICE_MEMORY_KIND,
    HOST_MEMORY_KIND,
    place_tree_on_memory_kind,
)


class MixedPrecisionOptimizer(nnx.ModelAndOptimizer):
    """AdamW-style optimizer with T5X mixed-precision semantics.

    * Optimizer state (momentum, second-moment) is stored in fp32.
    * Gradients are upcast to fp32 before the optimizer step.
    * The parameter update (``param + delta``) is computed in fp32, then
      cast back to the original param dtype (e.g. bf16).
    * Weight-decay is applied to fp32 params.

    When ``offload_optimizer_state`` is set (via :meth:`enable_state_offload`,
    typically only on a coherent-host platform — see
    :mod:`omegalax.trainers.offload`), the fp32 moment buffers live on
    ``pinned_host`` between steps and are staged to ``device`` only for the
    update, then placed back on host. This is memory-kind-only movement: shapes,
    dtypes, partition specs and the arithmetic are all unchanged, so the update
    is bit-identical to the non-offloaded path.
    """

    def enable_state_offload(self) -> None:
        """Move the optimizer moment buffers to ``pinned_host`` and arm staging.

        Call this at build time, BEFORE any checkpoint restore, so the restored
        shardings (captured via ``value.sharding``, which includes the memory
        kind) match the host-resident optimizer state. Idempotent.

        The offload flag is stored as a plain Python attribute (not an NNX
        variable), so it is invisible to ``nnx.state`` / checkpointing and does
        not affect the graphdef used for jit-cache keys beyond selecting the
        staging branch.
        """
        # Re-place every ``opt_state`` array leaf on host. ``self.opt_state`` is
        # an NNX node; ``nnx.state`` gives its array leaves, which we re-place
        # and write back. The large fp32 moment buffers (mu/nu) are what we care
        # about; the handful of tiny scalar counters inside opt_state (e.g.
        # MultiSteps' mini-step) ride along to keep the whole subtree on one
        # memory kind, and ``update`` stages the entire subtree back to device
        # for the step. (The optimizer's own ``self.step`` is separate from
        # opt_state and stays on device.)
        host_state = place_tree_on_memory_kind(nnx.state(self.opt_state), HOST_MEMORY_KIND)
        nnx.update(self.opt_state, host_state)
        object.__setattr__(self, "_offload_optimizer_state", True)

    @property
    def offload_optimizer_state(self) -> bool:
        return bool(getattr(self, "_offload_optimizer_state", False))

    def update(self, grads, **kwargs):  # type: ignore[override]
        """Compute and apply one optimizer step with fp32 accumulation."""
        offload = self.offload_optimizer_state

        param_arrays = nnx.pure(nnx.state(self.model, self.wrt))
        grad_arrays = nnx.pure(nnx.state(grads, self.wrt))
        opt_state_arrays = nnx.pure(self.opt_state)

        if offload:
            # Stage the host-resident moments onto the device for the step. This
            # is a pure memory-kind move (device_put): no arithmetic, no dtype
            # or shape change. Inside a jitted step XLA turns this into an async
            # H2D copy it can overlap with the forward/backward compute.
            opt_state_arrays = place_tree_on_memory_kind(opt_state_arrays, DEVICE_MEMORY_KIND)

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
            # Place the freshly-computed moments back on host. Under jit this is
            # the update's out-placement, so XLA can overlap the D2H copy with
            # the tail of the step; between steps the moments then reside on
            # host, off the accelerator's HBM.
            new_opt_state = place_tree_on_memory_kind(new_opt_state, HOST_MEMORY_KIND)

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1
