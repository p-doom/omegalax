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
    sharding_on_memory_kind,
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

    Staging correctness note: inside a jitted step the traced opt_state arrays
    report ``sharding is None``, so ``x.sharding.with_memory_kind(...)`` cannot
    be used to drive the staging from the tracer. We instead capture each
    moment leaf's *concrete device* ``NamedSharding`` at
    :meth:`enable_state_offload` time (outside jit) and stage with those stored
    shardings: ``device_put(x, stored.with_memory_kind("device"))`` on entry and
    ``...with_memory_kind("pinned_host")`` on the freshly-computed moments. The
    stored shardings are static (closed over as constants in the trace), so the
    ``device_put`` emits a real H2D/D2H transfer even on a memory-kind-less
    tracer. The jit input/output for the donated optimizer carries the host
    memory kind (opt_state resides on host before and after the step).
    """

    def enable_state_offload(self) -> None:
        """Move the optimizer moment buffers to ``pinned_host`` and arm staging.

        Call this at build time, BEFORE any checkpoint restore, so the restored
        shardings (captured via ``value.sharding``, which includes the memory
        kind) match the host-resident optimizer state. Idempotent (re-capturing
        the *device* shardings from the current on-device state before moving to
        host, so a second call after a host round-trip still records device
        shardings).

        The offload flag and the captured device-sharding pytree are stored as
        plain Python attributes (not NNX variables), so they are invisible to
        ``nnx.state`` / checkpointing and do not perturb the graphdef used for
        jit-cache keys beyond selecting the staging branch.
        """
        # Capture each opt_state leaf's CONCRETE sharding as it currently is,
        # then normalize to the "device" memory kind. These stored shardings are
        # what ``update`` uses to stage across the jit boundary (a tracer's
        # ``.sharding`` is None inside jit, so we cannot read it there). Capture
        # from ``nnx.pure(self.opt_state)`` — the exact structure ``update``
        # feeds to / receives from ``tx.update`` (an optax pytree of
        # tuples/namedtuples), so the stored sharding tree maps 1:1 onto both the
        # entry ``opt_state_arrays`` and the exit ``new_opt_state`` in
        # :meth:`_stage_opt_state`.
        opt_state_pure = nnx.pure(self.opt_state)
        device_shardings = jax.tree.map(
            lambda a: sharding_on_memory_kind(getattr(a, "sharding", None), DEVICE_MEMORY_KIND),
            opt_state_pure,
        )
        object.__setattr__(self, "_opt_state_device_shardings", device_shardings)
        # Re-place every ``opt_state`` array leaf on host. The large fp32 moment
        # buffers (mu/nu) are what we care about; the handful of tiny scalar
        # counters inside opt_state (e.g. MultiSteps' mini-step) ride along to
        # keep the whole subtree on one memory kind, and ``update`` stages the
        # entire subtree back to device for the step. (The optimizer's own
        # ``self.step`` is separate from opt_state and stays on device.)
        host_state = place_tree_on_memory_kind(nnx.state(self.opt_state), HOST_MEMORY_KIND)
        nnx.update(self.opt_state, host_state)
        object.__setattr__(self, "_offload_optimizer_state", True)

    @property
    def offload_optimizer_state(self) -> bool:
        return bool(getattr(self, "_offload_optimizer_state", False))

    def _stage_opt_state(self, opt_state_arrays, memory_kind: str):
        """Place ``opt_state_arrays`` on ``memory_kind`` using stored shardings.

        Uses the concrete device shardings captured in
        :meth:`enable_state_offload` (rewritten to ``memory_kind``) so the
        ``device_put`` fires even on jit tracers whose ``.sharding`` is None.
        """
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
            # Stage the host-resident moments onto the device for the step. This
            # is a pure memory-kind move (device_put with the stored device
            # sharding): no arithmetic, no dtype or shape change. Inside a jitted
            # step XLA lowers it to an async H2D copy it can overlap with the
            # forward/backward compute.
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
            # Place the freshly-computed moments back on host. Under jit this is
            # the update's out-placement, so XLA can overlap the D2H copy with
            # the tail of the step; between steps the moments then reside on
            # host, off the accelerator's HBM.
            new_opt_state = self._stage_opt_state(new_opt_state, HOST_MEMORY_KIND)

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1
