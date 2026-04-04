"""Mixed-precision optimizer (T5X-style: bf16 params, fp32 state & accumulation)."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import optax


class MixedPrecisionOptimizer(nnx.ModelAndOptimizer):
    """AdamW-style optimizer with T5X mixed-precision semantics.

    * Optimizer state (momentum, second-moment) is stored in fp32.
    * Gradients are upcast to fp32 before the optimizer step.
    * The parameter update (``param + delta``) is computed in fp32, then
      cast back to the original param dtype (e.g. bf16).
    * Weight-decay is applied to fp32 params.
    """

    def update(self, grads, **kwargs):  # type: ignore[override]
        """Compute and apply one optimizer step with fp32 accumulation."""
        param_arrays = nnx.pure(nnx.state(self.model, self.wrt))
        grad_arrays = nnx.pure(nnx.state(grads, self.wrt))
        opt_state_arrays = nnx.pure(self.opt_state)

        fp32_grads = jax.tree.map(lambda g: g.astype(jnp.float32), grad_arrays)

        updates, new_opt_state = self.tx.update(
            fp32_grads, opt_state_arrays, param_arrays, **nnx.pure(kwargs)
        )

        new_params = jax.tree.map(
            lambda p, u: (p.astype(jnp.float32) + u).astype(p.dtype),
            param_arrays,
            updates,
        )

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1
