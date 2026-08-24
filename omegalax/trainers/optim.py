"""Mixed-precision optimizer (T5X-style: bf16 params, fp32 state & accumulation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


@jax.jit(donate_argnums=0)
def accumulate_gradient_sum(gradient_sum, gradients):
    """Add one microbatch's unnormalized gradients to a donated device accumulator."""
    return jax.tree.map(jnp.add, gradient_sum, gradients)


@nnx.jit(donate_argnums=(0, 1))
def _apply_normalized_gradient_sum(optimizer, gradient_sum, supervised_tokens):
    normalized_gradients = jax.tree.map(lambda gradient: gradient / supervised_tokens, gradient_sum)
    grad_norm = optax.tree.norm(normalized_gradients)
    optimizer.update(normalized_gradients)
    return grad_norm


def apply_normalized_gradient_sum(optimizer, gradient_sum, supervised_tokens):
    """Normalize once by the global token count, then perform one optimizer update."""
    count = np.asarray(jax.device_get(supervised_tokens))
    if count.shape != () or not np.isfinite(count).item() or count.item() <= 0:
        raise ValueError(
            "Accumulated masked CE total supervised-token count must be positive and finite; "
            f"got {count!r}."
        )
    return _apply_normalized_gradient_sum(optimizer, gradient_sum, supervised_tokens)


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
