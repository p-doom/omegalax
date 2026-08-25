"""Mixed-precision optimizer (T5X-style: bf16 params, fp32 state & accumulation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx


@jax.jit
def initialize_gradient_sum(gradients):
    return jax.tree.map(lambda gradient: gradient.astype(jnp.float32), gradients)


@jax.jit(donate_argnums=0)
def accumulate_gradient_sum(gradient_sum, gradients):
    return jax.tree.map(
        lambda total, gradient: total + gradient.astype(jnp.float32),
        gradient_sum,
        gradients,
    )


@nnx.jit(donate_argnums=(0, 1))
def apply_normalized_gradient_sum(
    optimizer,
    gradient_sum,
    supervised_tokens,
    loss_sum,
):
    normalized_gradients = jax.tree.map(
        lambda gradient: gradient / jnp.maximum(supervised_tokens, 1.0),
        gradient_sum,
    )
    grad_norm = optax.tree.norm(normalized_gradients)
    healthy = (
        jnp.isfinite(supervised_tokens)
        & (supervised_tokens > 0)
        & jnp.isfinite(loss_sum)
        & (loss_sum >= 0)
        & jnp.isfinite(grad_norm)
    )
    optimizer.update(normalized_gradients)
    return grad_norm, healthy


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
