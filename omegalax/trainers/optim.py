"""Mixed-precision optax helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def fp32_wrap(tx: optax.GradientTransformation) -> optax.GradientTransformation:
    """Wrap an optax optimizer so internal state and accumulation use fp32.

    Mirrors the T5X recipe: bf16 params, fp32 optimizer states, fp32 gradient
    accumulation, cast updates back to param dtype after the step.

    Gradients are explicitly cast to fp32 before the inner optimizer. This is
    necessary for the second-moment estimate (``grad²`` in bf16 flushes small
    values to zero).
    """

    def init_fn(params):
        fp32_params = jax.tree.map(lambda p: jnp.zeros(p.shape, dtype=jnp.float32), params)
        return tx.init(fp32_params)

    def update_fn(updates, state, params=None):
        fp32_updates = jax.tree.map(lambda u: u.astype(jnp.float32), updates)
        new_updates, new_state = tx.update(fp32_updates, state, params)
        # Cast deltas back so apply_updates keeps params in their original dtype.
        new_updates = jax.tree.map(lambda u, p: u.astype(p.dtype), new_updates, params)
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
