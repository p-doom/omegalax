"""Mixed-precision optimizer (T5X-style: bf16 params, fp32 state & accumulation)."""

from __future__ import annotations

import enum

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


@jax.jit
def initialize_gradient_sum(gradients):
    """Start the device accumulator by upcasting every microgradient to fp32."""
    return jax.tree.map(lambda gradient: gradient.astype(jnp.float32), gradients)


@jax.jit(donate_argnums=0)
def accumulate_gradient_sum(gradient_sum, gradients):
    """Add one fp32-upcast microgradient to a donated device accumulator."""
    return jax.tree.map(
        lambda total, gradient: total + gradient.astype(jnp.float32),
        gradient_sum,
        gradients,
    )


class OptimizerFatalStatus(enum.IntEnum):
    HEALTHY = 0
    INVALID_LOSS = 1
    INVALID_SUPERVISION = 2
    INVALID_AUXILIARY_LOSS = 3
    INVALID_GRADIENT = 4
    INVALID_GRADIENT_NORM = 5
    INVALID_CLIP_NORM = 6
    INVALID_CURRENT_STATE = 7
    INVALID_CANDIDATE_STATE = 8
    INVALID_GENERATION = 9
    INVALID_LEARNING_RATE = 10


class OptimizerStatusBoundary(enum.StrEnum):
    LOG = "log"
    CHECKPOINT = "checkpoint"
    VALIDATION = "validation"
    FINAL = "final"


def generation_adamw(*, weight_decay: float) -> optax.GradientTransformationExtraArgs:
    """Build AdamW whose sole integer state is Adam's generation count."""
    direction = optax.chain(
        optax.scale_by_adam(),
        optax.add_decayed_weights(weight_decay),
    )

    def update(updates, state, params=None, *, learning_rate):
        updates, state = direction.update(updates, state, params)
        learning_rate = jnp.asarray(learning_rate)
        updates = jax.tree.map(
            lambda update_value: -learning_rate.astype(update_value.dtype) * update_value,
            updates,
        )
        return updates, state

    return optax.GradientTransformationExtraArgs(direction.init, update)


def _record_fatal(status, invalid, fatal: OptimizerFatalStatus):
    return jnp.where(
        jnp.logical_and(status == OptimizerFatalStatus.HEALTHY, invalid),
        jnp.asarray(fatal, dtype=jnp.uint8),
        status,
    )


def _inexact_tree_is_finite(tree):
    result = jnp.array(True)
    for value in jax.tree.leaves(tree):
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError(f"Expected inexact optimizer value, got {value.dtype}.")
        result = jnp.logical_and(result, jnp.all(jnp.isfinite(value)))
    return result


def _sole_adam_count(opt_state):
    counts = []
    for value in jax.tree.leaves(nnx.pure(opt_state)):
        if jnp.issubdtype(value.dtype, jnp.inexact):
            continue
        if value.shape != () or value.dtype != jnp.int32:
            raise TypeError(
                "Optimizer exact state must be one scalar int32 Adam count; "
                f"got shape={value.shape} dtype={value.dtype}."
            )
        counts.append(value)
    if len(counts) != 1:
        raise TypeError(f"Optimizer state must contain exactly one Adam count; got {len(counts)}.")
    return counts[0]


def _current_optimizer_state_is_healthy(optimizer, generation):
    result = _inexact_tree_is_finite(nnx.pure(nnx.state(optimizer.model, optimizer.wrt)))
    for value in jax.tree.leaves(nnx.pure(optimizer.opt_state)):
        if jnp.issubdtype(value.dtype, jnp.inexact):
            result = jnp.logical_and(result, jnp.all(jnp.isfinite(value)))
    step = optimizer.step[...]
    if step.shape != () or step.dtype != jnp.uint32:
        raise TypeError(
            f"NNX optimizer step must be scalar uint32; got shape={step.shape} dtype={step.dtype}."
        )
    adam_count = _sole_adam_count(optimizer.opt_state)
    prior_generation = generation - jnp.asarray(1, dtype=jnp.int32)
    counters_match = jnp.logical_and(
        step == prior_generation.astype(jnp.uint32),
        adam_count == prior_generation,
    )
    return result, counters_match


def _candidate_optimizer_state(optimizer, gradients, learning_rate, generation):
    params = nnx.pure(nnx.state(optimizer.model, optimizer.wrt))
    gradient_arrays = nnx.pure(nnx.state(gradients, optimizer.wrt))
    current_opt_state = nnx.pure(optimizer.opt_state)
    updates, candidate_opt_state = optimizer.tx.update(
        gradient_arrays,
        current_opt_state,
        params,
        learning_rate=learning_rate,
    )
    if jax.tree.structure(candidate_opt_state) != jax.tree.structure(current_opt_state):
        raise TypeError("Optimizer update changed its state tree structure.")
    candidate_params = jax.tree.map(
        lambda param, update: (param.astype(jnp.float32) + update).astype(param.dtype),
        params,
        updates,
    )
    result = _inexact_tree_is_finite(candidate_params)
    for current, candidate in zip(
        jax.tree.leaves(current_opt_state), jax.tree.leaves(candidate_opt_state), strict=True
    ):
        if (
            not hasattr(candidate, "shape")
            or not hasattr(candidate, "dtype")
            or current.shape != candidate.shape
            or current.dtype != candidate.dtype
        ):
            raise TypeError("Optimizer update changed an optimizer-state leaf schema.")
        if jnp.issubdtype(current.dtype, jnp.inexact):
            result = jnp.logical_and(result, jnp.all(jnp.isfinite(candidate)))
    candidate_adam_count = _sole_adam_count(candidate_opt_state)
    candidate_step = optimizer.step[...] + jnp.asarray(1, dtype=jnp.uint32)
    generation_matches = jnp.logical_and(
        candidate_step == generation.astype(jnp.uint32),
        candidate_adam_count == generation,
    )
    return candidate_params, candidate_opt_state, result, generation_matches


@nnx.jit(donate_argnums=(0, 1))
def apply_normalized_gradient_sum(
    optimizer,
    gradient_sum,
    ce_loss_sum,
    supervised_tokens,
    auxiliary_loss_abs_sum,
    fatal_status,
    max_grad_norm,
    learning_rate,
    generation,
):
    """Commit one named, clipped AdamW generation or set sticky fatal status."""
    if fatal_status.shape != () or fatal_status.dtype != jnp.uint8:
        raise TypeError(
            "Optimizer fatal status must be scalar uint8; "
            f"got shape={fatal_status.shape} dtype={fatal_status.dtype}."
        )
    if generation.shape != () or generation.dtype != jnp.int32:
        raise TypeError(
            "Named optimizer generation must be scalar int32; "
            f"got shape={generation.shape} dtype={generation.dtype}."
        )
    normalized_gradients = jax.tree.map(lambda gradient: gradient / supervised_tokens, gradient_sum)
    grad_norm = optax.tree.norm(normalized_gradients)
    clip_norm = jnp.asarray(max_grad_norm, dtype=grad_norm.dtype)
    clipped_gradients = jax.tree.map(
        lambda gradient: jax.lax.select(
            jnp.squeeze(grad_norm < clip_norm),
            gradient,
            (gradient / grad_norm.astype(gradient.dtype)) * clip_norm,
        ),
        normalized_gradients,
    )
    learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)

    status = fatal_status
    status = _record_fatal(
        status,
        jnp.logical_or(~jnp.isfinite(ce_loss_sum), ce_loss_sum < 0),
        OptimizerFatalStatus.INVALID_LOSS,
    )
    status = _record_fatal(
        status,
        jnp.logical_or(~jnp.isfinite(supervised_tokens), supervised_tokens <= 0),
        OptimizerFatalStatus.INVALID_SUPERVISION,
    )
    status = _record_fatal(
        status,
        jnp.logical_or(~jnp.isfinite(auxiliary_loss_abs_sum), auxiliary_loss_abs_sum != 0),
        OptimizerFatalStatus.INVALID_AUXILIARY_LOSS,
    )
    status = _record_fatal(
        status,
        ~_inexact_tree_is_finite(gradient_sum),
        OptimizerFatalStatus.INVALID_GRADIENT,
    )
    status = _record_fatal(
        status,
        ~jnp.isfinite(grad_norm),
        OptimizerFatalStatus.INVALID_GRADIENT_NORM,
    )
    status = _record_fatal(
        status,
        jnp.logical_or(~jnp.isfinite(clip_norm), clip_norm <= 0),
        OptimizerFatalStatus.INVALID_CLIP_NORM,
    )
    status = _record_fatal(
        status,
        jnp.logical_or(~jnp.isfinite(learning_rate), learning_rate < 0),
        OptimizerFatalStatus.INVALID_LEARNING_RATE,
    )
    current_is_healthy, current_generation_matches = _current_optimizer_state_is_healthy(
        optimizer, generation
    )
    status = _record_fatal(
        status,
        ~current_is_healthy,
        OptimizerFatalStatus.INVALID_CURRENT_STATE,
    )
    status = _record_fatal(
        status,
        jnp.logical_or(generation <= 0, ~current_generation_matches),
        OptimizerFatalStatus.INVALID_GENERATION,
    )
    current_params = nnx.pure(nnx.state(optimizer.model, optimizer.wrt))
    current_opt_state = nnx.pure(optimizer.opt_state)
    current_step = optimizer.step[...]
    candidate_params, candidate_opt_state, candidate_is_healthy, generation_matches = (
        _candidate_optimizer_state(optimizer, clipped_gradients, learning_rate, generation)
    )
    status = _record_fatal(
        status,
        ~candidate_is_healthy,
        OptimizerFatalStatus.INVALID_CANDIDATE_STATE,
    )
    status = _record_fatal(
        status,
        ~generation_matches,
        OptimizerFatalStatus.INVALID_GENERATION,
    )
    commit_candidate = status == OptimizerFatalStatus.HEALTHY
    next_params = jax.tree.map(
        lambda current, candidate: jax.lax.select(commit_candidate, candidate, current),
        current_params,
        candidate_params,
    )
    next_opt_state = jax.tree.map(
        lambda current, candidate: jax.lax.select(commit_candidate, candidate, current),
        current_opt_state,
        candidate_opt_state,
    )
    next_step = jax.lax.select(
        commit_candidate,
        current_step + jnp.asarray(1, dtype=jnp.uint32),
        current_step,
    )
    nnx.update(optimizer.model, next_params)
    nnx.update(optimizer.opt_state, nnx.state(next_opt_state))
    optimizer.step[...] = next_step
    return status, grad_norm


def require_healthy_optimizer_status(status, boundary: OptimizerStatusBoundary) -> None:
    if not isinstance(boundary, OptimizerStatusBoundary):
        raise TypeError(f"Unsupported optimizer status boundary: {boundary!r}.")
    value = np.asarray(jax.device_get(status))
    if value.shape != () or value.dtype != np.uint8:
        raise RuntimeError(
            f"Invalid optimizer fatal status at {boundary.value}: shape={value.shape} "
            f"dtype={value.dtype}."
        )
    try:
        fatal = OptimizerFatalStatus(int(value))
    except ValueError as error:
        raise RuntimeError(
            f"Invalid optimizer fatal status at {boundary.value}: code={int(value)}."
        ) from error
    if fatal is not OptimizerFatalStatus.HEALTHY:
        raise FloatingPointError(
            f"Optimizer fail-stop at {boundary.value}: {fatal.name.lower()} (code={int(fatal)})."
        )


class MixedPrecisionOptimizer(nnx.ModelAndOptimizer):
    """AdamW-style optimizer with bf16 params and fp32 state/accumulation."""

    def __init__(self, model, tx, *, wrt=nnx.Param):
        super().__init__(model, tx, wrt=wrt)
        fp32_state = jax.tree.map(
            lambda value: (
                value.astype(jnp.float32) if jnp.issubdtype(value.dtype, jnp.inexact) else value
            ),
            nnx.state(self.opt_state),
        )
        nnx.update(self.opt_state, fp32_state)

    def update(self, grads, **kwargs):  # type: ignore[override]
        """Compute and apply one optimizer step with fp32 accumulation."""
        param_arrays = nnx.pure(nnx.state(self.model, self.wrt))
        grad_arrays = nnx.pure(nnx.state(grads, self.wrt))
        opt_state_arrays = nnx.pure(self.opt_state)

        fp32_grads = jax.tree.map(lambda gradient: gradient.astype(jnp.float32), grad_arrays)
        updates, new_opt_state = self.tx.update(
            fp32_grads, opt_state_arrays, param_arrays, **nnx.pure(kwargs)
        )
        new_params = jax.tree.map(
            lambda param, update: (param.astype(jnp.float32) + update).astype(param.dtype),
            param_arrays,
            updates,
        )

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1
