"""Learning rate schedule builders for optax."""

from __future__ import annotations

import optax


def build_lr_schedule(
    *,
    peak_lr: float,
    num_steps: int,
    warmup_steps: int = 0,
    schedule: str = "linear",
    end_factor: float = 0.0,
    stable_fraction: float = 0.8,
) -> optax.Schedule | float:
    """Build a learning rate schedule.

    Args:
        peak_lr: Peak (maximum) learning rate.
        num_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps from 0 to ``peak_lr``.
        schedule: One of ``"linear"`` (constant after warmup), ``"cosine"``, or
            ``"wsd"`` (warmup-stable-decay).
        end_factor: Final LR as a fraction of ``peak_lr`` (used by cosine/wsd).
        stable_fraction: Fraction of post-warmup steps held at ``peak_lr``
            before decay begins (wsd only).
    """
    if schedule == "linear":
        if warmup_steps > 0:
            return optax.linear_schedule(
                init_value=0.0,
                end_value=peak_lr,
                transition_steps=warmup_steps,
            )
        return peak_lr

    decay_steps = max(num_steps - warmup_steps, 1)

    if schedule == "cosine":
        schedules = []
        boundaries = []
        if warmup_steps > 0:
            schedules.append(optax.linear_schedule(0.0, peak_lr, warmup_steps))
            boundaries.append(warmup_steps)
        schedules.append(
            optax.cosine_decay_schedule(
                init_value=peak_lr,
                decay_steps=decay_steps,
                alpha=end_factor,
            )
        )
        if len(schedules) == 1:
            return schedules[0]
        return optax.join_schedules(schedules, boundaries)

    if schedule == "wsd":
        stable_steps = int(decay_steps * stable_fraction)
        decay_phase_steps = max(decay_steps - stable_steps, 1)
        end_lr = peak_lr * end_factor
        schedules = []
        boundaries = []
        if warmup_steps > 0:
            schedules.append(optax.linear_schedule(0.0, peak_lr, warmup_steps))
            boundaries.append(warmup_steps)
        schedules.append(optax.constant_schedule(peak_lr))
        boundaries.append(warmup_steps + stable_steps)
        schedules.append(
            optax.linear_schedule(peak_lr, end_lr, decay_phase_steps)
        )
        return optax.join_schedules(schedules, boundaries)

    raise ValueError(f"Unknown lr_schedule={schedule!r}. Expected 'linear', 'cosine', or 'wsd'.")
