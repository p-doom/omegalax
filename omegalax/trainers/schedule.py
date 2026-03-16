"""Learning-rate schedule builders shared by text and VLM trainers."""

from __future__ import annotations

import optax


def build_lr_schedule(
    *,
    lr_schedule: str,
    learning_rate: float,
    num_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    wsd_decay_fraction: float = 0.1,
) -> optax.Schedule | float:
    peak_lr = learning_rate

    if lr_schedule == "constant":
        if warmup_steps > 0:
            return optax.linear_schedule(0.0, peak_lr, warmup_steps)
        return peak_lr

    min_lr = peak_lr * min_lr_ratio

    if lr_schedule == "cos":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=num_steps - warmup_steps,
            end_value=min_lr,
        )

    if lr_schedule == "wsd":
        decay_steps = max(1, int(num_steps * wsd_decay_fraction))
        stable_end = max(warmup_steps, num_steps - decay_steps)
        schedules = [
            optax.linear_schedule(0.0, peak_lr, max(1, warmup_steps)),
            optax.constant_schedule(peak_lr),
            optax.linear_schedule(peak_lr, min_lr, decay_steps),
        ]
        return optax.join_schedules(schedules, [warmup_steps, stable_end])

    raise ValueError(
        f"Unknown lr_schedule: {lr_schedule!r}. Must be 'constant', 'cos', or 'wsd'."
    )
