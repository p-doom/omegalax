"""GPU XLA perf flags for comm/compute overlap. XLA reads XLA_FLAGS lazily at first backend creation, so this must run before init_distributed()."""

from __future__ import annotations

import os

_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_all_reduce_combine_threshold_bytes=33554432 "
    "--xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 "
    "--xla_gpu_all_gather_combine_threshold_bytes=33554432 "
    "--xla_gpu_enable_all_gather_combine_by_dim=false "
    "--xla_gpu_enable_reduce_scatter_combine_by_dim=false "
    "--xla_gpu_enable_pipelined_all_gather=true "
    "--xla_gpu_enable_pipelined_reduce_scatter=true "
    "--xla_gpu_enable_pipelined_all_reduce=true "
    "--xla_gpu_enable_while_loop_double_buffering=true"
)


def configure_gpu_xla_flags() -> None:
    """Prepend the GPU perf flags to XLA_FLAGS (user flags appended last -> win on duplicates)."""
    os.environ["XLA_FLAGS"] = (_FLAGS + " " + os.environ.get("XLA_FLAGS", "")).strip()
