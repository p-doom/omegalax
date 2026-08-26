"""Distributed runtime helpers."""

from .mesh import (
    ParallelismConfig,
    derive_ici_dcn,
    ensure_mesh,
    make_hierarchical_mesh,
    make_mesh,
    set_default_mesh,
)
from .xla_flags import configure_gpu_xla_flags


def init_distributed():
    """Init JAX distributed for one-process-per-GPU launches (the multi-node convention)."""
    import jax

    jax.distributed.initialize()
    assert jax.local_device_count() == 1, (
        f"launch one process per GPU; this process sees {jax.local_device_count()} "
        "local devices -- use srun --ntasks-per-node=<gpus_per_node>"
    )


__all__ = [
    "configure_gpu_xla_flags",
    "init_distributed",
    "ParallelismConfig",
    "derive_ici_dcn",
    "ensure_mesh",
    "make_hierarchical_mesh",
    "make_mesh",
    "set_default_mesh",
]
