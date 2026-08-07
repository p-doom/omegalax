"""LoRA (Low-Rank Adaptation) for VLM SFT.

Wraps target ``nnx.Linear`` projections with low-rank adapters so only the
adapter weights are trained while base weights remain frozen.

Design follows the "LoRA Without Regret" recommendations:
* attach to all weight matrices in the transformer (attention q/k/v/o + MLP
  gate/up/down) — attention-only is empirically worse.
* default ``rank=32``, ``alpha=32`` (matches Tinker's SL default and the
  PEFT/HuggingFace convention; ``alpha/r = 1`` makes the optimal LR
  approximately rank-independent).
* init A from uniform with scale ``1/sqrt(d_in)``, B from zeros (so the
  adapter is the identity at step 0 and we get bit-exact forward parity
  with the unwrapped model).
* recommended LR ≈ 10× the FullFT LR.

Vision tower is intentionally excluded by the ``skip_paths`` default:
the BC objective is text-token-generation conditioned on visual features,
and we want to preserve the base ViT's UI/document grounding intact.
``lm_head`` and embeddings are also skipped (not standard LoRA targets;
``lm_head.kernel`` is consumed directly by the chunked-CE loss in the
trainer, so wrapping it would require trainer changes for no expected
benefit).

Trainable-vs-frozen separation is handled at the optimizer/grad layer via
the ``wrt`` filter: ``nnx.value_and_grad(loss, wrt=LoRAParam)`` and
``MixedPrecisionOptimizer(model, tx, wrt=LoRAParam)`` cause every
non-LoRA ``nnx.Param`` (including the wrapped base linears, vision tower,
embedder, lm_head, layernorms) to receive zero gradient and be excluded
from the optimizer state. Base weights ride along through serialization
unchanged.
"""

from __future__ import annotations

from typing import Sequence

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard


# Standard LoRA target-module names for transformer-block projections.
# Match by attribute name on the *parent* module (e.g.
# ``TextAttention.q_proj``). MoE feed-forward layers store gate/up/down
# as raw ``nnx.Param`` (rank-3, expert-stacked), not ``nnx.Linear``, so
# they fall through silently — Qwen3-VL-2B-Instruct is dense (no MoE
# layers) so this doesn't bite us.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Subtree attribute names to skip during injection. Any module whose
# ``iter_modules`` path passes through one of these is left untouched.
DEFAULT_SKIP_PATHS: tuple[str, ...] = ("vision",)


class LoRAParam(nnx.Param):
    """Trainable LoRA adapter weight.

    Distinct ``nnx.Param`` subclass so a ``wrt=LoRAParam`` filter on
    ``nnx.value_and_grad`` and ``MixedPrecisionOptimizer`` cleanly selects
    only the adapter weights for gradient updates.
    """


class LoRALinear(nnx.Module):
    """Low-rank-adapted Linear: ``y = base(x) + (alpha/r) * (x @ A) @ B``.

    The wrapped ``base`` is an ordinary ``nnx.Linear`` whose ``kernel`` /
    ``bias`` remain ``nnx.Param`` (NOT ``LoRAParam``). With
    ``wrt=LoRAParam`` plumbed through grad+optimizer, the base weights
    are frozen at the gradient layer while still being checkpointed as
    part of the model state.

    ``lora_A`` / ``lora_B`` are stored replicated (they are tiny: ~r·d
    each); sharding is applied in the forward instead. ``__call__``
    reshards ``lora_B`` so the delta is *born* with the base projection's
    output sharding — mirroring how the base weight is tp-sharded — which
    keeps the delta activation tp-sharded under TP with no extra
    collective. At tp=1 it is a no-op.
    """

    def __init__(
        self,
        base: nnx.Linear,
        *,
        r: int,
        alpha: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype | None = None,
    ):
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r

        d_in = base.in_features
        d_out = base.out_features
        adapter_dtype = dtype if dtype is not None else base.dtype

        # PEFT/Tinker convention: A ~ Uniform(-1/sqrt(d_in), 1/sqrt(d_in))
        # under nnx.initializers.uniform's symmetric scale. B = 0.
        # No sharding metadata: A and B are tiny (~r·d each) and replicated
        # across the mesh; the surrounding matmul output sharding is
        # constrained explicitly via ``out_sharding`` in ``__call__``.
        a_init_fn = nnx.initializers.uniform(scale=1.0 / (d_in**0.5))
        self.lora_A = LoRAParam(
            a_init_fn(rngs.params(), (d_in, r), adapter_dtype),
        )
        self.lora_B = LoRAParam(
            jnp.zeros((r, d_out), dtype=adapter_dtype),
        )

    # Forward-compat surface so callers that read base.kernel etc. still work.
    @property
    def kernel(self):
        return self.base.kernel

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def dtype(self):
        return self.base.dtype

    def __call__(self, inputs: jax.Array, *, out_sharding=None) -> jax.Array:
        base_out = self.base(inputs, out_sharding=out_sharding)
        a = self.lora_A[...]
        b = self.lora_B[...]
        if out_sharding is not None:
            # reshard lora_B (not the delta) so the delta is born tp-sharded; no-op at tp=1.
            b = reshard(b, P(None, out_sharding[-1]))
        # (..., d_in) @ (d_in, r) -> (..., r) -> @ (r, d_out) -> (..., d_out)
        delta = jnp.matmul(jnp.matmul(inputs, a), b) * self.scaling
        delta = delta.astype(base_out.dtype)
        if out_sharding is not None:
            # No-op given the born-sharded delta; kept as a defensive assertion.
            delta = jax.lax.with_sharding_constraint(delta, out_sharding)
        return base_out + delta

    def merge_into_base(self) -> jax.Array:
        """Return the merged kernel ``W + (alpha/r) * A @ B``.

        Used at HF-export time so downstream serving (sglang) sees a
        plain dense weight and needs no adapter awareness.
        """
        a = self.lora_A[...].astype(jnp.float32)
        b = self.lora_B[...].astype(jnp.float32)
        delta = (a @ b) * self.scaling
        merged = self.base.kernel[...].astype(jnp.float32) + delta
        return merged.astype(self.base.kernel[...].dtype)


def inject_lora(
    model: nnx.Module,
    *,
    r: int = 32,
    alpha: float = 32.0,
    rngs: nnx.Rngs,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    skip_paths: Sequence[str] = DEFAULT_SKIP_PATHS,
    dtype: jnp.dtype | None = None,
) -> int:
    """Replace each ``nnx.Linear`` named in ``target_modules`` with a
    ``LoRALinear`` wrapping it. In place. Returns the count of layers
    modified.

    Skips any module whose ``iter_modules`` path includes any name in
    ``skip_paths`` — by default this excludes the vision tower so its
    base weights are entirely untouched.

    Idempotent on already-wrapped modules: a ``LoRALinear`` is not an
    ``nnx.Linear``, so re-running this function won't double-wrap.
    """
    target_set = set(target_modules)
    skip_set = set(skip_paths)
    # Materialize iter_modules first because we mutate during iteration.
    modules = list(nnx.iter_modules(model))
    count = 0
    for path, module in modules:
        if any(p in skip_set for p in path):
            continue
        for name in target_set:
            child = getattr(module, name, None)
            if not isinstance(child, nnx.Linear):
                continue
            wrapped = LoRALinear(
                child,
                r=r,
                alpha=alpha,
                rngs=rngs,
                dtype=dtype,
            )
            setattr(module, name, wrapped)
            count += 1
    return count


def merge_lora_into_base(model: nnx.Module) -> int:
    """Walk the model; for each ``LoRALinear``, write the merged kernel
    back into ``base.kernel`` and re-attach the base ``nnx.Linear`` as
    the parent's attribute (replacing the wrapper). Returns count
    merged.

    After this call, the model is structurally indistinguishable from a
    full-FT model with the same effective weights — which is exactly
    what HF safetensors export wants.
    """
    modules = list(nnx.iter_modules(model))
    count = 0
    for path, module in modules:
        for name in list(vars(module)):
            child = getattr(module, name, None)
            if not isinstance(child, LoRALinear):
                continue
            merged_kernel = child.merge_into_base()
            base = child.base
            # Update the base kernel value in place. Keep the existing
            # nnx.Param wrapper (and its sharding metadata) so the
            # surrounding optimizer / serialization machinery sees an
            # unchanged variable structure.
            base.kernel[...] = merged_kernel
            setattr(module, name, base)
            count += 1
    return count
