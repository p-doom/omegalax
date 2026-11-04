"""LoRA (Low-Rank Adaptation) for VLM SFT.

Wraps target ``nnx.Linear`` projections with low-rank adapters so only the
adapter weights are trained while base weights remain frozen.

Design follows the "LoRA Without Regret" recommendations:
* attach to all weight matrices in the transformer (attention q/k/v/o + MLP
  gate/up/down) — attention-only is empirically worse. This includes the
  expert-stacked gate/up/down of MoE feed-forward blocks, which are stored
  as rank-3 ``nnx.Param`` (E, D, F) rather than ``nnx.Linear`` and so need a
  dedicated per-expert stacked adapter (see ``LoRAMoEExperts``); each expert
  gets its own low-rank adapter on each of its three projections.
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


def _logical_axis_mesh_active() -> bool:
    """True if an abstract mesh is active, so ``nnx.Param``'s eager sharding
    can resolve logical axis names. False under meshless CPU unit tests."""
    from jax.sharding import get_abstract_mesh

    return not get_abstract_mesh().empty


# Standard LoRA target-module names for transformer-block projections.
# Match by attribute name on the *parent* module (e.g.
# ``TextAttention.q_proj``). For dense linears these are wrapped as
# ``LoRALinear``. MoE feed-forward layers store gate/up/down as raw
# ``nnx.Param`` (rank-3, expert-stacked) rather than ``nnx.Linear``; those
# are matched by the same attribute names and adapted per-expert via
# ``LoRAMoEExperts`` (see ``inject_lora``).
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Expert-stacked projection attribute names on MoE feed-forward modules.
# These are rank-3 ``nnx.Param`` (E, D, F) / (E, F, D), not ``nnx.Linear``.
MOE_EXPERT_TARGET_MODULES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

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

    # ---- forward-compat surface so callers that read base.kernel etc. still work ----
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


class LoRAMoEExperts(nnx.Module):
    """Per-expert low-rank adapter for an expert-stacked MoE projection.

    MoE feed-forward blocks store their gate/up/down projections as a single
    rank-3 ``nnx.Param`` stacking all experts:

    * gate/up: shape ``(E, D, F)``, applied via ``einsum("BTD,EDF->BTEF")``.
    * down:    shape ``(E, F, D)``, applied via ``einsum("BTEF,EFD->BTED")``.

    A LoRA on these adds, *per expert*, a rank-``r`` correction to the
    per-expert linear map. Because the MoE forward is nonlinear (SiLU gate,
    top-k gather, router weighting), the correction MUST be injected inside
    the expert einsum — it cannot be folded into the module output. This
    module therefore only holds the adapter weights (``lora_A`` / ``lora_B``)
    and exposes ``delta`` to compute the per-expert output correction; the
    MoE ``__call__`` adds ``delta`` to the corresponding einsum result when an
    adapter is attached.

    Given the base weight ``W`` of shape ``(E, in, out)`` (``in``/``out`` are
    ``D``/``F`` for gate/up and ``F``/``D`` for down), the adapters are

        A: (E, in, r)   sharded on the ``in`` axis (mirrors the base weight)
        B: (E, r, out)  sharded on the ``out`` axis, initialized to zero,

    with the expert axis ``E`` and the rank axis ``r`` replicated. B=0 ⇒ the
    adapter is identity at step 0, giving bit-exact forward parity.

    Two forwards mirror the two einsum shapes in the MoE block:
    * ``delta_shared`` for gate/up, whose input ``BTD`` is shared across
      experts (base einsum ``BTD,EDF->BTEF``);
    * ``delta_per_expert`` for down, whose input ``BTEF`` is already
      per-expert (base einsum ``BTEF,EFD->BTED``).
    """

    def __init__(
        self,
        base: nnx.Param,
        *,
        in_axis_name: str | None,
        out_axis_name: str | None,
        r: int,
        alpha: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype | None = None,
    ):
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        shape = base[...].shape
        if len(shape) != 3:
            raise ValueError(
                f"LoRAMoEExperts expects a rank-3 expert-stacked Param, got shape {shape}"
            )
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r

        E, d_in, d_out = shape
        adapter_dtype = dtype if dtype is not None else base[...].dtype

        # PEFT/Tinker convention: A ~ Uniform(-1/sqrt(d_in), 1/sqrt(d_in)),
        # B = 0. Sharding mirrors the base expert weight: A is sharded on the
        # projection's input axis, B on its output axis, E and r replicated.
        # ``nnx.Param``'s eager sharding needs an active logical-axis mesh; the
        # trainer always injects under ``mesh_rules(mesh)`` so metadata is
        # attached there. When no such mesh is active (e.g. meshless CPU unit
        # tests) we omit the metadata — the physical sharding is still driven
        # by the surrounding einsums' ``out_sharding`` in ``__call__``.
        a_kwargs = {}
        b_kwargs = {}
        if _logical_axis_mesh_active():
            a_kwargs["sharding"] = (None, in_axis_name, None)
            b_kwargs["sharding"] = (None, None, out_axis_name)
        a_init_fn = nnx.initializers.uniform(scale=1.0 / (d_in**0.5))
        self.lora_A = LoRAParam(
            a_init_fn(rngs.params(), (E, d_in, r), adapter_dtype),
            **a_kwargs,
        )
        self.lora_B = LoRAParam(
            jnp.zeros((E, r, d_out), dtype=adapter_dtype),
            **b_kwargs,
        )

    def delta_shared(self, hidden_BTD: jax.Array, *, out_sharding=None) -> jax.Array:
        """LoRA correction for gate/up: input ``(B,T,D)`` shared across
        experts, returns ``(B,T,E,F)`` in ``hidden``'s dtype (matches the base
        ``BTD,EDF->BTEF`` einsum output)."""
        a = self.lora_A[...].astype(hidden_BTD.dtype)  # (E, D, r)
        b = self.lora_B[...].astype(hidden_BTD.dtype)  # (E, r, F)
        mid_BTEr = jnp.einsum("BTD,EDr->BTEr", hidden_BTD, a)
        delta_BTEF = jnp.einsum("BTEr,ErF->BTEF", mid_BTEr, b, out_sharding=out_sharding)
        return (delta_BTEF * self.scaling).astype(hidden_BTD.dtype)

    def delta_per_expert(self, hidden_BTEF: jax.Array, *, out_sharding=None) -> jax.Array:
        """LoRA correction for down: input ``(B,T,E,F)`` already per-expert,
        returns ``(B,T,E,D)`` in ``hidden``'s dtype (matches the base
        ``BTEF,EFD->BTED`` einsum output)."""
        a = self.lora_A[...].astype(hidden_BTEF.dtype)  # (E, F, r)
        b = self.lora_B[...].astype(hidden_BTEF.dtype)  # (E, r, D)
        mid_BTEr = jnp.einsum("BTEF,EFr->BTEr", hidden_BTEF, a)
        delta_BTED = jnp.einsum("BTEr,ErD->BTED", mid_BTEr, b, out_sharding=out_sharding)
        return (delta_BTED * self.scaling).astype(hidden_BTEF.dtype)

    def merge_into_base(self, base: nnx.Param) -> jax.Array:
        """Return the merged expert-stacked weight ``W + scaling * A @ B``
        (contracting over ``r`` per expert), in ``base``'s dtype."""
        a = self.lora_A[...].astype(jnp.float32)
        b = self.lora_B[...].astype(jnp.float32)
        delta = jnp.einsum("Eir,Ero->Eio", a, b) * self.scaling
        merged = base[...].astype(jnp.float32) + delta
        return merged.astype(base[...].dtype)


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

    MoE feed-forward blocks store their gate/up/down as rank-3 expert-stacked
    ``nnx.Param`` rather than ``nnx.Linear``. A module that declares an
    ``_EXPERT_LORA_SHARDING`` class attribute (mapping each expert-projection
    attribute name to its logical sharding tuple) opts into per-expert
    adapters: for each such projection named in ``target_modules`` a
    ``LoRAMoEExperts`` adapter is attached at ``{name}_lora`` and consumed by
    the module's forward. This is what lets the ~97%-of-parameters experts
    actually train under LoRA.

    Idempotent on already-wrapped modules: a ``LoRALinear`` is not an
    ``nnx.Linear``, so re-running this function won't double-wrap; expert
    adapters are only attached to slots that are currently ``None``.
    """
    target_set = set(target_modules)
    skip_set = set(skip_paths)
    # Materialize iter_modules first because we mutate during iteration.
    modules = list(nnx.iter_modules(model))
    count = 0
    for path, module in modules:
        if any(p in skip_set for p in path):
            continue
        expert_sharding = getattr(type(module), "_EXPERT_LORA_SHARDING", None)
        for name in target_set:
            child = getattr(module, name, None)
            if isinstance(child, nnx.Linear):
                wrapped = LoRALinear(
                    child,
                    r=r,
                    alpha=alpha,
                    rngs=rngs,
                    dtype=dtype,
                )
                setattr(module, name, wrapped)
                count += 1
                continue
            # MoE expert-stacked Param: attach a per-expert adapter into the
            # module's declared ``{name}_lora`` slot (guarded by the module).
            if (
                expert_sharding is not None
                and name in expert_sharding
                and isinstance(child, nnx.Param)
                and getattr(child, "ndim", None) == 3
                and getattr(module, f"{name}_lora", None) is None
            ):
                sharding = expert_sharding[name]
                adapter = LoRAMoEExperts(
                    child,
                    in_axis_name=sharding[1],
                    out_axis_name=sharding[2],
                    r=r,
                    alpha=alpha,
                    rngs=rngs,
                    dtype=dtype,
                )
                setattr(module, f"{name}_lora", adapter)
                count += 1
    return count


def merge_lora_into_base(model: nnx.Module) -> int:
    """Walk the model; for each ``LoRALinear``, write the merged kernel
    back into ``base.kernel`` and re-attach the base ``nnx.Linear`` as
    the parent's attribute (replacing the wrapper). Returns count
    merged.

    For MoE modules, each attached ``LoRAMoEExperts`` (at ``{name}_lora``) is
    folded back into its expert-stacked base ``nnx.Param`` and the slot reset
    to ``None`` so the forward reverts to the plain (now-merged) einsum.

    After this call, the model is structurally indistinguishable from a
    full-FT model with the same effective weights — which is exactly
    what HF safetensors export wants.
    """
    modules = list(nnx.iter_modules(model))
    count = 0
    for path, module in modules:
        for name in list(vars(module)):
            child = getattr(module, name, None)
            if isinstance(child, LoRALinear):
                merged_kernel = child.merge_into_base()
                base = child.base
                # Update the base kernel value in place. Keep the existing
                # nnx.Param wrapper (and its sharding metadata) so the
                # surrounding optimizer / serialization machinery sees an
                # unchanged variable structure.
                base.kernel[...] = merged_kernel
                setattr(module, name, base)
                count += 1
            elif isinstance(child, LoRAMoEExperts):
                # Fold the per-expert adapter into its base stacked Param.
                base_name = name[: -len("_lora")]
                base = getattr(module, base_name)
                base[...] = child.merge_into_base(base)
                setattr(module, name, None)
                count += 1
    return count
