"""Policy-gradient trainer: microbatched gradients, one optimizer step per batch of samples."""

from __future__ import annotations

import dataclasses
import functools
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from omegalax.models.qwen3_5.deltanet import CHUNK_SIZE
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.rl.loss import policy_loss, token_logprobs
from omegalax.rl.rollout import Sample
from omegalax.trainers.perf import forward_flops_per_token


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    lr: float
    weight_decay: float
    adam_b1: float
    adam_b2: float
    adam_eps: float
    max_grad_norm: float
    eps_low: float
    eps_high: float
    tokens_per_microbatch: int
    row_tokens: int
    logit_tile_tokens: int
    pad_id: int
    host_optimizer: bool
    loss_scale: float


def _metric_sums(logp_BT, entropy_BT, log_ratio_BT, clipped_BT, mask_BT, fresh_BT):
    stop = jax.lax.stop_gradient
    abs_lr = stop(jnp.abs(log_ratio_BT))
    k3 = stop(jnp.exp(log_ratio_BT) - 1.0 - log_ratio_BT)
    fresh = mask_BT * fresh_BT
    return {
        "tokens": jnp.sum(mask_BT),
        "entropy": jnp.sum(entropy_BT * mask_BT),
        "neg_logp": -jnp.sum(stop(logp_BT) * mask_BT),
        "clipped": jnp.sum(clipped_BT * mask_BT),
        "abs_log_ratio": jnp.sum(abs_lr * mask_BT),
        "kl_k3": jnp.sum(k3 * mask_BT),
        "fresh_tokens": jnp.sum(fresh),
        "fresh_abs_log_ratio": jnp.sum(abs_lr * fresh),
        "fresh_kl_k3": jnp.sum(k3 * fresh),
        "abs_log_ratio_BT": abs_lr * mask_BT,
    }


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _linear_dot(x, w, dimension_numbers, out_sharding):
    """``x @ w.astype(x.dtype)`` with float32 accumulation, output in ``x.dtype``. The weight
    gradient leaves the backward matmul in float32, so the FSDP reduce-scatter sums float32
    partials (as Megatron's fp32 gradient reduction does) instead of compute-dtype ones."""
    return jax.lax.dot_general(
        x, w.astype(x.dtype), dimension_numbers, preferred_element_type=jnp.float32, out_sharding=out_sharding
    ).astype(x.dtype)


def _linear_dot_fwd(x, w, dimension_numbers, out_sharding):
    if dimension_numbers != (((x.ndim - 1,), (0,)), ((), ())):
        raise ValueError(f"Unsupported Linear dimension numbers {dimension_numbers}.")
    return _linear_dot(x, w, dimension_numbers, out_sharding), (x, w)


def _linear_dot_bwd(dimension_numbers, out_sharding, residuals, g):
    x, w = residuals
    batch = tuple(range(x.ndim - 1))
    dx = jax.lax.dot_general(
        g, w.astype(g.dtype), (((g.ndim - 1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
        out_sharding=jax.typeof(x).sharding,
    ).astype(x.dtype)
    dw = jax.lax.dot_general(
        x, g, ((batch, batch), ((), ())), preferred_element_type=jnp.float32, out_sharding=jax.typeof(w).sharding
    )
    return dx, dw.astype(w.dtype)


_linear_dot.defvjp(_linear_dot_fwd, _linear_dot_bwd)


def _linear_dot_general(lhs, rhs, dimension_numbers, precision=None, out_sharding=None):
    return _linear_dot(lhs, rhs, dimension_numbers, out_sharding)


def _cast_inputs_only(args, *, dtype, **_):
    inputs, kernel, bias = args
    return inputs.astype(dtype), kernel, bias


class Trainer:
    """Owns float32 params on ``mesh`` and the optimizer state (on ``mesh`` or in host memory).

    ``step`` packs samples into microbatches of ``tokens_per_microbatch // row_tokens`` rows of
    ``row_tokens`` (one compiled shape), sums token-normalized gradients and applies one AdamW update.
    """

    def __init__(self, model: nnx.Module, model_cfg, cfg: TrainConfig):
        # Packed rows need a model that keeps segments apart (attention, positions, DeltaNet).
        if not isinstance(model, Qwen3_5ForCausalLM):
            raise ValueError(f"The trainer packs segments, which {type(model).__name__} does not separate.")
        if cfg.row_tokens % CHUNK_SIZE or cfg.tokens_per_microbatch % cfg.row_tokens:
            raise ValueError(
                f"row_tokens must be a multiple of {CHUNK_SIZE} and divide tokens_per_microbatch."
            )
        self.cfg = cfg
        self.model_cfg = model_cfg
        # Linear kernels enter their matmul as float32 params (cast inside _linear_dot), so their
        # gradients stay float32 through the cross-device reduction; the lm_head is float32 already.
        for _, m in nnx.iter_modules(model):
            if isinstance(m, nnx.Linear) and m is not model.lm_head:
                m.dot_general, m.promote_dtype = _linear_dot_general, _cast_inputs_only
        self.graphdef, self.params = nnx.split(model, nnx.Param)
        leaves = jax.tree.leaves(self.params)
        if len(leaves) != len(jax.tree.leaves(nnx.state(model))):
            raise ValueError("The trainer expects a model whose state is all nnx.Param.")
        self.mesh = leaves[0].sharding.mesh
        batch_axis = model_cfg.shd_cfg.act_btd[0]
        self._rows_multiple = 1 if batch_axis is None else int(
            np.prod([self.mesh.shape[a] for a in ((batch_axis,) if isinstance(batch_axis, str) else batch_axis)])
        )
        self._rows = cfg.tokens_per_microbatch // cfg.row_tokens
        if self._rows % self._rows_multiple:
            raise ValueError(f"{self._rows} rows per microbatch do not split over {self._rows_multiple} devices.")
        self._batch_sharding = NamedSharding(self.mesh, P(batch_axis, None))
        param_shardings = jax.tree.map(lambda x: x.sharding, self.params)
        decay_mask = jax.tree.map(lambda x: x.ndim >= 2, self.params)
        self.tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adamw(
                cfg.lr, b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps,
                weight_decay=cfg.weight_decay, mask=decay_mask,
            ),
        )
        self._param_shardings = param_shardings
        # With host_optimizer the float32 master copy and the Adam moments live in host memory and
        # the update runs on a one-device CPU mesh; the devices keep the params and gradient sum.
        if cfg.host_optimizer:
            self._opt_mesh = jax.make_mesh((1, 1, 1), self.mesh.axis_names, devices=jax.devices("cpu")[:1])
            self._opt_shardings = jax.tree.map(lambda _: NamedSharding(self._opt_mesh, P()), self.params)
        else:
            self._opt_mesh, self._opt_shardings = self.mesh, param_shardings
        self.master = jax.device_put(self.params, self._opt_shardings)
        with jax.set_mesh(self._opt_mesh):
            self.opt_state = jax.jit(self.tx.init)(self.master)
        self._zeros = jax.jit(
            lambda p: jax.tree.map(jnp.zeros_like, p), out_shardings=param_shardings
        )
        # Outputs fed back as inputs keep their input shardings, or the next call recompiles.
        self._accumulate = jax.jit(
            self._accumulate_fn, donate_argnums=(1,), out_shardings=(param_shardings, None)
        )
        self._apply = jax.jit(
            self._apply_fn,
            donate_argnums=(0, 1),
            out_shardings=(self._opt_shardings, jax.tree.map(lambda x: x.sharding, self.opt_state), None),
        )
        self._logprobs = jax.jit(self._logprobs_fn)
        self.version = 0

    def _forward(self, params, tokens_BT, segment_ids_BT):
        model = nnx.merge(self.graphdef, params)
        hidden_BTD, _ = model(tokens_BT, segment_ids_BT, None, jnp.array(0, jnp.int32))
        B, T = tokens_BT.shape
        return token_logprobs(
            hidden_BTD,
            model.output_weight(),
            tokens_BT,
            -(-B * T // self.cfg.logit_tile_tokens),
            self.model_cfg.shd_cfg.logits_btv,
        )

    def _accumulate_fn(self, params, acc, mb, inv_tokens):
        def loss_fn(params):
            logp_BT, entropy_BT = self._forward(params, mb["tokens"], mb["segment_ids"])
            loss, log_ratio_BT, clipped_BT = policy_loss(
                logp_BT, mb["behavior_logp"], mb["advantage"], mb["mask"], self.cfg.eps_low, self.cfg.eps_high
            )
            metrics = _metric_sums(
                logp_BT, entropy_BT, log_ratio_BT, clipped_BT, mb["mask"], mb["fresh"]
            )
            return loss * inv_tokens, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
        return jax.tree.map(jnp.add, acc, grads), metrics

    def _apply_fn(self, params, opt_state, grads):
        grads = jax.tree.map(lambda g: g / self.cfg.loss_scale, grads)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = self.tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, grad_norm

    def _logprobs_fn(self, params, tokens_BT, segment_ids_BT):
        return self._forward(params, tokens_BT, segment_ids_BT)[0]

    def _pack(self, lengths: Sequence[int]) -> tuple[int, list[tuple[int, int]]]:
        """First-fit decreasing into rows of ``row_tokens``: the number of rows (a whole number of
        microbatches) and each sequence's (row, start). Starts are multiples of CHUNK_SIZE, where
        the DeltaNet recurrence can restart."""
        T = self.cfg.row_tokens
        if max(lengths, default=0) > T:
            raise ValueError(f"Sequence length {max(lengths)} exceeds row_tokens={T}.")
        used, placed = [], [None] * len(lengths)
        for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
            slot = -(-lengths[i] // CHUNK_SIZE) * CHUNK_SIZE
            r = next((r for r, u in enumerate(used) if u + slot <= T), len(used))
            if r == len(used):
                used.append(0)
            placed[i] = (r, used[r])
            used[r] += slot
        return -(-len(used) // self._rows) * self._rows, placed

    def microbatches(self, samples: Sequence[Sample], advantages: Sequence[float]):
        """Host arrays per microbatch; sample ``i`` is segment ``i + 1`` of a packed row (0 = padding)."""
        rows, placed = self._pack([len(s.tokens) for s in samples])
        shape = (rows, self.cfg.row_tokens)
        packed = {
            "tokens": np.full(shape, self.cfg.pad_id, np.int32),
            "segment_ids": np.zeros(shape, np.int32),
            "mask": np.zeros(shape, np.float32),
            "behavior_logp": np.zeros(shape, np.float32),
            "fresh": np.zeros(shape, np.float32),
            "advantage": np.zeros(shape, np.float32),
        }
        for i, (s, (r, start)) in enumerate(zip(samples, placed)):
            seg = slice(start, start + len(s.tokens))
            packed["tokens"][r, seg] = s.tokens
            packed["segment_ids"][r, seg] = i + 1
            packed["mask"][r, seg] = s.mask
            packed["behavior_logp"][r, seg] = s.logprobs
            packed["fresh"][r, seg] = np.asarray(s.versions) == self.version
            packed["advantage"][r, seg] = advantages[i]
        return [{k: v[j : j + self._rows] for k, v in packed.items()} for j in range(0, rows, self._rows)]

    def _put(self, mb):
        return {k: jax.device_put(v, self._batch_sharding) for k, v in mb.items()}

    def warmup(self) -> float:
        """Compiles the gradient program on all-masked rows, and the update ahead of time (it then
        loads from the persistent compilation cache)."""
        t0 = time.perf_counter()
        shape = (self._rows, self.cfg.row_tokens)
        masked = self._put({
            k: np.zeros(shape, np.int32 if k in ("tokens", "segment_ids") else np.float32)
            for k in ("tokens", "segment_ids", "mask", "behavior_logp", "fresh", "advantage")
        })
        with jax.set_mesh(self.mesh):
            # The first trace of the model extends JAX's trace context, which keys every jit
            # cache; trace once before compiling anything, or the first step recompiles.
            self._accumulate.trace(self.params, self.params, masked, np.float32(0.0))
            acc, _ = self._accumulate(self.params, self._zeros(self.params), masked, np.float32(0.0))
        with jax.set_mesh(self._opt_mesh):
            self._apply.lower(self.master, self.opt_state, jax.device_put(acc, self._opt_shardings)).compile()
        jax.block_until_ready(acc)
        return time.perf_counter() - t0

    def step(self, samples: Sequence[Sample], advantages: Sequence[float]) -> dict[str, float]:
        t0 = time.perf_counter()
        total_tokens = float(sum(sum(s.mask) for s in samples))
        if total_tokens == 0:
            raise ValueError("A training batch needs at least one masked token.")
        # A zero advantage zeroes a sample's loss and gradient exactly; it only counts in the
        # token normalizer, so its forward/backward is skipped.
        active = [i for i, a in enumerate(advantages) if a != 0.0]
        mbs = self.microbatches([samples[i] for i in active], [advantages[i] for i in active])
        # The scale keeps float16 cotangents (~advantage / loss tokens) above the subnormal range.
        inv_tokens = np.float32(self.cfg.loss_scale / total_tokens)
        with jax.set_mesh(self.mesh):
            acc = self._zeros(self.params)
            sums = []
            for mb in mbs:
                acc, m = self._accumulate(self.params, acc, self._put(mb), inv_tokens)
                sums.append(m)
        with jax.set_mesh(self._opt_mesh):
            self.master, self.opt_state, grad_norm = self._apply(
                self.master, self.opt_state, jax.device_put(acc, self._opt_shardings)
            )
        self.params = jax.device_put(self.master, self._param_shardings)
        sums = jax.device_get(sums)
        if not np.isfinite(float(grad_norm)):
            raise FloatingPointError(f"Gradient norm {float(grad_norm)} at loss_scale {self.cfg.loss_scale}.")
        # Per-token |log pi - log mu| on trained tokens, for quantiles and Miles' sample mean.
        per_token, per_sample = [np.zeros(0)], [np.zeros(0)]
        for s, mb in zip(sums, mbs):
            a, m, seg = s.pop("abs_log_ratio_BT"), mb["mask"] > 0, mb["segment_ids"].ravel()
            per_token.append(a[m])
            num, den = np.bincount(seg, a.ravel()), np.bincount(seg, m.ravel().astype(np.float64))
            per_sample.append(num[den > 0] / den[den > 0])
        per_token, per_sample = np.concatenate(per_token), np.concatenate(per_sample)
        totals = {k: sum(float(s[k]) for s in sums) for k in (sums[0] if sums else ())}
        jax.block_until_ready(self.params)
        seconds = time.perf_counter() - t0
        self.version += 1
        flops = sum(
            forward_flops_per_token(self.model_cfg, len(samples[i].tokens)).model_flops(base_weights_trainable=True)
            * len(samples[i].tokens)
            for i in active
        )
        padded = max(sum(mb["tokens"].size for mb in mbs), 1)
        get = lambda k: totals.get(k, 0.0)
        n, nf = max(get("tokens"), 1.0), max(get("fresh_tokens"), 1.0)
        return {
            "train/seconds": seconds,
            "train/microbatches": len(mbs),
            "train/grad_norm": float(grad_norm),
            "train/loss_tokens": total_tokens,
            "train/active_frac": get("tokens") / total_tokens,
            "train/entropy": get("entropy") / n,
            "train/neg_logp": get("neg_logp") / n,
            "train/clip_frac": get("clipped") / n,
            "train/abs_log_ratio": get("abs_log_ratio") / n,
            "train/abs_log_ratio_sample_mean": float(per_sample.mean()) if per_sample.size else 0.0,
            "train/abs_log_ratio_p99": float(np.quantile(per_token, 0.99)) if per_token.size else 0.0,
            "train/abs_log_ratio_p999": float(np.quantile(per_token, 0.999)) if per_token.size else 0.0,
            "train/kl_k3": get("kl_k3") / n,
            "train/fresh_frac": get("fresh_tokens") / n,
            "train/fresh_abs_log_ratio": get("fresh_abs_log_ratio") / nf,
            "train/fresh_kl_k3": get("fresh_kl_k3") / nf,
            "train/pad_frac": 1.0 - sum(float(np.count_nonzero(mb["segment_ids"])) for mb in mbs) / padded,
            "train/model_tflop": flops / 1e12,
        }

    def logprobs(self, token_lists: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Trainer-forward ``logp[t]`` of each token given its prefix (``logp[0] = 0``), packed
        like training microbatches."""
        rows, placed = self._pack([len(t) for t in token_lists])
        tokens = np.full((rows, self.cfg.row_tokens), self.cfg.pad_id, np.int32)
        segment_ids = np.zeros_like(tokens)
        for i, (t, (r, start)) in enumerate(zip(token_lists, placed)):
            tokens[r, start : start + len(t)] = t
            segment_ids[r, start : start + len(t)] = i + 1
        with jax.set_mesh(self.mesh):
            logp = np.concatenate([
                np.asarray(self._logprobs(
                    self.params,
                    jax.device_put(tokens[j : j + self._rows], self._batch_sharding),
                    jax.device_put(segment_ids[j : j + self._rows], self._batch_sharding),
                ))
                for j in range(0, rows, self._rows)
            ])
        out = [logp[r, start : start + len(t)].copy() for t, (r, start) in zip(token_lists, placed)]
        for x in out:
            x[0] = 0.0  # scored from the previous segment's last token
        return out
