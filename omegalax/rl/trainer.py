"""Policy-gradient trainer: microbatched gradients, one optimizer step per batch of samples."""

from __future__ import annotations

import dataclasses
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

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
    length_buckets: tuple[int, ...]
    logit_tile_tokens: int
    pad_id: int
    host_optimizer: bool


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
    }


class Trainer:
    """Owns the float32 master params and the optimizer state on ``mesh``.

    ``step`` packs samples into length-bucketed microbatches (one compiled shape per bucket),
    sums token-normalized gradients and applies one AdamW update.
    """

    def __init__(self, model: nnx.Module, model_cfg, cfg: TrainConfig):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.graphdef, self.params = nnx.split(model, nnx.Param)
        leaves = jax.tree.leaves(self.params)
        if len(leaves) != len(jax.tree.leaves(nnx.state(model))):
            raise ValueError("The trainer expects a model whose state is all nnx.Param.")
        self.mesh = leaves[0].sharding.mesh
        batch_axis = model_cfg.shd_cfg.act_btd[0]
        self._rows_multiple = 1 if batch_axis is None else int(
            np.prod([self.mesh.shape[a] for a in ((batch_axis,) if isinstance(batch_axis, str) else batch_axis)])
        )
        self._batch_sharding = NamedSharding(self.mesh, P(batch_axis, None))
        self._row_sharding = NamedSharding(self.mesh, P(batch_axis))
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
        # the update runs on the CPU; the devices keep the params and the gradient sum only.
        self._opt_shardings = (
            jax.tree.map(lambda _: jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0]), self.params)
            if cfg.host_optimizer
            else param_shardings
        )
        self.master = jax.device_put(self.params, self._opt_shardings)
        with jax.set_mesh(self.mesh):
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
            model.lm_head.kernel[...],
            tokens_BT,
            -(-B * T // self.cfg.logit_tile_tokens),
            self.model_cfg.shd_cfg.logits_btv,
        )

    def _accumulate_fn(self, params, acc, mb, inv_tokens):
        def loss_fn(params):
            logp_BT, entropy_BT = self._forward(params, mb["tokens"], mb["segment_ids"])
            adv_BT = jnp.broadcast_to(mb["advantage"][:, None], logp_BT.shape)
            loss, log_ratio_BT, clipped_BT = policy_loss(
                logp_BT, mb["behavior_logp"], adv_BT, mb["mask"], self.cfg.eps_low, self.cfg.eps_high
            )
            metrics = _metric_sums(
                logp_BT, entropy_BT, log_ratio_BT, clipped_BT, mb["mask"], mb["fresh"]
            )
            return loss * inv_tokens, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
        return jax.tree.map(jnp.add, acc, grads), metrics

    def _apply_fn(self, params, opt_state, grads):
        grad_norm = optax.global_norm(grads)
        updates, opt_state = self.tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, grad_norm

    def _logprobs_fn(self, params, tokens_BT, segment_ids_BT):
        return self._forward(params, tokens_BT, segment_ids_BT)[0]

    def _bucket(self, length: int) -> int:
        for b in self.cfg.length_buckets:
            if b >= length:
                return b
        raise ValueError(f"Sample length {length} exceeds the largest bucket {self.cfg.length_buckets[-1]}.")

    def _rows(self, T: int) -> int:
        R = self._rows_multiple
        return max(R, self.cfg.tokens_per_microbatch // T // R * R)

    def microbatches(self, samples: Sequence[Sample], advantages: Sequence[float]):
        """Host arrays per microbatch. Samples go longest first; each microbatch takes the next
        ``rows`` samples, where its first (longest) sample picks the length bucket ``T`` and
        ``rows = tokens_per_microbatch // T``. Only the last microbatch has fully masked rows."""
        order = sorted(range(len(samples)), key=lambda i: -len(samples[i].tokens))
        out = []
        while order:
            T = self._bucket(len(samples[order[0]].tokens))
            rows = self._rows(T)
            chunk, order = order[:rows], order[rows:]
            mb = {
                "tokens": np.full((rows, T), self.cfg.pad_id, np.int32),
                "segment_ids": np.zeros((rows, T), np.int32),
                "mask": np.zeros((rows, T), np.float32),
                "behavior_logp": np.zeros((rows, T), np.float32),
                "fresh": np.zeros((rows, T), np.float32),
                "advantage": np.zeros(rows, np.float32),
            }
            for r, i in enumerate(chunk):
                s = samples[i]
                L = len(s.tokens)
                mb["tokens"][r, :L] = s.tokens
                mb["segment_ids"][r, :L] = 1
                mb["mask"][r, :L] = s.mask
                mb["behavior_logp"][r, :L] = s.logprobs
                mb["fresh"][r, :L] = np.asarray(s.versions) == self.version
                mb["advantage"][r] = advantages[i]
            out.append(mb)
        return out

    def _put(self, mb):
        return {
            k: jax.device_put(v, self._row_sharding if v.ndim == 1 else self._batch_sharding)
            for k, v in mb.items()
        }

    def warmup(self) -> float:
        """Compiles the gradient program of every length bucket on all-masked rows, and the update
        ahead of time (it then loads from the persistent compilation cache)."""
        t0 = time.perf_counter()
        with jax.set_mesh(self.mesh):
            acc = self._zeros(self.params)
            for T in self.cfg.length_buckets:
                rows = self._rows(T)
                mb = {k: np.zeros((rows, T), np.int32 if k in ("tokens", "segment_ids") else np.float32)
                      for k in ("tokens", "segment_ids", "mask", "behavior_logp", "fresh")}
                mb["advantage"] = np.zeros(rows, np.float32)
                acc, _ = self._accumulate(self.params, acc, self._put(mb), jnp.float32(0.0))
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
        inv_tokens = jnp.float32(1.0 / total_tokens)
        with jax.set_mesh(self.mesh):
            acc = self._zeros(self.params)
            sums = []
            for mb in mbs:
                acc, m = self._accumulate(self.params, acc, self._put(mb), inv_tokens)
                sums.append(m)
            self.master, self.opt_state, grad_norm = self._apply(
                self.master, self.opt_state, jax.device_put(acc, self._opt_shardings)
            )
            self.params = jax.device_put(self.master, self._param_shardings)
        sums = jax.device_get(sums)
        totals = {k: sum(float(s[k]) for s in sums) for k in (sums[0] if sums else ())}
        jax.block_until_ready(self.params)
        seconds = time.perf_counter() - t0
        self.version += 1
        flops = sum(
            forward_flops_per_token(self.model_cfg, mb["tokens"].shape[1]).model_flops(
                base_weights_trainable=True
            )
            * float(mb["segment_ids"].sum())
            for mb in mbs
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
            "train/kl_k3": get("kl_k3") / n,
            "train/fresh_frac": get("fresh_tokens") / n,
            "train/fresh_abs_log_ratio": get("fresh_abs_log_ratio") / nf,
            "train/fresh_kl_k3": get("fresh_kl_k3") / nf,
            "train/pad_frac": 1.0 - sum(float(mb["segment_ids"].sum()) for mb in mbs) / padded,
            "train/model_tflop": flops / 1e12,
        }

    def logprobs(self, token_lists: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Trainer-forward ``logp[t]`` of each token given its prefix (``logp[0] = 0``)."""
        T = self._bucket(max(len(t) for t in token_lists))
        R = self._rows_multiple
        B = -(-len(token_lists) // R) * R
        tokens = np.full((B, T), self.cfg.pad_id, np.int32)
        segment_ids = np.zeros((B, T), np.int32)
        for b, t in enumerate(token_lists):
            tokens[b, : len(t)] = t
            segment_ids[b, : len(t)] = 1
        with jax.set_mesh(self.mesh):
            logp = self._logprobs(
                self.params,
                jax.device_put(tokens, self._batch_sharding),
                jax.device_put(segment_ids, self._batch_sharding),
            )
        logp = np.asarray(logp)
        return [logp[b, : len(t)] for b, t in enumerate(token_lists)]
