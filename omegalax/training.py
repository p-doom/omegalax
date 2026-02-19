import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from .model import ModelConfig, Qwen3, forward


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 8
    seq_len: int = 64
    num_steps: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    print_every: int = 1

    @classmethod
    def smoke(cls):
        return cls()


def init_model(cfg: ModelConfig, rng: jax.Array) -> Qwen3:
    return Qwen3(cfg, rngs=nnx.Rngs(rng))


def build_optimizer(model: Qwen3, train_cfg: TrainConfig) -> nnx.ModelAndOptimizer:
    tx = optax.adamw(learning_rate=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    return nnx.ModelAndOptimizer(model, tx)


def make_train_step(cfg: ModelConfig, pad_id: int = 0):
    @nnx.jit(donate_argnums=0)
    def train_step(optimizer: nnx.ModelAndOptimizer, tokens: jax.Array):
        def loss_fn(model: Qwen3):
            logits = forward(model, tokens, pad_id).astype(jnp.float32)
            targets = tokens[:, 1:]
            lm_logits = logits[:, :-1, :]
            mask = (targets != pad_id).astype(jnp.float32)
            ce = optax.softmax_cross_entropy_with_integer_labels(lm_logits, targets)
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            loss = jnp.sum(ce * mask) / denom
            acc = jnp.sum((jnp.argmax(lm_logits, axis=-1) == targets).astype(jnp.float32) * mask) / denom
            return loss, acc

        (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
        optimizer.update(grads)
        metrics = {
            "loss": loss,
            "token_accuracy": acc,
            "grad_norm": optax.tree.norm(grads),
        }
        return loss, metrics

    return train_step
