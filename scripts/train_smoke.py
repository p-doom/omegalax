import argparse

import jax
import jax.numpy as jnp

from omegalax.model import ModelConfig, decode
from omegalax.training import TrainConfig, build_optimizer, init_model, make_train_step


def make_synthetic_batch(rng: jax.Array, batch_size: int, seq_len: int, vocab_size: int) -> jax.Array:
    return jax.random.randint(rng, (batch_size, seq_len), minval=0, maxval=vocab_size, dtype=jnp.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic smoke-training loop for omegalax.")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = ModelConfig.smoke()
    train_cfg = TrainConfig.smoke()
    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        print_every=args.print_every,
    )

    rng = jax.random.key(train_cfg.seed)
    rng, init_rng = jax.random.split(rng)

    model = init_model(model_cfg, init_rng)
    optimizer = build_optimizer(model, train_cfg)
    train_step = make_train_step(model_cfg, pad_id=0)

    for step in range(train_cfg.num_steps):
        rng, batch_rng = jax.random.split(rng)
        batch = make_synthetic_batch(batch_rng, train_cfg.batch_size, train_cfg.seq_len, model_cfg.vocab_size)
        _, metrics = train_step(optimizer, batch)
        if step % train_cfg.print_every == 0:
            host_metrics = jax.device_get(metrics)
            print(
                f"step={step + 1} "
                f"loss={float(host_metrics['loss']):.4f} "
                f"acc={float(host_metrics['token_accuracy']):.4f} "
                f"grad_norm={float(host_metrics['grad_norm']):.4f}"
            )

    prompt = make_synthetic_batch(rng, batch_size=1, seq_len=8, vocab_size=model_cfg.vocab_size)
    cache = model.init_cache(model_cfg, batch_size=1, token_len=8, generate_steps=4, dtype=jnp.float32)
    logits, cache = decode(model, cache, prompt, pad_id=0)
    _ = jax.device_get((logits.shape, len(cache)))
    print("smoke training loop completed")


if __name__ == "__main__":
    main()
