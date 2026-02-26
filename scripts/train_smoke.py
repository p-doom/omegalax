import argparse

import jax
import jax.numpy as jnp

from omegalax.text import api as text_api
from omegalax.trainers import text as text_trainer


def make_synthetic_batch(rng: jax.Array, batch_size: int, seq_len: int, vocab_size: int) -> jax.Array:
    return jax.random.randint(rng, (batch_size, seq_len), minval=0, maxval=vocab_size, dtype=jnp.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic smoke-training loop for omegalax.")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--fsdp-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.distributed.initialize()

    model_cfg = text_api.registry.build_config(args.model_id)
    train_cfg = text_trainer.TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        learning_rate=text_trainer.TrainConfig.smoke().learning_rate,
        weight_decay=text_trainer.TrainConfig.smoke().weight_decay,
        print_every=args.print_every,
    )

    optimizer, _ = text_trainer.run_training(
        model_cfg,
        train_cfg,
        log_every=args.print_every,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
    )

    rng = jax.random.key(args.seed)
    prompt = make_synthetic_batch(rng, batch_size=1, seq_len=8, vocab_size=model_cfg.vocab_size)
    cache = text_api.make_cache(model_cfg, batch_size=1, token_len=8, generate_steps=4, dtype=jnp.float32)
    logits, cache, aux_loss = text_api.decode(optimizer.model, cache, prompt, pad_id=0, cfg=model_cfg)
    _ = jax.block_until_ready(logits)
    _ = jax.block_until_ready(aux_loss)
    print("smoke training loop completed")


if __name__ == "__main__":
    main()
