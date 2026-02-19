import numpy as np
from absl.testing import absltest
import jax
import jax.numpy as jnp

from omegalax.model import ModelConfig, decode
from omegalax.training import TrainConfig, build_optimizer, init_model, make_train_step


class SmokeTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.model_cfg = ModelConfig.smoke()
        self.train_cfg = TrainConfig.smoke()
        self.rng = jax.random.key(0)
        self.rng, init_rng = jax.random.split(self.rng)
        self.model = init_model(self.model_cfg, init_rng)
        self.optimizer = build_optimizer(self.model, self.train_cfg)

    def _tokens(self, batch_size: int, seq_len: int) -> jax.Array:
        self.rng, rng = jax.random.split(self.rng)
        return jax.random.randint(
            rng, (batch_size, seq_len), minval=0, maxval=self.model_cfg.vocab_size, dtype=jnp.int32
        )

    def test_decode_shape(self):
        tokens = self._tokens(batch_size=2, seq_len=16)
        cache = self.model.init_cache(self.model_cfg, batch_size=2, token_len=16, generate_steps=4, dtype=jnp.float32)
        logits, _ = decode(self.model, cache, tokens, pad_id=0)
        self.assertEqual(logits.shape, (2, self.model_cfg.vocab_size))
        self.assertTrue(np.isfinite(np.asarray(logits)).all())

    def test_train_step_smoke(self):
        train_step = make_train_step(self.model_cfg, pad_id=0)
        batch = self._tokens(batch_size=self.train_cfg.batch_size, seq_len=self.train_cfg.seq_len)
        _, metrics = train_step(self.optimizer, batch)
        host = jax.device_get(metrics)
        self.assertTrue(np.isfinite(float(host["loss"])))
        self.assertTrue(np.isfinite(float(host["grad_norm"])))
        self.assertTrue(np.isfinite(float(host["token_accuracy"])))


if __name__ == "__main__":
    absltest.main()
