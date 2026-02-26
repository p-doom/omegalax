import dataclasses
import numpy as np
from absl.testing import absltest
import jax
import jax.numpy as jnp

from omegalax.text import api
from omegalax.trainers import text as text_trainer


class SmokeTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.model_cfg = api.registry.build_config("qwen3-smoke")
        self.train_cfg = text_trainer.TrainConfig.smoke()
        self.rng = jax.random.key(0)
        self.rng, init_rng = jax.random.split(self.rng)
        self.model = text_trainer.init_model(self.model_cfg, init_rng, tp_size=1, fsdp_size=1)

    def _tokens(self, batch_size: int, seq_len: int) -> jax.Array:
        self.rng, rng = jax.random.split(self.rng)
        return jax.random.randint(
            rng, (batch_size, seq_len), minval=0, maxval=self.model_cfg.vocab_size, dtype=jnp.int32
        )

    def test_decode_shape(self):
        token_ids_BT = self._tokens(batch_size=2, seq_len=16)
        cache = api.make_cache(self.model_cfg, batch_size=2, token_len=16, generate_steps=4, dtype=self.model_cfg.dtype)
        logits_BV, _, aux_loss = api.decode(self.model, cache, token_ids_BT, pad_id=0, cfg=self.model_cfg)
        self.assertEqual(logits_BV.shape, (2, self.model_cfg.vocab_size))
        self.assertTrue(np.isfinite(np.asarray(logits_BV)).all())
        self.assertTrue(np.isfinite(float(aux_loss)))

    def test_train_step_smoke(self):
        one_step_cfg = dataclasses.replace(self.train_cfg, num_steps=1, batch_size=2, seq_len=8)
        _, metrics = text_trainer.run_training(
            self.model_cfg,
            one_step_cfg,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
        )
        self.assertTrue(np.isfinite(float(metrics["loss"])))
        self.assertTrue(np.isfinite(float(metrics["grad_norm"])))

    def test_moe_forward_smoke(self):
        moe_cfg = api.registry.build_config("qwen3-smoke-moe")
        moe_model = text_trainer.init_model(moe_cfg, self.rng, tp_size=1, fsdp_size=1)
        token_ids_BT = self._tokens(batch_size=2, seq_len=8)
        logits_BTV, aux_loss = api.forward(moe_model, token_ids_BT, pad_id=0, cfg=moe_cfg)
        self.assertEqual(logits_BTV.shape[:2], (2, 8))
        self.assertTrue(np.isfinite(np.asarray(logits_BTV)).all())
        self.assertTrue(np.isfinite(float(aux_loss)))


if __name__ == "__main__":
    absltest.main()
