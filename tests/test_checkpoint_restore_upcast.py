"""Full resume upcasts bf16-parameter checkpoints into fp32-parameter train states."""

import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5.config import make_config
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.trainers import checkpoint_utils
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.pretrain import _abstract_curriculum_train_state, _curriculum_train_state


def _optimizer(param_dtype):
    cfg = dataclasses.replace(make_config("qwen3.5-smoke").text_config, param_dtype=param_dtype)
    model = Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(0))
    return MixedPrecisionOptimizer(model, optax.adamw(1e-3), wrt=nnx.Param)


class CheckpointRestoreUpcastTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1))

    def test_bf16_param_checkpoint_resumes_into_fp32_params(self):
        rng = jax.device_put(jax.random.key(0), jax.sharding.SingleDeviceSharding(jax.devices()[0]))
        old = _optimizer(jnp.bfloat16)
        old.update(jax.tree.map(lambda p: jnp.ones_like(p) * 1e-3, nnx.state(old.model, nnx.Param)))
        saved = _curriculum_train_state(old, rng, global_step=7, phase_idx=1, phase_step=3)
        path = os.path.join(self.create_tempdir().full_path, "ckpt")
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, saved)

        new = _optimizer(jnp.float32)
        args = checkpoint_utils.make_grain_restore_args(
            _abstract_curriculum_train_state(new, rng), None
        )
        restored = checkpointer.restore(path, args=args["train_state"])
        nnx.update(new, restored["optimizer"])

        for leaf in jax.tree.leaves(nnx.state(new.model, nnx.Param)):
            self.assertEqual(leaf.dtype, jnp.float32)
        for before, after in zip(
            jax.tree.leaves(nnx.state(old.model, nnx.Param)),
            jax.tree.leaves(nnx.state(new.model, nnx.Param)),
        ):
            self.assertTrue(bool((before.astype(jnp.float32) == after).all()))
        self.assertEqual(int(restored["global_step"]), 7)
        self.assertEqual(int(restored["phase_step"]), 3)


if __name__ == "__main__":
    absltest.main()
