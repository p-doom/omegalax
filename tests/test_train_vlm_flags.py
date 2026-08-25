"""VLM front-door flag validation."""

import json
import subprocess
import sys

from absl.testing import absltest

_VALID = {
    "model_snapshot": "/snapshot",
    "data_path": "/data",
    "data_mix": None,
    "max_length": 8,
    "num_steps": 10,
    "schedule_horizon": 100,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "warmup_steps": 0,
    "lr_schedule": "linear",
    "lr_end_factor": None,
    "lr_stable_fraction": None,
    "max_grad_norm": 1.0,
    "grad_accum_steps": 1,
    "gc_period": 0,
    "seed": 0,
    "tp_size": 1,
    "fsdp_size": 1,
    "dp_size": 1,
    "save_dir": "/checkpoints",
    "jax_cache_dir": "/jax-cache",
    "save_every": 5,
    "keep_period": 0,
    "keep_latest": 1,
    "log_every": 1,
    "log_memory": False,
    "resume": "never",
    "resume_step": None,
    "pad_id": 0,
    "peak_tflops": "auto",
    "grain_read_threads": 1,
    "grain_read_buffer_size": 1,
    "grain_workers": 0,
    "grain_worker_buffer_size": 1,
    "max_vision_patches_per_sample": 0,
    "max_vision_images_per_sample": 0,
    "num_loss_tiles": 4,
    "text_attn_backend": "xla",
    "enable_lora": False,
    "lora_rank": None,
    "lora_alpha": None,
    "freeze_vision_tower": False,
    "val_data_path": None,
    "val_every": None,
    "val_steps": None,
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_group": None,
    "wandb_name": None,
    "wandb_tags": None,
}


class TrainVLMFlagsTest(absltest.TestCase):
    def test_valid_and_invalid_contracts_under_optimized_python(self):
        code = """
import json, sys
from absl.testing import flagsaver
from scripts import train_vlm_sft
train_vlm_sft.FLAGS(['test'])
values = json.loads(sys.argv[1])
with flagsaver.flagsaver(**values):
    train_vlm_sft._validate_flags()
values['batch_size'] = 0
with flagsaver.flagsaver(**values):
    try:
        train_vlm_sft._validate_flags()
    except ValueError as error:
        if 'batch_size must be > 0' not in str(error):
            raise
    else:
        raise RuntimeError('invalid batch_size was accepted')
values['batch_size'] = 2
values['save_every'] = 0
values['keep_period'] = 5
with flagsaver.flagsaver(**values):
    try:
        train_vlm_sft._validate_flags()
    except ValueError as error:
        if 'keep_period requires save_every > 0' not in str(error):
            raise
    else:
        raise RuntimeError('unreachable retention policy was accepted')
"""
        payload = json.dumps(_VALID)
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", code, payload])
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    absltest.main()
