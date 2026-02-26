"""Tests for training FLOP counting and throughput metrics."""

from absl.testing import absltest

from omegalax.models.qwen3.dense.config import make_dense_config
from omegalax.models.qwen3.moe.config import make_moe_config
from omegalax.models.qwen3_5.config import make_config as make_qwen3_5_config
from omegalax.trainers.perf import (
    PEAK_TFLOPS,
    training_flops_per_token,
    per_device_flops_per_step,
    step_metrics,
    StepTimer,
)


class TrainingFlopsPerTokenTest(absltest.TestCase):
    """Smoke tests for training_flops_per_token with small configs."""

    def test_qwen3_smoke_dense_positive(self):
        cfg = make_dense_config("qwen3-smoke")
        seq_len = 8
        flops = training_flops_per_token(cfg, seq_len)
        self.assertGreater(flops, 0)
        # qwen3-smoke: D=128, H=4, G=4, K=32, F=512, V=1024, L=2
        # per layer attn ~ 2*128*12*32 + 4*8*4*32 + 2*4*32*128 = 98304+4096+32768 = 135168
        # per layer mlp = 6*128*512 = 393216; embed = 2*128*1024 = 262144
        # forward = 2*(135168+393216)+262144 = 1318912; *3 = 3956736
        self.assertGreaterEqual(flops, 3_000_000)
        self.assertLessEqual(flops, 5_000_000)

    def test_qwen3_smoke_moe_positive(self):
        cfg = make_moe_config("qwen3-smoke-moe")
        seq_len = 8
        flops = training_flops_per_token(cfg, seq_len)
        self.assertGreater(flops, 0)
        # MoE smoke: 2 layers, small experts
        self.assertGreaterEqual(flops, 1_000_000)
        self.assertLessEqual(flops, 20_000_000)

    def test_qwen3_5_smoke_positive(self):
        full_cfg = make_qwen3_5_config("qwen3.5-smoke")
        cfg = full_cfg.text_config
        seq_len = 8
        flops = training_flops_per_token(cfg, seq_len)
        self.assertGreater(flops, 0)
        # 4 layers (linear + full), small dims
        self.assertGreaterEqual(flops, 500_000)
        self.assertLessEqual(flops, 50_000_000)

    def test_training_flops_per_token_accepts_qwen3_5_full_config(self):
        full_cfg = make_qwen3_5_config("qwen3.5-smoke")
        seq_len = 8
        flops = training_flops_per_token(full_cfg, seq_len)
        self.assertGreater(flops, 0)
        self.assertEqual(flops, training_flops_per_token(full_cfg.text_config, seq_len))


class PerDeviceFlopsStepTest(absltest.TestCase):
    def test_per_device_flops_per_step_positive(self):
        cfg = make_dense_config("qwen3-smoke")
        flops = per_device_flops_per_step(cfg, seq_len=8, batch_size=2)
        self.assertGreater(flops, 0)


class StepMetricsTest(absltest.TestCase):
    def test_step_metrics_zero_delta(self):
        import datetime
        out = step_metrics(1e12, datetime.timedelta(0), 64, 312.0)
        self.assertEqual(out["step_time_s"], 0.0)
        self.assertEqual(out["tokens_per_sec_per_device"], 0.0)
        self.assertEqual(out["tflops_per_device"], 0.0)
        self.assertEqual(out["mfu"], 0.0)

    def test_step_metrics_positive_delta(self):
        import datetime
        # 1e12 FLOPs in 1 second -> 1 TFLOP/s; peak 312 -> mfu = 1/312
        out = step_metrics(1e12, datetime.timedelta(seconds=1), 64, 312.0)
        self.assertAlmostEqual(out["step_time_s"], 1.0)
        self.assertGreater(out["tokens_per_sec_per_device"], 0)
        self.assertAlmostEqual(out["tflops_per_device"], 1.0)
        self.assertAlmostEqual(out["mfu"], 1.0 / 312.0)

    def test_step_metrics_no_peak_skips_mfu(self):
        import datetime
        out = step_metrics(1e12, datetime.timedelta(seconds=1), 64, None)
        self.assertEqual(out["mfu"], 0.0)
        self.assertGreater(out["tflops_per_device"], 0)


class StepTimerTest(absltest.TestCase):
    def test_warmup_returns_zero_delta(self):
        t = StepTimer(warmup=2)
        self.assertEqual(t.step().total_seconds(), 0)
        self.assertEqual(t.step().total_seconds(), 0)

    def test_after_warmup_returns_positive_delta(self):
        t = StepTimer(warmup=0)
        d = t.step()
        self.assertGreaterEqual(d.total_seconds(), 0)


class PeakTflopsTest(absltest.TestCase):
    def test_peak_tflops_entries_positive(self):
        for name, v in PEAK_TFLOPS.items():
            self.assertGreater(v, 0, msg=name)


if __name__ == "__main__":
    absltest.main()
