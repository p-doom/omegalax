"""Tests for training FLOP counting and throughput metrics."""

import datetime
from unittest import mock

from absl.testing import absltest

from omegalax.distributed.mesh import process_local_batch_size
from omegalax.models.qwen3.config import make_config as make_qwen3_config
from omegalax.models.qwen3_5.config import make_config as make_qwen3_5_config
from omegalax.models.qwen3_vl.config import make_vl_config as make_qwen3_vl_config
from omegalax.trainers import perf
from omegalax.trainers.perf import (
    ForwardFlops,
    PEAK_TFLOPS,
    StepFlops,
    StepTimer,
    forward_flops_per_token,
    maybe_log_step_metrics,
    per_device_step_flops,
    qwen3_vl_vision_flops,
    qwen3_vl_vision_training_flops,
    resolve_peak_tflops,
    step_metrics,
    training_flops_per_token,
)


class ForwardFlopsPerTokenTest(absltest.TestCase):
    """Smoke tests + decomposition invariants for forward_flops_per_token."""

    def test_qwen3_smoke_dense_positive(self):
        cfg = make_qwen3_config("qwen3-smoke")
        fwd = forward_flops_per_token(cfg, seq_len=8)
        self.assertGreater(fwd.forward, 0)
        self.assertEqual(fwd.forward, fwd.weighted_layers + fwd.attention + fwd.head)
        self.assertEqual(training_flops_per_token(cfg, 8), 3 * fwd.forward)
        # qwen3-smoke: D=128, H=4, G=4, K=32, F=512, V=1024, L=2, T=8.
        # weighted_layers = 2*(qkv 98304 + o_proj 32768 + mlp 393216) = 1048576
        # attention = 2 * (2*T*H*K = 2048) = 4096 ; head = 2*D*V = 262144
        self.assertEqual(fwd.weighted_layers, 1_048_576)
        self.assertEqual(fwd.attention, 4_096)
        self.assertEqual(fwd.head, 262_144)
        self.assertEqual(fwd.forward, 1_314_816)

    def test_qwen3_smoke_moe_positive(self):
        cfg = make_qwen3_config("qwen3-smoke-moe")
        fwd = forward_flops_per_token(cfg, seq_len=8)
        self.assertGreater(fwd.forward, 0)
        self.assertEqual(fwd.forward, fwd.weighted_layers + fwd.attention + fwd.head)

    def test_qwen3_5_smoke_positive(self):
        full_cfg = make_qwen3_5_config("qwen3.5-smoke")
        fwd = forward_flops_per_token(full_cfg.text_config, seq_len=8)
        self.assertGreater(fwd.forward, 0)
        self.assertEqual(fwd.forward, fwd.weighted_layers + fwd.attention + fwd.head)

    def test_accepts_qwen3_5_full_config(self):
        full_cfg = make_qwen3_5_config("qwen3.5-smoke")
        self.assertEqual(
            forward_flops_per_token(full_cfg, 8).forward,
            forward_flops_per_token(full_cfg.text_config, 8).forward,
        )

    def test_attention_grows_with_seqlen(self):
        # Weightless attention scales with T; weighted/head do not.
        cfg = make_qwen3_config("qwen3-smoke")
        a = forward_flops_per_token(cfg, 8)
        b = forward_flops_per_token(cfg, 16)
        self.assertGreater(b.attention, a.attention)
        self.assertEqual(a.weighted_layers, b.weighted_layers)
        self.assertEqual(a.head, b.head)


class ModelHardwareFlopsTest(absltest.TestCase):
    """The core fix: LoRA-aware model FLOPs and remat-aware hardware FLOPs."""

    def _fwd(self):
        return ForwardFlops(weighted_layers=100, attention=30, head=10)

    def test_full_ft_model_is_3x_forward(self):
        fwd = self._fwd()
        self.assertEqual(fwd.model_flops(base_weights_trainable=True), 3 * fwd.forward)

    def test_lora_model_skips_frozen_weight_grads(self):
        # LoRA: frozen weighted matmuls 2x, weightless attention still 3x.
        fwd = self._fwd()
        expected = 3 * fwd.forward - (fwd.weighted_layers + fwd.head)
        self.assertEqual(fwd.model_flops(base_weights_trainable=False), expected)
        self.assertEqual(
            fwd.model_flops(base_weights_trainable=False),
            2 * (fwd.weighted_layers + fwd.head) + 3 * fwd.attention,
        )

    def test_hardware_no_remat_equals_model(self):
        fwd = self._fwd()
        for trainable in (True, False):
            self.assertEqual(
                fwd.hardware_flops(base_weights_trainable=trainable, decoder_remat=False),
                fwd.model_flops(base_weights_trainable=trainable),
            )

    def test_hardware_with_remat_adds_layer_recompute(self):
        # Remat recomputes the layer forward (weighted_layers + attention), not the head.
        fwd = self._fwd()
        recompute = fwd.weighted_layers + fwd.attention
        for trainable in (True, False):
            m = fwd.model_flops(base_weights_trainable=trainable)
            h = fwd.hardware_flops(base_weights_trainable=trainable, decoder_remat=True)
            self.assertEqual(h, m + recompute)

    def test_full_ft_remat_hfu_exceeds_mfu(self):
        fwd = self._fwd()
        m = fwd.model_flops(base_weights_trainable=True)
        h = fwd.hardware_flops(base_weights_trainable=True, decoder_remat=True)
        self.assertGreater(h, m)


class VisionFlopsTest(absltest.TestCase):
    def test_vision_flops_positive_and_block_subset(self):
        cfg = make_qwen3_vl_config("qwen3-vl-smoke")
        vf = qwen3_vl_vision_flops(cfg, [[1, 4, 4]])
        self.assertGreater(vf.forward, 0)
        self.assertGreater(vf.block_forward, 0)
        self.assertLessEqual(vf.block_forward, vf.forward)

    def test_vision_training_flops_frozen_is_forward_only(self):
        cfg = make_qwen3_vl_config("qwen3-vl-smoke")
        grid = [[1, 4, 4]]
        trained = qwen3_vl_vision_training_flops(cfg, grid, vision_trainable=True)
        frozen = qwen3_vl_vision_training_flops(cfg, grid, vision_trainable=False)
        self.assertEqual(trained, 3 * frozen)

    def test_vision_block_diagonal_attention(self):
        cfg = make_qwen3_vl_config("qwen3-vl-smoke")
        single = qwen3_vl_vision_flops(cfg, [[1, 4, 4]]).forward
        doubled = qwen3_vl_vision_flops(cfg, [[1, 4, 4], [1, 4, 4]]).forward
        self.assertEqual(doubled, 2 * single)


class PerDeviceStepFlopsTest(absltest.TestCase):
    def _kwargs(self, **overrides):
        kw = dict(
            base_weights_trainable=True,
            vision_trainable=True,
            decoder_remat=False,
            vision_remat=False,
        )
        kw.update(overrides)
        return kw

    def test_positive(self):
        cfg = make_qwen3_config("qwen3-smoke")
        sf = per_device_step_flops(cfg, seq_len=8, batch_size=2, **self._kwargs())
        self.assertGreater(sf.model, 0)
        self.assertEqual(sf.model, sf.hardware)  # remat off

    def test_lora_model_below_full_ft(self):
        cfg = make_qwen3_config("qwen3-smoke")
        full = per_device_step_flops(
            cfg, seq_len=8, batch_size=2, **self._kwargs(base_weights_trainable=True)
        )
        lora = per_device_step_flops(
            cfg, seq_len=8, batch_size=2, **self._kwargs(base_weights_trainable=False)
        )
        self.assertLess(lora.model, full.model)

    def test_remat_raises_hardware_above_model(self):
        cfg = make_qwen3_config("qwen3-smoke")
        sf = per_device_step_flops(cfg, seq_len=8, batch_size=2, **self._kwargs(decoder_remat=True))
        self.assertGreater(sf.hardware, sf.model)

    def test_vl_adds_vision_cost(self):
        cfg = make_qwen3_vl_config("qwen3-vl-smoke")
        with mock.patch("jax.device_count", return_value=1):
            base = per_device_step_flops(cfg, seq_len=8, batch_size=2, **self._kwargs())
            with_images = per_device_step_flops(
                cfg, seq_len=8, batch_size=2, image_grid_thw=[[1, 4, 4]], **self._kwargs()
            )
        self.assertGreater(with_images.model, base.model)

    def test_vl_frozen_vision_forward_only(self):
        # Frozen vision (x1) vs trained (x3): vision delta shrinks to 1/3, no recompute.
        cfg = make_qwen3_vl_config("qwen3-vl-smoke")
        grid = [[1, 4, 4]]
        with mock.patch("jax.device_count", return_value=1):
            base = per_device_step_flops(cfg, seq_len=8, batch_size=2, **self._kwargs())
            trained = per_device_step_flops(
                cfg, seq_len=8, batch_size=2, image_grid_thw=grid, **self._kwargs()
            )
            frozen = per_device_step_flops(
                cfg,
                seq_len=8,
                batch_size=2,
                image_grid_thw=grid,
                **self._kwargs(vision_trainable=False, vision_remat=True),
            )
        self.assertEqual(frozen.model, base.model + (trained.model - base.model) / 3)
        self.assertEqual(frozen.model, frozen.hardware)


class StepMetricsTest(absltest.TestCase):
    def test_zero_delta(self):
        out = step_metrics(StepFlops(1e12, 1e12), datetime.timedelta(0), 64, 312.0)
        self.assertEqual(out["step_time_s"], 0.0)
        self.assertEqual(out["mfu"], 0.0)
        self.assertEqual(out["hfu"], 0.0)
        self.assertEqual(out["tflops_per_device"], 0.0)

    def test_mfu_uses_model_hfu_uses_hardware(self):
        # model 1e12 FLOP/s, hardware 2e12 FLOP/s in 1s; peak 312.
        out = step_metrics(
            StepFlops(model=1e12, hardware=2e12), datetime.timedelta(seconds=1), 64, 312.0
        )
        self.assertAlmostEqual(out["model_tflops_per_device"], 1.0)
        self.assertAlmostEqual(out["hardware_tflops_per_device"], 2.0)
        self.assertAlmostEqual(out["tflops_per_device"], 1.0)  # alias of model
        self.assertAlmostEqual(out["mfu"], 1.0 / 312.0)
        self.assertAlmostEqual(out["hfu"], 2.0 / 312.0)
        self.assertAlmostEqual(out["global_tokens_per_sec"], 64.0)

    def test_no_peak_skips_utilization(self):
        out = step_metrics(StepFlops(1e12, 2e12), datetime.timedelta(seconds=1), 64, None)
        self.assertEqual(out["mfu"], 0.0)
        self.assertEqual(out["hfu"], 0.0)
        self.assertGreater(out["model_tflops_per_device"], 0)


class MaybeLogStepMetricsTest(absltest.TestCase):
    """The utilisation numbers a run is judged on come out of here.

    They were all read with a ``0.0`` default, so renaming one in ``step_metrics``
    logged and returned zero for it instead of raising -- on exactly the figures
    with a history of being misread.
    """

    #: What both trainers put in `window_metrics` (vlm.py, text.py).
    _CALLER_METRICS = {
        "loss": 1.5,
        "grad_norm": 0.5,
        "supervised_tokens": 64.0,
        "total_tokens": 128.0,
        "lr": 1e-4,
    }

    def _log(self, metrics=None):
        return maybe_log_step_metrics(
            1,
            dict(self._CALLER_METRICS if metrics is None else metrics),
            datetime.timedelta(seconds=1),
            is_primary_process=True,
            log_every=1,
            step_flops=StepFlops(model=1e12, hardware=2e12),
            global_tokens_per_step=64,
            peak_tflops=312.0,
            batch_size=8,
        )

    def test_logs_the_utilization_metrics_step_metrics_produced(self):
        out = self._log()
        self.assertAlmostEqual(out["mfu"], 1.0 / 312.0)
        self.assertAlmostEqual(out["hfu"], 2.0 / 312.0)
        self.assertEqual(out["total_samples"], 8)

    def test_a_renamed_step_metric_raises(self):
        renamed = dict(step_metrics(StepFlops(1e12, 2e12), datetime.timedelta(seconds=1), 64, 312.0))
        renamed["model_flops_utilization"] = renamed.pop("mfu")
        with mock.patch.object(perf, "step_metrics", return_value=renamed):
            with self.assertRaisesRegex(KeyError, "mfu"):
                self._log()

    def test_a_caller_metric_the_print_reads_is_required(self):
        for key in ("lr", "supervised_tokens", "total_tokens"):
            metrics = dict(self._CALLER_METRICS)
            metrics.pop(key)
            with self.assertRaisesRegex(KeyError, key):
                self._log(metrics)


class ProcessLocalBatchSizeTest(absltest.TestCase):
    def test_returns_process_local_batch_size(self):
        self.assertEqual(process_local_batch_size(8, dp_size=4, fsdp_size=1), 2)
        self.assertEqual(process_local_batch_size(8, dp_size=2, fsdp_size=2), 2)

    def test_rejects_non_divisible_global_batch_size(self):
        with self.assertRaisesRegex(ValueError, "divisible by data_parallel_size=3"):
            process_local_batch_size(8, dp_size=3, fsdp_size=1)


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
    """``resolve_peak_tflops`` is what reads the table; the table itself is a literal."""

    def test_every_preset_name_resolves_to_its_table_value(self):
        for name, expected in PEAK_TFLOPS.items():
            self.assertEqual(resolve_peak_tflops(name), expected, msg=name)

    def test_h100_sxm_is_the_name_the_launchers_pass(self):
        self.assertEqual(resolve_peak_tflops("h100_sxm"), 989.0)

    def test_numeric_spec_passes_through(self):
        self.assertEqual(resolve_peak_tflops("312"), 312.0)
        self.assertEqual(resolve_peak_tflops(989.0), 989.0)

    def test_none_disables_utilization(self):
        self.assertIsNone(resolve_peak_tflops(None))

    def test_unknown_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown peak_tflops 'h100'"):
            resolve_peak_tflops("h100")


if __name__ == "__main__":
    absltest.main()
