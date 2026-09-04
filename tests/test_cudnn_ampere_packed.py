from __future__ import annotations

from unittest import mock

from absl.testing import absltest

from omegalax.compat import cudnn_ampere_packed


class CudnnAmperePackedTest(absltest.TestCase):
    def test_packed_attention_disables_the_packed_validator_gate(self):
        received = None

        def check(
            query,
            key,
            value,
            layout,
            cudnn_version,
            has_bias,
            is_training,
            is_packed=False,
            is_paged_attention=False,
            is_fp8=False,
        ):
            nonlocal received
            received = (
                query,
                key,
                value,
                layout,
                cudnn_version,
                has_bias,
                is_training,
                is_packed,
                is_paged_attention,
                is_fp8,
            )

        with (
            mock.patch.object(
                cudnn_ampere_packed._fused_attention,
                "check_is_flash_attention",
                check,
            ),
            mock.patch.object(
                cudnn_ampere_packed._fused_attention,
                "check_compute_capability",
                side_effect=lambda capability: capability == "8.0",
            ),
        ):
            cudnn_ampere_packed.enable_ampere_packed_attention()
            cudnn_ampere_packed._fused_attention.check_is_flash_attention(
                "query", "key", "value", 0, 0, False, True, True, True, False
            )

        self.assertEqual(
            received,
            ("query", "key", "value", 0, 0, False, True, False, True, False),
        )

    def test_unsupported_gpu_fails_before_patch(self):
        check = cudnn_ampere_packed._fused_attention.check_is_flash_attention
        with mock.patch.object(
            cudnn_ampere_packed._fused_attention,
            "check_compute_capability",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "compute capability 8.0"):
                cudnn_ampere_packed.enable_ampere_packed_attention()

        self.assertIs(cudnn_ampere_packed._fused_attention.check_is_flash_attention, check)


if __name__ == "__main__":
    absltest.main()
