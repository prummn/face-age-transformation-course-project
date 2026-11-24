"""Regression checks for configuration and FP32 style projection."""
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_quality import _fp32_modulation, resolve_wplus_precision


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.torch = Mock()
        self.torch.cuda.device.return_value = nullcontext()

    def test_auto_uses_native_bf16_for_constrained(self):
        self.torch.cuda.is_bf16_supported.return_value = True
        self.assertEqual(resolve_wplus_precision(self.torch, "cuda:1", "constrained", "auto"), "bf16")
        self.torch.cuda.device.assert_called_once_with("cuda:1")
        self.torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)

    def test_auto_falls_back_and_explicit_bf16_reports_unsupported_device(self):
        self.torch.cuda.is_bf16_supported.return_value = False
        self.assertEqual(resolve_wplus_precision(self.torch, "cuda:0", "constrained", "auto"), "fp32")
        with self.assertRaisesRegex(ValueError, "Native BF16 is unavailable"):
            resolve_wplus_precision(self.torch, "cuda:0", "constrained", "bf16")

    def test_fp32_and_other_auto_profiles_do_not_probe_cuda(self):
        self.assertEqual(resolve_wplus_precision(self.torch, "cuda:0", "constrained", "fp32"), "fp32")
        self.assertEqual(resolve_wplus_precision(self.torch, "cuda:0", "texture", "auto"), "fp32")
        self.torch.cuda.device.assert_not_called()

    def test_explicit_bf16_can_select_a_different_wplus_profile(self):
        self.torch.cuda.is_bf16_supported.return_value = True
        self.assertEqual(resolve_wplus_precision(self.torch, "cuda:0", "texture", "bf16"), "bf16")

    def test_unknown_precision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be auto, fp32, or bf16"):
            resolve_wplus_precision(self.torch, "cuda:0", "constrained", "fp16")

    def test_projection_receives_fp32_inputs_with_autocast_disabled(self):
        self.torch.autocast.return_value = nullcontext()
        style = Mock()
        forward = Mock()
        result = _fp32_modulation(self.torch, forward)(style)
        self.torch.autocast.assert_called_once_with(device_type="cuda", enabled=False)
        style.float.assert_called_once_with()
        forward.assert_called_once_with(style.float.return_value)
        self.assertIs(result, forward.return_value)
