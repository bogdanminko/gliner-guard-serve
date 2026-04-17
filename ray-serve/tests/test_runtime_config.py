import os
import unittest
from unittest.mock import patch

import torch

from runtime_config import resolve_torch_dtype


class ResolveTorchDTypeTest(unittest.TestCase):
    def test_defaults_to_bf16(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TORCH_DTYPE", None)
            config = resolve_torch_dtype()
        self.assertEqual(config.name, "bf16")
        self.assertEqual(config.torch_dtype, torch.bfloat16)
        self.assertEqual(config.runtime_prefix, "pytorch-bf16")

    def test_float16_aliases_normalize_to_fp16(self) -> None:
        config = resolve_torch_dtype("float16")
        self.assertEqual(config.name, "fp16")
        self.assertEqual(config.torch_dtype, torch.float16)
        self.assertEqual(config.runtime_prefix, "pytorch-fp16")

    def test_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_torch_dtype("int8")
