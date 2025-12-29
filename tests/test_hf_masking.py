from __future__ import annotations

import unittest

import os
import sys

import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from models.hf_at import _infer_feature_vector_attention_mask  # noqa: E402


class _DummySpeechEncoder(nn.Module):
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = int(stride)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:  # noqa: SLF001
        s = int(self.stride)
        return torch.ceil(input_lengths.to(dtype=torch.float32) / float(s)).to(dtype=torch.long)


class TestHFAudioMasking(unittest.TestCase):
    def test_mask_passthrough_when_aligned(self):
        enc = _DummySpeechEncoder(stride=2)
        m = torch.ones((3, 10), dtype=torch.long)
        out = _infer_feature_vector_attention_mask(enc, m, hidden_len=10)
        self.assertTrue(out is m)

    def test_mask_downsamples_with_encoder_lengths(self):
        enc = _DummySpeechEncoder(stride=10)
        # Two samples: lengths 100 and 73.
        m = torch.zeros((2, 100), dtype=torch.long)
        m[0, :100] = 1
        m[1, :73] = 1
        out = _infer_feature_vector_attention_mask(enc, m, hidden_len=10)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(tuple(out.shape), (2, 10))
        self.assertTrue(torch.all(out[0] == 1))
        # ceil(73/10)=8 -> first 8 ones, then zeros.
        self.assertTrue(torch.all(out[1, :8] == 1))
        self.assertTrue(torch.all(out[1, 8:] == 0))

    def test_mask_fallback_ratio(self):
        class NoLenEncoder(nn.Module):
            pass

        enc = NoLenEncoder()
        m = torch.zeros((1, 100), dtype=torch.long)
        m[0, :50] = 1
        out = _infer_feature_vector_attention_mask(enc, m, hidden_len=10)
        self.assertIsNotNone(out)
        assert out is not None
        # ratio = 10/100, lengths=50 -> ceil(5)=5
        self.assertTrue(torch.all(out[0, :5] == 1))
        self.assertTrue(torch.all(out[0, 5:] == 0))


if __name__ == "__main__":
    unittest.main()

