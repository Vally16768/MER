from __future__ import annotations

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from augmentation import apply_modality_dropout  # noqa: E402


class TestAugmentation(unittest.TestCase):
    def test_modality_dropout_never_drops_both(self):
        # Use non-zero constant tensors so "dropped" == all-zeros.
        bsz = 256
        audio = torch.ones(bsz, 8)
        text = torch.ones(bsz, 9) * 2.0

        # With p=1.0, a naive implementation would drop both; ours must keep one.
        a2, t2, stats = apply_modality_dropout(audio=audio, text=text, p=1.0)
        self.assertIsNotNone(stats)
        assert a2 is not None and t2 is not None

        both_zero = (a2.abs().sum(dim=1) == 0) & (t2.abs().sum(dim=1) == 0)
        self.assertFalse(bool(both_zero.any()))


if __name__ == "__main__":
    unittest.main()

