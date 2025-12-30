from __future__ import annotations

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from models.flexible_at import FlexibleATModel  # noqa: E402
from reproducibility import set_seed  # noqa: E402


class TestReproducibility(unittest.TestCase):
    def test_seed_deterministic_forward(self):
        device = torch.device("cpu")
        audio = torch.randn(2, 160, device=device)
        text = torch.randn(2, 768, device=device)

        set_seed(123)
        m1 = FlexibleATModel(
            input_dim_audio=160,
            input_dim_text=768,
            gated_dim=128,
            n_classes=7,
            drop=0.0,
            modalities=["A", "T"],
        ).to(device)
        y1 = m1({"audio": audio, "text": text}).detach().cpu()

        set_seed(123)
        m2 = FlexibleATModel(
            input_dim_audio=160,
            input_dim_text=768,
            gated_dim=128,
            n_classes=7,
            drop=0.0,
            modalities=["A", "T"],
        ).to(device)
        y2 = m2({"audio": audio, "text": text}).detach().cpu()

        self.assertTrue(torch.allclose(y1, y2, atol=0.0, rtol=0.0))


if __name__ == "__main__":
    unittest.main()

