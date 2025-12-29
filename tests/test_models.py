from __future__ import annotations

import unittest

import torch

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from models.flexible_at import FlexibleATModel  # noqa: E402
from models.robust_at import RobustATModel  # noqa: E402


class TestModelShapes(unittest.TestCase):
    def test_flexible_at_shapes(self):
        model = FlexibleATModel(
            input_dim_audio=160,
            input_dim_text=768,
            gated_dim=128,
            n_classes=7,
            drop=0.0,
            modalities=["A", "T"],
        ).cpu()

        batch = {"audio": torch.randn(4, 160), "text": torch.randn(4, 768)}
        out = model(batch)
        self.assertEqual(tuple(out.shape), (4, 7))

        model_a = FlexibleATModel(
            input_dim_audio=160,
            input_dim_text=768,
            gated_dim=128,
            n_classes=7,
            drop=0.0,
            modalities=["A"],
        ).cpu()
        out_a = model_a({"audio": torch.randn(2, 160), "text": None})
        self.assertEqual(tuple(out_a.shape), (2, 7))

        model_t = FlexibleATModel(
            input_dim_audio=160,
            input_dim_text=768,
            gated_dim=128,
            n_classes=7,
            drop=0.0,
            modalities=["T"],
        ).cpu()
        out_t = model_t({"audio": None, "text": torch.randn(3, 768)})
        self.assertEqual(tuple(out_t.shape), (3, 7))

    def test_robust_at_shapes(self):
        model = RobustATModel(
            input_dim_audio=160,
            input_dim_text=768,
            hidden_dim=256,
            n_classes=7,
            num_layers=2,
            num_heads=8,
            ffn_mult=4,
            modalities=["A", "T"],
        ).cpu()

        batch = {"audio": torch.randn(4, 160), "text": torch.randn(4, 768)}
        out = model(batch)
        self.assertEqual(tuple(out.shape), (4, 7))

        model_a = RobustATModel(
            input_dim_audio=160,
            input_dim_text=768,
            hidden_dim=256,
            n_classes=7,
            num_layers=2,
            num_heads=8,
            ffn_mult=4,
            modalities=["A"],
        ).cpu()
        out_a = model_a({"audio": torch.randn(2, 160), "text": None})
        self.assertEqual(tuple(out_a.shape), (2, 7))

        model_t = RobustATModel(
            input_dim_audio=160,
            input_dim_text=768,
            hidden_dim=256,
            n_classes=7,
            num_layers=2,
            num_heads=8,
            ffn_mult=4,
            modalities=["T"],
        ).cpu()
        out_t = model_t({"audio": None, "text": torch.randn(3, 768)})
        self.assertEqual(tuple(out_t.shape), (3, 7))


if __name__ == "__main__":
    unittest.main()
