from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from models.flexible_at import FlexibleATModel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize FlexibleATModel (summary + optional graph).")
    p.add_argument("--input_dim_a", type=int, default=160)
    p.add_argument("--input_dim_t", type=int, default=768)
    p.add_argument("--gated_dim", type=int, default=128)
    p.add_argument("--n_classes", type=int, default=7)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--modalities", nargs="+", default=["A", "T"], choices=["A", "T"])
    p.add_argument("--out_dir", type=Path, default=Path("outputs/model_viz"))
    p.add_argument("--graph", action="store_true", help="Try to render a graph (requires torchviz + Graphviz).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FlexibleATModel(
        input_dim_audio=int(args.input_dim_a),
        input_dim_text=int(args.input_dim_t),
        gated_dim=int(args.gated_dim),
        n_classes=int(args.n_classes),
        drop=float(args.dropout),
        modalities=args.modalities,
    ).eval()

    (out_dir / "model_repr.txt").write_text(repr(model) + "\n", encoding="utf-8")

    try:
        from torchinfo import summary  # type: ignore

        batch = {
            "audio": torch.randn(2, int(args.input_dim_a)) if "A" in args.modalities else None,
            "text": torch.randn(2, int(args.input_dim_t)) if "T" in args.modalities else None,
        }
        s = summary(model, input_data=(batch,), verbose=0)
        (out_dir / "model_summary.txt").write_text(str(s) + "\n", encoding="utf-8")
    except ModuleNotFoundError:
        (out_dir / "model_summary.txt").write_text(
            "torchinfo not installed. Install with: pip install torchinfo\n",
            encoding="utf-8",
        )

    if args.graph:
        try:
            from torchviz import make_dot  # type: ignore

            batch = {
                "audio": torch.randn(1, int(args.input_dim_a)) if "A" in args.modalities else None,
                "text": torch.randn(1, int(args.input_dim_t)) if "T" in args.modalities else None,
            }
            y = model(batch)
            dot = make_dot(y.sum(), params=dict(model.named_parameters()))
            dot.render(str(out_dir / "flexible_at_graph"), format="png", cleanup=True)
        except ModuleNotFoundError:
            (out_dir / "graph_error.txt").write_text(
                "torchviz not installed. Install with: pip install torchviz\nAlso install Graphviz on PATH.\n",
                encoding="utf-8",
            )
        except Exception as exc:
            (out_dir / "graph_error.txt").write_text(f"Graph render failed: {exc}\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
