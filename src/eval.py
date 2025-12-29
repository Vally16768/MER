from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import ClassificationMetrics, compute_classification_metrics


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    if "label" not in out:
        raise ValueError("Batch missing required key: 'label'")
    return out


@dataclass(frozen=True)
class EvalResult:
    metrics: ClassificationMetrics
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score_pos: np.ndarray | None


@torch.no_grad()
def run_evaluation(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    labels: list[int] | None = None,
) -> EvalResult:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score_pos: list[float] = []
    collect_scores = False

    it = tqdm(dataloader, desc="eval", leave=False)
    for batch in it:
        batch_dev = _move_batch_to_device(batch, device)
        labels_t = batch_dev["label"]
        logits = model(batch_dev)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels_t.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        if logits.shape[1] == 2:
            collect_scores = True
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_score_pos.extend(probs.detach().cpu().numpy().tolist())

    metrics, cm = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        labels=labels,
    )

    return EvalResult(
        metrics=metrics,
        confusion_matrix=cm,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_score_pos=np.asarray(y_score_pos) if collect_scores else None,
    )
