from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    uar: float
    per_class: dict[str, dict[str, float]]
    support: dict[str, int]


def _as_numpy_1d(x: Iterable[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x)
    return arr.reshape(-1)


def compute_classification_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    class_names: list[str] | None = None,
    labels: list[int] | None = None,
) -> tuple[ClassificationMetrics, np.ndarray]:
    """Compute sklearn-aligned metrics + confusion matrix.

    Definitions (sklearn):
      - accuracy = accuracy_score(...)
      - macro-F1 = f1_score(..., average="macro")
      - weighted-F1 (wF1) = f1_score(..., average="weighted")
      - UAR = recall_score(..., average="macro")
    """

    y_true_np = _as_numpy_1d(y_true)
    y_pred_np = _as_numpy_1d(y_pred)

    if labels is None:
        labels = sorted(set(y_true_np.tolist()) | set(y_pred_np.tolist()))

    if class_names is None:
        class_names = [str(i) for i in labels]
    if len(class_names) != len(labels):
        raise ValueError("class_names length must match labels length")

    acc = float(accuracy_score(y_true_np, y_pred_np))
    macro_f1 = float(f1_score(y_true_np, y_pred_np, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_np, y_pred_np, labels=labels, average="weighted", zero_division=0))
    uar = float(recall_score(y_true_np, y_pred_np, labels=labels, average="macro", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {}
    support_dict: dict[str, int] = {}
    for name, p, r, f, s in zip(class_names, precision, recall, f1, support, strict=False):
        per_class[name] = {"precision": float(p), "recall": float(r), "f1": float(f)}
        support_dict[name] = int(s)

    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)

    return (
        ClassificationMetrics(
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            uar=uar,
            per_class=per_class,
            support=support_dict,
        ),
        cm,
    )


def compute_epoch_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    y_true_np = _as_numpy_1d(y_true)
    y_pred_np = _as_numpy_1d(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "wf1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "uar": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
    }


def to_jsonable(metrics: ClassificationMetrics) -> dict[str, Any]:
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "uar": metrics.uar,
        "per_class": metrics.per_class,
        "support": metrics.support,
    }

