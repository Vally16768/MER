from __future__ import annotations

import csv
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        val = row.get(key, None)
        if val is None or val == "":
            out.append(float("nan"))
        else:
            out.append(float(val))
    return out


def _to_int_list(rows: list[dict[str, Any]], key: str) -> list[int]:
    out: list[int] = []
    for row in rows:
        out.append(int(float(row[key])))
    return out


def plot_train_val_curves(
    *,
    metrics_train_csv: str,
    metrics_val_csv: str,
    plots_dir: str,
) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    train_rows = _read_csv(metrics_train_csv)
    val_rows = _read_csv(metrics_val_csv)

    epochs = _to_int_list(train_rows, "epoch")

    def _plot_pair(key: str, filename: str, ylabel: str) -> None:
        train_vals = _to_float_list(train_rows, key)
        val_vals = _to_float_list(val_rows, key)

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, train_vals, label=f"train_{key}")
        plt.plot(epochs, val_vals, label=f"val_{key}")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=200)
        plt.close()

    _plot_pair("loss", "loss.png", "loss")
    _plot_pair("accuracy", "accuracy.png", "accuracy")
    _plot_pair("wf1", "wf1.png", "weighted F1")
    _plot_pair("uar", "uar.png", "UAR (macro recall)")


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: list[str],
    out_path: str,
    title: str = "Confusion matrix",
) -> None:
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got shape={cm.shape}")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_binary_roc_pr(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    plots_dir: str,
) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_score, ax=ax)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "roc.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_score, ax=ax)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pr.png"), dpi=200)
    plt.close(fig)

