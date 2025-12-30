from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from config_utils import build_run_name, expand_path, normalize_modalities, save_yaml  # noqa: E402
from config_utils import apply_overrides, load_yaml  # noqa: E402
from data.dataset import AVTDictDataset, collate_avt_dict  # noqa: E402
from eval import run_evaluation  # noqa: E402
from logging_utils import ensure_dir, get_git_commit_hash, list_feature_files, load_checkpoint, write_json  # noqa: E402
from metrics import to_jsonable  # noqa: E402
from models.flexible_at import FlexibleATModel  # noqa: E402
from models.robust_at import RobustATModel  # noqa: E402
from plotting import plot_binary_roc_pr, plot_confusion_matrix  # noqa: E402
from reproducibility import set_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MER checkpoint (no training required).")
    p.add_argument("--config", required=True, help="Path to YAML config (e.g. outputs/<run>/config_resolved.yaml).")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values as KEY=VALUE (VALUE parsed as YAML). Repeatable.",
    )
    p.add_argument("--modalities", nargs="+", choices=["A", "T"], help="Override modality set.")
    p.add_argument("--seed", type=int, help="Override seed.")
    p.add_argument("--device", type=str, help="Override device (cpu/cuda).")
    p.add_argument("--output_dir", type=str, help="Override output directory (default: inferred from config).")
    return p.parse_args()


def _select_device(device_str: str) -> torch.device:
    device_str = (device_str or "cpu").strip().lower()
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: training.device is CUDA but torch.cuda.is_available() is False. "
            "Falling back to CPU. Install a CUDA-enabled PyTorch build to use the GPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")

    return torch.device(device_str)


def _resolve_repo_path(path: str) -> str:
    path = expand_path(path)
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    return os.path.normpath(path)


def _dataset_from_feature_path(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "UNKNOWN"


def _filter_feature_files(
    files: list[str],
    *,
    include_datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
) -> list[str]:
    include = {str(x).strip().upper() for x in (include_datasets or []) if str(x).strip()}
    exclude = {str(x).strip().upper() for x in (exclude_datasets or []) if str(x).strip()}
    if include and exclude and (include & exclude):
        overlap = sorted(include & exclude)
        raise ValueError(f"data.filter include/exclude overlap: {overlap}")
    if not include and not exclude:
        return files
    out: list[str] = []
    for p in files:
        ds = _dataset_from_feature_path(p).strip().upper()
        if include and ds not in include:
            continue
        if exclude and ds in exclude:
            continue
        out.append(p)
    return out


def _infer_output_dir(cfg: dict, args: argparse.Namespace, dataset_name: str, modalities: list[str]) -> str:
    if args.output_dir:
        return _resolve_repo_path(args.output_dir)

    out_cfg = cfg.get("output", {})
    run_dir = out_cfg.get("run_dir")
    if run_dir:
        return _resolve_repo_path(str(run_dir))

    config_path = os.path.abspath(args.config)
    if os.path.basename(config_path).lower() == "config_resolved.yaml":
        return os.path.dirname(config_path)

    root_dir = _resolve_repo_path(str(out_cfg.get("root_dir", "outputs")))
    run_name = out_cfg.get("run_name") or build_run_name(dataset_name, modalities)
    return os.path.join(root_dir, run_name)


def _write_confusion_csv(cm, class_names: list[str], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for name, row in zip(class_names, cm.tolist(), strict=False):
            w.writerow([name] + [int(x) for x in row])


def main() -> None:
    args = _parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg.setdefault("training", {})

    if args.modalities is not None:
        cfg["modalities"] = args.modalities
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    if args.device is not None:
        cfg["training"]["device"] = str(args.device)

    dataset_cfg = cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "dataset"))
    num_classes = int(dataset_cfg.get("num_classes", 0))
    if num_classes <= 1:
        raise ValueError("Config must set dataset.num_classes (>=2).")

    modalities = normalize_modalities(cfg.get("modalities", []))

    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    device = _select_device(str(cfg.get("training", {}).get("device", "cpu")))

    out_dir = _infer_output_dir(cfg, args, dataset_name, modalities)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))

    # Persist a resolved config snapshot next to eval outputs (useful when config was not from train.py).
    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["modalities"] = modalities
    resolved_cfg.setdefault("output", {})
    resolved_cfg["output"]["run_dir"] = out_dir
    resolved_cfg.setdefault("meta", {})
    resolved_cfg["meta"]["resolved_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    resolved_cfg["meta"]["git_commit"] = get_git_commit_hash(REPO_ROOT)
    resolved_cfg["meta"]["config_path"] = os.path.abspath(args.config)
    resolved_cfg["meta"]["cli_overrides"] = list(args.set)
    save_yaml(resolved_cfg, os.path.join(out_dir, "config_resolved.yaml"))

    data_cfg = cfg.get("data", {})
    eval_dir_raw = str(data_cfg.get("eval_dir") or "").strip()
    val_dir_raw = str(data_cfg.get("val_dir") or "").strip()

    eval_dir = _resolve_repo_path(eval_dir_raw) if eval_dir_raw else ""
    val_dir = _resolve_repo_path(val_dir_raw) if val_dir_raw else ""

    if eval_dir and os.path.isdir(eval_dir):
        eval_dir_final = eval_dir
    elif val_dir and os.path.isdir(val_dir):
        eval_dir_final = val_dir
    else:
        raise FileNotFoundError(
            "Invalid evaluation data directory. "
            f"data.eval_dir={eval_dir!r} data.val_dir={val_dir!r} (both missing or non-existent)"
        )

    backend = str(data_cfg.get("backend", "pickle")).strip().lower()
    if backend not in {"pickle", "memmap"}:
        raise ValueError("data.backend must be 'pickle' or 'memmap'")

    filter_cfg = data_cfg.get("filter", {}) or {}
    if backend == "pickle":
        feature_exts = data_cfg.get("extensions", [".pkl", ".pickle"])
        eval_files = list_feature_files(eval_dir_final, extensions=feature_exts)
        if not eval_files:
            raise FileNotFoundError(f"No feature files found in eval_dir={eval_dir_final!r}")

        eval_files = _filter_feature_files(
            eval_files,
            include_datasets=filter_cfg.get("include_datasets_eval", filter_cfg.get("include_datasets")),
            exclude_datasets=filter_cfg.get("exclude_datasets_eval", filter_cfg.get("exclude_datasets")),
        )
        if not eval_files:
            raise FileNotFoundError("After filtering, no eval files remain. Check data.filter.include_datasets_*.")

        eval_ds = AVTDictDataset(eval_files, modalities=modalities)
    else:
        from data.memmap_at import MemmapATDataset

        if filter_cfg:
            raise ValueError("data.filter is not supported for data.backend=memmap (build separate memmaps per subset).")

        eval_ds = MemmapATDataset(root_dir=Path(eval_dir_final), modalities=modalities)
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    loader_cfg = data_cfg.get("loader", {}) or {}
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=int(cfg.get("training", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_avt_dict,
        **loader_kwargs,
    )

    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "flexible_at")).strip().lower()
    if model_type in {"flexible_at", "gfl", "baseline"}:
        model = FlexibleATModel(
            input_dim_audio=int(model_cfg.get("input_dim_a", 512)),
            input_dim_text=int(model_cfg.get("input_dim_t", 768)),
            gated_dim=int(model_cfg.get("gated_dim", 128)),
            n_classes=num_classes,
            drop=float(model_cfg.get("dropout", 0.0)),
            modalities=modalities,
        )
    elif model_type in {"robust_at", "robust"}:
        hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("gated_dim", 256)))
        model = RobustATModel(
            input_dim_audio=int(model_cfg.get("input_dim_a", 512)),
            input_dim_text=int(model_cfg.get("input_dim_t", 768)),
            hidden_dim=hidden_dim,
            n_classes=num_classes,
            num_layers=int(model_cfg.get("num_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_mult=int(model_cfg.get("ffn_mult", 4)),
            modalities=modalities,
        )
    else:
        raise ValueError(f"Unknown model.type: {model_type!r}")

    ckpt_obj = load_checkpoint(args.ckpt, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        state = ckpt_obj["model_state"]
    elif isinstance(ckpt_obj, dict):
        state = ckpt_obj
    else:
        raise ValueError("Unsupported checkpoint format")

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    model = model.to(device)

    class_names = dataset_cfg.get("class_names")
    if isinstance(class_names, list) and len(class_names) == num_classes:
        class_names = [str(x) for x in class_names]
    else:
        class_names = [str(i) for i in range(num_classes)]

    labels = list(range(num_classes))
    res = run_evaluation(model=model, dataloader=eval_loader, device=device, class_names=class_names, labels=labels)

    metrics_json = {
        "dataset": dataset_name,
        "modalities": modalities,
        "num_classes": num_classes,
        "num_samples": int(res.y_true.shape[0]),
        "metrics": to_jsonable(res.metrics),
    }
    write_json(os.path.join(out_dir, "metrics_eval.json"), metrics_json)

    _write_confusion_csv(res.confusion_matrix, class_names, os.path.join(out_dir, "confusion_matrix.csv"))
    plot_confusion_matrix(
        res.confusion_matrix,
        class_names=class_names,
        out_path=os.path.join(out_dir, "confusion_matrix.png"),
    )

    if res.y_score_pos is not None:
        plot_binary_roc_pr(y_true=res.y_true, y_score=res.y_score_pos, plots_dir=os.path.join(out_dir, "plots"))

    print(f"Eval outputs: {out_dir}")
    print(
        f"acc={res.metrics.accuracy:.4f} macro_f1={res.metrics.macro_f1:.4f} "
        f"wf1={res.metrics.weighted_f1:.4f} uar={res.metrics.uar:.4f}"
    )


if __name__ == "__main__":
    main()
