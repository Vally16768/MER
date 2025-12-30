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
from data.manifest_at import ManifestATDataset  # noqa: E402
from data.hf_manifest_collate import ManifestCollator  # noqa: E402
from eval import run_evaluation  # noqa: E402
from logging_utils import ensure_dir, get_git_commit_hash, load_checkpoint, write_json  # noqa: E402
from metrics import to_jsonable  # noqa: E402
from models.hf_at import HFAudioTextModel  # noqa: E402
from plotting import plot_binary_roc_pr, plot_confusion_matrix  # noqa: E402
from reproducibility import set_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate end-to-end HF Audio+Text checkpoint.")
    p.add_argument("--config", required=True, help="Path to YAML config (or outputs/<run>/config_resolved.yaml).")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values as KEY=VALUE (VALUE parsed as YAML). Repeatable.",
    )
    p.add_argument("--modalities", nargs="+", choices=["A", "T"], help="Override modality set.")
    p.add_argument("--seed", type=int, help="Override seed.")
    p.add_argument("--device", type=str, help="Override device (cpu/cuda/auto).")
    p.add_argument("--output_dir", type=str, help="Override output directory.")
    return p.parse_args()


def _select_device(device_str: str) -> torch.device:
    device_str = (device_str or "cpu").strip().lower()
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_repo_path(path: str) -> str:
    path = expand_path(path)
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    return os.path.normpath(path)


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

    class_names = dataset_cfg.get("class_names")
    if class_names is None:
        class_names = None
    elif isinstance(class_names, list):
        class_names = [str(x) for x in class_names]
        if len(class_names) != num_classes:
            raise ValueError("dataset.class_names length must match dataset.num_classes")
    else:
        raise ValueError("dataset.class_names must be a list of strings (or omitted)")

    label_map = dataset_cfg.get("label_map")
    if label_map is None:
        label_map = None
    elif isinstance(label_map, dict):
        label_map = {str(k): str(v) for k, v in label_map.items()}
    else:
        raise ValueError("dataset.label_map must be a mapping (or omitted)")

    drop_unknown_labels = bool(dataset_cfg.get("drop_unknown_labels", False))

    modalities = normalize_modalities(cfg.get("modalities", ["A", "T"]))

    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    device = _select_device(str(cfg.get("training", {}).get("device", "cpu")))

    out_dir = _infer_output_dir(cfg, args, dataset_name, modalities)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))

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
    processed_dir = Path(_resolve_repo_path(str(data_cfg.get("processed_dir", "mer_dataset_builder/data/processed"))))
    manifest_path = Path(_resolve_repo_path(str(data_cfg.get("manifest", processed_dir / "meta_manifest.jsonl"))))
    audio_root = Path(_resolve_repo_path(str(data_cfg.get("audio_root", processed_dir / "audio"))))

    eval_splits = data_cfg.get("eval_splits", ["testB"])
    if not isinstance(eval_splits, list) or not eval_splits:
        raise ValueError("data.eval_splits must be a non-empty list")

    max_audio_sec = float(data_cfg.get("max_audio_sec", 10.0) or 0.0)
    text_max_tokens = int(data_cfg.get("text_max_tokens", 256) or 256)
    include_speaker_in_text = bool(data_cfg.get("include_speaker_in_text", False))
    meld_context_window = int(data_cfg.get("meld_context_window", 0) or 0)
    meld_context_sep = str(data_cfg.get("meld_context_sep", " [SEP] "))

    filter_cfg = data_cfg.get("filter", {}) or {}
    include_eval = filter_cfg.get("include_datasets_eval", filter_cfg.get("include_datasets"))
    exclude_eval = filter_cfg.get("exclude_datasets_eval", filter_cfg.get("exclude_datasets"))
    include_speaker_regex_eval = filter_cfg.get("include_speaker_regex_eval", filter_cfg.get("include_speaker_regex"))
    exclude_speaker_regex_eval = filter_cfg.get("exclude_speaker_regex_eval", filter_cfg.get("exclude_speaker_regex"))
    include_id_regex_eval = filter_cfg.get("include_id_regex_eval", filter_cfg.get("include_id_regex"))
    exclude_id_regex_eval = filter_cfg.get("exclude_id_regex_eval", filter_cfg.get("exclude_id_regex"))

    eval_ds = ManifestATDataset(
        manifest_path=manifest_path,
        audio_root=audio_root,
        splits=eval_splits,
        class_names=class_names,
        label_map=label_map,
        drop_unknown_labels=drop_unknown_labels,
        include_datasets=include_eval,
        exclude_datasets=exclude_eval,
        include_speaker_regex=include_speaker_regex_eval,
        exclude_speaker_regex=exclude_speaker_regex_eval,
        include_id_regex=include_id_regex_eval,
        exclude_id_regex=exclude_id_regex_eval,
        include_speaker_in_text=include_speaker_in_text,
        meld_context_window=meld_context_window,
        meld_context_sep=meld_context_sep,
        max_audio_sec=max_audio_sec,
        crop_seed=seed,
        deterministic_crop=True,
    )

    model_cfg = cfg.get("model", {})
    audio_model = str(model_cfg.get("audio_model", "microsoft/wavlm-base"))
    text_model = str(model_cfg.get("text_model", "roberta-base"))
    pool_audio = str(model_cfg.get("pool_audio", "mean"))
    pool_text = str(model_cfg.get("pool_text", "cls"))
    freeze_audio = bool(model_cfg.get("freeze_audio", False))
    freeze_text = bool(model_cfg.get("freeze_text", False))
    freeze_audio_feature_encoder = bool(model_cfg.get("freeze_audio_feature_encoder", False))
    gradient_checkpointing = bool(model_cfg.get("gradient_checkpointing", False))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    num_layers = int(model_cfg.get("num_layers", 4))
    num_heads = int(model_cfg.get("num_heads", 8))
    ffn_mult = int(model_cfg.get("ffn_mult", 4))
    model_moddrop = float(model_cfg.get("modality_dropout_p", 0.0) or 0.0)

    model = HFAudioTextModel(
        audio_model=audio_model,
        text_model=text_model,
        n_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        pool_audio=pool_audio,
        pool_text=pool_text,
        freeze_audio=freeze_audio,
        freeze_text=freeze_text,
        freeze_audio_feature_encoder=freeze_audio_feature_encoder,
        gradient_checkpointing=gradient_checkpointing,
        modalities=modalities,
        modality_dropout_p=model_moddrop,
    )

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

    collate_manifest = ManifestCollator(
        audio_model=audio_model,
        text_model=text_model,
        modalities=modalities,
        text_max_tokens=int(text_max_tokens),
    )

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
        batch_size=int(cfg.get("training", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_manifest,
        **loader_kwargs,
    )

    if class_names is None:
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
    plot_confusion_matrix(res.confusion_matrix, class_names=class_names, out_path=os.path.join(out_dir, "confusion_matrix.png"))

    if res.y_score_pos is not None:
        plot_binary_roc_pr(y_true=res.y_true, y_score=res.y_score_pos, plots_dir=os.path.join(out_dir, "plots"))

    print(f"Eval outputs: {out_dir}")
    print(
        f"acc={res.metrics.accuracy:.4f} macro_f1={res.metrics.macro_f1:.4f} "
        f"wf1={res.metrics.weighted_f1:.4f} uar={res.metrics.uar:.4f}"
    )


if __name__ == "__main__":
    main()
