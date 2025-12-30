from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime, timezone
from typing import Any
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from config_utils import build_run_name, expand_path, normalize_modalities, resolve_run_paths  # noqa: E402
from config_utils import apply_overrides, load_yaml, save_yaml  # noqa: E402
from data.manifest_at import ManifestATDataset  # noqa: E402
from data.hf_manifest_collate import ManifestCollator  # noqa: E402
from logging_utils import CSVLogger, ensure_dir, get_git_commit_hash, load_checkpoint, save_checkpoint  # noqa: E402
from models.hf_at import HFAudioTextModel  # noqa: E402
from plotting import plot_train_val_curves  # noqa: E402
from reproducibility import set_seed  # noqa: E402
from train_loop import train_val_loop  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train end-to-end HF Audio+Text model from the unified manifest.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values as KEY=VALUE (VALUE parsed as YAML). Repeatable.",
    )
    p.add_argument("--modalities", nargs="+", choices=["A", "T"], help="Override modality set.")
    p.add_argument("--seed", type=int, help="Override seed.")
    p.add_argument("--run_name", type=str, help="Override run name.")
    return p.parse_args()


def _select_device(device_str: str) -> torch.device:
    device_str = (device_str or "cpu").strip().lower()
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: training.device is CUDA but torch.cuda.is_available() is False. Falling back to CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_repo_path(path: str) -> str:
    path = expand_path(path)
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    return os.path.normpath(path)


def _load_init_checkpoint(model: torch.nn.Module, *, path: str) -> None:
    ckpt_path = _resolve_repo_path(path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"training.init_ckpt not found: {ckpt_path}")
    ckpt_obj = load_checkpoint(ckpt_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        state = ckpt_obj["model_state"]
    elif isinstance(ckpt_obj, dict):
        state = ckpt_obj
    else:
        raise ValueError("Unsupported checkpoint format for init_ckpt")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"Loaded init_ckpt with strict=False (missing={len(missing)} unexpected={len(unexpected)}).",
            file=sys.stderr,
        )


def _build_train_sampler(
    *,
    datasets: list[str],
    labels: list[int] | None,
    sampling_cfg: dict | None,
) -> torch.utils.data.Sampler | None:
    if not sampling_cfg:
        return None
    sampler_type = str(sampling_cfg.get("type", "none")).strip().lower()
    if sampler_type in {"none", "", "null"}:
        return None

    valid = {"dataset_balanced", "weighted", "class_balanced", "dataset_class_balanced"}
    if sampler_type not in valid:
        raise ValueError(f"Unknown training.sampling.type: {sampler_type!r}. Valid: {sorted(valid)}")

    ds_u = [str(d).strip() for d in datasets]
    ds_counts = Counter(ds_u)

    dataset_weights: dict[str, float] = {}
    if sampler_type in {"dataset_balanced", "weighted", "dataset_class_balanced"}:
        if sampler_type == "dataset_balanced":
            dataset_weights = {d: 1.0 for d in ds_counts}
        else:
            raw = sampling_cfg.get("dataset_weights", {}) or {}
            dataset_weights = {str(k): float(v) for k, v in dict(raw).items()}

    weights: list[float] = []
    class_counts: Counter[int] | None = None
    class_power = float(sampling_cfg.get("class_power", 1.0) or 1.0)
    if sampler_type in {"class_balanced", "dataset_class_balanced"}:
        if labels is None:
            raise ValueError("sampling.type requires labels but labels is None")
        class_counts = Counter(int(x) for x in labels)

    for i, d in enumerate(ds_u):
        w = 1.0
        if sampler_type in {"dataset_balanced", "weighted", "dataset_class_balanced"}:
            w *= float(dataset_weights.get(d, 1.0)) / float(ds_counts[d])
        if sampler_type in {"class_balanced", "dataset_class_balanced"}:
            assert labels is not None and class_counts is not None
            y = int(labels[i])
            w *= 1.0 / float(class_counts[y] ** class_power)
        weights.append(float(w))

    num_samples = int(sampling_cfg.get("num_samples", len(ds_u)))
    replacement = bool(sampling_cfg.get("replacement", True))
    return torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=replacement)


def main() -> None:
    args = _parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.set)

    if args.modalities is not None:
        cfg["modalities"] = args.modalities

    cfg.setdefault("training", {})
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)

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

    training_cfg = cfg.get("training", {})
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    device = _select_device(str(training_cfg.get("device", "cpu")))

    output_cfg = cfg.get("output", {})
    output_root = _resolve_repo_path(str(output_cfg.get("root_dir", "outputs")))
    run_name = args.run_name or output_cfg.get("run_name") or build_run_name(dataset_name, modalities)
    run_paths = resolve_run_paths(output_root, run_name)

    ensure_dir(run_paths.run_dir)
    ensure_dir(run_paths.checkpoints_dir)
    ensure_dir(run_paths.plots_dir)

    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["modalities"] = modalities
    resolved_cfg.setdefault("output", {})
    resolved_cfg["output"]["root_dir"] = output_root
    resolved_cfg["output"]["run_name"] = run_name
    resolved_cfg["output"]["run_dir"] = run_paths.run_dir
    resolved_cfg.setdefault("meta", {})
    resolved_cfg["meta"]["resolved_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    resolved_cfg["meta"]["git_commit"] = get_git_commit_hash(REPO_ROOT)
    resolved_cfg["meta"]["config_path"] = os.path.abspath(args.config)
    resolved_cfg["meta"]["cli_overrides"] = list(args.set)
    save_yaml(resolved_cfg, run_paths.resolved_config_yaml)

    data_cfg = cfg.get("data", {})
    processed_dir = Path(_resolve_repo_path(str(data_cfg.get("processed_dir", "mer_dataset_builder/data/processed"))))
    manifest_path = Path(_resolve_repo_path(str(data_cfg.get("manifest", processed_dir / "meta_manifest.jsonl"))))
    audio_root = Path(_resolve_repo_path(str(data_cfg.get("audio_root", processed_dir / "audio"))))

    train_splits = data_cfg.get("train_splits", ["train", "meld_train"])
    val_splits = data_cfg.get("val_splits", ["val", "meld_dev"])
    if not isinstance(train_splits, list) or not isinstance(val_splits, list):
        raise ValueError("data.train_splits and data.val_splits must be lists")

    max_audio_sec = float(data_cfg.get("max_audio_sec", 10.0) or 0.0)
    text_max_tokens = int(data_cfg.get("text_max_tokens", 256) or 256)
    include_speaker_in_text = bool(data_cfg.get("include_speaker_in_text", False))
    meld_context_window = int(data_cfg.get("meld_context_window", 0) or 0)
    meld_context_sep = str(data_cfg.get("meld_context_sep", " [SEP] "))

    filter_cfg = data_cfg.get("filter", {}) or {}
    include_train = filter_cfg.get("include_datasets_train", filter_cfg.get("include_datasets"))
    exclude_train = filter_cfg.get("exclude_datasets_train", filter_cfg.get("exclude_datasets"))
    include_val = filter_cfg.get("include_datasets_val", filter_cfg.get("include_datasets"))
    exclude_val = filter_cfg.get("exclude_datasets_val", filter_cfg.get("exclude_datasets"))

    include_speaker_regex_train = filter_cfg.get("include_speaker_regex_train", filter_cfg.get("include_speaker_regex"))
    exclude_speaker_regex_train = filter_cfg.get("exclude_speaker_regex_train", filter_cfg.get("exclude_speaker_regex"))
    include_id_regex_train = filter_cfg.get("include_id_regex_train", filter_cfg.get("include_id_regex"))
    exclude_id_regex_train = filter_cfg.get("exclude_id_regex_train", filter_cfg.get("exclude_id_regex"))

    include_speaker_regex_val = filter_cfg.get("include_speaker_regex_val", filter_cfg.get("include_speaker_regex"))
    exclude_speaker_regex_val = filter_cfg.get("exclude_speaker_regex_val", filter_cfg.get("exclude_speaker_regex"))
    include_id_regex_val = filter_cfg.get("include_id_regex_val", filter_cfg.get("include_id_regex"))
    exclude_id_regex_val = filter_cfg.get("exclude_id_regex_val", filter_cfg.get("exclude_id_regex"))

    train_ds = ManifestATDataset(
        manifest_path=manifest_path,
        audio_root=audio_root,
        splits=train_splits,
        class_names=class_names,
        label_map=label_map,
        drop_unknown_labels=drop_unknown_labels,
        include_datasets=include_train,
        exclude_datasets=exclude_train,
        include_speaker_regex=include_speaker_regex_train,
        exclude_speaker_regex=exclude_speaker_regex_train,
        include_id_regex=include_id_regex_train,
        exclude_id_regex=exclude_id_regex_train,
        include_speaker_in_text=include_speaker_in_text,
        meld_context_window=meld_context_window,
        meld_context_sep=meld_context_sep,
        max_audio_sec=max_audio_sec,
        crop_seed=seed,
        deterministic_crop=True,
    )
    val_ds = ManifestATDataset(
        manifest_path=manifest_path,
        audio_root=audio_root,
        splits=val_splits,
        class_names=class_names,
        label_map=label_map,
        drop_unknown_labels=drop_unknown_labels,
        include_datasets=include_val,
        exclude_datasets=exclude_val,
        include_speaker_regex=include_speaker_regex_val,
        exclude_speaker_regex=exclude_speaker_regex_val,
        include_id_regex=include_id_regex_val,
        exclude_id_regex=exclude_id_regex_val,
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
    ).to(device)

    init_ckpt = training_cfg.get("init_ckpt", None)
    if init_ckpt:
        _load_init_checkpoint(model, path=str(init_ckpt))

    collate_manifest = ManifestCollator(
        audio_model=audio_model,
        text_model=text_model,
        modalities=modalities,
        text_max_tokens=int(text_max_tokens),
    )

    batch_size = int(training_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    loader_cfg = data_cfg.get("loader", {}) or {}
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))

    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    sampling_cfg = training_cfg.get("sampling", None)
    train_sampler = _build_train_sampler(datasets=train_ds.datasets, labels=train_ds.labels, sampling_cfg=sampling_cfg if isinstance(sampling_cfg, dict) else None)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_manifest,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_manifest,
        **loader_kwargs,
    )

    class_weights_mode = training_cfg.get("class_weights", "balanced")
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0) or 0.0)
    if class_weights_mode in ("balanced", True):
        train_labels = np.asarray(train_ds.labels, dtype=int)
        present = np.unique(train_labels)
        weights_present = compute_class_weight("balanced", classes=present, y=train_labels)
        weights = np.ones((num_classes,), dtype=np.float32)
        weights[present] = weights_present.astype(np.float32, copy=False)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_t, label_smoothing=label_smoothing)
    elif class_weights_mode in (None, False, "none"):
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError("training.class_weights must be 'balanced' or null/none/false")

    opt_name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("learning_rate", 1e-4))
    lr_audio = float(training_cfg.get("lr_audio", lr * 0.2))
    lr_text = float(training_cfg.get("lr_text", lr * 0.2))
    wd = float(training_cfg.get("weight_decay", 0.0))

    param_groups: list[dict[str, Any]] = []
    audio_params = [p for p in model.audio_encoder.parameters() if p.requires_grad]
    text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    fusion_params = [p for p in model.fusion.parameters() if p.requires_grad]
    if audio_params:
        param_groups.append({"params": audio_params, "lr": lr_audio})
    if text_params:
        param_groups.append({"params": text_params, "lr": lr_text})
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": lr})

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError("train_hf_e2e.py supports optimizer: adam/adamw")

    scheduler = None
    scheduler_step_per_batch = False
    scheduler_metric = None
    sched_cfg = training_cfg.get("scheduler", None)
    if isinstance(sched_cfg, dict):
        sched_type = str(sched_cfg.get("type", "none")).lower()
        if sched_type and sched_type != "none":
            params = {k: v for k, v in sched_cfg.items() if k != "type"}
            if sched_type == "steplr":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
            elif sched_type == "reduceonplateau":
                scheduler_metric = params.pop("metric", None)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
            else:
                raise ValueError(f"Unknown scheduler type: {sched_type!r}")

    best_metric = str(training_cfg.get("best_metric", "UAR")).lower()
    best_metric = {"wf1": "wf1", "w_f1": "wf1", "weighted_f1": "wf1", "uar": "uar", "acc": "accuracy"}.get(best_metric, best_metric)

    epochs = int(training_cfg.get("epochs", 1))
    patience = training_cfg.get("patience", None)
    patience_int = int(patience) if patience is not None else None
    min_delta = float(training_cfg.get("min_delta", 0.0) or 0.0)
    grad_clip_norm = training_cfg.get("grad_clip_norm", None)
    grad_clip_norm_f = float(grad_clip_norm) if grad_clip_norm is not None else None
    amp = bool(training_cfg.get("amp", False))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1) or 1)

    train_logger = CSVLogger(
        run_paths.metrics_train_csv,
        ["epoch", "loss", "accuracy", "wf1", "uar", "lr", "time_epoch_sec", "optimizer_steps"],
    )
    val_logger = CSVLogger(
        run_paths.metrics_val_csv,
        ["epoch", "loss", "accuracy", "wf1", "uar", "lr", "time_epoch_sec"],
    )

    def on_best(epoch: int, metric_value: float) -> None:
        save_checkpoint(
            os.path.join(run_paths.checkpoints_dir, "best.pt"),
            model_state={k: v.detach().clone() for k, v in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            extra={"best_metric": best_metric, "best_value": metric_value},
        )

    def on_epoch_end(epoch: int, train_row: dict, val_row: dict) -> None:
        lr_now = float(optimizer.param_groups[0].get("lr", lr))
        train_row["lr"] = lr_now
        val_row["lr"] = lr_now
        train_logger.log(train_row)
        val_logger.log(val_row)
        save_checkpoint(
            os.path.join(run_paths.checkpoints_dir, "last.pt"),
            model_state={k: v.detach().clone() for k, v in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            extra={"best_metric": best_metric},
        )
        t_sec = float(train_row.get("time_epoch_sec", float("nan")))
        print(
            f"[epoch {epoch:03d}] lr={lr_now:.3e} t={t_sec:.1f}s "
            f"train loss={train_row['loss']:.4f} acc={train_row['accuracy']:.4f} wf1={train_row['wf1']:.4f} uar={train_row['uar']:.4f} | "
            f"val loss={val_row['loss']:.4f} acc={val_row['accuracy']:.4f} wf1={val_row['wf1']:.4f} uar={val_row['uar']:.4f}"
        )

    result = train_val_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        best_metric_name=best_metric,
        patience=patience_int,
        min_delta=min_delta,
        scheduler=scheduler,
        scheduler_metric=str(scheduler_metric) if scheduler_metric is not None else None,
        scheduler_step_per_batch=scheduler_step_per_batch,
        amp=amp,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=grad_clip_norm_f,
        on_best=on_best,
        on_epoch_end=on_epoch_end,
    )

    plot_train_val_curves(
        metrics_train_csv=run_paths.metrics_train_csv,
        metrics_val_csv=run_paths.metrics_val_csv,
        plots_dir=run_paths.plots_dir,
    )

    print(f"Run dir: {run_paths.run_dir}")
    print(f"Best: epoch={result.best_epoch} {best_metric}={result.best_metric:.6f}")


if __name__ == "__main__":
    main()
