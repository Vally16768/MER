from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentation import apply_feature_noise, apply_modality_dropout
from ema import EMA
from metrics import compute_epoch_metrics
import time


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    if "label" not in out:
        raise ValueError("Batch missing required key: 'label'")
    return out


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: dict[str, float]


def train_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    amp: bool = False,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    scheduler: Any | None = None,
    scheduler_step_per_batch: bool = False,
    epoch_idx: int | None = None,
    ema: EMA | None = None,
    feature_noise_std_audio: float = 0.0,
    feature_noise_std_text: float = 0.0,
    modality_dropout_p: float = 0.0,
) -> EpochResult:
    model.train()
    running_loss = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    drop_a = 0
    drop_t = 0

    accum = int(grad_accum_steps or 1)
    if accum < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    it = tqdm(dataloader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    opt_steps = 0
    for step_idx, batch in enumerate(it):
        batch_dev = _move_batch_to_device(batch, device)
        labels = batch_dev["label"]

        # Train-time feature augmentation (does not affect evaluation).
        audio = apply_feature_noise(batch_dev.get("audio"), std=feature_noise_std_audio)
        text = apply_feature_noise(batch_dev.get("text"), std=feature_noise_std_text)
        audio, text, stats = apply_modality_dropout(audio=audio, text=text, p=modality_dropout_p)
        if stats is not None:
            drop_a += stats.dropped_audio
            drop_t += stats.dropped_text
        batch_dev["audio"] = audio
        batch_dev["text"] = text

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch_dev)
            loss_raw = criterion(logits, labels)
            loss = loss_raw / float(accum)

        scaler.scale(loss).backward()

        do_step = ((step_idx + 1) % accum == 0) or (step_idx + 1 == len(dataloader))
        if do_step:
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1

            if ema is not None:
                ema.update(model)

        if scheduler is not None and scheduler_step_per_batch:
            if do_step:
                if epoch_idx is None:
                    scheduler.step()
                else:
                    scheduler.step(epoch_idx + (step_idx + 1) / max(1, len(dataloader)))

        bs = int(labels.shape[0])
        running_loss += float(loss_raw.item()) * bs
        n += bs

        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy().tolist()
        trues = labels.detach().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(trues)

        it.set_postfix(loss=float(loss_raw.item()))

    avg_loss = running_loss / max(1, n)
    metrics = compute_epoch_metrics(y_true, y_pred)
    if n > 0 and modality_dropout_p > 0:
        metrics["drop_audio_rate"] = float(drop_a) / float(n)
        metrics["drop_text_rate"] = float(drop_t) / float(n)
    metrics["optimizer_steps"] = float(opt_steps)
    return EpochResult(loss=float(avg_loss), metrics=metrics)


@torch.no_grad()
def eval_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()
    running_loss = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    it = tqdm(dataloader, desc="val", leave=False)
    for batch in it:
        batch_dev = _move_batch_to_device(batch, device)
        labels = batch_dev["label"]

        logits = model(batch_dev)
        loss = criterion(logits, labels)

        bs = int(labels.shape[0])
        running_loss += float(loss.item()) * bs
        n += bs

        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        trues = labels.cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(trues)

        it.set_postfix(loss=float(loss.item()))

    avg_loss = running_loss / max(1, n)
    metrics = compute_epoch_metrics(y_true, y_pred)
    return EpochResult(loss=float(avg_loss), metrics=metrics)


def _metric_better(metric_name: str, new: float, best: float, *, min_delta: float = 0.0) -> bool:
    md = float(min_delta or 0.0)
    if metric_name.lower() == "loss":
        return new < (best - md)
    return new > (best + md)


def _metric_init(metric_name: str) -> float:
    if metric_name.lower() == "loss":
        return float("inf")
    return float("-inf")


@dataclass(frozen=True)
class TrainLoopResult:
    best_epoch: int
    best_metric: float
    stopped_early: bool
    history_train: list[dict[str, Any]]
    history_val: list[dict[str, Any]]


def train_val_loop(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    best_metric_name: str = "uar",
    patience: int | None = None,
    min_delta: float = 0.0,
    scheduler: Any | None = None,
    scheduler_metric: str | None = None,
    scheduler_step_per_batch: bool = False,
    amp: bool = False,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    ema: EMA | None = None,
    ema_use_for_eval: bool = False,
    feature_noise_std_audio: float = 0.0,
    feature_noise_std_text: float = 0.0,
    modality_dropout_p: float = 0.0,
    on_best: Callable[[int, float], None] | None = None,
    on_epoch_end: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
) -> TrainLoopResult:
    history_train: list[dict[str, Any]] = []
    history_val: list[dict[str, Any]] = []

    best_metric = _metric_init(best_metric_name)
    best_epoch = -1
    epochs_since_improve = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        wall_t0 = time.perf_counter()
        epoch_t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        epoch_t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if epoch_t0 is not None:
            torch.cuda.synchronize()
            epoch_t0.record()

        tr = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp=amp,
            grad_accum_steps=grad_accum_steps,
            grad_clip_norm=grad_clip_norm,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
            epoch_idx=epoch - 1,
            ema=ema,
            feature_noise_std_audio=feature_noise_std_audio,
            feature_noise_std_text=feature_noise_std_text,
            modality_dropout_p=modality_dropout_p,
        )

        if ema is not None and ema_use_for_eval:
            with ema.apply_to(model):
                va = eval_one_epoch(model=model, dataloader=val_loader, criterion=criterion, device=device)
        else:
            va = eval_one_epoch(model=model, dataloader=val_loader, criterion=criterion, device=device)

        if epoch_t1 is not None and epoch_t0 is not None:
            epoch_t1.record()
            torch.cuda.synchronize()
            epoch_ms = float(epoch_t0.elapsed_time(epoch_t1))
            epoch_sec = epoch_ms / 1000.0
        else:
            epoch_sec = float(time.perf_counter() - wall_t0)

        train_row = {"epoch": epoch, "loss": tr.loss, **tr.metrics, "time_epoch_sec": epoch_sec}
        val_row = {"epoch": epoch, "loss": va.loss, **va.metrics, "time_epoch_sec": epoch_sec}
        history_train.append(train_row)
        history_val.append(val_row)

        metric_key = best_metric_name.lower()
        val_metric_val = float(val_row.get(metric_key, val_row.get(best_metric_name, float("nan"))))

        if _metric_better(best_metric_name, val_metric_val, best_metric, min_delta=min_delta):
            best_metric = val_metric_val
            best_epoch = epoch
            epochs_since_improve = 0
            if on_best is not None:
                on_best(epoch, val_metric_val)
        else:
            epochs_since_improve += 1

        if scheduler is not None and not scheduler_step_per_batch:
            try:
                from torch.optim.lr_scheduler import ReduceLROnPlateau  # type: ignore

                is_plateau = isinstance(scheduler, ReduceLROnPlateau)
            except Exception:
                is_plateau = False

            if is_plateau:
                metric_key = str(scheduler_metric or best_metric_name).lower()
                metric_val = val_row.get(metric_key, None)
                if metric_val is None:
                    metric_val = val_row.get("loss", None)
                if metric_val is None:
                    metric_val = val_metric_val
                scheduler.step(float(metric_val))
            else:
                scheduler.step()

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_row, val_row)

        if patience is not None and epochs_since_improve >= patience:
            stopped_early = True
            break

    if best_epoch == -1:
        best_epoch = len(history_val)
        best_metric = float(history_val[-1].get(best_metric_name.lower(), float("nan"))) if history_val else float("nan")

    return TrainLoopResult(
        best_epoch=best_epoch,
        best_metric=float(best_metric),
        stopped_early=stopped_early,
        history_train=history_train,
        history_val=history_val,
    )
