from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


class ConfigError(ValueError):
    pass


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency `pyyaml`. Install it (e.g. `pip install pyyaml`) to use YAML configs."
        ) from exc
    return yaml


def load_yaml(path: str) -> dict[str, Any]:
    yaml = _require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping/dict, got: {type(data)}")
    return data


def save_yaml(data: dict[str, Any], path: str) -> None:
    yaml = _require_yaml()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def parse_override_value(raw: str) -> Any:
    yaml = _require_yaml()
    return yaml.safe_load(raw)


def set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts:
        raise ConfigError("Empty override key")
    node: dict[str, Any] = cfg
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ConfigError(f"Cannot set {dotted_key}: {part} is not a dict")
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must be KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Invalid override key in: {item!r}")
        value = parse_override_value(raw_value)
        set_by_dotted_key(cfg, key, value)
    return cfg


CANONICAL_MODALITY_ORDER = ("A", "T")


def normalize_modalities(modalities: Iterable[str]) -> list[str]:
    mods = [m.strip().upper() for m in modalities if str(m).strip()]
    invalid = sorted({m for m in mods if m not in CANONICAL_MODALITY_ORDER})
    if invalid:
        raise ConfigError(f"Invalid modalities: {invalid}. Allowed: {list(CANONICAL_MODALITY_ORDER)}")
    ordered = [m for m in CANONICAL_MODALITY_ORDER if m in set(mods)]
    if not ordered:
        raise ConfigError("At least one modality must be selected.")
    return ordered


def modalities_tag(modalities: Iterable[str]) -> str:
    return "".join(normalize_modalities(modalities))


def utc_timestamp(compact: bool = True) -> str:
    fmt = "%Y%m%d_%H%M%S" if compact else "%Y-%m-%d_%H-%M-%S"
    return datetime.now(tz=timezone.utc).strftime(fmt)


def expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


@dataclass(frozen=True)
class RunPaths:
    run_dir: str
    checkpoints_dir: str
    plots_dir: str
    metrics_train_csv: str
    metrics_val_csv: str
    metrics_eval_json: str
    confusion_csv: str
    confusion_png: str
    resolved_config_yaml: str


def build_run_name(dataset: str, modalities: Iterable[str], timestamp: str | None = None) -> str:
    dataset = str(dataset).strip() or "dataset"
    ts = timestamp or utc_timestamp(compact=True)
    return f"{dataset}_{modalities_tag(modalities)}_{ts}"


def resolve_run_paths(root_dir: str, run_name: str) -> RunPaths:
    run_dir = os.path.join(root_dir, run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    plots_dir = os.path.join(run_dir, "plots")
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        plots_dir=plots_dir,
        metrics_train_csv=os.path.join(run_dir, "metrics_train.csv"),
        metrics_val_csv=os.path.join(run_dir, "metrics_val.csv"),
        metrics_eval_json=os.path.join(run_dir, "metrics_eval.json"),
        confusion_csv=os.path.join(run_dir, "confusion_matrix.csv"),
        confusion_png=os.path.join(run_dir, "confusion_matrix.png"),
        resolved_config_yaml=os.path.join(run_dir, "config_resolved.yaml"),
    )
