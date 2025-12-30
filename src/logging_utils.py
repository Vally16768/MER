from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)


def write_json(path: str, payload: Any) -> None:
    ensure_parent_dir(path)

    if is_dataclass(payload):
        payload = asdict(payload)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        ensure_parent_dir(self.path)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k) for k in self.fieldnames})


def get_git_commit_hash(repo_root: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def save_checkpoint(
    path: str,
    *,
    model_state: dict[str, Any],
    epoch: int | None = None,
    optimizer_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    import torch

    payload: dict[str, Any] = {"model_state": model_state}
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra is not None:
        payload["extra"] = extra

    ensure_parent_dir(path)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | None = None) -> dict[str, Any]:
    import torch

    try:
        # Prefer safe loading (no arbitrary code execution) when possible.
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def list_feature_files(dir_path: str, *, extensions: Iterable[str] | None = None) -> list[str]:
    exts = None
    if extensions is not None:
        exts = {e.lower() for e in extensions}

    files: list[str] = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if not os.path.isfile(full):
            continue
        if exts is not None and os.path.splitext(name)[1].lower() not in exts:
            continue
        files.append(full)
    files.sort()
    return files
