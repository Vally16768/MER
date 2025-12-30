from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MemmapMeta:
    count: int
    audio_dim: int | None
    text_dim: int | None
    audio_path: str | None
    text_path: str | None
    labels_path: str
    ids_path: str | None


def _read_meta(path: Path) -> MemmapMeta:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MemmapMeta(
        count=int(data["count"]),
        audio_dim=int(data["audio_dim"]) if data.get("audio_dim") is not None else None,
        text_dim=int(data["text_dim"]) if data.get("text_dim") is not None else None,
        audio_path=str(data.get("audio_path")) if data.get("audio_path") else None,
        text_path=str(data.get("text_path")) if data.get("text_path") else None,
        labels_path=str(data["labels_path"]),
        ids_path=str(data.get("ids_path")) if data.get("ids_path") else None,
    )


class MemmapATDataset(Dataset):
    """Fast A/T dataset backed by memmapped .npy arrays.

    Expected layout under `root_dir`:
      - meta.json
      - labels.npy
      - optionally: audio.npy, text.npy, ids.txt
    """

    def __init__(self, *, root_dir: Path, modalities: Iterable[str]) -> None:
        self.root_dir = Path(root_dir)
        meta_path = self.root_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing meta.json: {meta_path}")
        self.meta = _read_meta(meta_path)

        self.modalities = {str(m).strip().upper() for m in modalities if str(m).strip()}
        invalid = sorted([m for m in self.modalities if m not in {"A", "T"}])
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Allowed: ['A','T']")
        if not self.modalities:
            raise ValueError("At least one modality must be selected.")

        self.labels = np.load(self.root_dir / self.meta.labels_path, mmap_mode="r")
        if int(self.labels.shape[0]) != int(self.meta.count):
            raise ValueError("labels.npy count mismatch")

        self.audio = None
        self.text = None
        if "A" in self.modalities:
            if not self.meta.audio_path or self.meta.audio_dim is None:
                raise ValueError("meta.json missing audio_path/audio_dim but modalities include 'A'")
            self.audio = np.load(self.root_dir / self.meta.audio_path, mmap_mode="r")
        if "T" in self.modalities:
            if not self.meta.text_path or self.meta.text_dim is None:
                raise ValueError("meta.json missing text_path/text_dim but modalities include 'T'")
            self.text = np.load(self.root_dir / self.meta.text_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.meta.count)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = int(idx)
        y = int(self.labels[i])
        out: dict[str, Any] = {"label": y}

        if "A" in self.modalities:
            assert self.audio is not None
            out["audio"] = torch.from_numpy(self.audio[i]).to(dtype=torch.float32, copy=False)
        else:
            out["audio"] = None

        if "T" in self.modalities:
            assert self.text is not None
            out["text"] = torch.from_numpy(self.text[i]).to(dtype=torch.float32, copy=False)
        else:
            out["text"] = None

        return out

