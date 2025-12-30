from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    dataset: str
    split: str
    speaker_id: str
    raw_audio_path: Path
    source_relpath: str
    transcript: str
    emotion: str
    source_label: str
    notes: str | None = None

    id: str | None = None
    audio_relpath: str | None = None
    duration_sec: float | None = None

    def to_manifest_row(self) -> dict[str, Any]:
        if self.id is None or self.audio_relpath is None or self.duration_sec is None:
            raise ValueError("Sample missing id/audio_relpath/duration_sec")
        row: dict[str, Any] = {
            "id": self.id,
            "dataset": self.dataset,
            "split": self.split,
            "speaker_id": self.speaker_id,
            "audio_path": self.audio_relpath,
            "transcript": self.transcript,
            "emotion": self.emotion,
            "duration_sec": float(self.duration_sec),
            "source_label": self.source_label,
        }
        if self.notes:
            row["notes"] = self.notes
        return row

