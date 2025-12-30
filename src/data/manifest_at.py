from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


DEFAULT_CLASS_NAMES: list[str] = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


_MELD_DIALOG_RE = re.compile(r"dia(\d+)_utt(\d+)", flags=re.IGNORECASE)


def _speaker_prefix(speaker_id: str) -> str:
    spk = str(speaker_id or "").strip()
    if not spk:
        return ""
    return f"speaker={spk} "


def _meld_dialogue_utt(sample_id: str) -> tuple[str, int] | None:
    m = _MELD_DIALOG_RE.search(str(sample_id))
    if not m:
        return None
    dia = f"dia{int(m.group(1))}"
    utt = int(m.group(2))
    return dia, utt


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _stable_int(seed: int, sample_id: str) -> int:
    h = hashlib.blake2b(f"{seed}:{sample_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


@dataclass(frozen=True)
class ManifestRow:
    id: str
    dataset: str
    split: str
    speaker_id: str
    audio_path: str
    transcript: str
    emotion: str


def _to_row(r: dict[str, Any]) -> ManifestRow:
    return ManifestRow(
        id=str(r["id"]),
        dataset=str(r.get("dataset", "")),
        split=str(r.get("split", "")),
        speaker_id=str(r.get("speaker_id", "")),
        audio_path=str(r.get("audio_path", "")),
        transcript=str(r.get("transcript", "")),
        emotion=str(r.get("emotion", "")),
    )


class ManifestATDataset(Dataset):
    """Dataset that reads normalized WAV + transcript from the unified manifest.

    This is intended for end-to-end training with HuggingFace encoders (wav2vec2/WavLM + BERT-like text).
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        audio_root: Path,
        splits: Iterable[str],
        class_names: list[str] | None = None,
        label_map: dict[str, str] | None = None,
        drop_unknown_labels: bool = False,
        include_datasets: Iterable[str] | None = None,
        exclude_datasets: Iterable[str] | None = None,
        include_speaker_regex: str | None = None,
        exclude_speaker_regex: str | None = None,
        include_id_regex: str | None = None,
        exclude_id_regex: str | None = None,
        include_speaker_in_text: bool = False,
        meld_context_window: int = 0,
        meld_context_sep: str = " [SEP] ",
        max_audio_sec: float = 0.0,
        crop_seed: int = 42,
        deterministic_crop: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root)
        self.splits = {str(s).strip() for s in splits if str(s).strip()}
        if not self.splits:
            raise ValueError("splits must be non-empty")

        self.class_names = [str(x).strip().lower() for x in (class_names or DEFAULT_CLASS_NAMES) if str(x).strip()]
        if not self.class_names:
            raise ValueError("class_names must be non-empty")
        self.emotion_to_index: dict[str, int] = {name: i for i, name in enumerate(self.class_names)}
        self.label_map = {str(k).strip().lower(): str(v).strip().lower() for k, v in dict(label_map or {}).items()}
        self.drop_unknown_labels = bool(drop_unknown_labels)

        include = {str(x).strip().upper() for x in (include_datasets or []) if str(x).strip()}
        exclude = {str(x).strip().upper() for x in (exclude_datasets or []) if str(x).strip()}
        if include and exclude and (include & exclude):
            overlap = sorted(include & exclude)
            raise ValueError(f"include/exclude overlap: {overlap}")

        self.include = include
        self.exclude = exclude

        self._include_speaker_re = re.compile(include_speaker_regex, flags=re.IGNORECASE) if include_speaker_regex else None
        self._exclude_speaker_re = re.compile(exclude_speaker_regex, flags=re.IGNORECASE) if exclude_speaker_regex else None
        self._include_id_re = re.compile(include_id_regex, flags=re.IGNORECASE) if include_id_regex else None
        self._exclude_id_re = re.compile(exclude_id_regex, flags=re.IGNORECASE) if exclude_id_regex else None

        self.include_speaker_in_text = bool(include_speaker_in_text)
        self.meld_context_window = int(meld_context_window or 0)
        self.meld_context_sep = str(meld_context_sep)
        self.max_audio_sec = float(max_audio_sec or 0.0)
        self.crop_seed = int(crop_seed)
        self.deterministic_crop = bool(deterministic_crop)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")
        if not self.audio_root.is_dir():
            raise FileNotFoundError(f"Missing audio_root: {self.audio_root}")

        self.rows: list[ManifestRow] = []
        for r in read_jsonl(self.manifest_path):
            row = _to_row(r)
            if row.split not in self.splits:
                continue
            ds_u = row.dataset.strip().upper()
            if self.include and ds_u not in self.include:
                continue
            if self.exclude and ds_u in self.exclude:
                continue
            if self._include_speaker_re and not self._include_speaker_re.search(str(row.speaker_id)):
                continue
            if self._exclude_speaker_re and self._exclude_speaker_re.search(str(row.speaker_id)):
                continue
            if self._include_id_re and not self._include_id_re.search(str(row.id)):
                continue
            if self._exclude_id_re and self._exclude_id_re.search(str(row.id)):
                continue

            emo = row.emotion.strip().lower()
            if self.label_map:
                emo = self.label_map.get(emo, emo)
            if emo not in self.emotion_to_index:
                if self.drop_unknown_labels:
                    continue
                raise ValueError(f"Unknown emotion {row.emotion!r} for id={row.id} (class_names={self.class_names})")

            self.rows.append(row)

        if not self.rows:
            raise RuntimeError("No samples selected; check splits/include/exclude filters.")

        self.transcripts: list[str] = []
        for row in self.rows:
            text = str(row.transcript or "")
            if self.include_speaker_in_text:
                text = (_speaker_prefix(row.speaker_id) + text).strip()
            self.transcripts.append(text)

        if self.meld_context_window > 0:
            by_dialogue: dict[tuple[str, str], list[tuple[int, int]]] = {}
            for i, row in enumerate(self.rows):
                if row.dataset.strip().upper() != "MELD":
                    continue
                key = _meld_dialogue_utt(row.id)
                if key is None:
                    continue
                dia, utt = key
                by_dialogue.setdefault((row.split, dia), []).append((int(utt), int(i)))

            win = int(self.meld_context_window)
            sep = str(self.meld_context_sep)
            for _, pairs in by_dialogue.items():
                pairs.sort(key=lambda x: x[0])
                idxs = [i for _, i in pairs]
                for pos, idx in enumerate(idxs):
                    start = max(0, pos - win)
                    ctx = [self.transcripts[j] for j in idxs[start:pos] if self.transcripts[j]]
                    if ctx:
                        self.transcripts[idx] = sep.join(ctx + [self.transcripts[idx]]).strip()

        self.labels: list[int] = []
        self.datasets: list[str] = []
        for row in self.rows:
            emo = row.emotion.strip().lower()
            if self.label_map:
                emo = self.label_map.get(emo, emo)
            y = self.emotion_to_index.get(emo)
            if y is None:
                if self.drop_unknown_labels:
                    raise RuntimeError("Internal error: unknown label survived filtering")
                raise ValueError(f"Unknown emotion {row.emotion!r} for id={row.id} (class_names={self.class_names})")
            self.labels.append(int(y))
            self.datasets.append(row.dataset)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        audio_path = self.audio_root / row.audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=16000)
            sr = 16000
        wav = wav.squeeze(0).to(dtype=torch.float32, copy=False)

        if self.max_audio_sec > 0:
            max_len = int(self.max_audio_sec * 16000)
            if max_len > 0 and wav.numel() > max_len:
                if self.deterministic_crop:
                    start = _stable_int(self.crop_seed, row.id) % int(wav.numel() - max_len + 1)
                else:
                    start = 0
                wav = wav[int(start) : int(start) + int(max_len)]

        return {
            "id": row.id,
            "dataset": row.dataset,
            "speaker_id": row.speaker_id,
            "transcript": self.transcripts[int(idx)],
            "wav": wav,
            "label": torch.tensor(int(self.labels[int(idx)]), dtype=torch.long),
        }
