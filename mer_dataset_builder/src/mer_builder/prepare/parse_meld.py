from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

from mer_builder.config import EMOTIONS_7
from mer_builder.prepare.types import Sample
from mer_builder.utils.io import find_dataset_dir, iter_audio_files, relpath_posix
from mer_builder.utils.text_norm import normalize_transcript


def _find_first(dataset_root: Path, filename: str) -> Path | None:
    direct = dataset_root / filename
    if direct.exists():
        return direct
    matches = sorted(dataset_root.rglob(filename))
    return matches[0] if matches else None


def _index_clips(dataset_root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for clip in iter_audio_files(dataset_root, exts=(".wav", ".mp4")):
        idx[clip.stem.lower()].append(clip)
    return dict(idx)


def _pick_clip(candidates: list[Path], *, split_hint: str) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    hint = split_hint.lower()
    for p in candidates:
        if hint in str(p).lower():
            return p
    return candidates[0]


def _read_split_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def parse_meld(raw_dir: Path) -> list[Sample]:
    """
    Requires official split CSVs: train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
    and corresponding audio clips (MELD.Raw commonly provides .mp4 clips; some mirrors provide .wav).
    Expected filename convention: dia<Dialogue_ID>_utt<Utterance_ID>.(wav|mp4).
    """
    logger = logging.getLogger("mer_builder.prepare.parse_meld")
    dataset_root = find_dataset_dir(raw_dir, ["MELD", "meld"])
    if dataset_root is None:
        raise FileNotFoundError("MELD not found under raw_dir (expected data/raw/MELD).")

    train_csv = _find_first(dataset_root, "train_sent_emo.csv")
    dev_csv = _find_first(dataset_root, "dev_sent_emo.csv")
    test_csv = _find_first(dataset_root, "test_sent_emo.csv")
    if not train_csv or not dev_csv or not test_csv:
        raise FileNotFoundError(
            "Missing MELD split CSV(s). Expected train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv under data/raw/MELD/."
        )

    clip_index = _index_clips(dataset_root)
    if not clip_index:
        raise FileNotFoundError(f"No audio clips found under {dataset_root} (expected .wav or .mp4)")

    samples: list[Sample] = []
    for csv_path, split_name in [
        (train_csv, "meld_train"),
        (dev_csv, "meld_dev"),
        (test_csv, "testB"),
    ]:
        rows = _read_split_csv(csv_path)
        for row in rows:
            d_id = row.get("Dialogue_ID") or row.get("Dialogue Id") or row.get("DialogueID")
            u_id = row.get("Utterance_ID") or row.get("Utterance Id") or row.get("UtteranceID")
            if d_id is None or u_id is None:
                raise RuntimeError(f"Missing Dialogue_ID/Utterance_ID columns in {csv_path}")

            key = f"dia{int(d_id)}_utt{int(u_id)}"
            cands = clip_index.get(key.lower())
            if not cands:
                key2 = f"dia{int(d_id)}_utt{int(u_id):03d}"
                cands = clip_index.get(key2.lower())
            if not cands:
                raise FileNotFoundError(
                    f"Missing MELD audio for {key} (from {csv_path.name}). "
                    "Expected a clip named like dia<Dialogue_ID>_utt<Utterance_ID>.(wav|mp4)."
                )
            split_hint = "train" if split_name == "meld_train" else "dev" if split_name == "meld_dev" else "test"
            clip_path = _pick_clip(cands, split_hint=split_hint)

            speaker = (row.get("Speaker") or row.get("speaker") or "unknown").strip()
            utter = row.get("Utterance") or row.get("utterance") or ""
            emotion = (row.get("Emotion") or row.get("emotion") or "").strip().lower()
            if emotion not in EMOTIONS_7:
                raise RuntimeError(f"Unexpected MELD emotion '{emotion}' in {csv_path}")

            rel = relpath_posix(clip_path, dataset_root)
            samples.append(
                Sample(
                    dataset="MELD",
                    split=split_name,
                    speaker_id=f"meld_{speaker}",
                    raw_audio_path=clip_path,
                    source_relpath=rel,
                    transcript=normalize_transcript(utter),
                    emotion=emotion,
                    source_label=emotion,
                    notes=None,
                )
            )

    logger.info("Parsed MELD: %d samples", len(samples))
    return samples
