from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from mer_builder.prepare.map_labels import map_emotion
from mer_builder.prepare.types import Sample
from mer_builder.utils.io import (
    find_dataset_dir,
    iter_audio_files,
    normalize_whitespace,
    read_text_maybe,
    relpath_posix,
)
from mer_builder.utils.text_norm import normalize_transcript

_EMO_KEYS = {"neutral", "angry", "happy", "sad", "surprise"}
_SPK_RE = re.compile(r"^\d+$")


def _infer_emotion(path: Path, *, dataset_root: Path) -> str | None:
    rel = path.relative_to(dataset_root)
    for part in rel.parts:
        p = part.lower()
        if p in _EMO_KEYS:
            return p
    stem = path.stem.lower()
    for key in _EMO_KEYS:
        if key in stem:
            return key
    return None


def _infer_speaker_id(path: Path, *, dataset_root: Path) -> str:
    rel = path.relative_to(dataset_root)
    for part in rel.parts:
        if _SPK_RE.match(part):
            return part
    return path.parent.name


def _try_sibling_transcript(wav_path: Path) -> str | None:
    txt_path = wav_path.with_suffix(".txt")
    if txt_path.exists():
        return normalize_whitespace(read_text_maybe(txt_path))
    return None


def _speaker_dir_and_id(wav_path: Path, *, dataset_root: Path) -> tuple[Path | None, str | None]:
    """
    ESD common layout:
      .../<speaker_id>/<Emotion>/<utt_id>.wav
      .../<speaker_id>/<speaker_id>.txt
    """
    try:
        rel = wav_path.relative_to(dataset_root)
    except Exception:
        return None, None
    parts = list(rel.parts)
    for i, part in enumerate(parts):
        if _SPK_RE.match(part):
            return dataset_root.joinpath(*parts[: i + 1]), part
    return None, None


def _parse_speaker_transcript_file(path: Path) -> dict[str, str]:
    """
    Parses per-speaker transcript files like 0001/0001.txt.
    Format is typically tab-separated:
      utt_id<TAB>text<TAB>...
    """
    tx: dict[str, str] = {}
    for line in read_text_maybe(path).splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split("\t")
        if len(parts) >= 2:
            utt_id = parts[0].strip()
            text = parts[1].strip()
        else:
            # Fallback: "utt_id|text" or "utt_id text..."
            for sep in ["|", ","]:
                if sep in s:
                    a, b = s.split(sep, 1)
                    utt_id, text = a.strip(), b.strip()
                    break
            else:
                toks = s.split(maxsplit=1)
                if len(toks) < 2:
                    continue
                utt_id, text = toks[0].strip(), toks[1].strip()
        if utt_id and text:
            tx[Path(utt_id).stem] = normalize_whitespace(text)
    return tx


def _load_global_transcripts(dataset_root: Path) -> dict[str, str]:
    """
    Best-effort parsing of dataset-provided transcript tables.
    """
    logger = logging.getLogger("mer_builder.prepare.parse_esd")
    transcripts: dict[str, str] = {}

    candidates: list[Path] = []
    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        # General transcript tables (various repos/mirrors).
        if p.suffix.lower() in {".txt", ".csv", ".tsv"} and any(k in name for k in ["trans", "script", "text"]):
            candidates.append(p)
            continue

        # ESD official release often uses per-speaker transcript files named "<speaker_id>.txt"
        # under each speaker directory (e.g., 0001/0001.txt). Include those too.
        if p.suffix.lower() == ".txt" and _SPK_RE.match(p.stem):
            candidates.append(p)
    candidates = sorted(candidates)[:50]

    for p in candidates:
        try:
            if p.suffix.lower() == ".csv":
                with p.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    cols = [c.lower() for c in (reader.fieldnames or [])]
                    key_col = next(
                        (c for c in cols if c in {"utt_id", "utterance_id", "id", "wav", "filename", "file"}),
                        None,
                    )
                    txt_col = next(
                        (c for c in cols if c in {"text", "transcript", "transcription", "sentence"}), None
                    )
                    if not key_col or not txt_col:
                        continue
                    for row in reader:
                        k = (row.get(key_col) or "").strip()
                        t = (row.get(txt_col) or "").strip()
                        if not k or not t:
                            continue
                        transcripts[Path(k).stem] = normalize_whitespace(t)
            else:
                text = read_text_maybe(p)
                for line in text.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    for sep in ["|", "\t", ","]:
                        if sep not in s:
                            continue
                        k, t = s.split(sep, 1)
                        k = k.strip()
                        t = t.strip()
                        if k and t:
                            transcripts[Path(k).stem] = normalize_whitespace(t)
                        break
        except Exception:
            logger.debug("Failed parsing transcript candidate: %s", p)
            continue

    return transcripts


def parse_esd(raw_dir: Path) -> list[Sample]:
    """
    Parses Emotional Speech Dataset (ESD).
    Keeps samples even though ESD lacks fear/disgust; notes indicate subset coverage.
    """
    logger = logging.getLogger("mer_builder.prepare.parse_esd")
    dataset_root = find_dataset_dir(raw_dir, ["ESD", "esd"])
    if dataset_root is None:
        raise FileNotFoundError("ESD not found under raw_dir (expected data/raw/ESD).")

    wavs = sorted(iter_audio_files(dataset_root, exts=(".wav",)))
    if not wavs:
        raise FileNotFoundError(f"No WAVs found under {dataset_root}")

    global_tx = _load_global_transcripts(dataset_root)
    samples: list[Sample] = []
    speaker_tx_cache: dict[str, dict[str, str]] = {}

    for wav in wavs:
        emotion_raw = _infer_emotion(wav, dataset_root=dataset_root)
        if emotion_raw is None:
            logger.debug("Skipping (emotion not found): %s", wav)
            continue

        mapped = map_emotion("esd", emotion_raw)
        if mapped.emotion is None:
            continue

        transcript = _try_sibling_transcript(wav)
        if not transcript:
            transcript = global_tx.get(wav.stem)
        if not transcript:
            spk_dir, spk_id = _speaker_dir_and_id(wav, dataset_root=dataset_root)
            if spk_dir is not None and spk_id is not None:
                if spk_id not in speaker_tx_cache:
                    spk_txt = spk_dir / f"{spk_id}.txt"
                    if spk_txt.exists():
                        speaker_tx_cache[spk_id] = _parse_speaker_transcript_file(spk_txt)
                    else:
                        speaker_tx_cache[spk_id] = {}
                transcript = speaker_tx_cache[spk_id].get(wav.stem)

        if not transcript:
            raise FileNotFoundError(
                f"Missing transcript for ESD file {wav}. "
                "Provide per-utterance .txt files or a transcript table under data/raw/ESD/."
            )

        notes_parts = []
        if mapped.notes:
            notes_parts.append(mapped.notes)
        notes_parts.append("labels_subset(no_fear,no_disgust)")
        notes = "; ".join(notes_parts) if notes_parts else None

        speaker = _infer_speaker_id(wav, dataset_root=dataset_root)
        rel = relpath_posix(wav, dataset_root)
        samples.append(
            Sample(
                dataset="ESD",
                split="__UNASSIGNED__",
                speaker_id=f"esd_{speaker}",
                raw_audio_path=wav,
                source_relpath=rel,
                transcript=normalize_transcript(transcript),
                emotion=mapped.emotion,
                source_label=emotion_raw,
                notes=notes,
            )
        )

    logger.info("Parsed ESD: %d samples", len(samples))
    return samples
