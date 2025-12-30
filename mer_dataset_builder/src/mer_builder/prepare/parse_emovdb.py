from __future__ import annotations

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

_TXT_DONE_RE = re.compile(r'^\(\s*(?P<utt>[^\s]+)\s+"(?P<text>.*)"\s*\)\s*$')
_ID4_RE = re.compile(r"(?P<id>\d{4})$")
_KNOWN_SPEAKERS = {"bea", "jenie", "josh", "sam"}


def _load_cmuarctic_prompts(path: Path) -> dict[str, str]:
    """
    Parses http://www.festvox.org/cmu_arctic/cmuarctic.data
    Example line:
      ( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
    Returns: {"0001": "Author of ...", ...}
    """
    tx: dict[str, str] = {}
    for line in read_text_maybe(path).splitlines():
        m = _TXT_DONE_RE.match(line.strip())
        if not m:
            continue
        utt = m.group("utt").strip()
        text = normalize_whitespace(m.group("text").strip())
        m2 = _ID4_RE.search(utt)
        if not m2:
            continue
        tx[m2.group("id")] = text
    return tx


def _infer_speaker_from_path(wav_path: Path, *, dataset_root: Path) -> str:
    rel = wav_path.relative_to(dataset_root)
    for part in rel.parts:
        p = part.lower().strip()
        if p.startswith(("spk-", "spk_")):
            p = p[4:]
        p0 = p.split("_", 1)[0]
        if p0 in _KNOWN_SPEAKERS:
            return p0
    return "unknown"


def _collect_emovdb_wavs(dataset_root: Path) -> list[Path]:
    """
    Preferred layout is under data/raw/EmoV-DB/extracted/**.wav.

    Some users may also have stray/duplicate WAVs in the dataset root (e.g., manually copied).
    We de-duplicate by stem, preferring the extracted/ version when both exist.
    """
    extracted = dataset_root / "extracted"
    archives = dataset_root / "archives"

    extracted_wavs = sorted(iter_audio_files(extracted, exts=(".wav",))) if extracted.exists() else []
    extracted_names = {p.name.lower() for p in extracted_wavs}

    root_wavs: list[Path] = []
    for p in iter_audio_files(dataset_root, exts=(".wav",)):
        # Skip any WAVs that are already under extracted/ or archives/ (archives typically contain tarballs).
        try:
            p.relative_to(extracted)
            continue
        except Exception:
            pass
        try:
            p.relative_to(archives)
            continue
        except Exception:
            pass
        root_wavs.append(p)

    dup = 0
    extra = 0
    extra_wavs: list[Path] = []
    for p in root_wavs:
        # EmoV-DB filenames are not unique across speakers (same sentence IDs repeat), so we cannot
        # de-dup by stem. For stray root WAVs, we only treat them as duplicates if their filename
        # exists anywhere under extracted/.
        if p.name.lower() in extracted_names:
            dup += 1
            continue
        extra_wavs.append(p)
        extra += 1

    logger = logging.getLogger("mer_builder.prepare.parse_emovdb")
    if dup:
        logger.warning("EmoV-DB: found %d duplicate WAV(s) outside extracted/; ignoring duplicates", dup)
    if extra:
        logger.warning("EmoV-DB: found %d extra WAV(s) outside extracted/; including them", extra)

    return sorted(extracted_wavs + extra_wavs)


def parse_emovdb(
    raw_dir: Path,
    *,
    emovdb_sleepy: str = "drop",
) -> tuple[list[Sample], list[dict[str, str]]]:
    """
    Parses EmoV-DB (OpenSLR SLR115).
    Transcript source: CMU Arctic prompts (cmuarctic.data).
    Drops samples whose emotion label cannot be cleanly mapped into the 7-class space and logs them.
    """
    logger = logging.getLogger("mer_builder.prepare.parse_emovdb")
    dataset_root = find_dataset_dir(raw_dir, ["EmoV-DB", "emovdb", "EmoV_DB"])
    if dataset_root is None:
        raise FileNotFoundError("EmoV-DB not found under raw_dir (expected data/raw/EmoV-DB).")

    prompts_path = dataset_root / "cmuarctic.data"
    if not prompts_path.exists():
        raise FileNotFoundError(
            "Missing CMU Arctic prompts file for EmoV-DB.\n\n"
            "Expected: data/raw/EmoV-DB/cmuarctic.data\n"
            "Re-run: python -m mer_builder download --datasets emovdb"
        )
    transcripts = _load_cmuarctic_prompts(prompts_path)
    if not transcripts:
        raise FileNotFoundError(f"Failed to parse CMU Arctic prompts from: {prompts_path}")

    wavs = _collect_emovdb_wavs(dataset_root)
    if not wavs:
        raise FileNotFoundError(f"No WAVs found under {dataset_root}")

    samples: list[Sample] = []
    dropped: list[dict[str, str]] = []

    for wav in wavs:
        stem = wav.stem
        parts = stem.split("_")
        if len(parts) < 2:
            dropped.append({"path": str(wav), "reason": "unexpected_filename"})
            continue
        emotion_raw = parts[0].lower()
        m_id = _ID4_RE.search(stem)
        if not m_id:
            dropped.append({"path": str(wav), "reason": "missing_sentence_id"})
            continue
        sent_id4 = m_id.group("id")

        mapped = map_emotion("emovdb", emotion_raw, emovdb_sleepy=emovdb_sleepy)  # type: ignore[arg-type]
        if mapped.emotion is None:
            dropped.append({"path": str(wav), "reason": mapped.notes or "unmapped_emotion"})
            continue

        transcript = transcripts.get(sent_id4)
        if not transcript:
            dropped.append({"path": str(wav), "reason": "missing_transcript"})
            continue

        speaker = _infer_speaker_from_path(wav, dataset_root=dataset_root)
        rel = relpath_posix(wav, dataset_root)
        samples.append(
            Sample(
                dataset="EmoV-DB",
                split="__UNASSIGNED__",
                speaker_id=f"emovdb_{speaker}",
                raw_audio_path=wav,
                source_relpath=rel,
                transcript=normalize_transcript(transcript),
                emotion=mapped.emotion,
                source_label=emotion_raw,
                notes=mapped.notes,
            )
        )

    logger.info("Parsed EmoV-DB: %d samples (dropped %d)", len(samples), len(dropped))
    return samples, dropped
