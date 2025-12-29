from __future__ import annotations

import logging
from pathlib import Path

from mer_builder.prepare.map_labels import map_emotion
from mer_builder.prepare.types import Sample
from mer_builder.utils.io import find_dataset_dir, iter_audio_files, relpath_posix
from mer_builder.utils.text_norm import normalize_transcript

_STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}

_EMOTION_CODE = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess(raw_dir: Path) -> list[Sample]:
    """
    Expects RAVDESS speech audio files with names: MM-VC-EE-II-SS-RR-AA.wav
    where EE is emotion code and SS is statement code.
    """
    logger = logging.getLogger("mer_builder.prepare.parse_ravdess")
    dataset_root = find_dataset_dir(raw_dir, ["RAVDESS", "ravdess"])
    if dataset_root is None:
        raise FileNotFoundError("RAVDESS not found under raw_dir (expected data/raw/RAVDESS).")

    speech_root = None
    for cand in [
        dataset_root / "Audio_Speech_Actors_01-24",
        dataset_root / "Audio_Speech_Actors_01-24" / "Audio_Speech_Actors_01-24",
        dataset_root,  # alternate layout: Actor_01..Actor_24 at root
    ]:
        if cand.exists() and any(p.is_dir() and p.name.lower().startswith("actor_") for p in cand.iterdir()):
            speech_root = cand
            break
    if speech_root is None:
        raise FileNotFoundError(
            "RAVDESS speech folder not found. Expected Audio_Speech_Actors_01-24 under data/raw/RAVDESS/."
        )

    samples: list[Sample] = []
    wavs = sorted(iter_audio_files(speech_root, exts=(".wav",)))
    if not wavs:
        raise FileNotFoundError(f"No WAVs found under {speech_root}")

    for wav in wavs:
        stem = wav.stem
        parts = stem.split("-")
        if len(parts) != 7:
            logger.warning("Skipping unexpected filename: %s", wav.name)
            continue

        modality = parts[0]
        if modality != "03":  # 03 = audio-only in RAVDESS naming
            continue

        emotion_code = parts[2]
        statement_code = parts[4]
        actor_id = parts[6]

        source_label = _EMOTION_CODE.get(emotion_code)
        if not source_label:
            logger.warning("Unknown emotion code %s in %s", emotion_code, wav.name)
            continue

        transcript = _STATEMENTS.get(statement_code)
        if not transcript:
            raise RuntimeError(f"Unknown statement code {statement_code} in {wav.name}")

        mapped = map_emotion("ravdess", source_label)
        if mapped.emotion is None:
            continue

        speaker_id = f"ravdess_{actor_id}"
        rel = relpath_posix(wav, dataset_root)
        samples.append(
            Sample(
                dataset="RAVDESS",
                split="__UNASSIGNED__",
                speaker_id=speaker_id,
                raw_audio_path=wav,
                source_relpath=rel,
                transcript=normalize_transcript(transcript),
                emotion=mapped.emotion,
                source_label=source_label,
                notes=mapped.notes,
            )
        )

    logger.info("Parsed RAVDESS: %d samples", len(samples))
    return samples
