from __future__ import annotations

import logging
import re
from pathlib import Path

from mer_builder.prepare.map_labels import map_emotion
from mer_builder.prepare.types import Sample
from mer_builder.utils.io import find_dataset_dir, iter_audio_files, relpath_posix
from mer_builder.utils.text_norm import normalize_transcript

_SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "MTI": "Maybe tomorrow it will be cold.",
    "TAI": "The airplane is almost full.",
}

_FILENAME_RE = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>[A-Z]{3})_(?P<rest>.+)$"
)


def parse_cremad(raw_dir: Path) -> list[Sample]:
    """
    Expects CREMA-D WAVs with names like: 1001_DFA_ANG_XX.wav
    """
    logger = logging.getLogger("mer_builder.prepare.parse_cremad")
    dataset_root = find_dataset_dir(raw_dir, ["CREMA-D", "CREMAD", "cremad", "crema-d"])
    if dataset_root is None:
        raise FileNotFoundError("CREMA-D not found under raw_dir (expected data/raw/CREMA-D).")

    wavs = sorted(iter_audio_files(dataset_root, exts=(".wav",)))
    if not wavs:
        raise FileNotFoundError(f"No WAVs found under {dataset_root}")

    samples: list[Sample] = []
    for wav in wavs:
        m = _FILENAME_RE.match(wav.stem)
        if not m:
            logger.debug("Skipping unexpected filename: %s", wav.name)
            continue

        actor = m.group("actor")
        sentence_code = m.group("sentence")
        emotion_code = m.group("emotion")

        sentence_notes = None
        if sentence_code not in _SENTENCES:
            sentence_notes = f"unknown_sentence_code={sentence_code}"
            transcript = f"[CREMA-D:{sentence_code}]"
        else:
            transcript = normalize_transcript(_SENTENCES[sentence_code])

        mapped = map_emotion("cremad", emotion_code)
        if mapped.emotion is None:
            logger.debug("Dropping unmapped label %s (%s)", emotion_code, wav.name)
            continue
        notes = mapped.notes
        if sentence_notes:
            notes = "; ".join([p for p in [notes, sentence_notes] if p])

        speaker_id = f"cremad_{actor}"
        rel = relpath_posix(wav, dataset_root)
        samples.append(
            Sample(
                dataset="CREMA-D",
                split="__UNASSIGNED__",
                speaker_id=speaker_id,
                raw_audio_path=wav,
                source_relpath=rel,
                transcript=transcript,
                emotion=mapped.emotion,
                source_label=emotion_code,
                notes=notes,
            )
        )

    logger.info("Parsed CREMA-D: %d samples", len(samples))
    return samples
