from __future__ import annotations

import logging
import os
from pathlib import Path

from mer_builder.utils.io import download_file, extract_archive, iter_audio_files

_DEFAULT_ZENODO_URL = (
    "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)


def _validate_ravdess(dataset_dir: Path) -> None:
    audio_dirs = [
        # Common extracted layout
        dataset_dir / "Audio_Speech_Actors_01-24",
        dataset_dir / "Audio_Speech_Actors_01-24" / "Audio_Speech_Actors_01-24",
        # Alternate layout (zip extracts Actor_01..Actor_24 at root)
        dataset_dir,
    ]
    for d in audio_dirs:
        if not d.exists():
            continue
        wavs = list(iter_audio_files(d, exts=(".wav",)))
        if wavs:
            return
    raise FileNotFoundError(
        "RAVDESS not found or incomplete.\n\n"
        "Expected a folder containing speech WAVs, e.g.:\n"
        "  data/raw/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/*.wav\n\n"
        "If you downloaded manually, place the extracted contents under:\n"
        "  data/raw/RAVDESS/\n"
    )


def download_ravdess(raw_dir: Path, *, force: bool = False) -> None:
    """
    Downloads RAVDESS speech-only audio from Zenodo (resumable) and extracts it.
    """
    logger = logging.getLogger("mer_builder.download.ravdess")
    dataset_dir = raw_dir / "RAVDESS"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        try:
            _validate_ravdess(dataset_dir)
            logger.info("RAVDESS present: %s", dataset_dir)
            return
        except FileNotFoundError:
            pass

    url = os.environ.get("RAVDESS_SPEECH_URL", _DEFAULT_ZENODO_URL)
    archive = dataset_dir / "Audio_Speech_Actors_01-24.zip"
    download_file(url, archive, force=force, desc="RAVDESS speech zip")
    extract_archive(archive, dataset_dir, force=force)
    _validate_ravdess(dataset_dir)
    logger.info("RAVDESS ready: %s", dataset_dir)
