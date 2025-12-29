from __future__ import annotations

import csv
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mer_builder.utils.gdrive import download_gdrive_file
from mer_builder.utils.io import download_file, extract_archive, iter_audio_files, which

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


_CREMAD_SUMMARY_TABLE_URL = (
    "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv"
)
_CREMAD_WAV_BASE_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/"
_CREMAD_BAD_FILES = {
    "FileName",
}


def _validate_cremad(dataset_dir: Path) -> None:
    wav_count = sum(1 for _ in iter_audio_files(dataset_dir, exts=(".wav",)))
    # CREMA-D audio-only release contains 7,442 clips. Allow a tiny margin for mirror inconsistencies.
    if wav_count >= 7400:
        return
    raise FileNotFoundError(
        "CREMA-D not found or incomplete.\n\n"
        "Expected WAVs under something like:\n"
        "  data/raw/CREMA-D/AudioWAV/*.wav\n\n"
        "Manual download options:\n"
        "  - Kaggle dataset (requires Kaggle API token)\n"
        "  - Official CREMA-D website (may require agreement)\n"
    )


def _download_cremad_from_github(dataset_dir: Path, *, force: bool) -> None:
    """
    Downloads the audio-only WAV files directly from the official GitHub repository.

    The repository uses Git LFS; downloading the repo ZIP is not sufficient. Instead, we use the same
    strategy as TFDS: enumerate FileName entries from summaryTable.csv and download each WAV via the
    GitHub media endpoint.
    """
    logger = logging.getLogger("mer_builder.download.cremad")
    audio_dir = dataset_dir / "AudioWAV"
    audio_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = dataset_dir / "summaryTable.csv"
    if force or not summary_csv.exists():
        download_file(_CREMAD_SUMMARY_TABLE_URL, summary_csv, force=force, desc="CREMA-D summaryTable.csv")

    wav_names: list[str] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("FileName") or "").strip().strip('"')
            if not name or name in _CREMAD_BAD_FILES:
                continue
            wav_names.append(name)

    urls: list[tuple[str, Path]] = []
    for name in wav_names:
        dst = audio_dir / f"{name}.wav"
        if dst.exists() and not force:
            continue
        urls.append((_CREMAD_WAV_BASE_URL + f"{name}.wav", dst))

    if not urls:
        logger.info("CREMA-D WAVs already present: %s", audio_dir)
        return

    logger.info("Downloading CREMA-D WAVs from GitHub (%d files)...", len(urls))

    def _dl(u: str, dst: Path) -> None:
        # Disable per-file progress/logging; use the global tqdm instead.
        download_file(u, dst, force=force, resume=False, progress=False, log=False)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("CREMAD_NUM_WORKERS", "8"))) as ex:
        futures = {ex.submit(_dl, u, dst): (u, dst) for u, dst in urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CREMA-D WAVs"):
            u, dst = futures[fut]
            try:
                fut.result()
            except Exception:
                failures.append(dst.name)

    if failures:
        raise RuntimeError(
            "Failed downloading some CREMA-D WAV files from the GitHub mirror.\n"
            f"Count={len(failures)} (example: {failures[0]}).\n\n"
            "To get a complete CREMA-D release, use an official source:\n"
            "  - Set CREMAD_URL / CREMAD_GDRIVE_ID to an archive you obtained, or\n"
            "  - Install kaggle CLI and set KAGGLE_DOWNLOAD=1 (requires Kaggle API credentials).\n"
        )


def download_cremad(raw_dir: Path, *, force: bool = False) -> None:
    """
    CREMA-D often requires either Kaggle credentials or manual download.

    Supported modes:
    - If CREMAD_URL is set, download and extract that archive.
    - If KAGGLE_CREMA_D is set (default: ejlok1/cremad) and `kaggle` CLI is available, download via Kaggle.
    - Otherwise validate manual placement under data/raw/CREMA-D/.
    """
    logger = logging.getLogger("mer_builder.download.cremad")
    dataset_dir = raw_dir / "CREMA-D"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        try:
            _validate_cremad(dataset_dir)
            logger.info("CREMA-D present: %s", dataset_dir)
            return
        except FileNotFoundError:
            pass

    url = os.environ.get("CREMAD_URL", "").strip()
    gdrive_id = os.environ.get("CREMAD_GDRIVE_ID", "").strip()
    kaggle_slug = os.environ.get("KAGGLE_CREMA_D", "ejlok1/cremad").strip()
    kaggle_wanted = os.environ.get("KAGGLE_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "y"}
    if url:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "cremad.zip"
        archive = dataset_dir / filename
        download_file(url, archive, force=force, desc="CREMA-D archive")
        extract_archive(archive, dataset_dir, force=force)
    elif gdrive_id:
        archive = download_gdrive_file(
            gdrive_id,
            dataset_dir / "CREMA-D",
            force=force,
            desc="CREMA-D (gdrive)",
        )
        extract_archive(archive, dataset_dir, force=force)
    elif which("kaggle") and kaggle_wanted:
        logger.info("Downloading CREMA-D via Kaggle (%s) ...", kaggle_slug)
        cmd = ["kaggle", "datasets", "download", "-d", kaggle_slug, "-p", str(dataset_dir), "--unzip"]
        subprocess.check_call(cmd)
    else:
        # Default: download from official GitHub repo (works without Kaggle credentials).
        _download_cremad_from_github(dataset_dir, force=force)

    try:
        _validate_cremad(dataset_dir)
        logger.info("CREMA-D ready: %s", dataset_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + "\n\nManual download required:\n"
            "1) Download CREMA-D.\n"
            "2) Place/extract into: data/raw/CREMA-D/\n"
            "3) Re-run: python -m mer_builder download --datasets cremad\n\n"
            "Optional automation:\n"
            "  - Set CREMAD_URL to a direct archive URL, or\n"
            "  - Set CREMAD_GDRIVE_ID to a Google Drive file id, or\n"
            "  - Install kaggle CLI and set KAGGLE_DOWNLOAD=1 (requires Kaggle API credentials).\n"
        ) from e
