from __future__ import annotations

import logging
import os
from pathlib import Path

from mer_builder.utils.gdrive import download_gdrive_file
from mer_builder.utils.io import download_file, extract_archive, iter_audio_files


_DEFAULT_MELD_RAW_URL = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
_DEFAULT_MELD_CSV_BASE_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/"


def _find_first(dataset_dir: Path, filename: str) -> Path | None:
    direct = dataset_dir / filename
    if direct.exists():
        return direct
    matches = sorted(dataset_dir.rglob(filename))
    return matches[0] if matches else None


def _validate_meld(dataset_dir: Path) -> None:
    required = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]
    found = {name: _find_first(dataset_dir, name) for name in required}
    if not all(found.values()):
        missing = [k for k, v in found.items() if v is None]
        raise FileNotFoundError(
            f"MELD CSVs not found (missing: {missing}).\n\n"
            "Place the official MELD split CSVs under:\n"
            "  data/raw/MELD/train_sent_emo.csv\n"
            "  data/raw/MELD/dev_sent_emo.csv\n"
            "  data/raw/MELD/test_sent_emo.csv\n"
        )

    # Copy them to top-level for predictability (no-op if already there).
    for name, src in found.items():
        if src is None:
            continue
        dst = dataset_dir / name
        if dst.exists():
            continue
        try:
            dst.write_bytes(src.read_bytes())
        except Exception:
            pass

    # MELD.Raw is commonly distributed as MP4 clips (audio in container); some mirrors provide WAV.
    if next(iter_audio_files(dataset_dir, exts=(".wav", ".mp4")), None) is None:
        raise FileNotFoundError(
            "MELD audio not found.\n\n"
            "Expected extracted clips under data/raw/MELD (commonly under train_splits/, dev_splits_complete/, output_repeated_splits_test/).\n"
            "If you have MELD.Raw.tar.gz, note it can contain nested archives (train.tar.gz/dev.tar.gz/test.tar.gz) that must be extracted.\n"
        )


def _ensure_meld_csvs(dataset_dir: Path, *, force: bool) -> None:
    """
    Ensures train/dev/test CSVs exist at data/raw/MELD/*.csv.

    Preferred behavior:
    1) If present anywhere under dataset_dir, copy to the dataset root.
    2) Otherwise, download from the official public GitHub repository (override with MELD_CSV_BASE_URL).
    """
    base_env = (os.environ.get("MELD_CSV_BASE_URL") or "").strip()
    if base_env:
        bases = [base_env.rstrip("/") + "/"]
    else:
        bases = [
            _DEFAULT_MELD_CSV_BASE_URL.rstrip("/") + "/",
            _DEFAULT_MELD_CSV_BASE_URL.replace("/master/", "/main/").rstrip("/") + "/",
        ]
        bases = list(dict.fromkeys(bases))  # de-dup while preserving order
    for name in ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]:
        dst = dataset_dir / name
        if dst.exists() and not force:
            continue
        src = _find_first(dataset_dir, name)
        if src and src.exists() and src != dst:
            try:
                dst.write_bytes(src.read_bytes())
            except Exception:
                pass
        if dst.exists() and not force:
            continue
        last_err: Exception | None = None
        for base in bases:
            try:
                download_file(base + name, dst, force=force, desc=f"MELD {name}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if last_err is not None and not dst.exists():
            raise RuntimeError(f"Failed downloading MELD CSV: {name} (tried: {bases})") from last_err


def _extract_split_archives(dataset_dir: Path, *, force: bool) -> None:
    """
    The UMich MELD.Raw.tar.gz contains nested archives inside MELD.Raw/:
      - train.tar.gz (contains train_splits/*.mp4 + train_sent_emo.csv)
      - dev.tar.gz   (contains dev_splits_complete/*.mp4)
      - test.tar.gz  (contains output_repeated_splits_test/*.mp4)
    """
    logger = logging.getLogger("mer_builder.download.meld")
    nested = {
        "train": _find_first(dataset_dir, "train.tar.gz"),
        "dev": _find_first(dataset_dir, "dev.tar.gz"),
        "test": _find_first(dataset_dir, "test.tar.gz"),
    }
    for split, archive in nested.items():
        if archive is None:
            continue
        marker = f".extracted_{split}_tar_gz"
        logger.info("Extracting MELD %s split archive: %s", split, archive)
        extract_archive(archive, dataset_dir, force=force, marker_name=marker)


def download_meld(raw_dir: Path, *, force: bool = False) -> None:
    """
    Downloads MELD.Raw using an official mirror and extracts it.

    Sources (in priority order):
    - MELD_RAW_URL (direct archive URL)
    - MELD_RAW_GDRIVE_ID (Google Drive file id)
    - Official UMich mirror (default)
    """
    logger = logging.getLogger("mer_builder.download.meld")
    dataset_dir = raw_dir / "MELD"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        try:
            _validate_meld(dataset_dir)
            logger.info("MELD ready: %s", dataset_dir)
            return
        except FileNotFoundError:
            # If only CSVs are missing, try fetching them before downloading the full archive.
            try:
                _ensure_meld_csvs(dataset_dir, force=False)
                _validate_meld(dataset_dir)
                logger.info("MELD ready: %s", dataset_dir)
                return
            except Exception:
                pass

    url = os.environ.get("MELD_RAW_URL", "").strip()
    gdrive_id = os.environ.get("MELD_RAW_GDRIVE_ID", "").strip()

    archive: Path | None = None
    if url:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "MELD.Raw.tar.gz"
        archive = dataset_dir / filename
        download_file(url, archive, force=force, desc="MELD.Raw archive")
    elif gdrive_id:
        archive = download_gdrive_file(
            gdrive_id,
            dataset_dir / "MELD.Raw",
            force=force,
            desc="MELD.Raw (gdrive)",
        )
    else:
        archive = dataset_dir / "MELD.Raw.tar.gz"
        download_file(_DEFAULT_MELD_RAW_URL, archive, force=force, desc="MELD.Raw archive")

    if archive:
        extract_archive(archive, dataset_dir, force=force)
        _extract_split_archives(dataset_dir, force=force)
        _ensure_meld_csvs(dataset_dir, force=force)

    try:
        _validate_meld(dataset_dir)
        logger.info("MELD ready: %s", dataset_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + "\n\nManual download required:\n"
            "1) Obtain MELD.Raw (audio) and the official CSVs from the MELD authors/repository.\n"
            "2) Place/extract into: data/raw/MELD/\n"
            "3) Re-run: python -m mer_builder download --datasets meld\n\n"
            "Optional automation:\n"
            "  - Set MELD_RAW_URL to a direct archive URL, or\n"
            "  - Set MELD_RAW_GDRIVE_ID to a Google Drive file id.\n"
        ) from e
