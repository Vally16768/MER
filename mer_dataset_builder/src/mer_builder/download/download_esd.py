from __future__ import annotations

import logging
import os
import re
import urllib.parse
from pathlib import Path

from mer_builder.utils.gdrive import download_gdrive_file
from mer_builder.utils.io import download_file, env_truthy, extract_archive, iter_audio_files


_DEFAULT_ESD_GDRIVE_ID = "1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v"
_GDRIVE_FILE_RE = re.compile(r"/file/d/(?P<id>[0-9A-Za-z_-]{15,})", flags=re.IGNORECASE)


def _validate_esd(dataset_dir: Path) -> None:
    wavs = list(iter_audio_files(dataset_dir, exts=(".wav",)))
    if wavs:
        return
    raise FileNotFoundError(
        "ESD not found or incomplete.\n\n"
        "Place/extract ESD under:\n"
        "  data/raw/ESD/\n"
    )


def _gdrive_id_from_url(url: str) -> str | None:
    """
    Extracts a Google Drive file id from common share/download URLs.
    Examples:
      - https://drive.google.com/file/d/<id>/view
      - https://docs.google.com/uc?export=download&id=<id>
      - https://drive.usercontent.google.com/download?id=<id>&...
    """
    m = _GDRIVE_FILE_RE.search(url)
    if m:
        return m.group("id")
    try:
        parts = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qs(parts.query)
        if "id" in q and q["id"]:
            return q["id"][0]
    except Exception:
        return None
    return None


def download_esd(raw_dir: Path, *, force: bool = False) -> None:
    """
    ESD distribution may require manual acceptance depending on the mirror.
    Supports an optional direct-URL archive download via ESD_URL env var.
    """
    logger = logging.getLogger("mer_builder.download.esd")
    dataset_dir = raw_dir / "ESD"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        try:
            _validate_esd(dataset_dir)
            logger.info("ESD present: %s", dataset_dir)
            return
        except FileNotFoundError:
            pass

    url = os.environ.get("ESD_URL", "").strip()
    gdrive_id = os.environ.get("ESD_GDRIVE_ID", "").strip()
    if url and not gdrive_id:
        # Allow users to paste a Google Drive URL into ESD_URL; we will treat it like ESD_GDRIVE_ID.
        maybe = _gdrive_id_from_url(url)
        if maybe:
            gdrive_id = maybe
            url = ""
    if url:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "esd.tar.gz"
        archive = dataset_dir / filename
        download_file(url, archive, force=force, desc="ESD archive")
        extract_archive(archive, dataset_dir, force=force)
    elif gdrive_id:
        archive = download_gdrive_file(
            gdrive_id,
            dataset_dir / "ESD",
            force=force,
            desc="ESD (gdrive)",
        )
        extract_archive(archive, dataset_dir, force=force)
    else:
        # ESD is distributed via a public Google Drive link but requires a signed license agreement.
        # We require explicit opt-in to auto-download.
        if not env_truthy("ESD_ACCEPT_LICENSE"):
            raise FileNotFoundError(
                "ESD not present and auto-download requires license acceptance.\n\n"
                "1) Review/accept ESD terms from the official repository:\n"
                "   https://github.com/HLTSingapore/Emotional-Speech-Data\n"
                "2) Then set:\n"
                "   ESD_ACCEPT_LICENSE=1\n"
                "3) Re-run: python -m mer_builder download --datasets esd\n\n"
                "Or set ESD_URL / ESD_GDRIVE_ID to override the download source."
            )
        archive = download_gdrive_file(
            _DEFAULT_ESD_GDRIVE_ID,
            dataset_dir / "ESD",
            force=force,
            desc="ESD (official gdrive)",
        )
        extract_archive(archive, dataset_dir, force=force)

    try:
        _validate_esd(dataset_dir)
        logger.info("ESD ready: %s", dataset_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + "\n\nManual download required:\n"
            "1) Obtain ESD from the official distribution/mirror.\n"
            "2) Place/extract into: data/raw/ESD/\n"
            "3) Re-run: python -m mer_builder download --datasets esd\n\n"
            "Optional automation:\n"
            "  - Set ESD_URL to a direct archive URL, or\n"
            "  - Set ESD_GDRIVE_ID to a Google Drive file id.\n"
            "  - Or set ESD_ACCEPT_LICENSE=1 to use the official Google Drive download.\n"
        ) from e
