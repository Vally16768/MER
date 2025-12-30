from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from mer_builder.utils.gdrive import download_gdrive_file, list_gdrive_folder
from mer_builder.utils.io import atomic_write_text, download_file, ensure_dir, extract_archive, iter_audio_files


_DEFAULT_MEAD_PART0_GDRIVE_FOLDER_ID = "1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j"
_DEFAULT_MEAD_SUPP_PDF_URL = "https://wywu.github.io/projects/MEAD/support/MEAD-supp.pdf"
_AUTO_MARKER = ".mead_gdrive_complete"


def _validate_mead(dataset_dir: Path) -> None:
    audio_files = list(iter_audio_files(dataset_dir))
    if not audio_files:
        raise FileNotFoundError(
            "MEAD not found or incomplete.\n\n"
            "Place/extract MEAD Part0 under:\n"
            "  data/raw/MEAD/\n\n"
            "This builder expects speech audio files to exist somewhere under that directory.\n"
        )


def download_mead(raw_dir: Path, *, force: bool = False) -> None:
    """
    MEAD Part0 is commonly distributed via a manual link (e.g., Google Drive).
    This function supports an optional direct-URL archive download via MEAD_PART0_URL env var.
    """
    logger = logging.getLogger("mer_builder.download.mead")
    dataset_dir = raw_dir / "MEAD"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # If MEAD audio is already present, avoid network-heavy crawling unless we need to resume an
    # unfinished auto-download. A marker written after a successful full crawl indicates completion.
    has_audio = False
    if not force:
        try:
            _validate_mead(dataset_dir)
            has_audio = True
        except FileNotFoundError:
            pass

    has_sent_csv = (dataset_dir / "mead_sentences.csv").exists()
    has_supp_pdf = (dataset_dir / "MEAD-supp.pdf").exists() or any(dataset_dir.rglob("MEAD-supp.pdf"))
    is_complete = (dataset_dir / _AUTO_MARKER).exists()

    auto_dirs_present = (dataset_dir / "archives").exists() or (dataset_dir / "extracted").exists()
    if not force and has_audio and (has_supp_pdf or has_sent_csv) and (is_complete or not auto_dirs_present):
        logger.info("MEAD present: %s", dataset_dir)
        return

    url = os.environ.get("MEAD_PART0_URL", "").strip()
    gdrive_id = os.environ.get("MEAD_PART0_GDRIVE_ID", "").strip()
    gdrive_folder_id = os.environ.get("MEAD_PART0_GDRIVE_FOLDER_ID", _DEFAULT_MEAD_PART0_GDRIVE_FOLDER_ID).strip()
    if url:
        if has_audio and not force:
            logger.info("MEAD audio present locally; skipping MEAD_PART0_URL download.")
        else:
            filename = url.split("?")[0].rstrip("/").split("/")[-1] or "mead_part0.tar.gz"
            archive = dataset_dir / filename
            download_file(url, archive, force=force, desc="MEAD Part0 archive")
            extract_archive(archive, dataset_dir, force=force)
    elif gdrive_id:
        if has_audio and not force:
            logger.info("MEAD audio present locally; skipping MEAD_PART0_GDRIVE_ID download.")
        else:
            archive = download_gdrive_file(
                gdrive_id,
                dataset_dir / "MEAD_Part0",
                force=force,
                desc="MEAD Part0 (gdrive)",
            )
            extract_archive(archive, dataset_dir, force=force)
    else:
        # Auto-download from the official shared Google Drive folder. The folder contains per-actor subfolders,
        # each of which includes "audio.tar" and "video.tar". We download/extract only "audio.tar".
        logger.info("Discovering MEAD actors from Google Drive folder id=%s ...", gdrive_folder_id)
        try:
            actors = [x for x in list_gdrive_folder(gdrive_folder_id) if x.is_folder]
        except Exception as e:
            if not has_audio:
                raise FileNotFoundError(
                    "Could not list MEAD Google Drive folder contents.\n\n"
                    "Set MEAD_PART0_GDRIVE_FOLDER_ID to the shared folder id, or provide:\n"
                    "  - MEAD_PART0_URL (direct archive URL), or\n"
                    "  - MEAD_PART0_GDRIVE_ID (single archive file id)\n"
                ) from e
            logger.warning(
                "Could not list MEAD Google Drive folder contents; using existing local MEAD audio. (%s)",
                str(e).splitlines()[0] if str(e) else "error",
            )
            actors = []

        if not actors:
            if not has_audio:
                raise FileNotFoundError(
                    "Could not list MEAD Google Drive folder contents.\n\n"
                    "Set MEAD_PART0_GDRIVE_FOLDER_ID to the shared folder id, or provide:\n"
                    "  - MEAD_PART0_URL (direct archive URL), or\n"
                    "  - MEAD_PART0_GDRIVE_ID (single archive file id)\n"
                )
            logger.warning("MEAD actor listing empty; using existing local MEAD audio.")
        else:
            wanted_actors_raw = os.environ.get("MEAD_ACTORS", "").strip()
            if wanted_actors_raw:
                wanted = {
                    t.strip().upper()
                    for t in re.split(r"[,\s;]+", wanted_actors_raw)
                    if t.strip()
                }
                before = len(actors)
                actors = [a for a in actors if a.name.strip().upper() in wanted]
                logger.info("MEAD_ACTORS filter: %d -> %d actors", before, len(actors))

            archives_dir = ensure_dir(dataset_dir / "archives")
            extracted_dir = ensure_dir(dataset_dir / "extracted")

            for actor in sorted(actors, key=lambda x: x.name):
                actor_name = actor.name.strip()
                actor_folder_id = actor.id
                actor_items = list_gdrive_folder(actor_folder_id)
                audio_item = next((it for it in actor_items if not it.is_folder and it.name.lower() == "audio.tar"), None)
                if audio_item is None:
                    logger.warning("MEAD: missing audio.tar in actor folder %s", actor_name)
                    continue

                actor_archive = archives_dir / f"{actor_name}_audio.tar"
                actor_out = ensure_dir(extracted_dir / actor_name)
                download_gdrive_file(audio_item.id, actor_archive, force=force, desc=f"MEAD {actor_name} audio.tar")
                extract_archive(actor_archive, actor_out, force=force, marker_name=".extracted_audio")

            if not wanted_actors_raw:
                atomic_write_text(dataset_dir / _AUTO_MARKER, f"completed_at={time.time()}\n")

    # Transcripts: download the official supplementary PDF that contains the speech corpus lists.
    supp_url = os.environ.get("MEAD_SUPP_PDF_URL", _DEFAULT_MEAD_SUPP_PDF_URL).strip()
    supp_dst = dataset_dir / "MEAD-supp.pdf"
    if not has_supp_pdf and not has_sent_csv and (force or not supp_dst.exists()):
        try:
            download_file(supp_url, supp_dst, force=force, desc="MEAD-supp.pdf")
        except Exception as e:
            raise FileNotFoundError(
                "Failed to download MEAD-supp.pdf (required for transcript derivation).\n\n"
                "Either:\n"
                f"  - Download manually from: {supp_url}\n"
                "    and place it at: data/raw/MEAD/MEAD-supp.pdf\n"
                "Or:\n"
                "  - Set MEAD_SUPP_PDF_URL to an accessible URL and re-run download.\n"
            ) from e

    try:
        _validate_mead(dataset_dir)
        logger.info("MEAD ready: %s", dataset_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + "\n\nManual download required:\n"
            "1) Obtain MEAD Part0 (speech audio) from the official distribution.\n"
            "2) Place/extract into: data/raw/MEAD/\n"
            "3) (For transcripts) also provide scripts/metadata or create data/raw/MEAD/mead_sentences.csv.\n"
            "4) Re-run: python -m mer_builder download --datasets mead\n\n"
            "Optional automation:\n"
            "  - Set MEAD_PART0_URL to a direct archive URL, or\n"
            "  - Set MEAD_PART0_GDRIVE_ID to a Google Drive file id.\n"
            "  - Or set MEAD_PART0_GDRIVE_FOLDER_ID to the shared folder id (default is the official MEAD link).\n"
        ) from e
