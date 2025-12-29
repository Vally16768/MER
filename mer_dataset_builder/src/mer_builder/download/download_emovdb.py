from __future__ import annotations

import logging
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from mer_builder.utils.io import download_file, extract_archive, iter_audio_files

_OPENSLR_PAGE = "https://www.openslr.org/115/"
_CMU_ARCTIC_URL_DEFAULT = "http://www.festvox.org/cmu_arctic/cmuarctic.data"

def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_archive_ext(filename: str) -> str:
    lower = filename.lower()
    for ext in [".tar.gz", ".tgz", ".zip"]:
        if lower.endswith(ext):
            return filename[: -len(ext)]
    return Path(filename).stem


def _discover_emovdb_archives() -> dict[str, list[str]]:
    """
    Returns mapping: archive filename -> list of mirror URLs.
    """
    html = _fetch(_OPENSLR_PAGE)
    hrefs = re.findall(r'href=[\'"](?P<h>[^\'"]+)[\'"]', html, flags=re.IGNORECASE)
    cleaned: list[str] = []
    for h in hrefs:
        h2 = re.sub(r"\s+", "", h.strip())
        if not h2:
            continue
        low = h2.lower()
        if "resources/115/" not in low:
            continue
        if not (low.endswith(".tar.gz") or low.endswith(".tgz")):
            continue
        cleaned.append(urllib.parse.urljoin(_OPENSLR_PAGE, h2))

    by_name: dict[str, list[str]] = {}
    for u in cleaned:
        name = Path(urllib.parse.urlsplit(u).path).name
        if not name:
            continue
        by_name.setdefault(name, []).append(u)

    if not by_name:
        raise RuntimeError(f"Could not discover EmoV-DB archives from {_OPENSLR_PAGE}")

    mirror_pref = (os.environ.get("EMOVDB_MIRROR") or os.environ.get("OPENSLR_MIRROR") or "").strip().lower()
    default_order = ["openslr.trmal.net", "openslr.elda.org", "openslr.magicdatatech.com"]

    def _sort_urls(urls_: list[str]) -> list[str]:
        if mirror_pref:
            return sorted(urls_, key=lambda x: (mirror_pref not in x.lower(), x))
        return sorted(urls_, key=lambda x: (next((i for i, h in enumerate(default_order) if h in x.lower()), 99), x))

    return {k: _sort_urls(v) for k, v in by_name.items()}


def _download_any(urls: list[str], dst: Path, *, force: bool, desc: str) -> str:
    logger = logging.getLogger("mer_builder.download.emovdb")
    last_err: Exception | None = None
    for u in urls:
        try:
            part = dst.with_suffix(dst.suffix + ".part")
            if part.exists():
                part.unlink()
            download_file(u, dst, force=force, desc=desc)
            return u
        except Exception as e:
            last_err = e
            logger.warning("Mirror failed for %s: %s", dst.name, u)
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"No mirrors provided for {dst}")


def _validate_emovdb(dataset_dir: Path) -> None:
    prompts = dataset_dir / "cmuarctic.data"
    extracted_dir = dataset_dir / "extracted"
    if not prompts.exists():
        raise FileNotFoundError("Missing cmuarctic.data (required for transcripts).")
    if not extracted_dir.exists():
        raise FileNotFoundError("Missing extracted/ directory (did you run the downloader to extract OpenSLR archives?).")
    wavs = list(iter_audio_files(extracted_dir, exts=(".wav",)))
    if not wavs:
        raise FileNotFoundError("No WAVs found (did you extract the OpenSLR archives?).")


def download_emovdb(raw_dir: Path, *, force: bool = False) -> None:
    """
    Downloads EmoV-DB from OpenSLR (SLR115) and extracts it.
    """
    logger = logging.getLogger("mer_builder.download.emovdb")
    dataset_dir = raw_dir / "EmoV-DB"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Ensure CMU Arctic prompts file (transcripts).
    prompts_path = dataset_dir / "cmuarctic.data"
    prompts_url = (os.environ.get("CMUARCTIC_URL") or _CMU_ARCTIC_URL_DEFAULT).strip()
    if force or not prompts_path.exists():
        download_file(prompts_url, prompts_path, force=force, desc="cmuarctic.data")

    if not force:
        try:
            _validate_emovdb(dataset_dir)
            logger.info("EmoV-DB present: %s", dataset_dir)
            return
        except FileNotFoundError:
            pass

    url = os.environ.get("EMOVDB_URL")
    if not url:
        archives = _discover_emovdb_archives()
        archives_dir = dataset_dir / "archives"
        extracted_dir = dataset_dir / "extracted"
        archives_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        for name in sorted(archives.keys()):
            urls = archives[name]
            archive_path = archives_dir / name
            _download_any(urls, archive_path, force=force, desc=f"EmoV-DB {name}")
            extract_to = extracted_dir / _strip_archive_ext(name)
            extract_archive(archive_path, extract_to, force=force)
    else:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "emovdb.tar.gz"
        archive = dataset_dir / filename
        download_file(url, archive, force=force, desc="EmoV-DB archive")
        extract_archive(archive, dataset_dir / "extracted" / _strip_archive_ext(filename), force=force)

    _validate_emovdb(dataset_dir)
    logger.info("EmoV-DB ready: %s", dataset_dir)
