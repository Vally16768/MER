from __future__ import annotations

from dataclasses import dataclass
import http.cookiejar
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path


_CONFIRM_RE = re.compile(r"(?:confirm=|name=\"confirm\"\s+value=\")(?P<t>[0-9A-Za-z_]+)")


@dataclass(frozen=True)
class GDriveFolderItem:
    id: str
    name: str
    kind: str
    is_folder: bool


_FOLDER_ITEM_RE = re.compile(
    r'data-id="(?P<id>[0-9A-Za-z_-]{15,})"[^>]{0,400}?(?:data-tooltip|aria-label)="(?P<label>[^"]{1,200})"',
    flags=re.IGNORECASE,
)


def _split_drive_label(label: str) -> tuple[str, str, bool]:
    # Examples we rely on:
    #   "M003 Shared folder"
    #   "audio.tar Compressed archive"
    for suffix in [" Shared folder", " Compressed archive"]:
        if label.endswith(suffix):
            name = label[: -len(suffix)].strip()
            kind = suffix.strip()
            return name, kind, "folder" in kind.lower()
    # Fallback: treat as file.
    return label.strip(), "unknown", False


def list_gdrive_folder(folder_id: str, *, timeout_sec: int = 30) -> list[GDriveFolderItem]:
    """
    Best-effort listing of a *public* Google Drive folder without auth, by scraping the folder HTML.

    This is intentionally minimal (no Drive API dependency). It works for folders whose contents are
    embedded in the initial HTML response (common for shared dataset folders).
    """
    url = f"https://drive.google.com/drive/folders/{urllib.parse.quote(folder_id)}?usp=sharing"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=timeout_sec).read().decode("utf-8", errors="replace")

    items: list[GDriveFolderItem] = []
    seen: set[str] = set()
    for m in _FOLDER_ITEM_RE.finditer(html):
        item_id = m.group("id").strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        label = m.group("label").strip()
        name, kind, is_folder = _split_drive_label(label)
        items.append(GDriveFolderItem(id=item_id, name=name, kind=kind, is_folder=is_folder))
    return items


def _tqdm(total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return tqdm(total=total, desc=desc, unit="B", unit_scale=True)


def _guess_ext(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            magic = f.read(4)
        if magic.startswith(b"PK\x03\x04"):
            return ".zip"
        if magic[:2] == b"\x1f\x8b":
            return ".tar.gz"
    except Exception:
        return None
    return None


def download_gdrive_file(
    file_id: str,
    dst_path: Path,
    *,
    force: bool = False,
    timeout_sec: int = 60,
    chunk_size: int = 1024 * 1024,
    desc: str | None = None,
) -> Path:
    """
    Downloads a public Google Drive file by id (best-effort, no auth).

    If dst_path has no extension, the downloader will try to infer archive type and add .zip or .tar.gz.
    """
    logger = logging.getLogger("mer_builder.download.gdrive")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # If user passed a directory, we'll name using detected filename (fallback: gdrive_file).
    dst_is_dir = dst_path.exists() and dst_path.is_dir()

    if dst_path.exists() and dst_path.is_file() and not force:
        logger.info("Skip download (exists): %s", dst_path)
        return dst_path

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    base_url = "https://docs.google.com/uc?export=download&id=" + urllib.parse.quote(file_id)

    def _open(url: str):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return opener.open(req, timeout=timeout_sec)

    def _download_stream(resp, out_path: Path) -> None:
        total = resp.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        pbar = _tqdm(total=total_int, desc=desc or out_path.name)
        part = out_path.with_suffix(out_path.suffix + ".part")
        with part.open("wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                if pbar:
                    pbar.update(len(chunk))
        if pbar:
            pbar.close()
        part.replace(out_path)

    def _content_disposition_filename(cd: str) -> str | None:
        # content-disposition: attachment;filename="foo.tar.gz"
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^\";]+)"?', cd, flags=re.IGNORECASE)
        if not m:
            return None
        name = m.group(1).strip()
        return Path(name).name if name else None

    logger.info("Downloading Google Drive file id=%s", file_id)
    resp = _open(base_url)
    cd = resp.headers.get("Content-Disposition", "") or ""

    if "attachment" not in cd.lower():
        html = resp.read().decode("utf-8", errors="replace")
        m = _CONFIRM_RE.search(html)
        if not m:
            raise RuntimeError(
                "Google Drive download did not return a file and no confirm token was found. "
                "The file may require permission or has exceeded download quota."
            )
        token = m.group("t")
        url2 = base_url + "&confirm=" + urllib.parse.quote(token)
        resp = _open(url2)
        cd = resp.headers.get("Content-Disposition", "") or ""

    filename = _content_disposition_filename(cd)
    final_path = dst_path
    if dst_is_dir:
        final_path = dst_path / (filename or "gdrive_file")
    elif dst_path.suffix == "" and not dst_path.name.lower().endswith(".tar.gz"):
        # If we got a filename, preserve it; else infer later.
        if filename:
            final_path = dst_path.parent / filename

    if final_path.exists() and not force:
        logger.info("Skip download (exists): %s", final_path)
        return final_path

    _download_stream(resp, final_path)

    # Infer extension if missing (helps extract_archive).
    if final_path.suffix == "" and not final_path.name.lower().endswith(".tar.gz"):
        guessed = _guess_ext(final_path)
        if guessed:
            renamed = final_path.with_name(final_path.name + guessed)
            final_path.replace(renamed)
            final_path = renamed

    logger.info("Downloaded: %s", final_path)
    return final_path
