from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected dataclass or dict, got: {type(obj)}")


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def relpath_posix(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def find_dataset_dir(raw_dir: Path, candidates: list[str], *, extra_roots: list[Path] | None = None) -> Path | None:
    """
    Returns the first matching dataset directory.

    Checks (in order):
      1) `raw_dir/<candidate>`
      2) any direct subdirectory of `raw_dir` matching a candidate (case-insensitive)
      3) repeats (1)-(2) for each directory in `extra_roots` (if provided)
    """

    def _check(base: Path) -> Path | None:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
        if not base.exists():
            return None
        wanted = {c.lower(): c for c in candidates}
        for child in base.iterdir():
            if not child.is_dir():
                continue
            key = child.name.lower()
            if key in wanted:
                return child
        return None

    p0 = _check(raw_dir)
    if p0 is not None:
        return p0

    for base in extra_roots or []:
        p = _check(base)
        if p is not None:
            return p
    return None


def iter_audio_files(root: Path, *, exts: tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a")) -> Iterator[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # macOS metadata sidecar files sometimes leak into archives and can look like media.
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() in exts:
            yield p


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.1f}{u}"
        size /= 1024.0
    return f"{n}B"


def _tqdm(total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return tqdm(total=total, desc=desc, unit="B", unit_scale=True)


def download_file(
    url: str,
    dst_path: Path,
    *,
    force: bool = False,
    resume: bool = True,
    timeout_sec: int = 60,
    chunk_size: int = 1024 * 1024,
    desc: str | None = None,
    progress: bool = True,
    log: bool = True,
) -> Path:
    """
    Resumable HTTP(S) download using stdlib urllib.
    Writes to `<dst>.part` then atomically renames on success.
    """
    logger = logging.getLogger("mer_builder.download")
    ensure_dir(dst_path.parent)

    if dst_path.exists() and not force:
        if log:
            logger.info("Skip download (exists): %s", dst_path)
        return dst_path

    part_path = dst_path.with_suffix(dst_path.suffix + ".part")
    existing = part_path.stat().st_size if part_path.exists() else 0
    headers: dict[str, str] = {}
    if resume and existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    if log:
        logger.info("Downloading %s -> %s", url, dst_path)

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = getattr(resp, "status", None)
            content_len = resp.headers.get("Content-Length")
            total = int(content_len) + existing if content_len and status == 206 else int(content_len) if content_len else None

            if existing > 0 and status != 206:
                if log:
                    logger.warning("Server did not resume; restarting download for %s", url)
                existing = 0
                if part_path.exists():
                    part_path.unlink()

            mode = "ab" if existing > 0 else "wb"
            pbar = _tqdm(total=total, desc=desc or dst_path.name) if progress else None
            if pbar and existing:
                pbar.update(existing)
            bytes_done = existing
            with part_path.open(mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_done += len(chunk)
                    if pbar:
                        pbar.update(len(chunk))
            if pbar:
                pbar.close()
            if log:
                logger.info("Downloaded %s (%s)", dst_path.name, human_bytes(bytes_done))
    except Exception:
        if log:
            logger.exception("Download failed: %s", url)
        else:
            logger.error("Download failed: %s", url)
        raise

    part_path.replace(dst_path)
    return dst_path


def extract_archive(
    archive_path: Path,
    dst_dir: Path,
    *,
    force: bool = False,
    marker_name: str = ".extracted",
) -> None:
    logger = logging.getLogger("mer_builder.download")
    ensure_dir(dst_dir)

    if "/" in marker_name or "\\" in marker_name:
        raise ValueError(f"marker_name must be a filename, got: {marker_name!r}")
    marker = dst_dir / marker_name
    if marker.exists() and not force:
        logger.info("Skip extract (marker exists): %s", marker)
        return

    if archive_path.suffix.lower() == ".zip":
        import zipfile

        logger.info("Extracting zip: %s", archive_path)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dst_dir)
    elif (
        archive_path.suffix.lower() in {".tar", ".gz", ".tgz"}
        or archive_path.name.lower().endswith((".tar.gz", ".tar.tgz"))
    ):
        import tarfile

        logger.info("Extracting tar: %s", archive_path)
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dst_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")

    atomic_write_text(marker, f"extracted_at={time.time()}\n")


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def read_text_maybe(path: Path, *, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding).strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").strip()


_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def env_truthy(key: str) -> bool:
    v = os.environ.get(key, "")
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def die(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)
