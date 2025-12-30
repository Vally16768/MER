from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from subprocess import PIPE

from mer_builder.config import AUDIO_CHANNELS, AUDIO_CODEC, AUDIO_SAMPLE_RATE
from mer_builder.utils.io import which


def _prepend_path(dir_path: Path) -> None:
    p = str(dir_path)
    cur = os.environ.get("PATH", "")
    parts = [x for x in cur.split(os.pathsep) if x]
    if p not in parts:
        os.environ["PATH"] = p + os.pathsep + cur


def _try_autofind_ffmpeg_windows() -> bool:
    """
    Winget installs FFmpeg but doesn't always update PATH for already-open terminals.
    Try common install locations and, if found, prepend to PATH for this process.
    """
    if os.name != "nt":
        return False

    # User overrides (directory that contains ffmpeg(.exe) and ffprobe(.exe))
    user_dir = (os.environ.get("FFMPEG_BIN_DIR") or "").strip().strip('"')
    if user_dir:
        d = Path(user_dir)
        if d.exists():
            _prepend_path(d)
            return which("ffmpeg") is not None and which("ffprobe") is not None

    candidates: list[Path] = []

    # Common manual installs.
    for base in filter(None, [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]):
        candidates.extend(
            [
                Path(base) / "ffmpeg" / "bin",
                Path(base) / "FFmpeg" / "bin",
                Path(base) / "Gyan" / "FFmpeg" / "bin",
            ]
        )

    # Winget typical path:
    #   %LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-...\bin
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        root = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
        if root.exists():
            for pkg in root.glob("Gyan.FFmpeg*"):
                for bin_dir in pkg.rglob("bin"):
                    candidates.append(bin_dir)

    # Chocolatey typical path: C:\ProgramData\chocolatey\bin (exe shims)
    choco = os.environ.get("ChocolateyInstall")
    if choco:
        candidates.append(Path(choco) / "bin")
    candidates.append(Path("C:/ProgramData/chocolatey/bin"))

    # Filter to dirs that contain both binaries.
    ok_dirs: list[Path] = []
    for d in candidates:
        if not d.exists():
            continue
        if (d / "ffmpeg.exe").exists() and (d / "ffprobe.exe").exists():
            ok_dirs.append(d)

    if not ok_dirs:
        return False

    # Prefer the shortest path (usually the intended bin dir).
    ok_dirs.sort(key=lambda p: (len(str(p)), str(p).lower()))
    _prepend_path(ok_dirs[0])
    return which("ffmpeg") is not None and which("ffprobe") is not None


def ensure_ffmpeg() -> None:
    if which("ffmpeg") and which("ffprobe"):
        return
    _try_autofind_ffmpeg_windows()
    if which("ffmpeg") and which("ffprobe"):
        return
    raise RuntimeError(
        "ffmpeg/ffprobe not found on PATH.\n\n"
        "Install:\n"
        "  - Windows (winget): winget install Gyan.FFmpeg\n"
        "  - macOS (brew): brew install ffmpeg\n"
        "  - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
        "\n"
        "If it's already installed, add its bin directory to PATH, or set:\n"
        "  FFMPEG_BIN_DIR=C:\\path\\to\\ffmpeg\\bin\n"
    )


def ffprobe_duration_sec(audio_path: Path) -> float:
    ensure_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    try:
        return float(out)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to parse duration from ffprobe output: {out}") from e


def convert_to_wav_mono_16k(src_path: Path, dst_path: Path, *, force: bool = False) -> None:
    ensure_ffmpeg()
    logger = logging.getLogger("mer_builder.ffmpeg")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and not force:
        return

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_path),
        "-ac",
        str(AUDIO_CHANNELS),
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-c:a",
        AUDIO_CODEC,
        str(dst_path),
    ]
    try:
        # Capture stderr/stdout to avoid noisy per-file console spam; surface a concise message.
        r = subprocess.run(cmd, stdout=PIPE, stderr=PIPE, text=True)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "").strip().splitlines()
            short = msg[-1] if msg else "unknown_error"
            raise RuntimeError(short)
    except Exception as e:
        logger.error("ffmpeg failed for %s", src_path)
        raise RuntimeError(f"ffmpeg failed for {src_path}: {e}") from e
