from __future__ import annotations

import logging
from pathlib import Path

from mer_builder.utils.io import find_dataset_dir, iter_audio_files


def _normalize_iemocap_root(candidate: Path) -> Path:
    """
    Handles common extraction layouts:
      - IEMOCAP_full_release/Session1/...
      - IEMOCAP_full_release/IEMOCAP_full_release/Session1/...
    """
    if (candidate / "Session1").is_dir():
        return candidate

    nested = candidate / "IEMOCAP_full_release"
    if (nested / "Session1").is_dir():
        return nested

    try:
        children = [p for p in candidate.iterdir() if p.is_dir() and not p.name.startswith("._")]
    except Exception:
        children = []
    if len(children) == 1 and (children[0] / "Session1").is_dir():
        return children[0]

    return candidate


def _validate_iemocap(dataset_root: Path) -> None:
    # Must have sessions.
    sessions = [dataset_root / f"Session{i}" for i in range(1, 6)]
    if not any(s.is_dir() for s in sessions):
        raise FileNotFoundError(
            "IEMOCAP not found or incomplete.\n\n"
            "Expected a directory containing Session1..Session5.\n"
        )

    # Must have evaluation and wavs (sentence-level) somewhere.
    has_eval = False
    has_wavs = False
    for s in sessions:
        if not s.is_dir():
            continue
        eval_dir = s / "dialog" / "EmoEvaluation"
        wav_dir = s / "sentences" / "wav"
        if eval_dir.is_dir():
            for p in eval_dir.glob("*.txt"):
                if p.is_file() and not p.name.startswith("._"):
                    has_eval = True
                    break
        if wav_dir.is_dir() and next(iter_audio_files(wav_dir, exts=(".wav",)), None) is not None:
            has_wavs = True
        if has_eval and has_wavs:
            break

    if not has_eval or not has_wavs:
        raise FileNotFoundError(
            "IEMOCAP not found or incomplete.\n\n"
            "Expected files like:\n"
            "  Session1/dialog/EmoEvaluation/*.txt\n"
            "  Session1/sentences/wav/**.wav\n"
        )


def download_iemocap(raw_dir: Path, *, force: bool = False) -> None:
    """
    IEMOCAP requires manual download (license/terms). This function validates that the dataset is present.

    Supported placements:
      - data/raw/IEMOCAP_full_release/
      - data/raw/IEMOCAP/
      - <repo>/IEMOCAP_full_release/   (common when the archive is extracted next to this project)
    """
    _ = force  # reserved for parity with other downloaders
    logger = logging.getLogger("mer_builder.download.iemocap")

    dataset_dir = find_dataset_dir(
        raw_dir,
        ["IEMOCAP_full_release", "IEMOCAP"],
        extra_roots=[raw_dir.parent.parent, Path.cwd()],
    )
    if dataset_dir is None:
        raise FileNotFoundError(
            "IEMOCAP not found.\n\n"
            "Manual download required:\n"
            "1) Obtain the official IEMOCAP_full_release.\n"
            "2) Place/extract into one of:\n"
            "   - data/raw/IEMOCAP_full_release/\n"
            "   - data/raw/IEMOCAP/\n"
            "   - <repo>/IEMOCAP_full_release/\n"
            "3) Re-run: python -m mer_builder download --datasets iemocap\n"
        )

    dataset_root = _normalize_iemocap_root(dataset_dir)
    _validate_iemocap(dataset_root)
    logger.info("IEMOCAP present: %s", dataset_root)

