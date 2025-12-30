from __future__ import annotations

import csv
import logging
import re
import unicodedata
from pathlib import Path

from mer_builder.prepare.map_labels import map_emotion
from mer_builder.prepare.types import Sample
from mer_builder.utils.io import (
    find_dataset_dir,
    iter_audio_files,
    normalize_whitespace,
    read_text_maybe,
    relpath_posix,
)
from mer_builder.utils.text_norm import normalize_transcript

_SPEAKER_RE = re.compile(r"[MW]\d{3}", flags=re.IGNORECASE)
_MEAD_CORPUS_SECTION_RE = re.compile(r"^\s*6\s+Speech Corpus of Mead\s*$", flags=re.IGNORECASE)
_MEAD_LIST_NUM_RE = re.compile(r"^\s*(?P<n>\d+)\.(?:\s+)?(?P<text>\S.*)\s*$")

_EMO_CANDIDATES = {
    "anger": {"anger", "angry"},
    "disgust": {"disgust", "disgusted"},
    "fear": {"fear", "fearful"},
    "happy": {"happy"},
    "neutral": {"neutral"},
    "sad": {"sad", "sadness"},
    "surprise": {"surprise", "surprised"},
    "contempt": {"contempt"},
}


def _load_sentence_map_from_csv(path: Path) -> dict[tuple[str | None, int], str]:
    """
    Supports:
      - 2-column CSV: sentence_id,text (header optional) -> applies globally
      - 3-column CSV with header: emotion,sentence_id,text -> per-emotion overrides
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}
    header = [c.strip().lower() for c in rows[0]]
    has_header = "sentence_id" in header and "text" in header
    emotion_idx: int | None = None
    sid_idx = 0
    text_idx = 1
    start = 0
    if has_header:
        start = 1
        sid_idx = header.index("sentence_id")
        text_idx = header.index("text")
        if "emotion" in header:
            emotion_idx = header.index("emotion")

    out: dict[tuple[str | None, int], str] = {}
    for r in rows[start:]:
        if len(r) <= max(sid_idx, text_idx):
            continue
        sid_raw = r[sid_idx].strip()
        text = r[text_idx].strip()
        if not sid_raw or not text:
            continue
        emo: str | None = None
        if emotion_idx is not None and emotion_idx < len(r):
            emo_raw = r[emotion_idx].strip()
            if emo_raw:
                emo = emo_raw.lower()
        try:
            sid = int(sid_raw)
        except ValueError:
            continue
        out[(emo, sid)] = normalize_whitespace(text)
    return out


def _normalize_corpus_text(text: str) -> str:
    # NFKC expands common ligatures (e.g., "ﬁ" -> "fi") and normalizes symbols.
    text = unicodedata.normalize("NFKC", text)
    return normalize_whitespace(text)


def _load_mead_corpus_from_supp_pdf(pdf_path: Path) -> dict[str, list[str]]:
    """
    Parses MEAD-supp.pdf (Section 6: Speech Corpus of Mead) and returns:
      { "common": [...3...], "generic": [...10...], "anger": [...17...], ..., "neutral": [...27...] }
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "MEAD transcripts require parsing the official MEAD supplementary PDF (MEAD-supp.pdf).\n\n"
            "Install dependency:\n"
            "  python -m pip install pypdf\n\n"
            "Or provide data/raw/MEAD/mead_sentences.csv."
        ) from e

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    lines: list[str] = []
    for p in pages:
        lines.extend(p.splitlines())

    in_section = False
    current_key: str | None = None
    current_num: int | None = None
    buf: dict[str, dict[int, str]] = {}

    header_to_key = {
        "Common Sentences Read in Eight Emotions": "common",
        "Generic Sentences Read in Eight Emotions": "generic",
        "Angry": "anger",
        "Disgust": "disgust",
        "Contempt": "contempt",
        "Fear": "fear",
        "Happy": "happy",
        "Sad": "sad",
        "Surprise": "surprise",
        "Neutral": "neutral",
    }
    heading_set = {h.lower() for h in header_to_key.keys()}

    def _split_multi_numbered(line: str) -> list[str]:
        # pypdf extraction sometimes merges two numbered items onto the same line, e.g.:
        # "9. ... 10. ..."
        starts = [m.start() for m in re.finditer(r"\b\d+\.", line)]
        if len(starts) <= 1:
            return [line]
        out: list[str] = []
        for a, b in zip(starts, starts[1:] + [len(line)]):
            seg = line[a:b].strip()
            if seg:
                out.append(seg)
        return out

    def _start_item(num: int, text: str) -> None:
        nonlocal current_num
        if current_key is None:
            return
        buf.setdefault(current_key, {})
        buf[current_key][num] = _normalize_corpus_text(text)
        current_num = num

    def _append(text: str) -> None:
        if current_key is None or current_num is None:
            return
        cur = buf.get(current_key, {}).get(current_num, "")
        add = _normalize_corpus_text(text)
        if not add:
            return
        if cur.endswith("-"):
            cur = cur[:-1] + add
        else:
            cur = (cur + " " + add).strip() if cur else add
        buf[current_key][current_num] = cur

    for raw in lines:
        raw_line = raw.strip()
        if not raw_line:
            continue
        if not in_section:
            if _MEAD_CORPUS_SECTION_RE.match(raw_line):
                in_section = True
            continue

        for line in _split_multi_numbered(raw_line):
        # Stop if we already parsed everything we need.
            if (
                "common" in buf
                and "generic" in buf
                and all(
                    k in buf
                    for k in ["anger", "disgust", "contempt", "fear", "happy", "sad", "surprise", "neutral"]
                )
                and len(buf["common"]) >= 3
                and len(buf["generic"]) >= 10
                and len(buf["neutral"]) >= 27
            ):
                break

            lower = line.lower()
            if lower in heading_set:
                current_key = header_to_key[next(h for h in header_to_key.keys() if h.lower() == lower)]
                current_num = None
                continue

            m = _MEAD_LIST_NUM_RE.match(line)
            if m:
                _start_item(int(m.group("n")), m.group("text"))
                continue

            _append(line)

    expected_counts = {
        "common": 3,
        "generic": 10,
        "anger": 17,
        "disgust": 17,
        "contempt": 17,
        "fear": 17,
        "happy": 17,
        "sad": 17,
        "surprise": 17,
        "neutral": 27,
    }
    corpus: dict[str, list[str]] = {}
    for key, n in expected_counts.items():
        items = buf.get(key, {})
        if len(items) < n:
            raise RuntimeError(f"Failed to parse MEAD speech corpus from {pdf_path} (section={key}, got={len(items)})")
        corpus[key] = [_normalize_corpus_text(items[i]) for i in range(1, n + 1)]
    return corpus


def _resolve_mead_transcript(
    *,
    emotion_raw: str,
    sentence_id: int,
    corpus: dict[str, list[str]],
    overrides: dict[tuple[str | None, int], str] | None = None,
) -> str:
    # Per-emotion override first, then global override.
    if overrides:
        if (emotion_raw, sentence_id) in overrides:
            return overrides[(emotion_raw, sentence_id)]
        if (None, sentence_id) in overrides:
            return overrides[(None, sentence_id)]

    if not corpus:
        raise RuntimeError(
            f"MEAD transcript missing for emotion={emotion_raw} sentence_id={sentence_id}. "
            "Provide data/raw/MEAD/MEAD-supp.pdf or extend data/raw/MEAD/mead_sentences.csv."
        )

    if sentence_id <= 0:
        raise RuntimeError(f"Invalid MEAD sentence_id={sentence_id}")
    if sentence_id <= 3:
        return corpus["common"][sentence_id - 1]
    if sentence_id <= 13:
        return corpus["generic"][sentence_id - 4]
    idx = sentence_id - 14
    if emotion_raw == "neutral":
        arr = corpus["neutral"]
    else:
        if emotion_raw not in corpus:
            raise RuntimeError(f"Unknown MEAD emotion for transcript lookup: {emotion_raw}")
        arr = corpus[emotion_raw]
    if idx < 0 or idx >= len(arr):
        raise RuntimeError(f"MEAD sentence_id out of range for emotion={emotion_raw}: {sentence_id}")
    return arr[idx]


def _load_sentence_map(dataset_root: Path) -> tuple[dict[str, list[str]], dict[tuple[str | None, int], str] | None]:
    logger = logging.getLogger("mer_builder.prepare.parse_mead")
    overrides: dict[tuple[str | None, int], str] | None = None
    explicit = dataset_root / "mead_sentences.csv"
    if explicit.exists():
        overrides = _load_sentence_map_from_csv(explicit)
        if overrides:
            logger.info("MEAD sentence overrides from %s (%d entries)", explicit, len(overrides))

    # Prefer the official supplementary PDF (contains the full speech corpus lists).
    supp_pdf = dataset_root / "MEAD-supp.pdf"
    if not supp_pdf.exists():
        # also accept alternate filenames
        for cand in dataset_root.rglob("MEAD-supp.pdf"):
            supp_pdf = cand
            break
    if supp_pdf.exists():
        corpus = _load_mead_corpus_from_supp_pdf(supp_pdf)
        logger.info("MEAD speech corpus from %s", supp_pdf)
        return corpus, overrides

    if overrides:
        logger.warning("MEAD-supp.pdf not found; using mead_sentences.csv-only transcript lookup.")
        return {}, overrides

    raise FileNotFoundError(
        "Could not locate MEAD speech corpus for transcripts.\n\n"
        "Expected one of:\n"
        "  - data/raw/MEAD/MEAD-supp.pdf (downloaded automatically by the MEAD downloader), or\n"
        "  - data/raw/MEAD/mead_sentences.csv (sentence_id,text) to override.\n"
    )


def _infer_speaker_id(path: Path) -> str | None:
    m = _SPEAKER_RE.search(str(path))
    if not m:
        return None
    return m.group(0).upper()


def _infer_emotion(path: Path) -> str | None:
    text = str(path).lower()
    for emo, keys in _EMO_CANDIDATES.items():
        for k in keys:
            if k in text:
                return emo
    return None


def _infer_sentence_id_simple(path: Path, speaker_id: str | None) -> int | None:
    """
    MEAD Part0 audio clips are typically named like "001.m4a" (1-indexed sentence id).
    Falls back to extracting the last numeric token from the filename stem.
    """
    stem = path.stem.strip()
    if stem.isdigit():
        return int(stem)

    nums = [int(x) for x in re.findall(r"\d+", stem)]
    if speaker_id:
        try:
            nums = [n for n in nums if n != int(speaker_id[1:])]
        except Exception:
            pass
    return nums[-1] if nums else None


def parse_mead(raw_dir: Path, *, mead_contempt: str) -> list[Sample]:
    """
    Parses MEAD Part0 speech audio.

    Transcript derivation:
    - Attempts to locate sentence scripts in raw_dir/MEAD.
    - If not found, requires user-provided: data/raw/MEAD/mead_sentences.csv (sentence_id,text).

    Assumption:
    - Audio filenames contain a numeric sentence identifier that matches sentence_id in the scripts table.
    """
    logger = logging.getLogger("mer_builder.prepare.parse_mead")
    dataset_root = find_dataset_dir(raw_dir, ["MEAD", "mead"])
    if dataset_root is None:
        raise FileNotFoundError("MEAD not found under raw_dir (expected data/raw/MEAD).")

    corpus, overrides = _load_sentence_map(dataset_root)

    audio_files = sorted(iter_audio_files(dataset_root))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {dataset_root}")

    samples: list[Sample] = []
    dropped_contempt = 0

    for audio in audio_files:
        speaker = _infer_speaker_id(audio)
        if not speaker:
            logger.debug("Skipping (speaker id not found): %s", audio)
            continue

        emotion_raw = _infer_emotion(audio)
        if emotion_raw is None:
            logger.debug("Skipping (emotion not found): %s", audio)
            continue

        mapped = map_emotion("mead", emotion_raw, mead_contempt=mead_contempt)  # type: ignore[arg-type]
        if mapped.emotion is None:
            dropped_contempt += 1
            continue

        # NOTE: transcript lookup uses the raw MEAD emotion category (e.g., "happy", "angry") even if we map
        # to the unified 7-class space (e.g., happy->joy).
        sid = _infer_sentence_id_simple(audio, speaker)
        if sid is None:
            raise RuntimeError(
                f"Could not extract MEAD sentence_id for {audio.name}. "
                "Provide/verify data/raw/MEAD/mead_sentences.csv and ensure filenames contain the sentence id."
            )
        transcript = _resolve_mead_transcript(
            emotion_raw=emotion_raw,
            sentence_id=sid,
            corpus=corpus,
            overrides=overrides,
        )

        rel = relpath_posix(audio, dataset_root)
        samples.append(
            Sample(
                dataset="MEAD",
                split="__UNASSIGNED__",
                speaker_id=f"mead_{speaker}",
                raw_audio_path=audio,
                source_relpath=rel,
                transcript=normalize_transcript(transcript),
                emotion=mapped.emotion,
                source_label=emotion_raw,
                notes=mapped.notes,
            )
        )

    logger.info("Parsed MEAD: %d samples (dropped contempt=%d)", len(samples), dropped_contempt)
    return samples
