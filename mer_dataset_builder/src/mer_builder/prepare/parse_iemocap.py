from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mer_builder.prepare.map_labels import map_emotion
from mer_builder.prepare.types import Sample
from mer_builder.utils.io import find_dataset_dir, iter_audio_files, read_text_maybe, relpath_posix
from mer_builder.utils.text_norm import normalize_transcript

_EVAL_HEADER_RE = re.compile(
    r"^\[(?P<start>[-\d.]+)\s*-\s*(?P<end>[-\d.]+)\]\s*(?P<utt>\S+)\s+(?P<label>\S+)",
    flags=re.IGNORECASE,
)
_SPEAKER_RE = re.compile(r"^(?P<spk>Ses\d{2}[FM])_", flags=re.IGNORECASE)

_TX_RE_1 = re.compile(r"^(?P<utt>\S+)\s+\[[^\]]+\]:\s*(?P<text>.*)$")
_TX_RE_2 = re.compile(r"^(?P<utt>\S+)\s*:\s*(?P<text>.*)$")


@dataclass(frozen=True)
class _EvalItem:
    utt_id: str
    source_label: str
    categorical_votes: tuple[str, ...]


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


def _find_iemocap_root(raw_dir: Path) -> Path | None:
    # Prefer conventional placement under raw_dir, but support "next to project" layouts too.
    cand = find_dataset_dir(
        raw_dir,
        ["IEMOCAP_full_release", "IEMOCAP"],
        extra_roots=[raw_dir.parent.parent, Path.cwd()],
    )
    if cand is None:
        return None
    return _normalize_iemocap_root(cand)


def _parse_transcriptions(dataset_root: Path) -> dict[str, str]:
    """
    Builds utterance_id -> transcript.
    """
    logger = logging.getLogger("mer_builder.prepare.parse_iemocap")
    tx: dict[str, str] = {}
    sessions = [dataset_root / f"Session{i}" for i in range(1, 6)]
    for session in sessions:
        tdir = session / "dialog" / "transcriptions"
        if not tdir.is_dir():
            continue
        for p in sorted(tdir.glob("*.txt")):
            if p.name.startswith("._"):
                continue
            for line in read_text_maybe(p).splitlines():
                s = line.strip()
                if not s:
                    continue
                m = _TX_RE_1.match(s) or _TX_RE_2.match(s)
                if not m:
                    continue
                utt = m.group("utt").strip()
                if not _SPEAKER_RE.match(utt):
                    # Some transcription files contain extra lines like "M: ..." / "F: ..." (not utterance IDs).
                    continue
                text = normalize_transcript(m.group("text").strip())
                if not utt:
                    continue
                key = utt.lower()
                if key in tx and tx[key] != text:
                    logger.warning("IEMOCAP transcript mismatch for %s (keeping first)", utt)
                    continue
                tx[key] = text
    return tx


def _parse_eval_file(path: Path) -> list[_EvalItem]:
    items: list[_EvalItem] = []
    current_utt: str | None = None
    current_label: str | None = None
    votes: list[str] = []

    for line in read_text_maybe(path).splitlines():
        s = line.strip()
        if not s:
            continue

        m = _EVAL_HEADER_RE.match(s)
        if m:
            if current_utt and current_label is not None:
                items.append(_EvalItem(current_utt, current_label, tuple(votes)))
            current_utt = m.group("utt").strip()
            current_label = m.group("label").strip()
            votes = []
            continue

        # Categorical votes (one per coder; we use the first label if multiple are provided).
        if current_utt and s.startswith("C-") and ":" in s:
            rest = s.split(":", 1)[1].strip()
            if not rest:
                continue
            cat_field = rest.split("\t", 1)[0].strip()
            if not cat_field:
                continue
            first = cat_field.split(";", 1)[0].strip()
            if first:
                votes.append(first)

    if current_utt and current_label is not None:
        items.append(_EvalItem(current_utt, current_label, tuple(votes)))
    return items


def _derive_from_categorical(votes: tuple[str, ...]) -> tuple[str | None, str | None]:
    """
    Derives a 7-class emotion from categorical coder votes.
    Returns: (emotion, notes)
    """
    mapped: list[str] = []
    for v in votes:
        r = map_emotion("iemocap", v)
        if r.emotion:
            mapped.append(r.emotion)

    if not mapped:
        return None, None

    c = Counter(mapped)
    best = max(c.values())
    tied = sorted([k for k, v in c.items() if v == best])
    emotion = tied[0]

    # Keep notes short but informative (original votes can be useful for audits).
    vote_str = ",".join(v.strip().replace(" ", "_") for v in votes if v.strip())
    notes = f"derived_from_categorical={emotion};votes={vote_str}" if vote_str else f"derived_from_categorical={emotion}"
    return emotion, notes


def _speaker_id_from_utt(utt_id: str) -> str | None:
    m = _SPEAKER_RE.match(utt_id)
    if not m:
        return None
    return m.group("spk")


def parse_iemocap(raw_dir: Path) -> tuple[list[Sample], list[dict[str, str]]]:
    """
    Parses IEMOCAP (original full release).

    - Emotions from: Session*/dialog/EmoEvaluation/*.txt
    - Transcripts from: Session*/dialog/transcriptions/*.txt
    - Audio from: Session*/sentences/wav/**/<utt_id>.wav
    """
    logger = logging.getLogger("mer_builder.prepare.parse_iemocap")

    dataset_root = _find_iemocap_root(raw_dir)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            "IEMOCAP not found under raw_dir.\n\n"
            "Expected one of:\n"
            "  - data/raw/IEMOCAP_full_release/\n"
            "  - data/raw/IEMOCAP/\n"
            "  - <repo>/IEMOCAP_full_release/\n"
        )

    # Build audio index first (utterance wavs).
    wav_index: dict[str, Path] = {}
    for i in range(1, 6):
        wav_root = dataset_root / f"Session{i}" / "sentences" / "wav"
        if not wav_root.is_dir():
            continue
        for wav in iter_audio_files(wav_root, exts=(".wav",)):
            key = wav.stem.lower()
            if key in wav_index:
                continue
            wav_index[key] = wav
    if not wav_index:
        raise FileNotFoundError(f"IEMOCAP wavs not found under {dataset_root} (expected Session*/sentences/wav/**.wav)")

    transcripts = _parse_transcriptions(dataset_root)
    if not transcripts:
        raise FileNotFoundError(
            f"IEMOCAP transcripts not found under {dataset_root} (expected Session*/dialog/transcriptions/*.txt)"
        )

    samples: list[Sample] = []
    dropped: list[dict[str, str]] = []

    for i in range(1, 6):
        eval_dir = dataset_root / f"Session{i}" / "dialog" / "EmoEvaluation"
        if not eval_dir.is_dir():
            continue
        for eval_file in sorted(eval_dir.glob("*.txt")):
            if eval_file.name.startswith("._"):
                continue
            for item in _parse_eval_file(eval_file):
                utt_id = item.utt_id.strip()
                if not utt_id:
                    continue

                speaker = _speaker_id_from_utt(utt_id)
                if not speaker:
                    dropped.append(
                        {
                            "utt_id": utt_id,
                            "source_label": item.source_label,
                            "reason": "missing_speaker_id",
                            "eval_file": str(eval_file),
                        }
                    )
                    continue

                wav = wav_index.get(utt_id.lower())
                if wav is None:
                    raise FileNotFoundError(
                        f"Missing IEMOCAP utterance wav for {utt_id} (referenced in {eval_file}). "
                        "Expected files under Session*/sentences/wav/**/<utt_id>.wav."
                    )

                transcript = transcripts.get(utt_id.lower())
                if transcript is None:
                    raise FileNotFoundError(
                        f"Missing IEMOCAP transcript for {utt_id} (referenced in {eval_file}). "
                        "Expected transcripts under Session*/dialog/transcriptions/."
                    )

                mapped = map_emotion("iemocap", item.source_label)
                notes_parts: list[str] = []
                emotion = mapped.emotion
                if mapped.notes and emotion is not None:
                    notes_parts.append(mapped.notes)

                if emotion is None:
                    derived, derived_notes = _derive_from_categorical(item.categorical_votes)
                    if derived is None:
                        dropped.append(
                            {
                                "utt_id": utt_id,
                                "source_label": item.source_label,
                                "reason": mapped.notes or "unmapped_label",
                                "votes": ",".join(item.categorical_votes),
                                "eval_file": str(eval_file),
                            }
                        )
                        continue
                    emotion = derived
                    if derived_notes:
                        notes_parts.append(derived_notes)

                rel = relpath_posix(wav, dataset_root)
                notes = "; ".join([p for p in notes_parts if p]) if notes_parts else None
                samples.append(
                    Sample(
                        dataset="IEMOCAP",
                        split="__UNASSIGNED__",
                        speaker_id=f"iemocap_{speaker}",
                        raw_audio_path=wav,
                        source_relpath=rel,
                        transcript=transcript,
                        emotion=emotion,
                        source_label=item.source_label,
                        notes=notes,
                    )
                )

    logger.info("Parsed IEMOCAP: %d samples (dropped %d)", len(samples), len(dropped))
    return samples, dropped
