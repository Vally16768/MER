from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from mer_builder.config import ACTED_DATASET_KEYS, DATASET_DISPLAY_NAMES, EMOTIONS_7
from mer_builder.prepare.split_speakers import speaker_overlap
from mer_builder.utils.io import read_jsonl


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def check_manifest_integrity(
    manifest_path: Path,
    *,
    out_dir: Path | None = None,
    print_limit: int = 20,
) -> int:
    """
    Returns exit code:
      0 -> OK
      2 -> integrity issues found
    """
    rows = list(read_jsonl(manifest_path))
    if not rows:
        print(f"ERROR: Empty or missing manifest: {manifest_path}")
        return 2

    out_dir = out_dir or manifest_path.parent
    audio_root = out_dir / "audio"

    required = ["id", "dataset", "split", "speaker_id", "audio_path", "transcript", "emotion", "duration_sec", "source_label"]

    dup_ids: list[str] = []
    seen: set[str] = set()

    missing_fields = 0
    bad_emotion = 0
    empty_transcript = 0
    bad_duration = 0
    missing_audio: list[str] = []
    empty_audio: list[str] = []

    for r in rows:
        if any(k not in r for k in required):
            missing_fields += 1
            continue

        rid = str(r["id"])
        if rid in seen:
            dup_ids.append(rid)
        else:
            seen.add(rid)

        emo = str(r["emotion"])
        if emo not in EMOTIONS_7:
            bad_emotion += 1

        tx = str(r["transcript"]).strip()
        if not tx:
            empty_transcript += 1

        dur = _as_float(r["duration_sec"])
        if dur is None or dur <= 0:
            bad_duration += 1

        rel = Path(str(r["audio_path"]))
        audio_path = (audio_root / rel).resolve()
        try:
            audio_path.relative_to(audio_root.resolve())
        except Exception:
            missing_audio.append(str(rel))
            continue
        if not audio_path.exists():
            missing_audio.append(str(rel))
        else:
            try:
                if audio_path.stat().st_size <= 0:
                    empty_audio.append(str(rel))
            except Exception:
                empty_audio.append(str(rel))

    # Speaker overlap check (acted datasets only)
    overlap_fail = 0
    for key in sorted(ACTED_DATASET_KEYS):
        ds_name = DATASET_DISPLAY_NAMES[key]
        ds_rows = [r for r in rows if r.get("dataset") == ds_name]
        if not ds_rows:
            continue
        speakers_by_split: dict[str, set[str]] = {}
        for r in ds_rows:
            speakers_by_split.setdefault(str(r["split"]), set()).add(str(r["speaker_id"]))
        overlaps = speaker_overlap(speakers_by_split)
        if overlaps:
            overlap_fail += 1

    issues = (
        len(dup_ids)
        + missing_fields
        + bad_emotion
        + empty_transcript
        + bad_duration
        + len(missing_audio)
        + len(empty_audio)
        + overlap_fail
    )

    by_dataset = Counter(r.get("dataset") for r in rows)
    print("\n=== Integrity summary ===")
    print(f"manifest={manifest_path}")
    print(f"rows={len(rows)} datasets={len(by_dataset)}")
    print(f"missing_fields={missing_fields} dup_ids={len(dup_ids)} bad_emotion={bad_emotion}")
    print(f"empty_transcript={empty_transcript} bad_duration={bad_duration}")
    print(f"missing_audio_files={len(missing_audio)} empty_audio_files={len(empty_audio)}")
    print(f"speaker_overlap_acted_failures={overlap_fail}")

    if missing_audio:
        print("\nMissing audio examples:")
        for x in missing_audio[:print_limit]:
            print(f"  {x}")
    if empty_audio:
        print("\nEmpty audio examples:")
        for x in empty_audio[:print_limit]:
            print(f"  {x}")
    if dup_ids:
        print("\nDuplicate id examples:")
        for x in dup_ids[:print_limit]:
            print(f"  {x}")

    if issues:
        print("\nERROR: Integrity check failed.")
        return 2
    print("\nOK: Manifest + audio files look consistent.")
    return 0

