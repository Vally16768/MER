from __future__ import annotations

import json
import logging
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

from mer_builder.config import ACTED_DATASET_KEYS, DATASET_DISPLAY_NAMES, DEFAULT_SEED, SplitRatios
from mer_builder.prepare.normalize_audio import normalize_audio
from mer_builder.prepare.parse_cremad import parse_cremad
from mer_builder.prepare.parse_emovdb import parse_emovdb
from mer_builder.prepare.parse_esd import parse_esd
from mer_builder.prepare.parse_iemocap import parse_iemocap
from mer_builder.prepare.parse_mead import parse_mead
from mer_builder.prepare.parse_meld import parse_meld
from mer_builder.prepare.parse_ravdess import parse_ravdess
from mer_builder.prepare.split_speakers import split_speakers
from mer_builder.prepare.types import Sample
from mer_builder.utils.ffmpeg import ensure_ffmpeg
from mer_builder.utils.hashing import stable_sample_id
from mer_builder.utils.io import atomic_write_text, ensure_dir, write_csv, write_jsonl


def stable_hash_int(text: str) -> int:
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return int(h)


def _assign_ids_and_paths(samples: list[Sample]) -> None:
    for s in samples:
        s.id = stable_sample_id(s.dataset, s.speaker_id, s.source_relpath)
        # Relative to out_dir/audio/
        s.audio_relpath = f"{s.dataset}/{s.id}.wav"


def _assign_acted_splits(samples: list[Sample], *, seed: int) -> None:
    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_dataset[s.dataset].append(s)

    for key in sorted(ACTED_DATASET_KEYS):
        ds_name = DATASET_DISPLAY_NAMES[key]
        ds_samples = by_dataset.get(ds_name, [])
        if not ds_samples:
            continue

        mapping = split_speakers(
            [s.speaker_id for s in ds_samples],
            seed=seed + stable_hash_int(ds_name),
            ratios=SplitRatios(),
        )
        for s in ds_samples:
            s.split = mapping.get(s.speaker_id, "train")

        # Evaluation set naming requested by spec
        if key == "mead":
            for s in ds_samples:
                if s.split == "test":
                    s.split = "testA"


def prepare_all(
    *,
    raw_dir: Path,
    out_dir: Path,
    datasets: list[str],
    mead_contempt: str = "drop",
    emovdb_sleepy: str = "drop",
    audio_failure: str = "drop",
    num_workers: int = 8,
    seed: int = DEFAULT_SEED,
    continue_on_error: bool = False,
) -> list[tuple[str, str]]:
    logger = logging.getLogger("mer_builder.prepare")
    ensure_ffmpeg()

    ensure_dir(out_dir / "audio")
    stats_dir = ensure_dir(out_dir / "stats")

    samples: list[Sample] = []
    dropped_rows: dict[str, list[dict[str, str]]] = {}
    failures: list[tuple[str, str]] = []

    for ds in datasets:
        try:
            if ds == "meld":
                samples.extend(parse_meld(raw_dir))
            elif ds == "ravdess":
                samples.extend(parse_ravdess(raw_dir))
            elif ds == "cremad":
                samples.extend(parse_cremad(raw_dir))
            elif ds == "esd":
                samples.extend(parse_esd(raw_dir))
            elif ds == "emovdb":
                s, dropped = parse_emovdb(raw_dir, emovdb_sleepy=emovdb_sleepy)
                samples.extend(s)
                dropped_rows["emovdb"] = dropped
            elif ds == "iemocap":
                s, dropped = parse_iemocap(raw_dir)
                samples.extend(s)
                dropped_rows["iemocap"] = dropped
            elif ds == "mead":
                samples.extend(parse_mead(raw_dir, mead_contempt=mead_contempt))
            else:
                raise ValueError(f"Unknown dataset key: {ds}")
        except Exception as e:
            if not continue_on_error:
                raise
            failures.append((ds, str(e)))
            logger.error("Prepare failed for dataset=%s: %s", ds, str(e).splitlines()[0] if str(e) else "error")
            continue

    if not samples:
        raise RuntimeError("No samples parsed. Check raw_dir paths and dataset downloads.")

    _assign_acted_splits(samples, seed=seed)
    _assign_ids_and_paths(samples)

    logger.info("Normalizing audio to mono 16kHz PCM WAV (%d workers)...", num_workers)
    replace = audio_failure == "replace_with_silence"
    audio_errors, replaced_audio = normalize_audio(
        samples,
        out_dir=out_dir,
        num_workers=num_workers,
        drop_failed=True,
        replace_with_silence=replace,
    )

    if replaced_audio:
        out = stats_dir / "replaced_audio.csv"
        rows = []
        for s in samples:
            if not s.id or s.id not in replaced_audio:
                continue
            rows.append(
                {
                    "dataset": s.dataset,
                    "speaker_id": s.speaker_id,
                    "source_relpath": s.source_relpath,
                    "raw_audio_path": str(s.raw_audio_path),
                    "audio_path": s.audio_relpath or "",
                    "reason": replaced_audio[s.id],
                }
            )
        write_csv(
            out,
            rows,
            fieldnames=["dataset", "speaker_id", "source_relpath", "raw_audio_path", "audio_path", "reason"],
        )
        logger.warning("Replaced %d audio files with silence; see %s", len(replaced_audio), out)

    dropped_audio: list[dict[str, str]] = []
    kept: list[Sample] = []
    for s in samples:
        if s.duration_sec is None:
            try:
                raw_bytes = str(s.raw_audio_path.stat().st_size)
            except Exception:
                raw_bytes = ""
            dropped_audio.append(
                {
                    "dataset": s.dataset,
                    "speaker_id": s.speaker_id,
                    "source_relpath": s.source_relpath,
                    "raw_audio_path": str(s.raw_audio_path),
                    "reason": "audio_decode_failed",
                    "raw_bytes": raw_bytes,
                    "error": audio_errors.get(s.id or "", ""),
                }
            )
            continue
        kept.append(s)
    samples = kept
    if dropped_audio:
        out = stats_dir / "dropped_audio.csv"
        write_csv(
            out,
            dropped_audio,
            fieldnames=["dataset", "speaker_id", "source_relpath", "raw_audio_path", "reason", "raw_bytes", "error"],
        )
        logger.warning("Dropped %d samples due to audio failures; see %s", len(dropped_audio), out)

    rows = [s.to_manifest_row() for s in samples]
    manifest_jsonl = out_dir / "meta_manifest.jsonl"
    manifest_csv = out_dir / "meta_manifest.csv"

    write_jsonl(manifest_jsonl, rows)
    write_csv(
        manifest_csv,
        rows,
        fieldnames=[
            "id",
            "dataset",
            "split",
            "speaker_id",
            "audio_path",
            "transcript",
            "emotion",
            "duration_sec",
            "source_label",
            "notes",
        ],
    )

    for ds_key, dropped in dropped_rows.items():
        if not dropped:
            continue
        out = stats_dir / f"dropped_{ds_key}.csv"
        fieldnames = sorted({k for r in dropped for k in r.keys()})
        write_csv(out, dropped, fieldnames=fieldnames)

    build_info = {
        "created_at_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "datasets": datasets,
        "datasets_failed": [d for d, _ in failures],
        "mead_contempt": mead_contempt,
        "num_workers": num_workers,
        "seed": seed,
        "num_samples": len(samples),
    }
    atomic_write_text(stats_dir / "build_info.json", json.dumps(build_info, indent=2))
    logger.info("Wrote manifest: %s", manifest_jsonl)
    return failures
