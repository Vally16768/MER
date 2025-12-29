from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from mer_builder.config import ACTED_DATASET_KEYS, DATASET_DISPLAY_NAMES
from mer_builder.prepare.split_speakers import speaker_overlap
from mer_builder.utils.io import read_jsonl


def _percentiles(values: list[float], ps: list[float]) -> dict[float, float]:
    if not values:
        return {p: math.nan for p in ps}
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(values, dtype=float)
        return {p: float(np.percentile(arr, p)) for p in ps}
    except Exception:
        s = sorted(values)
        out: dict[float, float] = {}
        for p in ps:
            k = (len(s) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                out[p] = float(s[int(k)])
            else:
                out[p] = float(s[f] * (c - k) + s[c] * (k - f))
        return out


def validate_manifest(manifest_path: Path) -> None:
    rows = list(read_jsonl(manifest_path))
    if not rows:
        raise FileNotFoundError(f"Empty or missing manifest: {manifest_path}")

    by_dataset = Counter(r["dataset"] for r in rows)
    by_emotion = Counter(r["emotion"] for r in rows)
    by_split = Counter(r["split"] for r in rows)

    print("\n=== Counts per dataset ===")
    for k, v in sorted(by_dataset.items()):
        print(f"{k}: {v}")

    print("\n=== Counts per emotion ===")
    for k, v in sorted(by_emotion.items()):
        print(f"{k}: {v}")

    print("\n=== Counts per split ===")
    for k, v in sorted(by_split.items()):
        print(f"{k}: {v}")

    print("\n=== Speaker overlap (acted datasets) ===")
    for key in sorted(ACTED_DATASET_KEYS):
        ds_name = DATASET_DISPLAY_NAMES[key]
        ds_rows = [r for r in rows if r["dataset"] == ds_name]
        if not ds_rows:
            continue
        speakers_by_split: dict[str, set[str]] = defaultdict(set)
        for r in ds_rows:
            speakers_by_split[r["split"]].add(r["speaker_id"])
        overlaps = speaker_overlap(speakers_by_split)
        if overlaps:
            print(f"{ds_name}: FAIL (speaker overlap detected)")
            for pair, speakers in overlaps.items():
                print(f"  {pair}: {len(speakers)} overlapping speakers")
            raise SystemExit(2)
        print(f"{ds_name}: OK")

    durations: list[float] = []
    for r in rows:
        try:
            durations.append(float(r["duration_sec"]))
        except Exception:
            continue
    if durations:
        p = _percentiles(durations, [0, 1, 5, 50, 95, 99, 100])
        mean = sum(durations) / len(durations)
        print("\n=== Duration summary (sec) ===")
        print(f"n={len(durations)} min={min(durations):.3f} mean={mean:.3f} max={max(durations):.3f}")
        print(
            "p0={:.3f} p1={:.3f} p5={:.3f} p50={:.3f} p95={:.3f} p99={:.3f} p100={:.3f}".format(
                p[0], p[1], p[5], p[50], p[95], p[99], p[100]
            )
        )

