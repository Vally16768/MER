from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _split_to_group(split: str) -> str | None:
    s = str(split).strip()
    if s in {"train", "meld_train"}:
        return "train"
    if s in {"val", "meld_dev"}:
        return "val"
    if s in {"testA", "test"}:
        return "testA"
    if s in {"testB", "meld_test"}:
        return "testB"
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check manifest split grouping + speaker overlaps (acted datasets).")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("mer_dataset_builder/data/processed/meta_manifest.jsonl"),
        help="Path to unified meta_manifest.jsonl",
    )
    p.add_argument(
        "--acted_datasets",
        nargs="*",
        default=None,
        help="Datasets to treat as 'acted' for speaker-disjoint checks (default: all except MELD).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest: Path = args.manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    counts_split = Counter()
    counts_dataset = Counter()
    counts_group = Counter()
    counts_group_dataset = Counter()

    speakers = defaultdict(lambda: defaultdict(set))

    # First pass: counts.
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            dataset = str(r.get("dataset"))
            split = str(r.get("split"))
            grp = _split_to_group(split)

            counts_split[split] += 1
            counts_dataset[dataset] += 1
            if grp is not None:
                counts_group[grp] += 1
                counts_group_dataset[(grp, dataset)] += 1

    acted = set(args.acted_datasets) if args.acted_datasets else set(counts_dataset) - {"MELD"}

    # Second pass: speaker sets for acted datasets (train/val/testA only).
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            dataset = str(r.get("dataset"))
            if dataset not in acted:
                continue
            grp = _split_to_group(str(r.get("split")))
            if grp not in {"train", "val", "testA"}:
                continue
            speakers[dataset][grp].add(str(r.get("speaker_id")))

    print("=== Manifest counts (raw splits) ===")
    for k, v in sorted(counts_split.items()):
        print(f"{k}: {v}")

    print("\n=== Manifest counts (datasets) ===")
    for k, v in sorted(counts_dataset.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k}: {v}")

    print("\n=== Grouped split counts ===")
    for k, v in sorted(counts_group.items()):
        print(f"{k}: {v}")

    print("\n=== testB dataset distribution ===")
    items = [(d, n) for (g, d), n in counts_group_dataset.items() if g == "testB"]
    for d, n in sorted(items, key=lambda kv: (-kv[1], kv[0])):
        print(f"{d}: {n}")

    print("\n=== Speaker overlap checks (acted datasets) ===")
    for d in sorted(acted):
        tr = speakers[d].get("train", set())
        va = speakers[d].get("val", set())
        te = speakers[d].get("testA", set())
        print(
            f"{d}: speakers train={len(tr)} val={len(va)} testA={len(te)} "
            f"overlaps tr/va={len(tr & va)} tr/te={len(tr & te)} va/te={len(va & te)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

