from __future__ import annotations

import random
from collections.abc import Iterable

from mer_builder.config import SplitRatios


def split_speakers(
    speaker_ids: Iterable[str],
    *,
    seed: int,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, str]:
    speakers = sorted(set(speaker_ids))
    if not speakers:
        return {}

    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * ratios.train)
    n_val = int(n * ratios.val)
    n_test = n - n_train - n_val

    # Ensure non-empty splits when possible.
    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
    elif n == 2:
        n_train, n_val, n_test = 1, 0, 1
    else:
        n_train, n_val, n_test = 1, 0, 0

    mapping: dict[str, str] = {}
    for i, spk in enumerate(speakers):
        if i < n_train:
            mapping[spk] = "train"
        elif i < n_train + n_val:
            mapping[spk] = "val"
        else:
            mapping[spk] = "test"
    return mapping


def speaker_overlap(speakers_by_split: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Returns non-empty intersections for all split-pairs, keyed as "splitA|splitB".
    """
    splits = sorted(speakers_by_split.keys())
    overlaps: dict[str, set[str]] = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            inter = speakers_by_split[a].intersection(speakers_by_split[b])
            if inter:
                overlaps[f"{a}|{b}"] = inter
    return overlaps

