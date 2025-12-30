from __future__ import annotations

import hashlib


def sha1_short(text: str, *, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def stable_sample_id(dataset: str, speaker_id: str, source_relpath: str) -> str:
    base = f"{dataset}|{speaker_id}|{source_relpath}"
    return f"{dataset}_{sha1_short(base, n=16)}"

