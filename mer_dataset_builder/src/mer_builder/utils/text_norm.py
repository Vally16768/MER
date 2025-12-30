from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    return text

