from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Emotion7 = Literal["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

EMOTIONS_7: Final[list[str]] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]

DATASET_KEYS: Final[list[str]] = ["mead", "meld", "ravdess", "cremad", "esd", "emovdb", "iemocap"]

DATASET_DISPLAY_NAMES: Final[dict[str, str]] = {
    "mead": "MEAD",
    "meld": "MELD",
    "ravdess": "RAVDESS",
    "cremad": "CREMA-D",
    "esd": "ESD",
    "emovdb": "EmoV-DB",
    "iemocap": "IEMOCAP",
}

ACTED_DATASET_KEYS: Final[set[str]] = {"mead", "ravdess", "cremad", "esd", "emovdb", "iemocap"}

DEFAULT_SEED: Final[int] = 1337

AUDIO_SAMPLE_RATE: Final[int] = 16000
AUDIO_CHANNELS: Final[int] = 1
AUDIO_CODEC: Final[str] = "pcm_s16le"

MEAD_CONTEMPT_MODES: Final[set[str]] = {"drop", "map_to_disgust"}
EMOVDB_SLEEPY_MODES: Final[set[str]] = {"drop", "map_to_neutral"}


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
