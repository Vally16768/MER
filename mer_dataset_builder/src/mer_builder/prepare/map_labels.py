from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mer_builder.config import EMOTIONS_7


@dataclass(frozen=True)
class LabelMapResult:
    emotion: str | None
    notes: str | None = None


def _norm(label: str) -> str:
    return label.strip().lower().replace("-", "_")


def map_emotion(
    dataset: str,
    source_label: str,
    *,
    mead_contempt: Literal["drop", "map_to_disgust"] = "drop",
    emovdb_sleepy: Literal["drop", "map_to_neutral"] = "drop",
) -> LabelMapResult:
    ds = dataset.strip().lower()
    lbl = _norm(source_label)

    if ds == "emovdb":
        if lbl == "amused":
            return LabelMapResult("joy", "mapped amused->joy")
        if lbl in {"sleepy", "sleepiness"}:
            if emovdb_sleepy == "drop":
                return LabelMapResult(None, "dropped sleepy")
            return LabelMapResult("neutral", "mapped sleepy->neutral")

    # Already-in-target labels
    if lbl in EMOTIONS_7:
        return LabelMapResult(lbl, None)

    # Common synonyms
    common = {
        "angry": "anger",
        "anger": "anger",
        "sad": "sadness",
        "sadness": "sadness",
        "happy": "joy",
        "happiness": "joy",
        "joy": "joy",
        "surprised": "surprise",
        "surprise": "surprise",
        "neutral": "neutral",
        "calm": "neutral",
        "disgusted": "disgust",
        "disgust": "disgust",
        "fearful": "fear",
        "fear": "fear",
    }
    if lbl in common:
        mapped = common[lbl]
        notes = None
        if ds in {"crema-d", "cremad"} and lbl == "happy":
            notes = "mapped happy->joy"
        if ds == "ravdess" and lbl in {"happy", "calm"}:
            notes = f"mapped {lbl}->" + ("joy" if lbl == "happy" else "neutral")
        if ds == "esd" and lbl in {"angry", "happy", "sad"}:
            notes = f"mapped {lbl}->" + common[lbl]
        return LabelMapResult(mapped, notes)

    # Dataset-specific codes / labels
    if ds in {"crema-d", "cremad"}:
        crema = {
            "hap": ("joy", "mapped happy->joy"),
            "ang": ("anger", None),
            "dis": ("disgust", None),
            "fea": ("fear", None),
            "neu": ("neutral", None),
            "sad": ("sadness", None),
        }
        if lbl in crema:
            emotion, notes = crema[lbl]
            return LabelMapResult(emotion, notes)

    if ds == "ravdess":
        rav = {
            "01": "neutral",
            "neutral": "neutral",
            "02": "neutral",  # calm -> neutral
            "calm": "neutral",
            "03": "joy",  # happy -> joy
            "happy": "joy",
            "04": "sadness",
            "sad": "sadness",
            "05": "anger",
            "angry": "anger",
            "06": "fear",
            "fearful": "fear",
            "07": "disgust",
            "disgust": "disgust",
            "08": "surprise",
            "surprised": "surprise",
        }
        if lbl in rav:
            emotion = rav[lbl]
            notes = None
            if lbl in {"02", "calm"}:
                notes = "mapped calm->neutral"
            if lbl in {"03", "happy"}:
                notes = "mapped happy->joy"
            return LabelMapResult(emotion, notes)

    if ds == "iemocap":
        # IEMOCAP categorical codes + common label spellings.
        iemocap = {
            "ang": ("anger", None),
            "sad": ("sadness", None),
            "neu": ("neutral", None),
            "sur": ("surprise", None),
            "fea": ("fear", None),
            "dis": ("disgust", None),
            "hap": ("joy", "mapped hap->joy"),
            "exc": ("joy", "mapped exc->joy"),
            "fru": ("anger", "mapped fru->anger"),
            # Tokens from coder "Categorical" lines in EmoEvaluation files.
            "frustration": ("anger", "mapped frustration->anger"),
            "happiness": ("joy", "mapped happiness->joy"),
            "excited": ("joy", "mapped excited->joy"),
            # Labels outside the unified 7-class space (may be derived from coder votes in the parser).
            "oth": (None, "unmapped oth"),
            "other": (None, "unmapped other"),
            "xxx": (None, "unmapped xxx"),
        }
        if lbl in iemocap:
            emotion, notes = iemocap[lbl]
            return LabelMapResult(emotion, notes)

    if ds == "esd":
        esd = {
            "angry": ("anger", "mapped angry->anger"),
            "happy": ("joy", "mapped happy->joy"),
            "sad": ("sadness", "mapped sad->sadness"),
            "neutral": ("neutral", None),
            "surprise": ("surprise", None),
        }
        if lbl in esd:
            emotion, notes = esd[lbl]
            return LabelMapResult(emotion, notes)

    if ds == "mead":
        if lbl == "contempt":
            if mead_contempt == "drop":
                return LabelMapResult(None, "dropped contempt")
            return LabelMapResult("disgust", "mapped contempt->disgust")
        mead = {
            "angry": "anger",
            "anger": "anger",
            "disgust": "disgust",
            "disgusted": "disgust",
            "fear": "fear",
            "fearful": "fear",
            "happy": "joy",
            "neutral": "neutral",
            "sad": "sadness",
            "sadness": "sadness",
            "surprised": "surprise",
            "surprise": "surprise",
        }
        if lbl in mead:
            mapped = mead[lbl]
            notes = None
            if lbl == "happy":
                notes = "mapped happy->joy"
            if lbl == "sad":
                notes = "mapped sad->sadness"
            return LabelMapResult(mapped, notes)

    # MELD already 7-class; if label isn't in EMOTIONS_7, drop loudly by returning None.
    if ds == "meld":
        return LabelMapResult(None, f"unknown meld label: {source_label}")

    return LabelMapResult(None, f"unmapped label: {source_label}")
