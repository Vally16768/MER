from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

_AUDIO_PROCESSOR = None
_TEXT_TOKENIZER = None
_AUDIO_MODEL = None
_TEXT_MODEL = None


def _load_processors(*, audio_model: str, text_model: str):
    global _AUDIO_PROCESSOR, _TEXT_TOKENIZER, _AUDIO_MODEL, _TEXT_MODEL

    if _AUDIO_PROCESSOR is not None and _TEXT_TOKENIZER is not None and _AUDIO_MODEL == audio_model and _TEXT_MODEL == text_model:
        return _AUDIO_PROCESSOR, _TEXT_TOKENIZER

    try:
        from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency `transformers`. Install it to use HF end-to-end training.") from exc

    # Some speech encoders (e.g., WavLM) don't ship a tokenizer, so AutoProcessor may fail.
    try:
        try:
            audio_processor = AutoProcessor.from_pretrained(audio_model)
        except Exception:
            audio_processor = AutoFeatureExtractor.from_pretrained(audio_model)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio processor/feature extractor for {audio_model!r}.") from exc

    try:
        text_tokenizer = AutoTokenizer.from_pretrained(text_model)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer for {text_model!r}.") from exc

    _AUDIO_PROCESSOR = audio_processor
    _TEXT_TOKENIZER = text_tokenizer
    _AUDIO_MODEL = audio_model
    _TEXT_MODEL = text_model
    return _AUDIO_PROCESSOR, _TEXT_TOKENIZER


@dataclass(frozen=True)
class ManifestCollator:
    audio_model: str
    text_model: str
    modalities: list[str]
    text_max_tokens: int = 256

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audio_processor, text_tokenizer = _load_processors(audio_model=str(self.audio_model), text_model=str(self.text_model))

        labels = torch.stack([b["label"] for b in batch], dim=0)
        out: dict[str, Any] = {"label": labels}

        modalities = [str(m).strip().upper() for m in self.modalities if str(m).strip()]

        if "A" in modalities:
            wavs = [b["wav"].detach().cpu().numpy() for b in batch]
            aud = audio_processor(
                wavs,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            if "input_values" not in aud:
                raise ValueError("Audio processor did not return input_values")
            out["audio_input_values"] = aud["input_values"]
            if "attention_mask" in aud:
                out["audio_attention_mask"] = aud["attention_mask"].to(dtype=torch.bool)

        if "T" in modalities:
            texts = [str(b.get("transcript", "")) for b in batch]
            enc = text_tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=int(self.text_max_tokens),
                padding=True,
            )
            out["text_input_ids"] = enc["input_ids"]
            if "attention_mask" in enc:
                out["text_attention_mask"] = enc["attention_mask"].to(dtype=torch.bool)

        return out
