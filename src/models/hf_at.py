from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from augmentation import apply_modality_dropout
from .robust_at import RobustATModel


def _infer_feature_vector_attention_mask(
    encoder: nn.Module,
    input_attention_mask: torch.Tensor | None,
    *,
    hidden_len: int,
) -> torch.Tensor | None:
    """Convert input attention_mask (aligned to input_values length) to feature-vector length.

    HF speech encoders (wav2vec2/HuBERT/WavLM) internally downsample time, so `last_hidden_state`
    length is typically shorter than the input waveform length. The model accepts the input mask,
    but for pooling we need a mask aligned to `last_hidden_state.shape[1]`.
    """

    if input_attention_mask is None:
        return None

    if int(input_attention_mask.shape[1]) == int(hidden_len):
        return input_attention_mask

    lengths = input_attention_mask.to(dtype=torch.long).sum(dim=1)
    hidden_len_i = int(hidden_len)
    if hidden_len_i <= 0:
        return None

    out_lengths: torch.Tensor | None = None
    fn = getattr(encoder, "_get_feat_extract_output_lengths", None)
    if callable(fn):
        try:
            out_lengths = fn(lengths)  # type: ignore[misc]
        except Exception:
            out_lengths = None

    if out_lengths is None:
        # Fallback: scale by ratio between padded input length and output hidden length.
        max_in = int(input_attention_mask.shape[1])
        ratio = float(hidden_len_i) / float(max_in) if max_in > 0 else 1.0
        out_lengths = torch.ceil(lengths.to(dtype=torch.float32) * ratio).to(dtype=torch.long)

    out_lengths = out_lengths.clamp(min=1, max=hidden_len_i)
    rng = torch.arange(hidden_len_i, device=out_lengths.device).unsqueeze(0)
    mask = rng < out_lengths.unsqueeze(1)
    return mask.to(dtype=input_attention_mask.dtype)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    # hidden: (B, T, D), mask: (B, T)
    if mask is None:
        return hidden.mean(dim=1)
    m = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (hidden * m).sum(dim=1) / denom


def _masked_std(hidden: torch.Tensor, mask: torch.Tensor | None, *, mean: torch.Tensor | None = None) -> torch.Tensor:
    # Returns sqrt(E[x^2] - mean^2). Uses the same mask as _masked_mean.
    if mask is None:
        return hidden.std(dim=1)
    m = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)
    ex = (hidden * m).sum(dim=1) / denom
    ex2 = ((hidden * hidden) * m).sum(dim=1) / denom
    var = (ex2 - ex * ex).clamp(min=0.0)
    return var.sqrt()


class HFAudioTextModel(nn.Module):
    """End-to-end Audio+Text model using HuggingFace encoders + RobustAT fusion.

    Expects batches with keys:
      - audio_input_values: (B, T) float
      - audio_attention_mask: (B, T) optional int/bool
      - text_input_ids: (B, L) int
      - text_attention_mask: (B, L) optional int/bool
      - label: (B,) long
    """

    def __init__(
        self,
        *,
        audio_model: str,
        text_model: str,
        n_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        pool_audio: str = "mean",
        pool_text: str = "cls",
        freeze_audio: bool = False,
        freeze_text: bool = False,
        freeze_audio_feature_encoder: bool = False,
        gradient_checkpointing: bool = False,
        modalities: Iterable[str] = ("A", "T"),
        modality_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.modalities = [str(m).strip().upper() for m in modalities if str(m).strip()]
        invalid = sorted({m for m in self.modalities if m not in {"A", "T"}})
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Allowed: ['A','T']")
        if not self.modalities:
            raise ValueError("At least one modality must be selected.")

        self.pool_audio = str(pool_audio or "mean").strip().lower()
        self.pool_text = str(pool_text or "cls").strip().lower()
        if self.pool_audio not in {"mean", "mean_std"}:
            raise ValueError("pool_audio must be 'mean' or 'mean_std'")
        if self.pool_text not in {"cls", "mean"}:
            raise ValueError("pool_text must be 'cls' or 'mean'")

        self.modality_dropout_p = float(modality_dropout_p or 0.0)

        try:
            from transformers import AutoModel  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency `transformers`. Install it to use HFAudioTextModel.") from exc

        try:
            self.audio_encoder = AutoModel.from_pretrained(audio_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio encoder {audio_model!r}. If you are offline, pass a local path."
            ) from exc

        try:
            self.text_encoder = AutoModel.from_pretrained(text_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load text encoder {text_model!r}. If you are offline, pass a local path."
            ) from exc

        audio_dim = int(getattr(self.audio_encoder.config, "hidden_size", 0) or 0)
        text_dim = int(getattr(self.text_encoder.config, "hidden_size", 0) or 0)
        if audio_dim <= 0 or text_dim <= 0:
            raise ValueError(f"Could not infer encoder dims: audio_dim={audio_dim} text_dim={text_dim}")

        input_dim_a = audio_dim * 2 if self.pool_audio == "mean_std" else audio_dim
        input_dim_t = text_dim

        self.fusion = RobustATModel(
            input_dim_audio=input_dim_a,
            input_dim_text=input_dim_t,
            hidden_dim=int(hidden_dim),
            n_classes=int(n_classes),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            ffn_mult=int(ffn_mult),
            modalities=self.modalities,
        )

        if freeze_audio_feature_encoder:
            fn = getattr(self.audio_encoder, "freeze_feature_encoder", None)
            if callable(fn):
                fn()

        if gradient_checkpointing:
            for enc in (self.audio_encoder, self.text_encoder):
                fn = getattr(enc, "gradient_checkpointing_enable", None)
                if callable(fn):
                    fn()

        if freeze_audio:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def _pool_audio(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.pool_audio == "mean_std":
            mean = _masked_mean(hidden, mask)
            std = _masked_std(hidden, mask, mean=mean)
            return torch.cat([mean, std], dim=-1)
        return _masked_mean(hidden, mask)

    def _pool_text(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.pool_text == "mean":
            return _masked_mean(hidden, mask)
        return hidden[:, 0, :]

    def forward(self, batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
        audio_emb: torch.Tensor | None = None
        text_emb: torch.Tensor | None = None

        if "A" in self.modalities:
            audio_in = batch.get("audio_input_values")
            if audio_in is None:
                raise ValueError("Missing audio_input_values for audio modality")
            audio_mask = batch.get("audio_attention_mask")
            out_a = self.audio_encoder(audio_in, attention_mask=audio_mask)
            hidden_a = out_a.last_hidden_state
            hidden_mask = _infer_feature_vector_attention_mask(self.audio_encoder, audio_mask, hidden_len=int(hidden_a.shape[1]))
            audio_emb = self._pool_audio(hidden_a, hidden_mask)

        if "T" in self.modalities:
            text_ids = batch.get("text_input_ids")
            if text_ids is None:
                raise ValueError("Missing text_input_ids for text modality")
            text_mask = batch.get("text_attention_mask")
            out_t = self.text_encoder(input_ids=text_ids, attention_mask=text_mask)
            hidden_t = out_t.last_hidden_state
            text_emb = self._pool_text(hidden_t, text_mask)

        if self.training and self.modality_dropout_p > 0 and audio_emb is not None and text_emb is not None:
            audio_emb, text_emb, _ = apply_modality_dropout(
                audio=audio_emb, text=text_emb, p=self.modality_dropout_p, generator=None
            )

        return self.fusion({"audio": audio_emb, "text": text_emb})
