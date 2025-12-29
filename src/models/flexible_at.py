from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .architectures import GFL


class SwiGLU(nn.Module):
    """SwiGLU activation block (SiLU-gated linear unit).

    Given input x, computes: silu(W1 x) * (W2 x), implemented via one linear layer
    producing 2*out_dim features.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)
        init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return F.silu(a) * b


class FlexibleATModel(nn.Module):
    """Audio/Text fusion model with modality selection.

    - AT: uses the existing GFL (text,audio) branch.
    - A or T: bypasses fusion; projects the active modality embedding to `gated_dim`.
    """

    def __init__(
        self,
        input_dim_audio: int,
        input_dim_text: int,
        gated_dim: int,
        n_classes: int,
        drop: float,
        modalities: Iterable[str],
    ) -> None:
        super().__init__()

        self.modalities = self._normalize_modalities(modalities)

        self.norm_audio_in = nn.LayerNorm(input_dim_audio)
        self.norm_text_in = nn.LayerNorm(input_dim_text)

        self.fc_audio = nn.Linear(input_dim_audio, input_dim_audio)
        self.drop_audio = nn.Dropout(p=drop)
        self.fc_text = nn.Linear(input_dim_text, input_dim_text)
        self.drop_text = nn.Dropout(p=drop)

        init.xavier_uniform_(self.fc_audio.weight)
        init.xavier_uniform_(self.fc_text.weight)

        self.gal_ta = GFL(input_dim_text, input_dim_audio, gated_dim)

        self.norm_fused = nn.LayerNorm(gated_dim)
        self.act_fused = SwiGLU(gated_dim, gated_dim)

        self.drop_fused = nn.Dropout(p=drop)
        self.classifier = nn.Linear(gated_dim, n_classes)
        init.xavier_uniform_(self.classifier.weight)

        self.proj_audio = SwiGLU(input_dim_audio, gated_dim)
        self.proj_text = SwiGLU(input_dim_text, gated_dim)

    @staticmethod
    def _normalize_modalities(modalities: Iterable[str]) -> list[str]:
        mods = [str(m).strip().upper() for m in modalities if str(m).strip()]
        invalid = sorted({m for m in mods if m not in {"A", "T"}})
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Allowed: ['A', 'T']")
        if not mods:
            raise ValueError("At least one modality must be selected.")
        order = ["A", "T"]
        return [m for m in order if m in set(mods)]

    @staticmethod
    def _require_tensor(x: torch.Tensor | None, name: str) -> torch.Tensor:
        if x is None:
            raise ValueError(f"Missing required modality input: {name}")
        return x

    def forward(self, batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
        mods = self.modalities

        audio = batch.get("audio")
        text = batch.get("text")

        if mods == ["A", "T"]:
            audio = self._require_tensor(audio, "audio")
            text = self._require_tensor(text, "text")
            audio_n = self.norm_audio_in(audio)
            text_n = self.norm_text_in(text)
            fc_audio = self.drop_audio(self.fc_audio(audio_n))
            fc_text = self.drop_text(self.fc_text(text_n))
            h = self.gal_ta(fc_text, fc_audio)  # (text, audio) matches baseline ordering
            h = self.norm_fused(h)
            h = self.act_fused(h)
            h = self.drop_fused(h)
            return self.classifier(h)

        if mods == ["A"]:
            audio = self._require_tensor(audio, "audio")
            audio_n = self.norm_audio_in(audio)
            fc_audio = self.drop_audio(self.fc_audio(audio_n))
            h = self.proj_audio(fc_audio)
            h = self.drop_fused(h)
            return self.classifier(h)

        if mods == ["T"]:
            text = self._require_tensor(text, "text")
            text_n = self.norm_text_in(text)
            fc_text = self.drop_text(self.fc_text(text_n))
            h = self.proj_text(fc_text)
            h = self.drop_fused(h)
            return self.classifier(h)

        raise RuntimeError(f"Unsupported modality set: {mods}")
