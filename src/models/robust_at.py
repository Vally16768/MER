from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)
        init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return F.silu(a) * b


class FeedForward(nn.Module):
    def __init__(self, dim: int, *, mult: int = 4) -> None:
        super().__init__()
        hidden = int(dim) * int(mult)
        self.ln = nn.LayerNorm(dim)
        self.up = SwiGLU(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        init.xavier_uniform_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.up(y)
        y = self.down(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by num_heads={num_heads}")

        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=int(num_heads), dropout=0.0, batch_first=True)
        self.ffn = FeedForward(dim, mult=int(ffn_mult))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


class RobustATModel(nn.Module):
    """Robust Audio/Text fusion model.

    - Projects audio/text embeddings into a shared `hidden_dim` using SwiGLU.
    - Uses a small attention stack over tokens [CLS, audio, text] (no dropout).
    - Works with modality selection: A, T, or AT.
    """

    def __init__(
        self,
        *,
        input_dim_audio: int,
        input_dim_text: int,
        hidden_dim: int,
        n_classes: int,
        num_layers: int,
        num_heads: int,
        ffn_mult: int,
        modalities: Iterable[str],
    ) -> None:
        super().__init__()

        self.modalities = self._normalize_modalities(modalities)

        self.norm_audio_in = nn.LayerNorm(input_dim_audio)
        self.norm_text_in = nn.LayerNorm(input_dim_text)

        self.proj_audio = SwiGLU(input_dim_audio, hidden_dim)
        self.proj_text = SwiGLU(input_dim_text, hidden_dim)

        # Token type ids: 0=CLS, 1=audio, 2=text.
        self.type_embed = nn.Embedding(3, hidden_dim)
        init.xavier_uniform_(self.type_embed.weight)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        init.normal_(self.cls_token, mean=0.0, std=0.02)

        layers = [TransformerBlock(hidden_dim, num_heads=int(num_heads), ffn_mult=int(ffn_mult)) for _ in range(int(num_layers))]
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            SwiGLU(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
        init.xavier_uniform_(self.classifier.weight)

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

    def _build_tokens(self, *, audio: torch.Tensor | None, text: torch.Tensor | None) -> torch.Tensor:
        bsz = None
        tokens: list[torch.Tensor] = []
        type_ids: list[int] = []

        if audio is not None:
            bsz = int(audio.shape[0])
            a = self.norm_audio_in(audio)
            a = self.proj_audio(a).unsqueeze(1)
            tokens.append(a)
            type_ids.append(1)

        if text is not None:
            bsz = int(text.shape[0]) if bsz is None else bsz
            t = self.norm_text_in(text)
            t = self.proj_text(t).unsqueeze(1)
            tokens.append(t)
            type_ids.append(2)

        if not tokens or bsz is None:
            raise ValueError("At least one modality input must be present.")

        device = tokens[0].device
        cls = self.cls_token.expand(bsz, -1, -1)
        cls = cls + self.type_embed(torch.zeros((bsz, 1), dtype=torch.long, device=device))

        x = torch.cat(tokens, dim=1)  # (B, n_tokens, D)
        type_t = torch.tensor(type_ids, dtype=torch.long, device=device).view(1, -1).expand(bsz, -1)
        x = x + self.type_embed(type_t)

        return torch.cat([cls, x], dim=1)  # (B, 1+n_tokens, D)

    def forward(self, batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
        audio = batch.get("audio")
        text = batch.get("text")

        mods = self.modalities
        if mods == ["A", "T"]:
            audio = self._require_tensor(audio, "audio")
            text = self._require_tensor(text, "text")
        elif mods == ["A"]:
            audio = self._require_tensor(audio, "audio")
            text = None
        elif mods == ["T"]:
            text = self._require_tensor(text, "text")
            audio = None
        else:
            raise RuntimeError(f"Unsupported modality set: {mods}")

        x = self._build_tokens(audio=audio, text=text)
        x = self.encoder(x)
        pooled = x[:, 0, :]
        pooled = pooled + self.head(pooled)
        return self.classifier(pooled)

