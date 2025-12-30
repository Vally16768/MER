from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModalityDropoutStats:
    batch_size: int
    dropped_audio: int
    dropped_text: int

    @property
    def drop_audio_rate(self) -> float:
        return float(self.dropped_audio) / float(max(1, self.batch_size))

    @property
    def drop_text_rate(self) -> float:
        return float(self.dropped_text) / float(max(1, self.batch_size))


def _rand(shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


@torch.no_grad()
def apply_feature_noise(x: torch.Tensor | None, *, std: float, generator: torch.Generator | None = None) -> torch.Tensor | None:
    s = float(std or 0.0)
    if x is None or s <= 0.0:
        return x
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * s
    return x + noise


@torch.no_grad()
def apply_modality_dropout(
    *,
    audio: torch.Tensor | None,
    text: torch.Tensor | None,
    p: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, ModalityDropoutStats | None]:
    """Randomly zero-out entire modality vectors (per-sample).

    - Only applies when BOTH `audio` and `text` are present.
    - Ensures at least one modality remains per sample.
    """
    prob = float(p or 0.0)
    if prob <= 0.0 or audio is None or text is None:
        return audio, text, None

    if audio.shape[0] != text.shape[0]:
        raise ValueError(f"Batch size mismatch: audio={audio.shape} text={text.shape}")

    bsz = int(audio.shape[0])
    device = audio.device
    drop_a = _rand((bsz,), device=device, generator=generator) < prob
    drop_t = _rand((bsz,), device=device, generator=generator) < prob

    both = drop_a & drop_t
    if bool(both.any()):
        idx = both.nonzero(as_tuple=False).squeeze(1)
        keep_audio = _rand((int(idx.numel()),), device=device, generator=generator) < 0.5
        drop_t[idx[keep_audio]] = False
        drop_a[idx[~keep_audio]] = False

    audio2 = audio
    text2 = text
    if bool(drop_a.any()):
        audio2 = audio.clone()
        audio2[drop_a] = 0
    if bool(drop_t.any()):
        text2 = text.clone()
        text2[drop_t] = 0

    stats = ModalityDropoutStats(
        batch_size=bsz,
        dropped_audio=int(drop_a.sum().item()),
        dropped_text=int(drop_t.sum().item()),
    )
    return audio2, text2, stats

