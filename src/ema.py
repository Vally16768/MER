from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EMAConfig:
    decay: float = 0.999
    use_for_eval: bool = True


class EMA:
    """Exponential Moving Average (EMA) of model parameters.

    This is a lightweight generalization helper. Typical usage:
      ema = EMA(model, decay=0.999)
      ... after each optimizer step: ema.update(model)
      ... for evaluation: with ema.apply_to(model): eval()
    """

    def __init__(self, model: torch.nn.Module, *, decay: float = 0.999) -> None:
        self.decay = float(decay)
        if not (0.0 < self.decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")

        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        backup: dict[str, torch.Tensor] = {}
        try:
            for name, p in model.named_parameters():
                if name not in self.shadow:
                    continue
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])
            yield
        finally:
            for name, p in model.named_parameters():
                if name not in backup:
                    continue
                p.data.copy_(backup[name])

