from __future__ import annotations

from pathlib import Path

from mer_builder.prepare.normalize_audio import _write_silence_wav


def test_write_silence_wav(tmp_path: Path):
    out = tmp_path / "x.wav"
    dur = _write_silence_wav(out, sample_rate=16000, seconds=0.5)
    assert out.exists()
    assert out.stat().st_size > 44  # WAV header + some samples
    assert 0.49 <= dur <= 0.51

