from __future__ import annotations

from pathlib import Path

from mer_builder.prepare.parse_emovdb import _collect_emovdb_wavs


def test_collect_emovdb_wavs_dedups_by_stem(tmp_path: Path):
    root = tmp_path / "EmoV-DB"
    extracted = root / "extracted" / "bea_Amused"
    extracted.mkdir(parents=True, exist_ok=True)

    # Same stem appears both under extracted/ and at root -> prefer extracted path and count once.
    (extracted / "amused_1-15_0001.wav").write_bytes(b"x")
    (root / "amused_1-15_0001.wav").write_bytes(b"y")

    wavs = _collect_emovdb_wavs(root)
    assert len(wavs) == 1
    assert "extracted" in str(wavs[0]).replace("\\", "/")

