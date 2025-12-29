from __future__ import annotations

from pathlib import Path

from mer_builder.prepare.integrity import check_manifest_integrity
from mer_builder.utils.io import write_jsonl


def test_integrity_ok(tmp_path: Path):
    out_dir = tmp_path / "processed"
    audio_dir = out_dir / "audio" / "MELD"
    audio_dir.mkdir(parents=True, exist_ok=True)

    wav_rel = Path("MELD/meld_abc.wav")
    (out_dir / "audio" / wav_rel).write_bytes(b"RIFF0000WAVEfmt ")  # minimal non-empty placeholder

    manifest = out_dir / "meta_manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "meld_abc",
                "dataset": "MELD",
                "split": "meld_train",
                "speaker_id": "meld_spk",
                "audio_path": wav_rel.as_posix(),
                "transcript": "hello",
                "emotion": "neutral",
                "duration_sec": 1.0,
                "source_label": "neutral",
            }
        ],
    )
    assert check_manifest_integrity(manifest, out_dir=out_dir) == 0


def test_integrity_missing_audio(tmp_path: Path):
    out_dir = tmp_path / "processed"
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "meta_manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "x",
                "dataset": "MELD",
                "split": "meld_train",
                "speaker_id": "meld_spk",
                "audio_path": "MELD/missing.wav",
                "transcript": "hello",
                "emotion": "neutral",
                "duration_sec": 1.0,
                "source_label": "neutral",
            }
        ],
    )
    assert check_manifest_integrity(manifest, out_dir=out_dir) == 2
