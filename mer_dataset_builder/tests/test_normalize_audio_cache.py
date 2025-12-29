from __future__ import annotations

from pathlib import Path

from mer_builder.prepare.normalize_audio import normalize_audio
from mer_builder.prepare.types import Sample
from mer_builder.utils.io import write_jsonl


def test_normalize_audio_reuses_duration_cache(tmp_path: Path):
    out_dir = tmp_path / "processed"
    audio_rel = "MELD/meld_abc.wav"
    audio_path = out_dir / "audio" / audio_rel
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"RIFF0000WAVEfmt ")  # placeholder, we won't ffprobe it because cache is used

    manifest = out_dir / "meta_manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "meld_abc",
                "dataset": "MELD",
                "split": "meld_train",
                "speaker_id": "meld_spk",
                "audio_path": audio_rel,
                "transcript": "hello",
                "emotion": "neutral",
                "duration_sec": 1.23,
                "source_label": "neutral",
            }
        ],
    )

    s = Sample(
        dataset="MELD",
        split="meld_train",
        speaker_id="meld_spk",
        raw_audio_path=tmp_path / "missing_input.wav",
        source_relpath="x.wav",
        transcript="hello",
        emotion="neutral",
        source_label="neutral",
    )
    s.id = "meld_abc"
    s.audio_relpath = audio_rel

    failures, replaced = normalize_audio([s], out_dir=out_dir, num_workers=1, replace_with_silence=False)
    assert failures == {}
    assert replaced == {}
    assert s.duration_sec == 1.23

