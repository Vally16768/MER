from __future__ import annotations

import logging
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):  # type: ignore
        return it

from mer_builder.prepare.types import Sample
from mer_builder.utils.ffmpeg import convert_to_wav_mono_16k, ffprobe_duration_sec
from mer_builder.utils.io import read_jsonl


def _write_silence_wav(dst_path: Path, *, sample_rate: int = 16000, seconds: float = 0.5) -> float:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    nframes = max(1, int(sample_rate * seconds))
    data = b"\x00\x00" * nframes  # 16-bit mono silence
    with wave.open(str(dst_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return float(nframes) / float(sample_rate)


def normalize_audio(
    samples: list[Sample],
    *,
    out_dir: Path,
    num_workers: int = 8,
    force: bool = False,
    drop_failed: bool = True,
    replace_with_silence: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    logger = logging.getLogger("mer_builder.prepare.normalize_audio")

    # Speed-up for reruns: if a previous manifest exists, reuse duration_sec for samples whose
    # output WAV already exists. This avoids running ffprobe for tens of thousands of files.
    duration_cache: dict[str, float] = {}
    manifest_path = out_dir / "meta_manifest.jsonl"
    if not force and manifest_path.exists():
        try:
            for r in read_jsonl(manifest_path):
                sid = r.get("id")
                dur = r.get("duration_sec")
                if sid is None or dur is None:
                    continue
                try:
                    duration_cache[str(sid)] = float(dur)
                except Exception:
                    continue
            if duration_cache:
                logger.info("Reusing durations from existing manifest: %s (%d rows)", manifest_path, len(duration_cache))
        except Exception:
            duration_cache = {}

    def _work(sample: Sample) -> tuple[str, float | None, str | None]:
        if sample.audio_relpath is None:
            raise ValueError("Sample missing audio_relpath")
        dst_path = out_dir / "audio" / sample.audio_relpath

        # Fully idempotent rerun behavior:
        # If output already exists, and we have a cached duration, do not touch the file or run ffprobe.
        if not force and sample.id and dst_path.exists():
            cached = duration_cache.get(sample.id)
            if cached and cached > 0:
                return sample.id, float(cached), None

        try:
            # Some corpora contain placeholder media files with no actual audio stream (very tiny containers).
            # Skip calling ffmpeg for these to avoid noisy errors.
            size = sample.raw_audio_path.stat().st_size
            if size < 1024:
                if replace_with_silence:
                    if not force and dst_path.exists():
                        dur = ffprobe_duration_sec(dst_path)
                        return sample.id or "", float(dur), f"kept_existing_audio(input_too_small_bytes={size})"
                    dur = _write_silence_wav(dst_path, seconds=0.5)
                    return sample.id or "", float(dur), f"replaced_with_silence(input_too_small_bytes={size})"
                return sample.id or "", None, f"input_too_small_bytes={size}"
        except Exception:
            # If we can't stat it, ffmpeg will fail anyway; proceed.
            pass
        try:
            convert_to_wav_mono_16k(sample.raw_audio_path, dst_path, force=force)
            dur = ffprobe_duration_sec(dst_path)
            return sample.id or "", float(dur), None
        except Exception as e:
            if replace_with_silence:
                if not force and dst_path.exists():
                    dur = ffprobe_duration_sec(dst_path)
                    return sample.id or "", float(dur), f"kept_existing_audio({e})"
                dur = _write_silence_wav(dst_path, seconds=0.5)
                return sample.id or "", float(dur), f"replaced_with_silence({e})"
            return sample.id or "", None, str(e)

    futures = []
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        for s in samples:
            futures.append(ex.submit(_work, s))

        id_to_sample = {s.id: s for s in samples if s.id}
        failures: list[tuple[str, str]] = []
        replaced: list[tuple[str, str]] = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Normalize audio"):
            sample_id, dur, err = fut.result()
            if sample_id in id_to_sample and dur is not None:
                id_to_sample[sample_id].duration_sec = float(dur)
                if err:
                    replaced.append((sample_id, err))
            elif sample_id:
                failures.append((sample_id, err or "unknown_error"))

    missing = [s for s in samples if s.duration_sec is None]
    if missing and not drop_failed:
        logger.error("Missing duration for %d samples", len(missing))
        raise RuntimeError("Audio normalization incomplete (missing durations)")

    if missing:
        logger.warning("Dropping %d samples due to audio decode/normalize failure", len(missing))
    return ({k: v for k, v in failures}, {k: v for k, v in replaced})
