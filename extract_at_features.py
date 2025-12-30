from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLASS_NAMES: list[str] = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TO_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _split_to_group(split: str) -> str | None:
    s = str(split).strip()
    if s in {"train", "meld_train"}:
        return "train"
    if s in {"val", "meld_dev"}:
        return "val"
    if s in {"testA", "test"}:
        return "testA"
    if s in {"testB", "meld_test"}:
        return "testB"
    return None


@dataclass(frozen=True)
class Sample:
    id: str
    dataset: str
    split_group: str
    speaker_id: str
    audio_relpath: str
    transcript: str
    emotion: str


_MELD_DIALOG_RE = re.compile(r"dia(\d+)_utt(\d+)", flags=re.IGNORECASE)


def _speaker_prefix(speaker_id: str) -> str:
    spk = str(speaker_id or "").strip()
    if not spk:
        return ""
    # Keep it simple and tokenizable for HashingVectorizer.
    return f"speaker={spk} "


def _meld_dialogue_utt(sample: Sample) -> tuple[str, int] | None:
    # Expected examples:
    #   id: "MELD_dia125_utt3"
    #   id: "dia125_utt3"
    for s in (sample.id, sample.audio_relpath):
        m = _MELD_DIALOG_RE.search(str(s))
        if m:
            dia = f"dia{int(m.group(1))}"
            utt = int(m.group(2))
            return dia, utt
    return None


def _apply_text_transforms(
    samples: list[Sample],
    *,
    include_speaker_in_text: bool,
    meld_context_window: int,
    meld_context_sep: str,
) -> list[Sample]:
    if not include_speaker_in_text and meld_context_window <= 0:
        return samples

    base_text: list[str] = []
    for s in samples:
        text = str(s.transcript or "")
        if include_speaker_in_text:
            text = (_speaker_prefix(s.speaker_id) + text).strip()
        base_text.append(text)

    if meld_context_window <= 0:
        return [Sample(**{**s.__dict__, "transcript": base_text[i]}) for i, s in enumerate(samples)]

    by_dialogue: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for i, s in enumerate(samples):
        if str(s.dataset).strip().upper() != "MELD":
            continue
        key = _meld_dialogue_utt(s)
        if key is None:
            continue
        dia, utt = key
        by_dialogue.setdefault((s.split_group, dia), []).append((int(utt), int(i)))

    new_samples: list[Sample] = []
    id_to_ctx: dict[int, str] = {}
    win = int(meld_context_window)
    sep = str(meld_context_sep)
    for _, pairs in by_dialogue.items():
        pairs.sort(key=lambda x: x[0])
        idxs = [i for _, i in pairs]
        for pos, sample_idx in enumerate(idxs):
            start = max(0, pos - win)
            ctx_parts = [base_text[j] for j in idxs[start:pos] if base_text[j]]
            cur = base_text[sample_idx]
            if ctx_parts:
                id_to_ctx[sample_idx] = sep.join(ctx_parts + [cur]).strip()
            else:
                id_to_ctx[sample_idx] = cur

    for i, s in enumerate(samples):
        text = id_to_ctx.get(i, base_text[i])
        new_samples.append(Sample(**{**s.__dict__, "transcript": text}))

    return new_samples


def _setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("extract_at_features")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _read_manifest(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


_MFCC = None
_VEC = None
_AUDIO_DIM = None
_TEXT_DIM = None
_FAILURE_MODE = None


def _init_worker(n_mfcc: int, text_dim: int, failure_mode: str) -> None:
    global _MFCC, _VEC, _AUDIO_DIM, _TEXT_DIM, _FAILURE_MODE

    import torch
    import torchaudio
    from sklearn.feature_extraction.text import HashingVectorizer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    _AUDIO_DIM = int(n_mfcc) * 2
    _TEXT_DIM = int(text_dim)
    _FAILURE_MODE = str(failure_mode)

    # torchaudio requires n_mfcc <= n_mels; also n_mels must be <= n_freqs (= n_fft//2 + 1).
    n_fft = 400
    n_freqs = n_fft // 2 + 1  # 201
    n_mfcc_i = int(n_mfcc)
    # Keep a reasonably high mel resolution when requesting larger MFCC counts.
    n_mels = max(64, n_mfcc_i)
    n_mels = min(n_mels, n_freqs)
    if n_mfcc_i > n_mels:
        raise ValueError(f"Invalid MFCC params: n_mfcc={n_mfcc_i} > n_mels={n_mels}. Reduce n_mfcc.")

    _MFCC = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=n_mfcc_i,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": 160,
            "n_mels": n_mels,
            "center": True,
            "power": 2.0,
        },
    )
    _VEC = HashingVectorizer(
        n_features=int(text_dim),
        alternate_sign=False,
        norm=None,
        lowercase=True,
    )


def _extract_one_task(task: tuple[Sample, str], audio_dir: str) -> dict[str, Any]:
    sample, out_path = task
    return _extract_one(sample, audio_dir=audio_dir, out_path=out_path)


def _zero_features(*, audio_dim: int, text_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((audio_dim,), dtype=np.float32), np.zeros((text_dim,), dtype=np.float32)


def _extract_one(sample: Sample, *, audio_dir: str, out_path: str) -> dict[str, Any]:
    global _MFCC, _VEC, _AUDIO_DIM, _TEXT_DIM, _FAILURE_MODE
    assert _AUDIO_DIM is not None and _TEXT_DIM is not None and _FAILURE_MODE is not None

    out_p = Path(out_path)
    if out_p.exists():
        return {"status": "skipped", "id": sample.id, "split": sample.split_group}

    audio_path = Path(audio_dir) / sample.audio_relpath

    audio_vec: np.ndarray
    text_vec: np.ndarray
    notes: list[str] = []

    try:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=16000)

        mfcc = _MFCC(wav)  # (1, n_mfcc, frames)
        mfcc = mfcc.squeeze(0)
        mean = mfcc.mean(dim=-1)
        std = mfcc.std(dim=-1)
        feats = torch.cat([mean, std], dim=0).to(dtype=torch.float32).cpu().numpy()
        audio_vec = feats.astype(np.float32, copy=False)
    except Exception as exc:
        if _FAILURE_MODE == "skip":
            return {"status": "failed", "id": sample.id, "split": sample.split_group, "error": f"audio: {exc}"}
        audio_vec, _ = _zero_features(audio_dim=int(_AUDIO_DIM), text_dim=int(_TEXT_DIM))
        notes.append(f"audio_failed:{type(exc).__name__}")

    try:
        sparse = _VEC.transform([sample.transcript or ""])
        dense = sparse.toarray()[0].astype(np.float32, copy=False)
        if dense.shape[0] != int(_TEXT_DIM):
            raise ValueError(f"Unexpected text dim: {dense.shape}")
        text_vec = dense
    except Exception as exc:
        if _FAILURE_MODE == "skip":
            return {"status": "failed", "id": sample.id, "split": sample.split_group, "error": f"text: {exc}"}
        _, text_vec = _zero_features(audio_dim=int(_AUDIO_DIM), text_dim=int(_TEXT_DIM))
        notes.append(f"text_failed:{type(exc).__name__}")

    label = EMOTION_TO_INDEX.get(str(sample.emotion).strip().lower())
    if label is None:
        if _FAILURE_MODE == "skip":
            return {
                "status": "failed",
                "id": sample.id,
                "split": sample.split_group,
                "error": f"unknown emotion: {sample.emotion!r}",
            }
        label = 0
        notes.append(f"unknown_emotion_mapped_to:{CLASS_NAMES[0]}")

    payload: dict[str, Any] = {
        "id": sample.id,
        "dataset": sample.dataset,
        "split": sample.split_group,
        "speaker_id": sample.speaker_id,
        "emotion": sample.emotion,
        "true_label": int(label),
        "audio_features": audio_vec,
        "text_features": text_vec,
    }
    if notes:
        payload["notes"] = ";".join(notes)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {"status": "ok", "id": sample.id, "split": sample.split_group}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Audio+Text feature pickles from mer_dataset_builder manifest.")
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("mer_dataset_builder/data/processed"),
        help="Path to mer_dataset_builder processed dir (contains meta_manifest.jsonl and audio/).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/features/mer_builder_at_simple"),
        help="Output feature root dir. Creates train/ val/ testA/ testB subdirs.",
    )
    p.add_argument("--n_mfcc", type=int, default=80, help="MFCC count (audio_dim = 2*n_mfcc).")
    p.add_argument("--text_dim", type=int, default=768, help="HashingVectorizer output dim (text features).")
    p.add_argument("--include_speaker_in_text", action="store_true", help="Prefix transcript with speaker_id.")
    p.add_argument(
        "--meld_context_window",
        type=int,
        default=0,
        help="For MELD only: prepend up to N previous utterances from the same dialogue (no future context).",
    )
    p.add_argument("--meld_context_sep", type=str, default=" [SEP] ", help="Separator used when building MELD context.")
    p.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    p.add_argument("--limit", type=int, default=0, help="If >0, process only first N samples (debug).")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if output pickle exists.")
    p.add_argument(
        "--failure_mode",
        choices=["zero", "skip"],
        default="zero",
        help="On feature extraction failure: write zero vectors (zero) or skip sample (skip).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logger = _setup_logger(args.verbose)

    processed_dir: Path = args.processed_dir
    manifest_path = processed_dir / "meta_manifest.jsonl"
    audio_dir = processed_dir / "audio"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Missing audio dir: {audio_dir}")

    out_dir: Path = args.out_dir
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    logger.info("manifest=%s", manifest_path)
    logger.info("audio_dir=%s", audio_dir)
    logger.info("out_dir=%s", out_dir)

    samples: list[Sample] = []
    counts_in: dict[str, int] = {}
    skipped_unknown_split = 0

    for row in _read_manifest(manifest_path):
        split_group = _split_to_group(row.get("split", ""))
        if split_group is None:
            skipped_unknown_split += 1
            continue

        counts_in[split_group] = counts_in.get(split_group, 0) + 1

        samples.append(
            Sample(
            id=str(row["id"]),
            dataset=str(row.get("dataset", "")),
            split_group=split_group,
            speaker_id=str(row.get("speaker_id", "")),
            audio_relpath=str(row.get("audio_path", "")),
            transcript=str(row.get("transcript", "")),
            emotion=str(row.get("emotion", "")),
            )
        )

    samples = _apply_text_transforms(
        samples,
        include_speaker_in_text=bool(args.include_speaker_in_text),
        meld_context_window=int(args.meld_context_window),
        meld_context_sep=str(args.meld_context_sep),
    )

    tasks: list[tuple[Sample, str]] = []
    skipped_existing = 0
    for sample in samples:
        if args.limit and len(tasks) >= int(args.limit):
            break

        out_path = out_dir / sample.split_group / f"{sample.id}.pkl"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        if out_path.exists() and args.overwrite:
            try:
                out_path.unlink()
            except OSError:
                pass

        tasks.append((sample, str(out_path)))

    total_selected = sum(counts_in.values())
    logger.info(
        "selected=%d (train=%d val=%d testA=%d testB=%d)",
        total_selected,
        counts_in.get("train", 0),
        counts_in.get("val", 0),
        counts_in.get("testA", 0),
        counts_in.get("testB", 0),
    )
    if skipped_unknown_split:
        logger.warning("skipped_unknown_split=%d", skipped_unknown_split)
    if skipped_existing and not args.overwrite:
        logger.info("skipped_existing=%d (rerun is idempotent)", skipped_existing)

    if not tasks:
        logger.info("Nothing to do.")
        return 0

    failures_path = stats_dir / "feature_extraction_failures.jsonl"
    ok = skipped = failed = 0

    import itertools
    from concurrent.futures import ProcessPoolExecutor

    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:
        tqdm = None  # type: ignore[assignment]

    bar = tqdm(total=len(tasks), desc="extract features") if tqdm else None

    with ProcessPoolExecutor(
        max_workers=int(args.num_workers),
        initializer=_init_worker,
        initargs=(int(args.n_mfcc), int(args.text_dim), str(args.failure_mode)),
    ) as ex:
        with failures_path.open("w", encoding="utf-8") as f_fail:
            for res in ex.map(_extract_one_task, tasks, itertools.repeat(str(audio_dir)), chunksize=32):
                status = res.get("status")
                if status == "ok":
                    ok += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    f_fail.write(json.dumps(res, ensure_ascii=False) + "\n")
                if bar:
                    bar.update(1)

    if bar:
        bar.close()

    build_info = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "processed_dir": str(processed_dir),
        "manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
        "out_dir": str(out_dir),
        "feature_backend": "mfcc+hashing",
        "class_names": CLASS_NAMES,
        "n_mfcc": int(args.n_mfcc),
        "audio_dim": int(args.n_mfcc) * 2,
        "text_dim": int(args.text_dim),
        "include_speaker_in_text": bool(args.include_speaker_in_text),
        "meld_context_window": int(args.meld_context_window),
        "meld_context_sep": str(args.meld_context_sep),
        "num_workers": int(args.num_workers),
        "failure_mode": str(args.failure_mode),
        "selected_counts": counts_in,
        "processed_ok": int(ok),
        "processed_failed": int(failed),
        "processed_skipped": int(skipped),
        "skipped_existing": int(skipped_existing),
    }
    (stats_dir / "build_info.json").write_text(json.dumps(build_info, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("done ok=%d failed=%d", ok, failed)
    if failed:
        logger.warning("Some samples failed; see %s", failures_path)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
