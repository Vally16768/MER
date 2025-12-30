from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


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
    return f"speaker={spk} "


def _meld_dialogue_utt(sample: Sample) -> tuple[str, int] | None:
    m = _MELD_DIALOG_RE.search(str(sample.id))
    if not m:
        return None
    dia = f"dia{int(m.group(1))}"
    utt = int(m.group(2))
    return dia, utt


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

    id_to_ctx: dict[int, str] = {}
    win = int(meld_context_window)
    sep = str(meld_context_sep)
    for _, pairs in by_dialogue.items():
        pairs.sort(key=lambda x: x[0])
        idxs = [i for _, i in pairs]
        for pos, idx in enumerate(idxs):
            start = max(0, pos - win)
            ctx_parts = [base_text[j] for j in idxs[start:pos] if base_text[j]]
            cur = base_text[idx]
            id_to_ctx[idx] = sep.join(ctx_parts + [cur]).strip() if ctx_parts else cur

    out: list[Sample] = []
    for i, s in enumerate(samples):
        out.append(Sample(**{**s.__dict__, "transcript": id_to_ctx.get(i, base_text[i])}))
    return out


def _setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("extract_at_features_hf")
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


def _resolve_device(device: str) -> str:
    device = str(device or "auto").strip().lower()
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _masked_mean(hidden, mask):
    if mask is None:
        return hidden.mean(dim=1)
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (hidden * mask_f).sum(dim=1) / denom


def _infer_feature_vector_attention_mask(model, input_attention_mask, hidden_len: int):
    if input_attention_mask is None:
        return None
    if int(input_attention_mask.shape[1]) == int(hidden_len):
        return input_attention_mask

    lengths = input_attention_mask.to(dtype=torch.long).sum(dim=1)
    hidden_len_i = int(hidden_len)
    if hidden_len_i <= 0:
        return None

    out_lengths = None
    fn = getattr(model, "_get_feat_extract_output_lengths", None)
    if callable(fn):
        try:
            out_lengths = fn(lengths)
        except Exception:
            out_lengths = None

    if out_lengths is None:
        max_in = int(input_attention_mask.shape[1])
        ratio = float(hidden_len_i) / float(max_in) if max_in > 0 else 1.0
        out_lengths = torch.ceil(lengths.to(dtype=torch.float32) * ratio).to(dtype=torch.long)

    out_lengths = out_lengths.clamp(min=1, max=hidden_len_i)
    rng = torch.arange(hidden_len_i, device=out_lengths.device).unsqueeze(0)
    mask = rng < out_lengths.unsqueeze(1)
    return mask.to(dtype=input_attention_mask.dtype)


def _load_audio_mono_16k(path: Path):
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=16000)
        sr = 16000
    wav = wav.squeeze(0).to(dtype=torch.float32, copy=False)
    return wav, int(sr)


def _chunk_1d(wav, sr: int, max_sec: float) -> list:
    if max_sec <= 0:
        return [wav]
    max_len = int(max_sec * sr)
    if wav.numel() <= max_len:
        return [wav]
    out = []
    for start in range(0, wav.numel(), max_len):
        out.append(wav[start : start + max_len])
    return out


@torch.no_grad()
def _extract_audio_embedding(
    *,
    model,
    processor,
    wav,
    sr: int,
    device: str,
    max_audio_sec: float,
) -> np.ndarray:
    chunks = _chunk_1d(wav, sr, max_audio_sec)
    embs = []
    for chunk in chunks:
        inputs = processor(
            chunk.cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.get("input_values")
        if input_values is None:
            raise ValueError("Audio processor did not return input_values")
        attn = inputs.get("attention_mask")
        input_values = input_values.to(device)
        attn = attn.to(device) if attn is not None else None

        out = model(input_values, attention_mask=attn)
        hidden = out.last_hidden_state
        feat_mask = _infer_feature_vector_attention_mask(model, attn, int(hidden.shape[1]))
        pooled = _masked_mean(hidden, feat_mask)
        embs.append(pooled.squeeze(0).detach().cpu())

    emb = torch.stack(embs, dim=0).mean(dim=0)
    return emb.numpy().astype(np.float32, copy=False)


@torch.no_grad()
def _extract_text_embedding(*, model, tokenizer, text: str, device: str, max_tokens: int, pool: str) -> np.ndarray:
    enc = tokenizer(
        text or "",
        return_tensors="pt",
        truncation=True,
        max_length=int(max_tokens),
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    hidden = out.last_hidden_state  # (1, seq, dim)

    if pool == "mean":
        mask = enc.get("attention_mask")
        pooled = _masked_mean(hidden, mask)
    else:
        pooled = hidden[:, 0, :]

    return pooled.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strong Audio+Text features using HuggingFace encoders.")
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("mer_dataset_builder/data/processed"),
        help="Path to mer_dataset_builder processed dir (contains meta_manifest.jsonl and audio/).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/features/mer_builder_at_hf"),
        help="Output feature root dir. Creates train/ val/ testA/ testB subdirs.",
    )
    p.add_argument(
        "--audio_model",
        type=str,
        default="microsoft/wavlm-base",
        help="HF audio encoder name or local path (outputs hidden states).",
    )
    p.add_argument(
        "--text_model",
        type=str,
        default="roberta-base",
        help="HF text encoder name or local path (outputs hidden states).",
    )
    p.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size (keep small on CPU).")
    p.add_argument("--max_audio_sec", type=float, default=30.0, help="Chunk long audio to <= this many seconds.")
    p.add_argument("--max_text_tokens", type=int, default=256)
    p.add_argument("--text_pool", choices=["cls", "mean"], default="cls")
    p.add_argument("--include_speaker_in_text", action="store_true", help="Prefix transcript with speaker_id.")
    p.add_argument(
        "--meld_context_window",
        type=int,
        default=0,
        help="For MELD only: prepend up to N previous utterances from the same dialogue (no future context).",
    )
    p.add_argument("--meld_context_sep", type=str, default=" [SEP] ", help="Separator used when building MELD context.")
    p.add_argument(
        "--failure_mode",
        choices=["zero", "skip"],
        default="zero",
        help="On failure: write zero vectors (zero) or skip sample (skip).",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, process only first N samples (debug).")
    p.add_argument("--overwrite", action="store_true")
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

    device = _resolve_device(args.device)
    logger.info("device=%s", device)

    # Lazy imports to keep help fast.
    from tqdm import tqdm
    from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor, AutoTokenizer

    try:
        # Some speech encoders (e.g., WavLM) don't ship a tokenizer, so AutoProcessor may fail.
        try:
            audio_processor = AutoProcessor.from_pretrained(args.audio_model)
        except Exception:
            audio_processor = AutoFeatureExtractor.from_pretrained(args.audio_model)
        audio_model = AutoModel.from_pretrained(args.audio_model)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load audio model {args.audio_model!r}. "
            "If you are offline, pass a local path to --audio_model."
        ) from exc

    try:
        text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        text_model = AutoModel.from_pretrained(args.text_model)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load text model {args.text_model!r}. "
            "If you are offline, pass a local path to --text_model."
        ) from exc

    audio_model.eval().to(device)
    text_model.eval().to(device)

    audio_dim = int(getattr(audio_model.config, "hidden_size", 0) or 0)
    text_dim = int(getattr(text_model.config, "hidden_size", 0) or 0)
    if audio_dim <= 0 or text_dim <= 0:
        raise ValueError(f"Could not infer dims (audio_dim={audio_dim}, text_dim={text_dim})")

    logger.info("audio_model=%s (dim=%d)", args.audio_model, audio_dim)
    logger.info("text_model=%s (dim=%d)", args.text_model, text_dim)

    counts_in: dict[str, int] = {}
    skipped_existing = 0
    skipped_unknown_split = 0
    processed = 0
    failures = 0

    failures_path = stats_dir / "feature_extraction_failures.jsonl"
    f_fail = failures_path.open("w", encoding="utf-8")
    try:
        batch: list[Sample] = []

        def _flush_batch() -> None:
            nonlocal processed, failures
            if not batch:
                return

            # Keep only items that still need processing.
            to_process: list[Sample] = []
            out_paths: list[Path] = []
            for sample in batch:
                out_path = out_dir / sample.split_group / f"{sample.id}.pkl"
                if out_path.exists() and not args.overwrite:
                    continue
                to_process.append(sample)
                out_paths.append(out_path)

            if not to_process:
                batch.clear()
                return

            # Text embeddings in batch.
            try:
                enc = text_tokenizer(
                    [s.transcript or "" for s in to_process],
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(args.max_text_tokens),
                    padding=True,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = text_model(**enc)
                    hidden = out.last_hidden_state
                    if str(args.text_pool) == "mean":
                        t_pooled = _masked_mean(hidden, enc.get("attention_mask"))
                    else:
                        t_pooled = hidden[:, 0, :]
                text_embs = t_pooled.detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception:
                text_embs = np.zeros((len(to_process), text_dim), dtype=np.float32)
                for i, s in enumerate(to_process):
                    try:
                        text_embs[i] = _extract_text_embedding(
                            model=text_model,
                            tokenizer=text_tokenizer,
                            text=s.transcript,
                            device=device,
                            max_tokens=int(args.max_text_tokens),
                            pool=str(args.text_pool),
                        )
                    except Exception as exc_i:
                        failures += 1
                        f_fail.write(
                            json.dumps({"id": s.id, "split": s.split_group, "error": f"text: {exc_i}"}, ensure_ascii=False)
                            + "\n"
                        )
                        if str(args.failure_mode) == "skip":
                            continue
                        text_embs[i] = np.zeros((text_dim,), dtype=np.float32)

            # Audio embeddings: batch short clips, chunk long clips.
            audio_embs = np.zeros((len(to_process), audio_dim), dtype=np.float32)
            short_wavs: list[np.ndarray] = []
            short_map: list[int] = []
            max_sec = float(args.max_audio_sec)

            for i, s in enumerate(to_process):
                audio_path = audio_dir / s.audio_relpath
                try:
                    wav, sr = _load_audio_mono_16k(audio_path)
                    if max_sec > 0 and wav.numel() > int(max_sec * sr):
                        audio_embs[i] = _extract_audio_embedding(
                            model=audio_model,
                            processor=audio_processor,
                            wav=wav,
                            sr=sr,
                            device=device,
                            max_audio_sec=max_sec,
                        )
                    else:
                        short_wavs.append(wav.cpu().numpy())
                        short_map.append(i)
                except Exception as exc:
                    failures += 1
                    f_fail.write(
                        json.dumps({"id": s.id, "split": s.split_group, "audio": s.audio_relpath, "error": str(exc)}, ensure_ascii=False) + "\n"
                    )
                    if str(args.failure_mode) == "skip":
                        continue
                    audio_embs[i] = np.zeros((audio_dim,), dtype=np.float32)

            if short_wavs:
                try:
                    inputs = audio_processor(
                        short_wavs,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                        return_attention_mask=True,
                    )
                    input_values = inputs.get("input_values")
                    if input_values is None:
                        raise ValueError("Audio processor did not return input_values")
                    attn = inputs.get("attention_mask")

                    input_values = input_values.to(device)
                    attn = attn.to(device) if attn is not None else None

                    with torch.no_grad():
                        out = audio_model(input_values, attention_mask=attn)
                        hidden = out.last_hidden_state
                        feat_mask = _infer_feature_vector_attention_mask(audio_model, attn, int(hidden.shape[1]))
                        pooled = _masked_mean(hidden, feat_mask)
                    pooled_np = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                    for idx, emb in zip(short_map, pooled_np, strict=False):
                        audio_embs[idx] = emb
                except Exception as exc:
                    # If batch audio fails, fall back to per-sample extraction.
                    for idx in short_map:
                        s = to_process[idx]
                        audio_path = audio_dir / s.audio_relpath
                        try:
                            wav, sr = _load_audio_mono_16k(audio_path)
                            audio_embs[idx] = _extract_audio_embedding(
                                model=audio_model,
                                processor=audio_processor,
                                wav=wav,
                                sr=sr,
                                device=device,
                                max_audio_sec=max_sec,
                            )
                        except Exception as exc_i:
                            failures += 1
                            f_fail.write(
                                json.dumps(
                                    {"id": s.id, "split": s.split_group, "audio": s.audio_relpath, "error": str(exc_i)},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            if str(args.failure_mode) == "skip":
                                continue
                            audio_embs[idx] = np.zeros((audio_dim,), dtype=np.float32)

            # Write pickles.
            for i, (s, out_path) in enumerate(zip(to_process, out_paths, strict=False)):
                label = EMOTION_TO_INDEX.get(s.emotion.strip().lower())
                if label is None:
                    failures += 1
                    f_fail.write(
                        json.dumps({"id": s.id, "split": s.split_group, "error": f"unknown emotion: {s.emotion!r}"}, ensure_ascii=False) + "\n"
                    )
                    if str(args.failure_mode) == "skip":
                        continue
                    label = 0

                payload: dict[str, Any] = {
                    "id": s.id,
                    "dataset": s.dataset,
                    "split": s.split_group,
                    "speaker_id": s.speaker_id,
                    "emotion": s.emotion,
                    "true_label": int(label),
                    "audio_features": audio_embs[i],
                    "text_features": text_embs[i],
                    "audio_model": str(args.audio_model),
                    "text_model": str(args.text_model),
                }

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                processed += 1

            batch.clear()

        total_rows = 0
        samples: list[Sample] = []
        for row in _read_manifest(manifest_path):
            total_rows += 1
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

        it = tqdm(samples, desc="extract hf features", unit="sample")
        for sample in it:
            if args.limit and processed >= int(args.limit):
                break

            out_path = out_dir / sample.split_group / f"{sample.id}.pkl"
            if out_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            batch.append(sample)
            if len(batch) >= int(args.batch_size):
                _flush_batch()
            it.set_postfix(ok=processed, fail=failures, skip=skipped_existing)

        if batch:
            _flush_batch()
            it.set_postfix(ok=processed, fail=failures, skip=skipped_existing)

    finally:
        f_fail.close()

    build_info = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "processed_dir": str(processed_dir),
        "manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "failure_mode": str(args.failure_mode),
        "overwrite": bool(args.overwrite),
        "feature_backend": "huggingface",
        "audio_model": str(args.audio_model),
        "text_model": str(args.text_model),
        "audio_dim": int(audio_dim),
        "text_dim": int(text_dim),
        "text_pool": str(args.text_pool),
        "max_audio_sec": float(args.max_audio_sec),
        "max_text_tokens": int(args.max_text_tokens),
        "include_speaker_in_text": bool(args.include_speaker_in_text),
        "meld_context_window": int(args.meld_context_window),
        "meld_context_sep": str(args.meld_context_sep),
        "class_names": CLASS_NAMES,
        "counts_by_split": counts_in,
        "processed_ok": int(processed),
        "processed_failed": int(failures),
        "skipped_existing": int(skipped_existing),
        "skipped_unknown_split": int(skipped_unknown_split),
        "rows_seen": int(total_rows),
    }
    (stats_dir / "build_info.json").write_text(json.dumps(build_info, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("done ok=%d failed=%d skipped_existing=%d", processed, failures, skipped_existing)
    if failures:
        logger.warning("Some samples failed; see %s", failures_path)

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
