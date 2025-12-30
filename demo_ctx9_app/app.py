from __future__ import annotations

import json
import sys
import threading
import time
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


CLASS_NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / np.sum(e)).astype(np.float64, copy=False)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if sr == 16000:
        return wav
    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")
    # Lightweight fallback resampler (linear interpolation).
    # For best quality, provide audio already at 16kHz mono.
    duration = wav.shape[0] / float(sr)
    n_out = max(1, int(round(duration * 16000.0)))
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float64)
    out = np.interp(x_new, x_old, wav.astype(np.float64)).astype(np.float32)
    return out


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported in this demo (sampwidth=2).")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    elif n_channels != 1:
        raise ValueError(f"Unsupported channel count: {n_channels}")

    return audio, int(framerate)


def _write_wav_16bit(path: Path, wav: np.ndarray, sr: int) -> None:
    wav = np.asarray(wav, dtype=np.float32)
    wav_i16 = np.clip(wav * 32768.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(wav_i16.tobytes())


def _parse_history_lines(text: str) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        speaker: str | None = None
        utter = line
        if "\t" in line:
            left, right = line.split("\t", 1)
            speaker = left.strip() or None
            utter = right.strip()
        elif ":" in line:
            left, right = line.split(":", 1)
            # Heuristic: short prefix treated as speaker.
            if 1 <= len(left.strip()) <= 24:
                speaker = left.strip() or None
                utter = right.strip()
        out.append((speaker, utter))
    return out


def _build_meld_text(
    *,
    speaker_id: str,
    history_text: str,
    current_text: str,
    context_window: int = 9,
    sep: str = " </s> ",
    include_speaker: bool = True,
) -> str:
    speaker_id = str(speaker_id or "").strip()
    history = _parse_history_lines(history_text)
    history = history[-int(context_window) :] if context_window > 0 else []

    def fmt(spk: str | None, utt: str) -> str:
        utt = str(utt or "").strip()
        if not utt:
            return ""
        if not include_speaker:
            return utt
        spk_final = (spk or speaker_id or "").strip()
        return (f"speaker={spk_final} {utt}").strip() if spk_final else utt

    parts: list[str] = []
    for spk, utt in history:
        p = fmt(spk, utt)
        if p:
            parts.append(p)

    cur = fmt(speaker_id or None, current_text)
    if cur:
        parts.append(cur)
    return sep.join(parts).strip()


@dataclass(frozen=True)
class DemoConfig:
    ckpt_path: Path
    audio_model: str = "microsoft/wavlm-base"
    text_model: str = "roberta-base"
    pool_audio: str = "mean_std"
    pool_text: str = "cls"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_mult: int = 4
    freeze_audio_feature_encoder: bool = True
    text_max_tokens: int = 320
    max_audio_sec: float = 20.0
    context_window: int = 9
    context_sep: str = " </s> "
    include_speaker_in_text: bool = True


class Predictor:
    def __init__(self, cfg: DemoConfig, *, device: str = "cpu") -> None:
        import torch
        import warnings
        from transformers import AutoFeatureExtractor, AutoTokenizer  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore

        from src.models.hf_at import HFAudioTextModel

        self.cfg = cfg
        self.device = torch.device(device)

        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*clean_up_tokenization_spaces.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_only=False.*")
        hf_logging.set_verbosity_error()

        # Feature extractor for WavLM-family models (AutoProcessor may require a tokenizer; avoid that).
        self.audio_fx = AutoFeatureExtractor.from_pretrained(cfg.audio_model)
        self.tok = AutoTokenizer.from_pretrained(cfg.text_model, use_fast=True)

        ckpt_obj = torch.load(cfg.ckpt_path, map_location="cpu")
        if not isinstance(ckpt_obj, dict) or "model_state" not in ckpt_obj:
            raise ValueError("Unsupported checkpoint format; expected dict with key 'model_state'.")

        self.model = HFAudioTextModel(
            audio_model=cfg.audio_model,
            text_model=cfg.text_model,
            n_classes=len(CLASS_NAMES),
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            ffn_mult=cfg.ffn_mult,
            pool_audio=cfg.pool_audio,
            pool_text=cfg.pool_text,
            freeze_audio=False,
            freeze_text=False,
            freeze_audio_feature_encoder=cfg.freeze_audio_feature_encoder,
            gradient_checkpointing=False,
            modalities=("A", "T"),
            modality_dropout_p=0.0,
        )
        self.model.load_state_dict(ckpt_obj["model_state"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        *,
        wav: np.ndarray,
        sr: int,
        speaker_id: str,
        history_text: str,
        current_text: str,
    ) -> dict[str, Any]:
        import torch

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1:
            wav = wav.reshape(-1)

        wav = _resample_to_16k(wav, int(sr))
        max_len = int(self.cfg.max_audio_sec * 16000)
        if max_len > 0 and wav.shape[0] > max_len:
            wav = wav[:max_len]

        text = _build_meld_text(
            speaker_id=speaker_id,
            history_text=history_text,
            current_text=current_text,
            context_window=self.cfg.context_window,
            sep=self.cfg.context_sep,
            include_speaker=self.cfg.include_speaker_in_text,
        )

        if not text:
            raise ValueError("Empty text input.")

        audio_inputs = self.audio_fx([wav], sampling_rate=16000, return_tensors="pt", padding=True)
        text_inputs = self.tok(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.cfg.text_max_tokens),
        )

        batch: dict[str, Any] = {
            "audio_input_values": audio_inputs["input_values"].to(self.device),
            "audio_attention_mask": audio_inputs.get("attention_mask", None),
            "text_input_ids": text_inputs["input_ids"].to(self.device),
            "text_attention_mask": text_inputs.get("attention_mask", None),
        }
        if batch["audio_attention_mask"] is not None:
            batch["audio_attention_mask"] = batch["audio_attention_mask"].to(self.device)
        if batch["text_attention_mask"] is not None:
            batch["text_attention_mask"] = batch["text_attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0].astype(np.float64)

        pred_idx = int(np.argmax(probs))
        pred = CLASS_NAMES[pred_idx]
        return {
            "pred": pred,
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        }


def _try_import_sounddevice():
    try:
        import sounddevice as sd  # type: ignore

        return sd
    except Exception:
        return None


@dataclass(frozen=True)
class SelectedRun:
    run_dir: Path
    ckpt_path: Path
    metrics_path: Path
    wf1: float
    uar: float
    acc: float


def _parse_yaml_scalar(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    low = raw.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none", "~"}:
        return None
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        return int(raw)
    except Exception:
        pass
    try:
        return float(raw)
    except Exception:
        pass
    return raw


def _load_yaml_scalars(path: Path) -> dict[str, Any]:
    """
    Minimal YAML parser for our resolved configs.

    - Supports nested dicts via indentation.
    - Captures only scalar `key: value` pairs (ignores lists/complex types).
    - Avoids PyYAML dependency so the demo works in minimal environments.
    """
    out: dict[str, Any] = {}
    stack: list[str] = []
    indents: list[int] = [-1]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while len(stack) > 0 and indent <= indents[-1]:
            stack.pop()
            indents.pop()

        if line.startswith("- "):
            continue
        if ":" not in line:
            continue

        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if not key:
            continue

        if rest == "":
            stack.append(key)
            indents.append(indent)
            continue

        dotted = ".".join([*stack, key])
        out[dotted] = _parse_yaml_scalar(rest)

    return out


def _read_metrics(metrics_path: Path) -> tuple[float, float, float]:
    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    m = d.get("metrics", d)
    wf1 = float(m.get("weighted_f1", m.get("wf1", -1.0)))
    uar = float(m.get("uar", -1.0))
    acc = float(m.get("accuracy", m.get("acc", -1.0)))
    return wf1, uar, acc


def _iter_testb_metrics_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    files.extend(sorted(root.glob("*/eval_testB/metrics_eval.json")))
    files.extend(sorted(root.glob("*/eval_meld_testB/metrics_eval.json")))
    return files


def _run_is_hf_e2e(run_dir: Path) -> bool:
    cfg_path = run_dir / "config_resolved.yaml"
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not (cfg_path.exists() and ckpt_path.exists()):
        return False
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return ("audio_model:" in text) and ("text_model:" in text)


def _select_best_run() -> SelectedRun:
    """
    Select the best available HF end-to-end run across `outputs_paper/` and `outputs/`.

    Primary criterion: maximize TestB `weighted_f1`.
    Tie-breakers: `uar`, then `accuracy`.
    """
    candidates: list[SelectedRun] = []
    for root in (REPO_ROOT / "outputs_paper", REPO_ROOT / "outputs"):
        for metrics_path in _iter_testb_metrics_files(root):
            run_dir = metrics_path.parent.parent
            if not _run_is_hf_e2e(run_dir):
                continue
            ckpt_path = run_dir / "checkpoints" / "best.pt"
            try:
                wf1, uar, acc = _read_metrics(metrics_path)
            except Exception:
                continue
            if wf1 < 0:
                continue
            candidates.append(
                SelectedRun(
                    run_dir=run_dir,
                    ckpt_path=ckpt_path,
                    metrics_path=metrics_path,
                    wf1=wf1,
                    uar=uar,
                    acc=acc,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "No suitable run found. Expected at least one of:\n"
            "  outputs/*/eval_testB/metrics_eval.json\n"
            "  outputs_paper/*/eval_testB/metrics_eval.json\n"
            "and a corresponding checkpoints/best.pt + config_resolved.yaml."
        )

    candidates.sort(key=lambda r: (r.wf1, r.uar, r.acc), reverse=True)
    return candidates[0]


def _demo_config_from_run(run: SelectedRun) -> DemoConfig:
    cfg_path = run.run_dir / "config_resolved.yaml"
    scalars = _load_yaml_scalars(cfg_path)

    def get(key: str, default: Any) -> Any:
        return scalars.get(key, default)

    # Model
    audio_model = str(get("model.audio_model", "microsoft/wavlm-base"))
    text_model = str(get("model.text_model", "roberta-base"))
    pool_audio = str(get("model.pool_audio", "mean_std"))
    pool_text = str(get("model.pool_text", "cls"))
    hidden_dim = int(get("model.hidden_dim", 256))
    num_layers = int(get("model.num_layers", 4))
    num_heads = int(get("model.num_heads", 8))
    ffn_mult = int(get("model.ffn_mult", 4))
    freeze_audio_feature_encoder = bool(get("model.freeze_audio_feature_encoder", True))

    # Data/context
    text_max_tokens = int(get("data.text_max_tokens", get("data.text_max_len", 320)))
    max_audio_sec = float(get("data.max_audio_sec", 20.0))
    context_window = int(get("data.meld_context_window", 9))
    context_sep = str(get("data.meld_context_sep", " </s> "))
    include_speaker_in_text = bool(get("data.include_speaker_in_text", True))

    return DemoConfig(
        ckpt_path=run.ckpt_path,
        audio_model=audio_model,
        text_model=text_model,
        pool_audio=pool_audio,
        pool_text=pool_text,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        freeze_audio_feature_encoder=freeze_audio_feature_encoder,
        text_max_tokens=text_max_tokens,
        max_audio_sec=max_audio_sec,
        context_window=context_window,
        context_sep=context_sep,
        include_speaker_in_text=include_speaker_in_text,
    )


def main() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    selected = _select_best_run()
    default_ckpt = selected.ckpt_path
    default_cfg = _demo_config_from_run(selected)

    state = {
        "audio_path": None,
        "audio_wav": None,
        "audio_sr": None,
        "predictor": None,
        "busy": False,
        "is_recording": False,
        "record_stream": None,
        "record_frames": [],
        "record_t0": None,
        "record_timer_id": None,
        "record_sr": None,
        "is_playing": False,
        "play_t0": None,
        "play_timer_id": None,
        "play_tmp_wav": None,
        "play_backend": None,  # "sounddevice" | "winmm"
    }

    sd = _try_import_sounddevice()

    root = tk.Tk()
    root.title(f"MER Demo — {selected.run_dir.name} (TestB wf1={selected.wf1:.4f}, uar={selected.uar:.4f})")
    root.geometry("900x650")

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    # Speaker
    speaker_var = tk.StringVar(value="MELD_SPK")
    ttk.Label(frm, text="Speaker ID:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=speaker_var, width=30).grid(row=0, column=1, sticky="w")

    # History + current text
    ttk.Label(frm, text="Dialogue history (optional). One utterance per line. Formats: `spk\\ttext` or `spk: text`").grid(
        row=1, column=0, columnspan=3, sticky="w"
    )
    history = tk.Text(frm, height=8, width=110)
    history.grid(row=2, column=0, columnspan=3, sticky="nsew")

    ttk.Label(frm, text="Current utterance text:").grid(row=3, column=0, columnspan=3, sticky="w")
    current = tk.Text(frm, height=4, width=110)
    current.grid(row=4, column=0, columnspan=3, sticky="nsew")

    # Audio controls
    audio_status = tk.StringVar(value="No audio loaded/recorded.")
    ttk.Label(frm, textvariable=audio_status).grid(row=5, column=0, columnspan=3, sticky="w")

    wave_status = tk.StringVar(value="Duration: —")
    ttk.Label(frm, textvariable=wave_status).grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 0))

    audio_level = tk.StringVar(value="Level: —")
    ttk.Label(frm, textvariable=audio_level).grid(row=6, column=2, sticky="e", pady=(2, 0))

    wave_canvas = tk.Canvas(frm, height=120, background="#111111", highlightthickness=1, highlightbackground="#333333")
    wave_canvas.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def _winmm_send(cmd: str) -> None:
        import ctypes

        winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
        buf = ctypes.create_unicode_buffer(256)
        rc = winmm.mciSendStringW(cmd, buf, len(buf), 0)
        if rc != 0:
            err = ctypes.create_unicode_buffer(256)
            try:
                winmm.mciGetErrorStringW(rc, err, len(err))
                msg = err.value.strip() or f"mci error code={rc}"
            except Exception:
                msg = f"mci error code={rc}"
            raise RuntimeError(f"WinMM failed: {cmd!r} -> {msg}")

    def _set_audio(wav: np.ndarray, sr: int, *, source_path: str | None) -> None:
        state["audio_path"] = source_path
        state["audio_wav"] = np.asarray(wav, dtype=np.float32).reshape(-1)
        state["audio_sr"] = int(sr)
        dur = 0.0 if state["audio_sr"] <= 0 else (state["audio_wav"].shape[0] / state["audio_sr"])
        wave_status.set(f"Duration: {dur:.2f}s (sr={state['audio_sr']})")
        if state["audio_wav"].size:
            rms = float(np.sqrt(np.mean(np.square(state["audio_wav"]))))
            peak = float(np.max(np.abs(state["audio_wav"])))
            audio_level.set(f"Level: rms={rms:.4f} peak={peak:.4f}")
        else:
            audio_level.set("Level: —")
        _draw_waveform()

    def _draw_waveform(*, play_progress: float | None = None) -> None:
        wave_canvas.delete("all")
        w = max(1, int(wave_canvas.winfo_width()))
        h = max(1, int(wave_canvas.winfo_height()))
        mid = h // 2

        wav = state["audio_wav"]
        sr = state["audio_sr"]
        if wav is None or sr is None or wav.size == 0:
            wave_canvas.create_text(10, 10, anchor="nw", fill="#bbbbbb", text="No audio loaded/recorded.")
            return

        wave_canvas.create_line(0, mid, w, mid, fill="#333333")

        # Downsample to canvas width and draw envelope (min/max per bin).
        wav = np.asarray(wav, dtype=np.float32)
        n = wav.shape[0]
        step = max(1, n // w)
        xs = range(0, n, step)
        if len(list(xs)) <= 1:
            xs = [0, n - 1]

        # Efficient envelope: slice bins by step.
        x_pix = 0
        for i in range(0, n, step):
            seg = wav[i : min(n, i + step)]
            if seg.size == 0:
                continue
            lo = float(np.min(seg))
            hi = float(np.max(seg))
            y1 = mid - int(hi * (h * 0.45))
            y2 = mid - int(lo * (h * 0.45))
            wave_canvas.create_line(x_pix, y1, x_pix, y2, fill="#4aa3ff")
            x_pix += 1
            if x_pix >= w:
                break

        # Optional playhead.
        if play_progress is not None:
            p = float(play_progress)
            p = 0.0 if p < 0 else 1.0 if p > 1 else p
            x = int(p * (w - 1))
            wave_canvas.create_line(x, 0, x, h, fill="#ffb000")

    def _stop_playback() -> None:
        if not state["is_playing"]:
            return
        state["is_playing"] = False
        if state["play_timer_id"] is not None:
            try:
                root.after_cancel(state["play_timer_id"])
            except Exception:
                pass
            state["play_timer_id"] = None
        try:
            if state.get("play_backend") == "sounddevice" and sd is not None:
                sd.stop()
            elif state.get("play_backend") == "winmm":
                try:
                    _winmm_send("stop mer_demo_audio")
                finally:
                    _winmm_send("close mer_demo_audio")
            else:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        state["play_backend"] = None
        _draw_waveform(play_progress=None)

    def _start_playback() -> None:
        wav = state["audio_wav"]
        sr = state["audio_sr"]
        if wav is None or sr is None or wav.size == 0:
            messagebox.showwarning("Missing audio", "Load a WAV file or record audio first.")
            return

        _stop_playback()

        wav16 = _resample_to_16k(wav, int(sr))
        dur = wav16.shape[0] / 16000.0
        state["is_playing"] = True
        state["play_t0"] = time.perf_counter()

        def _tick():
            if not state["is_playing"]:
                return
            t = time.perf_counter() - float(state["play_t0"] or 0.0)
            p = 0.0 if dur <= 0 else (t / dur)
            if p >= 1.0:
                _stop_playback()
                return
            _draw_waveform(play_progress=p)
            state["play_timer_id"] = root.after(50, _tick)

        def _play_job():
            try:
                # Prefer sounddevice if available; otherwise use WinMM (Windows MCI).
                if sd is not None:
                    try:
                        state["play_backend"] = "sounddevice"
                        sd.play(wav16, samplerate=16000)
                        sd.wait()
                        return
                    except Exception:
                        state["play_backend"] = None

                # Use a unique temp file per playback to avoid file-lock issues.
                with tempfile.NamedTemporaryFile(prefix="mer_demo_", suffix=".wav", delete=False) as f:
                    tmp_path = Path(f.name)
                _write_wav_16bit(tmp_path, wav16, 16000)
                state["play_tmp_wav"] = str(tmp_path)

                # WinMM playback (more reliable than winsound on some Windows setups).
                state["play_backend"] = "winmm"
                _winmm_send("close mer_demo_audio")
                _winmm_send(f'open "{str(tmp_path)}" type waveaudio alias mer_demo_audio')
                _winmm_send("play mer_demo_audio")
            except Exception as exc:
                root.after(0, lambda: messagebox.showerror("Playback error", str(exc)))
            finally:
                # Ensure UI stops (even if playback backend doesn't notify).
                if state["is_playing"]:
                    root.after(0, _stop_playback)

        threading.Thread(target=_play_job, daemon=True).start()
        _tick()

    def pick_audio():
        p = filedialog.askopenfilename(title="Select WAV", filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if not p:
            return
        try:
            wav, sr = _read_wav(Path(p))
        except Exception as exc:
            messagebox.showerror("Audio error", str(exc))
            return
        _set_audio(wav, sr, source_path=p)
        audio_status.set(f"Loaded WAV: {p} (sr={sr}, sec={wav.shape[0]/sr:.2f})")

    device_row = ttk.Frame(frm)
    device_row.grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))

    input_dev_var = tk.StringVar(value="default")
    input_devices: list[tuple[str, int | None]] = [("default", None)]
    if sd is not None:
        try:
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                try:
                    if int(d.get("max_input_channels", 0)) <= 0:
                        continue
                    name = str(d.get("name", f"dev{idx}"))
                    input_devices.append((f"{idx}: {name}", idx))
                except Exception:
                    continue
        except Exception:
            pass

    ttk.Label(device_row, text="Input device:").pack(side="left")
    input_dev_combo = ttk.Combobox(
        device_row,
        textvariable=input_dev_var,
        values=[label for label, _ in input_devices],
        width=55,
        state=("readonly" if len(input_devices) > 1 else "disabled"),
    )
    input_dev_combo.current(0)
    input_dev_combo.pack(side="left", padx=(6, 0))

    def _get_selected_input_device():
        sel = input_dev_var.get()
        for label, idx in input_devices:
            if label == sel:
                return idx
        return None

    def _stop_recording() -> None:
        if not state["is_recording"]:
            return
        state["is_recording"] = False
        if state["record_timer_id"] is not None:
            try:
                root.after_cancel(state["record_timer_id"])
            except Exception:
                pass
            state["record_timer_id"] = None
        stream = state["record_stream"]
        state["record_stream"] = None
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:
            pass
        frames = state["record_frames"] or []
        sr = int(state.get("record_sr") or 16000)
        state["record_sr"] = None
        state["record_frames"] = []
        if frames:
            wav = np.concatenate(frames, axis=0).astype(np.float32, copy=False)
            _set_audio(wav, sr, source_path=None)
            audio_status.set(f"Recorded audio (sr={sr}, sec={wav.shape[0]/float(sr):.2f}).")
        else:
            audio_status.set("Recording stopped (no audio captured).")

    def _start_recording() -> None:
        if sd is None:
            messagebox.showwarning(
                "Recording unavailable",
                "Microphone recording requires `sounddevice`. Install it and restart the app.",
            )
            return
        if state["is_recording"]:
            return
        _stop_playback()
        state["record_frames"] = []
        state["is_recording"] = True
        state["record_t0"] = time.perf_counter()
        audio_status.set("Recording… (press Stop Recording to finish)")

        input_device = _get_selected_input_device()
        try:
            if input_device is None:
                in_sr = int(getattr(sd.default, "samplerate", 16000) or 16000)
            else:
                in_sr = int(sd.query_devices(input_device, "input").get("default_samplerate", 16000) or 16000)
        except Exception:
            in_sr = 16000
        # Use device-native sample rate for cleaner capture; the model resamples to 16k internally.
        state["record_sr"] = in_sr

        def _callback(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
            if status:
                # Don't spam message boxes from audio thread; keep UI stable.
                pass
            if not state["is_recording"]:
                return
            state["record_frames"].append(np.asarray(indata, dtype=np.float32).reshape(-1).copy())

        stream = sd.InputStream(
            samplerate=in_sr,
            device=input_device,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        state["record_stream"] = stream
        stream.start()

        def _tick():
            if not state["is_recording"]:
                return
            t = time.perf_counter() - float(state["record_t0"] or 0.0)
            wave_status.set(f"Recording… {t:.1f}s (sr={in_sr})")
            # Show a simple live peak estimate from the last chunk.
            try:
                frames = state["record_frames"]
                if frames:
                    last = frames[-1]
                    peak = float(np.max(np.abs(np.asarray(last, dtype=np.float32))))
                    audio_level.set(f"Level: peak={peak:.4f}")
            except Exception:
                pass
            state["record_timer_id"] = root.after(100, _tick)

        _tick()

    btn_row = ttk.Frame(frm)
    btn_row.grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Button(btn_row, text="Load WAV", command=pick_audio).pack(side="left")

    rec_btn = ttk.Button(
        btn_row,
        text="Start Recording",
        command=_start_recording,
        state=("normal" if sd is not None else "disabled"),
    )
    rec_btn.pack(side="left", padx=(10, 0))

    stop_rec_btn = ttk.Button(
        btn_row,
        text="Stop Recording",
        command=_stop_recording,
        state=("normal" if sd is not None else "disabled"),
    )
    stop_rec_btn.pack(side="left", padx=(6, 0))

    ttk.Button(btn_row, text="Play", command=_start_playback).pack(side="left", padx=(10, 0))
    ttk.Button(btn_row, text="Stop", command=_stop_playback).pack(side="left", padx=(6, 0))

    # Output
    out_pred = tk.StringVar(value="—")
    ttk.Label(frm, text="Prediction:").grid(row=10, column=0, sticky="w", pady=(10, 0))
    ttk.Label(frm, textvariable=out_pred, font=("Segoe UI", 14, "bold")).grid(row=10, column=1, sticky="w", pady=(10, 0))

    probs_box = tk.Text(frm, height=10, width=110)
    probs_box.grid(row=11, column=0, columnspan=3, sticky="nsew")

    def set_probs(probs: dict[str, float]) -> None:
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        probs_box.delete("1.0", "end")
        for k, v in items:
            probs_box.insert("end", f"{k:9s}  {v:.4f}\n")

    def ensure_predictor() -> Predictor:
        if state["predictor"] is not None:
            return state["predictor"]
        if not default_ckpt.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {default_ckpt}")
        pred = Predictor(default_cfg, device="cuda" if _torch_cuda_available() else "cpu")
        state["predictor"] = pred
        return pred

    def _torch_cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def run_infer():
        if state["busy"]:
            return
        if state["audio_wav"] is None or state["audio_sr"] is None:
            messagebox.showwarning("Missing audio", "Load a WAV file or record audio first.")
            return
        cur_text = current.get("1.0", "end").strip()
        if not cur_text:
            messagebox.showwarning("Missing text", "Enter the current utterance text.")
            return

        state["busy"] = True
        out_pred.set("Running...")
        probs_box.delete("1.0", "end")

        def _job():
            try:
                pred = ensure_predictor()
                res = pred.predict(
                    wav=state["audio_wav"],
                    sr=int(state["audio_sr"]),
                    speaker_id=speaker_var.get(),
                    history_text=history.get("1.0", "end").strip(),
                    current_text=cur_text,
                )
                out_pred.set(res["pred"])
                set_probs(res["probs"])
            except Exception as exc:
                out_pred.set("Error")
                messagebox.showerror("Inference error", str(exc))
            finally:
                state["busy"] = False

        threading.Thread(target=_job, daemon=True).start()

    ttk.Button(frm, text="Infer emotion", command=run_infer).grid(row=12, column=0, sticky="w", pady=(10, 0))

    # Layout weights
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(2, weight=1)
    frm.rowconfigure(4, weight=1)
    frm.rowconfigure(11, weight=1)

    def _on_canvas_resize(event):  # type: ignore[no-untyped-def]
        _draw_waveform(play_progress=None)

    wave_canvas.bind("<Configure>", _on_canvas_resize)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
