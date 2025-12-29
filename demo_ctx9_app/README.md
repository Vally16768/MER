# Demo UI: Audio + Text → Emotion

This is a small, standalone demo UI that:
- records (or loads) an audio clip,
- accepts text + optional dialogue history,
- auto-selects the best HF end-to-end checkpoint from your training runs,
- outputs the predicted emotion + class probabilities.

## Run

From `MER/`:

```powershell
python demo_ctx9_app\app.py
```

Optional (if you want microphone recording inside the UI):

```powershell
python -m pip install sounddevice
```

## Model selection

At startup, the app scans:
- `outputs_paper/`
- `outputs/`

It selects the run with the highest **MELD TestB** `weighted_f1` from:
- `*/eval_testB/metrics_eval.json` (preferred), or
- `*/eval_meld_testB/metrics_eval.json` (fallback),

and always loads that run’s `checkpoints/best.pt`.

## Audio controls

- **Start Recording / Stop Recording** records from the default microphone (requires `sounddevice`).
- **Play / Stop** plays the currently loaded/recorded audio.
- The waveform view shows the audio amplitude over time and a playhead during playback.
