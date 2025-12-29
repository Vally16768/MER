# Baseline Report — MER Audio+Text (MFCC + Hashing)

This report documents a reproducible baseline for the **`mer_dataset_builder` → MFCC+hash features → A+T model** pipeline.

## Environment (this machine)

- Python: 3.12.10
- OS: Windows 11 (10.0.22631)
- CPU: 28 logical cores
- PyTorch: 2.5.1+cu121 (CUDA 12.1)
- GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU

## Baseline: commands (train + evaluate)

Assumptions:

- You already ran `mer_dataset_builder` and have `mer_dataset_builder/data/processed/meta_manifest.jsonl` + `audio/`.
- You are in the repo root: `MER/`
- Use the venv interpreter: `.\.venv\Scripts\python.exe` (Windows).

### 0) Feature extraction (idempotent)

This creates `data/features/mer_builder_at_simple/{train,val,testA,testB}`.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe extract_at_features.py `
  --processed_dir mer_dataset_builder/data/processed `
  --out_dir data/features/mer_builder_at_simple `
  --n_mfcc 80 `
  --num_workers 16
```

Linux/macOS:

```bash
./.venv/bin/python extract_at_features.py \
  --processed_dir mer_dataset_builder/data/processed \
  --out_dir data/features/mer_builder_at_simple \
  --n_mfcc 80 \
  --num_workers 16
```

### 1) Train baseline (FlexibleAT)

```powershell
.\.venv\Scripts\python.exe train.py `
  --config configs/mer_builder_at_simple.yaml `
  --modalities A T `
  --run_name baseline_mfcc80_bs128
```

### 2) Evaluate on TestA (acted) vs TestB (MELD)

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --config outputs/baseline_mfcc80_bs128/config_resolved.yaml `
  --ckpt outputs/baseline_mfcc80_bs128/checkpoints/best.pt `
  --set data.eval_dir=data/features/mer_builder_at_simple/testA `
  --output_dir outputs/baseline_mfcc80_bs128/eval_testA

.\.venv\Scripts\python.exe evaluate.py `
  --config outputs/baseline_mfcc80_bs128/config_resolved.yaml `
  --ckpt outputs/baseline_mfcc80_bs128/checkpoints/best.pt `
  --set data.eval_dir=data/features/mer_builder_at_simple/testB `
  --output_dir outputs/baseline_mfcc80_bs128/eval_testB
```

### Single-command entrypoint (Windows)

To run the whole baseline as a single command, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline_at.ps1 -RunName baseline_mfcc80_bs128
```

## Baseline metrics (recorded in this repo)

These are the metrics already present under:

- TestA: `outputs/mfcc80_bs128_testA/metrics_eval.json`
- TestB: `outputs/mfcc80_bs128/metrics_eval.json`

| set | accuracy | macro_f1 | wf1 | uar |
|---|---:|---:|---:|---:|
| TestA (acted) | 0.7233 | 0.7142 | 0.7244 | 0.7196 |
| TestB (MELD) | 0.4544 | 0.2369 | 0.4232 | 0.2377 |

## Split protocol verification (leakage checks)

Using the unified manifest (`mer_dataset_builder/data/processed/meta_manifest.jsonl`):

- `testB` contains only `MELD` samples (official split preserved).
- Acted datasets are **speaker-disjoint** between `train`, `val`, and `testA` (overlap count = 0 per dataset).

If you want to re-check locally:

```powershell
.\.venv\Scripts\python.exe scripts/check_splits.py --manifest mer_dataset_builder/data/processed/meta_manifest.jsonl
```
