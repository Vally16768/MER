# MER (Audio+Text) — Architecture & Data Flow

This document describes the **audio+text** pipeline used with `mer_dataset_builder`, and the two A/T models trained via `train.py` / evaluated via `evaluate.py`.

## Repo map (key folders)

```
MER/
  configs/                         # YAML configs for A+T experiments
  mer_dataset_builder/             # Multi-corpus dataset builder + unified manifest
  src/
    data/                          # Feature dataset loaders
    models/                        # A+T models (Flexible / Robust)
    augmentation.py                # Train-time feature augmentation (modality dropout, noise)
    ema.py                         # Exponential Moving Average weights
    train_loop.py                  # Train/val loop + early stopping + scheduler stepping
    eval.py                        # Evaluation loop (no training)
    metrics.py                     # Accuracy / wF1 / UAR (+ confusion matrix)
  extract_at_features.py           # Build MFCC+hash text features from mer_dataset_builder manifest
  train.py                         # Train A/T model from feature pickles
  evaluate.py                      # Evaluate checkpoint on a chosen split directory
```

## Datasets supported (current repo)

Via `MER/mer_dataset_builder`, the unified manifest can include:

- `MEAD` (Part0 speech)
- `MELD` (Raw audio + official CSVs)
- `RAVDESS` (speech only)
- `CREMA-D`
- `ESD`
- `EmoV-DB` (OpenSLR SLR115)
- `IEMOCAP` (full release)

This repo does **not** include builders/parsers for MSP-Podcast or CMU-MOSEI.

## End-to-end data flow

1) `mer_dataset_builder` creates:
   - `mer_dataset_builder/data/processed/meta_manifest.jsonl`
   - `mer_dataset_builder/data/processed/audio/<DATASET>/*.wav` (normalized audio)

2) `extract_at_features.py` reads the manifest and writes **per-sample feature pickles**:
   - `data/features/<feature_set>/{train,val,testA,testB}/*.pkl`

3) `train.py` trains on:
   - `data.features.train_dir` → train loader
   - `data.features.val_dir` → val loader
   - checkpoints + logs → `outputs/<run_name>/`

4) `evaluate.py` evaluates a checkpoint on a directory:
   - `data.eval_dir` (or `data.val_dir` fallback)
   - metrics + confusion matrix → `--output_dir` (recommended) or inferred run dir

### Optional end-to-end (HuggingFace) pipeline

For stronger performance (especially on MELD), you can fine-tune pretrained encoders directly from the unified manifest:

- `train_hf_e2e.py`: loads audio + text from `mer_dataset_builder/data/processed/` and trains `HFAudioTextModel`
- `evaluate_hf_e2e.py`: evaluates checkpoints on manifest splits (e.g., `testB`)

This path does not require `extract_at_features*.py`, but it requires HuggingFace models to be available (downloaded or local paths).

## Splits: TestA vs TestB

`extract_at_features.py` groups manifest `split` values into four directories:

- `train`: `{train, meld_train}`
- `val`: `{val, meld_dev}`
- `testA`: `{testA, test}` (acted evaluation)
- `testB`: `{testB, meld_test}` (MELD conversational evaluation)

## Feature schema (what each `*.pkl` contains)

Each feature file is a Python `dict` with:

- `id`: global id from the unified manifest
- `dataset`: dataset name (e.g., `MELD`, `MEAD`, `IEMOCAP`, …)
- `split`: split-group (`train|val|testA|testB`)
- `speaker_id`: speaker identifier (for disjointness checks)
- `emotion`: canonical label string (one of 7 classes)
- `true_label`: integer label in `[0..6]`
- `audio_features`: `float32` vector, shape `(2*n_mfcc,)`
- `text_features`: `float32` vector, shape `(text_dim,)`
- optional `notes`: semicolon-separated notes for recoveries/failures

### Label order (must be consistent everywhere)

The label index mapping is fixed as:

`["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]`

See:
- `MER/extract_at_features.py` (`CLASS_NAMES`)
- `MER/configs/*.yaml` (`dataset.class_names`)

## Feature extraction details (`extract_at_features.py`)

### Audio → MFCC summary

Input audio is read via `torchaudio` and resampled to 16 kHz mono if needed. MFCC features are computed as:

- MFCC tensor: `(1, n_mfcc, frames)`
- Mean over time: `(n_mfcc,)`
- Std over time: `(n_mfcc,)`
- Concatenate: `audio_features = [mean; std]` → `(2*n_mfcc,)`

So for `n_mfcc=80`, `input_dim_a=160`.

### Text → hashing features

Text uses `sklearn.feature_extraction.text.HashingVectorizer` with:

- output dimension `text_dim` (default `768`)
- lowercase enabled
- `alternate_sign=False`, `norm=None`

## Models

Both models take a batch dict:

- `batch["audio"]`: `(B, D_a)` or `None`
- `batch["text"]`: `(B, D_t)` or `None`

and return logits `(B, num_classes)`.

### 1) `FlexibleATModel` (`src/models/flexible_at.py`)

Modes:

- **AT**: uses gated fusion (`GFL`) between text and audio vectors.
- **A only / T only**: projects the active modality to `gated_dim`, bypassing fusion.

Shapes (AT path):

```
audio: (B, D_a) -> LayerNorm -> Linear -> (B, D_a)
text:  (B, D_t) -> LayerNorm -> Linear -> (B, D_t)
GFL(text, audio) -> (B, gated_dim)
LayerNorm -> SwiGLU -> classifier -> logits: (B, C)
```

### 2) `RobustATModel` (`src/models/robust_at.py`)

Token-based fusion over `[CLS, audio, text]` with a small transformer encoder.

```
audio: (B, D_a) -> LN -> SwiGLU -> (B, D_h) -> token (B,1,D_h)
text:  (B, D_t) -> LN -> SwiGLU -> (B, D_h) -> token (B,1,D_h)

tokens = concat([CLS, audio?, text?]) -> (B, 1+n_tokens, D_h)
encoder (MHSA+FFN) x L
pool CLS -> (B, D_h)
head (LN + SwiGLU) + residual
classifier -> logits (B, C)
```

Notes:
- No attention masks are needed because sequence length is always 2–3 tokens.
- No dropout inside the transformer stack (regularization is done via sampling + modality dropout).

## Training loop (`train.py` + `src/train_loop.py`)

- Dataset: `src/data/dataset.py::AVTDictDataset` loads one pickle per sample and yields dict batches via `collate_avt_dict`.
- Loss: `CrossEntropyLoss` with optional:
  - balanced class weights (`training.class_weights: balanced`)
  - label smoothing (`training.label_smoothing`)
- Optimizer: Adam / AdamW / SGD (`training.optimizer`)
- LR scheduler (optional): StepLR / CosineAnnealingWarmRestarts / ReduceLROnPlateau (`training.scheduler`)
- Early stopping: `training.patience` + `training.min_delta`, tracking `training.best_metric` (typically `uar`)
- Train-time augmentation (`training.augmentation`):
  - `feature_noise_std_audio`, `feature_noise_std_text`
  - `modality_dropout_p` (drops whole modality vectors per-sample, never both)
- EMA (optional): `training.ema.enabled`, used for eval/best checkpoint when enabled

## Evaluation (`evaluate.py` + `src/eval.py`)

Computes (sklearn-aligned):

- accuracy
- macro-F1
- weighted-F1 (wF1)
- UAR (= macro recall)

Artifacts written:

- `metrics_eval.json`
- `confusion_matrix.csv` + `confusion_matrix.png`
- optional ROC/PR plots for binary tasks

To keep TestA and TestB separate, pass distinct `--output_dir` values.
