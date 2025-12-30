# Diagnosis — Audio+Text MER (MFCC + Hashing)

Scope: the A+T pipeline driven by `extract_at_features.py` → `train.py` → `evaluate.py`, using features generated from `mer_dataset_builder`.

The format below is: **Finding → Evidence (file/symbol) → Impact → Fix**.

## A) Data & evaluation correctness

### 1) Feature dimension mismatches can crash at runtime

- Evidence: `MER/train.py` (`_infer_feature_dims`, sanity check before model init).
- Impact: prevents silent mismatches when `n_mfcc` changes but `model.input_dim_a` is not updated.
- Fix: fail fast with a clear error and remediation (`--set model.input_dim_a=<dim>` or re-extract features).

### 2) Split grouping preserves MELD official test split

- Evidence: `MER/extract_at_features.py` (`_split_to_group`).
- Impact: `testB` is conversational/out-of-domain (MELD test), while `testA` is acted.
- Fix: always report both TestA and TestB.

### 3) Speaker leakage checks are supported for acted datasets

- Evidence: `speaker_id` stored in `mer_dataset_builder` manifest and propagated into feature payloads.
- Impact: enables speaker-disjoint verification for acted corpora.
- Fix: verify overlaps with `MER/scripts/check_splits.py`.

### 4) Eval outputs can be overwritten without `--output_dir`

- Evidence: `MER/evaluate.py` (`--output_dir`, `_infer_output_dir`).
- Impact: running TestA then TestB into the same folder overwrites `metrics_eval.json`.
- Fix: always pass distinct `--output_dir` values per evaluation set.

## B) Representation & fusion

### 5) MFCC+hashing features are a ceiling for MELD

- Evidence: `MER/extract_at_features.py` computes MFCC mean/std and `HashingVectorizer` text vectors.
- Impact: strong on acted corpora, but weak on conversational MELD (domain shift + limited semantics).
- Fix: use pretrained encoders (HF frozen or end-to-end).

### 6) Text dominance risk in multi-corpus training

- Evidence: hashed text features + shallow fusion can allow shortcut learning.
- Impact: hurts robustness on TestB when acted corpora dominate training.
- Fix: use modality dropout (train-time) and MELD-aware sampling weights.

### 7) MELD is conversational; utterance-only modeling leaves performance on the table

- Evidence: current A+T pipeline builds one embedding per utterance; no dialogue-context modeling.
- Impact: many strong MELD results use dialogue context and speaker-aware modeling; utterance-only classifiers underperform.
- Fix: fine-tune encoders end-to-end and/or extend the manifest with dialogue metadata and build context-window models.

## C) Optimization stability

### 8) EMA selection must save EMA weights

- Evidence: `MER/train.py` (`on_best` saves EMA weights when EMA eval is enabled).
- Impact: without this, the “best” checkpoint can be worse than the metric used to select it.
- Fix: save EMA-weight checkpoint for best selection (implemented).

### 9) Mixed precision + grad accumulation improve feasibility of larger models

- Evidence: `MER/src/train_loop.py` supports `amp` + `grad_accum_steps`.
- Impact: enables larger backbones/batches on GPU and improves training throughput.
- Fix: set `training.amp: true`, `training.grad_accum_steps: <N>` in configs.

## D) Performance bottlenecks

### 10) Per-sample pickle I/O is the main throughput limiter

- Evidence: `MER/src/data/dataset.py::AVTDictDataset` loads one pickle per `__getitem__`.
- Impact: Python unpickling + filesystem overhead can dominate training time (especially on Windows).
- Fix: enable `persistent_workers` / `prefetch_factor`, or switch to memmapped arrays.

### 11) “Database” optimization: use memmapped arrays instead of pickles

- Evidence: `MER/build_memmap_features.py`, `MER/src/data/memmap_at.py`.
- Impact: much faster feature reads + less Python overhead.
- Fix: convert splits once and train from memmaps.

## E) Audio resampling to 16 kHz

- Evidence: `mer_dataset_builder` normalizes audio to mono 16 kHz PCM WAV.
- Impact: 16 kHz is the expected rate for most speech encoders (Wav2Vec2/HuBERT/WavLM/Whisper) and is usually not the quality bottleneck.
- Fix: focus first on representation quality (pretrained encoders) + MELD context modeling rather than changing sample rate assumptions.

