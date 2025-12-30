# Master Project Report (Scientific, Complete)

This document is a *single, end-to-end* scientific report of the repository in `MER/`. It consolidates:

- The **data pipeline** (multi-corpus dataset building into a unified manifest).
- The **feature pipelines** (classical MFCC+text hashing, optional offline HF embeddings, and end-to-end HF fine-tuning).
- The **model architectures** implemented in this codebase (legacy AVT models and the Audio+Text MER models).
- The **training/evaluation system** (metrics, logging, early stopping, LR scheduling, sampling, and reproducibility).
- **All experiments and results** currently recorded in `outputs/` and `outputs_paper/` (full inventories included in appendices).

The scope is **Audio+Text emotion recognition** using a unified 7-class label space compatible with MELD:
`{anger, disgust, fear, joy, sadness, surprise, neutral}`.

---

## 0) Repository map and file-by-file guide

### 0.1 Top-level structure

At a high level, the repo contains two major subsystems:

1) **Original PRL repository** (legacy multimodal AVT pipeline: audio+video+text, feature extraction, training, inference).
2) **Audio+Text multi-corpus pipeline** built on top of a unified dataset builder and modernized training/evaluation scripts.

Top-level directory map (abridged):

```
MER/
  configs/                    # YAML configs for training variants
  demo_ctx9_app/              # Separate interactive demo app (ctx=9 model)
  mer_dataset_builder/        # Dataset builder: download/prepare/validate -> unified manifest
  outputs/                    # Exploratory runs (historical experiments)
  outputs_paper/              # Paper suite runs (final experiment suite)
  scripts/                    # Orchestration scripts + reporting utilities
  src/                        # Library code (models, data, training loop, utils)
  static/                     # Website assets (from upstream PRL repo)
  tests/                      # Unit tests (some optional deps)
  train.py                    # Feature-based training (pickle/memmap features)
  evaluate.py                 # Feature-based evaluation
  train_hf_e2e.py             # End-to-end HF training from unified manifest
  evaluate_hf_e2e.py          # End-to-end HF evaluation
  extract_at_features.py      # MFCC + HashingVectorizer feature extraction
  extract_at_features_hf.py   # Offline HF embedding feature extraction
  build_memmap_features.py    # Convert per-sample pickles to memmap for speed
  PROJECT_REPORT.md           # Academic project report (narrative + methodology)
  EXPERIMENTS_REPORT.md       # Detailed paper-suite results with full inventory
```

### 0.2 Documentation files (what each contains)

- `PROJECT_REPORT.md`: academic narrative (datasets, processing, architectures, experimental framing).
- `EXPERIMENTS_REPORT.md`: detailed results and tables from `outputs_paper/` plus a full run inventory.
- `outputs_paper/comparison_report.md`: auto-generated run table (paper suite), derived from `outputs_paper/summary.csv`.
- `outputs/comparison_report.md`: auto-generated run table (exploratory), derived from `outputs/summary.csv`.
- `ARCHITECTURE.md`, `BASELINE_REPORT.md`, `DIAGNOSIS.md`, `IMPROVEMENTS_REPORT.md`, `ABLATION.md`: incremental engineering notes and intermediate analyses. Their essential content is incorporated into the present report.

### 0.3 Core execution scripts (what they do)

**Dataset construction**
- `mer_dataset_builder/`: a separate Python project that downloads/validates raw corpora (where possible), normalizes audio, derives transcripts, standardizes labels, and writes a unified manifest.

**Feature-based pipeline (classical features or offline embeddings)**
- `extract_at_features.py`: classical features: MFCC(mean+std) + text hashing.
- `extract_at_features_hf.py`: offline embeddings from HF encoders: WavLM (audio) + RoBERTa (text).
- `train.py`: trains a classifier/fusion network on those features.
- `evaluate.py`: evaluates a saved checkpoint on `testA` / `testB` feature folders.

**End-to-end pipeline (best quality)**
- `train_hf_e2e.py`: trains encoders + fusion end-to-end directly from the unified manifest WAV+transcripts.
- `evaluate_hf_e2e.py`: evaluates end-to-end checkpoints (with split/dataset filters).

**Experiment orchestration**
- `scripts/run_full_paper_suite.ps1`: the full paper suite driver (stage1/stage2 + MELD sweeps + transfer + LOCO + IEMOCAP CV).
- `scripts/run_iemocap4_cv.ps1`: IEMOCAP-4 5-fold CV driver (A/T/AT).
- `scripts/collect_results.py`: rebuilds `summary.csv` and `comparison_report.md` by searching all `metrics_eval.json` files.

### 0.4 File catalog (code + configs)

This section provides a concise file-by-file catalog of the *code-bearing* parts of the repository (excluding large generated artifacts such as `outputs/`, `outputs_paper/`, and any virtual environments).

#### Root-level scripts and docs (`MER/`)

- `.flake8`: linting configuration (style checks).
- `.gitignore`: Git ignore rules (excludes large artifacts and environment files from version control).
- `ABLATION.md`: intermediate ablation notes.
- `ARCHITECTURE.md`: intermediate architecture notes (shapes and data flow).
- `BASELINE_REPORT.md`: baseline reproduction notes.
- `DIAGNOSIS.md`: code-level diagnosis notes.
- `IMPROVEMENTS_REPORT.md`: incremental improvement notes and measured outcomes.
- `PROJECT_REPORT.md`: academic narrative report (methodology + selected results).
- `EXPERIMENTS_REPORT.md`: detailed paper-suite results (tables + inventory).
- `MASTER_PROJECT_REPORT.md`: this document.
- `README.md`: upstream PRL repository README plus Audio+Text usage notes.
- `index.html` and `static/`: website entrypoint and static assets for the upstream project webpage.
- `IEMOCAP_full_release.tar.gz`: a local archive placeholder used during builder integration (not required by the training code once the unified manifest is built).
- `train.py`: feature-based training entrypoint (pickle/memmap features).
- `evaluate.py`: feature-based evaluation entrypoint.
- `train_hf_e2e.py`: end-to-end HF training entrypoint (manifest -> WAV+text).
- `evaluate_hf_e2e.py`: end-to-end HF evaluation entrypoint.
- `extract_at_features.py`: MFCC(mean+std) + HashingVectorizer feature extraction.
- `extract_at_features_hf.py`: offline HF embedding feature extraction (audio+text encoders).
- `build_memmap_features.py`: convert per-sample feature pickles to memory-mapped arrays to reduce I/O overhead.
- `visualize_flexible_at.py`: model visualization utilities (textual summaries; graph export if Graphviz is installed).

#### Configs (`MER/configs/`)

Configs are grouped by pipeline:

- Classical features (MFCC + hashing; trained by `train.py`):
  - `mer_builder_at_simple.yaml`: baseline feature-based Audio+Text model.
  - `mer_builder_at_simple_robust*.yaml`: RobustAT-based variants with class balancing, modality dropout, and optional EMA.
  - `mer_builder_at_simple_robust_moddrop_ema_meld_ft.yaml`: MELD-focused fine-tune settings for the feature pipeline.

- Offline HF embeddings (precompute embeddings; trained by `train.py`):
  - `mer_builder_at_hf.yaml`: feature-based training on precomputed WavLM/RoBERTa embeddings.

- End-to-end HF (trained by `train_hf_e2e.py`):
  - `mer_builder_at_hf_e2e.yaml`: baseline end-to-end run.
  - `mer_builder_at_hf_e2e_all_v2.yaml`: stronger “v2” setting (more context/longer audio, tuned for MELD).
  - `mer_builder_at_hf_e2e_large.yaml`: larger setting (more capacity; uses gradient checkpointing).
  - `mer_builder_at_hf_e2e_all_stage1_head_warmup.yaml`: stage1 head warmup (encoders frozen).
  - `mer_builder_at_hf_e2e_all_stage2_finetune.yaml`: stage2 full fine-tune (encoders trainable).
  - `mer_builder_at_hf_e2e_meld_ft*.yaml`: MELD-only fine-tune variants (context window and max audio duration sweep).
  - `mer_builder_at_hf_e2e_iemocap4_cv.yaml`: IEMOCAP-4 configuration (4 classes, CV slicing by session).

#### Orchestration scripts (`MER/scripts/`)

- `run_full_paper_suite.ps1`: master “paper suite” runner (stage1/stage2 + MELD sweeps + transfer + LOCO + IEMOCAP CV).
- `run_iemocap4_cv.ps1`: IEMOCAP-4 5-fold CV runner (A/T/AT, multi-seed).
- `run_paper_experiments.ps1`: compact runner for the core paper suite (without the heavy extras).
- `run_meld_ft_experiments.ps1`: MELD-only fine-tune experiments (ctx7/ctx9).
- `run_meld_ft_from_best_stage2.ps1`: MELD fine-tune starting from the newest stage2 checkpoint.
- `run_baseline_at.ps1`, `run_baseline_at.sh`: baseline automation helpers.
- `run_full_pipeline.ps1`: pipeline helper (dataset -> features -> train -> eval) for the Audio+Text track.
- `check_splits.py`: debugging utility for split validation.
- `collect_results.py`: generates `summary.csv` and `comparison_report.md` from `metrics_eval.json`.

#### Library code (`MER/src/`)

Top-level modules:
- `augmentation.py`: feature noise + modality dropout utilities used in training.
- `config_utils.py`: config loading, override application, run directory naming and resolution.
- `ema.py`: exponential moving average utility for model weights.
- `eval.py`: evaluation loop producing logits, metrics, confusion matrices.
- `logging_utils.py`: checkpoint saving/loading, CSV logging, directory utilities.
- `metrics.py`: metric computation (acc, macro-F1, weighted-F1, UAR, per-class metrics).
- `plotting.py`: plotting utilities (train/val curves, confusion matrices).
- `reproducibility.py`: deterministic seeding utilities.
- `train_loop.py`: the core training loop (AMP, grad accumulation, clipping, LR schedules, early stopping).

Data subpackage:
- `src/data/dataset.py`: feature datasets (pickle-based) and legacy AVT datasets.
- `src/data/memmap_at.py`: memmap dataset backend for faster feature loading.
- `src/data/manifest_at.py`: dataset that reads WAV+transcript directly from the unified manifest (end-to-end HF).
- `src/data/hf_manifest_collate.py`: HF collator that builds tokenized batches (audio processor + tokenizer).

Models subpackage:
- `src/models/flexible_at.py`: GFL-based audio/text fusion model (feature pipeline).
- `src/models/robust_at.py`: token-based transformer fusion model (feature + end-to-end).
- `src/models/hf_at.py`: HF wrapper model combining WavLM/RoBERTa encoders with RobustAT fusion.
- `src/models/architectures.py`: legacy PRL architectures (ResNet, wav2vec2 wrappers, AVT fusion, GFL definition).

Legacy trainer:
- `src/net_trainer/net_trainer.py`: legacy training loop used by the upstream PRL code path (AVT, binary-label options).

Optional embedded dependency:
- `src/data/face_detection/`: third-party face detection module (includes `face_detection_test.py` that requires OpenCV; not needed for Audio+Text).

#### Demo application (`MER/demo_ctx9_app/`)

- `demo_ctx9_app/app.py`: minimal GUI demo to record/load audio + enter text (and optional context) and run inference.
- `demo_ctx9_app/README.md`: usage notes for the demo.

#### Dataset builder (`MER/mer_dataset_builder/`)

This is a standalone, installable builder project with:
- CLI entrypoint: `mer_dataset_builder/src/mer_builder/cli.py`
- dataset download modules: `mer_dataset_builder/src/mer_builder/download/*.py`
- dataset parsing modules: `mer_dataset_builder/src/mer_builder/prepare/parse_*.py`
- audio normalization: `mer_dataset_builder/src/mer_builder/prepare/normalize_audio.py`
- validation/integrity: `mer_dataset_builder/src/mer_builder/prepare/validate.py`, `integrity.py`

---

## 1) Data: unified multi-corpus dataset

### 1.1 Builder overview (mer_dataset_builder)

The builder is located in `mer_dataset_builder/` and implements:

- **Download/validation**: `mer_dataset_builder/src/mer_builder/download/*`
- **Parsing + transcript derivation + label mapping**: `mer_dataset_builder/src/mer_builder/prepare/parse_*.py`
- **Audio normalization** to mono, 16 kHz, 16-bit PCM WAV via FFmpeg: `mer_dataset_builder/src/mer_builder/prepare/normalize_audio.py`
- **Speaker-disjoint splits** for acted corpora: `mer_dataset_builder/src/mer_builder/prepare/split_speakers.py`
- **Manifest writing**: `mer_dataset_builder/src/mer_builder/prepare/build_manifest.py`
- **Validation/integrity utilities**: `mer_dataset_builder/src/mer_builder/prepare/validate.py`, `mer_dataset_builder/src/mer_builder/prepare/integrity.py`

Unified output files:
- `mer_dataset_builder/data/processed/meta_manifest.jsonl`
- `mer_dataset_builder/data/processed/meta_manifest.csv`
- normalized audio under `mer_dataset_builder/data/processed/audio/<DATASET>/...`

### 1.2 Datasets integrated

This repository integrates the following corpora:

- **MEAD Part0**
- **MELD Raw** (official `train/dev/test_sent_emo.csv` and corresponding clips)
- **RAVDESS** (speech only; song excluded)
- **CREMA-D**
- **ESD**
- **EmoV-DB** (OpenSLR SLR115)
- **IEMOCAP** (full release; license-gated)

The builder standardizes each dataset into a common schema and label space, while retaining provenance (`dataset`, `source_label`, `notes`).

### 1.3 Label space and mapping

Target label space (7 classes):
`anger, disgust, fear, joy, sadness, surprise, neutral`

Key mapping principles implemented by the builder:
- Dataset-specific labels are mapped into the 7-class space, logging non-trivial mappings in `notes`.
- If a dataset lacks some classes (e.g., ESD lacks fear/disgust), samples remain but naturally do not cover the missing labels.
- EmoV-DB includes labels like “sleepy”; the builder supports either dropping them or mapping to neutral depending on configuration.
- MEAD includes “contempt”; the builder supports either dropping or mapping it to disgust.

### 1.4 Transcripts: provenance and construction

Transcripts are stored per sample in the manifest as UTF-8 strings. Transcript derivation is dataset-specific:

- **MELD**: transcripts are parsed from official split CSVs (utterance-level).
- **RAVDESS / CREMA-D**: transcripts are derived from fixed sentence inventories or metadata tables.
- **ESD**: transcripts are parsed from the dataset’s transcript files/tables (supports common official layouts and per-utterance `.txt` sidecars).
- **EmoV-DB**: transcripts are derived from the CMU Arctic prompts file `cmuarctic.data` (prompt id -> text).
- **MEAD**: transcripts are derived from `MEAD-supp.pdf` (speech corpus list) with robust parsing and fallbacks (`mead_sentences.csv` override supported).
- **IEMOCAP**: transcripts are parsed from the session transcription files; emotion labels are derived from evaluation annotations with additional handling of ambiguous labels.

The training code treats transcripts as opaque text strings; any language assumptions are external to the pipeline.

### 1.5 Audio normalization

All audio is normalized to:
- mono
- 16 kHz sampling rate
- PCM WAV

This ensures consistent audio I/O for both classical MFCC feature extraction and HF end-to-end training.

### 1.6 Splits and evaluation sets

The unified manifest contains split labels that are used to build training and evaluation datasets:

- Acted corpora (speaker-disjoint): `train`, `val`, and `test` (or `testA` for MEAD).
- MELD (official split preserved): `meld_train`, `meld_dev`, and `testB`.

Two principal evaluation sets are defined:
- **TestA (acted, in-domain)**: `split=testA` (used as the acted evaluation anchor in this project).
- **TestB (conversational, out-of-domain)**: `split=testB` (MELD test).

Note: some acted corpora use `split=test` rather than `testA`; many training/evaluation scripts map `test` into the “acted evaluation” group depending on the experiment.

### 1.7 Dataset size statistics (from the current manifest)

All counts below are computed from `mer_dataset_builder/data/processed/meta_manifest.jsonl` present in this repository.

**Total samples:** 106,256 across 7 datasets.

Per-dataset totals:
- ESD: 35,000
- MEAD: 31,734
- MELD: 13,708
- IEMOCAP: 10,039
- CREMA-D: 7,442
- EmoV-DB: 6,893
- RAVDESS: 1,440

Acted vs conversational:
- Acted total (MEAD, RAVDESS, CREMA-D, ESD, EmoV-DB, IEMOCAP): 92,548
- Conversational total (MELD): 13,708

Split totals:
- train: 71,062
- val: 10,378
- test: 7,148
- testA: 3,960
- meld_train: 9,989
- meld_dev: 1,109
- testB: 2,610

Emotion distribution (overall):
- neutral: 21,770
- anger: 19,880
- joy: 18,982
- sadness: 14,883
- surprise: 13,216
- disgust: 11,354
- fear: 6,171

MELD (test/train/dev combined) emotion distribution:
- neutral: 6,436
- joy: 2,308
- surprise: 1,636
- anger: 1,607
- sadness: 1,002
- disgust: 361
- fear: 358

---

## 2) Feature engineering and representations

This codebase supports three representation regimes for Audio+Text MER:

1) **Classical features** (fast, cheap, weaker generalization):
   - audio: MFCC(mean + std)
   - text: HashingVectorizer bag-of-words hashing
2) **Offline neural embeddings** (middle ground):
   - audio: WavLM embeddings pooled to a vector
   - text: RoBERTa embeddings pooled to a vector
3) **End-to-end neural encoders** (best quality):
   - raw waveform -> WavLM encoder (fine-tuned)
   - raw transcript -> RoBERTa encoder (fine-tuned)

### 2.1 MFCC (Mel-Frequency Cepstral Coefficients)

MFCCs are a classical audio representation designed to approximate human auditory perception:

1) compute the short-time Fourier transform (STFT)
2) apply a mel-scaled filterbank to the power spectrum
3) take log-mel energies
4) apply a discrete cosine transform (DCT) to decorrelate components -> MFCCs

In this project (`extract_at_features.py`):
- MFCC frames are computed at 16 kHz with `n_fft=400` and `hop_length=160`.
- Features are aggregated to a fixed-length utterance vector by concatenating:
  - mean across time (frames)
  - standard deviation across time
- Resulting audio vector dimension: `2 * n_mfcc`.

### 2.2 HashingVectorizer for text features

`HashingVectorizer` (from scikit-learn) is a stateless bag-of-words featurizer:
- tokenizes text into word n-grams (default tokenization rules)
- applies a hashing trick to map tokens to a fixed-dimensional sparse vector
- avoids building a vocabulary (memory-efficient, fast, but collisions occur)

In this project (`extract_at_features.py`):
- `alternate_sign=False` and `lowercase=True`
- output dimension is set by `text_dim` in the script

This representation is robust and simple but discards word order and deep semantics, which limits performance on conversational MELD.

### 2.3 Offline HF embeddings

`extract_at_features_hf.py` computes feature vectors using HuggingFace encoders:
- audio encoder (commonly `microsoft/wavlm-base`)
- text encoder (commonly `roberta-base`)

The script:
- loads WAV and resamples to 16 kHz
- extracts encoder hidden states
- applies pooling (masked mean; optionally mean+std)
- writes per-sample pickle files compatible with `train.py`

### 2.4 End-to-end HF training

In the end-to-end setting (`train_hf_e2e.py`):
- the dataset reads normalized WAV + transcript directly from `meta_manifest.jsonl`
- WavLM and RoBERTa run inside the model forward pass
- gradients flow end-to-end into both encoders (unless frozen by config)

This is the highest-capacity pipeline and is the basis for the “paper suite” results under `outputs_paper/`.

---

## 3) Model architectures implemented

This repository contains both legacy multimodal AVT models (from the upstream PRL codebase) and the modern Audio+Text models used in the current experiments.

### 3.1 Legacy PRL architectures (AVT)

Files:
- `src/models/architectures.py`
- `src/data/dataset.py`
- `src/avt_feature_extraction.py`, `src/train_avt_model.py`, `src/inference.py`

Key components:
- Visual backbone: ResNet-50 (`ResNet50` implementation in `architectures.py`)
- Acoustic backbone: wav2vec2-based model wrapper (`AudioModel` in `architectures.py`)
- Cross-modal fusion: **Gated Fusion Layer (GFL)** operating on pairs of modalities
- An AVT fusion head (`AVTmodel` in `architectures.py`)

This legacy path is retained for completeness and provenance, but the focus of the present project is Audio+Text MER on the unified multi-corpus dataset.

### 3.2 FlexibleATModel (feature-based A/T or A+T)

File:
- `src/models/flexible_at.py`

Purpose:
- A small fusion network that can operate in three modes: Audio-only, Text-only, or Audio+Text.

Core mechanisms:
- Per-modality LayerNorm + linear projection + dropout.
- If both modalities are present, fusion is performed by **GFL** (`src/models/architectures.py`), which learns a soft gate to combine modality-specific transforms.
- The classifier head uses LayerNorm + SwiGLU projection + linear classifier.

This model is used by `train.py` for feature-based experiments.

### 3.3 RobustATModel (token-based transformer fusion)

File:
- `src/models/robust_at.py`

Purpose:
- A compact transformer-style fusion module operating on a tiny token set:
  - `[CLS]` token
  - one “audio token” (pooled audio embedding)
  - one “text token” (pooled text embedding)

Key design features:
- **SwiGLU** feed-forward blocks (SiLU-gated linear unit) instead of ReLU.
- A small stack of Transformer blocks:
  - LayerNorm -> Multihead self-attention -> residual
  - LayerNorm -> SwiGLU FFN -> residual
- Token-type embeddings (CLS/audio/text) are added to tokens.

Positional encoding:
- There is no sequence positional encoding because the token set is unordered (size 2–3). The architecture acts as a set/slot transformer for modality tokens.

This model is used:
- directly in `train.py` for feature-based training, and
- as the fusion head inside `HFAudioTextModel` for end-to-end HF training.

### 3.4 HFAudioTextModel (end-to-end WavLM + RoBERTa + RobustAT fusion)

File:
- `src/models/hf_at.py`

Components:
- Audio encoder: HF `AutoModel.from_pretrained(audio_model)` (e.g., WavLM).
- Text encoder: HF `AutoModel.from_pretrained(text_model)` (e.g., RoBERTa).
- Pooling:
  - audio: masked mean or mean+std (requires converting input attention masks to hidden-state lengths)
  - text: CLS pooling or masked mean
- Fusion head: `RobustATModel` over `[CLS, audio, text]` tokens.

Masking correctness:
- HF speech encoders downsample time internally. For correct pooling, `hf_at.py` implements `_infer_feature_vector_attention_mask` to convert waveform-length masks to hidden-state-length masks.

---

## 4) Training and evaluation system

### 4.1 Configuration and run directories

Training and evaluation scripts use YAML configs in `configs/` and write a run directory containing:
- `config_resolved.yaml` (fully resolved config snapshot)
- `checkpoints/best.pt` and `checkpoints/last.pt`
- `metrics_train.csv`, `metrics_val.csv`
- `plots/` artifacts (loss/metric curves, confusion matrices during evaluation)

### 4.2 Training loop (early stopping, LR scheduling, AMP, EMA)

File:
- `src/train_loop.py`

Key mechanisms:
- **Early stopping** via `patience` and `min_delta` on a chosen `best_metric`.
- **ReduceLROnPlateau** scheduler support (configured in YAML; typically monitoring `uar`).
- **Gradient accumulation** (`grad_accum_steps`) for effective large batch sizes.
- **Gradient clipping** (`grad_clip_norm`) for stability.
- **Mixed precision** (AMP) enabled automatically when CUDA is available.
- Optional **EMA** (exponential moving average of model weights) for evaluation stability.
- Optional training-time **feature noise** and **modality dropout** (for feature-based models).

### 4.3 Samplers and class imbalance

Both feature-based and end-to-end training support:
- Balanced class weights for the loss (`class_weights: balanced`).
- Weighted samplers that can upweight under-represented corpora (e.g., upweight MELD during mixed-corpus training).

### 4.4 Evaluation and artifacts

Feature-based evaluation:
- `evaluate.py` reads a trained checkpoint and evaluates on a feature directory (`data.eval_dir`).

End-to-end evaluation:
- `evaluate_hf_e2e.py` reads `meta_manifest.jsonl` and runs evaluation using split/dataset filters.

Evaluation produces:
- `metrics_eval.json` (machine-readable metrics)
- `confusion_matrix.csv` and `confusion_matrix.png`

### 4.5 Aggregated reporting

File:
- `scripts/collect_results.py`

This utility scans for all `metrics_eval.json` under a given outputs folder and regenerates:
- `summary.csv` (flat table)
- `comparison_report.md` (Markdown table)

This is how the “complete run inventories” in the appendices are kept consistent with on-disk artifacts.

---

## 5) Experiments: design, motivation, and results

This section explains **all experiments currently recorded** in:
- `outputs_paper/` (paper suite)
- `outputs/` (exploratory history)

### 5.0 Run naming conventions (how to interpret run IDs)

The run IDs in `outputs_paper/summary.csv` and `outputs/summary.csv` encode experimental intent:

**Paper suite (`outputs_paper/`)**
- `paper_all_stage1`, `paper_all_stage2`: 2-stage end-to-end HF training on all corpora.
- `paper_meld_ft_ctx{0,3,7,9}_a20`: MELD-only fine-tune from stage2 with context window sweep and max audio duration 20s.
- `paper_meld_ctx9_a20_{audio_only,text_only,audio_text}`: modality ablations on MELD (TestB).
- `paper_acted_stage{1,2}`: acted-only training (MELD excluded) evaluated on MELD test.
- `paper_transfer_acted_to_meld_ft_ctx9`: acted-only model adapted by MELD fine-tuning.
- `paper_loco_exclude_<CORPUS>`: LOCO runs (train excludes corpus; evaluation on held-out corpus).
- `paper_iemocap4_seedS_foldF_<A|T|AT>`: IEMOCAP-4 5-fold CV runs, repeated over seeds and modalities.

**Exploratory history (`outputs/`)**
- `mfcc80_*`, `robust_mfcc80_*`, `exp*_robust_*`: feature-based MFCC+hashing or RobustAT variants.
- `run_hf_e2e_*`: earlier end-to-end HF iterations and prototypes.
- `run_meld_ft_from_stage2_ctx*`: MELD fine-tunes from a strong stage2 checkpoint (pre-paper suite).
- `synth_*`: synthetic/sanity runs.

### 5.1 Why TestA vs TestB?

The unified dataset intentionally measures **domain shift**:
- **TestA** represents acted speech (controlled conditions; clearer acoustic emotion cues).
- **TestB** is MELD conversational speech (noisy, multi-speaker, contextual; emotion often depends on dialogue context and lexical cues).

In practice, models can score very high on acted TestA while still underperforming on conversational TestB—this is the core generalization challenge addressed by the paper suite.

### 5.2 Paper suite (outputs_paper): what was tested and what it shows

The paper suite focuses on the methodology expected by reviewers for modern SER/MER:
- 2-stage fine-tuning (frozen -> full)
- MELD context sweep
- modality ablation
- acted→conversational transfer and adaptation
- LOCO robustness
- IEMOCAP-4 5-fold CV with multi-seed reporting

#### 5.2.1 Two-stage training (ALL corpora)

Observation:
- Stage 2 dramatically improves TestA, indicating that end-to-end fine-tuning captures acted-domain cues strongly.
- TestB gains are smaller; conversational performance remains constrained by domain shift and text dominance.

#### 5.2.2 MELD context sweep

Observation:
- ctx=0 achieves the best **weighted F1** on TestB (paper_meld_ft_ctx0_a20).
- ctx=9 achieves the best **UAR** on TestB (paper_meld_ft_ctx9_a20).

Interpretation:
- Weighted F1 is dominated by majority classes (notably neutral in MELD). Shorter/no context may preserve precision on dominant classes.
- UAR is macro-average recall, more sensitive to minority classes; context can help recall for rare emotions.

#### 5.2.3 Modality ablation (MELD)

Observation:
- Text-only is strong on TestB (consistent with conversational datasets where lexical semantics strongly indicate emotion).
- Audio-only is weak, reflecting both conversational noise and the fact that many MELD emotions are linguistically expressed.
- Audio+Text improves UAR relative to text-only, indicating complementary acoustic cues for minority-class recall.

#### 5.2.4 Cross-domain transfer (Acted -> MELD)

Observation:
- Acted-only models generalize poorly to MELD.
- MELD fine-tuning is essential for adaptation.

#### 5.2.5 LOCO

Observation:
- Held-out MELD is the hardest case (lowest LOCO metrics), reinforcing the conversational domain shift.
- Acted corpora tend to remain easier to generalize within, given shared recording and acting structure.

#### 5.2.6 IEMOCAP-4 (5-fold CV, multi-seed)

Observation:
- A+T consistently outperforms A-only and T-only baselines across folds/seeds.
- Variance across folds is modest, indicating stable training behavior.

### 5.3 Exploratory experiments (outputs): what they were and why they matter

The `outputs/` directory captures iterative development:

1) **Classical MFCC+hashing baseline** (`mfcc80_bs128`, `run_best`, etc.)
   - fast, but weak on MELD (TestB).
2) **Robust feature-based models** (`exp*_robust_*`)
   - introduce class balancing, modality dropout, EMA; improve TestA but limited on TestB.
3) **Early end-to-end HF attempts** (`run_hf_e2e_1`)
   - showed failure modes (very low TestA), motivating stage-wise training and better config discipline.
4) **Successful end-to-end HF training** (`run_hf_e2e_all_stage1_*`, `run_hf_e2e_all_stage2_*`)
   - improved TestB substantially vs classical features.
5) **MELD fine-tunes from strong checkpoints** (`run_meld_ft_from_stage2_ctx7/ctx9`)
   - demonstrate the practical benefit of MELD adaptation.
6) **Synthetic sanity runs** (`synth_*`)
   - internal checks to ensure the training/evaluation pipeline can overfit or behave as expected in controlled conditions.

This history is scientifically relevant because it documents why the “paper suite” converged on end-to-end HF + staged fine-tuning + contextualization as the best-performing, reviewer-acceptable method.

---

## 6) Practical artifact guide (where to find what)

For any run `X` under `outputs/` or `outputs_paper/`:

- Config snapshot: `.../X/config_resolved.yaml`
- Checkpoint: `.../X/checkpoints/best.pt`
- Evaluation metrics: `.../X/eval_*/metrics_eval.json`
- Confusion matrices: `.../X/eval_*/confusion_matrix.csv` and `.../X/eval_*/confusion_matrix.png`

Aggregated tables:
- `outputs/summary.csv` and `outputs/comparison_report.md`
- `outputs_paper/summary.csv` and `outputs_paper/comparison_report.md`

---

## Appendix A) Complete results inventory: outputs_paper (paper suite)

The table below is an exact formatted view of `outputs_paper/summary.csv`.

| run | eval | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| paper_acted_stage1 | eval_meld_testB | 0.2257 | 0.1959 | 0.1341 | 0.1865 |
| paper_acted_stage2 | eval_meld_testB | 0.2410 | 0.1988 | 0.1837 | 0.2556 |
| paper_all_stage1 | eval_testA | 0.7278 | 0.7191 | 0.6820 | 0.6582 |
| paper_all_stage1 | eval_testB | 0.6207 | 0.5818 | 0.3789 | 0.3650 |
| paper_all_stage2 | eval_testA | 0.9391 | 0.9396 | 0.9336 | 0.9371 |
| paper_all_stage2 | eval_testB | 0.6126 | 0.6036 | 0.4272 | 0.4204 |
| paper_iemocap4_seed1_fold1_A | eval_fold | 0.6457 | 0.6382 | 0.6026 | 0.6394 |
| paper_iemocap4_seed1_fold1_AT | eval_fold | 0.6842 | 0.6841 | 0.6577 | 0.6784 |
| paper_iemocap4_seed1_fold1_T | eval_fold | 0.6157 | 0.6258 | 0.5967 | 0.6028 |
| paper_iemocap4_seed1_fold2_A | eval_fold | 0.6565 | 0.6490 | 0.6269 | 0.6331 |
| paper_iemocap4_seed1_fold2_AT | eval_fold | 0.7078 | 0.7120 | 0.6930 | 0.6933 |
| paper_iemocap4_seed1_fold2_T | eval_fold | 0.6791 | 0.6854 | 0.6501 | 0.6524 |
| paper_iemocap4_seed1_fold3_A | eval_fold | 0.6283 | 0.6355 | 0.5962 | 0.6053 |
| paper_iemocap4_seed1_fold3_AT | eval_fold | 0.6988 | 0.6986 | 0.6656 | 0.6730 |
| paper_iemocap4_seed1_fold3_T | eval_fold | 0.6441 | 0.6332 | 0.5787 | 0.5617 |
| paper_iemocap4_seed1_fold4_A | eval_fold | 0.6801 | 0.6909 | 0.6215 | 0.6451 |
| paper_iemocap4_seed1_fold4_AT | eval_fold | 0.7281 | 0.7314 | 0.6488 | 0.6769 |
| paper_iemocap4_seed1_fold4_T | eval_fold | 0.6680 | 0.6822 | 0.5812 | 0.6227 |
| paper_iemocap4_seed1_fold5_A | eval_fold | 0.6541 | 0.6468 | 0.6093 | 0.5895 |
| paper_iemocap4_seed1_fold5_AT | eval_fold | 0.7205 | 0.7263 | 0.6955 | 0.7069 |
| paper_iemocap4_seed1_fold5_T | eval_fold | 0.6654 | 0.6669 | 0.6269 | 0.6474 |
| paper_iemocap4_seed2_fold1_A | eval_fold | 0.6650 | 0.6621 | 0.6341 | 0.6477 |
| paper_iemocap4_seed2_fold1_AT | eval_fold | 0.6893 | 0.6975 | 0.6777 | 0.6962 |
| paper_iemocap4_seed2_fold1_T | eval_fold | 0.6423 | 0.6283 | 0.5945 | 0.5907 |
| paper_iemocap4_seed2_fold2_A | eval_fold | 0.6543 | 0.6514 | 0.6378 | 0.6560 |
| paper_iemocap4_seed2_fold2_AT | eval_fold | 0.7214 | 0.7205 | 0.7035 | 0.7199 |
| paper_iemocap4_seed2_fold2_T | eval_fold | 0.6943 | 0.6964 | 0.6624 | 0.6752 |
| paper_iemocap4_seed2_fold3_A | eval_fold | 0.6489 | 0.6564 | 0.6219 | 0.6315 |
| paper_iemocap4_seed2_fold3_AT | eval_fold | 0.6935 | 0.7006 | 0.6700 | 0.6838 |
| paper_iemocap4_seed2_fold3_T | eval_fold | 0.6225 | 0.6336 | 0.5915 | 0.5994 |
| paper_iemocap4_seed2_fold4_A | eval_fold | 0.6791 | 0.6751 | 0.5992 | 0.6141 |
| paper_iemocap4_seed2_fold4_AT | eval_fold | 0.7518 | 0.7494 | 0.6740 | 0.6873 |
| paper_iemocap4_seed2_fold4_T | eval_fold | 0.6587 | 0.6734 | 0.5668 | 0.6371 |
| paper_iemocap4_seed2_fold5_A | eval_fold | 0.6494 | 0.6509 | 0.6177 | 0.6145 |
| paper_iemocap4_seed2_fold5_AT | eval_fold | 0.7163 | 0.7159 | 0.6841 | 0.6802 |
| paper_iemocap4_seed2_fold5_T | eval_fold | 0.6532 | 0.6636 | 0.6280 | 0.6368 |
| paper_iemocap4_seed3_fold1_A | eval_fold | 0.6486 | 0.6489 | 0.6159 | 0.6403 |
| paper_iemocap4_seed3_fold1_AT | eval_fold | 0.6808 | 0.6818 | 0.6549 | 0.6918 |
| paper_iemocap4_seed3_fold1_T | eval_fold | 0.6327 | 0.6247 | 0.5830 | 0.5875 |
| paper_iemocap4_seed3_fold2_A | eval_fold | 0.6543 | 0.6483 | 0.6246 | 0.6335 |
| paper_iemocap4_seed3_fold2_AT | eval_fold | 0.7490 | 0.7459 | 0.7287 | 0.7432 |
| paper_iemocap4_seed3_fold2_T | eval_fold | 0.6954 | 0.6996 | 0.6682 | 0.6828 |
| paper_iemocap4_seed3_fold3_A | eval_fold | 0.6293 | 0.6247 | 0.5765 | 0.5755 |
| paper_iemocap4_seed3_fold3_AT | eval_fold | 0.7031 | 0.6985 | 0.6569 | 0.6614 |
| paper_iemocap4_seed3_fold3_T | eval_fold | 0.6542 | 0.6438 | 0.5948 | 0.5884 |
| paper_iemocap4_seed3_fold4_A | eval_fold | 0.6859 | 0.6921 | 0.6143 | 0.6402 |
| paper_iemocap4_seed3_fold4_AT | eval_fold | 0.7300 | 0.7363 | 0.6557 | 0.6869 |
| paper_iemocap4_seed3_fold4_T | eval_fold | 0.6626 | 0.6745 | 0.5651 | 0.5989 |
| paper_iemocap4_seed3_fold5_A | eval_fold | 0.6461 | 0.6526 | 0.6274 | 0.6360 |
| paper_iemocap4_seed3_fold5_AT | eval_fold | 0.7215 | 0.7228 | 0.6941 | 0.6988 |
| paper_iemocap4_seed3_fold5_T | eval_fold | 0.6762 | 0.6774 | 0.6328 | 0.6417 |
| paper_loco_exclude_CREMA-D | eval_holdout | 0.7171 | 0.7145 | 0.6145 | 0.6182 |
| paper_loco_exclude_EmoV-DB | eval_holdout | 0.9149 | 0.9163 | 0.5151 | 0.5048 |
| paper_loco_exclude_ESD | eval_holdout | 1.0000 | 1.0000 | 0.7143 | 0.7143 |
| paper_loco_exclude_IEMOCAP | eval_holdout | 0.6504 | 0.6481 | 0.5044 | 0.5087 |
| paper_loco_exclude_MEAD | eval_holdout | 0.9197 | 0.9201 | 0.9148 | 0.9134 |
| paper_loco_exclude_MELD | eval_holdout | 0.5475 | 0.5598 | 0.3952 | 0.4130 |
| paper_loco_exclude_RAVDESS | eval_holdout | 0.7667 | 0.7607 | 0.7533 | 0.7579 |
| paper_meld_ctx9_a20_audio_only | eval_testB | 0.3054 | 0.3441 | 0.2504 | 0.2869 |
| paper_meld_ctx9_a20_audio_text | eval_testB | 0.5770 | 0.5842 | 0.4255 | 0.4470 |
| paper_meld_ctx9_a20_text_only | eval_testB | 0.6011 | 0.5932 | 0.4163 | 0.4218 |
| paper_meld_ft_ctx0_a20 | eval_testA | 0.9399 | 0.9401 | 0.9366 | 0.9374 |
| paper_meld_ft_ctx0_a20 | eval_testB | 0.6157 | 0.6092 | 0.4368 | 0.4330 |
| paper_meld_ft_ctx3_a20 | eval_testA | 0.9394 | 0.9394 | 0.9366 | 0.9340 |
| paper_meld_ft_ctx3_a20 | eval_testB | 0.5969 | 0.5976 | 0.4268 | 0.4375 |
| paper_meld_ft_ctx7_a20 | eval_testA | 0.9391 | 0.9393 | 0.9365 | 0.9346 |
| paper_meld_ft_ctx7_a20 | eval_testB | 0.6023 | 0.6007 | 0.4245 | 0.4302 |
| paper_meld_ft_ctx9_a20 | eval_testA | 0.9404 | 0.9404 | 0.9373 | 0.9352 |
| paper_meld_ft_ctx9_a20 | eval_testB | 0.5801 | 0.5870 | 0.4356 | 0.4639 |
| paper_transfer_acted_to_meld_ft_ctx9 | eval_testB | 0.5375 | 0.5657 | 0.4171 | 0.4639 |

---

## Appendix B) Complete results inventory: outputs (exploratory history)

The table below is an exact formatted view of `outputs/summary.csv`.

| run | eval | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| _tmp_eval_smoke | eval | 0.4272 | 0.4224 | 0.2657 | 0.2641 |
| exp1_robust_classbal | eval_testA | 0.7442 | 0.7591 | 0.7469 | 0.7501 |
| exp1_robust_classbal | eval_testB | 0.4483 | 0.4232 | 0.2384 | 0.2358 |
| exp2_robust_classbal_moddrop | eval_testA | 0.7442 | 0.7611 | 0.7441 | 0.7513 |
| exp2_robust_classbal_moddrop | eval_testB | 0.4268 | 0.4161 | 0.2481 | 0.2469 |
| exp3_robust_classbal_moddrop_ema | eval_testA | 0.7603 | 0.7684 | 0.7566 | 0.7581 |
| exp3_robust_classbal_moddrop_ema | eval_testB | 0.4203 | 0.4158 | 0.2469 | 0.2501 |
| exp4_robust_moddrop | eval_testA | 0.7548 | 0.7565 | 0.7447 | 0.7512 |
| exp4_robust_moddrop | eval_testB | 0.4310 | 0.4228 | 0.2592 | 0.2570 |
| exp5_robust_moddrop_ema | eval_testA | 0.7572 | 0.7655 | 0.7508 | 0.7591 |
| exp5_robust_moddrop_ema | eval_testB | 0.4272 | 0.4224 | 0.2657 | 0.2641 |
| mfcc80_bs128 | eval | 0.4544 | 0.4232 | 0.2369 | 0.2377 |
| mfcc80_bs128_testA | eval | 0.7233 | 0.7244 | 0.7142 | 0.7196 |
| robust_mfcc80_bs128 | eval | 0.7670 | 0.7708 | 0.7597 | 0.7593 |
| robust_mfcc80_bs128_testB | eval | 0.4375 | 0.4254 | 0.2531 | 0.2500 |
| run_1 | eval | 0.1540 | 0.0411 | 0.0381 | 0.1429 |
| run_2 | eval | 0.7572 | 0.7584 | 0.7498 | 0.7521 |
| run_best | eval_testA | 0.7540 | 0.7646 | 0.7521 | 0.7557 |
| run_best | eval_testB | 0.4521 | 0.4399 | 0.2660 | 0.2632 |
| run_ctx | eval_testA | 0.7427 | 0.7510 | 0.7414 | 0.7420 |
| run_ctx | eval_testB | 0.4444 | 0.4250 | 0.2483 | 0.2444 |
| run_ctx_meld_ft | eval_testA | 0.7419 | 0.7506 | 0.7407 | 0.7455 |
| run_ctx_meld_ft | eval_testB | 0.4031 | 0.4120 | 0.2558 | 0.2568 |
| run_hf_e2e_1 | eval_testA | 0.0606 | 0.0069 | 0.0163 | 0.1429 |
| run_hf_e2e_1 | eval_testB | 0.4812 | 0.3127 | 0.0928 | 0.1429 |
| run_hf_e2e_all_1 | eval_testA | 0.6955 | 0.6952 | 0.6812 | 0.6755 |
| run_hf_e2e_all_1 | eval_testB | 0.4176 | 0.4265 | 0.2653 | 0.2690 |
| run_hf_e2e_all_stage1_1 | eval_testA | 0.7278 | 0.7191 | 0.6820 | 0.6582 |
| run_hf_e2e_all_stage1_1 | eval_testB | 0.6207 | 0.5818 | 0.3789 | 0.3650 |
| run_hf_e2e_all_stage1_auto | eval_testA | 0.7278 | 0.7191 | 0.6820 | 0.6582 |
| run_hf_e2e_all_stage1_auto | eval_testB | 0.6207 | 0.5818 | 0.3789 | 0.3650 |
| run_hf_e2e_all_stage2_1 | eval_testA | 0.9328 | 0.9328 | 0.9288 | 0.9260 |
| run_hf_e2e_all_stage2_1 | eval_testB | 0.6000 | 0.5861 | 0.4120 | 0.4064 |
| run_hf_e2e_all_stage2_auto | eval_testA | 0.9268 | 0.9268 | 0.9263 | 0.9166 |
| run_hf_e2e_all_stage2_auto | eval_testB | 0.5981 | 0.5954 | 0.4250 | 0.4265 |
| run_hf_e2e_meld_ft_1 | eval_testA | 0.1808 | 0.1859 | 0.1646 | 0.1790 |
| run_hf_e2e_meld_ft_1 | eval_testB | 0.5579 | 0.5917 | 0.4387 | 0.5027 |
| run_hf_e2e_meld_ft_ctx7_a15 | eval_testA | 0.9278 | 0.9277 | 0.9241 | 0.9163 |
| run_hf_e2e_meld_ft_ctx7_a15 | eval_testB | 0.6000 | 0.5979 | 0.4234 | 0.4224 |
| run_hf_e2e_meld_ft_ctx9_a20 | eval_testA | 0.9308 | 0.9307 | 0.9264 | 0.9217 |
| run_hf_e2e_meld_ft_ctx9_a20 | eval_testB | 0.5943 | 0.5939 | 0.4196 | 0.4270 |
| run_iemocap_mfcc80 | eval | 0.4544 | 0.4232 | 0.2369 | 0.2377 |
| run_meld_ft_from_stage2_ctx7 | eval_testA | 0.9364 | 0.9362 | 0.9322 | 0.9237 |
| run_meld_ft_from_stage2_ctx7 | eval_testB | 0.6199 | 0.6116 | 0.4333 | 0.4257 |
| run_meld_ft_from_stage2_ctx9 | eval_testA | 0.9333 | 0.9331 | 0.9292 | 0.9238 |
| run_meld_ft_from_stage2_ctx9 | eval_testB | 0.6088 | 0.6067 | 0.4359 | 0.4339 |
| synth2_AT | eval | 0.5833 | 0.5806 | 0.5889 | 0.6333 |
| synth_A | eval | 0.5833 | 0.5242 | 0.5440 | 0.6222 |
| synth_AT | eval | 0.0833 | 0.0556 | 0.0556 | 0.0833 |
| synth_AVT | eval | 0.2500 | 0.2523 | 0.2315 | 0.2333 |
| synth_T | eval | 0.3333 | 0.2372 | 0.2650 | 0.3611 |
