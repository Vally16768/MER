# Multimodal Emotion Recognition (MER): Academic Report

## Abstract
This document describes (1) the construction of a multimodal **audio+text** emotion recognition dataset by unifying multiple public corpora and (2) an empirical study of audio–text fusion models with a focus on robustness. All labels are mapped into a shared 7-class label space compatible with MELD:
**{anger, disgust, fear, joy, sadness, surprise, neutral}**.

Robustness is assessed across two distinct domains:
- **TestA (acted / in-domain)**: controlled, acted speech.
- **TestB (conversational / out-of-domain)**: natural conversational speech from MELD.

## 1. Unified dataset

### 1.1 Source corpora: what they are and what they provide
The unified dataset aggregates the following corpora (all provide speech audio and emotion labels; most provide transcripts, either directly or derivable from metadata):

- **MEAD (Part0, speech)**: acted, multi-speaker emotional speech with scripted utterances across emotions; well-suited for speaker-disjoint evaluation.
- **MELD (Raw)**: conversational dialogue extracted from *Friends*, with official utterance transcripts and emotion labels; the primary out-of-domain benchmark in this project.
- **RAVDESS (speech only)**: acted emotional speech with controlled sentences; only the spoken modality is used (no singing).
- **CREMA-D**: acted emotional speech with standardized sentences across a large pool of speakers.
- **ESD (Emotional Speech Dataset)**: acted emotional speech with bilingual speakers; contains transcripts in two languages (see §1.2).
- **EmoV-DB (OpenSLR SLR115)**: acted emotional speech based on CMU Arctic-style prompts; emotion labels per utterance.
- **IEMOCAP (full release)**: dialogue-based emotional speech (acted / semi-improvised) with transcripts and speaker identifiers.

### 1.2 Language origin (based on observed transcripts)
Based on the transcripts present in the unified manifest:
- **MEAD, MELD, RAVDESS, CREMA-D, EmoV-DB, IEMOCAP**: predominantly **English** (Latin script).
- **ESD**: strictly **bilingual** with two balanced sub-collections:
  - **17,500** utterances with **English** transcripts (latin_only)
  - **17,500** utterances with **Mandarin Chinese** transcripts (cjk_only)

### 1.3 Shared label space and mapping rules
To unify heterogeneous corpora, all labels are mapped into the 7-class space:
**anger, disgust, fear, joy, sadness, surprise, neutral**.

Key mapping rules:
- **MELD**: already in the 7-class space; kept as-is.
- **CREMA-D**: `happy → joy` (others map directly).
- **RAVDESS**: `happy → joy`, `calm → neutral`; `song` is excluded and only speech is used.
- **ESD**: available labels {neutral, angry, happy, sad, surprise} map to {neutral, anger, joy, sadness, surprise}.
- **MEAD**: includes `contempt`; two policies are supported:
  - drop `contempt`, or
  - map `contempt → disgust` and record the decision in notes.
- **EmoV-DB**: some labels do not map cleanly into the 7-class space (e.g., `sleepy`); two policies are supported:
  - drop such samples, or
  - map `sleepy → neutral` and record the decision in notes.
- **IEMOCAP**: `hap/exc → joy`, `fru → anger`; core labels are retained when they fit the 7-class space.

### 1.4 Dataset size: per corpus, total, acted vs conversational
The unified manifest contains the following number of samples (utterances):

| dataset | samples |
| --- | ---: |
| MEAD | 31,734 |
| MELD | 13,708 |
| RAVDESS | 1,440 |
| CREMA-D | 7,442 |
| ESD | 35,000 |
| EmoV-DB | 6,893 |
| IEMOCAP | 10,039 |
| **Total** | **106,256** |

Domain grouping:
- **Acted (MEAD, RAVDESS, CREMA-D, ESD, EmoV-DB, IEMOCAP)**: **92,548**
- **Conversational (MELD)**: **13,708**

### 1.5 Splits and evaluation protocol
The protocol targets leakage avoidance and robustness measurement:
- **Acted corpora**: **speaker-disjoint** train/val/test splits by speaker identity.
- **MELD**: official train/dev/test splits are preserved.

Evaluation sets:
- **TestA (acted)**: in-domain evaluation on acted speech.
- **TestB (MELD test)**: out-of-domain evaluation on conversational speech.

Comparability note:
- In the **feature-based (MFCC+hashing)** experiments, TestA contains **10,247** samples (acted aggregate) and TestB contains **2,610** samples (MELD test).
- In the **end-to-end (WavLM+RoBERTa)** experiments, TestA was reported on **3,960** samples (acted subset) and TestB on **2,610** samples.

### 1.6 Audio normalization
All audio is normalized to a shared format:
- mono
- 16 kHz
- PCM WAV

For operational robustness, undecodable/corrupted sources are handled via:
- controlled exclusion (drop), or
- replacement with short silence (to preserve dataset cardinality), with explicit logging.

### 1.7 Transcript derivation and audio–text alignment
Transcripts are obtained as follows:
- **MELD**: official CSV utterance text.
- **RAVDESS / CREMA-D**: fixed sentences; transcripts are derived deterministically from sentence codes.
- **ESD**: explicit transcripts in per-utterance text files or per-speaker transcript tables.
- **EmoV-DB**: CMU Arctic-style prompts from metadata.
- **MEAD**: sentence text resolved from corpus metadata, with a robust fallback mapping.
- **IEMOCAP**: official dialogue transcripts mapped by utterance ID.

## 2. Representations (features / embeddings)

### 2.1 MFCC (audio): definition and usage
**MFCC (Mel-Frequency Cepstral Coefficients)** are cepstral features computed on the Mel scale to approximate human auditory perception. In speech, MFCCs capture the spectral envelope and are widely used as compact features for speech and emotion recognition.

In the feature-based pipeline, for each utterance:
- MFCC are computed with **n_mfcc = 80**
- temporal aggregation uses **mean** and **standard deviation**
- the final audio vector has **160 dimensions** (= 2 × n_mfcc)

### 2.2 HashingVectorizer (text): definition and usage
**HashingVectorizer** projects bag-of-words features into a fixed-dimensional space using hashing. Pros: constant memory, no explicit vocabulary, fast. Cons: collisions may mix unrelated tokens.

In the feature-based pipeline, transcripts are mapped into a **768-dimensional** vector.

### 2.3 Conversational context (MELD)
Because MELD is conversational, an utterance can be ambiguous without dialogue history. Two textual context augmentations are used:
- **speaker prefix** (e.g., `speaker=<id>`) to explicitly encode who is speaking;
- **context window**: concatenation of up to N preceding utterances from the same dialogue (no future context), separated by a delimiter.

Experiments evaluate larger context windows (e.g., **7** and **9**) and longer audio segments (e.g., **15s**, **20s**) to improve TestB.

### 2.4 Pretrained encoders (end-to-end)
The end-to-end pipeline uses:
- **WavLM-base** (audio): self-supervised speech encoder producing robust temporal representations.
- **RoBERTa-base** (text): pretrained language encoder producing contextual token representations.

Typical pooling:
- audio: **mean** or **mean+std** over time (mask-aware to ignore padding);
- text: **CLS** representation or mean over tokens.

## 3. Model architectures

### 3.1 Gated fusion (feature-based baseline)
The baseline fuses two vectors (audio and text) via a gating mechanism:
1) LayerNorm and linear projection per modality;
2) gated fusion into a **128-dimensional** latent;
3) **SwiGLU** activation and linear classification into 7 classes.

This architecture is lightweight and efficient, but ultimately limited by the MFCC+hashing representation quality.

### 3.2 Robust fusion via a modality-token Transformer
To improve robustness, fusion is performed by treating modalities as **tokens**:
- token sequence: `[CLS, audio_token, text_token]` (or a subset if a modality is missing);
- each token is projected into **hidden_dim = 256** using **SwiGLU**;
- a stack of **L = 4** Transformer blocks is applied, each with:
  - **Multi-Head Self-Attention** with **8 heads**
  - a feed-forward network with **ffn_mult = 4**
  - residual connections and pre-norm LayerNorm
- pooling uses the CLS token; a small head maps to class logits.

Note: classical positional encoding is unnecessary here because the sequence is not temporal—it is a fixed ordering of modality-type tokens.

### 3.3 End-to-end audio+text (WavLM + RoBERTa + modality-token Transformer)
The end-to-end model uses:
1) an audio encoder (WavLM) producing a time-indexed representation;
2) a text encoder (RoBERTa) producing token-indexed representations;
3) pooling into fixed-dimensional audio/text vectors;
4) fusion using the modality-token Transformer from §3.2.

For stability during fine-tuning:
- the audio feature extractor may be temporarily frozen early in training;
- discriminative learning rates are used (separate rates for encoders vs fusion/head).

## 4. Training and evaluation protocol

### 4.1 Objective
The task is 7-way multi-class classification optimized with cross-entropy.

### 4.2 Metrics
Reported metrics:
- Accuracy
- Macro-F1
- Weighted-F1 (wF1)
- UAR (Unweighted Average Recall): mean recall across classes

UAR is the primary selection metric, as it is more informative under class imbalance (particularly for MELD).

### 4.3 Optimization and regularization (conceptual)
Across experiments, the following techniques are used:
- AdamW with weight decay
- ReduceLROnPlateau scheduling (reacting to UAR stagnation)
- early stopping (patience)
- gradient clipping (max norm)
- mixed precision (AMP) and gradient accumulation (efficiency and stability)
- label smoothing (0.1) and balanced class weights
- EMA (Exponential Moving Average) for stabilizing evaluation (feature-based pipeline)
- modality dropout (dropping one modality during training to reduce shortcut reliance)
- dataset reweighting (upweighting MELD) to counter dominance of acted corpora

### 4.4 Two-stage training (end-to-end)
End-to-end models use a two-stage regime:
- **Stage 1 (head warmup)**: encoders frozen; train fusion/head with higher learning rate.
- **Stage 2 (fine-tuning)**: encoders unfrozen; lower learning rates with encoder-specific rates.

This regime substantially improves TestA and stabilizes training dynamics.

## 5. Experiments and results

### 5.1 Family A: feature-based (MFCC + hashing)
Typical setting:
- audio: MFCC-80 with mean+std aggregation (160-dim)
- text: hashing (768-dim)
- fusion: gated baseline or modality-token Transformer (robust)
- sampling: MELD reweighting (×4) in some experiments
- regularization: label smoothing (0.1); optional modality dropout and EMA

Results (TestA = 10,247; TestB = 2,610):

| experiment | core idea | TestA acc | TestA UAR | TestB acc | TestB UAR |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | baseline gated fusion | 0.7233 | 0.7196 | 0.4544 | 0.2377 |
| A1 | Transformer fusion (robust) | 0.7670 | 0.7593 | 0.4375 | 0.2500 |
| A2 | dataset+class balancing | 0.7442 | 0.7501 | 0.4483 | 0.2358 |
| A3 | A2 + modality dropout (0.2) | 0.7442 | 0.7513 | 0.4268 | 0.2469 |
| A4 | A3 + EMA | 0.7603 | 0.7581 | 0.4203 | 0.2501 |
| A5 | MELD reweighting (×4) + moddrop | 0.7548 | 0.7512 | 0.4310 | 0.2570 |
| A6 | A5 + EMA | 0.7572 | 0.7591 | 0.4272 | 0.2641 |
| A7 | A6 (replicated tuning) | 0.7540 | 0.7557 | 0.4521 | 0.2632 |
| A8 | conversational context (MFCC+hashing) | 0.7427 | 0.7420 | 0.4444 | 0.2444 |
| A9 | A8 + MELD-only fine-tuning | 0.7419 | 0.7455 | 0.4031 | 0.2568 |

Interpretation:
- Changing fusion (A1) improves TestA and slightly improves TestB UAR, but MFCC+hashing remains a ceiling for conversational speech.
- MELD reweighting + moddrop + EMA (A6/A7) yields the best feature-based TestB UAR (~0.26), still far below end-to-end performance.

### 5.2 Family B: end-to-end (WavLM-base + RoBERTa-base)
Typical setting:
- encoders: WavLM-base + RoBERTa-base
- pooling: audio mean+std; text CLS
- fusion: modality-token Transformer (hidden_dim 256, 4 layers, 8 heads)
- training: two-stage; MELD upweighting (×6) during multi-corpus training

Results (TestB = 2,610; TestA = 3,960 for end-to-end reports):

| experiment | core idea | TestA acc | TestA UAR | TestB acc | TestB UAR |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 | initial e2e (under-trained/unstable) | 0.0606 | 0.1429 | 0.4812 | 0.1429 |
| B1 | e2e multi-corpus baseline | 0.6955 | 0.6755 | 0.4176 | 0.2690 |
| B2 | Stage 1 (encoders frozen) | 0.7278 | 0.6582 | 0.6207 | 0.3650 |
| B3 | Stage 2 (fine-tune encoders) | 0.9328 | 0.9260 | 0.6000 | 0.4064 |
| B4 | Stage 2 (replica) | 0.9268 | 0.9166 | 0.5981 | 0.4265 |
| B5 | MELD-only fine-tune (no retention control) | 0.1808 | 0.1790 | 0.5579 | 0.5027 |
| B6 | B3 + MELD fine-tune, ctx=7, 15s audio | 0.9278 | 0.9163 | 0.6000 | 0.4224 |
| B7 | B3 + MELD fine-tune, ctx=9, 20s audio | 0.9308 | 0.9217 | 0.5943 | 0.4270 |
| B8 | best Stage2 + MELD fine-tune, ctx=7, 15s audio | 0.9364 | 0.9237 | 0.6199 | 0.4257 |
| B9 | best Stage2 + MELD fine-tune, ctx=9, 20s audio | 0.9333 | 0.9238 | 0.6088 | 0.4339 |

Interpretation:
- Two-stage training (B2→B3) is critical: it sharply improves TestA and raises TestB vs B1.
- MELD-only fine-tuning (B5) maximizes TestB UAR but severely degrades TestA (over-adaptation).
- MELD fine-tuning from a strong multi-corpus checkpoint (B6–B9) yields the best TestA/TestB trade-off.

### 5.3 Best overall trade-off (good TestA and TestB simultaneously)
Among the reported models, the best overall balance is:
- **B9**: TestA UAR **0.9238**, TestB UAR **0.4339** (TestB acc **0.6088**).

Practically:
- **ctx=9** tends to improve TestB UAR (better class balance),
- **ctx=7** tends to slightly improve TestB accuracy/wF1 (better on dominant classes).

## 5.4 SOTA-aligned experiment suite (recommended TOC)
This section specifies the experiment suite that reviewers typically expect for 2023–2025 **audio+text (no video)** MER work. It is framed as an experiment “table of contents” for a camera-ready paper, emphasizing comparability, methodological correctness, and reproducibility of conclusions.

### 5.4.1 Primary benchmark: MELD (7-class, official split)
**Goal:** establish a main benchmark that is directly comparable to recent work. MELD is conversational; therefore, utterance-only results are informative but not sufficient for SOTA-aligned comparison.

#### (A) Modality ablation (utterance-level)
**Purpose:** quantify the contribution of each modality.

Experiments (same training protocol, MELD official split):
- Text-only
- Audio-only
- Audio+Text

Metrics to report (all three):
- Accuracy
- weighted-F1 (**primary**)
- macro-F1
- UAR

Interpretation guideline:
Audio-only and text-only are baselines; Audio+Text is the primary configuration.

#### (B) Conversational context ablation (Audio+Text)
**Purpose:** quantify the impact of dialogue history, which is a key factor for MELD.

Experiments (same architecture and training policy):
- Context window = 0 (utterance-only)
- Context window = 3 previous utterances
- Context window = 7 previous utterances
- Context window = 9 previous utterances

Metrics:
- weighted-F1 (**primary**)
- Accuracy
- UAR

Interpretation guideline:
Improved performance with context supports the hypothesis that MELD requires conversational modeling.

#### (C) Frozen vs fine-tuned encoders (end-to-end)
**Purpose:** standard end-to-end ablation in recent literature.

Experiments (Audio+Text, using the best context setting):
- Encoders frozen (train fusion/head only)
- Full fine-tuning (two-stage regime)

Reporting:
- MELD test metrics for both
- relative improvements (%)

### 5.4.2 Acted benchmark: IEMOCAP-4 (Audio+Text, 5-fold CV)
**Why:** IEMOCAP-4 is the most widely reported acted SER/MER benchmark for A+T.

Protocol (paper-standard):
- 4 emotions: angry, happy/excited, sad, neutral
- 5-fold cross-validation (speaker/session disjoint)

Experiments:
- Text-only
- Audio-only
- Audio+Text

Metrics:
- weighted-F1 (**primary**)
- (optional but recommended) Accuracy
- mean ± std across folds

Interpretation guideline:
This validates performance on acted speech under a standardized evaluation protocol.

### 5.4.3 Generalization experiments (strongly recommended)
#### (A) Cross-domain transfer: acted → conversational
**Purpose:** explicitly quantify domain shift and adaptation.

Experiments:
1) Train on acted-only → test on MELD
2) Train on all corpora → test on MELD
3) Train on all corpora → fine-tune on MELD → test on MELD

Metrics:
- weighted-F1
- UAR

Interpretation guideline:
Fine-tuning on MELD is typically necessary to adapt to conversational speech.

#### (B) LOCO (Leave-One-Corpus-Out) robustness (A+T)
**Purpose:** “paper-grade” robustness analysis across corpora.

Protocol:
- Exclude one corpus from training
- Evaluate exclusively on the excluded corpus

Reporting:
- weighted-F1 / UAR per excluded corpus
- global mean across held-out corpora

### 5.4.4 Architecture ablations (minimum)
**Purpose:** justify architectural choices.

Experiments (same dataset + same context setting):
- Classic gated fusion baseline
- Modality-token Transformer fusion (RobustAT)

Report on:
- MELD weighted-F1
- MELD UAR

### 5.4.5 Regularization ablations (minimum)
**Purpose:** identify which regularizers materially affect robustness.

Experiments:
- No modality dropout / no EMA
- Modality dropout only
- Modality dropout + EMA

Report:
- MELD test metrics (weighted-F1, UAR)

### 5.4.6 Stability (recommended)
**Purpose:** demonstrate that conclusions are not seed-specific.

Protocol:
- Run at least 3 random seeds for MELD and IEMOCAP-4
- Report mean ± std (weighted-F1 and UAR)

### 5.4.7 Mandatory tables and recommended figures
Tables (minimum):
1) MELD: modality ablation + context ablation
2) IEMOCAP-4: A / T / A+T (5-fold CV)
3) Cross-domain: acted → conversational transfer
4) Architecture: baseline vs robust fusion

Figures (recommended):
- MELD confusion matrix
- Context window vs performance (bar chart)

### 5.4.8 Status relative to this project
The current project already contains:
- Multi-corpus unified dataset (acted + conversational) and a 7-class unified label space.
- End-to-end A+T models with conversational context windows (7 and 9) and two-stage fine-tuning results on TestA/TestB.

The following reviewer-facing items are not yet fully standardized within the current results table and should be added for a paper-complete experimental section:
- MELD modality ablation (A-only / T-only / A+T) under a fixed protocol.
- Context window sweep including context=0 (utterance-only) and context=3 (in addition to 7/9).
- IEMOCAP-4 5-fold cross-validation protocol (4-class mapping and fold aggregation).
- LOCO robustness (leave-one-corpus-out) across all corpora.
- Multi-seed statistics (≥3 seeds) for MELD and IEMOCAP-4.

## 6. Discussion: why TestB remains harder
The central observation is a clear **domain shift** between acted and conversational speech:
- acted: controlled delivery with more “clean” acoustic emotion cues
- MELD: dialogue, interruptions/overlap, broad speaking variability, pragmatic context, and class imbalance (neutral dominates)

Implications:
- MFCC+hashing has an inherent ceiling on MELD (insufficient semantic depth and limited prosodic nuance).
- Pretrained encoders substantially increase performance, yet TestB remains constrained without:
  - richer dialogue context modeling,
  - stronger imbalance-aware training,
  - robustness-oriented augmentation (e.g., transcript corruption, conversational noise).

## 7. Conclusions
1) A unified 7-class multi-corpus dataset enables systematic robustness measurement across acted vs conversational domains.
2) In the feature-based setting, sampling/moddrop/EMA yield incremental gains; representation quality dominates the ceiling.
3) In the end-to-end setting, two-stage training plus MELD fine-tuning with dialogue context provides the best TestA/TestB trade-off.

## Glossary
- **UAR**: mean per-class recall; robust under class imbalance.
- **wF1**: support-weighted F1; reflects global performance dominated by frequent classes.
- **Modality dropout**: dropping one modality during training to reduce shortcut reliance.
- **EMA**: exponential moving average of parameters for more stable evaluation.
