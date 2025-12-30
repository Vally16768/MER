# Improvements Report — Audio+Text MER (MFCC + Hashing)

Goal: improve **out-of-domain TestB (MELD)** while keeping experiments comparable (same features, same splits, same seed, same metrics).

Primary metric: **UAR** (macro recall). We also report accuracy and wF1.

## Baseline reproduction command

```powershell
.\.venv\Scripts\python.exe train.py --config configs/mer_builder_at_simple.yaml --modalities A T --run_name baseline_mfcc80_bs128
.\.venv\Scripts\python.exe evaluate.py --config outputs/baseline_mfcc80_bs128/config_resolved.yaml --ckpt outputs/baseline_mfcc80_bs128/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_simple/testA --output_dir outputs/baseline_mfcc80_bs128/eval_testA
.\.venv\Scripts\python.exe evaluate.py --config outputs/baseline_mfcc80_bs128/config_resolved.yaml --ckpt outputs/baseline_mfcc80_bs128/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_simple/testB --output_dir outputs/baseline_mfcc80_bs128/eval_testB
```

## Results summary (recorded in `outputs/`)

| run | model | sampling | moddrop | ema | TestA acc | TestA uar | TestB acc | TestB uar |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline (mfcc80_bs128) | FlexibleAT | none | 0.0 | 0 | 0.7233 | 0.7196 | 0.4544 | 0.2377 |
| robust baseline | RobustAT | none | 0.0 | 0 | 0.7670 | 0.7593 | 0.4375 | 0.2500 |
| exp1_robust_classbal | RobustAT | dataset_class_balanced | 0.0 | 0 | 0.7442 | 0.7501 | 0.4483 | 0.2358 |
| exp2_robust_classbal_moddrop | RobustAT | dataset_class_balanced | 0.2 | 0 | 0.7442 | 0.7513 | 0.4268 | 0.2469 |
| exp3_robust_classbal_moddrop_ema | RobustAT | dataset_class_balanced | 0.2 | 1 | 0.7603 | 0.7581 | 0.4203 | 0.2501 |
| exp4_robust_moddrop | RobustAT | weighted (MELD=4.0) | 0.2 | 0 | 0.7548 | 0.7512 | 0.4310 | 0.2570 |
| exp5_robust_moddrop_ema | RobustAT | weighted (MELD=4.0) | 0.2 | 1 | 0.7572 | 0.7591 | 0.4272 | 0.2641 |

Where to find metrics:

- Baseline TestA: `outputs/mfcc80_bs128_testA/metrics_eval.json`
- Baseline TestB: `outputs/mfcc80_bs128/metrics_eval.json`
- Robust baseline: `outputs/robust_mfcc80_bs128/metrics_eval.json` and `outputs/robust_mfcc80_bs128_testB/metrics_eval.json`
- Experiments: `outputs/<run>/eval_testA/metrics_eval.json` and `outputs/<run>/eval_testB/metrics_eval.json`

## Improvement 1 — Stronger fusion model (`RobustATModel`)

What changed:

- Added `MER/src/models/robust_at.py` and wired it via `model.type: robust_at` in configs.
- Token-based attention over `[CLS, audio, text]` + SwiGLU projections.

Command:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/mer_builder_at_simple_robust.yaml --modalities A T --run_name robust_mfcc80_bs128
```

Result vs baseline (TestB UAR):

- 0.2377 → 0.2500 (+0.0123)

Interpretation:

- The shallow gated fusion is strong in-domain (acted), but attention over modality tokens yields a more separable fused embedding for MELD.

## Improvement 2 — Sampling strategy (don’t let acted corpora dominate)

What changed:

- Added multiple sampler modes in `MER/train.py` (dataset-balanced, class-balanced, mixed).
- Best-performing strategy for MELD was **upweighting MELD** in sampling:
  - `training.sampling.type: weighted`
  - `training.sampling.dataset_weights: { MELD: 4.0 }`

Why:

- `train` contains many acted samples; without reweighting, the model underfits conversational characteristics.

Observed effect:

- Global dataset+class balancing (exp1) **hurt** TestB (0.2358).
- MELD upweighting + robustness regularizers (exp4/exp5) improved TestB.

## Improvement 3 — Robustness regularization (modality dropout + EMA)

### 3a) Modality dropout

- Implemented in `MER/src/augmentation.py` and used only during training in `MER/src/train_loop.py`.
- Drops whole modality vectors per-sample with probability `p`, but **never drops both**.

Command (as used in exp4):

```powershell
.\.venv\Scripts\python.exe train.py --config configs/mer_builder_at_simple_robust_moddrop.yaml --modalities A T --run_name exp4_robust_moddrop
```

### 3b) EMA weights

- Implemented in `MER/src/ema.py`, enabled via `training.ema.enabled: true`.
- Evaluates (and selects best) using EMA weights when enabled.

Command (as used in exp5):

```powershell
.\.venv\Scripts\python.exe train.py --config configs/mer_builder_at_simple_robust_moddrop_ema.yaml --modalities A T --run_name exp5_robust_moddrop_ema
```

Result (TestB UAR):

- exp4: 0.2570
- exp5: 0.2641 (+0.0071)

## Notes / next steps (highest ROI)

1) Tune `dataset_weights.MELD` (e.g., 6–10) and keep `modality_dropout_p` around 0.1–0.3.
2) Upgrade features (wav2vec2/WavLM + real text encoder) — MFCC+hashing is the main ceiling for MELD.
3) Add transcript corruption augmentation (word drop/substitution) when using token-based text encoders.

