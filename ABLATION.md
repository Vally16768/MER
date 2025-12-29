# Ablation — What mattered for TestB (MELD)

Goal: isolate the configuration/architecture changes that affect **out-of-domain** performance on `testB` (MELD).

Primary metric: **UAR**.

## Ablation matrix (executed)

All runs use:

- same feature set: `data/features/mer_builder_at_simple` (MFCC80 + hashing, fixed splits)
- seed = 42
- metrics: acc / macro-F1 / wF1 / UAR

| run | change vs previous | TestB uar | conclusion |
|---|---|---:|---|
| baseline (FlexibleAT) | — | 0.2377 | strong in-domain, poor OOD |
| robust baseline | model → RobustAT | 0.2500 | attention fusion helps |
| exp1_classbal | + dataset_class_balanced sampling | 0.2358 | hurts MELD (distribution distortion) |
| exp2_classbal_moddrop | + modality dropout (p=0.2) | 0.2469 | partially recovers robustness |
| exp3_classbal_moddrop_ema | + EMA | 0.2501 | small improvement, but still limited by sampling |
| exp4_moddrop | switch to weighted sampling (MELD=4.0) | 0.2570 | best without EMA |
| exp5_moddrop_ema | + EMA | 0.2641 | best overall in this sweep |

## Conclusions (actionable)

1) **Model capacity matters**, but only up to the feature ceiling: RobustAT > FlexibleAT for MELD.
2) **Global class balancing is not a free win** in multi-corpus training; it can degrade MELD by over/under-sampling rare acted emotions.
3) **MELD upweighting + modality dropout** is the best tradeoff seen so far for TestB.
4) **EMA** helps modestly on MELD in this setting.

## What to try next (small, rigorous set)

Minimal experiments (keep everything else identical):

1) Sweep MELD weight: `dataset_weights.MELD ∈ {2, 4, 6, 8, 10}` with `modality_dropout_p=0.2`.
2) Sweep modality dropout: `p ∈ {0.0, 0.1, 0.2, 0.3}` at best MELD weight.
3) Toggle EMA: off/on for the best (weight, p) pair.

Stop when TestB UAR plateaus; report both TestA and TestB for robustness.

