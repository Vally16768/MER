# Experiments and Results Report (Paper Suite)

This document summarizes all experiments currently recorded under `outputs_paper/`. Metrics are sourced from `outputs_paper/summary.csv` (generated from each evaluation’s `metrics_eval.json`).

## 1) Evaluation protocol

Metrics reported:
- Accuracy (acc)
- Weighted F1 (wf1) [primary for MELD and IEMOCAP]
- Macro F1 (macro_f1)
- Unweighted Average Recall (uar)

Splits and evaluation sets:
- TestA (acted, in-domain): aggregated acted corpora held-out split (`testA` in the unified manifest).
- TestB (conversational, out-of-domain): MELD official test split (`testB` in the unified manifest).
- LOCO holdout: test-like split for the held-out corpus (MELD uses `testB`, MEAD uses `testA`, others use `test`).
- IEMOCAP-4 CV: 5-fold session-disjoint protocol, 4-class label set.

## 2) Two-stage training (ALL corpora)

Stage 1 (head warmup, encoders frozen) and Stage 2 (full fine-tuning). Metrics below are TestA and TestB.

| Run | Eval | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| paper_all_stage1 | eval_testA | 0.7278 | 0.7191 | 0.6820 | 0.6582 |
| paper_all_stage1 | eval_testB | 0.6207 | 0.5818 | 0.3789 | 0.3650 |
| paper_all_stage2 | eval_testA | 0.9391 | 0.9396 | 0.9336 | 0.9371 |
| paper_all_stage2 | eval_testB | 0.6126 | 0.6036 | 0.4272 | 0.4204 |

Interpretation:
- Stage 2 substantially improves TestA (acted) and modestly improves TestB (MELD), confirming a domain shift toward conversational data.

## 3) MELD context sweep (Audio+Text, MELD fine-tuning from stage2)

### TestB (MELD)
| Run | Context | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| paper_meld_ft_ctx0_a20 | ctx=0 | 0.6157 | 0.6092 | 0.4368 | 0.4330 |
| paper_meld_ft_ctx3_a20 | ctx=3 | 0.5969 | 0.5976 | 0.4268 | 0.4375 |
| paper_meld_ft_ctx7_a20 | ctx=7 | 0.6023 | 0.6007 | 0.4245 | 0.4302 |
| paper_meld_ft_ctx9_a20 | ctx=9 | 0.5801 | 0.5870 | 0.4356 | 0.4639 |

### TestA (acted)
| Run | Context | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| paper_meld_ft_ctx0_a20 | ctx=0 | 0.9399 | 0.9401 | 0.9366 | 0.9374 |
| paper_meld_ft_ctx3_a20 | ctx=3 | 0.9394 | 0.9394 | 0.9366 | 0.9340 |
| paper_meld_ft_ctx7_a20 | ctx=7 | 0.9391 | 0.9393 | 0.9365 | 0.9346 |
| paper_meld_ft_ctx9_a20 | ctx=9 | 0.9404 | 0.9404 | 0.9373 | 0.9352 |

Interpretation:
- Best TestB weighted F1 is achieved at ctx=0, while ctx=9 yields the highest TestB UAR.
- TestA remains consistently high across all context settings (small differences).

## 4) MELD modality ablation (ctx=9, MELD fine-tuning)

| Run | Modalities | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| paper_meld_ctx9_a20_audio_only | A | 0.3054 | 0.3441 | 0.2504 | 0.2869 |
| paper_meld_ctx9_a20_text_only | T | 0.6011 | 0.5932 | 0.4163 | 0.4218 |
| paper_meld_ctx9_a20_audio_text | A+T | 0.5770 | 0.5842 | 0.4255 | 0.4470 |

Interpretation:
- Text dominates MELD performance; audio-only is much weaker.
- Audio+Text yields higher UAR than text-only, but slightly lower weighted F1 on TestB.

## 5) Cross-domain transfer (Acted -> MELD)

| Run | Eval | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| paper_acted_stage1 | eval_meld_testB | 0.2257 | 0.1959 | 0.1341 | 0.1865 |
| paper_acted_stage2 | eval_meld_testB | 0.2410 | 0.1988 | 0.1837 | 0.2556 |
| paper_transfer_acted_to_meld_ft_ctx9 | eval_testB | 0.5375 | 0.5657 | 0.4171 | 0.4639 |

Interpretation:
- Training only on acted data does not generalize to MELD.
- Fine-tuning on MELD recovers performance substantially.

## 6) LOCO (Leave-One-Corpus-Out) generalization

| Run | Held-out corpus | acc | wf1 | macro_f1 | uar |
| --- | --- | --- | --- | --- | --- |
| paper_loco_exclude_CREMA-D | CREMA-D | 0.7171 | 0.7145 | 0.6145 | 0.6182 |
| paper_loco_exclude_EmoV-DB | EmoV-DB | 0.9149 | 0.9163 | 0.5151 | 0.5048 |
| paper_loco_exclude_ESD | ESD | 1.0000 | 1.0000 | 0.7143 | 0.7143 |
| paper_loco_exclude_IEMOCAP | IEMOCAP | 0.6504 | 0.6481 | 0.5044 | 0.5087 |
| paper_loco_exclude_MEAD | MEAD | 0.9197 | 0.9201 | 0.9148 | 0.9134 |
| paper_loco_exclude_MELD | MELD | 0.5475 | 0.5598 | 0.3952 | 0.4130 |
| paper_loco_exclude_RAVDESS | RAVDESS | 0.7667 | 0.7607 | 0.7533 | 0.7579 |

Interpretation:
- Largest drop appears on MELD holdout (conversational domain shift).
- Very high results on MEAD and ESD reflect strong acted-domain consistency; note that ESD has fewer label classes, which can inflate metrics.

## 7) IEMOCAP-4 (5-fold CV, 4 classes)

Aggregated across 3 seeds and 5 folds (n=15 per modality). Values are mean +/- std.

| Modality | acc mean +/- std | wf1 mean +/- std | uar mean +/- std | n |
| --- | --- | --- | --- | --- |
| A | 0.6550 +/- 0.0162 | 0.6549 +/- 0.0181 | 0.6268 +/- 0.0218 | 15 |
| T | 0.6576 +/- 0.0228 | 0.6606 +/- 0.0257 | 0.6217 +/- 0.0339 | 15 |
| A+T | 0.7131 +/- 0.0211 | 0.7148 +/- 0.0203 | 0.6919 +/- 0.0194 | 15 |

Interpretation:
- Audio+Text consistently outperforms single-modality baselines.
- Variance across folds is small, indicating stable results.

## 8) Best results summary

Best TestA (acted, in-domain):
- `paper_meld_ft_ctx9_a20` with wf1=0.9404, acc=0.9404, uar=0.9352

Best TestB (MELD, conversational) by weighted F1:
- `paper_meld_ft_ctx0_a20` with wf1=0.6092, acc=0.6157, uar=0.4330

Best TestB by UAR:
- `paper_meld_ft_ctx9_a20` with uar=0.4639

## 9) Artifacts and traceability

For each run under `outputs_paper/<run_name>/`:
- `config_resolved.yaml` (exact config snapshot)
- `checkpoints/best.pt` and `checkpoints/last.pt`
- `metrics_train.csv`, `metrics_val.csv` (training logs)
- `eval_*/metrics_eval.json` (evaluation metrics)
- `eval_*/confusion_matrix.csv` and `eval_*/confusion_matrix.png`

Aggregate tables:
- `outputs_paper/summary.csv`
- `outputs_paper/comparison_report.md`

## Appendix A) Full run inventory (all entries from outputs_paper/summary.csv)

| Run | Eval | acc | wf1 | macro_f1 | uar |
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
| paper_loco_exclude_ESD | eval_holdout | 1.0000 | 1.0000 | 0.7143 | 0.7143 |
| paper_loco_exclude_EmoV-DB | eval_holdout | 0.9149 | 0.9163 | 0.5151 | 0.5048 |
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
