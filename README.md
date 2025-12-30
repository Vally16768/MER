# Multi-Corpus Emotion Recognition Method based on Cross-Modal Gated Attention Fusion

The official repository for "Multi-Corpus Emotion Recognition Method based on Cross-Modal Gated Attention Fusion", [Pattern Recognition Letters](https://www.sciencedirect.com/science/article/pii/S0167865525000662)

## Overview

This study addresses key challenges in **automatic emotion recognition (ER)**, particularly the limitations of single-corpus training that hinder generalizability. To overcome these issues, the authors introduce a novel **multi-corpus, multimodal ER method** evaluated using a **leave-one-corpus-out (LOCO) protocol**. This approach incorporates fine-tuned encoders for audio, video, and text, combined through a context-independent gated attention mechanism for cross-modal feature fusion.

## Pipeline Overview

![Pipeline Diagram](https://smil-spcras.github.io/MER/static/img/pipeline.png)

## Components of the Proposed Method

- **Visual Encoder**: Fine-tuned ResNet-50.
- **Acoustic Encoder**: Fine-tuned emotional wav2vec2.
- **Linguistic Encoder**: Fine-tuned RoBERTa model.
- **Audio/Video Segment Aggregation**: Extraction of feature statistics (mean and standard deviation (STD) values).
- **Multimodal Feature Aggregation**: Cross-modal, context-independent gated attention mechanism.

## Key Results

The proposed method achieves **state-of-the-art performance** across multiple benchmark corpora, including **MOSEI, MELD, IEMOCAP, and AFEW**. The study reveals that models trained on **MELD** demonstrate superior cross-corpus generalization. Additionally, **AFEW** annotations show strong alignment with other corpora, resulting in the best cross-corpus performance. These findings validate the robustness and applicability of the method across diverse real-world scenarios.

## User Guide

### Accessing the Models

The pre-trained models are available [here](https://drive.google.com/drive/folders/1NTVQatpihMwe5im_LZqAAouzAQWVjRwx?usp=sharing).

### Inference

To predict emotions for your multimodal files, configure the `config.toml` file with paths to the models and files, and then run `python src/inference.py`.

### Training

To train the multimodal model, first extract features from your data `python src/avt_feature_extraction.py`.

Then, initiate training with `python src/train_avt_model.py`.

Ensure the `config.toml` file is properly configured for both steps.

### Training on `mer_dataset_builder` (Audio+Text)

GPU note: `configs/mer_builder_at_simple.yaml` uses `training.device: auto` (CUDA if available, otherwise CPU). If you see a warning about CUDA not available, reinstall a CUDA-enabled PyTorch build.

1) Extract features (MFCC for audio + hashing for text) into pickle files:
`python extract_at_features.py --processed_dir mer_dataset_builder/data/processed --out_dir data/features/mer_builder_at_simple --n_mfcc 80 --num_workers 16`

2) Train:
`python train.py --config configs/mer_builder_at_simple.yaml --modalities A T --run_name run_1`

This creates `outputs/run_1/` (including `outputs/run_1/config_resolved.yaml` and `outputs/run_1/checkpoints/best.pt`).

3) Evaluate (after training finishes) on TestA (acted) vs TestB (MELD):
`python evaluate.py --config outputs/run_1/config_resolved.yaml --ckpt outputs/run_1/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_simple/testA`
`python evaluate.py --config outputs/run_1/config_resolved.yaml --ckpt outputs/run_1/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_simple/testB`

### Full pipeline (PowerShell) from scratch

1) Build the unified dataset (downloads where possible; some corpora require manual placement under `mer_dataset_builder/data/raw/*`):
`cd mer_dataset_builder`
`python -m pip install -e .`
`python -m mer_builder all --datasets mead meld ravdess cremadh esd emovdb iemocap --raw_dir data/raw --out_dir data/processed --mead_contempt map_to_disgust --emovdb_sleepy map_to_neutral --audio_failure replace_with_silence --num_workers 16`
`cd ..`

2) Extract features (MFCC mean+std + hashing text). For better MELD (TestB), include speaker + short dialogue history:
`python extract_at_features.py --processed_dir mer_dataset_builder/data/processed --out_dir data/features/mer_builder_at_simple_ctx --n_mfcc 80 --include_speaker_in_text --meld_context_window 3 --num_workers 16`

3) Train:
`python train.py --config configs/mer_builder_at_simple_robust_moddrop_ema.yaml --modalities A T --run_name run_ctx --set data.train_dir=data/features/mer_builder_at_simple_ctx/train --set data.val_dir=data/features/mer_builder_at_simple_ctx/val --set data.eval_dir=data/features/mer_builder_at_simple_ctx/testB`

4) Evaluate:
`python evaluate.py --config outputs/run_ctx/config_resolved.yaml --ckpt outputs/run_ctx/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_simple_ctx/testA --output_dir outputs/run_ctx/eval_testA`
`python evaluate.py --config outputs/run_ctx/config_resolved.yaml --ckpt outputs/run_ctx/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_simple_ctx/testB --output_dir outputs/run_ctx/eval_testB`

4) Optional: model visualization:
`python visualize_flexible_at.py --input_dim_a 160 --input_dim_t 768 --graph`

Graphviz is required for the PNG graph. Install on Windows:
`winget install Graphviz.Graphviz`
Then restart the terminal so `dot` is on PATH.

CUDA PyTorch (Windows) install example:
`python -m pip uninstall -y torch torchvision torchaudio`
`python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
Verify: `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"`

### Reports / audits

- `ARCHITECTURE.md`: data flow + model shapes
- `BASELINE_REPORT.md`: reproducible baseline commands + metrics
- `DIAGNOSIS.md`: correctness/performance audit findings
- `IMPROVEMENTS_REPORT.md`: experiment table + commands
- `ABLATION.md`: minimal ablation conclusions

Windows one-command baseline:
`powershell -ExecutionPolicy Bypass -File scripts/run_baseline_at.ps1 -RunName baseline_mfcc80_bs128`

### Strong features for better TestB (MELD)

The simple MFCC+hashing features are fast, but they generalize poorly to conversational MELD. For better TestB, extract embeddings with HuggingFace encoders:

1) Extract features (will download HF models on first run unless already cached):
`python extract_at_features_hf.py --processed_dir mer_dataset_builder/data/processed --out_dir data/features/mer_builder_at_hf --audio_model microsoft/wavlm-base --text_model roberta-base --device auto --batch_size 4 --include_speaker_in_text --meld_context_window 3`

2) Train (dropout is set to 0 in the config):
`python train.py --config configs/mer_builder_at_hf.yaml --modalities A T --run_name run_hf_1`

3) Evaluate on TestA vs TestB:
`python evaluate.py --config outputs/run_hf_1/config_resolved.yaml --ckpt outputs/run_hf_1/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_hf/testA`
`python evaluate.py --config outputs/run_hf_1/config_resolved.yaml --ckpt outputs/run_hf_1/checkpoints/best.pt --set data.eval_dir=data/features/mer_builder_at_hf/testB`

### End-to-end HuggingFace training (best quality)

Fine-tune the encoders end-to-end directly from the unified manifest (no feature pickles):

Train:
`python train_hf_e2e.py --config configs/mer_builder_at_hf_e2e.yaml --run_name run_hf_e2e_1`

Recommended (v2): more MELD context, longer audio, no modality-dropout:
`python train_hf_e2e.py --config configs/mer_builder_at_hf_e2e_all_v2.yaml --run_name run_hf_e2e_all_v2_1`

2-stage (often best): head warmup -> fine-tune
`python train_hf_e2e.py --config configs/mer_builder_at_hf_e2e_all_stage1_head_warmup.yaml --run_name run_hf_e2e_all_stage1_1`
`python train_hf_e2e.py --config configs/mer_builder_at_hf_e2e_all_stage2_finetune.yaml --run_name run_hf_e2e_all_stage2_1 --set training.init_ckpt=outputs/run_hf_e2e_all_stage1_1/checkpoints/best.pt`

Larger (more GPU RAM; enables gradient checkpointing):
`python train_hf_e2e.py --config configs/mer_builder_at_hf_e2e_large.yaml --run_name run_hf_e2e_large_1`

Optional: MELD-only fine-tune (often improves TestB). Start from a multi-corpus HF run:
`python train_hf_e2e.py --config configs/mer_builder_at_hf_e2e_meld_ft.yaml --run_name run_hf_e2e_meld_ft_1 --set training.init_ckpt=outputs/run_hf_e2e_1/checkpoints/best.pt`

Evaluate TestB (MELD):
`python evaluate_hf_e2e.py --config outputs/run_hf_e2e_1/config_resolved.yaml --ckpt outputs/run_hf_e2e_1/checkpoints/best.pt --set data.eval_splits=[testB] --output_dir outputs/run_hf_e2e_1/eval_testB`

Evaluate TestA (acted):
`python evaluate_hf_e2e.py --config outputs/run_hf_e2e_1/config_resolved.yaml --ckpt outputs/run_hf_e2e_1/checkpoints/best.pt --set data.eval_splits=[testA] --output_dir outputs/run_hf_e2e_1/eval_testA`

Offline note: pass local paths to `model.audio_model` / `model.text_model` (and set `TRANSFORMERS_OFFLINE=1`) if you can't download models.

Windows note: HF end-to-end uses a multiprocessing DataLoader; if you still hit worker issues, set `--set data.num_workers=0` to run single-process.

### Speeding up pickle features (optional)

Per-sample pickle I/O can bottleneck training. Convert a split directory into memmapped arrays:

`python build_memmap_features.py --in_dir data/features/mer_builder_at_simple/train --out_dir data/features/mer_builder_at_simple_memmap/train`

## Citation

If you are using our models in your research, please consider to cite research:

<div class="highlight highlight-text-bibtex notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">@article</span>{<span class="pl-en">RYUMINA2024</span>,
  <span class="pl-s">title</span>        = <span class="pl-s"><span class="pl-pds">{</span>Multi-Corpus Emotion Recognition Method based on Cross-Modal Gated Attention Fusion<span class="pl-pds">}</span></span>,
  <span class="pl-s">author</span>       = <span class="pl-s"><span class="pl-pds">{</span>Elena Ryumina and Dmitry Ryumin and Alexandr Axyonov and Denis Ivanko and Alexey Karpov<span class="pl-pds">}</span></span>,
  <span class="pl-s">journal</span>      = <span class="pl-s"><span class="pl-pds">{</span>Pattern Recognition Letters<span class="pl-pds">}</span></span>,
  <span class="pl-s">year</span>         = <span class="pl-s"><span class="pl-pds">{</span>2024<span class="pl-pds">}</span></span>,
}</div>

## Acknowledgments

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
