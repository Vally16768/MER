$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoRoot

New-Item -ItemType Directory -Force -Path outputs\base_stage2_best | Out-Null
$stage2Best = Get-ChildItem -Path outputs\run_hf_e2e_all_stage2_* -Filter best.pt -Recurse -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if (-not $stage2Best) {
  throw "No stage2 best checkpoint found under outputs\\run_hf_e2e_all_stage2_*\\checkpoints\\best.pt"
}
Copy-Item -Force $stage2Best.FullName outputs\base_stage2_best\best.pt

# MELD fine-tune ctx7
python train_hf_e2e.py `
  --config configs/mer_builder_at_hf_e2e_meld_ft_ctx7_a15.yaml `
  --run_name run_meld_ft_from_stage2_ctx7 `
  --set training.init_ckpt=outputs/base_stage2_best/best.pt

python evaluate_hf_e2e.py `
  --config outputs/run_meld_ft_from_stage2_ctx7/config_resolved.yaml `
  --ckpt outputs/run_meld_ft_from_stage2_ctx7/checkpoints/best.pt `
  --set "data.eval_splits=[testA]" `
  --set "data.filter={}" `
  --output_dir outputs/run_meld_ft_from_stage2_ctx7/eval_testA

python evaluate_hf_e2e.py `
  --config outputs/run_meld_ft_from_stage2_ctx7/config_resolved.yaml `
  --ckpt outputs/run_meld_ft_from_stage2_ctx7/checkpoints/best.pt `
  --set "data.eval_splits=[testB]" `
  --output_dir outputs/run_meld_ft_from_stage2_ctx7/eval_testB

# MELD fine-tune ctx9
python train_hf_e2e.py `
  --config configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml `
  --run_name run_meld_ft_from_stage2_ctx9 `
  --set training.init_ckpt=outputs/base_stage2_best/best.pt

python evaluate_hf_e2e.py `
  --config outputs/run_meld_ft_from_stage2_ctx9/config_resolved.yaml `
  --ckpt outputs/run_meld_ft_from_stage2_ctx9/checkpoints/best.pt `
  --set "data.eval_splits=[testA]" `
  --set "data.filter={}" `
  --output_dir outputs/run_meld_ft_from_stage2_ctx9/eval_testA

python evaluate_hf_e2e.py `
  --config outputs/run_meld_ft_from_stage2_ctx9/config_resolved.yaml `
  --ckpt outputs/run_meld_ft_from_stage2_ctx9/checkpoints/best.pt `
  --set "data.eval_splits=[testB]" `
  --output_dir outputs/run_meld_ft_from_stage2_ctx9/eval_testB

Pop-Location
