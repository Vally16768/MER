param(
  [int]$NumWorkers = 4
)

$ErrorActionPreference = "Stop"

function Eval-Run {
  param(
    [string]$RunName
  )
  Write-Host "`n==> Eval TestB: $RunName"
  python evaluate_hf_e2e.py --config "outputs/$RunName/config_resolved.yaml" --ckpt "outputs/$RunName/checkpoints/best.pt" --set "data.eval_splits=[testB]" --output_dir "outputs/$RunName/eval_testB"
  if ($LASTEXITCODE -ne 0) { throw "Eval TestB failed: $RunName" }

  Write-Host "`n==> Eval TestA: $RunName"
  python evaluate_hf_e2e.py --config "outputs/$RunName/config_resolved.yaml" --ckpt "outputs/$RunName/checkpoints/best.pt" --set "data.eval_splits=[testA]" --set "data.filter={}" --output_dir "outputs/$RunName/eval_testA"
  if ($LASTEXITCODE -ne 0) { throw "Eval TestA failed: $RunName" }
}

Write-Host "`n==> Stage1: head warmup"
python train_hf_e2e.py --config "configs/mer_builder_at_hf_e2e_all_stage1_head_warmup.yaml" --run_name "run_hf_e2e_all_stage1_auto" --set "data.num_workers=$NumWorkers"
if ($LASTEXITCODE -ne 0) { throw "Stage1 training failed" }

Eval-Run -RunName "run_hf_e2e_all_stage1_auto"

Write-Host "`n==> Stage2: fine-tune"
python train_hf_e2e.py --config "configs/mer_builder_at_hf_e2e_all_stage2_finetune.yaml" --run_name "run_hf_e2e_all_stage2_auto" --set "training.init_ckpt=outputs/run_hf_e2e_all_stage1_auto/checkpoints/best.pt" --set "data.num_workers=$NumWorkers"
if ($LASTEXITCODE -ne 0) { throw "Stage2 training failed" }

Eval-Run -RunName "run_hf_e2e_all_stage2_auto"

Write-Host "`nAll stages finished."
