param(
  [string]$BaseCkpt = "outputs/run_hf_e2e_all_stage2_1/checkpoints/best.pt",
  [int]$NumWorkers = 4
)

$ErrorActionPreference = "Stop"

function Run-Experiment {
  param(
    [string]$Config,
    [string]$RunName
  )

  Write-Host "`n==> Train: $RunName"
  python train_hf_e2e.py --config $Config --run_name $RunName --set "training.init_ckpt=$BaseCkpt" --set "data.num_workers=$NumWorkers"
  if ($LASTEXITCODE -ne 0) { throw "Train failed: $RunName" }

  Write-Host "`n==> Eval TestB: $RunName"
  python evaluate_hf_e2e.py --config "outputs/$RunName/config_resolved.yaml" --ckpt "outputs/$RunName/checkpoints/best.pt" --set "data.eval_splits=[testB]" --output_dir "outputs/$RunName/eval_testB"
  if ($LASTEXITCODE -ne 0) { throw "Eval TestB failed: $RunName" }

  Write-Host "`n==> Eval TestA: $RunName"
  python evaluate_hf_e2e.py --config "outputs/$RunName/config_resolved.yaml" --ckpt "outputs/$RunName/checkpoints/best.pt" --set "data.eval_splits=[testA]" --set "data.filter={}" --output_dir "outputs/$RunName/eval_testA"
  if ($LASTEXITCODE -ne 0) { throw "Eval TestA failed: $RunName" }
}

Run-Experiment -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx7_a15.yaml" -RunName "run_hf_e2e_meld_ft_ctx7_a15"
Run-Experiment -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "run_hf_e2e_meld_ft_ctx9_a20"

Write-Host "`nAll experiments finished."
