param(
  [string]$RunName = "baseline_mfcc80_bs128",
  [int]$NumWorkers = 16,
  [int]$Nmfcc = 80,
  [int]$BatchSize = 128
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Py = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
if (!(Test-Path $Py)) {
  $Py = "python"
}

$RunDir = Join-Path $RepoRoot ("outputs\\" + $RunName)
if (Test-Path $RunDir) {
  throw "Run dir already exists: $RunDir. Pick a new -RunName or delete the folder to avoid appending logs."
}

$FeatureRoot = Join-Path $RepoRoot "data\\features\\mer_builder_at_simple"
$RequiredSplits = @("train", "val", "testA", "testB")
$Missing = @($RequiredSplits | Where-Object { -not (Test-Path (Join-Path $FeatureRoot $_)) })
if ($Missing.Count -gt 0) {
  Write-Host "Missing feature splits ($($Missing -join ', ')); running feature extraction..."
  & $Py extract_at_features.py `
    --processed_dir mer_dataset_builder/data/processed `
    --out_dir data/features/mer_builder_at_simple `
    --n_mfcc $Nmfcc `
    --num_workers $NumWorkers
}

& $Py train.py `
  --config configs/mer_builder_at_simple.yaml `
  --modalities A T `
  --run_name $RunName `
  --set training.batch_size=$BatchSize

& $Py evaluate.py `
  --config ("outputs/" + $RunName + "/config_resolved.yaml") `
  --ckpt ("outputs/" + $RunName + "/checkpoints/best.pt") `
  --set data.eval_dir=data/features/mer_builder_at_simple/testA `
  --output_dir ("outputs/" + $RunName + "/eval_testA")

& $Py evaluate.py `
  --config ("outputs/" + $RunName + "/config_resolved.yaml") `
  --ckpt ("outputs/" + $RunName + "/checkpoints/best.pt") `
  --set data.eval_dir=data/features/mer_builder_at_simple/testB `
  --output_dir ("outputs/" + $RunName + "/eval_testB")

Write-Host "Done. Outputs: outputs\\$RunName"

