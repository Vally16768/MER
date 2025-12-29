param(
  [int]$NumWorkers = 4,
  [int]$Seeds = 1,
  [string]$Config = "configs/mer_builder_at_hf_e2e_iemocap4_cv.yaml",
  [string]$RunPrefix = "paper_iemocap4",
  [string]$OutputRoot = "outputs_paper"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoRoot

function Ensure-Train {
  param(
    [string]$RunName,
    [string[]]$ExtraArgs = @()
  )

  $ckpt = "$OutputRoot/$RunName/checkpoints/best.pt"
  if (Test-Path $ckpt) {
    Write-Host "==> Skip train (exists): $RunName"
    return
  }

  Write-Host "`n==> Train: $RunName"
  $args = @(
    "train_hf_e2e.py",
    "--config", $Config,
    "--run_name", $RunName,
    "--set", "output.root_dir=$OutputRoot",
    "--set", "data.num_workers=$NumWorkers"
  ) + $ExtraArgs
  python @args
  if ($LASTEXITCODE -ne 0) { throw "Train failed: $RunName" }
}

function Ensure-Eval {
  param(
    [string]$RunName,
    [string]$EvalName,
    [string[]]$SetArgs = @()
  )

  $outDir = "$OutputRoot/$RunName/$EvalName"
  $metrics = "$outDir/metrics_eval.json"
  if (Test-Path $metrics) {
    Write-Host "==> Skip eval (exists): $RunName/$EvalName"
    return
  }

  Write-Host "`n==> Eval: $RunName/$EvalName"
  $cfg = "$OutputRoot/$RunName/config_resolved.yaml"
  $ckpt = "$OutputRoot/$RunName/checkpoints/best.pt"
  $args = @("evaluate_hf_e2e.py", "--config", $cfg, "--ckpt", $ckpt, "--output_dir", $outDir) + $SetArgs
  python @args
  if ($LASTEXITCODE -ne 0) { throw "Eval failed: $RunName/$EvalName" }
}

for ($seed = 1; $seed -le $Seeds; $seed++) {
  for ($fold = 1; $fold -le 5; $fold++) {
    $ses = ("Ses0{0}" -f $fold)
    $excludeTrain = $ses
    $includeEval = $ses

    foreach ($mod in @("AT", "A", "T")) {
      $run = "$RunPrefix`_seed$seed`_fold$fold`_$mod"
      $extra = @(
        "--set", "training.seed=$seed",
        "--set", "data.filter.exclude_speaker_regex_train=$excludeTrain",
        "--set", "data.filter.include_speaker_regex_val=$includeEval",
        "--set", "data.filter.include_speaker_regex_eval=$includeEval"
      )
      if ($mod -eq "A") { $extra += @("--modalities", "A") }
      elseif ($mod -eq "T") { $extra += @("--modalities", "T") }
      else { $extra += @("--modalities", "A", "T") }

      Ensure-Train -RunName $run -ExtraArgs $extra
      Ensure-Eval -RunName $run -EvalName "eval_fold" -SetArgs @("--set", "data.eval_splits=[train,val,test]")
    }
  }
}

Write-Host "`n==> Refreshing outputs/summary.csv and outputs/comparison_report.md"
python scripts/collect_results.py $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "collect_results.py failed" }

Write-Host "`nAll done."

Pop-Location
