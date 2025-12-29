param(
  [int]$NumWorkers = 4,
  [switch]$SkipStage1,
  [switch]$SkipStage2,
  [switch]$SkipMeldSweep,
  [switch]$SkipMeldModalities,
  [string]$Stage1Run = "paper_all_stage1",
  [string]$Stage2Run = "paper_all_stage2"
)

$ErrorActionPreference = "Stop"

function Ensure-Train {
  param(
    [string]$Config,
    [string]$RunName,
    [string[]]$ExtraArgs = @()
  )

  $ckpt = "outputs/$RunName/checkpoints/best.pt"
  if (Test-Path $ckpt) {
    Write-Host "==> Skip train (exists): $RunName"
    return
  }

  Write-Host "`n==> Train: $RunName"
  $args = @("train_hf_e2e.py", "--config", $Config, "--run_name", $RunName, "--set", "data.num_workers=$NumWorkers") + $ExtraArgs
  python @args
  if ($LASTEXITCODE -ne 0) { throw "Train failed: $RunName" }
}

function Ensure-Eval {
  param(
    [string]$RunName,
    [string]$EvalName,
    [string[]]$SetArgs = @()
  )

  $outDir = "outputs/$RunName/$EvalName"
  $metrics = "$outDir/metrics_eval.json"
  if (Test-Path $metrics) {
    Write-Host "==> Skip eval (exists): $RunName/$EvalName"
    return
  }

  Write-Host "`n==> Eval: $RunName/$EvalName"
  $cfg = "outputs/$RunName/config_resolved.yaml"
  $ckpt = "outputs/$RunName/checkpoints/best.pt"
  $args = @("evaluate_hf_e2e.py", "--config", $cfg, "--ckpt", $ckpt, "--output_dir", $outDir) + $SetArgs
  python @args
  if ($LASTEXITCODE -ne 0) { throw "Eval failed: $RunName/$EvalName" }
}

function Resolve-BestCkpt {
  param([string]$RunName)
  $ckpt = "outputs/$RunName/checkpoints/best.pt"
  if (-not (Test-Path $ckpt)) { throw "Missing checkpoint: $ckpt" }
  return $ckpt
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoRoot

Write-Host "Running from repo root: $repoRoot"

if (-not $SkipStage1) {
  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_all_stage1_head_warmup.yaml" -RunName $Stage1Run
  Ensure-Eval -RunName $Stage1Run -EvalName "eval_testA" -SetArgs @("--set", "data.eval_splits=[testA]", "--set", "data.filter={}")
  Ensure-Eval -RunName $Stage1Run -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")
}

if (-not $SkipStage2) {
  $init = Resolve-BestCkpt -RunName $Stage1Run
  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_all_stage2_finetune.yaml" -RunName $Stage2Run -ExtraArgs @("--set", "training.init_ckpt=$init")
  Ensure-Eval -RunName $Stage2Run -EvalName "eval_testA" -SetArgs @("--set", "data.eval_splits=[testA]", "--set", "data.filter={}")
  Ensure-Eval -RunName $Stage2Run -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")
}

$stage2Best = Resolve-BestCkpt -RunName $Stage2Run

if (-not $SkipMeldSweep) {
  # MELD fine-tune context sweep (same architecture/training, only context changes).
  $sweep = @(
    @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx0_a20.yaml"; run = "paper_meld_ft_ctx0_a20" },
    @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx3_a20.yaml"; run = "paper_meld_ft_ctx3_a20" },
    @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx7_a20.yaml"; run = "paper_meld_ft_ctx7_a20" },
    @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml"; run = "paper_meld_ft_ctx9_a20" }
  )

  foreach ($e in $sweep) {
    Ensure-Train -Config $e.cfg -RunName $e.run -ExtraArgs @("--set", "training.init_ckpt=$stage2Best")
    Ensure-Eval -RunName $e.run -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")
    # Cross-domain acted evaluation set (global testA). Needs filter cleared.
    Ensure-Eval -RunName $e.run -EvalName "eval_testA" -SetArgs @("--set", "data.eval_splits=[testA]", "--set", "data.filter={}")
  }
}

if (-not $SkipMeldModalities) {
  # MELD modality ablation on the conversational benchmark (testB).
  # We keep context and training policy fixed, varying only modalities.
  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "paper_meld_ctx9_a20_text_only" -ExtraArgs @("--modalities", "T")
  Ensure-Eval -RunName "paper_meld_ctx9_a20_text_only" -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")

  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "paper_meld_ctx9_a20_audio_only" -ExtraArgs @("--modalities", "A")
  Ensure-Eval -RunName "paper_meld_ctx9_a20_audio_only" -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")

  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "paper_meld_ctx9_a20_audio_text" -ExtraArgs @("--modalities", "A", "T")
  Ensure-Eval -RunName "paper_meld_ctx9_a20_audio_text" -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")
}

Write-Host "`n==> Refreshing outputs/summary.csv and outputs/comparison_report.md"
python scripts/collect_results.py outputs
if ($LASTEXITCODE -ne 0) { throw "collect_results.py failed" }

Write-Host "`nAll done."

Pop-Location
