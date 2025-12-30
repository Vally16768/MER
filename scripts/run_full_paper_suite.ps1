param(
  # All new runs go under this folder, e.g. outputs_paper/<run_name>/...
  [string]$OutputRoot = "outputs_paper",

  # DataLoader workers (set conservatively on Windows).
  [int]$NumWorkers = 4,

  # Multi-seed for the key MELD best-setting (optional).
  [int]$MeldSeeds = 1,

  # IEMOCAP 5-fold CV seeds (optional; can be slow).
  [int]$IemocapSeeds = 1,

  # Optional long suites:
  [switch]$SkipIemocapCV,
  [switch]$SkipLOCO,
  [switch]$SkipTransfer
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoRoot

function Ensure-Train {
  param(
    [string]$Config,
    [string]$RunName,
    [string[]]$ExtraArgs = @()
  )

  $runDir = Join-Path $OutputRoot $RunName
  $ckpt = Join-Path $runDir "checkpoints/best.pt"
  if (Test-Path $ckpt) {
    Write-Host "==> Skip train (exists): $runDir"
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

  $runDir = Join-Path $OutputRoot $RunName
  $cfg = Join-Path $runDir "config_resolved.yaml"
  $ckpt = Join-Path $runDir "checkpoints/best.pt"
  $outDir = Join-Path $runDir $EvalName
  $metrics = Join-Path $outDir "metrics_eval.json"

  if (Test-Path $metrics) {
    Write-Host "==> Skip eval (exists): $outDir"
    return
  }

  Write-Host "`n==> Eval: $RunName/$EvalName"
  $args = @(
    "evaluate_hf_e2e.py",
    "--config", $cfg,
    "--ckpt", $ckpt,
    "--output_dir", $outDir
  ) + $SetArgs

  python @args
  if ($LASTEXITCODE -ne 0) { throw "Eval failed: $RunName/$EvalName" }
}

function Require-Ckpt {
  param([string]$RunName)
  $runDir = Join-Path $OutputRoot $RunName
  $ckpt = Join-Path $runDir "checkpoints/best.pt"
  if (-not (Test-Path $ckpt)) { throw "Missing checkpoint: $ckpt" }
  return $ckpt
}

function Refresh-Reports {
  Write-Host "`n==> Refreshing summary/comparison under $OutputRoot"
  python scripts/collect_results.py $OutputRoot
  if ($LASTEXITCODE -ne 0) { throw "collect_results.py failed" }
}

Write-Host "Repo root: $repoRoot"
Write-Host "Output root: $OutputRoot"

# ---------------------------------------------------------------------------
# 1) Two-stage training (ALL corpora): stage1 (head warmup) -> stage2 (finetune)
# ---------------------------------------------------------------------------
$stage1 = "paper_all_stage1"
$stage2 = "paper_all_stage2"

Ensure-Train -Config "configs/mer_builder_at_hf_e2e_all_stage1_head_warmup.yaml" -RunName $stage1
Ensure-Eval -RunName $stage1 -EvalName "eval_testA" -SetArgs @("--set", "data.eval_splits=[testA]", "--set", "data.filter={}")
Ensure-Eval -RunName $stage1 -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")

$initStage2 = Require-Ckpt -RunName $stage1
Ensure-Train -Config "configs/mer_builder_at_hf_e2e_all_stage2_finetune.yaml" -RunName $stage2 -ExtraArgs @("--set", "training.init_ckpt=$initStage2")
Ensure-Eval -RunName $stage2 -EvalName "eval_testA" -SetArgs @("--set", "data.eval_splits=[testA]", "--set", "data.filter={}")
Ensure-Eval -RunName $stage2 -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")

$bestAll = Require-Ckpt -RunName $stage2

# ---------------------------------------------------------------------------
# 2) MELD context sweep (utterance-only vs context)
# ---------------------------------------------------------------------------
$meldSweep = @(
  @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx0_a20.yaml"; run = "paper_meld_ft_ctx0_a20" },
  @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx3_a20.yaml"; run = "paper_meld_ft_ctx3_a20" },
  @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx7_a20.yaml"; run = "paper_meld_ft_ctx7_a20" },
  @{ cfg = "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml"; run = "paper_meld_ft_ctx9_a20" }
)

foreach ($e in $meldSweep) {
  Ensure-Train -Config $e.cfg -RunName $e.run -ExtraArgs @("--set", "training.init_ckpt=$bestAll")
  Ensure-Eval -RunName $e.run -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")
  Ensure-Eval -RunName $e.run -EvalName "eval_testA" -SetArgs @("--set", "data.eval_splits=[testA]", "--set", "data.filter={}")
}

# ---------------------------------------------------------------------------
# 3) MELD modality ablation (Text-only / Audio-only / Audio+Text) on testB
# ---------------------------------------------------------------------------
Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "paper_meld_ctx9_a20_text_only" `
  -ExtraArgs @("--modalities", "T", "--set", "training.init_ckpt=$bestAll")
Ensure-Eval -RunName "paper_meld_ctx9_a20_text_only" -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")

Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "paper_meld_ctx9_a20_audio_only" `
  -ExtraArgs @("--modalities", "A", "--set", "training.init_ckpt=$bestAll")
Ensure-Eval -RunName "paper_meld_ctx9_a20_audio_only" -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")

Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "paper_meld_ctx9_a20_audio_text" `
  -ExtraArgs @("--modalities", "A", "T", "--set", "training.init_ckpt=$bestAll")
Ensure-Eval -RunName "paper_meld_ctx9_a20_audio_text" -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")

# ---------------------------------------------------------------------------
# 4) Optional: Cross-domain transfer (Acted -> MELD) + adaptation
# ---------------------------------------------------------------------------
if (-not $SkipTransfer) {
  # Acted-only training (exclude MELD) -> evaluate on MELD testB.
  $actedStage1 = "paper_acted_stage1"
  $actedStage2 = "paper_acted_stage2"

  $excludeMeld = "MELD"
  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_all_stage1_head_warmup.yaml" -RunName $actedStage1 -ExtraArgs @(
    "--set", "data.filter.exclude_datasets_train=[$excludeMeld]",
    "--set", "data.filter.exclude_datasets_val=[$excludeMeld]"
  )
  Ensure-Eval -RunName $actedStage1 -EvalName "eval_meld_testB" -SetArgs @("--set", "data.eval_splits=[testB]", "--set", "data.filter.include_datasets_eval=[MELD]")

  $initActed2 = Require-Ckpt -RunName $actedStage1
  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_all_stage2_finetune.yaml" -RunName $actedStage2 -ExtraArgs @(
    "--set", "training.init_ckpt=$initActed2",
    "--set", "data.filter.exclude_datasets_train=[$excludeMeld]",
    "--set", "data.filter.exclude_datasets_val=[$excludeMeld]"
  )
  Ensure-Eval -RunName $actedStage2 -EvalName "eval_meld_testB" -SetArgs @("--set", "data.eval_splits=[testB]", "--set", "data.filter.include_datasets_eval=[MELD]")

  # Adaptation: fine-tune on MELD from acted-only stage2.
  $bestActed = Require-Ckpt -RunName $actedStage2
  Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName "paper_transfer_acted_to_meld_ft_ctx9" -ExtraArgs @("--set", "training.init_ckpt=$bestActed")
  Ensure-Eval -RunName "paper_transfer_acted_to_meld_ft_ctx9" -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")
}

# ---------------------------------------------------------------------------
# 5) Optional: LOCO (Leave-One-Corpus-Out) – very expensive
# ---------------------------------------------------------------------------
if (-not $SkipLOCO) {
  $corpora = @("MEAD", "RAVDESS", "CREMA-D", "ESD", "EmoV-DB", "IEMOCAP", "MELD")
  foreach ($c in $corpora) {
    $run = "paper_loco_exclude_$c"
    Ensure-Train -Config "configs/mer_builder_at_hf_e2e_all_stage2_finetune.yaml" -RunName $run -ExtraArgs @(
      "--set", "training.init_ckpt=$bestAll",
      "--set", "data.filter.exclude_datasets_train=[$c]",
      "--set", "data.filter.exclude_datasets_val=[$c]",
      "--set", "data.num_workers=0"
    )

    # Evaluate on the held-out corpus on its "test"-like split.
    if ($c -eq "MELD") {
      Ensure-Eval -RunName $run -EvalName "eval_holdout" -SetArgs @("--set", "data.eval_splits=[testB]", "--set", "data.filter.include_datasets_eval=[MELD]")
    } else {
      # NOTE: MEAD uses split=testA (acted in-domain eval) rather than split=test.
      if ($c -eq "MEAD") {
        Ensure-Eval -RunName $run -EvalName "eval_holdout" -SetArgs @("--set", "data.eval_splits=[testA]", "--set", "data.filter.include_datasets_eval=[$c]")
      } else {
        Ensure-Eval -RunName $run -EvalName "eval_holdout" -SetArgs @("--set", "data.eval_splits=[test]", "--set", "data.filter.include_datasets_eval=[$c]")
      }
    }
  }
}

# ---------------------------------------------------------------------------
# 6) Optional: Multi-seed stability (MELD best setting)
# ---------------------------------------------------------------------------
if ($MeldSeeds -gt 1) {
  for ($seed = 1; $seed -le $MeldSeeds; $seed++) {
    $run = "paper_meld_ft_ctx9_a20_seed$seed"
    Ensure-Train -Config "configs/mer_builder_at_hf_e2e_meld_ft_ctx9_a20.yaml" -RunName $run -ExtraArgs @(
      "--set", "training.init_ckpt=$bestAll",
      "--set", "training.seed=$seed"
    )
    Ensure-Eval -RunName $run -EvalName "eval_testB" -SetArgs @("--set", "data.eval_splits=[testB]")
  }
}

# ---------------------------------------------------------------------------
# 7) Optional: IEMOCAP-4 5-fold CV (runs in a separate script)
# ---------------------------------------------------------------------------
if (-not $SkipIemocapCV) {
  if ($IemocapSeeds -lt 1) { $IemocapSeeds = 1 }
  & "$repoRoot/scripts/run_iemocap4_cv.ps1" -NumWorkers $NumWorkers -Seeds $IemocapSeeds -RunPrefix "paper_iemocap4" -OutputRoot $OutputRoot | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "IEMOCAP CV script failed" }
}

Refresh-Reports
Write-Host "`nPaper suite finished."

Pop-Location
