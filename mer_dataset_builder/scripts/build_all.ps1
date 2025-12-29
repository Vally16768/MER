param(
  [string]$RawDir = "data/raw",
  [string]$OutDir = "data/processed",
  [int]$NumWorkers = 8,
  [ValidateSet("drop", "map_to_disgust")]
  [string]$MeadContempt = "drop"
)

$ErrorActionPreference = "Stop"

python -m pip install -e .
python -m mer_builder all --raw_dir $RawDir --out_dir $OutDir --num_workers $NumWorkers --mead_contempt $MeadContempt
