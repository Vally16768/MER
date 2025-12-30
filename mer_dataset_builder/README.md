# mer_dataset_builder

Production-grade, reproducible builder that unifies multiple **audio + text emotion recognition** corpora into a single manifest with a common **7-class MELD-compatible** label space:

`{anger, disgust, fear, joy, sadness, surprise, neutral}`

Datasets supported:
- **MEAD Part0**
- **MELD Raw**
- **RAVDESS (speech only)**
- **CREMA-D**
- **ESD**
- **EmoV-DB (OpenSLR SLR115)**
- **IEMOCAP (original full release)**

## Repo structure

```
mer_dataset_builder/
  README.md
  pyproject.toml
  src/mer_builder/...
  scripts/
  data/
    raw/
    processed/
```

## Requirements

- Python **3.10+**
- `ffmpeg` + `ffprobe` available on PATH

If `ffmpeg` is missing, the builder prints installation hints. Example installs:
- Windows (winget): `winget install Gyan.FFmpeg`
- macOS (brew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`

If you installed FFmpeg but your current terminal can't find it, set `FFMPEG_BIN_DIR` to the directory containing `ffmpeg.exe` and `ffprobe.exe`, e.g.:

```powershell
$env:FFMPEG_BIN_DIR="C:\path\to\ffmpeg\bin"
```

## Install

From `MER/mer_dataset_builder/`:

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Commands

All commands are run from `MER/mer_dataset_builder/`.
If you run from a different working directory, pass absolute paths (or prefix with `mer_dataset_builder/`) for `--raw_dir` / `--out_dir` to avoid re-downloading into a different folder.

### Download (or validate manual placement)

```bash
python -m mer_builder download --datasets mead meld ravdess cremadh esd emovdb iemocap --raw_dir data/raw
```

Notes:
- Some corpora have additional terms (license agreement / access form). The downloader supports automation but may require an explicit opt-in env var.
- Optional direct-download env vars (if you have a mirror link):
  - `MEAD_PART0_URL`, `MELD_RAW_URL`, `CREMAD_URL`, `ESD_URL`, `EMOVDB_URL`, `RAVDESS_SPEECH_URL`
  - `MELD_CSV_BASE_URL` (override where `train/dev/test_sent_emo.csv` are fetched from if missing)
  - Google Drive file id support: `MEAD_PART0_GDRIVE_ID`, `MELD_RAW_GDRIVE_ID`, `CREMAD_GDRIVE_ID`, `ESD_GDRIVE_ID`
  - Google Drive folder id support (MEAD): `MEAD_PART0_GDRIVE_FOLDER_ID`
  - Actor subset (MEAD): `MEAD_ACTORS="M003,M005,W009"` (optional)
  - License opt-in (ESD): `ESD_ACCEPT_LICENSE=1`
  - OpenSLR mirror selection for EmoV-DB: `OPENSLR_MIRROR` or `EMOVDB_MIRROR` (substring match, e.g. `elda` or `trmal`)
  - CMU Arctic prompts URL override: `CMUARCTIC_URL`
  - `KAGGLE_DOWNLOAD=1` (plus Kaggle credentials) optionally downloads CREMA-D if `kaggle` CLI is installed.
  - `--continue_on_error` continues other datasets and exits non-zero if any failed.

### Manual placement expectations (raw data)

Place raw datasets under `data/raw/<DATASET>/`:

- **MEAD** -> `data/raw/MEAD/`
  - Must contain speech audio files somewhere under the tree (`.wav/.flac/.mp3/.m4a` are scanned).
  - Auto-download: by default, `download mead` crawls the official Google Drive folder and downloads `audio.tar` for each actor into `data/raw/MEAD/archives/`, extracting to `data/raw/MEAD/extracted/<ACTOR>/`.
    - Override folder id: `MEAD_PART0_GDRIVE_FOLDER_ID`
    - Also downloads `data/raw/MEAD/MEAD-supp.pdf` (required for transcript derivation)
  - Transcripts: derived automatically from `MEAD-supp.pdf` (speech corpus list). Optional overrides: `data/raw/MEAD/mead_sentences.csv`
    - Format: `sentence_id,text` (2 columns; header optional) or `emotion,sentence_id,text` (with header)

- **MELD** -> `data/raw/MELD/`
  - Required CSVs:
    - `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`
  - Audio clips must exist under the folder (MELD.Raw commonly provides `.mp4` clips; some mirrors provide `.wav`).
    - Common layouts: `train_splits/`, `dev_splits_complete/`, `output_repeated_splits_test/`
  - Expected filename convention: `dia<Dialogue_ID>_utt<Utterance_ID>.(wav|mp4)`
  - Auto-download: by default, `download meld` fetches `MELD.Raw.tar.gz` from the official UMich mirror (unless `MELD_RAW_URL` / `MELD_RAW_GDRIVE_ID` is set) and extracts nested `train/dev/test` archives automatically.
  - If CSVs are not found in the extracted bundle, the downloader fetches them from the public `declare-lab/MELD` GitHub repo (override with `MELD_CSV_BASE_URL`).

- **RAVDESS** -> `data/raw/RAVDESS/Audio_Speech_Actors_01-24/Actor_*/`
  - Only speech audio is used (song modality is ignored).

- **CREMA-D** -> `data/raw/CREMA-D/`
  - WAVs are usually under `AudioWAV/`.
  - Expected filename convention: `<ACTORID>_<SENTENCECODE>_<EMOTIONCODE>_*.wav`
  - Auto-download: by default, `download cremad` fetches `summaryTable.csv` then downloads WAVs from the official GitHub repo (Git LFS-backed). If any files are missing on that mirror, the downloader fails with a message telling you to use Kaggle or an official archive (`CREMAD_URL`/`CREMAD_GDRIVE_ID`) to get the complete 7,442-clip release.

- **ESD** -> `data/raw/ESD/`
  - WAVs are scanned recursively.
  - Transcripts:
    - Preferred: sibling `.txt` next to each WAV with the transcript text.
    - Also supported (official release layout): per-speaker transcript files like `.../<speaker>/<speaker>.txt` containing `utt_id<TAB>text`.
    - Fallback: a transcript table (`.txt/.csv/.tsv`) under `data/raw/ESD/` containing `utt_id,text`-style mappings.
  - Auto-download: set `ESD_ACCEPT_LICENSE=1` to download from the official Google Drive link (see `https://github.com/HLTSingapore/Emotional-Speech-Data`).

- **EmoV-DB** -> `data/raw/EmoV-DB/`
  - Auto-downloads OpenSLR SLR115 speaker/emotion tarballs into `data/raw/EmoV-DB/archives/` and extracts to `data/raw/EmoV-DB/extracted/`.
  - Downloads `data/raw/EmoV-DB/cmuarctic.data` (CMU Arctic prompts) for transcript lookup.
  - Samples whose emotion labels cannot be cleanly mapped into the 7-class space (e.g., Sleepy) are dropped and logged to `data/processed/stats/dropped_emovdb.csv`.

- **IEMOCAP** -> `data/raw/IEMOCAP_full_release/` (or `data/raw/IEMOCAP/`)
  - Manual download required (license/terms).
  - Expected structure:
    - `Session1/dialog/EmoEvaluation/*.txt`
    - `Session1/dialog/transcriptions/*.txt`
    - `Session1/sentences/wav/**/<utt_id>.wav`
  - Also supported: placing `IEMOCAP_full_release/` next to this repo root (e.g., `MER/mer_dataset_builder/IEMOCAP_full_release/`).
  - Label handling:
    - Maps `hap/exc -> joy`, `fru -> anger`, and keeps core labels (`ang/sad/neu/sur/fea/dis`).
    - If a file has `xxx/oth`, the parser attempts to derive a usable label from categorical coder votes; otherwise it is logged to `data/processed/stats/dropped_iemocap.csv`.

### Prepare (audio normalize + transcripts + mapping + splits + manifest)

```bash
python -m mer_builder prepare --raw_dir data/raw --out_dir data/processed --mead_contempt map_to_disgust --emovdb_sleepy map_to_neutral --audio_failure replace_with_silence --num_workers 16
```

MEAD contempt handling:
- `--mead_contempt drop` (exclude contempt)
- `--mead_contempt map_to_disgust` (map contempt -> disgust, `notes` set)

EmoV-DB sleepy handling (not part of the 7-class space):
- `--emovdb_sleepy drop` (default, excluded from the unified 7-class dataset)
- `--emovdb_sleepy map_to_neutral` (keeps samples, sets `notes`)

Audio failures during normalization:
- `--audio_failure drop` (default; failed clips are removed and recorded in `data/processed/stats/dropped_audio.csv`)
- `--audio_failure replace_with_silence` (keeps samples by writing a short silent WAV and logs `data/processed/stats/replaced_audio.csv`)

### Validate (stats + split checks)

```bash
python -m mer_builder validate --manifest data/processed/meta_manifest.jsonl
```

### Integrity (missing files / duplicates / sanity)

```bash
python -m mer_builder integrity --manifest data/processed/meta_manifest.jsonl --out_dir data/processed
```

### All (download -> prepare -> validate)

```bash
python -m mer_builder all
```

To keep going even if a dataset is missing (and build a partial manifest from what's available):

```bash
python -m mer_builder all --continue_on_error
```

Or use the scripts:
- Linux/macOS: `bash scripts/build_all.sh`
- Windows: `powershell -ExecutionPolicy Bypass -File scripts/build_all.ps1`

## Output

Unified manifests:
- `data/processed/meta_manifest.jsonl`
- `data/processed/meta_manifest.csv`

Audio is normalized to **mono, 16kHz, PCM WAV** under:
- `data/processed/audio/<DATASET>/<id>.wav`

If some source clips are corrupted/unreadable (occasionally happens in large corpora), they are automatically **dropped** during normalization and recorded in:
- `data/processed/stats/dropped_audio.csv`

Manifest schema:
- `id` (unique global ID)
- `dataset` (e.g., `"MEAD"`, `"MELD"`, ...)
- `split` (`train/val/test/meld_train/meld_dev/testA/testB`)
- `speaker_id`
- `audio_path` (relative under `data/processed/audio/`)
- `transcript`
- `emotion` (one of the 7 classes)
- `duration_sec`
- `source_label`
- `notes` (optional)

## Splits and robustness evaluation (TestA vs TestB)

Speaker-disjoint acted splits (80/10/10 by speaker) are created for:
- MEAD, RAVDESS, CREMA-D, ESD, EmoV-DB, IEMOCAP

MELD official splits are preserved:
- `meld_train`, `meld_dev`, and `testB` (MELD official test set)

Two robustness evaluations:
- **TestA_in_domain_acted** -> `split == "testA"` (MEAD acted test speakers)
- **TestB_out_of_domain_conversational** -> `split == "testB"` (MELD conversational test set)

This lets you measure:
- in-domain acted generalization (TestA) vs
- out-of-domain conversational generalization (TestB)

## Transcript sources (by dataset)

- **MELD**: from official `*_sent_emo.csv` (`Utterance` column)
- **RAVDESS**: derived from statement id (2 fixed sentences)
- **CREMA-D**: derived from sentence code (mapping table in `src/mer_builder/prepare/parse_cremad.py`)
- **ESD**: parsed from per-utterance `.txt` files (preferred) or transcript tables found under `data/raw/ESD/`
- **EmoV-DB**: parsed from `txt.done.data` / `prompts` / `metadata.csv` (CMU Arctic style)
- **MEAD**: sentence text resolved via dataset metadata; if missing, provide `data/raw/MEAD/mead_sentences.csv` (`sentence_id,text`)
- **IEMOCAP**: transcripts from `Session*/dialog/transcriptions/*.txt` matched by utterance id
