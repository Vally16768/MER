from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert per-sample A/T pickle features into fast memmapped arrays.")
    p.add_argument("--in_dir", type=Path, required=True, help="Input split dir with *.pkl files.")
    p.add_argument("--out_dir", type=Path, required=True, help="Output dir (writes meta.json + *.npy).")
    p.add_argument("--modalities", nargs="+", default=["A", "T"], choices=["A", "T"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _list_pkls(in_dir: Path) -> list[Path]:
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pkl", ".pickle"}])
    return files


def _load_one(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")
    return obj


def main() -> int:
    args = _parse_args()
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    modalities = {m.strip().upper() for m in args.modalities if str(m).strip()}
    if not modalities:
        raise ValueError("At least one modality must be selected.")

    files = _list_pkls(in_dir)
    if not files:
        raise FileNotFoundError(f"No pickle files found in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        raise FileExistsError(f"{meta_path} exists. Use --overwrite to rebuild.")

    first = _load_one(files[0])
    audio_dim = int(np.asarray(first["audio_features"]).reshape(-1).shape[0]) if "A" in modalities else None
    text_dim = int(np.asarray(first["text_features"]).reshape(-1).shape[0]) if "T" in modalities else None

    n = len(files)
    labels_path = out_dir / "labels.npy"
    ids_path = out_dir / "ids.txt"

    labels_mm = open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(n,))
    audio_mm = None
    text_mm = None
    if "A" in modalities:
        assert audio_dim is not None
        audio_mm = open_memmap(out_dir / "audio.npy", mode="w+", dtype=np.float32, shape=(n, audio_dim))
    if "T" in modalities:
        assert text_dim is not None
        text_mm = open_memmap(out_dir / "text.npy", mode="w+", dtype=np.float32, shape=(n, text_dim))

    with ids_path.open("w", encoding="utf-8") as f_ids:
        for i, pkl_path in enumerate(tqdm(files, desc="memmap", unit="file")):
            d = _load_one(pkl_path)

            sample_id = str(d.get("id") or pkl_path.stem)
            f_ids.write(sample_id + "\n")

            y = int(d.get("true_label"))
            labels_mm[i] = y

            if audio_mm is not None:
                a = np.asarray(d["audio_features"], dtype=np.float32).reshape(-1)
                if a.shape[0] != audio_dim:
                    raise ValueError(f"audio dim mismatch at {pkl_path.name}: {a.shape[0]} != {audio_dim}")
                audio_mm[i, :] = a

            if text_mm is not None:
                t = np.asarray(d["text_features"], dtype=np.float32).reshape(-1)
                if t.shape[0] != text_dim:
                    raise ValueError(f"text dim mismatch at {pkl_path.name}: {t.shape[0]} != {text_dim}")
                text_mm[i, :] = t

    meta = {
        "count": int(n),
        "audio_dim": int(audio_dim) if audio_dim is not None else None,
        "text_dim": int(text_dim) if text_dim is not None else None,
        "audio_path": "audio.npy" if audio_mm is not None else None,
        "text_path": "text.npy" if text_mm is not None else None,
        "labels_path": "labels.npy",
        "ids_path": "ids.txt",
        "modalities": sorted(modalities),
        "in_dir": str(in_dir),
        "created_by": os.path.basename(__file__),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OK: wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

