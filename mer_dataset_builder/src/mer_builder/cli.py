from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mer_builder.config import DATASET_KEYS, DEFAULT_SEED, MEAD_CONTEMPT_MODES
from mer_builder.config import EMOVDB_SLEEPY_MODES
from mer_builder.download.download_cremad import download_cremad
from mer_builder.download.download_emovdb import download_emovdb
from mer_builder.download.download_esd import download_esd
from mer_builder.download.download_iemocap import download_iemocap
from mer_builder.download.download_mead import download_mead
from mer_builder.download.download_meld import download_meld
from mer_builder.download.download_ravdess import download_ravdess
from mer_builder.prepare.build_manifest import prepare_all
from mer_builder.prepare.integrity import check_manifest_integrity
from mer_builder.prepare.validate import validate_manifest
from mer_builder.utils.io import setup_logging


BUILDER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = BUILDER_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = BUILDER_ROOT / "data" / "processed"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )


def _parse_datasets(values: list[str] | None) -> list[str]:
    if not values:
        return DATASET_KEYS.copy()
    normalized = []
    for value in values:
        v = value.strip().lower()
        if v == "cremadh":
            v = "cremad"
        normalized.append(v)
    unknown = sorted(set(normalized) - set(DATASET_KEYS))
    if unknown:
        raise SystemExit(f"Unknown dataset keys: {unknown}. Valid: {DATASET_KEYS}")
    return normalized


def _default_prepare_datasets(raw_dir: Path) -> list[str]:
    candidates = [
        ("mead", "MEAD"),
        ("meld", "MELD"),
        ("ravdess", "RAVDESS"),
        ("cremad", "CREMA-D"),
        ("esd", "ESD"),
        ("emovdb", "EmoV-DB"),
    ]
    present: list[str] = []
    for key, folder in candidates:
        if (raw_dir / folder).exists():
            present.append(key)

    # IEMOCAP is sometimes extracted next to the project (not under raw_dir); we check both.
    if any(
        p.exists()
        for p in [
            raw_dir / "IEMOCAP_full_release",
            raw_dir / "IEMOCAP",
            Path.cwd() / "IEMOCAP_full_release",
            Path.cwd() / "IEMOCAP",
            DEFAULT_RAW_DIR / "IEMOCAP_full_release",
            DEFAULT_RAW_DIR / "IEMOCAP",
        ]
    ):
        present.append("iemocap")
    return present


def _raw_dir_has_any_dataset(raw_dir: Path) -> bool:
    folders = {"MEAD", "MELD", "RAVDESS", "CREMA-D", "ESD", "EmoV-DB", "IEMOCAP_full_release", "IEMOCAP"}
    try:
        entries = {p.name for p in raw_dir.iterdir() if p.exists()}
    except FileNotFoundError:
        return False
    return bool(entries & folders)


def _out_dir_looks_prepared(out_dir: Path) -> bool:
    return (out_dir / "meta_manifest.jsonl").is_file() and (out_dir / "audio").is_dir()


def _resolve_dirs_for_cmd(cmd: str, raw_dir: Path | None, out_dir: Path | None) -> tuple[Path | None, Path | None]:
    """Resolve raw/out dirs robustly when running from outside mer_dataset_builder/.

    If the user passes a relative path that doesn't contain expected datasets, but the
    default mer_dataset_builder/data/* directories do, automatically switch and log a warning.
    """

    log = logging.getLogger("mer_builder")

    resolved_raw: Path | None = None
    if raw_dir is not None:
        resolved_raw = raw_dir if raw_dir.is_absolute() else (Path.cwd() / raw_dir).resolve()
        if not _raw_dir_has_any_dataset(resolved_raw) and _raw_dir_has_any_dataset(DEFAULT_RAW_DIR):
            log.warning(
                "raw_dir=%s doesn't look like it contains datasets; using %s instead. "
                "Tip: run from mer_dataset_builder/ or pass --raw_dir %s",
                resolved_raw,
                DEFAULT_RAW_DIR,
                DEFAULT_RAW_DIR,
            )
            resolved_raw = DEFAULT_RAW_DIR

    resolved_out: Path | None = None
    if out_dir is not None:
        resolved_out = out_dir if out_dir.is_absolute() else (Path.cwd() / out_dir).resolve()
        if cmd in {"prepare", "all"}:
            if not _out_dir_looks_prepared(resolved_out) and _out_dir_looks_prepared(DEFAULT_OUT_DIR):
                log.warning(
                    "out_dir=%s doesn't look like a prepared dir; using %s instead. "
                    "Tip: run from mer_dataset_builder/ or pass --out_dir %s",
                    resolved_out,
                    DEFAULT_OUT_DIR,
                    DEFAULT_OUT_DIR,
                )
                resolved_out = DEFAULT_OUT_DIR

    return resolved_raw, resolved_out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mer_builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download or validate raw datasets.")
    p_dl.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to download/validate (default: all): {DATASET_KEYS}",
    )
    p_dl.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    p_dl.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives even if already present (where supported).",
    )
    p_dl.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue downloading other datasets if one fails; exits non-zero if any failed.",
    )
    _add_common_args(p_dl)

    p_prep = sub.add_parser("prepare", help="Prepare processed audio + unified manifest.")
    p_prep.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    p_prep.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p_prep.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to prepare (default: all present): {DATASET_KEYS}",
    )
    p_prep.add_argument(
        "--mead_contempt",
        default="drop",
        choices=sorted(MEAD_CONTEMPT_MODES),
        help="How to handle MEAD 'contempt' samples.",
    )
    p_prep.add_argument(
        "--emovdb_sleepy",
        default="drop",
        choices=sorted(EMOVDB_SLEEPY_MODES),
        help="How to handle EmoV-DB 'sleepy' samples (not in 7-class label space).",
    )
    p_prep.add_argument("--num_workers", type=int, default=8)
    p_prep.add_argument(
        "--audio_failure",
        default="drop",
        choices=["drop", "replace_with_silence"],
        help="On audio decode/normalize failure, either drop the sample or replace its audio with short silence.",
    )
    p_prep.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_prep.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue preparing other datasets if one fails; exits non-zero if any failed.",
    )
    _add_common_args(p_prep)

    p_val = sub.add_parser("validate", help="Validate a built manifest and print stats.")
    p_val.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUT_DIR / "meta_manifest.jsonl",
    )
    _add_common_args(p_val)

    p_int = sub.add_parser("integrity", help="Check manifest/audio consistency (missing files, duplicates, etc.).")
    p_int.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUT_DIR / "meta_manifest.jsonl",
    )
    p_int.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Processed output directory (defaults to the manifest's parent directory).",
    )
    p_int.add_argument("--print_limit", type=int, default=20)
    _add_common_args(p_int)

    p_all = sub.add_parser("all", help="Run download -> prepare -> validate.")
    p_all.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to run (default: all): {DATASET_KEYS}",
    )
    p_all.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    p_all.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p_all.add_argument(
        "--mead_contempt",
        default="drop",
        choices=sorted(MEAD_CONTEMPT_MODES),
    )
    p_all.add_argument(
        "--emovdb_sleepy",
        default="drop",
        choices=sorted(EMOVDB_SLEEPY_MODES),
    )
    p_all.add_argument("--num_workers", type=int, default=8)
    p_all.add_argument(
        "--audio_failure",
        default="drop",
        choices=["drop", "replace_with_silence"],
    )
    p_all.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue downloading other datasets if one fails; exits non-zero if any failed.",
    )
    _add_common_args(p_all)

    return p


def _run_download(
    datasets: list[str],
    raw_dir: Path,
    *,
    force: bool,
    continue_on_error: bool,
) -> tuple[list[str], list[tuple[str, str]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    successes: list[str] = []
    for ds in datasets:
        try:
            if ds == "mead":
                download_mead(raw_dir, force=force)
            elif ds == "meld":
                download_meld(raw_dir, force=force)
            elif ds == "ravdess":
                download_ravdess(raw_dir, force=force)
            elif ds == "cremad":
                download_cremad(raw_dir, force=force)
            elif ds == "esd":
                download_esd(raw_dir, force=force)
            elif ds == "emovdb":
                download_emovdb(raw_dir, force=force)
            elif ds == "iemocap":
                download_iemocap(raw_dir, force=force)
            else:
                raise SystemExit(f"Unhandled dataset: {ds}")
            successes.append(ds)
        except Exception as e:
            failures.append((ds, str(e)))
            if not continue_on_error:
                raise

    if failures:
        logging.getLogger("mer_builder").error("Download failures (%d): %s", len(failures), [f[0] for f in failures])
        for ds, msg in failures:
            logging.getLogger("mer_builder").error("[%s] %s", ds, msg.splitlines()[0] if msg else "unknown error")
    return successes, failures


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(getattr(args, "log_level", "INFO"))
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        # Robust dir resolution when running from outside mer_dataset_builder/.
        if hasattr(args, "raw_dir") or hasattr(args, "out_dir"):
            raw_dir, out_dir = _resolve_dirs_for_cmd(
                str(args.cmd),
                getattr(args, "raw_dir", None),
                getattr(args, "out_dir", None),
            )
            if raw_dir is not None:
                args.raw_dir = raw_dir
            if out_dir is not None:
                args.out_dir = out_dir

        if args.cmd == "download":
            datasets = _parse_datasets(args.datasets)
            _, failures = _run_download(
                datasets,
                args.raw_dir,
                force=args.force,
                continue_on_error=args.continue_on_error,
            )
            return 1 if failures else 0

        if args.cmd == "prepare":
            datasets = _parse_datasets(args.datasets) if args.datasets else _default_prepare_datasets(args.raw_dir)
            failures = prepare_all(
                raw_dir=args.raw_dir,
                out_dir=args.out_dir,
                datasets=datasets,
                mead_contempt=args.mead_contempt,
                emovdb_sleepy=args.emovdb_sleepy,
                audio_failure=args.audio_failure,
                num_workers=args.num_workers,
                seed=args.seed,
                continue_on_error=args.continue_on_error,
            )
            return 1 if failures else 0

        if args.cmd == "validate":
            validate_manifest(args.manifest)
            return 0

        if args.cmd == "integrity":
            return int(
                check_manifest_integrity(
                    args.manifest,
                    out_dir=args.out_dir,
                    print_limit=int(args.print_limit),
                )
            )

        if args.cmd == "all":
            datasets = _parse_datasets(args.datasets)
            downloaded, dl_failures = _run_download(
                datasets,
                args.raw_dir,
                force=args.force,
                continue_on_error=args.continue_on_error,
            )
            prep_datasets = downloaded if args.continue_on_error else datasets
            prep_failures = prepare_all(
                raw_dir=args.raw_dir,
                out_dir=args.out_dir,
                datasets=prep_datasets,
                mead_contempt=args.mead_contempt,
                emovdb_sleepy=args.emovdb_sleepy,
                audio_failure=args.audio_failure,
                num_workers=args.num_workers,
                seed=args.seed,
                continue_on_error=args.continue_on_error,
            )
            validate_manifest(args.out_dir / "meta_manifest.jsonl")
            return 1 if (dl_failures or prep_failures) else 0

        raise SystemExit(f"Unknown command: {args.cmd}")
    except (RuntimeError, FileNotFoundError) as e:
        logging.getLogger("mer_builder").error("%s", str(e).strip())
        return 2
