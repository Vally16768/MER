from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    run: str
    eval: str
    acc: float
    macro_f1: float
    wf1: float
    uar: float


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _find_metrics_files(outputs_dir: Path) -> list[Path]:
    # metrics files are written by evaluate.py / evaluate_hf_e2e.py as:
    #   outputs/<run>/eval_*/metrics_eval.json
    # We also accept:
    #   outputs/<run>/metrics_eval.json
    files = sorted(outputs_dir.rglob("metrics_eval.json"))
    return [p for p in files if p.is_file()]


def _infer_run_and_eval(metrics_path: Path) -> tuple[str, str]:
    # Expected:
    #   outputs/<run>/eval_xxx/metrics_eval.json
    # Or:
    #   outputs/<run>/metrics_eval.json
    parent = metrics_path.parent
    if parent.name.lower().startswith("eval_"):
        eval_name = parent.name
        run_name = parent.parent.name
    else:
        eval_name = "eval"
        run_name = parent.name
    return run_name, eval_name


def _load_one(metrics_path: Path) -> Row | None:
    try:
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    run, ev = _infer_run_and_eval(metrics_path)
    m = obj.get("metrics") or {}
    return Row(
        run=run,
        eval=ev,
        acc=_as_float(m.get("accuracy")),
        macro_f1=_as_float(m.get("macro_f1")),
        wf1=_as_float(m.get("weighted_f1")),
        uar=_as_float(m.get("uar")),
    )


def _write_summary_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "eval", "acc", "macro_f1", "wf1", "uar"])
        for r in rows:
            w.writerow([r.run, r.eval, r.acc, r.macro_f1, r.wf1, r.uar])


def _write_comparison_md(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Evaluation Comparison Report")
    lines.append("")
    lines.append("| run | eval | acc | macro_f1 | wf1 | uar |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(f"| {r.run} | {r.eval} | {r.acc} | {r.macro_f1} | {r.wf1} | {r.uar} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    outputs_dir = Path("outputs")
    if len(argv) >= 2:
        outputs_dir = Path(argv[1])
    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
        return 2

    rows: list[Row] = []
    for p in _find_metrics_files(outputs_dir):
        r = _load_one(p)
        if r is not None:
            rows.append(r)

    rows.sort(key=lambda x: (x.run.lower(), x.eval.lower()))
    if not rows:
        print("No metrics_eval.json files found.")
        return 0

    _write_summary_csv(rows, outputs_dir / "summary.csv")
    _write_comparison_md(rows, outputs_dir / "comparison_report.md")
    print(f"Wrote: {outputs_dir / 'summary.csv'}")
    print(f"Wrote: {outputs_dir / 'comparison_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

