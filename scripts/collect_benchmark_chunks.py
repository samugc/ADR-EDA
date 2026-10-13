#!/usr/bin/env python3
"""Collect chunked dynamic benchmark outputs.

The chunked runner writes one `per_run_metrics.csv` per problem/algorithm pair.
This helper merges completed chunks and produces a lightweight progress report
without depending on pandas.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List


PROBLEMS = [
    "dzdt1_step",
    "dzdt1_sine",
    "dzdt2_shape",
    "dzdt3_shocks",
    "fda1_lite",
    "fda2_lite",
    "dconstrained_zdt1",
    "ddtlz2_3obj",
    "df1_t10",
    "df2_t10",
    "df3_t10",
    "df4_t10",
    "df5_t10",
    "df6_t10",
]

ALGORITHMS = [
    "dnsga2_b",
    "mnsga2",
    "sgea",
    "trnsga2",
    "pfpa",
    "tda_dmoea",
    "ab_dmoea",
    "adr_eda",
]

METRICS = [
    "migd",
    "mhv",
    "final_igd",
    "final_hv",
    "post_change_igd",
    "post_change_hv",
    "recovery_time",
    "runtime_sec",
]


def _is_number(value: str) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect(outdir: Path) -> None:
    chunk_root = outdir / "chunks"
    all_rows: List[Dict[str, str]] = []
    chunk_rows: List[Dict[str, str]] = []

    for problem in PROBLEMS:
        for algorithm in ALGORITHMS:
            chunk_dir = chunk_root / problem / algorithm
            metrics_path = chunk_dir / "per_run_metrics.csv"
            summary_path = chunk_dir / "summary.json"
            run_log = chunk_dir / "run.log"
            status = "missing"
            n_rows = 0
            if metrics_path.exists():
                rows = _read_rows(metrics_path)
                all_rows.extend(rows)
                n_rows = len(rows)
                status = "complete" if summary_path.exists() and n_rows > 0 else "partial"
            elif run_log.exists():
                status = "started_no_metrics"
            chunk_rows.append(
                {
                    "problem": problem,
                    "algorithm": algorithm,
                    "status": status,
                    "n_rows": str(n_rows),
                    "chunk_dir": str(chunk_dir),
                }
            )

    if all_rows:
        _write_csv(outdir / "per_run_metrics.csv", all_rows, list(all_rows[0].keys()))
    _write_csv(outdir / "chunk_completion.csv", chunk_rows, list(chunk_rows[0].keys()))

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        if row.get("status") == "ok":
            grouped[row.get("algorithm", "")].append(row)

    summary_rows: List[Dict[str, str]] = []
    for algorithm in ALGORITHMS:
        rows = grouped.get(algorithm, [])
        out = {"algorithm": algorithm, "n_ok": str(len(rows))}
        for metric in METRICS:
            values = [float(r[metric]) for r in rows if _is_number(r.get(metric, ""))]
            out[f"{metric}_mean"] = f"{mean(values):.6g}" if values else ""
            out[f"{metric}_std"] = f"{stdev(values):.6g}" if len(values) > 1 else ""
        summary_rows.append(out)

    if summary_rows:
        _write_csv(outdir / "summary_compact.csv", summary_rows, list(summary_rows[0].keys()))

    n_complete = sum(1 for r in chunk_rows if r["status"] == "complete")
    n_failed_or_partial = sum(1 for r in chunk_rows if r["status"] not in {"complete", "missing"})
    report = [
        "# Chunked Campaign Collection",
        "",
        f"- Completed chunks: {n_complete}/{len(chunk_rows)}",
        f"- Started but incomplete chunks: {n_failed_or_partial}",
        f"- Run rows collected: {len(all_rows)}",
        "",
        "Outputs:",
        "- `per_run_metrics.csv`: merged rows from completed/partial chunks.",
        "- `chunk_completion.csv`: status for each problem/algorithm pair.",
        "- `summary_compact.csv`: lightweight algorithm-level means/stds over ok rows.",
        "",
    ]
    (outdir / "COLLECTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    collect(args.outdir)


if __name__ == "__main__":
    main()
