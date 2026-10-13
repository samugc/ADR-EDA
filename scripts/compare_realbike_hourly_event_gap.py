#!/usr/bin/env python3
"""Compare hourly aggregate and chronological event-level RealBike summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COMMON_METRICS = [
    ("utility_score", "utility_score_mean"),
    ("service_rate", "service_rate_mean"),
    ("relocation_km", "relocation_km_mean"),
    ("reoptimization_epochs", "reoptimization_epochs_mean"),
    ("reuse_epochs", "reuse_epochs_mean"),
    ("rejected", "rejected_mean"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly-summary", type=Path, required=True)
    parser.add_argument("--event-summary", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    hourly_rows = {row["policy"]: row for row in read_csv(args.hourly_summary)}
    event_rows = {
        (row["policy"], row["metric"]): row
        for row in read_csv(args.event_summary)
    }
    out = []
    for policy, hourly in sorted(hourly_rows.items()):
        for event_metric, hourly_metric in COMMON_METRICS:
            event = event_rows.get((policy, event_metric))
            if event is None or hourly_metric not in hourly:
                continue
            hourly_mean = float(hourly[hourly_metric])
            event_mean = float(event["mean"])
            out.append(
                {
                    "policy": policy,
                    "metric": event_metric,
                    "hourly_mean": round(hourly_mean, 6),
                    "event_mean": round(event_mean, 6),
                    "event_minus_hourly": round(event_mean - hourly_mean, 6),
                }
            )
    write_csv(args.out_csv, out)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
