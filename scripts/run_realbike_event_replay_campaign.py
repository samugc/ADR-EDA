#!/usr/bin/env python3
"""Run a compact RealBike event-replay validation campaign.

The campaign is intentionally scriptable and resumable-ish: each command writes
to its own output directory, and progress.csv records the status of every stage.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "docs" / "realbike_dmoo"
CAMPAIGN = OUT_ROOT / "event_replay_multicity_osrm_s10_h168_2026_05_25"
PROGRESS = CAMPAIGN / "progress.csv"

SEEDS = "2026,2027,2028,2029,2030,2031,2032,2033,2034,2035"
COMMON_EVENT_ARGS = [
    "--seeds",
    SEEDS,
    "--policies",
    "no_relocation,static_twice_daily_rebalance,rolling_horizon_rebalance,adr_eda_station",
    "--station-limit",
    "25",
    "--max-hours",
    "168",
    "--lookahead-hours",
    "4",
    "--relocation-budget-bikes",
    "8",
    "--selective-threshold",
    "10",
    "--adr-high-pressure",
    "4",
    "--adr-drift-threshold",
    "0.08",
    "--service-reward",
    "1.0",
    "--rejection-penalty",
    "2.0",
    "--relocation-km-cost",
    "0.035",
    "--relocation-action-cost",
    "0.02",
    "--eda-population",
    "16",
    "--eda-generations",
    "6",
    "--eda-elite-frac",
    "0.35",
    "--eda-initial-sigma",
    "0.18",
    "--eda-imbalance-penalty",
    "0.35",
]
COMMON_HOURLY_ARGS = [
    "--seeds",
    SEEDS,
    "--station-limit",
    "25",
    "--max-hours",
    "168",
    "--lookahead-hours",
    "4",
    "--relocation-budget-bikes",
    "8",
    "--selective-threshold",
    "10",
    "--adr-high-pressure",
    "4",
    "--adr-drift-threshold",
    "0.08",
    "--service-reward",
    "1.0",
    "--rejection-penalty",
    "2.0",
    "--relocation-km-cost",
    "0.035",
    "--relocation-action-cost",
    "0.02",
    "--eda-population",
    "16",
    "--eda-generations",
    "6",
    "--eda-elite-frac",
    "0.35",
    "--eda-initial-sigma",
    "0.18",
    "--eda-imbalance-penalty",
    "0.35",
]

DATASETS = [
    {
        "name": "citibike",
        "profile": OUT_ROOT / "citibike_202401_profile_top25",
        "fleet": "300",
    },
    {
        "name": "divvy",
        "profile": OUT_ROOT / "divvy_202401_profile_top25",
        "fleet": "70",
    },
]

GATES = [
    {"name": "gate1_4_aggressive", "min": "1", "max": "4"},
    {"name": "gate2_8_reuse", "min": "2", "max": "8"},
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_progress(row: dict[str, str]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    exists = PROGRESS.exists()
    with PROGRESS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["time", "dataset", "gate", "stage", "status", "output", "message"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_stage(
    *,
    dataset: str,
    gate: str,
    stage: str,
    output: Path,
    cmd: list[str],
) -> None:
    append_progress(
        {
            "time": now(),
            "dataset": dataset,
            "gate": gate,
            "stage": stage,
            "status": "STARTED",
            "output": str(output),
            "message": " ".join(cmd),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    log_path = output / f"{stage}.log"
    output.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
        encoding="utf-8",
    )
    status = "OK" if completed.returncode == 0 else "FAILED"
    append_progress(
        {
            "time": now(),
            "dataset": dataset,
            "gate": gate,
            "stage": stage,
            "status": status,
            "output": str(output),
            "message": f"returncode={completed.returncode}; log={log_path}",
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed for {dataset}/{gate}; see {log_path}")


def main() -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for gate in GATES:
            event_out = CAMPAIGN / f"{dataset['name']}_{gate['name']}_event"
            hourly_out = CAMPAIGN / f"{dataset['name']}_{gate['name']}_hourly"
            gap_out = event_out / "aggregation_gap.csv"
            matrix = dataset["profile"] / "station_osrm_matrix_driving.json"

            run_stage(
                dataset=dataset["name"],
                gate=gate["name"],
                stage="event_replay",
                output=event_out,
                cmd=[
                    sys.executable,
                    "scripts/run_real_bikeshare_event_replay.py",
                    "--profile-dir",
                    str(dataset["profile"]),
                    "--out-dir",
                    str(event_out),
                    "--fleet-size",
                    dataset["fleet"],
                    "--distance-matrix-json",
                    str(matrix),
                    "--adr-min-update-interval",
                    gate["min"],
                    "--adr-max-update-interval",
                    gate["max"],
                    *COMMON_EVENT_ARGS,
                ],
            )
            run_stage(
                dataset=dataset["name"],
                gate=gate["name"],
                stage="hourly_comparable",
                output=hourly_out,
                cmd=[
                    sys.executable,
                    "scripts/run_real_bikeshare_station_benchmark.py",
                    "--profile-dir",
                    str(dataset["profile"]),
                    "--out-dir",
                    str(hourly_out),
                    "--fleet-size",
                    dataset["fleet"],
                    "--distance-matrix-json",
                    str(matrix),
                    "--adr-min-update-interval",
                    gate["min"],
                    "--adr-max-update-interval",
                    gate["max"],
                    *COMMON_HOURLY_ARGS,
                ],
            )
            run_stage(
                dataset=dataset["name"],
                gate=gate["name"],
                stage="aggregation_gap",
                output=event_out,
                cmd=[
                    sys.executable,
                    "scripts/compare_realbike_hourly_event_gap.py",
                    "--hourly-summary",
                    str(hourly_out / "summary.csv"),
                    "--event-summary",
                    str(event_out / "event_summary.csv"),
                    "--out-csv",
                    str(gap_out),
                ],
            )
    append_progress(
        {
            "time": now(),
            "dataset": "all",
            "gate": "all",
            "stage": "campaign",
            "status": "DONE",
            "output": str(CAMPAIGN),
            "message": "Event replay multicity campaign completed.",
        }
    )


if __name__ == "__main__":
    main()
