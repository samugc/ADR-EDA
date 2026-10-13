#!/usr/bin/env python3
"""Minimal ablation for ADR-EDA on the real request stream."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ROOT = Path("/app")
DEFAULT_LOCAL_PYTHONPATH = ""
DEFAULT_TRAVEL_HOST = (
    REPO_ROOT / "tests" / "model" / "results" / "madrid_osrm_zone_matrix_congestion_2025_01.json"
)
DEFAULT_ZONE_BASE_HOST = (
    REPO_ROOT / "src" / "free_floating_robust" / "data" / "madrid_zone_base_demand_serious_baseline.json"
)
DEFAULT_TRAVEL_DOCKER = Path("/app/tests/model/results/madrid_osrm_zone_matrix_congestion_2025_01.json")
DEFAULT_ZONE_BASE_DOCKER = Path("/app/src/free_floating_robust/data/madrid_zone_base_demand_serious_baseline.json")


def _parse_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("The seed list cannot be empty.")
    return out


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    print("[adr-ablation]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * _std(values) / math.sqrt(len(values)))


def _git_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "python_executable": os.environ.get("PYTHON", "") or "python3",
    }
    try:
        branch = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--short"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        meta["git_branch"] = branch.stdout.strip() if branch.returncode == 0 else ""
        meta["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else ""
        meta["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except Exception:
        meta["git_branch"] = ""
        meta["git_commit"] = ""
        meta["git_dirty"] = None
    return meta


def _resolve_runner_mode(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode == "auto":
        return "local" if CONTAINER_ROOT.exists() else ("docker" if shutil.which("docker") else "local")
    return mode


def _effective_local_pythonpath(raw: str) -> str:
    value = str(raw).strip()
    if value:
        return value
    if CONTAINER_ROOT.exists():
        return str(CONTAINER_ROOT / "src")
    return str(REPO_ROOT / "src")


def _extract_row(name: str, seed: int, payload: Dict[str, Any]) -> Dict[str, object]:
    metrics = dict(payload.get("audit_summary", {}).get("metrics", {}))
    fronts = dict(payload.get("eda_fronts", {}))
    reopts = list(fronts.get("reoptimizations", []))
    real_reopts = [row for row in reopts if bool(row.get("reoptimized", True))]
    holds = [row for row in reopts if not bool(row.get("reoptimized", True))]
    vector_shift = [
        float(row["vector_shift_l1"])
        for row in reopts
        if row.get("vector_shift_l1") is not None
    ]
    return {
        "variant": str(name),
        "seed": int(seed),
        "profit_eur": float(metrics.get("profit_eur", 0.0)),
        "service_rate": float(metrics.get("service_rate", 0.0)),
        "served_requests": float(metrics.get("served_requests", 0.0)),
        "rejected_requests": float(metrics.get("rejected_requests", 0.0)),
        "drive_km_total": float(metrics.get("drive_km_total", 0.0)),
        "ops_deadhead_km_total": float(metrics.get("ops_deadhead_km_total", 0.0)),
        "relocation_cost_eur": float(metrics.get("relocation_cost_eur", 0.0)),
        "ops_cost_eur": float(metrics.get("ops_cost_eur", 0.0)),
        "epochs_observed": int(len(reopts)),
        "epochs_reoptimized": int(len(real_reopts)),
        "epochs_hold": int(len(holds)),
        "mean_shift_l1": _mean(vector_shift) if vector_shift else 0.0,
    }


def _summarize(rows: List[Dict[str, object]]) -> Dict[str, float]:
    profits = [float(r["profit_eur"]) for r in rows]
    services = [float(r["service_rate"]) for r in rows]
    drives = [float(r["drive_km_total"]) for r in rows]
    deadheads = [float(r["ops_deadhead_km_total"]) for r in rows]
    reloc_costs = [float(r["relocation_cost_eur"]) for r in rows]
    ops_costs = [float(r["ops_cost_eur"]) for r in rows]
    reopts = [float(r["epochs_reoptimized"]) for r in rows]
    holds = [float(r["epochs_hold"]) for r in rows]
    shifts = [float(r["mean_shift_l1"]) for r in rows]
    rejected = [float(r["rejected_requests"]) for r in rows]
    return {
        "n": int(len(rows)),
        "mean_profit_eur": _mean(profits),
        "std_profit_eur": _std(profits),
        "ci95_profit_eur": _ci95(profits),
        "mean_service_rate": _mean(services),
        "mean_drive_km_total": _mean(drives),
        "mean_ops_deadhead_km_total": _mean(deadheads),
        "mean_relocation_cost_eur": _mean(reloc_costs),
        "mean_ops_cost_eur": _mean(ops_costs),
        "mean_epochs_reoptimized": _mean(reopts),
        "mean_epochs_hold": _mean(holds),
        "mean_shift_l1": _mean(shifts),
        "mean_rejected_requests": _mean(rejected),
        "min_profit_eur": min(profits) if profits else float("nan"),
    }


def _pairwise_delta(
    rows: List[Dict[str, object]],
    left_variant: str,
    right_variant: str,
) -> Dict[str, float]:
    by_key = {(str(r["variant"]), int(r["seed"])): r for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    deltas: List[Dict[str, float]] = []
    for seed in seeds:
        left = by_key.get((left_variant, seed))
        right = by_key.get((right_variant, seed))
        if left is None or right is None:
            continue
        deltas.append(
            {
                "profit_delta": float(right["profit_eur"]) - float(left["profit_eur"]),
                "service_delta": float(right["service_rate"]) - float(left["service_rate"]),
                "deadhead_delta": float(right["ops_deadhead_km_total"]) - float(left["ops_deadhead_km_total"]),
                "reopt_delta": float(right["epochs_reoptimized"]) - float(left["epochs_reoptimized"]),
            }
        )
    if not deltas:
        return {}
    profit = [d["profit_delta"] for d in deltas]
    service = [d["service_delta"] for d in deltas]
    deadhead = [d["deadhead_delta"] for d in deltas]
    reopt = [d["reopt_delta"] for d in deltas]
    return {
        "scenario_count": int(len(deltas)),
        "mean_profit_delta": _mean(profit),
        "ci95_profit_delta": _ci95(profit),
        "mean_service_delta": _mean(service),
        "mean_deadhead_delta": _mean(deadhead),
        "mean_reopt_delta": _mean(reopt),
        "profit_wins": int(sum(1 for v in profit if v > 0.0)),
        "service_nonworse": int(sum(1 for v in service if v >= 0.0)),
        "deadhead_wins": int(sum(1 for v in deadhead if v < 0.0)),
    }


def _build_report(summary: Dict[str, Any]) -> str:
    summary_rows = {str(r["variant"]): r for r in summary["summary_rows"]}
    order = list(summary["meta"]["variant_order"])
    rows = list(summary["rows"])
    meta = dict(summary["meta"])
    variant_descriptions = {
        "full_static": "always-reoptimize baseline.",
        "fast_event": "problem-aware event-driven DynamicEDA baseline without ADR-specific guard layers.",
        "adr_base_v22": "base ADR-EDA profile without reservation-aware guard layer.",
        "adr_reservation_guard_v2": "ADR-EDA with reservation-aware smoothing and hold logic.",
        "adr_cost_guard_v3": "ADR-EDA with reservation-aware control plus cost guard and burst control.",
    }
    variant_lines = [f"- `{name}`: {variant_descriptions.get(name, '')}" for name in order]

    lines = [
        "# ADR-EDA Minimal Ablation",
        "",
        "## Configuration",
        f"- runner_mode: {meta['runner_mode']}",
        f"- repo_root: {meta.get('repo_root', '')}",
        f"- git_branch: {meta.get('git_branch', '')}",
        f"- git_commit: {meta.get('git_commit', '')}",
        f"- git_dirty: {meta.get('git_dirty', '')}",
        f"- seeds: {summary['meta']['seeds']}",
        f"- sim_duration_min: {summary['meta']['sim_duration_min']:.1f}",
        f"- decision_interval_min: {summary['meta']['decision_interval_min']}",
        f"- planning_horizon_min: {summary['meta']['planning_horizon_min']}",
        f"- horizon_hours: {summary['meta']['horizon_hours']}",
        f"- fleet_size: {summary['meta']['fleet_size']}",
        f"- charger_count: {summary['meta']['charger_count']}",
        f"- ops_resources: {summary['meta']['ops_resources']}",
        f"- base_demand_scale: {summary['meta']['base_demand_scale']:.2f}",
        f"- reservation_share: {summary['meta']['reservation_share']:.2f}",
        f"- ops_cost_profile: {summary['meta']['ops_cost_profile']}",
        "",
        "## Variants",
        "",
        *variant_lines,
        "",
        "## Summary",
        "",
        "| variant | profit_mean | service | drive_km | deadhead_km | reloc_cost | ops_cost | reopt_epochs | hold_epochs | mean_shift_l1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        row = summary_rows[name]
        lines.append(
            f"| {name} | {float(row['mean_profit_eur']):.2f} | {float(row['mean_service_rate']):.3f} | "
            f"{float(row['mean_drive_km_total']):.2f} | {float(row['mean_ops_deadhead_km_total']):.2f} | "
            f"{float(row['mean_relocation_cost_eur']):.2f} | {float(row['mean_ops_cost_eur']):.2f} | "
            f"{float(row['mean_epochs_reoptimized']):.2f} | {float(row['mean_epochs_hold']):.2f} | "
            f"{float(row['mean_shift_l1']):.3f} |"
        )

    lines.extend(["", "## Pairwise ADR Deltas", ""])
    for pair in summary["pairwise"]:
        left = pair["left"]
        right = pair["right"]
        delta = pair["delta"]
        lines.append(f"### `{right}` minus `{left}`")
        lines.append(f"- mean_profit_delta: {float(delta['mean_profit_delta']):.2f} EUR")
        lines.append(f"- ci95_profit_delta: {float(delta['ci95_profit_delta']):.2f} EUR")
        lines.append(f"- mean_service_delta: {float(delta['mean_service_delta']):.4f}")
        lines.append(f"- mean_deadhead_delta: {float(delta['mean_deadhead_delta']):.2f} km")
        lines.append(f"- mean_reopt_delta: {float(delta['mean_reopt_delta']):.2f}")
        lines.append(f"- profit_wins: {int(delta['profit_wins'])}/{int(delta['scenario_count'])}")
        lines.append(f"- service_nonworse: {int(delta['service_nonworse'])}/{int(delta['scenario_count'])}")
        lines.append(f"- deadhead_wins: {int(delta['deadhead_wins'])}/{int(delta['scenario_count'])}")
        lines.append("")

    lines.extend(
        [
            "## Detail by Seed",
            "",
            "| variant | seed | profit | service | drive_km | deadhead_km | reloc_cost | ops_cost | reopt_epochs | hold_epochs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (order.index(str(r["variant"])), int(r["seed"]))):
        lines.append(
            f"| {row['variant']} | {int(row['seed'])} | {float(row['profit_eur']):.2f} | "
            f"{float(row['service_rate']):.3f} | {float(row['drive_km_total']):.2f} | "
            f"{float(row['ops_deadhead_km_total']):.2f} | {float(row['relocation_cost_eur']):.2f} | "
            f"{float(row['ops_cost_eur']):.2f} | {int(row['epochs_reoptimized'])} | {int(row['epochs_hold'])} |"
        )
    return "\n".join(lines) + "\n"


def _run_variant_seed(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    variant: Dict[str, Any],
    seed: int,
) -> Dict[str, object]:
    host_results_dir = REPO_ROOT / "tests" / "model" / "results" / output_dir.name
    host_results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    host_output = output_dir / f"{variant['name']}_seed{seed}.json"
    output_json = str(host_output) if str(args.runner_mode) == "local" else str(
        Path("/app/tests/model/results") / output_dir.name / f"{variant['name']}_seed{seed}.json"
    )

    cmd = []
    env: Dict[str, str] | None = None
    if str(args.runner_mode) == "docker":
        cmd.extend(
            [
                "docker",
                "exec",
                str(args.docker_container),
                "env",
                f"PYTHONPATH={args.local_pythonpath}",
                "python3",
                "-m",
                "free_floating_robust.runner",
            ]
        )
        travel_metrics_json = str(args.travel_metrics_docker)
        zone_base_demand_json = str(args.zone_base_docker)
    else:
        cmd.extend(["python3", "-m", "free_floating_robust.runner"])
        env = dict(os.environ)
        env["PYTHONPATH"] = str(args.local_pythonpath)
        travel_metrics_json = str(args.travel_metrics_host)
        zone_base_demand_json = str(args.zone_base_host)

    cmd.extend(
        [
        "--mode",
        "online-eda",
        "--seed",
        str(int(seed)),
        "--sim-duration-min",
        str(float(args.sim_duration_min)),
        "--decision-interval-min",
        str(int(args.decision_interval_min)),
        "--planning-horizon-min",
        str(int(args.planning_horizon_min)),
        "--horizon-hours",
        str(int(args.horizon_hours)),
        "--fleet-size",
        str(int(args.fleet_size)),
        "--charger-count",
        str(int(args.charger_count)),
        "--ops-resources",
        str(int(args.ops_resources)),
        "--base-demand-scale",
        str(float(args.base_demand_scale)),
        "--size-gen",
        str(int(args.size_gen)),
        "--max-iter",
        str(int(args.max_iter)),
        "--dead-iter",
        str(int(args.dead_iter)),
        "--travel-metrics-json",
        travel_metrics_json,
        "--zone-base-demand-json",
        zone_base_demand_json,
        "--fare-mode",
        "time_only",
        "--fare-per-min-eur",
        str(float(args.fare_per_min_eur)),
        "--ops-cost-profile",
        str(args.ops_cost_profile),
        "--reservation-share",
        str(float(args.reservation_share)),
        "--reservation-booking-intensity-mult",
        str(float(args.reservation_booking_intensity_mult)),
        "--reservation-mean-lead-min",
        str(float(args.reservation_mean_lead_min)),
        "--eda-reopt-every-epochs",
        "1",
        "--output-json",
        output_json,
        "--no-simple-access-model",
        ]
    )
    cmd.extend(list(variant["extra_args"]))
    _run(cmd, env=env)
    preferred_output = host_output
    if str(args.runner_mode) == "docker":
        docker_host_output = host_results_dir / f"{variant['name']}_seed{seed}.json"
        if output_dir != host_results_dir:
            base_name = f"{variant['name']}_seed{seed}"
            for suffix in (
                ".json",
                "_audit_summary.json",
                "_audit_status.csv",
                "_fronts.json",
                "_events.jsonl",
            ):
                src = host_results_dir / f"{base_name}{suffix}"
                dst = output_dir / f"{base_name}{suffix}"
                if src.exists():
                    shutil.copy2(src, dst)
        if preferred_output.exists():
            pass
        elif docker_host_output.exists():
            preferred_output = docker_host_output
    payload = _read_json(preferred_output)
    row = _extract_row(str(variant["name"]), int(seed), payload)
    row["output_json"] = str(preferred_output)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal ablation for ADR-EDA.")
    p.add_argument(
        "--runner-mode",
        type=str,
        choices=("auto", "docker", "local"),
        default="auto",
    )
    p.add_argument("--docker-container", type=str, default="tesis_container")
    p.add_argument("--local-pythonpath", type=str, default=DEFAULT_LOCAL_PYTHONPATH)
    p.add_argument("--seeds", type=str, default="2026,2027,2028")
    p.add_argument("--sim-duration-min", type=float, default=60.0)
    p.add_argument("--decision-interval-min", type=int, default=5)
    p.add_argument("--planning-horizon-min", type=int, default=60)
    p.add_argument("--horizon-hours", type=int, default=1)
    p.add_argument("--fleet-size", type=int, default=220)
    p.add_argument("--charger-count", type=int, default=22)
    p.add_argument("--ops-resources", type=int, default=26)
    p.add_argument("--base-demand-scale", type=float, default=1.0)
    p.add_argument("--size-gen", type=int, default=24)
    p.add_argument("--max-iter", type=int, default=2)
    p.add_argument("--dead-iter", type=int, default=2)
    p.add_argument("--fare-per-min-eur", type=float, default=0.24)
    p.add_argument("--ops-cost-profile", type=str, default="serio")
    p.add_argument("--reservation-share", type=float, default=0.28)
    p.add_argument("--reservation-booking-intensity-mult", type=float, default=1.0)
    p.add_argument("--reservation-mean-lead-min", type=float, default=35.0)
    p.add_argument(
        "--variants",
        type=str,
        default="full_static,adr_base_v22,adr_reservation_guard_v2,adr_cost_guard_v3",
        help="Comma-separated subset of variants to execute.",
    )
    p.add_argument("--travel-metrics-host", type=str, default=str(DEFAULT_TRAVEL_HOST))
    p.add_argument("--zone-base-host", type=str, default=str(DEFAULT_ZONE_BASE_HOST))
    p.add_argument("--travel-metrics-docker", type=str, default=str(DEFAULT_TRAVEL_DOCKER))
    p.add_argument("--zone-base-docker", type=str, default=str(DEFAULT_ZONE_BASE_DOCKER))
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "tests" / "model" / "results" / "adr_minimal_ablation"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.runner_mode = _resolve_runner_mode(args.runner_mode)
    args.local_pythonpath = _effective_local_pythonpath(args.local_pythonpath)
    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_csv_ints(args.seeds)
    requested_variants = [token.strip() for token in str(args.variants).split(",") if token.strip()]

    variant_catalog: Dict[str, Dict[str, Any]] = {
        "full_static": {
            "name": "full_static",
            "extra_args": [
                "--eda-online-engine",
                "speda-static",
                "--online-reopt-profile",
                "full",
                "--optimizer-family",
                "SPBN",
            ],
        },
        "fast_event": {
            "name": "fast_event",
            "extra_args": [
                "--eda-online-engine",
                "dynamic-stateful",
                "--dynamic-online-profile",
                "fast_event",
                "--pm-family",
                "SPBN",
            ],
        },
        "adr_base_v22": {
            "name": "adr_base_v22",
            "extra_args": [
                "--eda-online-engine",
                "dynamic-stateful",
                "--dynamic-online-profile",
                "adr_eda_v2_2_carsharing",
                "--pm-family",
                "SPBN",
            ],
        },
        "adr_reservation_guard_v2": {
            "name": "adr_reservation_guard_v2",
            "extra_args": [
                "--eda-online-engine",
                "dynamic-stateful",
                "--dynamic-online-profile",
                "adr_eda_reservation_aware_v2",
                "--pm-family",
                "SPBN",
            ],
        },
        "adr_cost_guard_v3": {
            "name": "adr_cost_guard_v3",
            "extra_args": [
                "--eda-online-engine",
                "dynamic-stateful",
                "--dynamic-online-profile",
                "adr_eda_reservation_costaware_v3",
                "--pm-family",
                "SPBN",
            ],
        },
    }
    unknown = [name for name in requested_variants if name not in variant_catalog]
    if unknown:
        raise ValueError(f"Unknown variants requested: {unknown}")
    if not requested_variants:
        raise ValueError("The variant list cannot be empty.")
    variants = [variant_catalog[name] for name in requested_variants]

    rows: List[Dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            rows.append(
                _run_variant_seed(
                    args=args,
                    output_dir=out_dir,
                    variant=variant,
                    seed=int(seed),
                )
            )

    summary_rows = []
    for variant in variants:
        variant_rows = [row for row in rows if str(row["variant"]) == str(variant["name"])]
        summary_row = {"variant": str(variant["name"])}
        summary_row.update(_summarize(variant_rows))
        summary_rows.append(summary_row)

    variant_order = [str(v["name"]) for v in variants]
    canonical_pairs = [
        ("full_static", "fast_event"),
        ("full_static", "adr_reservation_guard_v2"),
        ("fast_event", "adr_base_v22"),
        ("adr_base_v22", "adr_reservation_guard_v2"),
        ("fast_event", "adr_reservation_guard_v2"),
        ("adr_reservation_guard_v2", "adr_cost_guard_v3"),
        ("fast_event", "adr_cost_guard_v3"),
        ("adr_base_v22", "adr_cost_guard_v3"),
        ("full_static", "adr_cost_guard_v3"),
        ("full_static", "adr_base_v22"),
    ]
    pairwise = []
    for left, right in canonical_pairs:
        if left in requested_variants and right in requested_variants:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "delta": _pairwise_delta(rows, left, right),
                }
            )

    summary = {
        "meta": {
            "runner_mode": str(args.runner_mode),
            "docker_container": str(args.docker_container),
            "local_pythonpath": str(args.local_pythonpath),
            "seeds": seeds,
            "sim_duration_min": float(args.sim_duration_min),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "horizon_hours": int(args.horizon_hours),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            "base_demand_scale": float(args.base_demand_scale),
            "size_gen": int(args.size_gen),
            "max_iter": int(args.max_iter),
            "dead_iter": int(args.dead_iter),
            "ops_cost_profile": str(args.ops_cost_profile),
            "reservation_share": float(args.reservation_share),
            "reservation_booking_intensity_mult": float(args.reservation_booking_intensity_mult),
            "reservation_mean_lead_min": float(args.reservation_mean_lead_min),
            "variant_order": variant_order,
            **_git_metadata(),
        },
        "summary_rows": summary_rows,
        "pairwise": pairwise,
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(rows, out_dir / "runs.csv")
    _write_csv(summary_rows, out_dir / "summary_rows.csv")
    (out_dir / "REPORT.md").write_text(_build_report(summary), encoding="utf-8")
    print(f"[adr-ablation] summary -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
