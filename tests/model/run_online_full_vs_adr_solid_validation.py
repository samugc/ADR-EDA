#!/usr/bin/env python3
"""Solid validation of online full vs selective dynamic controllers on the real request stream."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ROOT = Path("/app")
DEFAULT_LOCAL_PYTHONPATH = ""
DEFAULT_TRAVEL_HOST = REPO_ROOT / "tests" / "model" / "results" / "madrid_osrm_zone_matrix_congestion_2025_01.json"
DEFAULT_ZONE_BASE_HOST = REPO_ROOT / "src" / "free_floating_robust" / "data" / "madrid_zone_base_demand_serious_baseline.json"
DEFAULT_TRAVEL_DOCKER = Path("/app/tests/model/results/madrid_osrm_zone_matrix_congestion_2025_01.json")
DEFAULT_ZONE_BASE_DOCKER = Path("/app/src/free_floating_robust/data/madrid_zone_base_demand_serious_baseline.json")
SUPPORTED_DYNAMIC_PROFILES = [
    "custom",
    "default_event",
    "balanced_event",
    "fast_event",
    "fast_strong_react",
    "adr_eda_v2_1",
    "adr_eda_v2_1_carsharing",
    "adr_eda_v2_2_carsharing",
    "adr_eda_reliability_diagnostics_v1",
    "adr_eda_adaptive_weights_v1",
    "adr_eda_adaptive_weights_switch_gate_v1",
    "adr_eda_adaptive_weights_selector_v2",
    "selective_simple_carsharing",
    "adr_eda_reservation_aware_v1",
    "adr_eda_reservation_aware_v2",
    "adr_eda_reservation_overload_aware_v1",
    "adr_eda_service_guard_v1",
    "adr_eda_service_guard_v2",
    "adr_eda_service_guard_v3",
    "adr_eda_service_guard_v4",
    "adr_eda_reservation_costaware_v1",
    "adr_eda_reservation_costaware_v2",
    "adr_eda_reservation_costaware_v3",
]

RUNNER_DYNAMIC_PROFILE_ALIASES = {
    "adr_eda_reliability_diagnostics_v1": "adr_eda_adaptive_diagnostics_only",
    "adr_eda_adaptive_weights_v1": "adr_eda_adaptive_weights",
    "adr_eda_adaptive_weights_switch_gate_v1": "adr_eda_adaptive_weights_guarded",
    "adr_eda_adaptive_weights_selector_v2": "adr_eda_adaptive_weights_selector_v2",
}


def _parse_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("La lista de seeds no puede quedar vacia.")
    return out


def _parse_csv_floats(raw: str) -> List[float]:
    out: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("La lista de escalas no puede quedar vacia.")
    return out


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    print("[solid-validation]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_existing_payload(paths: List[Path]) -> Tuple[Dict[str, Any] | None, Path | None]:
    """Return a usable completed runner output from the first existing path."""
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        metrics = dict(payload.get("audit_summary", {}).get("metrics", {}))
        if metrics:
            return payload, path
    return None, None


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        return "docker" if shutil.which("docker") else "local"
    return mode


def _effective_local_pythonpath(args: argparse.Namespace, runner_mode: str) -> str:
    raw = str(args.local_pythonpath).strip()
    if raw:
        return raw
    if runner_mode == "docker":
        return str(CONTAINER_ROOT / "src")
    return str(REPO_ROOT / "src")


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


def _extract_row(
    *,
    variant: str,
    seed: int,
    demand_scale: float,
    payload: Dict[str, Any],
) -> Dict[str, object]:
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
        "variant": str(variant),
        "seed": int(seed),
        "base_demand_scale": float(demand_scale),
        "profit_eur": float(metrics.get("profit_eur", 0.0)),
        "service_rate": float(metrics.get("service_rate", 0.0)),
        "served_requests": int(metrics.get("served_requests", 0.0)),
        "rejected_requests": int(metrics.get("rejected_requests", 0.0)),
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
    served = [float(r["served_requests"]) for r in rows]
    return {
        "n": int(len(rows)),
        "mean_profit_eur": _mean(profits),
        "std_profit_eur": _std(profits),
        "ci95_profit_eur": _ci95(profits),
        "mean_service_rate": _mean(services),
        "min_service_rate": min(services) if services else float("nan"),
        "mean_drive_km_total": _mean(drives),
        "mean_ops_deadhead_km_total": _mean(deadheads),
        "mean_relocation_cost_eur": _mean(reloc_costs),
        "mean_ops_cost_eur": _mean(ops_costs),
        "mean_epochs_reoptimized": _mean(reopts),
        "mean_epochs_hold": _mean(holds),
        "mean_shift_l1": _mean(shifts),
        "mean_served_requests": _mean(served),
        "mean_rejected_requests": _mean(rejected),
        "max_rejected_requests": max(rejected) if rejected else float("nan"),
        "min_profit_eur": min(profits) if profits else float("nan"),
    }


def _paired_analysis(
    rows: List[Dict[str, object]],
    lhs_name: str,
    rhs_name: str,
) -> Dict[str, object]:
    by_key: Dict[Tuple[str, int, float], Dict[str, object]] = {
        (str(r["variant"]), int(r["seed"]), float(r["base_demand_scale"])): r for r in rows
    }
    scenario_keys = sorted(
        {
            (int(r["seed"]), float(r["base_demand_scale"]))
            for r in rows
        }
    )
    deltas: List[Dict[str, float]] = []
    for seed, scale in scenario_keys:
        lhs = by_key.get((lhs_name, seed, scale))
        rhs = by_key.get((rhs_name, seed, scale))
        if lhs is None or rhs is None:
            continue
        deltas.append(
            {
                "seed": float(seed),
                "base_demand_scale": float(scale),
                "profit_delta": float(lhs["profit_eur"]) - float(rhs["profit_eur"]),
                "service_delta": float(lhs["service_rate"]) - float(rhs["service_rate"]),
                "deadhead_delta": float(lhs["ops_deadhead_km_total"]) - float(rhs["ops_deadhead_km_total"]),
                "rejected_delta": float(lhs["rejected_requests"]) - float(rhs["rejected_requests"]),
            }
        )
    if not deltas:
        return {}

    profit_deltas = [float(d["profit_delta"]) for d in deltas]
    service_deltas = [float(d["service_delta"]) for d in deltas]
    deadhead_deltas = [float(d["deadhead_delta"]) for d in deltas]
    rejected_deltas = [float(d["rejected_delta"]) for d in deltas]
    return {
        "lhs_variant": str(lhs_name),
        "rhs_variant": str(rhs_name),
        "scenario_count": int(len(deltas)),
        "mean_profit_delta": _mean(profit_deltas),
        "ci95_profit_delta": _ci95(profit_deltas),
        "mean_service_delta": _mean(service_deltas),
        "mean_deadhead_delta": _mean(deadhead_deltas),
        "mean_rejected_delta": _mean(rejected_deltas),
        "profit_wins": int(sum(1 for d in profit_deltas if d > 0.0)),
        "service_nonworse": int(sum(1 for d in service_deltas if d >= 0.0)),
        "deadhead_wins": int(sum(1 for d in deadhead_deltas if d < 0.0)),
    }


def _dynamic_variant(name: str, profile: str) -> Dict[str, Any]:
    runner_profile = RUNNER_DYNAMIC_PROFILE_ALIASES.get(str(profile), str(profile))
    return {
        "name": str(name),
        "extra_args": [
            "--eda-online-engine",
            "dynamic-stateful",
            "--dynamic-online-profile",
            runner_profile,
            "--pm-family",
            "SPBN",
        ],
        "requested_profile": str(profile),
        "runner_profile": runner_profile,
    }


def _run_variant_seed(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    variant: Dict[str, Any],
    seed: int,
    demand_scale: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_tag = str(demand_scale).replace(".", "p")
    base_name = f"{variant['name']}_d{scale_tag}_seed{seed}"
    base_cmd = [
        "python3",
        "-m",
        "free_floating_robust.runner",
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
        str(float(demand_scale)),
        "--size-gen",
        str(int(args.size_gen)),
        "--max-iter",
        str(int(args.max_iter)),
        "--dead-iter",
        str(int(args.dead_iter)),
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
        "--no-simple-access-model",
    ]
    if str(args.optimizer_random_seed_mode).strip().lower() == "scenario_seed":
        base_cmd.extend(
            [
                "--optimizer-random-seed",
                str(int(seed) + int(args.optimizer_random_seed_base)),
            ]
        )

    runner_mode = _resolve_runner_mode(args.runner_mode)
    effective_local_pythonpath = _effective_local_pythonpath(args, runner_mode)

    local_env: Dict[str, str] | None = None
    if runner_mode == "docker":
        host_results_dir = REPO_ROOT / "tests" / "model" / "results" / output_dir.name
        host_results_dir.mkdir(parents=True, exist_ok=True)
        host_output = host_results_dir / f"{base_name}.json"
        preferred_output = output_dir / f"{base_name}.json"
        if bool(args.resume_existing):
            payload, existing_output = _read_existing_payload([preferred_output, host_output])
            if payload is not None and existing_output is not None:
                print("[solid-validation] resume-existing", existing_output, flush=True)
                row = _extract_row(
                    variant=str(variant["name"]),
                    seed=int(seed),
                    demand_scale=float(demand_scale),
                    payload=payload,
                )
                row["output_json"] = str(existing_output)
                return row
        runner_output = str(Path("/app/tests/model/results") / output_dir.name / f"{base_name}.json")
        cmd = [
            "docker",
            "exec",
            str(args.docker_container),
            "env",
            f"PYTHONPATH={effective_local_pythonpath}",
            *base_cmd,
            "--travel-metrics-json",
            str(args.travel_metrics_docker),
            "--zone-base-demand-json",
            str(args.zone_base_docker),
            "--output-json",
            runner_output,
        ]
        cmd.extend(list(variant["extra_args"]))
        _run(cmd)

        payload = _read_json(host_output)
        if output_dir.resolve() != host_results_dir.resolve():
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
        preferred_output = output_dir / f"{base_name}.json"
        if not preferred_output.exists():
            preferred_output = host_output
    else:
        preferred_output = output_dir / f"{base_name}.json"
        if bool(args.resume_existing):
            payload, existing_output = _read_existing_payload([preferred_output])
            if payload is not None and existing_output is not None:
                print("[solid-validation] resume-existing", existing_output, flush=True)
                row = _extract_row(
                    variant=str(variant["name"]),
                    seed=int(seed),
                    demand_scale=float(demand_scale),
                    payload=payload,
                )
                row["output_json"] = str(existing_output)
                return row
        cmd = [
            sys.executable,
            *base_cmd[1:],
            "--travel-metrics-json",
            str(args.travel_metrics_host),
            "--zone-base-demand-json",
            str(args.zone_base_host),
            "--output-json",
            str(preferred_output),
        ]
        cmd.extend(list(variant["extra_args"]))
        local_env = dict(os.environ)
        existing_pythonpath = str(local_env.get("PYTHONPATH", "")).strip()
        local_env["PYTHONPATH"] = (
            effective_local_pythonpath
            if not existing_pythonpath
            else effective_local_pythonpath + os.pathsep + existing_pythonpath
        )
        _run(cmd, env=local_env)
        payload = _read_json(preferred_output)

    row = _extract_row(
        variant=str(variant["name"]),
        seed=int(seed),
        demand_scale=float(demand_scale),
        payload=payload,
    )
    row["output_json"] = str(preferred_output)
    return row


def _build_report(summary: Dict[str, Any]) -> str:
    dynamic_name = str(summary["meta"]["dynamic_variant_name"])
    extra_dynamic_name = str(summary["meta"].get("extra_dynamic_variant_name") or "").strip()
    by_variant = {str(row["variant"]): row for row in summary["summary_rows"]}
    full = by_variant["full_static"]
    dyn = by_variant[dynamic_name]
    extra_dyn = by_variant.get(extra_dynamic_name) if extra_dynamic_name else None
    paired = dict(summary.get("paired", {}))
    pairwise_rows = [dict(row) for row in summary.get("pairwise_rows", [])]
    if not pairwise_rows and paired:
        pairwise_rows = [paired]
    meta = dict(summary["meta"])

    lines = [
        "# Solid Validation: Full vs Dynamic Controllers",
        "",
        "## Configuracion",
        f"- dynamic_variant: {dynamic_name}",
        f"- dynamic_profile: {summary['meta']['dynamic_online_profile']}",
        f"- extra_dynamic_variant: {extra_dynamic_name or 'none'}",
        f"- extra_dynamic_profile: {summary['meta'].get('extra_dynamic_online_profile', '') or 'none'}",
        f"- seeds: {summary['meta']['seeds']}",
        f"- base_demand_scales: {summary['meta']['base_demand_scales']}",
        f"- sim_duration_min: {summary['meta']['sim_duration_min']:.1f}",
        f"- decision_interval_min: {summary['meta']['decision_interval_min']}",
        f"- planning_horizon_min: {summary['meta']['planning_horizon_min']}",
        f"- horizon_hours: {summary['meta']['horizon_hours']}",
        f"- fleet_size: {summary['meta']['fleet_size']}",
        f"- charger_count: {summary['meta']['charger_count']}",
        f"- ops_resources: {summary['meta']['ops_resources']}",
        f"- repo_root: {meta.get('repo_root', '')}",
        f"- git_branch: {meta.get('git_branch', '')}",
        f"- git_commit: {meta.get('git_commit', '')}",
        f"- git_dirty: {meta.get('git_dirty', '')}",
        "",
        "## Resumen Global",
        "",
        "| variant | profit_mean | service_mean | service_min | drive_km | deadhead_km | reloc_cost | ops_cost | rejected_mean | rejected_max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_variants = ["full_static", dynamic_name]
    if extra_dynamic_name:
        ordered_variants.append(extra_dynamic_name)
    for variant_name in ordered_variants:
        row = by_variant.get(variant_name)
        if row is None:
            continue
        lines.append(
            f"| {variant_name} | {row['mean_profit_eur']:.2f} | {row['mean_service_rate']:.3f} | "
            f"{row['min_service_rate']:.3f} | {row['mean_drive_km_total']:.2f} | "
            f"{row['mean_ops_deadhead_km_total']:.2f} | {row['mean_relocation_cost_eur']:.2f} | "
            f"{row['mean_ops_cost_eur']:.2f} | {row['mean_rejected_requests']:.2f} | {row['max_rejected_requests']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Pareado Por Escenario",
            "",
        ]
    )
    if pairwise_rows:
        for block in pairwise_rows:
            lhs = str(block.get("lhs_variant", ""))
            rhs = str(block.get("rhs_variant", ""))
            label = f"{lhs} - {rhs}" if lhs and rhs else "comparacion"
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"- escenarios: {block['scenario_count']}",
                    f"- mean_profit_delta: {block['mean_profit_delta']:.2f} EUR",
                    f"- ci95_profit_delta: {block['ci95_profit_delta']:.2f}",
                    f"- mean_service_delta: {block['mean_service_delta']:.4f}",
                    f"- mean_deadhead_delta: {block['mean_deadhead_delta']:.2f} km",
                    f"- mean_rejected_delta: {block['mean_rejected_delta']:.2f}",
                    f"- profit_wins: {block['profit_wins']}/{block['scenario_count']}",
                    f"- service_nonworse: {block['service_nonworse']}/{block['scenario_count']}",
                    f"- deadhead_wins: {block['deadhead_wins']}/{block['scenario_count']}",
                    "",
                ]
            )
    else:
        lines.append("- sin comparacion pareada")
        lines.append("")

    if extra_dyn is not None:
        lines.extend(
            [
                "## Lectura Rapida",
                "",
                f"- {dynamic_name} vs full_static: {dyn['mean_profit_eur'] - full['mean_profit_eur']:.2f} EUR de margen medio.",
                f"- {extra_dynamic_name} vs full_static: {extra_dyn['mean_profit_eur'] - full['mean_profit_eur']:.2f} EUR de margen medio.",
                f"- {dynamic_name} vs {extra_dynamic_name}: {dyn['mean_profit_eur'] - extra_dyn['mean_profit_eur']:.2f} EUR de margen medio.",
                "",
            ]
        )

    lines.extend(
        [
            "## Resumen Por Demanda",
            "",
            "| demand_scale | variant | profit_mean | service_mean | service_min | deadhead_km | rejected_mean |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["per_scale_rows"]:
        lines.append(
            f"| {float(row['base_demand_scale']):.2f} | {row['variant']} | {float(row['mean_profit_eur']):.2f} | "
            f"{float(row['mean_service_rate']):.3f} | {float(row['min_service_rate']):.3f} | "
            f"{float(row['mean_ops_deadhead_km_total']):.2f} | {float(row['mean_rejected_requests']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Detalle",
            "",
            "| scale | variant | seed | profit | service | deadhead_km | rejected | reopt_epochs | hold_epochs | shift_l1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary["rows"], key=lambda r: (float(r["base_demand_scale"]), str(r["variant"]), int(r["seed"]))):
        lines.append(
            f"| {float(row['base_demand_scale']):.2f} | {row['variant']} | {int(row['seed'])} | "
            f"{float(row['profit_eur']):.2f} | {float(row['service_rate']):.3f} | "
            f"{float(row['ops_deadhead_km_total']):.2f} | {int(row['rejected_requests'])} | "
            f"{int(row['epochs_reoptimized'])} | {int(row['epochs_hold'])} | {float(row['mean_shift_l1']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Solid head-to-head validation for full vs selective dynamic controllers."
    )
    p.add_argument("--docker-container", type=str, default="tesis_container")
    p.add_argument("--runner-mode", type=str, default="auto", choices=["auto", "docker", "local"])
    p.add_argument("--local-pythonpath", type=str, default=DEFAULT_LOCAL_PYTHONPATH)
    p.add_argument("--travel-metrics-host", type=str, default=str(DEFAULT_TRAVEL_HOST))
    p.add_argument("--zone-base-host", type=str, default=str(DEFAULT_ZONE_BASE_HOST))
    p.add_argument("--seeds", type=str, default="2026,2027,2028,2029,2030")
    p.add_argument("--base-demand-scales", type=str, default="1.0,1.15,1.30")
    p.add_argument("--sim-duration-min", type=float, default=60.0)
    p.add_argument("--decision-interval-min", type=int, default=5)
    p.add_argument("--planning-horizon-min", type=int, default=60)
    p.add_argument("--horizon-hours", type=int, default=1)
    p.add_argument("--fleet-size", type=int, default=220)
    p.add_argument("--charger-count", type=int, default=22)
    p.add_argument("--ops-resources", type=int, default=26)
    p.add_argument("--size-gen", type=int, default=24)
    p.add_argument("--max-iter", type=int, default=2)
    p.add_argument("--dead-iter", type=int, default=2)
    p.add_argument("--fare-per-min-eur", type=float, default=0.24)
    p.add_argument(
        "--ops-cost-profile",
        type=str,
        default="serio",
        choices=["custom", "legacy", "conservador", "serio", "stress"],
    )
    p.add_argument("--reservation-share", type=float, default=0.28)
    p.add_argument("--reservation-booking-intensity-mult", type=float, default=1.0)
    p.add_argument("--reservation-mean-lead-min", type=float, default=35.0)
    p.add_argument(
        "--optimizer-random-seed-mode",
        type=str,
        default="disabled",
        choices=["disabled", "scenario_seed"],
        help=(
            "Controla la semilla RNG del optimizador EDA. Por defecto queda "
            "desactivada para no alterar resultados historicos; usa scenario_seed "
            "para reruns reproducibles por escenario."
        ),
    )
    p.add_argument(
        "--optimizer-random-seed-base",
        type=int,
        default=0,
        help="Offset sumado al seed del escenario si mode=scenario_seed.",
    )
    p.add_argument(
        "--dynamic-online-profile",
        type=str,
        default="adr_eda_v2_2_carsharing",
        choices=SUPPORTED_DYNAMIC_PROFILES,
    )
    p.add_argument("--dynamic-variant-name", type=str, default="dynamic_adr_eda_v2_2_carsharing")
    p.add_argument(
        "--extra-dynamic-online-profile",
        type=str,
        default="",
        choices=["", *SUPPORTED_DYNAMIC_PROFILES],
    )
    p.add_argument(
        "--extra-dynamic-variant-name",
        type=str,
        default="dynamic_selective_simple_carsharing",
    )
    p.add_argument("--travel-metrics-docker", type=str, default=str(DEFAULT_TRAVEL_DOCKER))
    p.add_argument("--zone-base-docker", type=str, default=str(DEFAULT_ZONE_BASE_DOCKER))
    p.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse completed JSON outputs in the output directory instead of rerunning them.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "tests" / "model" / "results" / "online_full_vs_adr_solid_validation"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runner_mode = _resolve_runner_mode(args.runner_mode)
    effective_local_pythonpath = _effective_local_pythonpath(args, runner_mode)
    seeds = _parse_csv_ints(args.seeds)
    demand_scales = _parse_csv_floats(args.base_demand_scales)
    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        {
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
        _dynamic_variant(str(args.dynamic_variant_name), str(args.dynamic_online_profile)),
    ]
    extra_profile = str(args.extra_dynamic_online_profile).strip()
    if extra_profile:
        variants.append(_dynamic_variant(str(args.extra_dynamic_variant_name), extra_profile))

    rows: List[Dict[str, object]] = []
    for demand_scale in demand_scales:
        for variant in variants:
            for seed in seeds:
                rows.append(
                    _run_variant_seed(
                        args=args,
                        output_dir=out_dir,
                        variant=variant,
                        seed=int(seed),
                        demand_scale=float(demand_scale),
                    )
                )

    summary_rows: List[Dict[str, object]] = []
    for variant in variants:
        variant_rows = [row for row in rows if str(row["variant"]) == str(variant["name"])]
        summary_row: Dict[str, object] = {"variant": str(variant["name"])}
        summary_row.update(_summarize(variant_rows))
        summary_rows.append(summary_row)

    per_scale_rows: List[Dict[str, object]] = []
    for demand_scale in demand_scales:
        for variant in variants:
            group = [
                row
                for row in rows
                if float(row["base_demand_scale"]) == float(demand_scale)
                and str(row["variant"]) == str(variant["name"])
            ]
            row: Dict[str, object] = {
                "base_demand_scale": float(demand_scale),
                "variant": str(variant["name"]),
            }
            row.update(_summarize(group))
            per_scale_rows.append(row)

    pairwise_rows: List[Dict[str, object]] = []
    main_vs_full = _paired_analysis(rows, str(args.dynamic_variant_name), "full_static")
    if main_vs_full:
        pairwise_rows.append(main_vs_full)
    if extra_profile:
        extra_vs_full = _paired_analysis(rows, str(args.extra_dynamic_variant_name), "full_static")
        if extra_vs_full:
            pairwise_rows.append(extra_vs_full)
        main_vs_extra = _paired_analysis(
            rows,
            str(args.dynamic_variant_name),
            str(args.extra_dynamic_variant_name),
        )
        if main_vs_extra:
            pairwise_rows.append(main_vs_extra)

    summary = {
        "meta": {
            "docker_container": str(args.docker_container),
            "runner_mode": runner_mode,
            "local_pythonpath": effective_local_pythonpath,
            "seeds": seeds,
            "base_demand_scales": demand_scales,
            "sim_duration_min": float(args.sim_duration_min),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "horizon_hours": int(args.horizon_hours),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            "size_gen": int(args.size_gen),
            "max_iter": int(args.max_iter),
            "dead_iter": int(args.dead_iter),
            "ops_cost_profile": str(args.ops_cost_profile),
            "optimizer_random_seed_mode": str(args.optimizer_random_seed_mode),
            "optimizer_random_seed_base": int(args.optimizer_random_seed_base),
            "dynamic_online_profile": str(args.dynamic_online_profile),
            "dynamic_variant_name": str(args.dynamic_variant_name),
            "extra_dynamic_online_profile": extra_profile,
            "extra_dynamic_variant_name": (
                str(args.extra_dynamic_variant_name) if extra_profile else ""
            ),
            **_git_metadata(),
        },
        "summary_rows": summary_rows,
        "per_scale_rows": per_scale_rows,
        "paired": main_vs_full,
        "pairwise_rows": pairwise_rows,
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(rows, out_dir / "runs.csv")
    (out_dir / "REPORT.md").write_text(_build_report(summary), encoding="utf-8")

    print("ONLINE_FULL_VS_ADR_SOLID_VALIDATION_OK")
    print({"output_dir": str(out_dir)})


if __name__ == "__main__":
    main()
