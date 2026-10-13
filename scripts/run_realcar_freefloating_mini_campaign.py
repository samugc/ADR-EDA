#!/usr/bin/env python3
"""Run a small RealCar free-floating Pareto/ADR campaign.

This campaign is intentionally small: it validates that the original
free-floating multiobjective model can consume real Munich trip demand without
turning the problem into the station-level bike-sharing proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONSUMPTION_KWH_KM = 0.145
DEFAULT_ENERGY_COST_PER_KWH = 0.23

COMPARABLE_PROFIT_COLUMNS = [
    "reported_profit_eur",
    "revenue_eur",
    "ops_charge_credit_eur",
    "explicit_relocation_cost_eur",
    "explicit_energy_cost_eur",
    "explicit_ops_cost_eur",
    "explicit_ops_deadhead_cost_eur",
    "explicit_penalty_cost_eur",
    "comparable_consumption_kwh_km",
    "comparable_energy_cost_per_kwh",
    "drive_energy_proxy_cost_eur",
    "comparable_operating_cost_eur",
    "comparable_profit_eur",
]

Q1_AUDIT_COLUMNS = [
    "budget_label",
    "size_gen",
    "max_iter",
    "window_count",
    "optimizer_eval_count",
    "selection_eval_count",
    "execution_eval_count",
    "total_rollout_eval_count",
    "optimizer_runtime_sec",
    "selection_runtime_sec",
    "execution_runtime_sec",
    "evaluation_runtime_sec",
    "reopt_wall_time_sec",
    "optimizer_step_count",
]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def write_progress(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "time",
                "seed",
                "mode",
                "status",
                "output",
                "seconds",
                "message",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_cmd(
    *,
    cmd: list[str],
    log_path: Path,
    progress_path: Path,
    seed: int,
    mode: str,
    output_path: Path,
) -> None:
    start = time.monotonic()
    write_progress(
        progress_path,
        {
            "time": now_iso(),
            "seed": seed,
            "mode": mode,
            "status": "STARTED",
            "output": str(output_path),
            "seconds": "",
            "message": " ".join(cmd),
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - start
    status = "OK" if proc.returncode == 0 else "FAILED"
    write_progress(
        progress_path,
        {
            "time": now_iso(),
            "seed": seed,
            "mode": mode,
            "status": status,
            "output": str(output_path),
            "seconds": f"{elapsed:.2f}",
            "message": f"returncode={proc.returncode}; log={log_path}",
        },
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} seed={seed} failed; see {log_path}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(data: dict[str, Any], *path: str, default: Any = "") -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_present(mapping: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in ("", None):
            return value
    return default


def empty_q1_audit_fields() -> dict[str, Any]:
    return {key: "" for key in Q1_AUDIT_COLUMNS}


def q1_audit_fields(
    data: dict[str, Any] | None = None,
    fronts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = data or {}
    fronts = fronts or {}
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    front_meta = fronts.get("meta", {}) if isinstance(fronts, dict) else {}
    windows = data.get("windows", []) if isinstance(data.get("windows", []), list) else []
    reoptimizations = (
        fronts.get("reoptimizations", [])
        if isinstance(fronts.get("reoptimizations", []), list)
        else []
    )

    size_gen = _first_present(
        meta,
        ["size_gen", "size_gen_used", "size_gen_requested"],
        _first_present(front_meta, ["eda_size_gen", "size_gen_requested"], ""),
    )
    max_iter = _first_present(
        meta,
        ["max_iter", "max_iter_per_window", "max_iter_per_update", "max_iter_requested"],
        _first_present(front_meta, ["eda_max_iter", "max_iter_requested"], ""),
    )
    window_count = _first_present(
        meta,
        ["window_count"],
        len(windows) if windows else len(reoptimizations) if reoptimizations else "",
    )

    optimizer_eval = _first_present(
        meta,
        ["optimizer_rollout_eval_count", "optimizer_eval_count"],
    )
    selection_eval = _first_present(
        meta,
        ["selection_rollout_eval_count", "selection_eval_count"],
    )
    execution_eval = _first_present(
        meta,
        ["execution_rollout_eval_count", "execution_eval_count"],
    )
    total_eval = _first_present(meta, ["total_rollout_eval_count"])

    optimizer_runtime = _first_present(
        meta,
        ["optimizer_wall_time_sec", "optimizer_runtime_sec", "optimizer_rollout_runtime_sec", "exec_time_sec"],
    )
    if optimizer_runtime in ("", None) and windows:
        values = [
            as_float(row.get("optimizer_wall_time_sec", row.get("optimizer_exec_time_sec", "")))
            for row in windows
            if row.get("optimizer_wall_time_sec", row.get("optimizer_exec_time_sec", "")) not in ("", None)
        ]
        values = [value for value in values if value is not None]
        if values:
            optimizer_runtime = sum(values)
    selection_runtime = _first_present(
        meta,
        ["selection_wall_time_sec", "selection_runtime_sec", "selection_rollout_runtime_sec"],
    )
    if selection_runtime in ("", None) and windows:
        values = [
            as_float(row.get("selection_wall_time_sec", ""))
            for row in windows
            if row.get("selection_wall_time_sec", "") not in ("", None)
        ]
        values = [value for value in values if value is not None]
        if values:
            selection_runtime = sum(values)
    execution_runtime = _first_present(
        meta,
        ["execution_wall_time_sec", "execution_runtime_sec", "execution_rollout_runtime_sec"],
    )
    if execution_runtime in ("", None) and windows:
        values = [
            as_float(row.get("execution_wall_time_sec", ""))
            for row in windows
            if row.get("execution_wall_time_sec", "") not in ("", None)
        ]
        values = [value for value in values if value is not None]
        if values:
            execution_runtime = sum(values)
    evaluation_runtime = _first_present(meta, ["total_rollout_runtime_sec"])

    reopt_wall = _first_present(front_meta, ["reopt_wall_time_sec"], "")
    if (
        reopt_wall in ("", None)
        and reoptimizations
        and any("reopt_wall_time_sec" in row for row in reoptimizations)
    ):
        reopt_wall = sum(as_float(row.get("reopt_wall_time_sec", 0.0)) for row in reoptimizations)
    optimizer_steps = _first_present(front_meta, ["optimizer_steps_executed"], "")
    if (
        optimizer_steps in ("", None)
        and reoptimizations
        and any("optimizer_steps_executed" in row for row in reoptimizations)
    ):
        optimizer_steps = sum(int(as_float(row.get("optimizer_steps_executed", 0.0))) for row in reoptimizations)

    budget_label = ""
    if size_gen not in ("", None) and max_iter not in ("", None):
        budget_label = f"{int(as_float(size_gen))}x{int(as_float(max_iter))}"

    return {
        "budget_label": budget_label,
        "size_gen": size_gen,
        "max_iter": max_iter,
        "window_count": window_count,
        "optimizer_eval_count": optimizer_eval,
        "selection_eval_count": selection_eval,
        "execution_eval_count": execution_eval,
        "total_rollout_eval_count": total_eval,
        "optimizer_runtime_sec": optimizer_runtime,
        "selection_runtime_sec": selection_runtime,
        "execution_runtime_sec": execution_runtime,
        "evaluation_runtime_sec": evaluation_runtime,
        "reopt_wall_time_sec": reopt_wall,
        "optimizer_step_count": optimizer_steps,
    }


def empty_comparable_profit_fields() -> dict[str, Any]:
    return {key: "" for key in COMPARABLE_PROFIT_COLUMNS}


def comparable_profit_fields(
    *,
    reported_profit_eur: Any,
    revenue_eur: Any,
    drive_km_total: Any,
    meta: dict[str, Any] | None = None,
    ops_charge_credit_eur: Any = 0.0,
    relocation_cost_eur: Any = 0.0,
    energy_cost_eur: Any = 0.0,
    ops_cost_eur: Any = 0.0,
    ops_deadhead_cost_eur: Any = 0.0,
    penalty_cost_eur: Any = 0.0,
) -> dict[str, Any]:
    """Add a post-hoc net-profit metric comparable across rollout modes.

    The MO objective `f2_profit` is not changed. The simulator's native profit
    is preserved as `reported_profit_eur`; for finite-horizon audit tables,
    `comparable_profit_eur` charges a homogeneous proxy energy cost to every
    driven kilometer, instead of depending on whether a policy happened to
    recharge vehicles within the horizon.
    """

    if revenue_eur in ("", None):
        fields = empty_comparable_profit_fields()
        fields["reported_profit_eur"] = reported_profit_eur
        return fields

    meta = meta or {}
    consumption_kwh_km = as_float(
        meta.get(
            "consumption_kwh_km",
            meta.get("consumption_kwh_per_km", DEFAULT_CONSUMPTION_KWH_KM),
        ),
        DEFAULT_CONSUMPTION_KWH_KM,
    )
    energy_cost_per_kwh = as_float(
        meta.get("energy_cost_per_kwh", DEFAULT_ENERGY_COST_PER_KWH),
        DEFAULT_ENERGY_COST_PER_KWH,
    )
    revenue = as_float(revenue_eur)
    ops_charge_credit = as_float(ops_charge_credit_eur)
    relocation_cost = as_float(relocation_cost_eur)
    explicit_energy_cost = as_float(energy_cost_eur)
    ops_cost = as_float(ops_cost_eur)
    ops_deadhead_cost = as_float(ops_deadhead_cost_eur)
    penalty_cost = as_float(penalty_cost_eur)
    drive_energy_proxy = (
        as_float(drive_km_total) * consumption_kwh_km * energy_cost_per_kwh
    )
    comparable_operating_cost = (
        drive_energy_proxy
        + relocation_cost
        + ops_cost
        + ops_deadhead_cost
        + penalty_cost
    )
    comparable_profit = revenue + ops_charge_credit - comparable_operating_cost
    return {
        "reported_profit_eur": reported_profit_eur,
        "revenue_eur": revenue_eur,
        "ops_charge_credit_eur": ops_charge_credit_eur,
        "explicit_relocation_cost_eur": relocation_cost_eur,
        "explicit_energy_cost_eur": energy_cost_eur,
        "explicit_ops_cost_eur": ops_cost_eur,
        "explicit_ops_deadhead_cost_eur": ops_deadhead_cost_eur,
        "explicit_penalty_cost_eur": penalty_cost_eur,
        "comparable_consumption_kwh_km": consumption_kwh_km,
        "comparable_energy_cost_per_kwh": energy_cost_per_kwh,
        "drive_energy_proxy_cost_eur": drive_energy_proxy,
        "comparable_operating_cost_eur": comparable_operating_cost,
        "comparable_profit_eur": comparable_profit,
    }


def load_progress_runtime(out_dir: Path) -> dict[tuple[int, str], float]:
    path = out_dir / "progress.csv"
    if not path.exists():
        return {}
    out: dict[tuple[int, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("status", "")).strip() != "OK":
                continue
            try:
                seed = int(row.get("seed", ""))
                seconds = float(row.get("seconds", ""))
            except ValueError:
                continue
            out[(seed, str(row.get("mode", "")))] = seconds
    return out


def last_selected_objectives(fronts: dict[str, Any]) -> tuple[list[float], int, int]:
    reopts = list(fronts.get("reoptimizations", [])) if isinstance(fronts, dict) else []
    for row in reversed(reopts):
        selected = row.get("selected_objectives")
        if isinstance(selected, list) and len(selected) >= 3:
            return [float(selected[0]), float(selected[1]), float(selected[2])], int(
                row.get("pareto_size", "")
                if str(row.get("pareto_size", "")).strip()
                else 0
            ), len(reopts)
    return [], 0, len(reopts)


def summarize(out_dir: Path, seeds: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    runtime = load_progress_runtime(out_dir)
    for seed in seeds:
        base_path = out_dir / f"seed_{seed}" / "baseline.json"
        if base_path.exists():
            d = load_json(base_path)
            details = d.get("details", {})
            meta = d.get("meta", {})
            served = details.get("served_requests", "")
            rejected = details.get("rejected_requests", "")
            requests = details.get("requests_total", details.get("raw_requests", ""))
            if requests in ("", None) and served not in ("", None) and rejected not in ("", None):
                requests = float(served) + float(rejected)
            profit_fields = comparable_profit_fields(
                reported_profit_eur=details.get("profit_eur", ""),
                revenue_eur=details.get("revenue_eur", ""),
                ops_charge_credit_eur=details.get("ops_charge_credit_eur", ""),
                drive_km_total=details.get("drive_km_total", ""),
                relocation_cost_eur=details.get("cost_relocation_eur", ""),
                energy_cost_eur=details.get("cost_energy_eur", ""),
                ops_cost_eur=details.get("cost_ops_eur", ""),
                ops_deadhead_cost_eur=details.get("cost_ops_deadhead_eur", ""),
                penalty_cost_eur=details.get("cost_penalty_eur", ""),
                meta=meta,
            )
            rows.append(
                {
                    "seed": seed,
                    "mode": "baseline",
                    "objective_source": "baseline_rollout",
                    "f1_service": metric(d, "objectives", "f1_service"),
                    "f2_profit": metric(d, "objectives", "f2_profit"),
                    "f3_health": metric(d, "objectives", "f3_health"),
                    "pareto_size": "",
                    "front_records": "",
                    "requests": requests,
                    "served": served,
                    "rejected": rejected,
                    "service_rate": details.get("service_rate", ""),
                    "profit_eur": details.get("profit_eur", ""),
                    **profit_fields,
                    "drive_km_total": details.get("drive_km_total", ""),
                    "eda_reopt_count": "",
                    **empty_q1_audit_fields(),
                    "runtime_sec": runtime.get((seed, "baseline"), ""),
                }
            )

        static_path = out_dir / f"seed_{seed}" / "static_eda.json"
        if static_path.exists():
            d = load_json(static_path)
            best = d.get("best_by_sum", {})
            details = best.get("details", {}) if isinstance(best, dict) else {}
            meta = d.get("meta", {}) if isinstance(d, dict) else {}
            served = details.get("served_requests", "")
            rejected = details.get("rejected_requests", "")
            requests = details.get("requests_total", details.get("raw_requests", details.get("total_requests", "")))
            if requests in ("", None) and served not in ("", None) and rejected not in ("", None):
                requests = float(served) + float(rejected)
            profit_fields = comparable_profit_fields(
                reported_profit_eur=details.get("profit_eur", ""),
                revenue_eur=details.get("revenue_eur", ""),
                ops_charge_credit_eur=details.get("ops_charge_credit_eur", ""),
                drive_km_total=details.get("drive_km_total", ""),
                relocation_cost_eur=details.get("cost_relocation_eur", ""),
                energy_cost_eur=details.get("cost_energy_eur", ""),
                ops_cost_eur=details.get("cost_ops_eur", ""),
                ops_deadhead_cost_eur=details.get("cost_ops_deadhead_eur", ""),
                penalty_cost_eur=details.get("cost_penalty_eur", ""),
                meta=meta,
            )
            rows.append(
                {
                    "seed": seed,
                    "mode": "static_eda",
                    "objective_source": "static_eda_best_by_sum",
                    "f1_service": best.get("f1", ""),
                    "f2_profit": best.get("f2", ""),
                    "f3_health": best.get("f3", ""),
                    "pareto_size": d.get("pareto_size", ""),
                    "front_records": "",
                    "requests": requests,
                    "served": served,
                    "rejected": rejected,
                    "service_rate": details.get("service_rate", ""),
                    "profit_eur": details.get("profit_eur", ""),
                    **profit_fields,
                    "drive_km_total": details.get("drive_km_total", ""),
                    "eda_reopt_count": "",
                    **q1_audit_fields(d),
                    "runtime_sec": runtime.get((seed, "static_eda"), ""),
                }
            )

        for mode_name, file_name in [
            ("pymoo_nsga2", "pymoo_nsga2.json"),
            ("pymoo_dnsga2", "pymoo_dnsga2.json"),
        ]:
            pymoo_path = out_dir / f"seed_{seed}" / file_name
            if not pymoo_path.exists():
                continue
            d = load_json(pymoo_path)
            best = d.get("best_by_sum", {})
            details = best.get("details", {}) if isinstance(best, dict) else {}
            meta = d.get("meta", {}) if isinstance(d, dict) else {}
            served = details.get("served_requests", "")
            rejected = details.get("rejected_requests", "")
            requests = details.get("requests_total", details.get("raw_requests", ""))
            if requests in ("", None) and served not in ("", None) and rejected not in ("", None):
                requests = float(served) + float(rejected)
            profit_fields = comparable_profit_fields(
                reported_profit_eur=details.get("profit_eur", ""),
                revenue_eur=details.get("revenue_eur", ""),
                ops_charge_credit_eur=details.get("ops_charge_credit_eur", ""),
                drive_km_total=details.get("drive_km_total", ""),
                relocation_cost_eur=details.get("cost_relocation_eur", ""),
                energy_cost_eur=details.get("cost_energy_eur", ""),
                ops_cost_eur=details.get("cost_ops_eur", ""),
                ops_deadhead_cost_eur=details.get("cost_ops_deadhead_eur", ""),
                penalty_cost_eur=details.get("cost_penalty_eur", ""),
                meta=meta,
            )
            rows.append(
                {
                    "seed": seed,
                    "mode": mode_name,
                    "objective_source": "pymoo_static_best_by_sum",
                    "f1_service": best.get("f1", ""),
                    "f2_profit": best.get("f2", ""),
                    "f3_health": best.get("f3", ""),
                    "pareto_size": d.get("pareto_size", ""),
                    "front_records": "",
                    "requests": requests,
                    "served": served,
                    "rejected": rejected,
                    "service_rate": details.get("service_rate", ""),
                    "profit_eur": details.get("profit_eur", ""),
                    **profit_fields,
                    "drive_km_total": details.get("drive_km_total", ""),
                    "eda_reopt_count": "",
                    **q1_audit_fields(d),
                    "runtime_sec": runtime.get((seed, mode_name), meta.get("exec_time_sec", "")),
                }
            )

        stateful_path = out_dir / f"seed_{seed}" / "pymoo_dnsga2_stateful.json"
        if stateful_path.exists():
            d = load_json(stateful_path)
            final = d.get("final_status", {})
            mean_objectives = d.get("mean_executed_objectives", [])
            profit_fields = comparable_profit_fields(
                reported_profit_eur=final.get("profit_eur", ""),
                revenue_eur=final.get("revenue_eur", ""),
                ops_charge_credit_eur=final.get("ops_charge_credit_eur", ""),
                drive_km_total=final.get("drive_km_total", ""),
                relocation_cost_eur=final.get("relocation_cost_eur", ""),
                energy_cost_eur=final.get("energy_cost_eur", ""),
                ops_cost_eur=final.get("ops_cost_eur", ""),
                ops_deadhead_cost_eur=final.get("ops_deadhead_cost_eur", ""),
                penalty_cost_eur=final.get("penalty_cost_eur", ""),
                meta=d.get("meta", {}),
            )
            rows.append(
                {
                    "seed": seed,
                    "mode": "pymoo_dnsga2_stateful",
                    "objective_source": "pymoo_stateful_mean_executed",
                    "f1_service": mean_objectives[0] if len(mean_objectives) >= 1 else "",
                    "f2_profit": mean_objectives[1] if len(mean_objectives) >= 2 else "",
                    "f3_health": mean_objectives[2] if len(mean_objectives) >= 3 else "",
                    "pareto_size": "",
                    "front_records": len(d.get("windows", [])),
                    "requests": final.get("requests", ""),
                    "served": final.get("served", ""),
                    "rejected": final.get("rejected", ""),
                    "service_rate": final.get("service_rate", ""),
                    "profit_eur": final.get("profit_eur", ""),
                    **profit_fields,
                    "drive_km_total": final.get("drive_km_total", ""),
                    "eda_reopt_count": d.get("pymoo_reopt_count", ""),
                    **q1_audit_fields(d),
                    "runtime_sec": runtime.get((seed, "pymoo_dnsga2_stateful"), ""),
                }
            )

        adr_stateful_path = out_dir / f"seed_{seed}" / "pymoo_dnsga2_adr_stateful.json"
        if adr_stateful_path.exists():
            d = load_json(adr_stateful_path)
            final = d.get("final_status", {})
            mean_objectives = d.get("mean_executed_objectives", [])
            profit_fields = comparable_profit_fields(
                reported_profit_eur=final.get("profit_eur", ""),
                revenue_eur=final.get("revenue_eur", ""),
                ops_charge_credit_eur=final.get("ops_charge_credit_eur", ""),
                drive_km_total=final.get("drive_km_total", ""),
                relocation_cost_eur=final.get("relocation_cost_eur", ""),
                energy_cost_eur=final.get("energy_cost_eur", ""),
                ops_cost_eur=final.get("ops_cost_eur", ""),
                ops_deadhead_cost_eur=final.get("ops_deadhead_cost_eur", ""),
                penalty_cost_eur=final.get("penalty_cost_eur", ""),
                meta=d.get("meta", {}),
            )
            rows.append(
                {
                    "seed": seed,
                    "mode": "pymoo_dnsga2_adr_stateful",
                    "objective_source": "pymoo_adr_stateful_mean_executed",
                    "f1_service": mean_objectives[0] if len(mean_objectives) >= 1 else "",
                    "f2_profit": mean_objectives[1] if len(mean_objectives) >= 2 else "",
                    "f3_health": mean_objectives[2] if len(mean_objectives) >= 3 else "",
                    "pareto_size": "",
                    "front_records": len(d.get("windows", [])),
                    "requests": final.get("requests", ""),
                    "served": final.get("served", ""),
                    "rejected": final.get("rejected", ""),
                    "service_rate": final.get("service_rate", ""),
                    "profit_eur": final.get("profit_eur", ""),
                    **profit_fields,
                    "drive_km_total": final.get("drive_km_total", ""),
                    "eda_reopt_count": d.get("pymoo_reopt_count", ""),
                    **q1_audit_fields(d),
                    "runtime_sec": runtime.get((seed, "pymoo_dnsga2_adr_stateful"), ""),
                }
            )

        online_summary = out_dir / f"seed_{seed}" / "online_adr_audit_summary.json"
        online_fronts = out_dir / f"seed_{seed}" / "online_adr_fronts.json"
        if online_summary.exists():
            d = load_json(online_summary)
            final = d.get("final_status", {})
            metrics = d.get("metrics", {})
            meta = d.get("meta", {})
            fronts = load_json(online_fronts) if online_fronts.exists() else {}
            selected_objectives, last_pareto_size, front_records = last_selected_objectives(fronts)
            reported_profit = final.get("profit_eur", metrics.get("profit_eur", ""))
            profit_fields = comparable_profit_fields(
                reported_profit_eur=reported_profit,
                revenue_eur=metrics.get("revenue_eur", final.get("revenue_eur", "")),
                ops_charge_credit_eur=metrics.get(
                    "ops_charge_credit_eur",
                    final.get("ops_charge_credit_eur", ""),
                ),
                drive_km_total=metrics.get("drive_km_total", ""),
                relocation_cost_eur=metrics.get(
                    "relocation_cost_eur",
                    final.get("relocation_cost_eur", ""),
                ),
                energy_cost_eur=metrics.get(
                    "energy_cost_eur",
                    final.get("energy_cost_eur", ""),
                ),
                ops_cost_eur=metrics.get("ops_cost_eur", final.get("ops_cost_eur", "")),
                ops_deadhead_cost_eur=metrics.get(
                    "ops_deadhead_cost_eur",
                    final.get("ops_deadhead_cost_eur", ""),
                ),
                penalty_cost_eur=metrics.get(
                    "penalty_cost_eur",
                    final.get("penalty_cost_eur", ""),
                ),
                meta=meta,
            )
            rows.append(
                {
                    "seed": seed,
                    "mode": "online_adr_eda",
                    "objective_source": "online_last_selected_front",
                    "f1_service": selected_objectives[0] if selected_objectives else "",
                    "f2_profit": selected_objectives[1] if selected_objectives else "",
                    "f3_health": selected_objectives[2] if selected_objectives else "",
                    "pareto_size": last_pareto_size or "",
                    "front_records": front_records,
                    "requests": final.get("requests", metrics.get("requests_total", "")),
                    "served": final.get("served", metrics.get("served_requests", "")),
                    "rejected": final.get("rejected", metrics.get("rejected_requests", "")),
                    "service_rate": final.get("service_rate", metrics.get("service_rate", "")),
                    "profit_eur": reported_profit,
                    **profit_fields,
                    "drive_km_total": metrics.get("drive_km_total", ""),
                    "eda_reopt_count": metric(fronts, "meta", "eda_reopt_count"),
                    **q1_audit_fields(d, fronts),
                    "runtime_sec": runtime.get((seed, "online_adr_eda"), ""),
                }
            )

    summary_path = out_dir / "summary.csv"
    if rows:
        with summary_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _mean(mode: str, key: str) -> float | None:
        values = []
        for row in rows:
            if row.get("mode") != mode:
                continue
            value = row.get(key)
            if value in ("", None):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return sum(values) / len(values)

    baseline_service = _mean("baseline", "service_rate")
    online_service = _mean("online_adr_eda", "service_rate")
    baseline_profit = _mean("baseline", "profit_eur")
    online_profit = _mean("online_adr_eda", "profit_eur")
    baseline_comparable_profit = _mean("baseline", "comparable_profit_eur")
    online_comparable_profit = _mean("online_adr_eda", "comparable_profit_eur")
    online_runtime = _mean("online_adr_eda", "runtime_sec")
    online_reopts = _mean("online_adr_eda", "eda_reopt_count")
    pymoo_nsga2_service = _mean("pymoo_nsga2", "service_rate")
    pymoo_dnsga2_service = _mean("pymoo_dnsga2", "service_rate")
    pymoo_nsga2_comparable_profit = _mean("pymoo_nsga2", "comparable_profit_eur")
    pymoo_dnsga2_comparable_profit = _mean("pymoo_dnsga2", "comparable_profit_eur")
    pymoo_nsga2_runtime = _mean("pymoo_nsga2", "runtime_sec")
    pymoo_dnsga2_runtime = _mean("pymoo_dnsga2", "runtime_sec")
    pymoo_stateful_service = _mean("pymoo_dnsga2_stateful", "service_rate")
    pymoo_stateful_comparable_profit = _mean("pymoo_dnsga2_stateful", "comparable_profit_eur")
    pymoo_stateful_runtime = _mean("pymoo_dnsga2_stateful", "runtime_sec")
    pymoo_stateful_reopts = _mean("pymoo_dnsga2_stateful", "eda_reopt_count")
    pymoo_adr_stateful_service = _mean("pymoo_dnsga2_adr_stateful", "service_rate")
    pymoo_adr_stateful_comparable_profit = _mean("pymoo_dnsga2_adr_stateful", "comparable_profit_eur")
    pymoo_adr_stateful_runtime = _mean("pymoo_dnsga2_adr_stateful", "runtime_sec")
    pymoo_adr_stateful_reopts = _mean("pymoo_dnsga2_adr_stateful", "eda_reopt_count")

    report_path = out_dir / "REALCAR_MINI_REPORT.md"
    report_path.write_text(
        "# RealCar-mini free-floating Pareto campaign\n\n"
        f"Generated: {now_iso()}\n\n"
        "This is a small validation campaign for the original free-floating "
        "multiobjective carsharing model using real Munich trip demand.\n\n"
        "## Reading\n\n"
        "This campaign keeps the original free-floating multiobjective "
        "carsharing model and replaces synthetic demand with observed Munich "
        "OD trips.\n\n"
        f"- Baseline mean service: {baseline_service if baseline_service is not None else 'n/a'}\n"
        f"- Online ADR-EDA mean service: {online_service if online_service is not None else 'n/a'}\n"
        f"- Baseline mean reported profit EUR: {baseline_profit if baseline_profit is not None else 'n/a'}\n"
        f"- Online ADR-EDA mean reported profit EUR: {online_profit if online_profit is not None else 'n/a'}\n"
        f"- Baseline mean comparable profit EUR: {baseline_comparable_profit if baseline_comparable_profit is not None else 'n/a'}\n"
        f"- Online ADR-EDA mean comparable profit EUR: {online_comparable_profit if online_comparable_profit is not None else 'n/a'}\n"
        f"- Online ADR-EDA mean reoptimizations: {online_reopts if online_reopts is not None else 'n/a'}\n"
        f"- Online ADR-EDA mean runtime sec: {online_runtime if online_runtime is not None else 'n/a'}\n\n"
        f"- pymoo NSGA-II mean service: {pymoo_nsga2_service if pymoo_nsga2_service is not None else 'n/a'}\n"
        f"- pymoo D-NSGA-II mean service: {pymoo_dnsga2_service if pymoo_dnsga2_service is not None else 'n/a'}\n"
        f"- pymoo NSGA-II mean comparable profit EUR: {pymoo_nsga2_comparable_profit if pymoo_nsga2_comparable_profit is not None else 'n/a'}\n"
        f"- pymoo D-NSGA-II mean comparable profit EUR: {pymoo_dnsga2_comparable_profit if pymoo_dnsga2_comparable_profit is not None else 'n/a'}\n"
        f"- pymoo NSGA-II mean runtime sec: {pymoo_nsga2_runtime if pymoo_nsga2_runtime is not None else 'n/a'}\n"
        f"- pymoo D-NSGA-II mean runtime sec: {pymoo_dnsga2_runtime if pymoo_dnsga2_runtime is not None else 'n/a'}\n\n"
        f"- pymoo stateful D-NSGA-II mean service: {pymoo_stateful_service if pymoo_stateful_service is not None else 'n/a'}\n"
        f"- pymoo stateful D-NSGA-II mean comparable profit EUR: {pymoo_stateful_comparable_profit if pymoo_stateful_comparable_profit is not None else 'n/a'}\n"
        f"- pymoo stateful D-NSGA-II mean reoptimizations: {pymoo_stateful_reopts if pymoo_stateful_reopts is not None else 'n/a'}\n"
        f"- pymoo stateful D-NSGA-II mean runtime sec: {pymoo_stateful_runtime if pymoo_stateful_runtime is not None else 'n/a'}\n\n"
        f"- pymoo ADR-controlled D-NSGA-II mean service: {pymoo_adr_stateful_service if pymoo_adr_stateful_service is not None else 'n/a'}\n"
        f"- pymoo ADR-controlled D-NSGA-II mean comparable profit EUR: {pymoo_adr_stateful_comparable_profit if pymoo_adr_stateful_comparable_profit is not None else 'n/a'}\n"
        f"- pymoo ADR-controlled D-NSGA-II mean reoptimizations: {pymoo_adr_stateful_reopts if pymoo_adr_stateful_reopts is not None else 'n/a'}\n"
        f"- pymoo ADR-controlled D-NSGA-II mean runtime sec: {pymoo_adr_stateful_runtime if pymoo_adr_stateful_runtime is not None else 'n/a'}\n\n"
        "The `objective_source` column clarifies whether `f1/f2/f3` come from "
        "a baseline rollout, a static EDA/Pymoo Pareto front, or the last selected online "
        "front. The multiobjective `f2_profit` is unchanged; operational KPIs "
        "such as requests, served, profit and km are reported separately. The "
        "original simulator/audit `profit_eur` is preserved as "
        "`reported_profit_eur`; `comparable_profit_eur` adds a homogeneous "
        "drive-energy proxy to both baseline and online modes for "
        "finite-horizon economic comparisons, but it is not used as the EDA "
        "objective.\n\n"
        "## Artifacts\n\n"
        f"- progress: `{out_dir / 'progress.csv'}`\n"
        f"- summary: `{summary_path}`\n"
        "- per-seed logs and JSON files under `seed_*` folders\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--travel-metrics-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="2026,2027,2028")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--fleet-size", type=int, default=70)
    ap.add_argument("--charger-count", type=int, default=12)
    ap.add_argument("--ops-resources", type=int, default=14)
    ap.add_argument("--size-gen", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=1)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.csv"
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    sim_duration_min = int(round(float(args.hours) * 60.0))

    if bool(args.summarize_only):
        summarize(out_dir=out_dir, seeds=seeds)
        return 0

    common = [
        sys.executable,
        "-m",
        "free_floating_robust.runner",
        "--real-demand-profile-dir",
        str(args.profile_dir),
        "--travel-metrics-json",
        str(args.travel_metrics_json),
        "--fleet-size",
        str(args.fleet_size),
        "--charger-count",
        str(args.charger_count),
        "--ops-resources",
        str(args.ops_resources),
        "--horizon-hours",
        str(args.hours),
        "--decision-interval-min",
        "30",
        "--planning-horizon-min",
        "60",
        "--fare-mode",
        "time_plus_km",
        "--fare-base-eur",
        "1.0",
        "--fare-per-min-eur",
        "0.22",
        "--fare-per-km-eur",
        "0.35",
    ]

    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        run_cmd(
            cmd=[
                *common,
                "--mode",
                "baseline",
                "--seed",
                str(seed),
                "--output-json",
                str(seed_dir / "baseline.json"),
            ],
            log_path=seed_dir / "baseline.log",
            progress_path=progress_path,
            seed=seed,
            mode="baseline",
            output_path=seed_dir / "baseline.json",
        )

        run_cmd(
            cmd=[
                *common,
                "--mode",
                "static-eda",
                "--seed",
                str(seed),
                "--optimizer-random-seed",
                str(seed),
                "--size-gen",
                str(args.size_gen),
                "--max-iter",
                str(args.max_iter),
                "--dead-iter",
                "1",
                "--output-json",
                str(seed_dir / "static_eda.json"),
            ],
            log_path=seed_dir / "static_eda.log",
            progress_path=progress_path,
            seed=seed,
            mode="static_eda",
            output_path=seed_dir / "static_eda.json",
        )

        for mode_name, algorithm, file_name in [
            ("pymoo_nsga2", "nsga2", "pymoo_nsga2.json"),
            ("pymoo_dnsga2", "dnsga2", "pymoo_dnsga2.json"),
        ]:
            run_cmd(
                cmd=[
                    *common,
                    "--mode",
                    "pymoo-static",
                    "--seed",
                    str(seed),
                    "--optimizer-random-seed",
                    str(seed),
                    "--pymoo-algorithm",
                    algorithm,
                    "--size-gen",
                    str(args.size_gen),
                    "--max-iter",
                    str(args.max_iter),
                    "--output-json",
                    str(seed_dir / file_name),
                ],
                log_path=seed_dir / f"{mode_name}.log",
                progress_path=progress_path,
                seed=seed,
                mode=mode_name,
                output_path=seed_dir / file_name,
            )

        run_cmd(
            cmd=[
                *common,
                "--mode",
                "pymoo-stateful",
                "--seed",
                str(seed),
                "--optimizer-random-seed",
                str(seed),
                "--pymoo-algorithm",
                "dnsga2",
                "--sim-duration-min",
                str(sim_duration_min),
                "--horizon-hours",
                "1",
                "--pymoo-execution-window-min",
                "30",
                "--size-gen",
                str(args.size_gen),
                "--max-iter",
                str(args.max_iter),
                "--output-json",
                str(seed_dir / "pymoo_dnsga2_stateful.json"),
            ],
            log_path=seed_dir / "pymoo_dnsga2_stateful.log",
            progress_path=progress_path,
            seed=seed,
            mode="pymoo_dnsga2_stateful",
            output_path=seed_dir / "pymoo_dnsga2_stateful.json",
        )

        run_cmd(
            cmd=[
                *common,
                "--mode",
                "pymoo-adr-stateful",
                "--seed",
                str(seed),
                "--optimizer-random-seed",
                str(seed),
                "--pymoo-algorithm",
                "dnsga2",
                "--sim-duration-min",
                str(sim_duration_min),
                "--horizon-hours",
                "1",
                "--pymoo-execution-window-min",
                "30",
                "--size-gen",
                str(args.size_gen),
                "--max-iter",
                str(args.max_iter),
                "--output-json",
                str(seed_dir / "pymoo_dnsga2_adr_stateful.json"),
            ],
            log_path=seed_dir / "pymoo_dnsga2_adr_stateful.log",
            progress_path=progress_path,
            seed=seed,
            mode="pymoo_dnsga2_adr_stateful",
            output_path=seed_dir / "pymoo_dnsga2_adr_stateful.json",
        )

        run_cmd(
            cmd=[
                *common,
                "--mode",
                "online-eda",
                "--seed",
                str(seed),
                "--optimizer-random-seed",
                str(seed),
                "--sim-duration-min",
                str(sim_duration_min),
                "--horizon-hours",
                "1",
                "--size-gen",
                str(args.size_gen),
                "--max-iter",
                str(args.max_iter),
                "--dead-iter",
                "1",
                "--eda-online-engine",
                "dynamic-stateful",
                "--dynamic-online-profile",
                "adr_eda_v2_2_carsharing",
                "--output-json",
                str(seed_dir / "online_adr.json"),
                "--audit-prefix",
                str(seed_dir / "online_adr_audit"),
                "--eda-fronts-json",
                str(seed_dir / "online_adr_fronts.json"),
            ],
            log_path=seed_dir / "online_adr.log",
            progress_path=progress_path,
            seed=seed,
            mode="online_adr_eda",
            output_path=seed_dir / "online_adr.json",
        )

    summarize(out_dir=out_dir, seeds=seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
