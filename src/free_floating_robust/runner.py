from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .evaluator import FreeFloatingRobustDMOOP
from .ops_cost_profiles import apply_ops_cost_profile
from .planner import RobustDecisionEncoding
from .scenario import (
    MadridFreeFloatingScenario,
    ScenarioConfig,
    build_free_floating_scenario,
    default_travel_metrics_json,
)
from .types import RobustModelConfig


def _to_list(a: np.ndarray) -> list:
    return np.asarray(a, dtype=float).tolist()


def _seed_optimizer_random_state(raw_seed: int | None) -> Optional[int]:
    if raw_seed is None:
        return None
    seed = int(raw_seed)
    if seed < 0:
        return None
    seed = int(seed % (2**32 - 1))
    random.seed(seed)
    np.random.seed(seed)
    return seed


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _evaluation_stats(problem: FreeFloatingRobustDMOOP) -> Dict[str, float]:
    return {key: float(value) for key, value in problem.evaluation_stats().items()}


def _evaluation_stats_delta(
    after: Dict[str, float],
    before: Dict[str, float],
) -> Dict[str, float]:
    keys = set(after) | set(before)
    return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys}


def _rollout_eval_count(stats: Dict[str, float]) -> int:
    return int(round(float(stats.get("rollout_call_count", 0.0))))


def _rollout_runtime_sec(stats: Dict[str, float]) -> float:
    return float(stats.get("rollout_wall_time_sec", 0.0))


def _best_by_sum_index(front: np.ndarray) -> Optional[int]:
    if front.ndim != 2 or front.shape[0] == 0:
        return None
    sums = np.sum(np.asarray(front, dtype=float), axis=1)
    return int(np.argmax(sums))


def _best_by_normalized_sum_index(front: np.ndarray) -> Optional[int]:
    if front.ndim != 2 or front.shape[0] == 0:
        return None
    values = np.asarray(front, dtype=float)
    lo = np.min(values, axis=0)
    hi = np.max(values, axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    scores = np.sum((values - lo) / span, axis=1)
    return int(np.argmax(scores))


def _state_metrics_payload(state: Any) -> Dict[str, float]:
    metrics = state.metrics
    return {
        "requests": float(metrics.requests_total),
        "served": float(metrics.served_requests),
        "rejected": float(metrics.rejected_requests),
        "service_rate": float(metrics.service_rate),
        "profit_eur": float(metrics.profit_eur),
        "revenue_eur": float(metrics.revenue_eur),
        "ops_charge_credit_eur": float(metrics.ops_charge_credit_eur),
        "relocation_cost_eur": float(metrics.relocation_cost_eur),
        "energy_cost_eur": float(metrics.energy_cost_eur),
        "ops_cost_eur": float(metrics.ops_cost_eur),
        "ops_deadhead_cost_eur": float(metrics.ops_deadhead_cost_eur),
        "penalty_cost_eur": float(metrics.penalty_cost_eur),
        "drive_km_total": float(metrics.drive_km_total),
        "ops_deadhead_km_total": float(metrics.ops_deadhead_km_total),
        "charged_kwh_total": float(metrics.charged_kwh_total),
        "soh_loss_total": float(metrics.soh_loss_total),
        "bookings_total": float(metrics.bookings_total),
        "canceled_bookings": float(metrics.canceled_bookings),
        "preassigned_bookings": float(metrics.preassigned_bookings),
        "reservation_pickups_due": float(metrics.reservation_pickups_due),
        "preassigned_pickups_honored": float(metrics.preassigned_pickups_honored),
        "preassigned_pickups_failed": float(metrics.preassigned_pickups_failed),
    }


def _scenario_meta(
    *,
    scenario: MadridFreeFloatingScenario,
    model_cfg: RobustModelConfig,
    state0: Any,
) -> Dict[str, Any]:
    return {
        "travel_metrics_source": (
            None if state0.travel_metrics is None else str(state0.travel_metrics.source)
        ),
        "travel_metrics_has_congestion_bands": bool(
            state0.travel_metrics is not None and getattr(state0.travel_metrics, "congestion_time_bands", [])
        ),
        "travel_metrics_congestion_band_names": (
            []
            if state0.travel_metrics is None
            else [
                str(band.get("name", f"band_{idx}"))
                for idx, band in enumerate(getattr(state0.travel_metrics, "congestion_time_bands", []))
                if isinstance(band, dict)
            ]
        ),
        "fare_mode": str(scenario.cfg.fare_mode),
        "fare_per_min_eur": float(scenario.cfg.fare_per_min_eur),
        "fare_per_km_eur": float(scenario.cfg.fare_per_km_eur),
        "fare_airport_pickup_fee_eur": float(scenario.cfg.fare_airport_pickup_fee_eur),
        "fare_airport_dropoff_fee_eur": float(scenario.cfg.fare_airport_dropoff_fee_eur),
        "reservation_share": float(scenario.cfg.reservation_share),
        "reservation_booking_intensity_mult": float(scenario.cfg.reservation_booking_intensity_mult),
        "immediate_request_intensity_mult": float(scenario.cfg.immediate_request_intensity_mult),
        "reservation_min_lead_min": float(scenario.cfg.reservation_min_lead_min),
        "reservation_mean_lead_min": float(scenario.cfg.reservation_mean_lead_min),
        "reservation_max_lead_min": float(scenario.cfg.reservation_max_lead_min),
        "reservation_cancel_base_prob": float(scenario.cfg.reservation_cancel_base_prob),
        "reservation_cancel_lead_sensitivity": float(scenario.cfg.reservation_cancel_lead_sensitivity),
        "reservation_cancel_cap": float(scenario.cfg.reservation_cancel_cap),
        "max_user_access_walk_min": float(model_cfg.max_user_access_walk_min),
        "ops_cost_profile": str(model_cfg.ops_cost_profile),
        "relocation_cost_per_km": float(model_cfg.relocation_cost_per_km),
        "ops_deadhead_cost_per_km": float(model_cfg.ops_deadhead_cost_per_km),
        "ops_cost_per_action_eur": float(model_cfg.ops_cost_per_action_eur),
        "energy_cost_per_kwh": float(model_cfg.energy_cost_per_kwh),
        "ops_relocation_handover_min": float(model_cfg.ops_relocation_handover_min),
        "ops_charge_plug_min": float(model_cfg.ops_charge_plug_min),
        "zone_base_demand_json": None if not scenario.cfg.zone_base_demand_json else str(scenario.cfg.zone_base_demand_json),
        "scenario_class": type(scenario).__name__,
        "demand_source": str(getattr(scenario, "demand_source", "synthetic")),
        "real_demand_profile_dir": str(getattr(scenario, "profile_dir", "")),
        "real_demand_request_count": int(getattr(scenario, "real_request_count", 0)),
        "real_demand_time_origin": getattr(scenario, "real_time_origin", None),
        "real_demand_forecast_history_min": float(
            getattr(scenario, "forecast_history_min", 0.0)
        ),
        "zone_base_demand_per_hour": {
            str(zone_id): float(zone.base_demand_per_hour) for zone_id, zone in state0.zones.items()
        },
        "zones": list(state0.zones.keys()),
    }


def _build_context(
    *,
    seed: int,
    decision_interval_min: int,
    planning_horizon_min: int,
    fleet_size: int,
    charger_count: int,
    ops_resources: int,
    travel_metrics_json: str = "",
    fare_mode: str = "time_only",
    fare_base_eur: float = 0.0,
    fare_per_km_eur: float = 0.0,
    fare_per_min_eur: float = 0.24,
    fare_min_total_eur: float = 0.0,
    fare_included_km: float = 200.0,
    fare_extra_km_eur: float = 0.39,
    reservation_share: float = 0.28,
    reservation_booking_intensity_mult: float = 1.0,
    immediate_request_intensity_mult: float = 1.0,
    reservation_mean_lead_min: float = 35.0,
    ops_charge_action_credit_eur: float = 0.0,
    ops_cost_profile: str = "custom",
    relocation_cost_per_km: float = 0.22,
    ops_deadhead_cost_per_km: float = 0.22,
    ops_cost_per_action_eur: float = 1.20,
    energy_cost_per_kwh: float = 0.23,
    max_user_access_walk_min: float = 12.0,
    ops_relocation_handover_min: float = 1.5,
    ops_charge_plug_min: float = 2.0,
    zone_base_demand_json: str = "",
    real_demand_profile_dir: str = "",
    real_demand_forecast_history_min: float = 1440.0,
) -> tuple[RobustModelConfig, MadridFreeFloatingScenario, Any, RobustDecisionEncoding]:
    resolved_travel_metrics_json = str(travel_metrics_json).strip() or default_travel_metrics_json()
    model_cfg = RobustModelConfig(
        decision_interval_min=int(decision_interval_min),
        planning_horizon_min=int(planning_horizon_min),
        ops_cost_profile=str(ops_cost_profile),
        relocation_cost_per_km=float(relocation_cost_per_km),
        ops_charge_action_credit_eur=float(ops_charge_action_credit_eur),
        ops_deadhead_cost_per_km=float(ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(ops_cost_per_action_eur),
        energy_cost_per_kwh=float(energy_cost_per_kwh),
        max_user_access_walk_min=float(max_user_access_walk_min),
        ops_relocation_handover_min=float(ops_relocation_handover_min),
        ops_charge_plug_min=float(ops_charge_plug_min),
    )
    scenario = build_free_floating_scenario(
        cfg=ScenarioConfig(
            fleet_size=int(fleet_size),
            charger_count=int(charger_count),
            ops_resources=int(ops_resources),
            travel_metrics_json=resolved_travel_metrics_json,
            fare_mode=str(fare_mode),
            fare_base_eur=float(fare_base_eur),
            fare_per_km_eur=float(fare_per_km_eur),
            fare_per_min_eur=float(fare_per_min_eur),
            fare_min_total_eur=float(fare_min_total_eur),
            fare_included_km=float(fare_included_km),
            fare_extra_km_eur=float(fare_extra_km_eur),
            reservation_share=float(reservation_share),
            reservation_booking_intensity_mult=float(reservation_booking_intensity_mult),
            immediate_request_intensity_mult=float(immediate_request_intensity_mult),
            reservation_mean_lead_min=float(reservation_mean_lead_min),
            zone_base_demand_json=str(zone_base_demand_json).strip() or None,
        ),
        real_demand_profile_dir=str(real_demand_profile_dir),
        real_demand_forecast_history_min=float(real_demand_forecast_history_min),
    )
    state0 = scenario.initial_state(seed=int(seed), model_cfg=model_cfg)
    encoding = RobustDecisionEncoding(zone_ids=list(state0.zones.keys()))
    return model_cfg, scenario, state0, encoding


def _build_evaluator(
    *,
    state_template: Any,
    scenario: MadridFreeFloatingScenario,
    encoding: RobustDecisionEncoding,
    config: RobustModelConfig,
    horizon_hours: float,
    random_seed: int,
) -> FreeFloatingRobustDMOOP:
    return FreeFloatingRobustDMOOP(
        state_template=state_template,
        scenario=scenario,
        encoding=encoding,
        config=config,
        horizon_hours=float(horizon_hours),
        random_seed=int(random_seed),
    )


def _full_shock(zone_ids: List[str], base: float, overrides: Dict[str, float]) -> Dict[str, float]:
    out = {z: float(base) for z in zone_ids}
    for zone_id, value in (overrides or {}).items():
        if zone_id in out:
            out[zone_id] = float(value)
    return out


def _build_windows(
    *,
    zone_ids: List[str],
    base_demand_scale: float,
    max_iter: int,
    window_pack: str,
    key: str,
) -> List[Dict[str, Any]]:
    core = [
        {
            "id": "weekday_baseline",
            key: _full_shock(zone_ids, base_demand_scale, {}),
            "max_iter": int(max_iter),
        },
        {
            "id": "airport_peak",
            key: _full_shock(
                zone_ids,
                base_demand_scale,
                {
                    "Airport": base_demand_scale * 1.35,
                    "Centro": base_demand_scale * 1.20,
                    "Moncloa": base_demand_scale * 1.15,
                },
            ),
            "max_iter": int(max_iter),
        },
        {
            "id": "late_evening_reversion",
            key: _full_shock(
                zone_ids,
                base_demand_scale,
                {
                    "Airport": base_demand_scale * 0.90,
                    "Centro": base_demand_scale * 1.00,
                },
            ),
            "max_iter": int(max_iter),
        },
    ]

    if window_pack == "core":
        return core

    extended = core + [
        {
            "id": "morning_commute_north",
            key: _full_shock(
                zone_ids,
                base_demand_scale,
                {
                    "Chamartin": base_demand_scale * 1.35,
                    "Tetuan": base_demand_scale * 1.25,
                    "Centro": base_demand_scale * 1.12,
                    "Airport": base_demand_scale * 0.92,
                },
            ),
            "max_iter": int(max_iter),
        },
        {
            "id": "stadium_event_evening",
            key: _full_shock(
                zone_ids,
                base_demand_scale,
                {
                    "Moncloa": base_demand_scale * 1.45,
                    "Arganzuela": base_demand_scale * 1.35,
                    "Centro": base_demand_scale * 1.18,
                },
            ),
            "max_iter": int(max_iter),
        },
        {
            "id": "rainy_day_compact",
            key: _full_shock(
                zone_ids,
                base_demand_scale,
                {
                    "Centro": base_demand_scale * 1.25,
                    "Salamanca": base_demand_scale * 1.22,
                    "Retiro": base_demand_scale * 1.18,
                    "Airport": base_demand_scale * 0.88,
                },
            ),
            "max_iter": int(max_iter),
        },
    ]

    if window_pack == "extended":
        return extended

    if window_pack == "extreme":
        return extended + [
            {
                "id": "airport_disruption_spike",
                key: _full_shock(
                    zone_ids,
                    base_demand_scale,
                    {
                        "Airport": base_demand_scale * 1.70,
                        "Centro": base_demand_scale * 1.30,
                        "Salamanca": base_demand_scale * 1.15,
                    },
                ),
                "max_iter": int(max_iter),
            },
            {
                "id": "night_low_demand",
                key: _full_shock(
                    zone_ids,
                    base_demand_scale,
                    {
                        "Centro": base_demand_scale * 0.82,
                        "Airport": base_demand_scale * 0.76,
                        "Moncloa": base_demand_scale * 0.80,
                        "Tetuan": base_demand_scale * 0.84,
                    },
                ),
                "max_iter": int(max_iter),
            },
        ]

    raise ValueError("--window-pack debe ser core, extended o extreme")


def _expand_stateful_windows(
    *,
    windows: List[Dict[str, Any]],
    coarse_horizon_hours: float,
    observation_interval_min: float,
) -> List[Dict[str, Any]]:
    coarse_duration_min = max(1.0, 60.0 * float(coarse_horizon_hours))
    obs_min = float(observation_interval_min)
    if obs_min <= 0.0 or obs_min >= coarse_duration_min - 1e-9:
        out: List[Dict[str, Any]] = []
        start_min = 0.0
        for window in windows:
            expanded = dict(window)
            expanded["parent_window_id"] = str(window.get("id", "window"))
            expanded["subwindow_index"] = 0
            expanded["start_min"] = float(start_min)
            expanded["horizon_hours"] = float(coarse_horizon_hours)
            out.append(expanded)
            start_min += coarse_duration_min
        return out

    out: List[Dict[str, Any]] = []
    global_start_min = 0.0
    for window in windows:
        parent_window_id = str(window.get("id", "window"))
        local_elapsed_min = 0.0
        subwindow_index = 0
        while local_elapsed_min < coarse_duration_min - 1e-9:
            duration_min = min(obs_min, coarse_duration_min - local_elapsed_min)
            expanded = dict(window)
            expanded["id"] = f"{parent_window_id}_t{int(round(global_start_min)):04d}"
            expanded["parent_window_id"] = parent_window_id
            expanded["subwindow_index"] = int(subwindow_index)
            expanded["start_min"] = float(global_start_min)
            expanded["horizon_hours"] = float(duration_min) / 60.0
            out.append(expanded)
            local_elapsed_min += duration_min
            global_start_min += duration_min
            subwindow_index += 1
    return out


def _candidate_eda_configs(
    *,
    requested_family: str,
    requested_size_gen: int,
    requested_w_noise: float,
    fallback_families: List[str],
    auto_fallback: bool,
) -> List[tuple[str, int, float]]:
    configs: List[tuple[str, int, float]] = [
        (str(requested_family), int(requested_size_gen), float(requested_w_noise))
    ]
    if not bool(auto_fallback):
        return configs

    pop_candidates = [
        int(requested_size_gen),
        max(int(requested_size_gen), 40),
        max(int(requested_size_gen), 64),
    ]
    noise_candidates = [
        float(requested_w_noise),
        max(0.20, float(requested_w_noise)),
        max(0.30, float(requested_w_noise)),
    ]
    seen = set(configs)
    for family in fallback_families:
        for pop in pop_candidates:
            for noise in noise_candidates:
                key = (str(family), int(pop), float(noise))
                if key not in seen:
                    seen.add(key)
                    configs.append(key)
    return configs


def run_demo(
    seed: int = 2026,
    horizon_hours: int = 8,
    decision_interval_min: int = 5,
    planning_horizon_min: int = 60,
    fleet_size: int = 220,
    charger_count: int = 22,
    ops_resources: int = 26,
    external_demand_multiplier: Dict[str, float] | None = None,
    travel_metrics_json: str = "",
    fare_mode: str = "time_only",
    fare_base_eur: float = 0.0,
    fare_per_km_eur: float = 0.0,
    fare_per_min_eur: float = 0.24,
    fare_min_total_eur: float = 0.0,
    fare_included_km: float = 200.0,
    fare_extra_km_eur: float = 0.39,
    reservation_share: float = 0.28,
    reservation_booking_intensity_mult: float = 1.0,
    immediate_request_intensity_mult: float = 1.0,
    reservation_mean_lead_min: float = 35.0,
    ops_charge_action_credit_eur: float = 0.0,
    ops_cost_profile: str = "custom",
    relocation_cost_per_km: float = 0.22,
    ops_deadhead_cost_per_km: float = 0.22,
    ops_cost_per_action_eur: float = 1.20,
    energy_cost_per_kwh: float = 0.23,
    max_user_access_walk_min: float = 12.0,
    ops_relocation_handover_min: float = 1.5,
    ops_charge_plug_min: float = 2.0,
    zone_base_demand_json: str = "",
    real_demand_profile_dir: str = "",
    real_demand_forecast_history_min: float = 1440.0,
) -> dict:
    model_cfg, scenario, state0, encoding = _build_context(
        seed=int(seed),
        decision_interval_min=int(decision_interval_min),
        planning_horizon_min=int(planning_horizon_min),
        fleet_size=int(fleet_size),
        charger_count=int(charger_count),
        ops_resources=int(ops_resources),
        travel_metrics_json=str(travel_metrics_json),
        fare_mode=str(fare_mode),
        fare_base_eur=float(fare_base_eur),
        fare_per_km_eur=float(fare_per_km_eur),
        fare_per_min_eur=float(fare_per_min_eur),
        fare_min_total_eur=float(fare_min_total_eur),
        fare_included_km=float(fare_included_km),
        fare_extra_km_eur=float(fare_extra_km_eur),
        reservation_share=float(reservation_share),
        reservation_booking_intensity_mult=float(reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(immediate_request_intensity_mult),
        reservation_mean_lead_min=float(reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(ops_charge_action_credit_eur),
        ops_cost_profile=str(ops_cost_profile),
        relocation_cost_per_km=float(relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(ops_cost_per_action_eur),
        energy_cost_per_kwh=float(energy_cost_per_kwh),
        max_user_access_walk_min=float(max_user_access_walk_min),
        ops_relocation_handover_min=float(ops_relocation_handover_min),
        ops_charge_plug_min=float(ops_charge_plug_min),
        zone_base_demand_json=str(zone_base_demand_json),
        real_demand_profile_dir=str(real_demand_profile_dir),
        real_demand_forecast_history_min=float(real_demand_forecast_history_min),
    )

    x0 = np.mean(encoding.bounds, axis=1)
    evaluator = _build_evaluator(
        state_template=state0,
        scenario=scenario,
        encoding=encoding,
        config=model_cfg,
        horizon_hours=int(horizon_hours),
        random_seed=int(seed),
    )
    objectives, details, _ = evaluator.rollout(
        x=x0,
        external_demand_multiplier=external_demand_multiplier,
    )
    return {
        "mode": "baseline",
        "objectives": {
            "f1_service": float(objectives[0]),
            "f2_profit": float(objectives[1]),
            "f3_health": float(objectives[2]),
        },
        "details": {k: float(v) for k, v in details.items()},
        "meta": {
            "seed": int(seed),
            "horizon_hours": int(horizon_hours),
            "decision_interval_min": int(model_cfg.decision_interval_min),
            "planning_horizon_min": int(model_cfg.planning_horizon_min),
            "fleet_size": int(len(state0.vehicles)),
            **_scenario_meta(scenario=scenario, model_cfg=model_cfg, state0=state0),
        },
    }


def run_static_eda(args: argparse.Namespace) -> Dict[str, Any]:
    from . import SPEDARunConfig, SPEDAStaticRunner

    model_cfg, scenario, state0, encoding = _build_context(
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
    )
    problem = _build_evaluator(
        state_template=state0,
        scenario=scenario,
        encoding=encoding,
        config=model_cfg,
        horizon_hours=int(args.horizon_hours),
        random_seed=int(args.seed),
    )
    candidate_cfgs = _candidate_eda_configs(
        requested_family=str(args.optimizer_family),
        requested_size_gen=int(args.size_gen),
        requested_w_noise=float(args.w_noise),
        fallback_families=[str(args.optimizer_family), "KDEBN", "SPBN"],
        auto_fallback=bool(args.auto_fallback),
    )
    raw: Dict[str, Any] | None = None
    used_family = None
    used_size_gen = None
    used_w_noise = None
    fallback_errors: List[str] = []
    for family, size_gen, w_noise in candidate_cfgs:
        try:
            cfg = SPEDARunConfig(
                size_gen=int(size_gen),
                max_iter=int(args.max_iter),
                dead_iter=int(args.dead_iter),
                alpha=float(args.alpha),
                archive_generations=float(args.archive_generations),
                w_noise=float(w_noise),
                parallelize=bool(args.parallelize),
                disp=bool(args.disp),
                optimizer_family=str(family),
            )
            runner = SPEDAStaticRunner(encoding=encoding, config=cfg)
            problem.reset_evaluation_stats()
            optimizer_wall_start = time.monotonic()
            raw = runner.optimize(problem)
            optimizer_wall_time_sec = float(time.monotonic() - optimizer_wall_start)
            optimizer_stats = _evaluation_stats(problem)
            used_family = str(family)
            used_size_gen = int(size_gen)
            used_w_noise = float(w_noise)
            break
        except Exception as exc:
            fallback_errors.append(f"{family}/N={size_gen}/noise={w_noise:.2f}: {type(exc).__name__}: {exc}")
    if raw is None:
        raise RuntimeError("EDA estática falló en todas las configuraciones: " + " | ".join(fallback_errors))

    front = np.asarray(raw.get("pareto_front_max", np.empty((0, 3))), dtype=float)
    pset = np.asarray(raw.get("pareto_set", np.empty((0, encoding.dimension))), dtype=float)
    best_idx = _best_by_sum_index(front)
    selection_stats: Dict[str, float] = {}
    selection_wall_time_sec = 0.0

    report: Dict[str, Any] = {
        "mode": "static-eda",
        "meta": {
            "seed": int(args.seed),
            "horizon_hours": int(args.horizon_hours),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            **_scenario_meta(scenario=scenario, model_cfg=model_cfg, state0=state0),
            "encoding_dimension": int(encoding.dimension),
            "optimizer_family_requested": str(args.optimizer_family),
            "optimizer_family_used": str(raw.get("optimizer_family", used_family or args.optimizer_family)),
            "size_gen_requested": int(args.size_gen),
            "size_gen_used": int(used_size_gen or args.size_gen),
            "max_iter": int(args.max_iter),
            "w_noise_requested": float(args.w_noise),
            "w_noise_used": float(used_w_noise or args.w_noise),
            "auto_fallback": bool(args.auto_fallback),
            "fallback_errors": fallback_errors,
            "optimizer_wall_time_sec": float(optimizer_wall_time_sec),
            "optimizer_rollout_eval_count": _rollout_eval_count(optimizer_stats),
            "optimizer_rollout_runtime_sec": _rollout_runtime_sec(optimizer_stats),
        },
        "pareto_size": int(front.shape[0]),
        "pareto_front_max": _to_list(front),
        "pareto_set": _to_list(pset),
    }
    if best_idx is not None and best_idx >= 0 and best_idx < pset.shape[0]:
        before_selection = _evaluation_stats(problem)
        selection_wall_start = time.monotonic()
        selected_obj, selected_details, _ = problem.rollout(np.asarray(pset[best_idx], dtype=float))
        selection_wall_time_sec = float(time.monotonic() - selection_wall_start)
        selection_stats = _evaluation_stats_delta(_evaluation_stats(problem), before_selection)
        report["best_by_sum"] = {
            "index": int(best_idx),
            "score": float(np.sum(front[best_idx])),
            "f1": float(selected_obj[0]),
            "f2": float(selected_obj[1]),
            "f3": float(selected_obj[2]),
            "x": _to_list(pset[best_idx]),
            "details": {key: float(value) for key, value in selected_details.items()},
        }
    total_stats = _evaluation_stats(problem)
    report["meta"].update(
        {
            "selection_wall_time_sec": float(selection_wall_time_sec),
            "selection_rollout_eval_count": _rollout_eval_count(selection_stats),
            "selection_rollout_runtime_sec": _rollout_runtime_sec(selection_stats),
            "total_rollout_eval_count": _rollout_eval_count(total_stats),
            "total_rollout_runtime_sec": _rollout_runtime_sec(total_stats),
        }
    )
    return report


_PYMOO_ALGORITHM_LABELS = {
    "nsga2": "NSGA-II",
    "dnsga2": "D-NSGA-II",
    "spea2": "SPEA2",
    "sms": "SMS-EMOA",
    "smsemoa": "SMS-EMOA",
}


def _pymoo_algorithm_key(raw_key: str) -> str:
    key = str(raw_key).strip().lower()
    if key not in _PYMOO_ALGORITHM_LABELS:
        raise ValueError(f"--pymoo-algorithm no soportado: {raw_key}")
    return key


def _pymoo_algorithm_label(algorithm_key: str) -> str:
    return _PYMOO_ALGORITHM_LABELS[_pymoo_algorithm_key(algorithm_key)]


def _build_pymoo_algorithm(algorithm_key: str, pop_size: int, sampling: Any | None = None) -> Any:
    key = _pymoo_algorithm_key(algorithm_key)
    kwargs: Dict[str, Any] = {"pop_size": max(4, int(pop_size))}
    if sampling is not None:
        kwargs["sampling"] = sampling
    try:
        if key == "dnsga2":
            from pymoo.algorithms.moo.dnsga2 import DNSGA2

            return DNSGA2(version="B", **kwargs)
        if key == "nsga2":
            from pymoo.algorithms.moo.nsga2 import NSGA2

            return NSGA2(**kwargs)
        if key == "spea2":
            from pymoo.algorithms.moo.spea2 import SPEA2

            return SPEA2(**kwargs)
        if key in {"sms", "smsemoa"}:
            from pymoo.algorithms.moo.sms import SMSEMOA

            return SMSEMOA(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo inicializar {key} desde pymoo en el entorno actual."
        ) from exc
    raise ValueError(f"--pymoo-algorithm no soportado: {algorithm_key}")


def run_pymoo_static(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
    except Exception as exc:
        raise RuntimeError(
            "pymoo no esta disponible. Instala pymoo en el entorno Docker/configurado."
        ) from exc

    model_cfg, scenario, state0, encoding = _build_context(
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
    )
    problem = _build_evaluator(
        state_template=state0,
        scenario=scenario,
        encoding=encoding,
        config=model_cfg,
        horizon_hours=float(args.horizon_hours),
        random_seed=int(args.seed),
    )

    lower = encoding.bounds[:, 0].astype(float)
    upper = encoding.bounds[:, 1].astype(float)

    class PymooFreeFloatingProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=int(encoding.dimension), n_obj=3, xl=lower, xu=upper)

        def _evaluate(self, x, out, *eval_args, **eval_kwargs):
            out["F"] = np.asarray(problem.objective_min(np.asarray(x, dtype=float)), dtype=float)

    algorithm_key = _pymoo_algorithm_key(args.pymoo_algorithm)
    algorithm = _build_pymoo_algorithm(algorithm_key, pop_size=max(4, int(args.size_gen)))
    algorithm_label = _pymoo_algorithm_label(algorithm_key)

    seed = _seed_optimizer_random_state(getattr(args, "optimizer_random_seed", -1))
    problem.reset_evaluation_stats()
    optimizer_wall_start = time.monotonic()
    result = minimize(
        PymooFreeFloatingProblem(),
        algorithm,
        termination=("n_gen", max(1, int(args.max_iter))),
        seed=int(seed if seed is not None else args.seed),
        verbose=bool(args.disp),
        save_history=False,
    )
    optimizer_wall_time_sec = float(time.monotonic() - optimizer_wall_start)
    optimizer_stats = _evaluation_stats(problem)
    front_min = np.asarray(getattr(result, "F", np.empty((0, 3))), dtype=float)
    if front_min.ndim == 1 and front_min.size:
        front_min = front_min.reshape(1, -1)
    front_max = -front_min if front_min.size else np.empty((0, 3), dtype=float)
    pset = np.asarray(getattr(result, "X", np.empty((0, encoding.dimension))), dtype=float)
    if pset.ndim == 1 and pset.size:
        pset = pset.reshape(1, -1)

    best_idx = _best_by_sum_index(front_max)
    report: Dict[str, Any] = {
        "mode": "pymoo-static",
        "meta": {
            "seed": int(args.seed),
            "optimizer_random_seed": None if seed is None else int(seed),
            "horizon_hours": float(args.horizon_hours),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            **_scenario_meta(scenario=scenario, model_cfg=model_cfg, state0=state0),
            "encoding_dimension": int(encoding.dimension),
            "algorithm": algorithm_label,
            "pymoo_algorithm": algorithm_key,
            "size_gen": int(args.size_gen),
            "max_iter": int(args.max_iter),
            "exec_time_sec": float(getattr(result, "exec_time", 0.0) or 0.0),
            "optimizer_wall_time_sec": float(optimizer_wall_time_sec),
            "optimizer_rollout_eval_count": _rollout_eval_count(optimizer_stats),
            "optimizer_rollout_runtime_sec": _rollout_runtime_sec(optimizer_stats),
        },
        "pareto_size": int(front_max.shape[0]),
        "pareto_front_max": _to_list(front_max),
        "pareto_set": _to_list(pset),
    }
    selection_stats: Dict[str, float] = {}
    selection_wall_time_sec = 0.0
    if best_idx is not None and best_idx >= 0 and best_idx < pset.shape[0]:
        x_sel = np.asarray(pset[best_idx], dtype=float)
        before_selection = _evaluation_stats(problem)
        selection_wall_start = time.monotonic()
        obj, details, _ = problem.rollout(x_sel)
        selection_wall_time_sec = float(time.monotonic() - selection_wall_start)
        selection_stats = _evaluation_stats_delta(_evaluation_stats(problem), before_selection)
        report["best_by_sum"] = {
            "index": int(best_idx),
            "score": float(np.sum(front_max[best_idx])),
            "f1": float(obj[0]),
            "f2": float(obj[1]),
            "f3": float(obj[2]),
            "x": _to_list(x_sel),
            "details": {key: float(value) for key, value in details.items()},
        }
    total_stats = _evaluation_stats(problem)
    report["meta"].update(
        {
            "selection_wall_time_sec": float(selection_wall_time_sec),
            "selection_rollout_eval_count": _rollout_eval_count(selection_stats),
            "selection_rollout_runtime_sec": _rollout_runtime_sec(selection_stats),
            "total_rollout_eval_count": _rollout_eval_count(total_stats),
            "total_rollout_runtime_sec": _rollout_runtime_sec(total_stats),
        }
    )
    return report


def run_pymoo_stateful(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.sampling import Sampling
        from pymoo.optimize import minimize
    except Exception as exc:
        raise RuntimeError(
            "pymoo no esta disponible. Instala pymoo en el entorno Docker/configurado."
        ) from exc

    model_cfg, scenario, state0, encoding = _build_context(
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
    )
    lower = encoding.bounds[:, 0].astype(float)
    upper = encoding.bounds[:, 1].astype(float)
    pop_size = max(4, int(args.size_gen))
    total_min = max(1.0, float(args.sim_duration_min))
    execution_window_min = float(args.pymoo_execution_window_min)
    if execution_window_min <= 0.0:
        execution_window_min = float(args.decision_interval_min)
    execution_window_min = max(1.0, execution_window_min)
    plan_horizon_hours = max(execution_window_min / 60.0, float(args.horizon_hours))
    algorithm_key = _pymoo_algorithm_key(args.pymoo_algorithm)
    algorithm_label = _pymoo_algorithm_label(algorithm_key)
    seed = _seed_optimizer_random_state(getattr(args, "optimizer_random_seed", -1))
    base_seed = int(seed if seed is not None else args.seed)

    carry_state = copy.deepcopy(state0)
    transfer_pool: np.ndarray | None = None
    windows: List[Dict[str, Any]] = []
    elapsed_min = 0.0
    window_idx = 0

    while elapsed_min < total_min - 1e-9:
        run_window_min = min(execution_window_min, total_min - elapsed_min)
        window_seed = int((base_seed + 1009 * window_idx) % (2**32 - 1))
        rng = np.random.default_rng(window_seed)
        eval_problem = _build_evaluator(
            state_template=carry_state,
            scenario=scenario,
            encoding=encoding,
            config=model_cfg,
            horizon_hours=plan_horizon_hours,
            random_seed=window_seed,
        )

        class PymooStatefulProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=int(encoding.dimension), n_obj=3, xl=lower, xu=upper)

            def _evaluate(self, x, out, *eval_args, **eval_kwargs):
                objectives, _, _ = eval_problem.rollout(
                    x=np.asarray(x, dtype=float),
                    base_state=carry_state,
                )
                out["F"] = -np.asarray(objectives, dtype=float)

        class TransferSampling(Sampling):
            def _do(self, problem, n_samples, **kwargs):
                samples: List[np.ndarray] = []
                if transfer_pool is not None and transfer_pool.size:
                    pool = np.asarray(transfer_pool, dtype=float)
                    if pool.ndim == 1:
                        pool = pool.reshape(1, -1)
                    for row in pool[:n_samples]:
                        samples.append(np.clip(np.asarray(row, dtype=float), lower, upper))
                    while len(samples) < n_samples and pool.shape[0] > 0:
                        base = np.asarray(pool[int(rng.integers(0, pool.shape[0]))], dtype=float)
                        noise = rng.normal(0.0, 0.04 * np.maximum(upper - lower, 1e-9))
                        samples.append(np.clip(base + noise, lower, upper))
                while len(samples) < n_samples:
                    samples.append(rng.uniform(low=lower, high=upper, size=int(encoding.dimension)))
                return np.asarray(samples[:n_samples], dtype=float)

        algorithm = _build_pymoo_algorithm(
            algorithm_key,
            pop_size=pop_size,
            sampling=TransferSampling(),
        )

        eval_problem.reset_evaluation_stats()
        optimizer_wall_start = time.monotonic()
        result = minimize(
            PymooStatefulProblem(),
            algorithm,
            termination=("n_gen", max(1, int(args.max_iter))),
            seed=window_seed,
            verbose=bool(args.disp),
            save_history=False,
        )
        optimizer_wall_time_sec = float(time.monotonic() - optimizer_wall_start)
        optimizer_stats = _evaluation_stats(eval_problem)
        front_min = np.asarray(getattr(result, "F", np.empty((0, 3))), dtype=float)
        if front_min.ndim == 1 and front_min.size:
            front_min = front_min.reshape(1, -1)
        front_max = -front_min if front_min.size else np.empty((0, 3), dtype=float)
        pset = np.asarray(getattr(result, "X", np.empty((0, encoding.dimension))), dtype=float)
        if pset.ndim == 1 and pset.size:
            pset = pset.reshape(1, -1)
        before_selection = _evaluation_stats(eval_problem)
        selection_wall_start = time.monotonic()
        if pset.size == 0:
            pset = np.asarray([np.mean(encoding.bounds, axis=1)], dtype=float)
            front_max = np.asarray([eval_problem.objective_max(pset[0])], dtype=float)

        selected_index = _best_by_normalized_sum_index(front_max)
        if selected_index is None or selected_index >= pset.shape[0]:
            selected_index = 0
        x_selected = np.asarray(pset[selected_index], dtype=float)
        selected_objectives, selected_details, _ = eval_problem.rollout(
            x=x_selected,
            base_state=carry_state,
        )
        selection_wall_time_sec = float(time.monotonic() - selection_wall_start)
        selection_stats = _evaluation_stats_delta(_evaluation_stats(eval_problem), before_selection)

        exec_problem = _build_evaluator(
            state_template=carry_state,
            scenario=scenario,
            encoding=encoding,
            config=model_cfg,
            horizon_hours=float(run_window_min) / 60.0,
            random_seed=int(window_seed + 17),
        )
        exec_problem.reset_evaluation_stats()
        execution_wall_start = time.monotonic()
        exec_objectives, exec_details, carry_state = exec_problem.rollout(
            x=x_selected,
            base_state=carry_state,
        )
        execution_wall_time_sec = float(time.monotonic() - execution_wall_start)
        execution_stats = _evaluation_stats(exec_problem)
        transfer_pool = np.vstack([x_selected.reshape(1, -1), pset])
        if transfer_pool.shape[0] > pop_size:
            transfer_pool = transfer_pool[:pop_size]

        windows.append(
            {
                "window_id": f"w{window_idx:03d}",
                "start_min": float(elapsed_min),
                "execution_window_min": float(run_window_min),
                "planning_horizon_hours": float(plan_horizon_hours),
                "pareto_size": int(front_max.shape[0]),
                "selected_index": int(selected_index),
                "selected_objectives": _to_list(selected_objectives),
                "selected_details": {key: float(value) for key, value in selected_details.items()},
                "executed_objectives": _to_list(exec_objectives),
                "executed_details": {key: float(value) for key, value in exec_details.items()},
                "transfer_pool_size": int(transfer_pool.shape[0]),
                "optimizer_exec_time_sec": float(getattr(result, "exec_time", 0.0) or 0.0),
                "optimizer_wall_time_sec": float(optimizer_wall_time_sec),
                "optimizer_rollout_eval_count": _rollout_eval_count(optimizer_stats),
                "optimizer_rollout_runtime_sec": _rollout_runtime_sec(optimizer_stats),
                "selection_wall_time_sec": float(selection_wall_time_sec),
                "selection_rollout_eval_count": _rollout_eval_count(selection_stats),
                "selection_rollout_runtime_sec": _rollout_runtime_sec(selection_stats),
                "execution_wall_time_sec": float(execution_wall_time_sec),
                "execution_rollout_eval_count": _rollout_eval_count(execution_stats),
                "execution_rollout_runtime_sec": _rollout_runtime_sec(execution_stats),
            }
        )
        elapsed_min += float(run_window_min)
        window_idx += 1

    exec_objective_rows = np.asarray(
        [row["executed_objectives"] for row in windows], dtype=float
    ) if windows else np.empty((0, 3), dtype=float)
    mean_exec_objectives = (
        np.mean(exec_objective_rows, axis=0).tolist()
        if exec_objective_rows.size
        else []
    )

    optimizer_eval_total = sum(int(row.get("optimizer_rollout_eval_count", 0)) for row in windows)
    selection_eval_total = sum(int(row.get("selection_rollout_eval_count", 0)) for row in windows)
    execution_eval_total = sum(int(row.get("execution_rollout_eval_count", 0)) for row in windows)
    optimizer_wall_total = sum(float(row.get("optimizer_wall_time_sec", 0.0)) for row in windows)
    selection_wall_total = sum(float(row.get("selection_wall_time_sec", 0.0)) for row in windows)
    execution_wall_total = sum(float(row.get("execution_wall_time_sec", 0.0)) for row in windows)
    optimizer_rollout_runtime_total = sum(float(row.get("optimizer_rollout_runtime_sec", 0.0)) for row in windows)
    selection_rollout_runtime_total = sum(float(row.get("selection_rollout_runtime_sec", 0.0)) for row in windows)
    execution_rollout_runtime_total = sum(float(row.get("execution_rollout_runtime_sec", 0.0)) for row in windows)

    return {
        "mode": "pymoo-stateful",
        "meta": {
            "seed": int(args.seed),
            "optimizer_random_seed": None if seed is None else int(seed),
            "sim_duration_min": float(total_min),
            "execution_window_min": float(execution_window_min),
            "planning_horizon_hours": float(plan_horizon_hours),
            "decision_interval_min": int(args.decision_interval_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            **_scenario_meta(scenario=scenario, model_cfg=model_cfg, state0=state0),
            "encoding_dimension": int(encoding.dimension),
            "algorithm": algorithm_label,
            "pymoo_algorithm": algorithm_key,
            "size_gen": int(args.size_gen),
            "max_iter_per_window": int(args.max_iter),
            "window_count": int(len(windows)),
            "population_transfer": True,
            "optimizer_wall_time_sec": float(optimizer_wall_total),
            "optimizer_rollout_eval_count": int(optimizer_eval_total),
            "optimizer_rollout_runtime_sec": float(optimizer_rollout_runtime_total),
            "selection_wall_time_sec": float(selection_wall_total),
            "selection_rollout_eval_count": int(selection_eval_total),
            "selection_rollout_runtime_sec": float(selection_rollout_runtime_total),
            "execution_wall_time_sec": float(execution_wall_total),
            "execution_rollout_eval_count": int(execution_eval_total),
            "execution_rollout_runtime_sec": float(execution_rollout_runtime_total),
            "total_rollout_eval_count": int(optimizer_eval_total + selection_eval_total + execution_eval_total),
            "total_rollout_runtime_sec": float(
                optimizer_rollout_runtime_total
                + selection_rollout_runtime_total
                + execution_rollout_runtime_total
            ),
        },
        "windows": windows,
        "mean_executed_objectives": mean_exec_objectives,
        "last_executed_objectives": windows[-1]["executed_objectives"] if windows else [],
        "final_status": _state_metrics_payload(carry_state),
        "pymoo_reopt_count": int(len(windows)),
    }


def run_pymoo_adr_stateful(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.sampling import Sampling
        from pymoo.optimize import minimize
    except Exception as exc:
        raise RuntimeError(
            "pymoo no esta disponible. Instala pymoo en el entorno Docker/configurado."
        ) from exc

    model_cfg, scenario, state0, encoding = _build_context(
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
    )
    lower = encoding.bounds[:, 0].astype(float)
    upper = encoding.bounds[:, 1].astype(float)
    pop_size = max(4, int(args.size_gen))
    total_min = max(1.0, float(args.sim_duration_min))
    execution_window_min = float(args.pymoo_execution_window_min)
    if execution_window_min <= 0.0:
        execution_window_min = float(args.decision_interval_min)
    execution_window_min = max(1.0, execution_window_min)
    plan_horizon_hours = max(execution_window_min / 60.0, float(args.horizon_hours))
    algorithm_key = _pymoo_algorithm_key(args.pymoo_algorithm)
    algorithm_label = _pymoo_algorithm_label(algorithm_key)
    seed = _seed_optimizer_random_state(getattr(args, "optimizer_random_seed", -1))
    base_seed = int(seed if seed is not None else args.seed)
    min_update_interval = max(0, int(args.pymoo_adr_min_update_interval))
    max_update_interval = max(1, int(args.pymoo_adr_max_update_interval))
    service_floor = float(args.pymoo_adr_service_floor)
    reject_threshold = float(args.pymoo_adr_reject_threshold)

    carry_state = copy.deepcopy(state0)
    transfer_pool: np.ndarray | None = None
    x_selected: np.ndarray | None = None
    last_selected_objectives: list[float] = []
    last_selected_details: Dict[str, float] = {}
    last_exec_details: Dict[str, float] = {}
    windows: List[Dict[str, Any]] = []
    elapsed_min = 0.0
    window_idx = 0
    update_count = 0
    reuse_count = 0
    windows_since_update = max_update_interval

    while elapsed_min < total_min - 1e-9:
        run_window_min = min(execution_window_min, total_min - elapsed_min)
        window_seed = int((base_seed + 1009 * window_idx) % (2**32 - 1))
        rng = np.random.default_rng(window_seed)
        recent_service = float(last_exec_details.get("service_rate", 1.0))
        recent_rejected = float(last_exec_details.get("rejected_requests", 0.0))
        trigger_reason = "reuse"
        should_update = False
        if x_selected is None:
            should_update = True
            trigger_reason = "bootstrap"
        elif windows_since_update < min_update_interval:
            should_update = False
            trigger_reason = "min_interval_hold"
        elif windows_since_update >= max_update_interval:
            should_update = True
            trigger_reason = "max_interval"
        elif recent_service < service_floor:
            should_update = True
            trigger_reason = "service_floor"
        elif recent_rejected >= reject_threshold:
            should_update = True
            trigger_reason = "recent_rejections"

        optimizer_exec_time = 0.0
        optimizer_wall_time_sec = 0.0
        optimizer_stats: Dict[str, float] = {}
        selection_wall_time_sec = 0.0
        selection_stats: Dict[str, float] = {}
        pareto_size = 0
        selected_index = None
        selected_details: Dict[str, float] = dict(last_selected_details)
        selected_objectives = list(last_selected_objectives)

        if should_update:
            eval_problem = _build_evaluator(
                state_template=carry_state,
                scenario=scenario,
                encoding=encoding,
                config=model_cfg,
                horizon_hours=plan_horizon_hours,
                random_seed=window_seed,
            )

            class PymooADRStatefulProblem(ElementwiseProblem):
                def __init__(self):
                    super().__init__(n_var=int(encoding.dimension), n_obj=3, xl=lower, xu=upper)

                def _evaluate(self, x, out, *eval_args, **eval_kwargs):
                    objectives, _, _ = eval_problem.rollout(
                        x=np.asarray(x, dtype=float),
                        base_state=carry_state,
                    )
                    out["F"] = -np.asarray(objectives, dtype=float)

            class TransferSampling(Sampling):
                def _do(self, problem, n_samples, **kwargs):
                    samples: List[np.ndarray] = []
                    if transfer_pool is not None and transfer_pool.size:
                        pool = np.asarray(transfer_pool, dtype=float)
                        if pool.ndim == 1:
                            pool = pool.reshape(1, -1)
                        for row in pool[:n_samples]:
                            samples.append(np.clip(np.asarray(row, dtype=float), lower, upper))
                        while len(samples) < n_samples and pool.shape[0] > 0:
                            base = np.asarray(pool[int(rng.integers(0, pool.shape[0]))], dtype=float)
                            noise = rng.normal(0.0, 0.04 * np.maximum(upper - lower, 1e-9))
                            samples.append(np.clip(base + noise, lower, upper))
                    while len(samples) < n_samples:
                        samples.append(rng.uniform(low=lower, high=upper, size=int(encoding.dimension)))
                    return np.asarray(samples[:n_samples], dtype=float)

            algorithm = _build_pymoo_algorithm(
                algorithm_key,
                pop_size=pop_size,
                sampling=TransferSampling(),
            )

            eval_problem.reset_evaluation_stats()
            optimizer_wall_start = time.monotonic()
            result = minimize(
                PymooADRStatefulProblem(),
                algorithm,
                termination=("n_gen", max(1, int(args.max_iter))),
                seed=window_seed,
                verbose=bool(args.disp),
                save_history=False,
            )
            optimizer_wall_time_sec = float(time.monotonic() - optimizer_wall_start)
            optimizer_stats = _evaluation_stats(eval_problem)
            front_min = np.asarray(getattr(result, "F", np.empty((0, 3))), dtype=float)
            if front_min.ndim == 1 and front_min.size:
                front_min = front_min.reshape(1, -1)
            front_max = -front_min if front_min.size else np.empty((0, 3), dtype=float)
            pset = np.asarray(getattr(result, "X", np.empty((0, encoding.dimension))), dtype=float)
            if pset.ndim == 1 and pset.size:
                pset = pset.reshape(1, -1)
            before_selection = _evaluation_stats(eval_problem)
            selection_wall_start = time.monotonic()
            if pset.size == 0:
                pset = np.asarray([np.mean(encoding.bounds, axis=1)], dtype=float)
                front_max = np.asarray([eval_problem.objective_max(pset[0])], dtype=float)

            selected_index = _best_by_normalized_sum_index(front_max)
            if selected_index is None or selected_index >= pset.shape[0]:
                selected_index = 0
            x_selected = np.asarray(pset[selected_index], dtype=float)
            selected_objectives_arr, selected_details_raw, _ = eval_problem.rollout(
                x=x_selected,
                base_state=carry_state,
            )
            selection_wall_time_sec = float(time.monotonic() - selection_wall_start)
            selection_stats = _evaluation_stats_delta(_evaluation_stats(eval_problem), before_selection)
            selected_objectives = _to_list(selected_objectives_arr)
            selected_details = {key: float(value) for key, value in selected_details_raw.items()}
            last_selected_objectives = list(selected_objectives)
            last_selected_details = dict(selected_details)
            transfer_pool = np.vstack([x_selected.reshape(1, -1), pset])
            if transfer_pool.shape[0] > pop_size:
                transfer_pool = transfer_pool[:pop_size]
            optimizer_exec_time = float(getattr(result, "exec_time", 0.0) or 0.0)
            pareto_size = int(front_max.shape[0])
            update_count += 1
            windows_since_update = 0
        else:
            reuse_count += 1
            windows_since_update += 1

        if x_selected is None:
            x_selected = np.asarray(np.mean(encoding.bounds, axis=1), dtype=float)

        exec_problem = _build_evaluator(
            state_template=carry_state,
            scenario=scenario,
            encoding=encoding,
            config=model_cfg,
            horizon_hours=float(run_window_min) / 60.0,
            random_seed=int(window_seed + 17),
        )
        exec_problem.reset_evaluation_stats()
        execution_wall_start = time.monotonic()
        exec_objectives, exec_details, carry_state = exec_problem.rollout(
            x=x_selected,
            base_state=carry_state,
        )
        execution_wall_time_sec = float(time.monotonic() - execution_wall_start)
        execution_stats = _evaluation_stats(exec_problem)
        last_exec_details = {key: float(value) for key, value in exec_details.items()}

        windows.append(
            {
                "window_id": f"w{window_idx:03d}",
                "start_min": float(elapsed_min),
                "execution_window_min": float(run_window_min),
                "planning_horizon_hours": float(plan_horizon_hours),
                "action": "update" if should_update else "reuse",
                "trigger_reason": str(trigger_reason),
                "windows_since_update_after": int(windows_since_update),
                "recent_service_before": float(recent_service),
                "recent_rejected_before": float(recent_rejected),
                "pareto_size": int(pareto_size),
                "selected_index": None if selected_index is None else int(selected_index),
                "selected_objectives": selected_objectives,
                "selected_details": selected_details,
                "executed_objectives": _to_list(exec_objectives),
                "executed_details": {key: float(value) for key, value in exec_details.items()},
                "transfer_pool_size": 0 if transfer_pool is None else int(transfer_pool.shape[0]),
                "optimizer_exec_time_sec": float(optimizer_exec_time),
                "optimizer_wall_time_sec": float(optimizer_wall_time_sec),
                "optimizer_rollout_eval_count": _rollout_eval_count(optimizer_stats),
                "optimizer_rollout_runtime_sec": _rollout_runtime_sec(optimizer_stats),
                "selection_wall_time_sec": float(selection_wall_time_sec),
                "selection_rollout_eval_count": _rollout_eval_count(selection_stats),
                "selection_rollout_runtime_sec": _rollout_runtime_sec(selection_stats),
                "execution_wall_time_sec": float(execution_wall_time_sec),
                "execution_rollout_eval_count": _rollout_eval_count(execution_stats),
                "execution_rollout_runtime_sec": _rollout_runtime_sec(execution_stats),
            }
        )
        elapsed_min += float(run_window_min)
        window_idx += 1

    exec_objective_rows = np.asarray(
        [row["executed_objectives"] for row in windows], dtype=float
    ) if windows else np.empty((0, 3), dtype=float)
    mean_exec_objectives = (
        np.mean(exec_objective_rows, axis=0).tolist()
        if exec_objective_rows.size
        else []
    )

    optimizer_eval_total = sum(int(row.get("optimizer_rollout_eval_count", 0)) for row in windows)
    selection_eval_total = sum(int(row.get("selection_rollout_eval_count", 0)) for row in windows)
    execution_eval_total = sum(int(row.get("execution_rollout_eval_count", 0)) for row in windows)
    optimizer_wall_total = sum(float(row.get("optimizer_wall_time_sec", 0.0)) for row in windows)
    selection_wall_total = sum(float(row.get("selection_wall_time_sec", 0.0)) for row in windows)
    execution_wall_total = sum(float(row.get("execution_wall_time_sec", 0.0)) for row in windows)
    optimizer_rollout_runtime_total = sum(float(row.get("optimizer_rollout_runtime_sec", 0.0)) for row in windows)
    selection_rollout_runtime_total = sum(float(row.get("selection_rollout_runtime_sec", 0.0)) for row in windows)
    execution_rollout_runtime_total = sum(float(row.get("execution_rollout_runtime_sec", 0.0)) for row in windows)

    return {
        "mode": "pymoo-adr-stateful",
        "meta": {
            "seed": int(args.seed),
            "optimizer_random_seed": None if seed is None else int(seed),
            "sim_duration_min": float(total_min),
            "execution_window_min": float(execution_window_min),
            "planning_horizon_hours": float(plan_horizon_hours),
            "decision_interval_min": int(args.decision_interval_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            **_scenario_meta(scenario=scenario, model_cfg=model_cfg, state0=state0),
            "encoding_dimension": int(encoding.dimension),
            "algorithm": algorithm_label,
            "pymoo_algorithm": algorithm_key,
            "size_gen": int(args.size_gen),
            "max_iter_per_update": int(args.max_iter),
            "window_count": int(len(windows)),
            "population_transfer": True,
            "adr_min_update_interval": int(min_update_interval),
            "adr_max_update_interval": int(max_update_interval),
            "adr_service_floor": float(service_floor),
            "adr_reject_threshold": float(reject_threshold),
            "optimizer_wall_time_sec": float(optimizer_wall_total),
            "optimizer_rollout_eval_count": int(optimizer_eval_total),
            "optimizer_rollout_runtime_sec": float(optimizer_rollout_runtime_total),
            "selection_wall_time_sec": float(selection_wall_total),
            "selection_rollout_eval_count": int(selection_eval_total),
            "selection_rollout_runtime_sec": float(selection_rollout_runtime_total),
            "execution_wall_time_sec": float(execution_wall_total),
            "execution_rollout_eval_count": int(execution_eval_total),
            "execution_rollout_runtime_sec": float(execution_rollout_runtime_total),
            "total_rollout_eval_count": int(optimizer_eval_total + selection_eval_total + execution_eval_total),
            "total_rollout_runtime_sec": float(
                optimizer_rollout_runtime_total
                + selection_rollout_runtime_total
                + execution_rollout_runtime_total
            ),
        },
        "windows": windows,
        "mean_executed_objectives": mean_exec_objectives,
        "last_executed_objectives": windows[-1]["executed_objectives"] if windows else [],
        "final_status": _state_metrics_payload(carry_state),
        "pymoo_reopt_count": int(update_count),
        "pymoo_reuse_count": int(reuse_count),
    }


def run_windowed_static_eda(args: argparse.Namespace) -> Dict[str, Any]:
    from . import SPEDARunConfig, SPEDAStaticRunner, WindowedDynamicSPEDA

    model_cfg, scenario, state0, encoding = _build_context(
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
    )
    windows = _build_windows(
        zone_ids=list(state0.zones.keys()),
        base_demand_scale=float(args.base_demand_scale),
        max_iter=int(args.max_iter),
        window_pack=str(args.window_pack),
        key="demand_multiplier",
    )

    def problem_builder(window: Dict[str, Any], carry_state: Optional[Any]) -> FreeFloatingRobustDMOOP:
        template = copy.deepcopy(carry_state) if carry_state is not None else copy.deepcopy(state0)
        return _build_evaluator(
            state_template=template,
            scenario=scenario,
            encoding=encoding,
            config=model_cfg,
            horizon_hours=int(args.horizon_hours),
            random_seed=int(args.seed),
        )

    candidate_cfgs = _candidate_eda_configs(
        requested_family=str(args.optimizer_family),
        requested_size_gen=int(args.size_gen),
        requested_w_noise=float(args.w_noise),
        fallback_families=[str(args.optimizer_family), "KDEBN", "SPBN"],
        auto_fallback=bool(args.auto_fallback),
    )
    runs = None
    used_family = None
    used_size_gen = None
    used_w_noise = None
    fallback_errors: List[str] = []
    for family, size_gen, w_noise in candidate_cfgs:
        try:
            cfg = SPEDARunConfig(
                size_gen=int(size_gen),
                max_iter=int(args.max_iter),
                dead_iter=int(args.dead_iter),
                alpha=float(args.alpha),
                archive_generations=float(args.archive_generations),
                w_noise=float(w_noise),
                parallelize=bool(args.parallelize),
                disp=bool(args.disp),
                optimizer_family=str(family),
            )
            static_runner = SPEDAStaticRunner(encoding=encoding, config=cfg)
            dynamic_runner = WindowedDynamicSPEDA(static_runner=static_runner)
            runs = dynamic_runner.optimize_windows(problem_builder=problem_builder, windows=windows)
            used_family = str(family)
            used_size_gen = int(size_gen)
            used_w_noise = float(w_noise)
            break
        except Exception as exc:
            fallback_errors.append(f"{family}/N={size_gen}/noise={w_noise:.2f}: {type(exc).__name__}: {exc}")
    if runs is None:
        raise RuntimeError("EDA por ventanas falló en todas las configuraciones: " + " | ".join(fallback_errors))

    out_windows: List[Dict[str, Any]] = []
    for wr in runs:
        front = np.asarray(wr.raw_result.get("pareto_front_max", np.empty((0, 3))), dtype=float)
        pset = np.asarray(wr.raw_result.get("pareto_set", np.empty((0, encoding.dimension))), dtype=float)
        out_windows.append(
            {
                "window_id": wr.window_id,
                "pareto_size": int(front.shape[0]),
                "pareto_front_max": _to_list(front),
                "pareto_set": _to_list(pset),
                "selected_index": None if wr.selected_index is None else int(wr.selected_index),
                "selected_objectives": wr.selected_objectives,
                "selected_details": wr.selected_details,
            }
        )

    return {
        "mode": "windowed-eda",
        "meta": {
            "seed": int(args.seed),
            "horizon_hours_per_window": int(args.horizon_hours),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            **_scenario_meta(scenario=scenario, model_cfg=model_cfg, state0=state0),
            "encoding_dimension": int(encoding.dimension),
            "size_gen_requested": int(args.size_gen),
            "size_gen_used": int(used_size_gen or args.size_gen),
            "max_iter": int(args.max_iter),
            "optimizer_family_requested": str(args.optimizer_family),
            "optimizer_family_used": str(used_family or args.optimizer_family),
            "window_pack": str(args.window_pack),
            "base_demand_scale": float(args.base_demand_scale),
            "w_noise_requested": float(args.w_noise),
            "w_noise_used": float(used_w_noise or args.w_noise),
            "auto_fallback": bool(args.auto_fallback),
            "fallback_errors": fallback_errors,
        },
        "windows": out_windows,
    }


def run_stateful_dynamic_eda(args: argparse.Namespace) -> Dict[str, Any]:
    from . import DynamicSPEDARunConfig, StatefulDynamicSPEDA

    model_cfg, scenario, state0, encoding = _build_context(
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
    )
    coarse_horizon_hours = float(args.horizon_hours)
    raw_windows = _build_windows(
        zone_ids=list(state0.zones.keys()),
        base_demand_scale=float(args.base_demand_scale),
        max_iter=int(args.max_iter),
        window_pack=str(args.window_pack),
        key="demand_shock",
    )
    windows = _expand_stateful_windows(
        windows=raw_windows,
        coarse_horizon_hours=coarse_horizon_hours,
        observation_interval_min=float(args.dynamic_observation_interval_min),
    )
    resolved_dynamic_profile = _resolve_dynamic_online_profile(args)

    def problem_builder(window: Dict[str, Any], carry_state: Optional[Any]) -> FreeFloatingRobustDMOOP:
        template = copy.deepcopy(carry_state) if carry_state is not None else copy.deepcopy(state0)
        return _build_evaluator(
            state_template=template,
            scenario=scenario,
            encoding=encoding,
            config=model_cfg,
            horizon_hours=float(window.get("horizon_hours", coarse_horizon_hours)),
            random_seed=int(args.seed),
        )

    candidate_cfgs = _candidate_eda_configs(
        requested_family=str(args.pm_family),
        requested_size_gen=int(args.size_gen),
        requested_w_noise=float(args.w_noise),
        fallback_families=[str(args.pm_family), "GBN", "KDEBN", "SPBN"],
        auto_fallback=bool(args.auto_fallback),
    )
    runs = None
    used_family = None
    used_size_gen = None
    used_w_noise = None
    fallback_errors: List[str] = []
    for family, size_gen, w_noise in candidate_cfgs:
        try:
            dyn_cfg = DynamicSPEDARunConfig(
                size_gen=int(size_gen),
                max_iter=int(args.max_iter),
                dead_iter=int(args.dead_iter),
                alpha=float(args.alpha),
                elite_factor=float(args.elite_factor),
                archive_generations=float(args.archive_generations),
                w_noise=float(w_noise),
                parallelize=bool(args.parallelize),
                disp=bool(args.disp),
                pm_family=str(family),
                num_folds=int(args.num_folds),
                archive_retention_after_change=float(args.archive_retention_after_change),
                with_reaction=bool(args.with_reaction),
                change_detector_mode=str(args.dynamic_change_detector_mode),
                reopt_trigger_mode=str(args.dynamic_reopt_trigger),
                detector={
                    "n_sentinels": int(args.n_sentinels),
                    "drift_alpha": float(args.drift_alpha),
                    "lambda_ewma": float(args.lambda_ewma),
                    "sentinel_refresh": int(args.sentinel_refresh),
                    "k_consecutive": int(args.k_consecutive),
                    "split_drift_signals": bool(resolved_dynamic_profile.get("split_drift_signals", False)),
                    "adaptive_lambda_enabled": bool(resolved_dynamic_profile.get("adaptive_lambda_enabled", False)),
                    "adaptive_lambda_cov_enabled": bool(
                        resolved_dynamic_profile.get("adaptive_lambda_cov_enabled", False)
                    ),
                    "adaptive_cooldown_enabled": bool(
                        resolved_dynamic_profile.get("adaptive_cooldown_enabled", False)
                    ),
                },
                reaction={
                    "react_strength": float(resolved_dynamic_profile["react_strength"]),
                    "immigrant_ratio": float(resolved_dynamic_profile["immigrant_ratio"]),
                    "react_strength_mild": float(resolved_dynamic_profile["react_strength_mild"]),
                    "immigrant_ratio_mild": float(resolved_dynamic_profile["immigrant_ratio_mild"]),
                    "reaction_min_drift_ratio": float(
                        resolved_dynamic_profile.get("reaction_min_drift_ratio", 1.10)
                    ),
                    "reaction_min_interval": int(
                        resolved_dynamic_profile.get("reaction_min_interval", 3)
                    ),
                    "adaptive_policy_enabled": bool(resolved_dynamic_profile["adaptive_policy_enabled"]),
                    "reaction_quality_gate_enabled": bool(
                        resolved_dynamic_profile["reaction_quality_gate_enabled"]
                    ),
                    "reaction_quality_gate_mo_enabled": bool(
                        resolved_dynamic_profile.get("reaction_quality_gate_mo_enabled", False)
                    ),
                    "diversity_block_enabled": bool(
                        resolved_dynamic_profile.get("diversity_block_enabled", True)
                    ),
                    "predictive_block_enabled": bool(
                        resolved_dynamic_profile.get("predictive_block_enabled", False)
                    ),
                    "continuity_block_enabled": bool(
                        resolved_dynamic_profile.get("continuity_block_enabled", True)
                    ),
                    "adr_front_ls_enabled": bool(
                        resolved_dynamic_profile.get("adr_front_ls_enabled", False)
                    ),
                    "adr_front_ls_context_gate_enabled": bool(
                        resolved_dynamic_profile.get("adr_front_ls_context_gate_enabled", False)
                    ),
                    "change_type_enabled": bool(
                        resolved_dynamic_profile.get("change_type_enabled", False)
                    ),
                    "adr_enabled": bool(resolved_dynamic_profile.get("adr_enabled", False)),
                },
                local_search={
                    "local_search_enabled": bool(
                        resolved_dynamic_profile.get("local_search_enabled", args.local_search_enabled)
                    ),
                    "local_search_neighbors": int(
                        resolved_dynamic_profile.get("local_search_neighbors", args.local_search_neighbors)
                    ),
                    "local_search_front_ratio": float(
                        resolved_dynamic_profile.get("local_search_front_ratio", args.local_search_front_ratio)
                    ),
                    "local_search_step": float(
                        resolved_dynamic_profile.get("local_search_step", args.local_search_step)
                    ),
                    "local_search_max_points": int(
                        resolved_dynamic_profile.get("local_search_max_points", args.local_search_max_points)
                    ),
                },
                problem_detector={
                    "threshold_mild": float(args.dynamic_threshold_mild),
                    "threshold_severe": float(args.dynamic_threshold_severe),
                    "reference_ewma_alpha": float(args.dynamic_regime_ewma_alpha),
                    "min_reopt_interval_min": float(args.dynamic_min_reopt_interval_min),
                    "max_reopt_interval_min": float(args.dynamic_max_reopt_interval_min),
                    "hard_low_soc_share": float(args.dynamic_hard_low_soc_share),
                    "hard_charger_pressure": float(args.dynamic_hard_charger_pressure),
                    "hard_reservation_pressure": float(args.dynamic_hard_reservation_pressure),
                    "hard_coverage_gap": float(args.dynamic_hard_coverage_gap),
                },
            )
            dynamic_runner = StatefulDynamicSPEDA(encoding=encoding, config=dyn_cfg)
            runs = dynamic_runner.optimize_windows(problem_builder=problem_builder, windows=windows)
            used_family = str(family)
            used_size_gen = int(size_gen)
            used_w_noise = float(w_noise)
            break
        except Exception as exc:
            fallback_errors.append(f"{family}/N={size_gen}/noise={w_noise:.2f}: {type(exc).__name__}: {exc}")
    if runs is None:
        raise RuntimeError("DynamicEDA stateful falló en todas las configuraciones: " + " | ".join(fallback_errors))

    out_windows: List[Dict[str, Any]] = []
    for wr in runs:
        front = np.asarray(wr.raw_result.get("pareto_front_max", np.empty((0, 3))), dtype=float)
        pset = np.asarray(wr.raw_result.get("pareto_set", np.empty((0, encoding.dimension))), dtype=float)
        out_windows.append(
            {
                "window_id": wr.window_id,
                "parent_window_id": wr.raw_result.get("window", {}).get("parent_window_id"),
                "subwindow_index": wr.raw_result.get("window", {}).get("subwindow_index"),
                "window_horizon_hours": wr.raw_result.get("window", {}).get("horizon_hours"),
                "pareto_size": int(front.shape[0]),
                "pareto_front_max": _to_list(front),
                "pareto_set": _to_list(pset),
                "selected_index": None if wr.selected_index is None else int(wr.selected_index),
                "selected_objectives": wr.selected_objectives,
                "selected_details": wr.selected_details,
                "step_info_last": wr.raw_result.get("step_info_last", {}),
                "regime": wr.raw_result.get("regime", {}),
                "reference_regime": wr.raw_result.get("reference_regime", {}),
                "change_info": wr.raw_result.get("change_info", {}),
                "trigger_info": wr.raw_result.get("trigger_info", {}),
                "adaptive_guard_applied": bool(
                    wr.raw_result.get("adaptive_guard_applied", False)
                ),
                "adaptive_guard_result": wr.raw_result.get("adaptive_guard_result"),
                "reoptimized": bool(wr.raw_result.get("reoptimized", True)),
            }
        )

    return {
        "mode": "stateful-eda",
        "meta": {
            "seed": int(args.seed),
            "coarse_horizon_hours": float(coarse_horizon_hours),
            "dynamic_observation_interval_min": float(args.dynamic_observation_interval_min),
            "observation_horizon_hours": float(windows[0].get("horizon_hours", coarse_horizon_hours)) if windows else float(coarse_horizon_hours),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            **_scenario_meta(scenario=scenario, model_cfg=model_cfg, state0=state0),
            "encoding_dimension": int(encoding.dimension),
            "size_gen_requested": int(args.size_gen),
            "size_gen_used": int(used_size_gen or args.size_gen),
            "max_iter_per_window": int(args.max_iter),
            "window_pack": str(args.window_pack),
            "base_demand_scale": float(args.base_demand_scale),
            "pm_family_requested": str(args.pm_family),
            "pm_family_used": str(used_family or args.pm_family),
            "with_reaction": bool(args.with_reaction),
            "change_detector_mode": str(args.dynamic_change_detector_mode),
            "reopt_trigger_mode": str(args.dynamic_reopt_trigger),
            "dynamic_threshold_mild": float(args.dynamic_threshold_mild),
            "dynamic_threshold_severe": float(args.dynamic_threshold_severe),
            "dynamic_regime_ewma_alpha": float(args.dynamic_regime_ewma_alpha),
            "dynamic_min_reopt_interval_min": float(args.dynamic_min_reopt_interval_min),
            "dynamic_max_reopt_interval_min": float(args.dynamic_max_reopt_interval_min),
            "dynamic_hard_low_soc_share": float(args.dynamic_hard_low_soc_share),
            "dynamic_hard_charger_pressure": float(args.dynamic_hard_charger_pressure),
            "dynamic_hard_reservation_pressure": float(args.dynamic_hard_reservation_pressure),
            "dynamic_hard_coverage_gap": float(args.dynamic_hard_coverage_gap),
            "w_noise_requested": float(args.w_noise),
            "w_noise_used": float(used_w_noise or args.w_noise),
            "auto_fallback": bool(args.auto_fallback),
            "fallback_errors": fallback_errors,
            "coarse_window_count": int(len(raw_windows)),
            "observation_window_count": int(len(windows)),
        },
        "windows": out_windows,
    }


def _derive_online_artifact_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    if str(args.output_json).strip():
        base = Path(str(args.output_json))
        stem = base.with_suffix("")
        audit_prefix = str(Path(str(stem) + "_audit"))
        fronts_json = str(Path(str(stem) + "_fronts.json"))
        events_jsonl = str(Path(str(stem) + "_events.jsonl"))
        return audit_prefix, fronts_json, events_jsonl

    base = Path("/tmp/ff_online_eda")
    audit_prefix = str(base / "audit")
    fronts_json = str(base / "fronts.json")
    events_jsonl = str(base / "events.jsonl")
    return audit_prefix, fronts_json, events_jsonl


def _resolve_online_reopt_profile(args: argparse.Namespace) -> Dict[str, Any]:
    profile = str(getattr(args, "online_reopt_profile", "full")).strip()
    aliases = {
        "archive-guided": "archive_guided",
        "incumbent-seed": "incumbent_seed",
        "transfer-only": "transfer_only",
    }
    profile = aliases.get(profile, profile)
    valid = {
        "custom",
        "naive",
        "transfer_only",
        "incumbent_seed",
        "archive",
        "archive_guided",
        "full",
    }
    if profile not in valid:
        supported = ", ".join(sorted(valid))
        raise ValueError(f"--online-reopt-profile='{profile}' no soportado. Usa: {supported}")

    resolved = {
        "profile": profile,
        "policy_smooth_lambda": float(args.policy_smooth_lambda),
        "policy_smooth_metric": str(args.policy_smooth_metric),
        "eda_warm_start_enabled": bool(args.eda_warm_start_enabled),
        "eda_temporal_archive_enabled": bool(args.eda_temporal_archive_enabled),
        "eda_temporal_archive_max_size": int(args.eda_temporal_archive_max_size),
        "eda_temporal_archive_keep_per_step": int(args.eda_temporal_archive_keep_per_step),
        "eda_incumbent_gate_enabled": bool(args.eda_incumbent_gate_enabled),
        "eda_incumbent_min_improvement": float(args.eda_incumbent_min_improvement),
        "eda_incumbent_neighbor_ratio": float(args.eda_incumbent_neighbor_ratio),
        "eda_incumbent_noise_scale": float(args.eda_incumbent_noise_scale),
        "eda_archive_guided_ratio": float(args.eda_archive_guided_ratio),
        "eda_archive_guided_noise_scale": float(args.eda_archive_guided_noise_scale),
    }

    if profile == "custom":
        return resolved

    profile_overrides: Dict[str, Dict[str, Any]] = {
        "naive": {
            "policy_smooth_lambda": 0.0,
            "eda_warm_start_enabled": False,
            "eda_temporal_archive_enabled": False,
            "eda_incumbent_gate_enabled": False,
            "eda_incumbent_min_improvement": 0.01,
            "eda_incumbent_neighbor_ratio": 0.0,
            "eda_incumbent_noise_scale": 0.04,
            "eda_archive_guided_ratio": 0.0,
            "eda_archive_guided_noise_scale": 0.05,
        },
        "transfer_only": {
            "policy_smooth_lambda": 0.0,
            "eda_warm_start_enabled": True,
            "eda_temporal_archive_enabled": False,
            "eda_incumbent_gate_enabled": False,
            "eda_incumbent_min_improvement": 0.01,
            "eda_incumbent_neighbor_ratio": 0.0,
            "eda_incumbent_noise_scale": 0.04,
            "eda_archive_guided_ratio": 0.0,
            "eda_archive_guided_noise_scale": 0.05,
        },
        "incumbent_seed": {
            "policy_smooth_lambda": 0.0,
            "eda_warm_start_enabled": True,
            "eda_temporal_archive_enabled": False,
            "eda_incumbent_gate_enabled": False,
            "eda_incumbent_min_improvement": 0.01,
            "eda_incumbent_neighbor_ratio": 0.25,
            "eda_incumbent_noise_scale": 0.04,
            "eda_archive_guided_ratio": 0.0,
            "eda_archive_guided_noise_scale": 0.05,
        },
        "archive": {
            "policy_smooth_lambda": 0.0,
            "eda_warm_start_enabled": True,
            "eda_temporal_archive_enabled": True,
            "eda_temporal_archive_max_size": 96,
            "eda_temporal_archive_keep_per_step": 12,
            "eda_incumbent_gate_enabled": False,
            "eda_incumbent_min_improvement": 0.01,
            "eda_incumbent_neighbor_ratio": 0.25,
            "eda_incumbent_noise_scale": 0.04,
            "eda_archive_guided_ratio": 0.0,
            "eda_archive_guided_noise_scale": 0.05,
        },
        "archive_guided": {
            "policy_smooth_lambda": 0.0,
            "eda_warm_start_enabled": True,
            "eda_temporal_archive_enabled": True,
            "eda_temporal_archive_max_size": 96,
            "eda_temporal_archive_keep_per_step": 12,
            "eda_incumbent_gate_enabled": False,
            "eda_incumbent_min_improvement": 0.01,
            "eda_incumbent_neighbor_ratio": 0.25,
            "eda_incumbent_noise_scale": 0.04,
            "eda_archive_guided_ratio": 0.15,
            "eda_archive_guided_noise_scale": 0.05,
        },
        "full": {
            "policy_smooth_lambda": 0.0,
            "eda_warm_start_enabled": True,
            "eda_temporal_archive_enabled": True,
            "eda_temporal_archive_max_size": 96,
            "eda_temporal_archive_keep_per_step": 12,
            "eda_incumbent_gate_enabled": True,
            "eda_incumbent_min_improvement": 0.01,
            "eda_incumbent_neighbor_ratio": 0.25,
            "eda_incumbent_noise_scale": 0.04,
            "eda_archive_guided_ratio": 0.15,
            "eda_archive_guided_noise_scale": 0.05,
        },
    }
    resolved.update(profile_overrides[profile])
    return resolved


def _resolve_dynamic_online_profile(args: argparse.Namespace) -> Dict[str, Any]:
    profile = str(getattr(args, "dynamic_online_profile", "custom")).strip()
    aliases = {
        "dynamic-fast": "fast_event",
        "dynamic-fast-event": "fast_event",
        "dynamic-balanced": "balanced_event",
        "dynamic-default": "default_event",
        "dynamic-fast-strong": "fast_strong_react",
        "adr-eda-v2.1": "adr_eda_v2_1",
        "adr_eda_v2.1": "adr_eda_v2_1",
        "adr-v2.1": "adr_eda_v2_1",
        "adr-eda-v2.1-carsharing": "adr_eda_v2_1_carsharing",
        "adr_eda_v2_1_carsharing": "adr_eda_v2_1_carsharing",
        "adr-v2.1-carsharing": "adr_eda_v2_1_carsharing",
        "adr-eda-v2.2-carsharing": "adr_eda_v2_2_carsharing",
        "adr_eda_v2_2_carsharing": "adr_eda_v2_2_carsharing",
        "adr-v2.2-carsharing": "adr_eda_v2_2_carsharing",
        "adr-eda-adaptive-weights": "adr_eda_adaptive_weights",
        "adr_eda_adaptive_weights": "adr_eda_adaptive_weights",
        "adr-adaptive-weights": "adr_eda_adaptive_weights",
        "adr-eda-adaptive-diagnostics-only": "adr_eda_adaptive_diagnostics_only",
        "adr_eda_adaptive_diagnostics_only": "adr_eda_adaptive_diagnostics_only",
        "adr-adaptive-diagnostics-only": "adr_eda_adaptive_diagnostics_only",
        "adr-eda-adaptive-weights-activation-probe": "adr_eda_adaptive_weights_activation_probe",
        "adr_eda_adaptive_weights_activation_probe": "adr_eda_adaptive_weights_activation_probe",
        "adr-adaptive-weights-activation-probe": "adr_eda_adaptive_weights_activation_probe",
        "adr-eda-adaptive-weights-guarded": "adr_eda_adaptive_weights_guarded",
        "adr_eda_adaptive_weights_guarded": "adr_eda_adaptive_weights_guarded",
        "adr-adaptive-weights-guarded": "adr_eda_adaptive_weights_guarded",
        "adr-eda-adaptive-weights-selector-v2": "adr_eda_adaptive_weights_selector_v2",
        "adr_eda_adaptive_weights_selector_v2": "adr_eda_adaptive_weights_selector_v2",
        "adr-adaptive-weights-selector-v2": "adr_eda_adaptive_weights_selector_v2",
        "adr-eda-contextual-weights": "adr_eda_contextual_weights",
        "adr_eda_contextual_weights": "adr_eda_contextual_weights",
        "adr-contextual-weights": "adr_eda_contextual_weights",
        "adr-eda-fixed-weights-activation-probe": "adr_eda_fixed_weights_activation_probe",
        "adr_eda_fixed_weights_activation_probe": "adr_eda_fixed_weights_activation_probe",
        "adr-fixed-weights-activation-probe": "adr_eda_fixed_weights_activation_probe",
        "selective-simple-carsharing": "selective_simple_carsharing",
        "selective_simple_carsharing": "selective_simple_carsharing",
        "event-simple-carsharing": "selective_simple_carsharing",
        "adr-eda-reservation-aware-v1": "adr_eda_reservation_aware_v1",
        "adr_eda_reservation_aware_v1": "adr_eda_reservation_aware_v1",
        "adr-reservation-aware-v1": "adr_eda_reservation_aware_v1",
        "adr-eda-reservation-aware-v2": "adr_eda_reservation_aware_v2",
        "adr_eda_reservation_aware_v2": "adr_eda_reservation_aware_v2",
        "adr-reservation-aware-v2": "adr_eda_reservation_aware_v2",
        "adr-eda-reservation-overload-aware-v1": "adr_eda_reservation_overload_aware_v1",
        "adr_eda_reservation_overload_aware_v1": "adr_eda_reservation_overload_aware_v1",
        "adr-reservation-overload-aware-v1": "adr_eda_reservation_overload_aware_v1",
        "adr-eda-service-guard-v1": "adr_eda_service_guard_v1",
        "adr_eda_service_guard_v1": "adr_eda_service_guard_v1",
        "adr-service-guard-v1": "adr_eda_service_guard_v1",
        "adr-eda-service-guard-v2": "adr_eda_service_guard_v2",
        "adr_eda_service_guard_v2": "adr_eda_service_guard_v2",
        "adr-service-guard-v2": "adr_eda_service_guard_v2",
        "adr-eda-service-guard-v3": "adr_eda_service_guard_v3",
        "adr_eda_service_guard_v3": "adr_eda_service_guard_v3",
        "adr-service-guard-v3": "adr_eda_service_guard_v3",
        "adr-eda-service-guard-v4": "adr_eda_service_guard_v4",
        "adr_eda_service_guard_v4": "adr_eda_service_guard_v4",
        "adr-service-guard-v4": "adr_eda_service_guard_v4",
        "adr-eda-service-constrained-selector": "adr_eda_service_constrained_selector",
        "adr_eda_service_constrained_selector": "adr_eda_service_constrained_selector",
        "adr-service-constrained-selector": "adr_eda_service_constrained_selector",
        "adr-eda-service-constrained-selector-floor50": "adr_eda_service_constrained_selector_floor50",
        "adr_eda_service_constrained_selector_floor50": "adr_eda_service_constrained_selector_floor50",
        "adr-service-constrained-selector-floor50": "adr_eda_service_constrained_selector_floor50",
        "adr-eda-service-constrained-selector-floor60": "adr_eda_service_constrained_selector_floor60",
        "adr_eda_service_constrained_selector_floor60": "adr_eda_service_constrained_selector_floor60",
        "adr-service-constrained-selector-floor60": "adr_eda_service_constrained_selector_floor60",
        "adr-eda-service-constrained-selector-floor65": "adr_eda_service_constrained_selector_floor65",
        "adr_eda_service_constrained_selector_floor65": "adr_eda_service_constrained_selector_floor65",
        "adr-service-constrained-selector-floor65": "adr_eda_service_constrained_selector_floor65",
        "adr-eda-service-constrained-selector-floor70": "adr_eda_service_constrained_selector_floor70",
        "adr_eda_service_constrained_selector_floor70": "adr_eda_service_constrained_selector_floor70",
        "adr-service-constrained-selector-floor70": "adr_eda_service_constrained_selector_floor70",
        "adr-eda-service-constrained-selector-floor75": "adr_eda_service_constrained_selector_floor75",
        "adr_eda_service_constrained_selector_floor75": "adr_eda_service_constrained_selector_floor75",
        "adr-service-constrained-selector-floor75": "adr_eda_service_constrained_selector_floor75",
        "adr-eda-reservation-costaware-v1": "adr_eda_reservation_costaware_v1",
        "adr_eda_reservation_costaware_v1": "adr_eda_reservation_costaware_v1",
        "adr-reservation-costaware-v1": "adr_eda_reservation_costaware_v1",
        "adr-eda-reservation-costaware-v2": "adr_eda_reservation_costaware_v2",
        "adr_eda_reservation_costaware_v2": "adr_eda_reservation_costaware_v2",
        "adr-reservation-costaware-v2": "adr_eda_reservation_costaware_v2",
        "adr-eda-reservation-costaware-v3": "adr_eda_reservation_costaware_v3",
        "adr_eda_reservation_costaware_v3": "adr_eda_reservation_costaware_v3",
        "adr-reservation-costaware-v3": "adr_eda_reservation_costaware_v3",
    }
    profile = aliases.get(profile, profile)
    valid = {
        "custom",
        "default_event",
        "balanced_event",
        "fast_event",
        "fast_strong_react",
        "adr_eda_v2_1",
        "adr_eda_v2_1_carsharing",
        "adr_eda_v2_2_carsharing",
        "adr_eda_adaptive_weights",
        "adr_eda_adaptive_diagnostics_only",
        "adr_eda_adaptive_weights_activation_probe",
        "adr_eda_adaptive_weights_guarded",
        "adr_eda_adaptive_weights_selector_v2",
        "adr_eda_contextual_weights",
        "adr_eda_fixed_weights_activation_probe",
        "selective_simple_carsharing",
        "adr_eda_reservation_aware_v1",
        "adr_eda_reservation_aware_v2",
        "adr_eda_reservation_overload_aware_v1",
        "adr_eda_service_guard_v1",
        "adr_eda_service_guard_v2",
        "adr_eda_service_guard_v3",
        "adr_eda_service_guard_v4",
        "adr_eda_service_constrained_selector",
        "adr_eda_service_constrained_selector_floor50",
        "adr_eda_service_constrained_selector_floor60",
        "adr_eda_service_constrained_selector_floor65",
        "adr_eda_service_constrained_selector_floor70",
        "adr_eda_service_constrained_selector_floor75",
        "adr_eda_reservation_costaware_v1",
        "adr_eda_reservation_costaware_v2",
        "adr_eda_reservation_costaware_v3",
    }
    if profile not in valid:
        supported = ", ".join(sorted(valid))
        raise ValueError(f"--dynamic-online-profile='{profile}' no soportado. Usa: {supported}")

    resolved = {
        "profile": profile,
        "with_reaction": bool(args.with_reaction),
        "adaptive_policy_enabled": bool(args.adaptive_policy_enabled),
        "reaction_quality_gate_enabled": bool(args.reaction_quality_gate_enabled),
        "dynamic_change_detector_mode": str(args.dynamic_change_detector_mode),
        "dynamic_reopt_trigger": str(args.dynamic_reopt_trigger),
        "dynamic_threshold_mild": float(args.dynamic_threshold_mild),
        "dynamic_threshold_severe": float(args.dynamic_threshold_severe),
        "dynamic_min_reopt_interval_min": float(args.dynamic_min_reopt_interval_min),
        "dynamic_max_reopt_interval_min": float(args.dynamic_max_reopt_interval_min),
        "react_strength": float(args.react_strength),
        "immigrant_ratio": float(args.immigrant_ratio),
        "react_strength_mild": float(args.react_strength_mild),
        "immigrant_ratio_mild": float(args.immigrant_ratio_mild),
        "reaction_min_drift_ratio": 1.10,
        "reaction_min_interval": 3,
        "selection_smooth_lambda": float(args.policy_smooth_lambda),
        "selection_smooth_metric": str(args.policy_smooth_metric),
        "incumbent_gate_enabled": bool(args.eda_incumbent_gate_enabled),
        "incumbent_min_improvement": float(args.eda_incumbent_min_improvement),
        "adaptive_response": {
            "enabled": bool(args.adaptive_response_enabled),
            "diagnostics_enabled": bool(args.adaptive_response_diagnostics),
            "score_scale": float(args.adaptive_response_score_scale),
            "similarity_decay": float(args.adaptive_response_similarity_decay),
        },
    }

    if profile == "custom":
        return resolved

    profile_overrides: Dict[str, Dict[str, Any]] = {
        "default_event": {
            "with_reaction": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
        },
        "balanced_event": {
            "with_reaction": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.35,
            "dynamic_min_reopt_interval_min": 5.0,
            "dynamic_max_reopt_interval_min": 20.0,
        },
        "fast_event": {
            "with_reaction": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.14,
            "dynamic_threshold_severe": 0.30,
            "dynamic_min_reopt_interval_min": 5.0,
            "dynamic_max_reopt_interval_min": 15.0,
        },
        "fast_strong_react": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.14,
            "dynamic_threshold_severe": 0.30,
            "dynamic_min_reopt_interval_min": 5.0,
            "dynamic_max_reopt_interval_min": 15.0,
            "react_strength": 2.8,
            "immigrant_ratio": 0.28,
            "react_strength_mild": 1.6,
            "immigrant_ratio_mild": 0.16,
        },
        "adr_eda_v2_1": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.14,
            "dynamic_threshold_severe": 0.30,
            "dynamic_min_reopt_interval_min": 5.0,
            "dynamic_max_reopt_interval_min": 15.0,
            "react_strength": 2.8,
            "immigrant_ratio": 0.28,
            "react_strength_mild": 1.6,
            "immigrant_ratio_mild": 0.16,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 6,
            "local_search_front_ratio": 0.55,
            "local_search_step": 0.06,
            "local_search_max_points": 36,
        },
        "adr_eda_v2_1_carsharing": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.18,
            "dynamic_threshold_severe": 0.36,
            "dynamic_min_reopt_interval_min": 10.0,
            "dynamic_max_reopt_interval_min": 20.0,
            "react_strength": 2.1,
            "immigrant_ratio": 0.16,
            "react_strength_mild": 1.2,
            "immigrant_ratio_mild": 0.08,
            "selection_smooth_lambda": 0.45,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.03,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": False,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 4,
            "local_search_front_ratio": 0.35,
            "local_search_step": 0.04,
            "local_search_max_points": 20,
        },
        "adr_eda_v2_2_carsharing": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.28,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.02,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
        },
        "selective_simple_carsharing": {
            "with_reaction": False,
            "adaptive_policy_enabled": False,
            "reaction_quality_gate_enabled": False,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.0,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": False,
            "incumbent_min_improvement": 0.0,
            "split_drift_signals": False,
            "adaptive_lambda_enabled": False,
            "adaptive_lambda_cov_enabled": False,
            "adaptive_cooldown_enabled": False,
            "diversity_block_enabled": True,
            "predictive_block_enabled": False,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": False,
            "adr_front_ls_context_gate_enabled": False,
            "change_type_enabled": False,
            "adr_enabled": False,
            "reaction_quality_gate_mo_enabled": False,
            "local_search_enabled": False,
            "local_search_neighbors": 4,
            "local_search_front_ratio": 0.35,
            "local_search_step": 0.04,
            "local_search_max_points": 20,
        },
        "adr_eda_reservation_aware_v1": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.28,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.02,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
            "reservation_guard": {
                "enabled": True,
                "commitment_horizon_fraction": 0.75,
                "due_soon_window_min": 25.0,
                "selection_smooth_lambda_gain": 0.32,
                "incumbent_min_improvement_gain": 0.025,
                "react_strength_damping": 0.18,
                "immigrant_ratio_damping": 0.30,
                "react_strength_floor_scale": 0.72,
                "immigrant_ratio_floor_scale": 0.60,
                "predictive_disable_threshold": 0.85,
                "front_ls_disable_threshold": 0.60,
                "mild_step_budget_scale": 0.85,
                "medium_step_budget_scale": 0.95,
                "max_interval_step_budget_scale": 0.90,
            },
        },
        "adr_eda_reservation_aware_v2": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.28,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.02,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
            "reservation_guard": {
                "enabled": True,
                "commitment_horizon_fraction": 0.75,
                "due_soon_window_min": 25.0,
                "selection_smooth_lambda_gain": 0.48,
                "incumbent_min_improvement_gain": 0.04,
                "react_strength_damping": 0.28,
                "immigrant_ratio_damping": 0.40,
                "react_strength_floor_scale": 0.62,
                "immigrant_ratio_floor_scale": 0.50,
                "predictive_disable_threshold": 0.55,
                "front_ls_disable_threshold": 0.42,
                "predictive_disable_relocation_threshold": 0.45,
                "front_ls_disable_relocation_threshold": 0.35,
                "mild_step_budget_scale": 0.80,
                "medium_step_budget_scale": 0.88,
                "max_interval_step_budget_scale": 0.72,
                "relocating_pressure_scale": 0.05,
                "recent_deadhead_km_scale": 4.0,
                "recent_ops_cost_scale": 3.6,
                "recent_relocation_cost_scale": 6.0,
                "relocating_pressure_weight": 0.35,
                "recent_deadhead_km_weight": 0.35,
                "recent_ops_cost_weight": 0.15,
                "recent_relocation_cost_weight": 0.15,
                "relocation_debt_signal_weight": 0.28,
                "max_interval_hold_threshold": 0.42,
                "hold_service_reject_rate_max": 0.05,
                "hold_service_coverage_gap_max": 0.25,
            },
        },
        "adr_eda_reservation_overload_aware_v1": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.28,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.02,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
            "reservation_guard": {
                "enabled": True,
                "commitment_horizon_fraction": 0.75,
                "due_soon_window_min": 25.0,
                "selection_smooth_lambda_gain": 0.48,
                "incumbent_min_improvement_gain": 0.04,
                "react_strength_damping": 0.28,
                "immigrant_ratio_damping": 0.40,
                "react_strength_floor_scale": 0.62,
                "immigrant_ratio_floor_scale": 0.50,
                "predictive_disable_threshold": 0.55,
                "front_ls_disable_threshold": 0.42,
                "predictive_disable_relocation_threshold": 0.45,
                "front_ls_disable_relocation_threshold": 0.35,
                "mild_step_budget_scale": 0.80,
                "medium_step_budget_scale": 0.88,
                "max_interval_step_budget_scale": 0.72,
                "relocating_pressure_scale": 0.05,
                "recent_deadhead_km_scale": 4.0,
                "recent_ops_cost_scale": 3.6,
                "recent_relocation_cost_scale": 6.0,
                "relocating_pressure_weight": 0.35,
                "recent_deadhead_km_weight": 0.35,
                "recent_ops_cost_weight": 0.15,
                "recent_relocation_cost_weight": 0.15,
                "relocation_debt_signal_weight": 0.28,
                "demand_pressure_scale": 6.0,
                "coverage_gap_overload_scale": 0.20,
                "recent_reject_rate_overload_scale": 0.08,
                "demand_pressure_overload_weight": 0.45,
                "coverage_gap_overload_weight": 0.30,
                "recent_reject_rate_overload_weight": 0.25,
                "overload_relief_weight": 0.55,
                "overload_unsuppress_threshold": 0.35,
                "max_interval_hold_threshold": 0.42,
                "hold_service_reject_rate_max": 0.05,
                "hold_service_coverage_gap_max": 0.25,
            },
        },
        "adr_eda_service_guard_v1": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.15,
            "dynamic_threshold_severe": 0.32,
            "dynamic_min_reopt_interval_min": 6.0,
            "dynamic_max_reopt_interval_min": 16.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.24,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.015,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
            "reservation_guard": {
                "enabled": True,
                "commitment_horizon_fraction": 0.75,
                "due_soon_window_min": 25.0,
                "selection_smooth_lambda_gain": 0.36,
                "incumbent_min_improvement_gain": 0.03,
                "react_strength_damping": 0.18,
                "immigrant_ratio_damping": 0.30,
                "react_strength_floor_scale": 0.72,
                "immigrant_ratio_floor_scale": 0.60,
                "predictive_disable_threshold": 0.68,
                "front_ls_disable_threshold": 0.50,
                "predictive_disable_relocation_threshold": 0.52,
                "front_ls_disable_relocation_threshold": 0.42,
                "mild_step_budget_scale": 0.88,
                "medium_step_budget_scale": 0.92,
                "max_interval_step_budget_scale": 0.82,
                "relocating_pressure_scale": 0.05,
                "recent_deadhead_km_scale": 4.0,
                "recent_ops_cost_scale": 3.6,
                "recent_relocation_cost_scale": 6.0,
                "relocating_pressure_weight": 0.30,
                "recent_deadhead_km_weight": 0.30,
                "recent_ops_cost_weight": 0.20,
                "recent_relocation_cost_weight": 0.20,
                "relocation_debt_signal_weight": 0.18,
                "demand_pressure_scale": 5.0,
                "coverage_gap_overload_scale": 0.16,
                "recent_reject_rate_overload_scale": 0.04,
                "demand_pressure_overload_weight": 0.35,
                "coverage_gap_overload_weight": 0.35,
                "recent_reject_rate_overload_weight": 0.30,
                "overload_relief_weight": 0.78,
                "overload_unsuppress_threshold": 0.18,
                "max_interval_hold_threshold": 0.46,
                "hold_service_reject_rate_max": 0.02,
                "hold_service_coverage_gap_max": 0.15,
            },
        },
        "adr_eda_reservation_costaware_v1": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.28,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.02,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
            "reservation_guard": {
                "enabled": True,
                "commitment_horizon_fraction": 0.75,
                "due_soon_window_min": 25.0,
                "selection_smooth_lambda_gain": 0.48,
                "incumbent_min_improvement_gain": 0.04,
                "react_strength_damping": 0.28,
                "immigrant_ratio_damping": 0.40,
                "react_strength_floor_scale": 0.62,
                "immigrant_ratio_floor_scale": 0.50,
                "predictive_disable_threshold": 0.55,
                "front_ls_disable_threshold": 0.42,
                "predictive_disable_relocation_threshold": 0.45,
                "front_ls_disable_relocation_threshold": 0.35,
                "mild_step_budget_scale": 0.80,
                "medium_step_budget_scale": 0.88,
                "max_interval_step_budget_scale": 0.72,
                "relocating_pressure_scale": 0.05,
                "recent_deadhead_km_scale": 4.0,
                "recent_ops_cost_scale": 3.6,
                "recent_relocation_cost_scale": 6.0,
                "relocating_pressure_weight": 0.35,
                "recent_deadhead_km_weight": 0.35,
                "recent_ops_cost_weight": 0.15,
                "recent_relocation_cost_weight": 0.15,
                "relocation_debt_signal_weight": 0.28,
                "max_interval_hold_threshold": 0.42,
                "hold_service_reject_rate_max": 0.05,
                "hold_service_coverage_gap_max": 0.25,
                "cost_guard_enabled": True,
                "cost_guard_activation_signal_min": 0.20,
                "cost_guard_activation_relocation_signal_min": 0.15,
                "candidate_extra_deadhead_km_max": 18.0,
                "candidate_extra_operational_cost_eur_max": 18.0,
                "candidate_extra_relocation_cost_eur_max": 18.0,
                "candidate_min_profit_gain_eur_for_costly_move": 6.0,
            },
        },
        "adr_eda_reservation_costaware_v2": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.28,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.02,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
            "reservation_guard": {
                "enabled": True,
                "commitment_horizon_fraction": 0.75,
                "due_soon_window_min": 25.0,
                "selection_smooth_lambda_gain": 0.48,
                "incumbent_min_improvement_gain": 0.04,
                "react_strength_damping": 0.28,
                "immigrant_ratio_damping": 0.40,
                "react_strength_floor_scale": 0.62,
                "immigrant_ratio_floor_scale": 0.50,
                "predictive_disable_threshold": 0.55,
                "front_ls_disable_threshold": 0.42,
                "predictive_disable_relocation_threshold": 0.45,
                "front_ls_disable_relocation_threshold": 0.35,
                "mild_step_budget_scale": 0.80,
                "medium_step_budget_scale": 0.88,
                "max_interval_step_budget_scale": 0.72,
                "relocating_pressure_scale": 0.05,
                "recent_deadhead_km_scale": 4.0,
                "recent_ops_cost_scale": 3.6,
                "recent_relocation_cost_scale": 6.0,
                "relocating_pressure_weight": 0.35,
                "recent_deadhead_km_weight": 0.35,
                "recent_ops_cost_weight": 0.15,
                "recent_relocation_cost_weight": 0.15,
                "relocation_debt_signal_weight": 0.28,
                "max_interval_hold_threshold": 0.42,
                "hold_service_reject_rate_max": 0.05,
                "hold_service_coverage_gap_max": 0.25,
                "cost_guard_enabled": True,
                "cost_guard_disable_on_bootstrap": True,
                "cost_guard_activation_signal_min": 0.20,
                "cost_guard_activation_relocation_signal_min": 0.15,
                "cost_guard_activation_demand_signal_min": 0.20,
                "cost_guard_activation_coverage_gap_signal_min": 0.25,
                "cost_guard_activation_recent_reject_signal_min": 0.20,
                "candidate_extra_deadhead_km_max": 18.0,
                "candidate_extra_operational_cost_eur_max": 18.0,
                "candidate_extra_relocation_cost_eur_max": 18.0,
                "candidate_min_profit_gain_eur_for_costly_move": 6.0,
            },
        },
        "adr_eda_reservation_costaware_v3": {
            "with_reaction": True,
            "adaptive_policy_enabled": True,
            "reaction_quality_gate_enabled": True,
            "dynamic_change_detector_mode": "problem-aware",
            "dynamic_reopt_trigger": "event",
            "dynamic_threshold_mild": 0.16,
            "dynamic_threshold_severe": 0.34,
            "dynamic_min_reopt_interval_min": 8.0,
            "dynamic_max_reopt_interval_min": 18.0,
            "react_strength": 2.4,
            "immigrant_ratio": 0.20,
            "react_strength_mild": 1.35,
            "immigrant_ratio_mild": 0.10,
            "selection_smooth_lambda": 0.28,
            "selection_smooth_metric": "l1",
            "incumbent_gate_enabled": True,
            "incumbent_min_improvement": 0.02,
            "split_drift_signals": True,
            "adaptive_lambda_enabled": True,
            "adaptive_lambda_cov_enabled": True,
            "adaptive_cooldown_enabled": True,
            "diversity_block_enabled": True,
            "predictive_block_enabled": True,
            "continuity_block_enabled": True,
            "adr_front_ls_enabled": True,
            "adr_front_ls_context_gate_enabled": True,
            "change_type_enabled": True,
            "adr_enabled": True,
            "reaction_quality_gate_mo_enabled": True,
            "local_search_enabled": True,
            "local_search_neighbors": 5,
            "local_search_front_ratio": 0.45,
            "local_search_step": 0.05,
            "local_search_max_points": 28,
            "reservation_guard": {
                "enabled": True,
                "commitment_horizon_fraction": 0.75,
                "due_soon_window_min": 25.0,
                "selection_smooth_lambda_gain": 0.48,
                "incumbent_min_improvement_gain": 0.04,
                "react_strength_damping": 0.28,
                "immigrant_ratio_damping": 0.40,
                "react_strength_floor_scale": 0.62,
                "immigrant_ratio_floor_scale": 0.50,
                "predictive_disable_threshold": 0.55,
                "front_ls_disable_threshold": 0.42,
                "predictive_disable_relocation_threshold": 0.45,
                "front_ls_disable_relocation_threshold": 0.35,
                "mild_step_budget_scale": 0.80,
                "medium_step_budget_scale": 0.88,
                "max_interval_step_budget_scale": 0.72,
                "relocating_pressure_scale": 0.05,
                "recent_deadhead_km_scale": 4.0,
                "recent_ops_cost_scale": 3.6,
                "recent_relocation_cost_scale": 6.0,
                "relocating_pressure_weight": 0.35,
                "recent_deadhead_km_weight": 0.35,
                "recent_ops_cost_weight": 0.15,
                "recent_relocation_cost_weight": 0.15,
                "relocation_debt_signal_weight": 0.28,
                "max_interval_hold_threshold": 0.42,
                "hold_service_reject_rate_max": 0.05,
                "hold_service_coverage_gap_max": 0.25,
                "cost_guard_enabled": True,
                "cost_guard_disable_on_bootstrap": True,
                "cost_guard_activation_signal_min": 0.20,
                "cost_guard_activation_relocation_signal_min": 0.15,
                "cost_guard_activation_demand_signal_min": 0.20,
                "cost_guard_activation_coverage_gap_signal_min": 0.25,
                "cost_guard_activation_recent_reject_signal_min": 0.20,
                "cost_guard_burst_enabled": True,
                "cost_guard_burst_commitment_signal_min": 0.16,
                "cost_guard_burst_relocation_signal_min": 0.20,
                "candidate_shift_penalty_burst_max": 0.16,
                "candidate_min_profit_gain_eur_for_burst_move": 10.0,
                "candidate_extra_deadhead_km_max": 18.0,
                "candidate_extra_operational_cost_eur_max": 18.0,
                "candidate_extra_relocation_cost_eur_max": 18.0,
                "candidate_min_profit_gain_eur_for_costly_move": 6.0,
            },
        },
    }
    profile_overrides["adr_eda_adaptive_weights"] = copy.deepcopy(
        profile_overrides["adr_eda_v2_2_carsharing"]
    )
    profile_overrides["adr_eda_adaptive_weights"]["adaptive_response"] = {
        "enabled": True,
        "diagnostics_enabled": True,
        "score_scale": 0.34,
        "similarity_decay": 1.0,
    }
    profile_overrides["adr_eda_adaptive_diagnostics_only"] = copy.deepcopy(
        profile_overrides["adr_eda_v2_2_carsharing"]
    )
    profile_overrides["adr_eda_adaptive_diagnostics_only"]["adaptive_response"] = {
        "enabled": False,
        "diagnostics_enabled": True,
        "score_scale": 0.34,
        "similarity_decay": 1.0,
    }
    profile_overrides["adr_eda_adaptive_weights_activation_probe"] = copy.deepcopy(
        profile_overrides["adr_eda_adaptive_weights"]
    )
    profile_overrides["adr_eda_adaptive_weights_activation_probe"].update(
        {
            "dynamic_threshold_mild": 0.01,
            "dynamic_threshold_severe": 0.03,
            "reaction_min_drift_ratio": 1.01,
            "reaction_min_interval": 1,
            "adr_front_ls_enabled": False,
            "adr_front_ls_context_gate_enabled": False,
            "local_search_enabled": False,
        }
    )
    profile_overrides["adr_eda_adaptive_weights_activation_probe"]["adaptive_response"].update(
        {
            "front_ls_low_scale": 0.0,
            "front_ls_high_scale": 0.0,
            "selection_policy_diagnostics_enabled": True,
        }
    )
    profile_overrides["adr_eda_adaptive_weights_guarded"] = copy.deepcopy(
        profile_overrides["adr_eda_adaptive_weights_activation_probe"]
    )
    profile_overrides["adr_eda_adaptive_weights_guarded"]["adaptive_response"].update(
        {
            "guard_enabled": True,
            "guard_shift_l1_threshold": 2.50,
            "guard_min_score_gain": 0.09,
            "guard_recent_relocation_cost_min": 6.0,
            "guard_ops_pressure_min": 0.25,
            "guard_relocating_pressure_min": 0.03,
            "guard_require_debt_signal": True,
        }
    )
    profile_overrides["adr_eda_adaptive_weights_selector_v2"] = copy.deepcopy(
        profile_overrides["adr_eda_adaptive_weights_activation_probe"]
    )
    profile_overrides["adr_eda_adaptive_weights_selector_v2"]["adaptive_response"].update(
        {
            "selection_policy_enabled": True,
            "selection_policy_diagnostics_enabled": True,
            "selection_policy_strength": 0.40,
            "selection_policy_objective_gain": 0.70,
            "selection_policy_continuity_gain": 1.00,
            "selection_policy_diversity_gain": 1.25,
            "selection_policy_max_shift_penalty": 0.80,
            "selection_policy_max_shift_reward": 0.25,
        }
    )
    profile_overrides["adr_eda_contextual_weights"] = copy.deepcopy(
        profile_overrides["adr_eda_adaptive_weights_selector_v2"]
    )
    profile_overrides["adr_eda_contextual_weights"]["adaptive_response"].update(
        {
            "weight_policy": "contextual_formula",
            "contextual_temperature": 0.70,
            "contextual_base_prior_strength": 0.35,
            "contextual_diversity_target": 0.18,
            "contextual_min_predictive": 0.05,
            "contextual_max_predictive": 0.45,
            "contextual_min_continuity": 0.10,
            "contextual_max_continuity": 0.50,
            "contextual_min_diversity": 0.15,
            "contextual_max_diversity": 0.70,
            "contextual_min_front_ls": 0.00,
            "contextual_max_front_ls": 0.00,
        }
    )
    profile_overrides["adr_eda_fixed_weights_activation_probe"] = copy.deepcopy(
        profile_overrides["adr_eda_adaptive_weights_activation_probe"]
    )
    profile_overrides["adr_eda_fixed_weights_activation_probe"]["adaptive_response"].update(
        {
            "enabled": False,
            "diagnostics_enabled": True,
        }
    )
    profile_overrides["adr_eda_service_guard_v2"] = copy.deepcopy(
        profile_overrides["adr_eda_service_guard_v1"]
    )
    profile_overrides["adr_eda_service_guard_v2"].update(
        {
            "dynamic_threshold_mild": 0.10,
            "dynamic_threshold_severe": 0.22,
            "dynamic_min_reopt_interval_min": 0.0,
            "dynamic_max_reopt_interval_min": 8.0,
            "selection_smooth_lambda": 0.12,
            "incumbent_gate_enabled": False,
            "incumbent_min_improvement": 0.0,
        }
    )
    profile_overrides["adr_eda_service_guard_v2"]["reservation_guard"].update(
        {
            "selection_smooth_lambda_gain": 0.12,
            "incumbent_min_improvement_gain": 0.0,
            "react_strength_damping": 0.0,
            "immigrant_ratio_damping": 0.0,
            "predictive_disable_threshold": 1.5,
            "front_ls_disable_threshold": 1.5,
            "predictive_disable_relocation_threshold": 1.5,
            "front_ls_disable_relocation_threshold": 1.5,
            "mild_step_budget_scale": 1.0,
            "medium_step_budget_scale": 1.0,
            "max_interval_step_budget_scale": 1.0,
            "demand_pressure_scale": 3.0,
            "coverage_gap_overload_scale": 0.08,
            "recent_reject_rate_overload_scale": 0.02,
            "demand_pressure_overload_weight": 0.25,
            "coverage_gap_overload_weight": 0.25,
            "recent_reject_rate_overload_weight": 0.50,
            "overload_relief_weight": 0.0,
            "overload_unsuppress_threshold": 0.0,
            "max_interval_hold_threshold": 1.5,
            "hold_service_reject_rate_max": 0.0,
            "hold_service_coverage_gap_max": 0.0,
        }
    )
    profile_overrides["adr_eda_service_guard_v3"] = copy.deepcopy(
        profile_overrides["adr_eda_service_guard_v2"]
    )
    profile_overrides["adr_eda_service_guard_v3"]["reservation_guard"].update(
        {
            "service_deployment_guard_enabled": True,
            "service_guard_disable_on_bootstrap": True,
            "service_guard_activation_demand_signal_min": 0.15,
            "service_guard_activation_coverage_gap_signal_min": 0.15,
            "service_guard_activation_recent_reject_signal_min": 0.15,
            "service_guard_max_service_drop": 0.0,
            "service_guard_max_extra_rejected": 0.0,
            "service_guard_max_extra_deadhead_km": 3.0,
            "service_guard_min_profit_gain_eur": -3.0,
        }
    )
    profile_overrides["adr_eda_service_guard_v4"] = copy.deepcopy(
        profile_overrides["adr_eda_service_guard_v3"]
    )
    profile_overrides["adr_eda_service_guard_v4"]["reservation_guard"].update(
        {
            "service_guard_require_opportunity": True,
            "service_guard_min_service_gain": 0.0,
            "service_guard_min_rejected_reduction": 0.0,
            "service_guard_neutral_min_profit_gain_eur": 3.0,
        }
    )
    profile_overrides["adr_eda_service_constrained_selector"] = copy.deepcopy(
        profile_overrides["adr_eda_v2_2_carsharing"]
    )
    profile_overrides["adr_eda_service_constrained_selector"]["adaptive_response"] = {
        "enabled": False,
        "diagnostics_enabled": True,
        "service_constrained_selection_enabled": True,
        "service_constrained_selection_diagnostics_enabled": True,
        "service_constrained_service_floor": 0.99,
        "service_constrained_max_rejected_requests": 0.0,
        "service_constrained_profit_weight": 1.00,
        "service_constrained_deadhead_weight": 0.35,
        "service_constrained_health_weight": 0.20,
        "service_constrained_shift_weight": 0.15,
        "service_constrained_rejected_weight": 0.35,
    }
    profile_overrides["adr_eda_service_constrained_selector_floor60"] = copy.deepcopy(
        profile_overrides["adr_eda_service_constrained_selector"]
    )
    profile_overrides["adr_eda_service_constrained_selector_floor60"]["adaptive_response"].update(
        {
            "service_constrained_service_floor": 0.60,
            "service_constrained_max_rejected_requests": 1e9,
        }
    )
    for floor_name, floor_value in (
        ("floor50", 0.50),
        ("floor65", 0.65),
        ("floor70", 0.70),
        ("floor75", 0.75),
    ):
        profile_name = f"adr_eda_service_constrained_selector_{floor_name}"
        profile_overrides[profile_name] = copy.deepcopy(
            profile_overrides["adr_eda_service_constrained_selector_floor60"]
        )
        profile_overrides[profile_name]["adaptive_response"].update(
            {
                "service_constrained_service_floor": float(floor_value),
                "service_constrained_max_rejected_requests": 1e9,
            }
        )
    resolved.update(profile_overrides[profile])
    return resolved


def run_online_eda(args: argparse.Namespace) -> Dict[str, Any]:
    from .live_requests_demo import run_stream

    optimizer_random_seed = _seed_optimizer_random_state(
        getattr(args, "optimizer_random_seed", -1)
    )
    audit_prefix, default_fronts_json, default_events_jsonl = _derive_online_artifact_paths(args)
    audit_prefix = str(args.audit_prefix).strip() or audit_prefix
    fronts_json = str(args.eda_fronts_json).strip() or default_fronts_json
    events_jsonl = str(args.output_jsonl).strip() or ""
    resolved_profile = _resolve_online_reopt_profile(args)
    resolved_dynamic_profile = _resolve_dynamic_online_profile(args)
    online_model_cfg, online_scenario, online_state0, _ = _build_context(
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
    )

    effective_policy_smooth_lambda = float(resolved_profile["policy_smooth_lambda"])
    effective_policy_smooth_metric = str(resolved_profile["policy_smooth_metric"])
    if str(args.eda_online_engine) == "dynamic-stateful":
        effective_policy_smooth_lambda = float(resolved_dynamic_profile["selection_smooth_lambda"])
        effective_policy_smooth_metric = str(resolved_dynamic_profile["selection_smooth_metric"])

    live_args = argparse.Namespace(
        mode="batch",
        seed=int(args.seed),
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        fleet_size=int(args.fleet_size),
        charger_count=int(args.charger_count),
        ops_resources=int(args.ops_resources),
        travel_metrics_json=str(args.travel_metrics_json),
        zone_base_demand_json=str(args.zone_base_demand_json),
        real_demand_profile_dir=str(args.real_demand_profile_dir),
        real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
        fare_mode=str(args.fare_mode),
        fare_base_eur=float(args.fare_base_eur),
        fare_per_km_eur=float(args.fare_per_km_eur),
        fare_per_min_eur=float(args.fare_per_min_eur),
        fare_min_total_eur=float(args.fare_min_total_eur),
        fare_included_km=float(args.fare_included_km),
        fare_extra_km_eur=float(args.fare_extra_km_eur),
        reservation_share=float(args.reservation_share),
        reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
        immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
        reservation_mean_lead_min=float(args.reservation_mean_lead_min),
        ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
        ops_cost_profile=str(args.ops_cost_profile),
        relocation_cost_per_km=float(args.relocation_cost_per_km),
        ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
        ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
        energy_cost_per_kwh=float(args.energy_cost_per_kwh),
        max_user_access_walk_min=float(args.max_user_access_walk_min),
        ops_relocation_handover_min=float(args.ops_relocation_handover_min),
        ops_charge_plug_min=float(args.ops_charge_plug_min),
        base_demand_scale=float(args.base_demand_scale),
        real_duration_sec=300.0,
        sim_min_per_real_sec=1.0,
        sim_duration_min=float(args.sim_duration_min),
        output_jsonl=str(events_jsonl),
        audit_prefix=str(audit_prefix),
        min_same_zone_trip_km=float(args.min_same_zone_trip_km),
        simple_access_model=bool(args.simple_access_model),
        eda_reopt_enabled=True,
        eda_reopt_every_epochs=int(args.eda_reopt_every_epochs),
        eda_online_engine=str(args.eda_online_engine),
        eda_horizon_hours=int(args.horizon_hours),
        eda_size_gen=int(args.size_gen),
        eda_max_iter=int(args.max_iter),
        eda_dead_iter=int(args.dead_iter),
        eda_alpha=float(args.alpha),
        eda_w_noise=float(args.w_noise),
        eda_optimizer_family=str(args.optimizer_family),
        eda_parallelize=bool(args.parallelize),
        eda_auto_fallback=bool(args.auto_fallback),
        eda_warm_start_enabled=bool(resolved_profile["eda_warm_start_enabled"]),
        eda_temporal_archive_enabled=bool(resolved_profile["eda_temporal_archive_enabled"]),
        eda_temporal_archive_max_size=int(resolved_profile["eda_temporal_archive_max_size"]),
        eda_temporal_archive_keep_per_step=int(resolved_profile["eda_temporal_archive_keep_per_step"]),
        eda_incumbent_gate_enabled=bool(resolved_profile["eda_incumbent_gate_enabled"]),
        eda_incumbent_min_improvement=float(resolved_profile["eda_incumbent_min_improvement"]),
        eda_incumbent_neighbor_ratio=float(resolved_profile["eda_incumbent_neighbor_ratio"]),
        eda_incumbent_noise_scale=float(resolved_profile["eda_incumbent_noise_scale"]),
        eda_archive_guided_ratio=float(resolved_profile["eda_archive_guided_ratio"]),
        eda_archive_guided_noise_scale=float(resolved_profile["eda_archive_guided_noise_scale"]),
        eda_fronts_json=str(fronts_json),
        eda_rollout_demand_mode=str(args.eda_rollout_demand_mode),
        policy_smooth_lambda=float(effective_policy_smooth_lambda),
        policy_smooth_metric=str(effective_policy_smooth_metric),
        elite_factor=float(args.elite_factor),
        archive_generations=float(args.archive_generations),
        pm_family=str(args.pm_family),
        num_folds=int(args.num_folds),
        archive_retention_after_change=float(args.archive_retention_after_change),
        dynamic_online_profile=str(resolved_dynamic_profile["profile"]),
        with_reaction=bool(resolved_dynamic_profile["with_reaction"]),
        adaptive_policy_enabled=bool(resolved_dynamic_profile["adaptive_policy_enabled"]),
        adaptive_response_enabled=bool(
            resolved_dynamic_profile.get("adaptive_response", {}).get("enabled", False)
        ),
        adaptive_response_diagnostics=bool(
            resolved_dynamic_profile.get("adaptive_response", {}).get("diagnostics_enabled", False)
        ),
        adaptive_response_score_scale=float(
            resolved_dynamic_profile.get("adaptive_response", {}).get("score_scale", 0.34)
        ),
        adaptive_response_similarity_decay=float(
            resolved_dynamic_profile.get("adaptive_response", {}).get("similarity_decay", 1.0)
        ),
        reaction_quality_gate_enabled=bool(resolved_dynamic_profile["reaction_quality_gate_enabled"]),
        local_search_enabled=bool(args.local_search_enabled),
        dynamic_change_detector_mode=str(resolved_dynamic_profile["dynamic_change_detector_mode"]),
        dynamic_reopt_trigger=str(resolved_dynamic_profile["dynamic_reopt_trigger"]),
        dynamic_threshold_mild=float(resolved_dynamic_profile["dynamic_threshold_mild"]),
        dynamic_threshold_severe=float(resolved_dynamic_profile["dynamic_threshold_severe"]),
        dynamic_regime_ewma_alpha=float(args.dynamic_regime_ewma_alpha),
        dynamic_min_reopt_interval_min=float(resolved_dynamic_profile["dynamic_min_reopt_interval_min"]),
        dynamic_max_reopt_interval_min=float(resolved_dynamic_profile["dynamic_max_reopt_interval_min"]),
        dynamic_hard_low_soc_share=float(args.dynamic_hard_low_soc_share),
        dynamic_hard_charger_pressure=float(args.dynamic_hard_charger_pressure),
        dynamic_hard_reservation_pressure=float(args.dynamic_hard_reservation_pressure),
        dynamic_hard_coverage_gap=float(args.dynamic_hard_coverage_gap),
        n_sentinels=int(args.n_sentinels),
        drift_alpha=float(args.drift_alpha),
        lambda_ewma=float(args.lambda_ewma),
        sentinel_refresh=int(args.sentinel_refresh),
        k_consecutive=int(args.k_consecutive),
        react_strength=float(resolved_dynamic_profile["react_strength"]),
        immigrant_ratio=float(resolved_dynamic_profile["immigrant_ratio"]),
        react_strength_mild=float(resolved_dynamic_profile["react_strength_mild"]),
        immigrant_ratio_mild=float(resolved_dynamic_profile["immigrant_ratio_mild"]),
        local_search_neighbors=int(args.local_search_neighbors),
        local_search_front_ratio=float(args.local_search_front_ratio),
        local_search_step=float(args.local_search_step),
        local_search_max_points=int(args.local_search_max_points),
        optimizer_random_seed=-1 if optimizer_random_seed is None else int(optimizer_random_seed),
    )
    run_stream(live_args)

    summary_path = Path(str(audit_prefix) + "_summary.json")
    status_path = Path(str(audit_prefix) + "_status.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    fronts = json.loads(Path(fronts_json).read_text(encoding="utf-8")) if Path(fronts_json).exists() else {}

    return {
        "mode": "online-eda",
        "meta": {
            "seed": int(args.seed),
            "optimizer_random_seed": None
            if optimizer_random_seed is None
            else int(optimizer_random_seed),
            "sim_duration_min": float(args.sim_duration_min),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            "travel_metrics_json": str(args.travel_metrics_json),
            "base_demand_scale": float(args.base_demand_scale),
            **_scenario_meta(scenario=online_scenario, model_cfg=online_model_cfg, state0=online_state0),
            "eda_horizon_hours": int(args.horizon_hours),
            "eda_reopt_every_epochs": int(args.eda_reopt_every_epochs),
            "eda_online_engine": str(args.eda_online_engine),
            "eda_rollout_demand_mode": str(args.eda_rollout_demand_mode),
            "online_reopt_profile": str(resolved_profile["profile"]),
            "dynamic_online_profile": str(resolved_dynamic_profile["profile"]),
            "policy_smooth_lambda": float(resolved_profile["policy_smooth_lambda"]),
            "policy_smooth_metric": str(resolved_profile["policy_smooth_metric"]),
            "dynamic_selection_smooth_lambda": float(
                resolved_dynamic_profile["selection_smooth_lambda"]
            ),
            "dynamic_selection_smooth_metric": str(
                resolved_dynamic_profile["selection_smooth_metric"]
            ),
            "eda_warm_start_enabled": bool(resolved_profile["eda_warm_start_enabled"]),
            "eda_temporal_archive_enabled": bool(resolved_profile["eda_temporal_archive_enabled"]),
            "eda_temporal_archive_max_size": int(resolved_profile["eda_temporal_archive_max_size"]),
            "eda_temporal_archive_keep_per_step": int(resolved_profile["eda_temporal_archive_keep_per_step"]),
            "eda_incumbent_gate_enabled": bool(resolved_profile["eda_incumbent_gate_enabled"]),
            "eda_incumbent_min_improvement": float(resolved_profile["eda_incumbent_min_improvement"]),
            "eda_incumbent_neighbor_ratio": float(resolved_profile["eda_incumbent_neighbor_ratio"]),
            "eda_incumbent_noise_scale": float(resolved_profile["eda_incumbent_noise_scale"]),
            "eda_archive_guided_ratio": float(resolved_profile["eda_archive_guided_ratio"]),
            "eda_archive_guided_noise_scale": float(resolved_profile["eda_archive_guided_noise_scale"]),
            "optimizer_family_requested": str(args.optimizer_family),
            "pm_family_requested": str(args.pm_family),
            "with_reaction": bool(resolved_dynamic_profile["with_reaction"]),
            "adaptive_response": dict(resolved_dynamic_profile.get("adaptive_response", {})),
            "dynamic_change_detector_mode": str(resolved_dynamic_profile["dynamic_change_detector_mode"]),
            "dynamic_reopt_trigger": str(resolved_dynamic_profile["dynamic_reopt_trigger"]),
            "dynamic_threshold_mild": float(resolved_dynamic_profile["dynamic_threshold_mild"]),
            "dynamic_threshold_severe": float(resolved_dynamic_profile["dynamic_threshold_severe"]),
            "dynamic_min_reopt_interval_min": float(resolved_dynamic_profile["dynamic_min_reopt_interval_min"]),
            "dynamic_max_reopt_interval_min": float(resolved_dynamic_profile["dynamic_max_reopt_interval_min"]),
            "dynamic_hard_low_soc_share": float(args.dynamic_hard_low_soc_share),
            "dynamic_hard_charger_pressure": float(args.dynamic_hard_charger_pressure),
            "dynamic_hard_reservation_pressure": float(args.dynamic_hard_reservation_pressure),
            "dynamic_hard_coverage_gap": float(args.dynamic_hard_coverage_gap),
            "size_gen_requested": int(args.size_gen),
            "max_iter_requested": int(args.max_iter),
            "dead_iter_requested": int(args.dead_iter),
            "auto_fallback": bool(args.auto_fallback),
        },
        "audit_summary": summary,
        "eda_fronts": fronts,
        "artifacts": {
            "audit_summary_json": str(summary_path),
            "audit_status_csv": str(status_path),
            "eda_fronts_json": str(fronts_json),
            "events_jsonl": str(events_jsonl) if events_jsonl else "",
        },
    }


def run_mode(args: argparse.Namespace) -> Dict[str, Any]:
    mode = str(args.mode)
    if mode == "baseline":
        return run_demo(
            seed=int(args.seed),
            horizon_hours=int(args.horizon_hours),
            decision_interval_min=int(args.decision_interval_min),
            planning_horizon_min=int(args.planning_horizon_min),
            fleet_size=int(args.fleet_size),
            charger_count=int(args.charger_count),
            ops_resources=int(args.ops_resources),
            travel_metrics_json=str(args.travel_metrics_json),
            zone_base_demand_json=str(args.zone_base_demand_json),
            fare_mode=str(args.fare_mode),
            fare_base_eur=float(args.fare_base_eur),
            fare_per_km_eur=float(args.fare_per_km_eur),
            fare_per_min_eur=float(args.fare_per_min_eur),
            fare_min_total_eur=float(args.fare_min_total_eur),
            fare_included_km=float(args.fare_included_km),
            fare_extra_km_eur=float(args.fare_extra_km_eur),
            reservation_share=float(args.reservation_share),
            reservation_booking_intensity_mult=float(args.reservation_booking_intensity_mult),
            immediate_request_intensity_mult=float(args.immediate_request_intensity_mult),
            reservation_mean_lead_min=float(args.reservation_mean_lead_min),
            ops_charge_action_credit_eur=float(args.ops_charge_action_credit_eur),
            ops_cost_profile=str(args.ops_cost_profile),
            relocation_cost_per_km=float(args.relocation_cost_per_km),
            ops_deadhead_cost_per_km=float(args.ops_deadhead_cost_per_km),
            ops_cost_per_action_eur=float(args.ops_cost_per_action_eur),
            energy_cost_per_kwh=float(args.energy_cost_per_kwh),
            max_user_access_walk_min=float(args.max_user_access_walk_min),
            ops_relocation_handover_min=float(args.ops_relocation_handover_min),
            ops_charge_plug_min=float(args.ops_charge_plug_min),
            real_demand_profile_dir=str(args.real_demand_profile_dir),
            real_demand_forecast_history_min=float(args.real_demand_forecast_history_min),
        )
    if mode == "online-eda":
        return run_online_eda(args)
    if mode == "static-eda":
        return run_static_eda(args)
    if mode == "pymoo-static":
        return run_pymoo_static(args)
    if mode == "pymoo-stateful":
        return run_pymoo_stateful(args)
    if mode == "pymoo-adr-stateful":
        return run_pymoo_adr_stateful(args)
    if mode == "windowed-eda":
        return run_windowed_static_eda(args)
    if mode == "stateful-eda":
        return run_stateful_dynamic_eda(args)
    raise ValueError(f"Modo no soportado: {mode}")


def run_demo_to_json(output_path: str) -> None:
    payload = run_demo()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Runner unificado del modelo free_floating_robust. "
            "Puede ejecutar baseline, online-eda, EDA estática, pymoo externo, EDA por ventanas y DynamicEDA stateful."
        )
    )
    p.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "online-eda",
            "static-eda",
            "pymoo-static",
            "pymoo-stateful",
            "pymoo-adr-stateful",
            "windowed-eda",
            "stateful-eda",
        ],
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--optimizer-random-seed",
        type=int,
        default=-1,
        help=(
            "Semilla opcional para RNG global usado por el optimizador/EDA. "
            "Por defecto queda desactivada para no alterar perfiles canonicos."
        ),
    )
    p.add_argument("--horizon-hours", type=float, default=8.0)
    p.add_argument("--sim-duration-min", type=float, default=120.0)
    p.add_argument("--decision-interval-min", type=int, default=5)
    p.add_argument("--planning-horizon-min", type=int, default=60)
    p.add_argument("--fleet-size", type=int, default=220)
    p.add_argument("--charger-count", type=int, default=22)
    p.add_argument("--ops-resources", type=int, default=26)
    p.add_argument(
        "--travel-metrics-json",
        type=str,
        default=default_travel_metrics_json() or "",
        help=(
            "Matriz de movilidad OSRM. Por defecto usa la matriz mantenida con "
            "congestion temporal y, si no existe, la matriz OSRM base."
        ),
    )
    p.add_argument(
        "--zone-base-demand-json",
        type=str,
        default="",
        help="JSON opcional con niveles base de demanda por zona: {\"Centro\": 22.0, ...}.",
    )
    p.add_argument(
        "--real-demand-profile-dir",
        type=str,
        default="",
        help=(
            "Perfil de demanda real con station_summary.csv y event_stream_top_stations.csv. "
            "Si se proporciona, reemplaza la demanda sintética por viajes observados."
        ),
    )
    p.add_argument(
        "--real-demand-forecast-history-min",
        type=float,
        default=1440.0,
        help="Ventana histórica usada para forecast online en perfiles de demanda real.",
    )
    p.add_argument(
        "--eda-rollout-demand-mode",
        type=str,
        default="observed",
        choices=["observed", "forecast"],
        help=(
            "Demanda usada solo en rollouts internos de EDA online. "
            "'observed' reproduce los eventos reales de la ventana; 'forecast' "
            "muestrea desde forecast histórico sin mirar eventos futuros."
        ),
    )
    p.add_argument("--fare-mode", type=str, default="time_only", choices=["time_only", "time_plus_km"])
    p.add_argument("--fare-base-eur", type=float, default=0.0)
    p.add_argument("--fare-per-km-eur", type=float, default=0.0)
    p.add_argument("--fare-per-min-eur", type=float, default=0.24)
    p.add_argument("--fare-min-total-eur", type=float, default=0.0)
    p.add_argument("--fare-included-km", type=float, default=200.0)
    p.add_argument("--fare-extra-km-eur", type=float, default=0.39)
    p.add_argument("--reservation-share", type=float, default=0.28)
    p.add_argument("--reservation-booking-intensity-mult", type=float, default=1.0)
    p.add_argument("--immediate-request-intensity-mult", type=float, default=1.0)
    p.add_argument("--reservation-mean-lead-min", type=float, default=35.0)
    p.add_argument("--ops-charge-action-credit-eur", type=float, default=0.0)
    p.add_argument(
        "--ops-cost-profile",
        type=str,
        default="custom",
        choices=["custom", "legacy", "conservador", "serio", "stress"],
        help=(
            "Preset de costes operativos. 'custom' respeta los valores CLI; "
            "'serio' es la recomendacion base para Madrid."
        ),
    )
    p.add_argument(
        "--relocation-cost-per-km",
        type=float,
        default=0.22,
        help="Coste variable por km cuando un operario conduce un EV en relocalizacion.",
    )
    p.add_argument(
        "--ops-deadhead-cost-per-km",
        type=float,
        default=0.22,
        help="Coste por km del desplazamiento en vacio del staff hasta el EV.",
    )
    p.add_argument(
        "--ops-cost-per-action-eur",
        type=float,
        default=1.20,
        help="Coste fijo por accion operativa de relocalizacion/carga del staff.",
    )
    p.add_argument(
        "--energy-cost-per-kwh",
        type=float,
        default=0.23,
        help="Coste energetico variable por kWh cargado.",
    )
    p.add_argument(
        "--ops-relocation-handover-min",
        type=float,
        default=1.5,
        help="Tiempo operativo adicional para aparcar/entregar una relocalizacion del staff.",
    )
    p.add_argument(
        "--ops-charge-plug-min",
        type=float,
        default=2.0,
        help="Tiempo adicional para enchufar y dejar operativo el EV al iniciar carga.",
    )
    p.add_argument(
        "--max-user-access-walk-min",
        type=float,
        default=12.0,
        help=(
            "Tiempo maximo de caminata usuario->vehiculo permitido en el modo serio. "
            "Por encima de este umbral el EV no se considera factible."
        ),
    )
    p.add_argument("--output-json", type=str, default="")

    p.add_argument("--size-gen", type=int, default=40)
    p.add_argument("--max-iter", type=int, default=6)
    p.add_argument("--dead-iter", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--archive-generations", type=float, default=4.0)
    p.add_argument("--w-noise", type=float, default=0.10)
    p.add_argument("--optimizer-family", type=str, default="SPBN", choices=["SPBN", "KDEBN"])
    p.add_argument(
        "--pymoo-algorithm",
        type=str,
        default="nsga2",
        choices=["nsga2", "dnsga2", "spea2", "sms", "smsemoa"],
    )
    p.add_argument(
        "--pymoo-execution-window-min",
        type=float,
        default=0.0,
        help=(
            "Ventana ejecutada por pymoo-stateful. Si es 0, usa decision_interval_min. "
            "El horizonte de optimizacion sigue siendo --horizon-hours."
        ),
    )
    p.add_argument("--pymoo-adr-min-update-interval", type=int, default=1)
    p.add_argument("--pymoo-adr-max-update-interval", type=int, default=4)
    p.add_argument("--pymoo-adr-service-floor", type=float, default=0.78)
    p.add_argument("--pymoo-adr-reject-threshold", type=float, default=2.0)
    p.add_argument("--parallelize", action="store_true")
    p.add_argument("--disp", action="store_true")
    p.add_argument("--auto-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eda-reopt-every-epochs", type=int, default=1)
    p.add_argument(
        "--simple-access-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Activa el atajo legacy v1: demanda solo en zonas con EV idle, "
            "asignacion solo misma zona origen y walk=0. "
            "Por defecto se usa el modo serio con acceso peatonal y asignacion entre zonas."
        ),
    )
    p.add_argument("--min-same-zone-trip-km", type=float, default=1.0)
    p.add_argument("--audit-prefix", type=str, default="")
    p.add_argument("--eda-fronts-json", type=str, default="")
    p.add_argument("--output-jsonl", type=str, default="")
    p.add_argument(
        "--online-reopt-profile",
        type=str,
        default="full",
        choices=["custom", "naive", "transfer_only", "incumbent_seed", "archive", "archive_guided", "full"],
    )
    p.add_argument(
        "--eda-online-engine",
        type=str,
        default="speda-static",
        choices=["speda-static", "dynamic-stateful"],
        help="Motor online del reoptimizador: SPEDA estático repetido o DynamicEDA stateful.",
    )
    p.add_argument(
        "--dynamic-online-profile",
        type=str,
        default="custom",
        choices=[
            "custom",
            "default_event",
            "balanced_event",
            "fast_event",
            "fast_strong_react",
            "adr_eda_v2_1",
            "adr_eda_v2_1_carsharing",
            "adr_eda_v2_2_carsharing",
            "adr_eda_adaptive_weights",
            "adr_eda_adaptive_diagnostics_only",
            "adr_eda_adaptive_weights_activation_probe",
            "adr_eda_adaptive_weights_guarded",
            "adr_eda_adaptive_weights_selector_v2",
            "adr_eda_contextual_weights",
            "adr_eda_fixed_weights_activation_probe",
            "selective_simple_carsharing",
            "adr_eda_reservation_aware_v1",
            "adr_eda_reservation_aware_v2",
            "adr_eda_reservation_overload_aware_v1",
            "adr_eda_service_guard_v1",
            "adr_eda_service_guard_v2",
            "adr_eda_service_guard_v3",
            "adr_eda_service_guard_v4",
            "adr_eda_service_constrained_selector",
            "adr_eda_service_constrained_selector_floor50",
            "adr_eda_service_constrained_selector_floor60",
            "adr_eda_service_constrained_selector_floor65",
            "adr_eda_service_constrained_selector_floor70",
            "adr_eda_service_constrained_selector_floor75",
            "adr_eda_reservation_costaware_v1",
            "adr_eda_reservation_costaware_v2",
            "adr_eda_reservation_costaware_v3",
        ],
        help="Preset reproducible para DynamicEDA online sobre el stream real.",
    )
    p.add_argument("--policy-smooth-lambda", type=float, default=0.15)
    p.add_argument("--policy-smooth-metric", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--eda-warm-start-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eda-temporal-archive-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eda-temporal-archive-max-size", type=int, default=96)
    p.add_argument("--eda-temporal-archive-keep-per-step", type=int, default=12)
    p.add_argument("--eda-incumbent-gate-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eda-incumbent-min-improvement", type=float, default=0.01)
    p.add_argument("--eda-incumbent-neighbor-ratio", type=float, default=0.25)
    p.add_argument("--eda-incumbent-noise-scale", type=float, default=0.04)
    p.add_argument("--eda-archive-guided-ratio", type=float, default=0.15)
    p.add_argument("--eda-archive-guided-noise-scale", type=float, default=0.05)

    p.add_argument("--window-pack", type=str, default="extended", choices=["core", "extended", "extreme"])
    p.add_argument("--base-demand-scale", type=float, default=1.0)

    p.add_argument("--elite-factor", type=float, default=0.35)
    p.add_argument("--pm-family", type=str, default="SPBN", choices=["SPBN", "GBN", "KDEBN"])
    p.add_argument("--num-folds", type=int, default=3)
    p.add_argument("--archive-retention-after-change", type=float, default=0.30)
    p.add_argument("--with-reaction", action="store_true")
    p.add_argument("--adaptive-policy-enabled", action="store_true")
    p.add_argument("--adaptive-response-enabled", action="store_true")
    p.add_argument("--adaptive-response-diagnostics", action="store_true")
    p.add_argument("--adaptive-response-score-scale", type=float, default=0.34)
    p.add_argument("--adaptive-response-similarity-decay", type=float, default=1.0)
    p.add_argument("--reaction-quality-gate-enabled", action="store_true")
    p.add_argument("--local-search-enabled", action="store_true")
    p.add_argument(
        "--dynamic-change-detector-mode",
        type=str,
        default="sentinel",
        choices=["sentinel", "problem-aware"],
    )
    p.add_argument(
        "--dynamic-reopt-trigger",
        type=str,
        default="always",
        choices=["always", "event"],
    )
    p.add_argument("--dynamic-threshold-mild", type=float, default=0.20)
    p.add_argument("--dynamic-threshold-severe", type=float, default=0.45)
    p.add_argument("--dynamic-observation-interval-min", type=float, default=0.0)
    p.add_argument("--dynamic-regime-ewma-alpha", type=float, default=0.35)
    p.add_argument("--dynamic-min-reopt-interval-min", type=float, default=10.0)
    p.add_argument("--dynamic-max-reopt-interval-min", type=float, default=25.0)
    p.add_argument("--dynamic-hard-low-soc-share", type=float, default=0.30)
    p.add_argument("--dynamic-hard-charger-pressure", type=float, default=0.85)
    p.add_argument("--dynamic-hard-reservation-pressure", type=float, default=0.50)
    p.add_argument("--dynamic-hard-coverage-gap", type=float, default=0.30)

    p.add_argument("--n-sentinels", type=int, default=12)
    p.add_argument("--drift-alpha", type=float, default=0.03)
    p.add_argument("--lambda-ewma", type=float, default=0.30)
    p.add_argument("--sentinel-refresh", type=int, default=8)
    p.add_argument("--k-consecutive", type=int, default=2)

    p.add_argument("--react-strength", type=float, default=2.4)
    p.add_argument("--immigrant-ratio", type=float, default=0.24)
    p.add_argument("--react-strength-mild", type=float, default=1.4)
    p.add_argument("--immigrant-ratio-mild", type=float, default=0.12)

    p.add_argument("--local-search-neighbors", type=int, default=4)
    p.add_argument("--local-search-front-ratio", type=float, default=0.35)
    p.add_argument("--local-search-step", type=float, default=0.04)
    p.add_argument("--local-search-max-points", type=int, default=24)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_ops_cost_profile(args)
    _seed_optimizer_random_state(getattr(args, "optimizer_random_seed", -1))
    out_path = Path(args.output_json) if str(args.output_json).strip() else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = run_mode(args)
        payload = json.dumps(result, indent=2, ensure_ascii=False, default=_json_default)
        if out_path is not None:
            out_path.write_text(payload, encoding="utf-8")
            print("FF_RUNNER_OK")
            print({"mode": str(args.mode), "output_json": str(out_path)})
        else:
            print(payload)
        return 0
    except Exception as exc:
        payload = {
            "status": "skipped",
            "mode": str(args.mode),
            "reason": f"{type(exc).__name__}: {exc}",
        }
        if out_path is not None:
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("FF_RUNNER_SKIPPED")
        print(payload["reason"][:320])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
