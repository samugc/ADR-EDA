from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dynamics import (
    choose_user_nearest_vehicle,
    feasible_vehicle_set_for_request,
    refresh_system_state,
    schedule_trip_for_request,
)
from .ops_cost_profiles import apply_ops_cost_profile
from .planner import RobustDecisionEncoding, RollingHorizonPlanner
from .scenario import MadridFreeFloatingScenario, ScenarioConfig, default_travel_metrics_json
from .types import Request, RobustModelConfig, SystemState, TaskType, VehicleTask, VehicleStatus


@dataclass
class StreamEvent:
    sim_time_min: float
    priority: int
    seq: int
    kind: str
    payload: dict


def _fmt_clock(sim_time_min: float) -> str:
    t = max(0.0, float(sim_time_min))
    hh = int(t // 60.0) % 24
    mm = int(t % 60.0)
    ss = int(round((t - int(t)) * 60.0))
    if ss == 60:
        ss = 0
        mm = (mm + 1) % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _emit(
    sim_time_min: float,
    message: str,
    out_events: Optional[List[dict]] = None,
    event_kind: Optional[str] = None,
    payload: Optional[dict] = None,
) -> None:
    print(f"[{_fmt_clock(sim_time_min)}] {message}", flush=True)
    if out_events is not None and event_kind is not None:
        row = {"sim_time_min": float(sim_time_min), "event_type": event_kind}
        if payload:
            row.update(payload)
        out_events.append(row)


def _status_row(state: SystemState) -> dict:
    counts = {s.value: 0 for s in VehicleStatus}
    for v in state.vehicles.values():
        counts[v.status.value] += 1
    mean_soc = float(np.mean([v.soc for v in state.vehicles.values()])) if state.vehicles else 0.0
    mean_soh = float(np.mean([v.soh for v in state.vehicles.values()])) if state.vehicles else 0.0
    m = state.metrics
    return {
        "bookings": int(m.bookings_total),
        "requests": int(m.requests_total),
        "served": int(m.served_requests),
        "rejected": int(m.rejected_requests),
        "canceled": int(m.canceled_bookings),
        "service_rate": float(m.service_rate),
        "profit_eur": float(m.profit_eur),
        "mean_soc": mean_soc,
        "mean_soh": mean_soh,
        "idle": counts.get("idle", 0),
        "serving": counts.get("serving", 0),
        "charging": counts.get("charging", 0),
        "relocating": counts.get("relocating", 0),
        "unavailable": counts.get("unavailable", 0),
    }


def _resolve_dynamic_online_profile(args: argparse.Namespace) -> dict:
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
        "selective_simple_carsharing",
        "adr_eda_reservation_aware_v1",
        "adr_eda_reservation_aware_v2",
        "adr_eda_reservation_overload_aware_v1",
        "adr_eda_reservation_costaware_v1",
        "adr_eda_reservation_costaware_v2",
        "adr_eda_reservation_costaware_v3",
    }
    if profile not in valid:
        supported = ", ".join(sorted(valid))
        raise ValueError(f"--dynamic-online-profile='{profile}' no soportado. Usa: {supported}")

    resolved = {
        "profile": profile,
        "with_reaction": bool(getattr(args, "with_reaction", False)),
        "adaptive_policy_enabled": bool(getattr(args, "adaptive_policy_enabled", False)),
        "reaction_quality_gate_enabled": bool(
            getattr(args, "reaction_quality_gate_enabled", False)
        ),
        "dynamic_change_detector_mode": str(
            getattr(args, "dynamic_change_detector_mode", "sentinel")
        ),
        "dynamic_reopt_trigger": str(getattr(args, "dynamic_reopt_trigger", "always")),
        "dynamic_threshold_mild": float(getattr(args, "dynamic_threshold_mild", 0.20)),
        "dynamic_threshold_severe": float(getattr(args, "dynamic_threshold_severe", 0.45)),
        "dynamic_min_reopt_interval_min": float(
            getattr(args, "dynamic_min_reopt_interval_min", 10.0)
        ),
        "dynamic_max_reopt_interval_min": float(
            getattr(args, "dynamic_max_reopt_interval_min", 25.0)
        ),
        "react_strength": float(getattr(args, "react_strength", 2.4)),
        "immigrant_ratio": float(getattr(args, "immigrant_ratio", 0.24)),
        "react_strength_mild": float(getattr(args, "react_strength_mild", 1.4)),
        "immigrant_ratio_mild": float(getattr(args, "immigrant_ratio_mild", 0.12)),
        "selection_smooth_lambda": float(getattr(args, "policy_smooth_lambda", 0.15)),
        "selection_smooth_metric": str(getattr(args, "policy_smooth_metric", "l1")),
        "incumbent_gate_enabled": bool(getattr(args, "eda_incumbent_gate_enabled", True)),
        "incumbent_min_improvement": float(
            getattr(args, "eda_incumbent_min_improvement", 0.01)
        ),
    }

    if profile == "custom":
        return resolved

    profile_overrides = {
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
    resolved.update(profile_overrides[profile])
    return resolved


def _build_audit_summary(
    state: SystemState,
    pending_reservations: int,
    status_series: List[dict],
    args: argparse.Namespace,
    total_sim_min: float,
) -> dict:
    m = state.metrics
    final_status = _status_row(state)
    fleet_size = len(state.vehicles)
    soc_values = [float(v.soc) for v in state.vehicles.values()]
    soh_values = [float(v.soh) for v in state.vehicles.values()]

    req_balance_lhs = int(m.served_requests + m.rejected_requests)
    req_balance_rhs = int(m.requests_total)
    book_balance_lhs = int(m.requests_total + m.canceled_bookings + pending_reservations)
    book_balance_rhs = int(m.bookings_total)
    status_sum = int(
        final_status["idle"]
        + final_status["serving"]
        + final_status["charging"]
        + final_status["relocating"]
        + final_status["unavailable"]
    )

    checks = {
        "request_balance_ok": req_balance_lhs == req_balance_rhs,
        "request_balance_lhs_served_plus_rejected": req_balance_lhs,
        "request_balance_rhs_requests": req_balance_rhs,
        "booking_balance_ok": book_balance_lhs == book_balance_rhs,
        "booking_balance_lhs_requests_plus_canceled_plus_pending": book_balance_lhs,
        "booking_balance_rhs_bookings": book_balance_rhs,
        "vehicle_status_sum_ok": status_sum == fleet_size,
        "vehicle_status_sum": status_sum,
        "fleet_size": fleet_size,
        "soc_range_ok": (min(soc_values) >= -1e-9 and max(soc_values) <= 1.0 + 1e-9) if soc_values else True,
        "soc_min": float(min(soc_values)) if soc_values else 0.0,
        "soc_max": float(max(soc_values)) if soc_values else 0.0,
        "soh_range_ok": (min(soh_values) >= -1e-9 and max(soh_values) <= 1.0 + 1e-9) if soh_values else True,
        "soh_min": float(min(soh_values)) if soh_values else 0.0,
        "soh_max": float(max(soh_values)) if soh_values else 0.0,
        "pending_reservations_nonnegative": pending_reservations >= 0,
        "pending_reservations_count": int(pending_reservations),
    }

    series_brief = {}
    if status_series:
        for key in ("requests", "served", "rejected", "canceled", "profit_eur", "mean_soc", "mean_soh"):
            vals = [float(r[key]) for r in status_series]
            series_brief[key] = {
                "min": float(min(vals)),
                "max": float(max(vals)),
                "mean": float(np.mean(vals)),
                "last": float(vals[-1]),
            }

    return {
        "meta": {
            "mode": args.mode,
            "seed": int(args.seed),
            "decision_interval_min": int(args.decision_interval_min),
            "planning_horizon_min": int(args.planning_horizon_min),
            "sim_total_min": float(total_sim_min),
            "fleet_size": int(args.fleet_size),
            "charger_count": int(args.charger_count),
            "ops_resources": int(args.ops_resources),
            "base_demand_scale": float(args.base_demand_scale),
            "ops_cost_profile": str(getattr(args, "ops_cost_profile", "custom")),
            "relocation_cost_per_km": float(getattr(args, "relocation_cost_per_km", 0.22)),
            "ops_deadhead_cost_per_km": float(getattr(args, "ops_deadhead_cost_per_km", 0.22)),
            "ops_cost_per_action_eur": float(getattr(args, "ops_cost_per_action_eur", 1.20)),
            "energy_cost_per_kwh": float(getattr(args, "energy_cost_per_kwh", 0.23)),
            "simple_access_model": bool(args.simple_access_model),
            "min_same_zone_trip_km": float(args.min_same_zone_trip_km),
        },
        "final_status": final_status,
        "metrics": {
            "bookings_total": int(m.bookings_total),
            "requests_total": int(m.requests_total),
            "served_requests": int(m.served_requests),
            "rejected_requests": int(m.rejected_requests),
            "canceled_bookings": int(m.canceled_bookings),
            "preassigned_bookings": int(m.preassigned_bookings),
            "preassigned_pickups_honored": int(m.preassigned_pickups_honored),
            "preassigned_pickups_failed": int(m.preassigned_pickups_failed),
            "service_rate": float(m.service_rate),
            "profit_eur": float(m.profit_eur),
            "revenue_eur": float(m.revenue_eur),
            "trip_fare_revenue_eur": float(m.trip_fare_revenue_eur),
            "airport_fee_revenue_eur": float(m.airport_fee_revenue_eur),
            "ops_charge_credit_eur": float(m.ops_charge_credit_eur),
            "relocation_cost_eur": float(m.relocation_cost_eur),
            "energy_cost_eur": float(m.energy_cost_eur),
            "ops_cost_eur": float(m.ops_cost_eur),
            "ops_deadhead_cost_eur": float(m.ops_deadhead_cost_eur),
            "charged_kwh_total": float(m.charged_kwh_total),
            "drive_km_total": float(m.drive_km_total),
            "ops_deadhead_km_total": float(m.ops_deadhead_km_total),
            "soh_loss_total": float(m.soh_loss_total),
        },
        "checks": checks,
        "status_series_brief": series_brief,
    }


def _write_audit_outputs(prefix: str, summary: dict, status_series: List[dict]) -> None:
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(str(base) + "_summary.json")
    status_path = Path(str(base) + "_status.csv")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if status_series:
        fieldnames = list(status_series[0].keys())
        with status_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(status_series)
    else:
        status_path.write_text("", encoding="utf-8")

    print(f"Audit summary: {summary_path}", flush=True)
    print(f"Audit status:  {status_path}", flush=True)


def _collect_task_end_events(
    state: SystemState,
    start_min: float,
    end_min: float,
    seq0: int,
) -> Tuple[List[StreamEvent], int]:
    out: List[StreamEvent] = []
    seq = int(seq0)

    for v in state.vehicles.values():
        task = v.current_task
        while task is not None:
            if start_min < task.end_min <= end_min:
                payload = {
                    "vehicle_id": v.vehicle_id,
                    "task_type": task.task_type.value,
                    "request_id": task.request_id,
                    "origin_zone_id": task.origin_zone_id,
                    "destination_zone_id": task.destination_zone_id,
                    "charger_id": task.charger_id,
                }
                out.append(
                    StreamEvent(
                        sim_time_min=float(task.end_min),
                        priority=0,  # completion first at same timestamp
                        seq=seq,
                        kind="task_end",
                        payload=payload,
                    )
                )
                seq += 1
            task = task.next_task

    return out, seq


def _push_event(heap: List[tuple], ev: StreamEvent) -> None:
    heapq.heappush(heap, (ev.sim_time_min, ev.priority, ev.seq, ev.kind, ev.payload))


def _choose_with_exclusions(
    request: Request,
    state: SystemState,
    cfg: RobustModelConfig,
    excluded_vehicle_ids: set[str],
):
    candidates = [
        (v, walk)
        for v, walk in feasible_vehicle_set_for_request(request=request, state=state, cfg=cfg)
        if v.vehicle_id not in excluded_vehicle_ids
    ]
    if not candidates:
        return choose_user_nearest_vehicle(request=request, state=state, cfg=cfg)
    vehicle, walk_min = min(candidates, key=lambda x: x[1])
    return vehicle, float(walk_min)


def _serve_pickup(
    req: Request,
    event_time_min: float,
    window_end_min: float,
    state: SystemState,
    cfg: RobustModelConfig,
    heap: List[tuple],
    seq: int,
    out_events: Optional[List[dict]],
) -> int:
    req_pick = copy.copy(req)
    if req_pick.pickup_time_min is None:
        req_pick.pickup_time_min = float(event_time_min)

    state.metrics.requests_total += 1
    refresh_system_state(state=state, now_min=event_time_min, cfg=cfg)

    pickup_clock = (
        float(req_pick.pickup_time_min)
        if req_pick.pickup_time_min is not None
        else float(req_pick.request_time_min)
    )
    reserved_vehicle_ids = {
        str(r.preassigned_vehicle_id)
        for r in state.pending_reservations.values()
        if r.preassigned_vehicle_id is not None
        and (r.pickup_time_min if r.pickup_time_min is not None else r.request_time_min) >= pickup_clock - 1e-9
        and r.request_id != req_pick.request_id
    }

    vehicle = None
    walk_min = 0.0
    if req_pick.preassigned_vehicle_id:
        preassigned_id = str(req_pick.preassigned_vehicle_id)
        for v, w in feasible_vehicle_set_for_request(request=req_pick, state=state, cfg=cfg):
            if v.vehicle_id == preassigned_id:
                vehicle, walk_min = v, float(w)
                state.metrics.preassigned_pickups_honored += 1
                break
        if vehicle is None:
            state.metrics.preassigned_pickups_failed += 1

    if vehicle is None:
        vehicle, walk_min = _choose_with_exclusions(
            request=req_pick,
            state=state,
            cfg=cfg,
            excluded_vehicle_ids=reserved_vehicle_ids,
        )
    if vehicle is None:
        state.metrics.rejected_requests += 1
        _emit(
            event_time_min,
            f"REQUEST_REJECTED id={req_pick.request_id} reason=no_feasible_vehicle",
            out_events,
            "request_rejected",
            {
                "request_id": req_pick.request_id,
                "origin_zone_id": req_pick.origin_zone_id,
                "destination_zone_id": req_pick.destination_zone_id,
            },
        )
        return seq

    soc_before = float(vehicle.soc)
    schedule_trip_for_request(request=req_pick, vehicle=vehicle, walk_min=walk_min, state=state)
    trip = vehicle.current_task
    _emit(
        event_time_min,
        (
            f"REQUEST_ASSIGNED id={req_pick.request_id} veh={vehicle.vehicle_id} "
            f"{req_pick.origin_zone_id}->{req_pick.destination_zone_id} "
            f"walk={walk_min:.1f}min soc={soc_before:.3f} eta_end={_fmt_clock(trip.end_min)}"
        ),
        out_events,
        "request_assigned",
        {
            "request_id": req_pick.request_id,
            "vehicle_id": vehicle.vehicle_id,
            "origin_zone_id": req_pick.origin_zone_id,
            "destination_zone_id": req_pick.destination_zone_id,
            "walk_min": float(walk_min),
            "soc_before": soc_before,
            "eta_trip_end_min": float(trip.end_min),
        },
    )
    _emit(
        float(trip.start_min),
        (
            f"TRIP_STARTED req={req_pick.request_id} veh={vehicle.vehicle_id} "
            f"from={trip.origin_zone_id} to={trip.destination_zone_id}"
        ),
        out_events,
        "trip_started",
        {
            "request_id": req_pick.request_id,
            "vehicle_id": vehicle.vehicle_id,
            "origin_zone_id": trip.origin_zone_id,
            "destination_zone_id": trip.destination_zone_id,
            "start_min": float(trip.start_min),
            "end_min": float(trip.end_min),
        },
    )

    # Emit completion in stream only if it happens inside this epoch.
    if trip.end_min <= window_end_min + 1e-9:
        _push_event(
            heap,
            StreamEvent(
                sim_time_min=float(trip.end_min),
                priority=0,
                seq=seq,
                kind="task_end",
                payload={
                    "vehicle_id": vehicle.vehicle_id,
                    "task_type": TaskType.TRIP.value,
                    "request_id": req_pick.request_id,
                    "origin_zone_id": req_pick.origin_zone_id,
                    "destination_zone_id": req_pick.destination_zone_id,
                    "charger_id": None,
                },
            ),
        )
        return seq + 1
    return seq


def _build_default_policy_vector(encoding: RobustDecisionEncoding) -> np.ndarray:
    return np.mean(encoding.bounds, axis=1)


def _policy_to_dict(policy) -> dict:
    data = asdict(policy)
    def _convert(value):
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in sorted(value.items())}
        return float(value)

    return {str(k): _convert(v) for k, v in data.items()}


def _best_by_sum_index(front: np.ndarray) -> Optional[int]:
    if front.ndim != 2 or front.shape[0] == 0:
        return None
    f = np.asarray(front, dtype=float)
    lo = np.min(f, axis=0)
    hi = np.max(f, axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    f_norm = (f - lo) / span
    return int(np.argmax(np.sum(f_norm, axis=1)))


def _normalized_front_scores(front: np.ndarray) -> np.ndarray:
    if front.ndim != 2 or front.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    f = np.asarray(front, dtype=float)
    lo = np.min(f, axis=0)
    hi = np.max(f, axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return (f - lo) / span


def _normalized_front_mean_scores(front: np.ndarray) -> np.ndarray:
    scores = _normalized_front_scores(front)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return np.empty((0,), dtype=float)
    return np.mean(scores, axis=1)


def _normalized_policy_shift(
    pset: np.ndarray,
    reference_vector: Optional[np.ndarray],
    bounds: np.ndarray,
    metric: str,
) -> np.ndarray:
    if pset.ndim != 2 or pset.shape[0] == 0:
        return np.empty((0,), dtype=float)
    if reference_vector is None:
        return np.zeros(pset.shape[0], dtype=float)

    lower = np.asarray(bounds[:, 0], dtype=float)
    upper = np.asarray(bounds[:, 1], dtype=float)
    span = np.where((upper - lower) > 1e-12, upper - lower, 1.0)
    ref = np.clip(np.asarray(reference_vector, dtype=float), lower, upper)
    delta = (np.asarray(pset, dtype=float) - ref.reshape(1, -1)) / span.reshape(1, -1)

    if str(metric).strip().lower() == "l2":
        return np.sqrt(np.mean(np.square(delta), axis=1))
    return np.mean(np.abs(delta), axis=1)


def _dedup_rows(
    rows: List[np.ndarray],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    max_rows: int,
) -> Optional[np.ndarray]:
    if not rows:
        return None

    dedup: List[np.ndarray] = []
    seen = set()
    for row in rows:
        clipped = np.clip(np.asarray(row, dtype=float), lower, upper)
        key = tuple(np.round(clipped, 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(clipped)
        if len(dedup) >= int(max_rows):
            break
    if not dedup:
        return None
    return np.vstack(dedup)


def _select_archive_rows(
    *,
    front: np.ndarray,
    pset: np.ndarray,
    max_keep: int,
) -> Optional[np.ndarray]:
    pareto = np.asarray(pset, dtype=float)
    objs = np.asarray(front, dtype=float)
    if pareto.ndim != 2 or pareto.shape[0] == 0:
        return None
    keep = max(1, int(max_keep))
    if objs.ndim != 2 or objs.shape[0] != pareto.shape[0]:
        return pareto[:keep]

    ranking = np.argsort(-_normalized_front_mean_scores(objs))
    chosen = [pareto[int(i)] for i in ranking[:keep]]
    return np.vstack(chosen) if chosen else None


def _update_temporal_archive(
    *,
    archive_pool: Optional[np.ndarray],
    front: np.ndarray,
    pset: np.ndarray,
    max_size: int,
    keep_per_step: int,
    encoding: RobustDecisionEncoding,
) -> Optional[np.ndarray]:
    new_rows = _select_archive_rows(front=front, pset=pset, max_keep=int(keep_per_step))
    lower = np.asarray(encoding.bounds[:, 0], dtype=float)
    upper = np.asarray(encoding.bounds[:, 1], dtype=float)

    rows: List[np.ndarray] = []
    if new_rows is not None:
        rows.extend([np.asarray(new_rows[i], dtype=float) for i in range(new_rows.shape[0])])
    if archive_pool is not None:
        old = np.asarray(archive_pool, dtype=float)
        if old.ndim == 2 and old.shape[1] == int(encoding.dimension) and old.shape[0] > 0:
            rows.extend([old[i] for i in range(old.shape[0])])
    return _dedup_rows(rows, lower=lower, upper=upper, max_rows=int(max_size))


def _build_warm_start_pool(
    *,
    encoding: RobustDecisionEncoding,
    incumbent_vector: Optional[np.ndarray],
    transfer_pool: Optional[np.ndarray],
    archive_pool: Optional[np.ndarray],
    size_gen: int,
    step_idx: int,
    seed: int,
    incumbent_neighbor_ratio: float,
    incumbent_noise_scale: float,
    archive_guided_ratio: float,
    archive_guided_noise_scale: float,
) -> tuple[Optional[np.ndarray], dict]:
    rows: List[np.ndarray] = []
    dim = int(encoding.dimension)
    lower = np.asarray(encoding.bounds[:, 0], dtype=float)
    upper = np.asarray(encoding.bounds[:, 1], dtype=float)
    span = np.where((upper - lower) > 1e-12, upper - lower, 1.0)
    meta = {
        "transfer_rows_used": 0,
        "archive_rows_used": 0,
        "incumbent_neighbors_used": 0,
        "archive_guided_immigrants_used": 0,
    }

    if transfer_pool is not None:
        pool = np.asarray(transfer_pool, dtype=float)
        if pool.ndim == 2 and pool.shape[1] == dim and pool.shape[0] > 0:
            clipped = [np.clip(pool[i], lower, upper) for i in range(pool.shape[0])]
            rows.extend(clipped)
            meta["transfer_rows_used"] = len(clipped)

    if bool(getattr(encoding, "dimension", dim)) and archive_pool is not None:
        pool = np.asarray(archive_pool, dtype=float)
        if pool.ndim == 2 and pool.shape[1] == dim and pool.shape[0] > 0:
            clipped = [np.clip(pool[i], lower, upper) for i in range(pool.shape[0])]
            rows.extend(clipped)
            meta["archive_rows_used"] = len(clipped)

    incumbent = None
    if incumbent_vector is not None:
        incumbent = np.clip(np.asarray(incumbent_vector, dtype=float), lower, upper)
        rows.insert(0, incumbent)

    if incumbent is not None:
        rng = np.random.default_rng(int(seed) + 7000 + int(step_idx))
        n_neighbors = max(2, int(round(float(size_gen) * max(0.0, float(incumbent_neighbor_ratio)))))
        noise = rng.normal(
            loc=0.0,
            scale=float(max(1e-6, incumbent_noise_scale)),
            size=(n_neighbors, dim),
        )
        neighbors = np.clip(incumbent.reshape(1, -1) + noise * span.reshape(1, -1), lower, upper)
        rows.extend([neighbors[i] for i in range(neighbors.shape[0])])
        meta["incumbent_neighbors_used"] = int(neighbors.shape[0])

    if archive_pool is not None:
        pool = np.asarray(archive_pool, dtype=float)
        if pool.ndim == 2 and pool.shape[1] == dim and pool.shape[0] > 0:
            ratio = float(max(0.0, archive_guided_ratio))
            noise_scale = float(max(1e-6, archive_guided_noise_scale))
            n_guided = int(round(float(size_gen) * ratio))
            if n_guided > 0:
                rng = np.random.default_rng(int(seed) + 11000 + int(step_idx))
                center_idx = rng.integers(0, pool.shape[0], size=n_guided)
                centers = np.asarray([pool[int(i)] for i in center_idx], dtype=float)
                noise = rng.normal(loc=0.0, scale=noise_scale, size=(n_guided, dim))
                guided = np.clip(centers + noise * span.reshape(1, -1), lower, upper)
                rows.extend([guided[i] for i in range(guided.shape[0])])
                meta["archive_guided_immigrants_used"] = int(guided.shape[0])

    if not rows:
        return None, meta

    init = _dedup_rows(rows, lower=lower, upper=upper, max_rows=int(size_gen))
    return init, meta


def _select_front_index(
    *,
    front: np.ndarray,
    pset: np.ndarray,
    reference_vector: Optional[np.ndarray],
    bounds: np.ndarray,
    smooth_lambda: float,
    smooth_metric: str,
) -> dict:
    if front.ndim != 2 or front.shape[0] == 0:
        return {
            "selected_index": None,
            "objective_score": None,
            "shift_penalty": None,
            "total_score": None,
            "selection_strategy": "empty_front",
        }

    objective_score = np.mean(_normalized_front_scores(front), axis=1)
    shift_penalty = _normalized_policy_shift(pset, reference_vector, bounds, smooth_metric)
    lam = max(0.0, float(smooth_lambda))

    if reference_vector is None or lam <= 1e-12:
        idx = int(np.argmax(objective_score))
        return {
            "selected_index": idx,
            "objective_score": float(objective_score[idx]),
            "shift_penalty": float(shift_penalty[idx]) if shift_penalty.size else 0.0,
            "total_score": float(objective_score[idx]),
            "selection_strategy": "normalized_front_mean",
        }

    total_score = objective_score - lam * shift_penalty
    idx = int(np.argmax(total_score))
    return {
        "selected_index": idx,
        "objective_score": float(objective_score[idx]),
        "shift_penalty": float(shift_penalty[idx]),
        "total_score": float(total_score[idx]),
        "selection_strategy": "normalized_front_mean_minus_policy_shift",
    }


def _candidate_eda_configs(
    requested_family: str,
    requested_size_gen: int,
    requested_w_noise: float,
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
    fam_candidates = [str(requested_family), "KDEBN", "SPBN"]
    seen = set(configs)
    for fam in fam_candidates:
        for pop in pop_candidates:
            for noi in noise_candidates:
                key = (str(fam), int(pop), float(noi))
                if key not in seen:
                    seen.add(key)
                    configs.append(key)
    return configs


def _eda_reoptimize_policy(
    *,
    state: SystemState,
    scenario: MadridFreeFloatingScenario,
    encoding: RobustDecisionEncoding,
    cfg: RobustModelConfig,
    seed: int,
    step_idx: int,
    args: argparse.Namespace,
    out_events: Optional[List[dict]],
    reference_vector: Optional[np.ndarray],
    transfer_pool: Optional[np.ndarray],
    archive_pool: Optional[np.ndarray],
    incumbent_vector: Optional[np.ndarray],
) -> dict:
    # Lazy import so normal stream mode does not require EDAspy stack.
    from .evaluator import FreeFloatingRobustDMOOP
    from .edaspy_bridge import SPEDARunConfig, SPEDAStaticRunner

    alpha_eff = max(0.10, float(args.eda_alpha))
    max_iter_eff = max(2, int(args.eda_max_iter))
    dead_iter_eff = min(max_iter_eff - 1, max(1, int(args.eda_dead_iter)))
    # EDAspy backend uses 10 folds internally; keep enough samples in elite pool.
    min_size_for_folds = int(np.ceil(12.0 / max(1e-9, alpha_eff)))
    size_gen_eff = max(int(args.eda_size_gen), min_size_for_folds)
    if size_gen_eff != int(args.eda_size_gen):
        _emit(
            float(state.current_time_min),
            (
                f"EDA_PARAM_ADJUST size_gen {int(args.eda_size_gen)}->"
                f"{size_gen_eff} (alpha={alpha_eff:.2f})"
            ),
            out_events,
            "eda_param_adjust",
            {
                "old_size_gen": int(args.eda_size_gen),
                "new_size_gen": int(size_gen_eff),
                "alpha": float(alpha_eff),
            },
        )

    candidate_cfgs = _candidate_eda_configs(
        requested_family=str(args.eda_optimizer_family),
        requested_size_gen=int(size_gen_eff),
        requested_w_noise=float(args.eda_w_noise),
        auto_fallback=bool(getattr(args, "eda_auto_fallback", True)),
    )
    if bool(getattr(args, "eda_warm_start_enabled", True)):
        init_data, warm_meta = _build_warm_start_pool(
            encoding=encoding,
            incumbent_vector=incumbent_vector,
            transfer_pool=transfer_pool,
            archive_pool=archive_pool,
            size_gen=int(size_gen_eff),
            step_idx=int(step_idx),
            seed=int(seed),
            incumbent_neighbor_ratio=float(getattr(args, "eda_incumbent_neighbor_ratio", 0.25)),
            incumbent_noise_scale=float(getattr(args, "eda_incumbent_noise_scale", 0.04)),
            archive_guided_ratio=float(getattr(args, "eda_archive_guided_ratio", 0.15)),
            archive_guided_noise_scale=float(getattr(args, "eda_archive_guided_noise_scale", 0.05)),
        )
    else:
        init_data = None
        warm_meta = {
            "transfer_rows_used": 0,
            "archive_rows_used": 0,
            "incumbent_neighbors_used": 0,
            "archive_guided_immigrants_used": 0,
        }
    raw = None
    used_family = None
    used_size_gen = None
    used_w_noise = None
    errors: List[str] = []
    for family, size_gen, w_noise in candidate_cfgs:
        try:
            problem = FreeFloatingRobustDMOOP(
                state_template=copy.deepcopy(state),
                scenario=scenario,
                encoding=encoding,
                config=cfg,
                horizon_hours=int(args.eda_horizon_hours),
                random_seed=int(seed + 10000 + step_idx),
            )
            runner = SPEDAStaticRunner(
                encoding=encoding,
                config=SPEDARunConfig(
                    size_gen=int(size_gen),
                    max_iter=int(max_iter_eff),
                    dead_iter=int(dead_iter_eff),
                    alpha=float(alpha_eff),
                    w_noise=float(w_noise),
                    parallelize=bool(args.eda_parallelize),
                    disp=False,
                    optimizer_family=str(family),
                ),
            )
            raw = runner.optimize(
                problem=problem,
                init_data=init_data,
                max_iter=int(max_iter_eff),
            )
            used_family = str(family)
            used_size_gen = int(size_gen)
            used_w_noise = float(w_noise)
            break
        except Exception as exc:
            errors.append(f"{family}/N={size_gen}/noise={w_noise:.2f}: {type(exc).__name__}: {exc}")

    if raw is None:
        raise RuntimeError(" | ".join(errors))

    front = np.asarray(raw.get("pareto_front_max", np.empty((0, 3))), dtype=float)
    pset = np.asarray(raw.get("pareto_set", np.empty((0, encoding.dimension))), dtype=float)
    selection = _select_front_index(
        front=front,
        pset=pset,
        reference_vector=reference_vector,
        bounds=encoding.bounds,
        smooth_lambda=float(getattr(args, "policy_smooth_lambda", 0.0)),
        smooth_metric=str(getattr(args, "policy_smooth_metric", "l1")),
    )
    idx = selection["selected_index"]
    incumbent_score = None
    candidate_score = None
    incumbent_gate_applied = False
    frontier_selected_index = idx
    if used_family != str(args.eda_optimizer_family) or used_size_gen != int(size_gen_eff) or abs(float(used_w_noise) - float(args.eda_w_noise)) > 1e-12:
        _emit(
            float(state.current_time_min),
            (
                f"EDA_FALLBACK_USED family={used_family} size_gen={used_size_gen} "
                f"w_noise={float(used_w_noise):.2f}"
            ),
            out_events,
            "eda_fallback_used",
            {
                "family_used": str(used_family),
                "size_gen_used": int(used_size_gen or size_gen_eff),
                "w_noise_used": float(used_w_noise or args.eda_w_noise),
            },
        )
    if idx is None or idx < 0 or idx >= pset.shape[0]:
        return {
            "selected_vector": None,
            "selected_policy": None,
            "selected_objectives": None,
            "selected_index": None,
            "selection_strategy": selection["selection_strategy"],
            "selection_objective_score": selection["objective_score"],
            "selection_shift_penalty": selection["shift_penalty"],
            "selection_total_score": selection["total_score"],
            "pareto_size": int(front.shape[0]),
            "pareto_set": pset,
            "pareto_front_max": front,
            "optimizer_family_used": str(used_family or args.eda_optimizer_family),
            "size_gen_used": int(used_size_gen or size_gen_eff),
            "w_noise_used": float(used_w_noise or args.eda_w_noise),
            "warm_start_pool_size": 0 if init_data is None else int(np.asarray(init_data).shape[0]),
            "transfer_rows_used": int(warm_meta["transfer_rows_used"]),
            "archive_rows_used": int(warm_meta["archive_rows_used"]),
            "incumbent_neighbors_used": int(warm_meta["incumbent_neighbors_used"]),
            "archive_guided_immigrants_used": int(warm_meta["archive_guided_immigrants_used"]),
            "incumbent_score": incumbent_score,
            "candidate_score": candidate_score,
            "incumbent_gate_applied": bool(incumbent_gate_applied),
            "frontier_selected_index": frontier_selected_index,
            "fallback_errors": list(errors),
        }
    x_sel = np.asarray(pset[idx], dtype=float)
    obj_sel = np.asarray(front[idx], dtype=float)
    candidate_score = float(np.mean(np.asarray(obj_sel, dtype=float)))

    if bool(getattr(args, "eda_incumbent_gate_enabled", True)) and incumbent_vector is not None:
        incumbent_obj, _, _ = problem.rollout(x=np.asarray(incumbent_vector, dtype=float))
        incumbent_score = float(np.mean(np.asarray(incumbent_obj, dtype=float)))
        min_gain = float(max(0.0, getattr(args, "eda_incumbent_min_improvement", 0.01)))
        if candidate_score < incumbent_score + min_gain:
            incumbent_gate_applied = True
            x_sel = np.asarray(incumbent_vector, dtype=float)
            obj_sel = np.asarray(incumbent_obj, dtype=float)
            idx = None

    event_kind = "eda_reopt"
    action_label = "update"
    selection_strategy = str(selection["selection_strategy"])
    if incumbent_gate_applied:
        event_kind = "eda_reopt_hold"
        action_label = "hold"
        selection_strategy = selection_strategy + "_with_incumbent_gate"
    _emit(
        float(state.current_time_min),
        (
            f"EDA_REOPT step={step_idx} action={action_label} pareto={int(front.shape[0])} "
            f"sel=[{obj_sel[0]:.3f},{obj_sel[1]:.1f},{obj_sel[2]:.3f}]"
        ),
        out_events,
        event_kind,
        {
            "step_idx": int(step_idx),
            "pareto_size": int(front.shape[0]),
            "sel_f1": float(obj_sel[0]),
            "sel_f2": float(obj_sel[1]),
            "sel_f3": float(obj_sel[2]),
            "selection_strategy": selection_strategy,
            "selection_score": float(selection["total_score"] or 0.0),
            "selection_shift_penalty": float(selection["shift_penalty"] or 0.0),
            "incumbent_score": None if incumbent_score is None else float(incumbent_score),
            "candidate_score": None if candidate_score is None else float(candidate_score),
            "incumbent_gate_applied": bool(incumbent_gate_applied),
            "warm_start_pool_size": 0 if init_data is None else int(np.asarray(init_data).shape[0]),
            "transfer_rows_used": int(warm_meta["transfer_rows_used"]),
            "archive_rows_used": int(warm_meta["archive_rows_used"]),
            "incumbent_neighbors_used": int(warm_meta["incumbent_neighbors_used"]),
            "archive_guided_immigrants_used": int(warm_meta["archive_guided_immigrants_used"]),
        },
    )
    return {
        "selected_vector": x_sel,
        "selected_policy": _policy_to_dict(encoding.decode(x_sel)),
        "selected_objectives": obj_sel,
        "selected_index": None if idx is None else int(idx),
        "selection_strategy": selection_strategy,
        "selection_objective_score": selection["objective_score"],
        "selection_shift_penalty": selection["shift_penalty"],
        "selection_total_score": selection["total_score"],
        "pareto_size": int(front.shape[0]),
        "pareto_set": pset,
        "pareto_front_max": front,
        "optimizer_family_used": str(used_family or args.eda_optimizer_family),
        "size_gen_used": int(used_size_gen or size_gen_eff),
        "w_noise_used": float(used_w_noise or args.eda_w_noise),
        "warm_start_pool_size": 0 if init_data is None else int(np.asarray(init_data).shape[0]),
        "transfer_rows_used": int(warm_meta["transfer_rows_used"]),
        "archive_rows_used": int(warm_meta["archive_rows_used"]),
        "incumbent_neighbors_used": int(warm_meta["incumbent_neighbors_used"]),
        "archive_guided_immigrants_used": int(warm_meta["archive_guided_immigrants_used"]),
        "incumbent_score": incumbent_score,
        "candidate_score": candidate_score,
        "incumbent_gate_applied": bool(incumbent_gate_applied),
        "frontier_selected_index": frontier_selected_index,
        "fallback_errors": list(errors),
    }


def _dynamic_reoptimize_policy(
    *,
    state: SystemState,
    scenario: MadridFreeFloatingScenario,
    encoding: RobustDecisionEncoding,
    cfg: RobustModelConfig,
    seed: int,
    step_idx: int,
    args: argparse.Namespace,
    out_events: Optional[List[dict]],
    runner_state: dict,
    recent_service_signal: Optional[dict] = None,
) -> dict:
    from .evaluator import FreeFloatingRobustDMOOP
    from .edaspy_dynamic_bridge import DynamicSPEDARunConfig, OnlineStatefulDynamicSPEDA

    dynamic_profile = _resolve_dynamic_online_profile(args)
    alpha_eff = max(0.10, float(args.eda_alpha))
    max_iter_eff = max(2, int(args.eda_max_iter))
    dead_iter_eff = min(max_iter_eff - 1, max(1, int(args.eda_dead_iter)))
    min_size_for_folds = int(np.ceil(12.0 / max(1e-9, alpha_eff)))
    size_gen_eff = max(int(args.eda_size_gen), min_size_for_folds)
    if size_gen_eff != int(args.eda_size_gen):
        _emit(
            float(state.current_time_min),
            (
                f"DYNAMIC_EDA_PARAM_ADJUST size_gen {int(args.eda_size_gen)}->"
                f"{size_gen_eff} (alpha={alpha_eff:.2f})"
            ),
            out_events,
            "eda_param_adjust",
            {
                "old_size_gen": int(args.eda_size_gen),
                "new_size_gen": int(size_gen_eff),
                "alpha": float(alpha_eff),
                "engine": "dynamic-stateful",
            },
        )

    runner = runner_state.get("runner")
    if runner is None:
        candidate_cfgs = _candidate_eda_configs(
            requested_family=str(args.pm_family),
            requested_size_gen=int(size_gen_eff),
            requested_w_noise=float(args.eda_w_noise),
            auto_fallback=bool(getattr(args, "eda_auto_fallback", True)),
        )
        init_errors: List[str] = []
        for family, size_gen, w_noise in candidate_cfgs:
            try:
                dyn_cfg = DynamicSPEDARunConfig(
                    size_gen=int(size_gen),
                    max_iter=int(max_iter_eff),
                    dead_iter=int(dead_iter_eff),
                    alpha=float(alpha_eff),
                    elite_factor=float(getattr(args, "elite_factor", 0.35)),
                    archive_generations=float(getattr(args, "archive_generations", 4.0)),
                    w_noise=float(w_noise),
                    parallelize=bool(args.eda_parallelize),
                    disp=False,
                    pm_family=str(family),
                    num_folds=int(getattr(args, "num_folds", 3)),
                    archive_retention_after_change=float(
                        getattr(args, "archive_retention_after_change", 0.30)
                    ),
                    incumbent_gate_enabled=bool(
                        dynamic_profile.get(
                            "incumbent_gate_enabled",
                            getattr(args, "eda_incumbent_gate_enabled", True),
                        )
                    ),
                    incumbent_min_improvement=float(
                        dynamic_profile.get(
                            "incumbent_min_improvement",
                            getattr(args, "eda_incumbent_min_improvement", 0.01),
                        )
                    ),
                    with_reaction=bool(dynamic_profile["with_reaction"]),
                    change_detector_mode=str(dynamic_profile["dynamic_change_detector_mode"]),
                    reopt_trigger_mode=str(dynamic_profile["dynamic_reopt_trigger"]),
                    detector={
                        "n_sentinels": int(getattr(args, "n_sentinels", 12)),
                        "drift_alpha": float(getattr(args, "drift_alpha", 0.03)),
                        "lambda_ewma": float(getattr(args, "lambda_ewma", 0.30)),
                        "sentinel_refresh": int(getattr(args, "sentinel_refresh", 8)),
                        "k_consecutive": int(getattr(args, "k_consecutive", 2)),
                        "split_drift_signals": bool(dynamic_profile.get("split_drift_signals", False)),
                        "adaptive_lambda_enabled": bool(
                            dynamic_profile.get("adaptive_lambda_enabled", False)
                        ),
                        "adaptive_lambda_cov_enabled": bool(
                            dynamic_profile.get("adaptive_lambda_cov_enabled", False)
                        ),
                        "adaptive_cooldown_enabled": bool(
                            dynamic_profile.get("adaptive_cooldown_enabled", False)
                        ),
                    },
                    reaction={
                        "react_strength": float(dynamic_profile["react_strength"]),
                        "immigrant_ratio": float(dynamic_profile["immigrant_ratio"]),
                        "react_strength_mild": float(dynamic_profile["react_strength_mild"]),
                        "immigrant_ratio_mild": float(dynamic_profile["immigrant_ratio_mild"]),
                        "adaptive_policy_enabled": bool(dynamic_profile["adaptive_policy_enabled"]),
                        "reaction_quality_gate_enabled": bool(
                            dynamic_profile["reaction_quality_gate_enabled"]
                        ),
                        "reaction_quality_gate_mo_enabled": bool(
                            dynamic_profile.get("reaction_quality_gate_mo_enabled", False)
                        ),
                        "diversity_block_enabled": bool(
                            dynamic_profile.get("diversity_block_enabled", True)
                        ),
                        "predictive_block_enabled": bool(
                            dynamic_profile.get("predictive_block_enabled", False)
                        ),
                        "continuity_block_enabled": bool(
                            dynamic_profile.get("continuity_block_enabled", True)
                        ),
                        "adr_front_ls_enabled": bool(
                            dynamic_profile.get("adr_front_ls_enabled", False)
                        ),
                        "adr_front_ls_context_gate_enabled": bool(
                            dynamic_profile.get("adr_front_ls_context_gate_enabled", False)
                        ),
                        "change_type_enabled": bool(
                            dynamic_profile.get("change_type_enabled", False)
                        ),
                        "adr_enabled": bool(dynamic_profile.get("adr_enabled", False)),
                    },
                    local_search={
                        "local_search_enabled": bool(
                            dynamic_profile.get("local_search_enabled", getattr(args, "local_search_enabled", False))
                        ),
                        "local_search_neighbors": int(
                            dynamic_profile.get("local_search_neighbors", getattr(args, "local_search_neighbors", 4))
                        ),
                        "local_search_front_ratio": float(
                            dynamic_profile.get(
                                "local_search_front_ratio", getattr(args, "local_search_front_ratio", 0.35)
                            )
                        ),
                        "local_search_step": float(
                            dynamic_profile.get("local_search_step", getattr(args, "local_search_step", 0.04))
                        ),
                        "local_search_max_points": int(
                            dynamic_profile.get("local_search_max_points", getattr(args, "local_search_max_points", 24))
                        ),
                    },
                    problem_detector={
                        "threshold_mild": float(dynamic_profile["dynamic_threshold_mild"]),
                        "threshold_severe": float(dynamic_profile["dynamic_threshold_severe"]),
                        "reference_ewma_alpha": float(
                            getattr(args, "dynamic_regime_ewma_alpha", 0.35)
                        ),
                        "min_reopt_interval_min": float(
                            dynamic_profile["dynamic_min_reopt_interval_min"]
                        ),
                        "max_reopt_interval_min": float(
                            dynamic_profile["dynamic_max_reopt_interval_min"]
                        ),
                        "hard_low_soc_share": float(
                            getattr(args, "dynamic_hard_low_soc_share", 0.30)
                        ),
                        "hard_charger_pressure": float(
                            getattr(args, "dynamic_hard_charger_pressure", 0.85)
                        ),
                        "hard_reservation_pressure": float(
                            getattr(args, "dynamic_hard_reservation_pressure", 0.50)
                        ),
                        "hard_coverage_gap": float(
                            getattr(args, "dynamic_hard_coverage_gap", 0.30)
                        ),
                    },
                    reservation_guard=dict(dynamic_profile.get("reservation_guard", {})),
                    selection_smooth_lambda=float(dynamic_profile["selection_smooth_lambda"]),
                    selection_smooth_metric=str(dynamic_profile["selection_smooth_metric"]),
                )
                runner = OnlineStatefulDynamicSPEDA(encoding=encoding, config=dyn_cfg)
                runner_state["runner"] = runner
                runner_state["used_family"] = str(family)
                runner_state["used_size_gen"] = int(size_gen)
                runner_state["used_w_noise"] = float(w_noise)
                runner_state["fallback_errors"] = list(init_errors)
                if (
                    str(family) != str(args.pm_family)
                    or int(size_gen) != int(size_gen_eff)
                    or abs(float(w_noise) - float(args.eda_w_noise)) > 1e-12
                ):
                    _emit(
                        float(state.current_time_min),
                        (
                            f"DYNAMIC_EDA_FALLBACK_USED family={family} size_gen={int(size_gen)} "
                            f"w_noise={float(w_noise):.2f}"
                        ),
                        out_events,
                        "eda_fallback_used",
                        {
                            "family_used": str(family),
                            "size_gen_used": int(size_gen),
                            "w_noise_used": float(w_noise),
                            "engine": "dynamic-stateful",
                        },
                    )
                break
            except Exception as exc:
                init_errors.append(
                    f"{family}/N={size_gen}/noise={w_noise:.2f}: {type(exc).__name__}: {exc}"
                )
        if runner is None:
            raise RuntimeError(" | ".join(init_errors))

    problem = FreeFloatingRobustDMOOP(
        state_template=copy.deepcopy(state),
        scenario=scenario,
        encoding=encoding,
        config=cfg,
        horizon_hours=float(args.eda_horizon_hours),
        random_seed=int(seed + 10000 + step_idx),
    )
    window = {
        "id": f"stream_step_{int(step_idx)}",
        "start_min": float(state.current_time_min),
        "horizon_hours": float(args.eda_horizon_hours),
        "demand_multiplier": {zone_id: float(args.base_demand_scale) for zone_id in scenario.zones},
        "max_iter": int(max_iter_eff),
        "recent_service_signal": dict(recent_service_signal or {}),
    }
    result = runner.optimize_window(problem=problem, window=window, carry_state=state)
    raw = dict(result.raw_result)
    selected_vector = np.asarray(raw.get("selected_vector"), dtype=float)
    selected_policy = _policy_to_dict(raw.get("selected_policy"))
    selected_objectives = (
        None if result.selected_objectives is None else np.asarray(result.selected_objectives, dtype=float)
    )
    trigger_info = dict(raw.get("trigger_info", {}))
    change_info = dict(raw.get("change_info", {}))
    regime = dict(raw.get("regime", {}))
    reference_regime = dict(raw.get("reference_regime", {}))
    reoptimized = bool(raw.get("reoptimized", True))
    incumbent_gate_applied = bool(raw.get("incumbent_gate_applied", False))
    action_label = "update" if reoptimized else "hold"
    event_kind = "eda_reopt" if reoptimized else "eda_reopt_hold"
    selection_strategy = (
        "dynamic_stateful_update"
        if reoptimized
        else f"dynamic_stateful_hold_{str(trigger_info.get('reason', 'stable')).strip().lower()}"
    )
    raw_selection_strategy = str(raw.get("selection_strategy", "")).strip()
    if raw_selection_strategy:
        selection_strategy = raw_selection_strategy if reoptimized else f"{selection_strategy}_{raw_selection_strategy}"
    if incumbent_gate_applied:
        selection_strategy = selection_strategy + "_with_incumbent_gate"
    if selected_objectives is not None:
        _emit(
            float(state.current_time_min),
            (
                f"DYNAMIC_EDA step={step_idx} action={action_label} "
                f"reason={trigger_info.get('reason', 'n/a')} "
                f"sel=[{selected_objectives[0]:.3f},{selected_objectives[1]:.1f},{selected_objectives[2]:.3f}]"
            ),
            out_events,
            event_kind,
            {
                "step_idx": int(step_idx),
                "engine": "dynamic-stateful",
                "pareto_size": int(np.asarray(raw.get("pareto_front_max", np.empty((0, 3)))).shape[0]),
                "sel_f1": float(selected_objectives[0]),
                "sel_f2": float(selected_objectives[1]),
                "sel_f3": float(selected_objectives[2]),
                "reoptimized": bool(reoptimized),
                "trigger_reason": str(trigger_info.get("reason", "")),
                "change_score": float(change_info.get("score", 0.0)),
                "profile_hint": str(change_info.get("profile_hint", "")),
            },
        )

    return {
        "selected_vector": selected_vector,
        "selected_policy": selected_policy,
        "selected_objectives": selected_objectives,
        "selected_index": None if result.selected_index is None else int(result.selected_index),
        "selection_strategy": selection_strategy,
        "selection_objective_score": raw.get("selection_objective_score"),
        "selection_shift_penalty": raw.get("selection_shift_penalty"),
        "selection_total_score": raw.get("selection_total_score"),
        "pareto_size": int(np.asarray(raw.get("pareto_front_max", np.empty((0, 3)))).shape[0]),
        "pareto_set": np.asarray(raw.get("pareto_set", np.empty((0, encoding.dimension))), dtype=float),
        "pareto_front_max": np.asarray(raw.get("pareto_front_max", np.empty((0, 3))), dtype=float),
        "optimizer_family_used": str(runner_state.get("used_family", getattr(args, "pm_family", "SPBN"))),
        "size_gen_used": int(runner_state.get("used_size_gen", size_gen_eff)),
        "w_noise_used": float(runner_state.get("used_w_noise", args.eda_w_noise)),
        "warm_start_pool_size": 0,
        "transfer_rows_used": 0,
        "archive_rows_used": 0,
        "incumbent_neighbors_used": 0,
        "archive_guided_immigrants_used": 0,
        "incumbent_score": raw.get("incumbent_score"),
        "candidate_score": raw.get("candidate_score"),
        "incumbent_gate_applied": bool(incumbent_gate_applied),
        "frontier_selected_index": None if result.selected_index is None else int(result.selected_index),
        "fallback_errors": list(runner_state.get("fallback_errors", [])),
        "trigger_info": trigger_info,
        "change_info": change_info,
        "regime": regime,
        "reference_regime": reference_regime,
        "reservation_guard": dict(raw.get("reservation_guard", {})),
        "recent_service_signal": dict(recent_service_signal or {}),
        "reoptimized": bool(reoptimized),
        "shadow_update_executed": bool(raw.get("shadow_update_executed", False)),
        "optimizer_steps_executed": int(raw.get("optimizer_steps_executed", 0)),
        "step_fallback_used": bool(raw.get("step_fallback_used", False)),
        "step_errors": list(raw.get("step_errors", [])),
    }


def run_stream(args: argparse.Namespace) -> None:
    scenario = MadridFreeFloatingScenario(
        cfg=ScenarioConfig(
            fleet_size=int(args.fleet_size),
            charger_count=int(args.charger_count),
            ops_resources=int(args.ops_resources),
            min_same_zone_trip_km=float(args.min_same_zone_trip_km),
            travel_metrics_json=str(getattr(args, "travel_metrics_json", "")).strip() or default_travel_metrics_json(),
            fare_mode=str(getattr(args, "fare_mode", "time_only")),
            fare_base_eur=float(getattr(args, "fare_base_eur", 0.0)),
            fare_per_km_eur=float(getattr(args, "fare_per_km_eur", 0.0)),
            fare_per_min_eur=float(getattr(args, "fare_per_min_eur", 0.24)),
            fare_min_total_eur=float(getattr(args, "fare_min_total_eur", 0.0)),
            fare_included_km=float(getattr(args, "fare_included_km", 200.0)),
            fare_extra_km_eur=float(getattr(args, "fare_extra_km_eur", 0.39)),
            reservation_share=float(getattr(args, "reservation_share", 0.28)),
            reservation_booking_intensity_mult=float(
                getattr(args, "reservation_booking_intensity_mult", 1.0)
            ),
            immediate_request_intensity_mult=float(
                getattr(args, "immediate_request_intensity_mult", 1.0)
            ),
            reservation_mean_lead_min=float(getattr(args, "reservation_mean_lead_min", 35.0)),
            zone_base_demand_json=str(getattr(args, "zone_base_demand_json", "")).strip() or None,
        )
    )
    cfg = RobustModelConfig(
        decision_interval_min=int(args.decision_interval_min),
        planning_horizon_min=int(args.planning_horizon_min),
        ops_cost_profile=str(getattr(args, "ops_cost_profile", "custom")),
        relocation_cost_per_km=float(getattr(args, "relocation_cost_per_km", 0.22)),
        ops_charge_action_credit_eur=float(getattr(args, "ops_charge_action_credit_eur", 0.0)),
        ops_deadhead_cost_per_km=float(getattr(args, "ops_deadhead_cost_per_km", 0.22)),
        ops_cost_per_action_eur=float(getattr(args, "ops_cost_per_action_eur", 1.20)),
        energy_cost_per_kwh=float(getattr(args, "energy_cost_per_kwh", 0.23)),
        max_user_access_walk_min=float(getattr(args, "max_user_access_walk_min", 12.0)),
        ops_relocation_handover_min=float(getattr(args, "ops_relocation_handover_min", 1.5)),
        ops_charge_plug_min=float(getattr(args, "ops_charge_plug_min", 2.0)),
        demand_only_zones_with_idle=bool(args.simple_access_model),
        assign_only_same_origin_zone=bool(args.simple_access_model),
        assume_zero_user_access_time=bool(args.simple_access_model),
    )
    state = scenario.initial_state(seed=int(args.seed), model_cfg=cfg)

    encoding = RobustDecisionEncoding(zone_ids=list(state.zones.keys()))
    planner = RollingHorizonPlanner(cfg=cfg)
    x_policy = _build_default_policy_vector(encoding)
    policy = encoding.decode(x_policy)
    reopt_transfer_pool: Optional[np.ndarray] = None
    reopt_archive_pool: Optional[np.ndarray] = None
    dynamic_runner_state: Optional[dict] = None
    eda_reopt_count = 0
    eda_fronts: List[dict] = []
    demand_multiplier = {zone_id: float(args.base_demand_scale) for zone_id in scenario.zones}
    if str(getattr(args, "eda_online_engine", "speda-static")) == "dynamic-stateful":
        dynamic_runner_state = {"runner": None, "used_family": None, "used_size_gen": None, "used_w_noise": None, "fallback_errors": []}

    if args.mode == "live":
        total_sim_min = float(args.real_duration_sec) * float(args.sim_min_per_real_sec)
    else:
        total_sim_min = float(args.sim_duration_min)

    delta = int(cfg.decision_interval_min)
    pending_reservations: Dict[str, Request] = {}
    out_events: Optional[List[dict]] = [] if args.output_jsonl else None
    status_series: List[dict] = []
    metrics_checkpoint = {
        "requests_total": int(state.metrics.requests_total),
        "served_requests": int(state.metrics.served_requests),
        "rejected_requests": int(state.metrics.rejected_requests),
        "canceled_bookings": int(state.metrics.canceled_bookings),
        "relocation_cost_eur": float(state.metrics.relocation_cost_eur),
        "ops_cost_eur": float(state.metrics.ops_cost_eur),
        "ops_deadhead_km_total": float(state.metrics.ops_deadhead_km_total),
    }
    recent_service_signal = {
        "requests": 0,
        "served": 0,
        "rejected": 0,
        "canceled": 0,
        "service_rate": 1.0,
        "relocation_cost_eur": 0.0,
        "ops_cost_eur": 0.0,
        "ops_deadhead_km": 0.0,
    }
    seq = 0

    _emit(
        0.0,
        (
            f"STREAM_START mode={args.mode} sim_total={total_sim_min:.1f}min "
            f"delta={delta}min seed={args.seed} "
            f"eda_reopt={int(bool(args.eda_reopt_enabled))} "
            f"engine={str(getattr(args, 'eda_online_engine', 'speda-static'))}"
        ),
        out_events,
        "stream_start",
    )

    total_steps = int(np.ceil(total_sim_min / max(1, delta)))
    for step_idx in range(total_steps):
        epoch_wall_start = time.monotonic()
        now = float(step_idx * delta)
        end = min(total_sim_min, now + delta)
        if now >= total_sim_min - 1e-9:
            break

        refresh_system_state(state=state, now_min=now, cfg=cfg)
        state.pending_reservations = dict(pending_reservations)

        if bool(args.eda_reopt_enabled) and (step_idx % max(1, int(args.eda_reopt_every_epochs)) == 0):
            try:
                prev_policy = policy
                prev_vector = np.asarray(x_policy, dtype=float)
                if str(getattr(args, "eda_online_engine", "speda-static")) == "dynamic-stateful":
                    reopt = _dynamic_reoptimize_policy(
                        state=state,
                        scenario=scenario,
                        encoding=encoding,
                        cfg=cfg,
                        seed=int(args.seed),
                        step_idx=int(step_idx),
                        args=args,
                        out_events=out_events,
                        runner_state=dynamic_runner_state or {},
                        recent_service_signal=recent_service_signal,
                    )
                else:
                    reopt = _eda_reoptimize_policy(
                        state=state,
                        scenario=scenario,
                        encoding=encoding,
                        cfg=cfg,
                        seed=int(args.seed),
                        step_idx=int(step_idx),
                        args=args,
                        out_events=out_events,
                        reference_vector=np.asarray(x_policy, dtype=float),
                        transfer_pool=reopt_transfer_pool,
                        archive_pool=reopt_archive_pool,
                        incumbent_vector=np.asarray(x_policy, dtype=float),
                    )
                x_new = reopt["selected_vector"]
                pset_new = np.asarray(
                    reopt.get("pareto_set", np.empty((0, encoding.dimension))),
                    dtype=float,
                )
                if str(getattr(args, "eda_online_engine", "speda-static")) != "dynamic-stateful":
                    reopt_transfer_pool = (
                        pset_new if pset_new.ndim == 2 and pset_new.shape[0] > 0 else reopt_transfer_pool
                    )
                    if bool(getattr(args, "eda_temporal_archive_enabled", True)):
                        reopt_archive_pool = _update_temporal_archive(
                            archive_pool=reopt_archive_pool,
                            front=np.asarray(reopt.get("pareto_front_max", np.empty((0, 3))), dtype=float),
                            pset=pset_new,
                            max_size=int(getattr(args, "eda_temporal_archive_max_size", 96)),
                            keep_per_step=int(getattr(args, "eda_temporal_archive_keep_per_step", 12)),
                            encoding=encoding,
                        )
                if x_new is not None:
                    x_policy = np.asarray(x_new, dtype=float)
                    policy = encoding.decode(x_policy)
                if bool(reopt.get("reoptimized", x_new is not None)):
                    eda_reopt_count += 1
                if int(reopt["pareto_size"]) > 0 or str(getattr(args, "eda_online_engine", "speda-static")) == "dynamic-stateful":
                    vector_shift_l1 = None
                    vector_shift_max_abs = None
                    if x_new is not None:
                        delta_vec = np.abs(np.asarray(x_policy, dtype=float) - prev_vector)
                        vector_shift_l1 = float(np.sum(delta_vec))
                        vector_shift_max_abs = float(np.max(delta_vec))
                    eda_fronts.append(
                        {
                            "step_idx": int(step_idx),
                            "sim_time_min": float(now),
                            "pareto_size": int(reopt["pareto_size"]),
                            "selected_index": reopt["selected_index"],
                            "selection_strategy": str(reopt["selection_strategy"]),
                            "selection_objective_score": reopt["selection_objective_score"],
                            "selection_shift_penalty": reopt["selection_shift_penalty"],
                            "selection_total_score": reopt["selection_total_score"],
                            "frontier_selected_index": reopt["frontier_selected_index"],
                            "incumbent_score": reopt["incumbent_score"],
                            "candidate_score": reopt["candidate_score"],
                            "incumbent_gate_applied": bool(reopt["incumbent_gate_applied"]),
                            "warm_start_pool_size": int(reopt["warm_start_pool_size"]),
                            "transfer_rows_used": int(reopt["transfer_rows_used"]),
                            "archive_rows_used": int(reopt["archive_rows_used"]),
                            "incumbent_neighbors_used": int(reopt["incumbent_neighbors_used"]),
                            "archive_guided_immigrants_used": int(reopt["archive_guided_immigrants_used"]),
                            "temporal_archive_pool_size": (
                                0
                                if reopt_archive_pool is None
                                else int(np.asarray(reopt_archive_pool).shape[0])
                            ),
                            "selected_objectives": (
                                None
                                if reopt["selected_objectives"] is None
                                else [
                                    float(reopt["selected_objectives"][0]),
                                    float(reopt["selected_objectives"][1]),
                                    float(reopt["selected_objectives"][2]),
                                ]
                            ),
                            "selected_vector": (
                                None if x_new is None else np.asarray(x_policy, dtype=float).tolist()
                            ),
                            "selected_policy": reopt["selected_policy"],
                            "vector_shift_l1": vector_shift_l1,
                            "vector_shift_max_abs": vector_shift_max_abs,
                            "previous_policy": _policy_to_dict(prev_policy),
                            "optimizer_family_used": str(reopt["optimizer_family_used"]),
                            "size_gen_used": int(reopt["size_gen_used"]),
                            "w_noise_used": float(reopt["w_noise_used"]),
                            "fallback_errors": list(reopt["fallback_errors"]),
                            "eda_online_engine": str(getattr(args, "eda_online_engine", "speda-static")),
                            "trigger_info": reopt.get("trigger_info", {}),
                            "change_info": reopt.get("change_info", {}),
                            "regime": reopt.get("regime", {}),
                            "reference_regime": reopt.get("reference_regime", {}),
                            "reservation_guard": reopt.get("reservation_guard", {}),
                            "recent_service_signal": reopt.get("recent_service_signal", {}),
                            "reoptimized": bool(reopt.get("reoptimized", True)),
                            "shadow_update_executed": bool(reopt.get("shadow_update_executed", False)),
                            "optimizer_steps_executed": int(reopt.get("optimizer_steps_executed", 0)),
                            "step_fallback_used": bool(reopt.get("step_fallback_used", False)),
                            "step_errors": list(reopt.get("step_errors", [])),
                            "pareto_front_max": np.asarray(reopt["pareto_front_max"], dtype=float).tolist(),
                        }
                    )
            except Exception as exc:
                _emit(
                    now,
                    f"EDA_REOPT_FAILED step={step_idx} reason={type(exc).__name__}: {exc}",
                    out_events,
                    "eda_reopt_failed",
                    {"step_idx": int(step_idx), "error": f"{type(exc).__name__}: {exc}"},
                )

        demand_forecast = scenario.forecast_demand(
            now_min=now,
            horizon_min=float(cfg.planning_horizon_min),
            external_multiplier=demand_multiplier,
        )
        planner.plan_epoch(
            state=state,
            epoch_start_min=now,
            epoch_end_min=end,
            demand_forecast=demand_forecast,
            policy=policy,
        )
        refresh_system_state(state=state, now_min=now, cfg=cfg)

        heap: List[tuple] = []

        task_events, seq = _collect_task_end_events(state=state, start_min=now, end_min=end, seq0=seq)
        for ev in task_events:
            _push_event(heap, ev)

        # Existing reservations: pickup/cancel events in this interval.
        for req in list(pending_reservations.values()):
            p_time = req.pickup_time_min if req.pickup_time_min is not None else req.request_time_min
            c_time = req.cancellation_time_min
            if c_time is not None and now < c_time <= end and c_time < p_time - 1e-9:
                _push_event(
                    heap,
                    StreamEvent(
                        sim_time_min=float(c_time),
                        priority=1,
                        seq=seq,
                        kind="cancel",
                        payload={"request_id": req.request_id},
                    ),
                )
                seq += 1
            if now < p_time <= end:
                _push_event(
                    heap,
                    StreamEvent(
                        sim_time_min=float(p_time),
                        priority=3,
                        seq=seq,
                        kind="pickup",
                        payload={"request_id": req.request_id},
                    ),
                )
                seq += 1

        # New booking arrivals in this interval.
        if cfg.demand_only_zones_with_idle:
            origin_filter = {
                v.zone_id
                for v in state.vehicles.values()
                if v.status == VehicleStatus.IDLE and v.available_at_min <= now + 1e-9
            }
        else:
            origin_filter = None
        reqs = scenario.sample_requests(
            state=state,
            start_min=now,
            end_min=end,
            rng=np.random.default_rng(args.seed + step_idx),
            external_multiplier=demand_multiplier,
            origin_zone_filter=origin_filter,
        )
        for req in reqs:
            _push_event(
                heap,
                StreamEvent(
                    sim_time_min=float(req.request_time_min),
                    priority=2,
                    seq=seq,
                    kind="booking",
                    payload={"request": req},
                ),
            )
            seq += 1

        while heap:
            event_time, _, _, kind, payload = heapq.heappop(heap)
            refresh_system_state(state=state, now_min=float(event_time), cfg=cfg)

            if kind == "task_end":
                vid = payload["vehicle_id"]
                task_type = payload["task_type"]
                v = state.vehicles.get(vid)
                if v is None:
                    continue
                if task_type == TaskType.TRIP.value:
                    _emit(
                        event_time,
                        (
                            f"TRIP_FINISHED req={payload.get('request_id')} veh={vid} "
                            f"zone={v.zone_id} soc={v.soc:.3f}"
                        ),
                        out_events,
                        "trip_finished",
                        {
                            "request_id": payload.get("request_id"),
                            "vehicle_id": vid,
                            "zone_id": v.zone_id,
                            "soc_after": float(v.soc),
                        },
                    )
                elif task_type == TaskType.RELOCATION.value:
                    _emit(
                        event_time,
                        f"RELOCATION_FINISHED veh={vid} zone={v.zone_id} soc={v.soc:.3f}",
                        out_events,
                        "relocation_finished",
                        {"vehicle_id": vid, "zone_id": v.zone_id, "soc_after": float(v.soc)},
                    )
                elif task_type == TaskType.CHARGE.value:
                    _emit(
                        event_time,
                        (
                            f"CHARGE_FINISHED veh={vid} charger={payload.get('charger_id')} "
                            f"zone={v.zone_id} soc={v.soc:.3f}"
                        ),
                        out_events,
                        "charge_finished",
                        {
                            "vehicle_id": vid,
                            "charger_id": payload.get("charger_id"),
                            "zone_id": v.zone_id,
                            "soc_after": float(v.soc),
                        },
                    )
                continue

            if kind == "booking":
                req: Request = payload["request"]
                state.metrics.bookings_total += 1
                _emit(
                    event_time,
                    (
                        f"REQUEST_CREATED id={req.request_id} {req.origin_zone_id}->{req.destination_zone_id} "
                        f"reserv={int(req.is_reservation)}"
                    ),
                    out_events,
                    "request_created",
                    {
                        "request_id": req.request_id,
                        "origin_zone_id": req.origin_zone_id,
                        "destination_zone_id": req.destination_zone_id,
                        "is_reservation": bool(req.is_reservation),
                        "pickup_time_min": float(req.pickup_time_min or req.request_time_min),
                    },
                )

                if req.is_reservation:
                    effective_policy = planner.materialize_policy(
                        state=state,
                        demand_forecast=demand_forecast,
                        policy=policy,
                        at_time_min=float(event_time),
                    )
                    preassigned = planner.try_preassign_reservation(
                        request=req,
                        state=state,
                        policy=effective_policy,
                        event_time_min=float(event_time),
                    )
                    if preassigned.preassigned_vehicle_id is not None:
                        state.metrics.preassigned_bookings += 1
                        _emit(
                            event_time,
                            (
                                f"RESERVATION_PREASSIGNED id={preassigned.request_id} "
                                f"veh={preassigned.preassigned_vehicle_id}"
                            ),
                            out_events,
                            "reservation_preassigned",
                            {
                                "request_id": preassigned.request_id,
                                "vehicle_id": str(preassigned.preassigned_vehicle_id),
                            },
                        )
                    pending_reservations[req.request_id] = preassigned
                    state.pending_reservations = dict(pending_reservations)
                    p_time = (
                        preassigned.pickup_time_min
                        if preassigned.pickup_time_min is not None
                        else preassigned.request_time_min
                    )
                    c_time = preassigned.cancellation_time_min
                    if c_time is not None and event_time < c_time <= end and c_time < p_time - 1e-9:
                        _push_event(
                            heap,
                            StreamEvent(
                                sim_time_min=float(c_time),
                                priority=1,
                                seq=seq,
                                kind="cancel",
                                payload={"request_id": preassigned.request_id},
                            ),
                        )
                        seq += 1
                    if event_time < p_time <= end:
                        _push_event(
                            heap,
                            StreamEvent(
                                sim_time_min=float(p_time),
                                priority=3,
                                seq=seq,
                                kind="pickup",
                                payload={"request_id": preassigned.request_id},
                            ),
                        )
                        seq += 1
                else:
                    seq = _serve_pickup(
                        req=req,
                        event_time_min=float(event_time),
                        window_end_min=float(end),
                        state=state,
                        cfg=cfg,
                        heap=heap,
                        seq=seq,
                        out_events=out_events,
                    )
                continue

            if kind == "cancel":
                rid = payload["request_id"]
                if rid in pending_reservations:
                    pending_reservations.pop(rid, None)
                    state.pending_reservations = dict(pending_reservations)
                    state.metrics.canceled_bookings += 1
                    _emit(
                        event_time,
                        f"REQUEST_CANCELED id={rid}",
                        out_events,
                        "request_canceled",
                        {"request_id": rid},
                    )
                continue

            if kind == "pickup":
                rid = payload["request_id"]
                if rid in pending_reservations:
                    req_pick = pending_reservations.pop(rid)
                    state.pending_reservations = dict(pending_reservations)
                    seq = _serve_pickup(
                        req=req_pick,
                        event_time_min=float(event_time),
                        window_end_min=float(end),
                        state=state,
                        cfg=cfg,
                        heap=heap,
                        seq=seq,
                        out_events=out_events,
                    )
                continue

        refresh_system_state(state=state, now_min=end, cfg=cfg)
        state.current_time_min = float(end)

        status = _status_row(state)
        status_series.append({"sim_time_min": float(end), **status})
        recent_requests = int(state.metrics.requests_total) - int(metrics_checkpoint["requests_total"])
        recent_served = int(state.metrics.served_requests) - int(metrics_checkpoint["served_requests"])
        recent_rejected = int(state.metrics.rejected_requests) - int(metrics_checkpoint["rejected_requests"])
        recent_canceled = int(state.metrics.canceled_bookings) - int(metrics_checkpoint["canceled_bookings"])
        recent_relocation_cost_eur = float(state.metrics.relocation_cost_eur) - float(
            metrics_checkpoint["relocation_cost_eur"]
        )
        recent_ops_cost_eur = float(state.metrics.ops_cost_eur) - float(
            metrics_checkpoint["ops_cost_eur"]
        )
        recent_ops_deadhead_km = float(state.metrics.ops_deadhead_km_total) - float(
            metrics_checkpoint["ops_deadhead_km_total"]
        )
        recent_service_signal = {
            "requests": int(max(0, recent_requests)),
            "served": int(max(0, recent_served)),
            "rejected": int(max(0, recent_rejected)),
            "canceled": int(max(0, recent_canceled)),
            "service_rate": (
                1.0
                if recent_requests <= 0
                else float(max(0.0, min(1.0, recent_served / max(1, recent_requests))))
            ),
            "relocation_cost_eur": float(max(0.0, recent_relocation_cost_eur)),
            "ops_cost_eur": float(max(0.0, recent_ops_cost_eur)),
            "ops_deadhead_km": float(max(0.0, recent_ops_deadhead_km)),
        }
        metrics_checkpoint = {
            "requests_total": int(state.metrics.requests_total),
            "served_requests": int(state.metrics.served_requests),
            "rejected_requests": int(state.metrics.rejected_requests),
            "canceled_bookings": int(state.metrics.canceled_bookings),
            "relocation_cost_eur": float(state.metrics.relocation_cost_eur),
            "ops_cost_eur": float(state.metrics.ops_cost_eur),
            "ops_deadhead_km_total": float(state.metrics.ops_deadhead_km_total),
        }
        _emit(
            end,
            (
                "STATUS "
                f"req={status['requests']} served={status['served']} rej={status['rejected']} "
                f"can={status['canceled']} sr={status['service_rate']:.3f} "
                f"idle={status['idle']} serving={status['serving']} charging={status['charging']} "
                f"profit={status['profit_eur']:.1f} soc={status['mean_soc']:.3f} "
                f"reopt={eda_reopt_count}"
            ),
            out_events,
            "status",
            status,
        )

        if args.mode == "live":
            target_real_sec = (end - now) / max(1e-9, float(args.sim_min_per_real_sec))
            elapsed_real_sec = time.monotonic() - epoch_wall_start
            if target_real_sec > elapsed_real_sec:
                time.sleep(target_real_sec - elapsed_real_sec)

    _emit(
        state.current_time_min,
        "STREAM_END",
        out_events,
        "stream_end",
        _status_row(state),
    )

    if args.audit_prefix:
        summary = _build_audit_summary(
            state=state,
            pending_reservations=len(pending_reservations),
            status_series=status_series,
            args=args,
            total_sim_min=float(total_sim_min),
        )
        _write_audit_outputs(prefix=args.audit_prefix, summary=summary, status_series=status_series)

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assert out_events is not None
        with output_path.open("w", encoding="utf-8") as f:
            for row in out_events:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nLog guardado en: {output_path}", flush=True)

    if args.eda_fronts_json:
        fronts_path = Path(args.eda_fronts_json)
        fronts_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "mode": args.mode,
                "seed": int(args.seed),
                "decision_interval_min": int(args.decision_interval_min),
                "planning_horizon_min": int(args.planning_horizon_min),
                "sim_total_min": float(total_sim_min),
                "eda_reopt_enabled": bool(args.eda_reopt_enabled),
                "eda_reopt_count": int(eda_reopt_count),
                "eda_horizon_hours": int(args.eda_horizon_hours),
                "eda_online_engine": str(getattr(args, "eda_online_engine", "speda-static")),
                "dynamic_online_profile": str(getattr(args, "dynamic_online_profile", "custom")),
                "eda_optimizer_family": str(args.eda_optimizer_family),
                "pm_family": str(getattr(args, "pm_family", "SPBN")),
                "policy_smooth_lambda": float(args.policy_smooth_lambda),
                "policy_smooth_metric": str(args.policy_smooth_metric),
                "fleet_size": int(args.fleet_size),
                "charger_count": int(args.charger_count),
                "ops_resources": int(args.ops_resources),
                "base_demand_scale": float(args.base_demand_scale),
            },
            "reoptimizations": eda_fronts,
        }
        fronts_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"EDA fronts:   {fronts_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Demo de flujo de demanda/eventos para free-floating EV. "
            "Modo live: tiempo real de pared. Modo batch: ejecucion acelerada."
        )
    )
    p.add_argument("--mode", choices=["live", "batch"], default="live")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--decision-interval-min", type=int, default=5)
    p.add_argument("--planning-horizon-min", type=int, default=60)
    p.add_argument("--fleet-size", type=int, default=220)
    p.add_argument("--charger-count", type=int, default=22)
    p.add_argument("--ops-resources", type=int, default=26)
    p.add_argument("--base-demand-scale", type=float, default=1.0)
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
    p.add_argument("--real-duration-sec", type=float, default=300.0)
    p.add_argument("--sim-min-per-real-sec", type=float, default=1.0)
    p.add_argument("--sim-duration-min", type=float, default=720.0)
    p.add_argument("--output-jsonl", type=str, default="")
    p.add_argument(
        "--audit-prefix",
        type=str,
        default="",
        help=(
            "Prefijo de salida para auditoria. "
            "Se generan <prefijo>_summary.json y <prefijo>_status.csv."
        ),
    )
    p.add_argument(
        "--min-same-zone-trip-km",
        type=float,
        default=1.0,
        help=(
            "Umbral minimo para viajes con origen=destino por microzona. "
            "Si no se alcanza, se remapea a otra zona."
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
    p.add_argument(
        "--eda-reopt-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activa reoptimizacion EDA online en cada cierta cantidad de epocas.",
    )
    p.add_argument(
        "--eda-reopt-every-epochs",
        type=int,
        default=1,
        help="Reoptimiza cada N epocas (N*decision_interval_min).",
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
            "selective_simple_carsharing",
            "adr_eda_reservation_aware_v1",
            "adr_eda_reservation_aware_v2",
        ],
        help="Preset reproducible para DynamicEDA online sobre el stream real.",
    )
    p.add_argument(
        "--eda-horizon-hours",
        type=int,
        default=2,
        help="Horizonte de evaluacion para cada reoptimizacion EDA.",
    )
    p.add_argument("--eda-size-gen", type=int, default=40)
    p.add_argument("--eda-max-iter", type=int, default=4)
    p.add_argument("--eda-dead-iter", type=int, default=3)
    p.add_argument("--eda-alpha", type=float, default=0.70)
    p.add_argument("--eda-w-noise", type=float, default=0.10)
    p.add_argument(
        "--eda-optimizer-family",
        type=str,
        default="SPBN",
        choices=["SPBN", "KDEBN"],
    )
    p.add_argument("--elite-factor", type=float, default=0.35)
    p.add_argument("--archive-generations", type=float, default=4.0)
    p.add_argument("--pm-family", type=str, default="SPBN", choices=["SPBN", "GBN", "KDEBN"])
    p.add_argument("--num-folds", type=int, default=3)
    p.add_argument("--archive-retention-after-change", type=float, default=0.30)
    p.add_argument("--with-reaction", action="store_true")
    p.add_argument("--adaptive-policy-enabled", action="store_true")
    p.add_argument("--reaction-quality-gate-enabled", action="store_true")
    p.add_argument("--local-search-enabled", action="store_true")
    p.add_argument("--dynamic-change-detector-mode", type=str, default="sentinel", choices=["sentinel", "problem-aware"])
    p.add_argument("--dynamic-reopt-trigger", type=str, default="always", choices=["always", "event"])
    p.add_argument("--dynamic-threshold-mild", type=float, default=0.20)
    p.add_argument("--dynamic-threshold-severe", type=float, default=0.45)
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
    p.add_argument("--eda-parallelize", action="store_true")
    p.add_argument("--eda-auto-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--eda-warm-start-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reutiliza frente anterior y vecinos del incumbente como poblacion inicial.",
    )
    p.add_argument(
        "--eda-temporal-archive-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mantiene un archivo temporal con soluciones Pareto de ventanas previas.",
    )
    p.add_argument(
        "--eda-temporal-archive-max-size",
        type=int,
        default=96,
        help="Tamano maximo del archivo temporal de soluciones.",
    )
    p.add_argument(
        "--eda-temporal-archive-keep-per-step",
        type=int,
        default=12,
        help="Numero de soluciones del frente actual que se conservan por ventana en el archivo.",
    )
    p.add_argument(
        "--eda-incumbent-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mantiene la politica incumbente si la nueva no mejora lo suficiente en el estado actual.",
    )
    p.add_argument(
        "--eda-incumbent-min-improvement",
        type=float,
        default=0.01,
        help="Mejora minima media en objetivos [0,1] para aceptar una nueva politica.",
    )
    p.add_argument(
        "--eda-incumbent-neighbor-ratio",
        type=float,
        default=0.25,
        help="Fraccion de la poblacion inicial sembrada como vecinos de la politica incumbente.",
    )
    p.add_argument(
        "--eda-incumbent-noise-scale",
        type=float,
        default=0.04,
        help="Escala relativa del ruido usada para sembrar vecinos alrededor de la politica incumbente.",
    )
    p.add_argument(
        "--eda-archive-guided-ratio",
        type=float,
        default=0.15,
        help="Fraccion de la poblacion inicial guiada alrededor de soluciones del archivo temporal.",
    )
    p.add_argument(
        "--eda-archive-guided-noise-scale",
        type=float,
        default=0.05,
        help="Escala relativa del ruido usada para guided immigrants alrededor del archivo temporal.",
    )
    p.add_argument(
        "--eda-fronts-json",
        type=str,
        default="",
        help="JSON de salida con fronteras de Pareto por cada reoptimizacion online.",
    )
    p.add_argument(
        "--policy-smooth-lambda",
        type=float,
        default=0.15,
        help=(
            "Penalizacion temporal al cambio de politica entre reoptimizaciones. "
            "0 desactiva el suavizado."
        ),
    )
    p.add_argument(
        "--policy-smooth-metric",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help="Metrica usada para penalizar el cambio del vector de decision.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_ops_cost_profile(args)
    run_stream(args)


if __name__ == "__main__":
    main()
