from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .evaluator import FreeFloatingRobustDMOOP
from .edaspy_runtime import enable_adaptive_bn_folds_patch, ensure_edaspy_importable
from .planner import RobustDecisionEncoding
from .travel_metrics import walk_time_min, zone_distance_km
from .types import OpsStatus, SystemState, VehicleStatus


def _normalize_pm_family(raw: Optional[str]) -> str:
    aliases = {
        "SPBN": "SPBN",
        "SPEDA": "SPBN",
        "GBN": "GBN",
        "KDEBN": "KDEBN",
        "KDE": "KDEBN",
    }
    key = str(raw or "SPBN").strip().upper()
    if key not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"pm_family='{raw}' no soportado. Usa: {supported}.")
    return aliases[key]


def _normalize_change_detector_mode(raw: Optional[str]) -> str:
    key = str(raw or "sentinel").strip().lower().replace("_", "-")
    aliases = {
        "sentinel": "sentinel",
        "problem-aware": "problem-aware",
        "problemaware": "problem-aware",
        "regime": "problem-aware",
    }
    if key not in aliases:
        raise ValueError(
            f"change_detector_mode='{raw}' no soportado. Usa: sentinel, problem-aware."
        )
    return aliases[key]


def _normalize_reopt_trigger_mode(raw: Optional[str]) -> str:
    key = str(raw or "always").strip().lower().replace("_", "-")
    aliases = {
        "always": "always",
        "event": "event",
        "event-driven": "event",
        "eventdriven": "event",
    }
    if key not in aliases:
        raise ValueError(f"reopt_trigger_mode='{raw}' no soportado. Usa: always, event.")
    return aliases[key]


_DEFAULT_PROBLEM_DETECTOR = {
    "weights": {
        "demand_pressure": 0.26,
        "reservation_pressure": 0.18,
        "low_soc_share": 0.18,
        "charger_pressure": 0.08,
        "spatial_imbalance": 0.14,
        "coverage_gap": 0.04,
        "recent_reject_rate": 0.12,
    },
    "scales": {
        "demand_pressure": 0.35,
        "reservation_pressure": 0.20,
        "low_soc_share": 0.10,
        "charger_pressure": 0.30,
        "spatial_imbalance": 0.20,
        "coverage_gap": 0.20,
        "recent_reject_rate": 0.10,
    },
    "threshold_mild": 0.20,
    "threshold_severe": 0.45,
    "low_soc_threshold": 0.30,
    "reference_ewma_alpha": 0.35,
    "replace_weights": False,
    "replace_scales": False,
    "hard_component_thresholds": {},
    "min_reopt_interval_min": 10.0,
    "max_reopt_interval_min": 25.0,
    "hard_low_soc_share": 0.30,
    "hard_charger_pressure": 0.85,
    "hard_reservation_pressure": 0.50,
    "hard_coverage_gap": 0.60,
    "hard_recent_reject_rate": 0.15,
    "hard_recent_request_count_min": 4,
}


def _merge_problem_detector_config(provided: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(_DEFAULT_PROBLEM_DETECTOR)
    if provided is None:
        return cfg
    if not isinstance(provided, dict):
        raise TypeError(
            f"problem_detector debe ser un dict, recibido {type(provided).__name__}"
        )

    for key, value in provided.items():
        if key in {"weights", "scales", "hard_component_thresholds"}:
            if not isinstance(value, dict):
                raise TypeError(f"problem_detector['{key}'] debe ser un dict.")
            replace_map = False
            if key == "weights":
                replace_map = bool(provided.get("replace_weights", False))
            elif key == "scales":
                replace_map = bool(provided.get("replace_scales", False))
            merged = {} if replace_map else dict(cfg[key])
            for sub_key, sub_value in value.items():
                merged[str(sub_key)] = float(sub_value)
            cfg[key] = merged
        else:
            cfg[key] = value

    cfg["threshold_mild"] = max(0.0, float(cfg["threshold_mild"]))
    cfg["threshold_severe"] = max(float(cfg["threshold_mild"]) + 1e-6, float(cfg["threshold_severe"]))
    cfg["low_soc_threshold"] = float(np.clip(float(cfg["low_soc_threshold"]), 0.05, 0.60))
    cfg["reference_ewma_alpha"] = float(np.clip(float(cfg["reference_ewma_alpha"]), 0.0, 1.0))
    cfg["replace_weights"] = bool(cfg.get("replace_weights", False))
    cfg["replace_scales"] = bool(cfg.get("replace_scales", False))
    cfg["min_reopt_interval_min"] = max(0.0, float(cfg["min_reopt_interval_min"]))
    cfg["max_reopt_interval_min"] = max(
        float(cfg["min_reopt_interval_min"]) + 1e-6,
        float(cfg["max_reopt_interval_min"]),
    )
    cfg["hard_low_soc_share"] = float(np.clip(float(cfg["hard_low_soc_share"]), 0.0, 1.0))
    cfg["hard_charger_pressure"] = max(0.0, float(cfg["hard_charger_pressure"]))
    cfg["hard_reservation_pressure"] = max(0.0, float(cfg["hard_reservation_pressure"]))
    cfg["hard_coverage_gap"] = float(np.clip(float(cfg.get("hard_coverage_gap", 0.30)), 0.0, 1.0))
    cfg["hard_recent_reject_rate"] = float(
        np.clip(float(cfg.get("hard_recent_reject_rate", 0.15)), 0.0, 1.0)
    )
    cfg["hard_recent_request_count_min"] = max(
        1.0, float(cfg.get("hard_recent_request_count_min", 4.0))
    )
    cfg["weights"] = {str(k): max(0.0, float(v)) for k, v in dict(cfg["weights"]).items()}
    cfg["scales"] = {str(k): max(1e-9, float(v)) for k, v in dict(cfg["scales"]).items()}
    cfg["hard_component_thresholds"] = {
        str(k): max(0.0, float(v)) for k, v in dict(cfg.get("hard_component_thresholds", {})).items()
    }
    return cfg


_DEFAULT_RESERVATION_GUARD = {
    "enabled": False,
    "commitment_horizon_fraction": 0.75,
    "due_soon_window_min": 25.0,
    "reservation_pressure_scale": 0.55,
    "due_soon_pressure_scale": 0.30,
    "preassigned_pressure_scale": 0.25,
    "commitment_pressure_scale": 0.55,
    "selection_smooth_lambda_gain": 0.32,
    "incumbent_min_improvement_gain": 0.025,
    "react_strength_damping": 0.18,
    "immigrant_ratio_damping": 0.30,
    "react_strength_floor_scale": 0.72,
    "immigrant_ratio_floor_scale": 0.60,
    "predictive_disable_threshold": 0.85,
    "front_ls_disable_threshold": 0.60,
    "predictive_disable_relocation_threshold": 2.0,
    "front_ls_disable_relocation_threshold": 2.0,
    "mild_step_budget_scale": 0.85,
    "medium_step_budget_scale": 0.95,
    "max_interval_step_budget_scale": 0.90,
    "relocating_pressure_scale": 0.08,
    "recent_deadhead_km_scale": 8.0,
    "recent_ops_cost_scale": 8.0,
    "recent_relocation_cost_scale": 8.0,
    "relocating_pressure_weight": 0.0,
    "recent_deadhead_km_weight": 0.0,
    "recent_ops_cost_weight": 0.0,
    "recent_relocation_cost_weight": 0.0,
    "relocation_debt_signal_weight": 0.0,
    "demand_pressure_scale": 6.0,
    "coverage_gap_overload_scale": 0.20,
    "recent_reject_rate_overload_scale": 0.08,
    "demand_pressure_overload_weight": 0.0,
    "coverage_gap_overload_weight": 0.0,
    "recent_reject_rate_overload_weight": 0.0,
    "overload_relief_weight": 0.0,
    "overload_unsuppress_threshold": 2.0,
    "max_interval_hold_threshold": 1.1,
    "hold_service_reject_rate_max": 0.08,
    "hold_service_coverage_gap_max": 0.35,
    "cost_guard_enabled": False,
    "cost_guard_disable_on_bootstrap": False,
    "cost_guard_activation_signal_min": 0.0,
    "cost_guard_activation_relocation_signal_min": 0.0,
    "cost_guard_activation_demand_signal_min": 0.0,
    "cost_guard_activation_coverage_gap_signal_min": 0.0,
    "cost_guard_activation_recent_reject_signal_min": 0.0,
    "cost_guard_burst_enabled": False,
    "cost_guard_burst_commitment_signal_min": 0.0,
    "cost_guard_burst_relocation_signal_min": 0.0,
    "candidate_shift_penalty_burst_max": 1e9,
    "candidate_min_profit_gain_eur_for_burst_move": 0.0,
    "candidate_extra_deadhead_km_max": 1e9,
    "candidate_extra_operational_cost_eur_max": 1e9,
    "candidate_extra_relocation_cost_eur_max": 1e9,
    "candidate_min_profit_gain_eur_for_costly_move": 0.0,
    "service_deployment_guard_enabled": False,
    "service_guard_disable_on_bootstrap": True,
    "service_guard_activation_demand_signal_min": 0.0,
    "service_guard_activation_coverage_gap_signal_min": 0.0,
    "service_guard_activation_recent_reject_signal_min": 0.0,
    "service_guard_max_service_drop": 0.0,
    "service_guard_max_extra_rejected": 0.0,
    "service_guard_max_extra_deadhead_km": 1e9,
    "service_guard_min_profit_gain_eur": -1e9,
    "service_guard_require_opportunity": False,
    "service_guard_min_service_gain": 0.0,
    "service_guard_min_rejected_reduction": 0.0,
    "service_guard_neutral_min_profit_gain_eur": -1e9,
}


_DEFAULT_ADAPTIVE_RESPONSE = {
    "enabled": False,
    "diagnostics_enabled": False,
    "score_scale": 0.34,
    "similarity_decay": 1.0,
    "drift_weight": 0.55,
    "similarity_weight": 0.45,
    "predictive_low_scale": 0.55,
    "predictive_high_scale": 1.25,
    "continuity_low_scale": 0.70,
    "continuity_high_scale": 1.25,
    "diversity_low_scale": 1.45,
    "diversity_high_scale": 0.85,
    "front_ls_low_scale": 0.80,
    "front_ls_high_scale": 1.15,
    "low_reliability_threshold": 0.45,
    "high_reliability_threshold": 0.75,
    "guard_enabled": False,
    "guard_shift_l1_threshold": 2.50,
    "guard_min_score_gain": 0.09,
    "guard_recent_relocation_cost_min": 6.0,
    "guard_ops_pressure_min": 0.25,
    "guard_relocating_pressure_min": 0.03,
    "guard_require_debt_signal": True,
    "selection_policy_enabled": False,
    "selection_policy_diagnostics_enabled": False,
    "selection_policy_strength": 0.35,
    "selection_policy_objective_gain": 0.70,
    "selection_policy_continuity_gain": 1.00,
    "selection_policy_diversity_gain": 1.20,
    "selection_policy_max_shift_penalty": 0.80,
    "selection_policy_max_shift_reward": 0.25,
    "service_constrained_selection_enabled": False,
    "service_constrained_selection_diagnostics_enabled": False,
    "service_constrained_service_floor": 0.99,
    "service_constrained_max_rejected_requests": 1e9,
    "service_constrained_profit_weight": 1.00,
    "service_constrained_deadhead_weight": 0.35,
    "service_constrained_health_weight": 0.20,
    "service_constrained_shift_weight": 0.15,
    "service_constrained_rejected_weight": 0.25,
    "weight_policy": "reliability",
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
    "contextual_max_front_ls": 0.25,
}


def _merge_adaptive_response_config(provided: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(_DEFAULT_ADAPTIVE_RESPONSE)
    if provided is None:
        return cfg
    if not isinstance(provided, dict):
        raise TypeError(
            f"adaptive_response debe ser un dict, recibido {type(provided).__name__}"
        )
    for key, value in provided.items():
        cfg[str(key)] = value

    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["diagnostics_enabled"] = bool(cfg.get("diagnostics_enabled", False))
    cfg["score_scale"] = max(1e-9, float(cfg.get("score_scale", 0.34)))
    cfg["similarity_decay"] = max(0.0, float(cfg.get("similarity_decay", 1.0)))
    cfg["drift_weight"] = max(0.0, float(cfg.get("drift_weight", 0.55)))
    cfg["similarity_weight"] = max(0.0, float(cfg.get("similarity_weight", 0.45)))
    if cfg["drift_weight"] + cfg["similarity_weight"] <= 1e-12:
        cfg["drift_weight"] = 1.0
        cfg["similarity_weight"] = 0.0
    for key in (
        "predictive_low_scale",
        "predictive_high_scale",
        "continuity_low_scale",
        "continuity_high_scale",
        "diversity_low_scale",
        "diversity_high_scale",
        "front_ls_low_scale",
        "front_ls_high_scale",
    ):
        cfg[key] = max(0.0, float(cfg.get(key, _DEFAULT_ADAPTIVE_RESPONSE[key])))
    cfg["low_reliability_threshold"] = float(
        np.clip(float(cfg.get("low_reliability_threshold", 0.45)), 0.0, 1.0)
    )
    cfg["high_reliability_threshold"] = float(
        np.clip(float(cfg.get("high_reliability_threshold", 0.75)), 0.0, 1.0)
    )
    if cfg["high_reliability_threshold"] < cfg["low_reliability_threshold"]:
        cfg["low_reliability_threshold"], cfg["high_reliability_threshold"] = (
            cfg["high_reliability_threshold"],
            cfg["low_reliability_threshold"],
        )
    cfg["guard_enabled"] = bool(cfg.get("guard_enabled", False))
    cfg["guard_shift_l1_threshold"] = max(
        0.0, float(cfg.get("guard_shift_l1_threshold", 2.50))
    )
    cfg["guard_min_score_gain"] = max(
        0.0, float(cfg.get("guard_min_score_gain", 0.09))
    )
    cfg["guard_recent_relocation_cost_min"] = max(
        0.0, float(cfg.get("guard_recent_relocation_cost_min", 6.0))
    )
    cfg["guard_ops_pressure_min"] = max(
        0.0, float(cfg.get("guard_ops_pressure_min", 0.25))
    )
    cfg["guard_relocating_pressure_min"] = max(
        0.0, float(cfg.get("guard_relocating_pressure_min", 0.03))
    )
    cfg["guard_require_debt_signal"] = bool(cfg.get("guard_require_debt_signal", True))
    cfg["selection_policy_enabled"] = bool(cfg.get("selection_policy_enabled", False))
    cfg["selection_policy_diagnostics_enabled"] = bool(
        cfg.get("selection_policy_diagnostics_enabled", False)
    )
    cfg["service_constrained_selection_enabled"] = bool(
        cfg.get("service_constrained_selection_enabled", False)
    )
    cfg["service_constrained_selection_diagnostics_enabled"] = bool(
        cfg.get("service_constrained_selection_diagnostics_enabled", False)
    )
    cfg["service_constrained_service_floor"] = float(
        np.clip(float(cfg.get("service_constrained_service_floor", 0.99)), 0.0, 1.0)
    )
    cfg["service_constrained_max_rejected_requests"] = max(
        0.0, float(cfg.get("service_constrained_max_rejected_requests", 1e9))
    )
    weight_policy = str(cfg.get("weight_policy", "reliability")).strip().lower()
    if weight_policy not in {"reliability", "contextual_formula"}:
        raise ValueError(
            "adaptive_response['weight_policy'] debe ser 'reliability' o "
            f"'contextual_formula', recibido {weight_policy!r}."
        )
    cfg["weight_policy"] = weight_policy
    for key in (
        "selection_policy_strength",
        "selection_policy_objective_gain",
        "selection_policy_continuity_gain",
        "selection_policy_diversity_gain",
        "selection_policy_max_shift_penalty",
        "selection_policy_max_shift_reward",
        "service_constrained_profit_weight",
        "service_constrained_deadhead_weight",
        "service_constrained_health_weight",
        "service_constrained_shift_weight",
        "service_constrained_rejected_weight",
        "contextual_temperature",
        "contextual_base_prior_strength",
        "contextual_diversity_target",
        "contextual_min_predictive",
        "contextual_max_predictive",
        "contextual_min_continuity",
        "contextual_max_continuity",
        "contextual_min_diversity",
        "contextual_max_diversity",
        "contextual_min_front_ls",
        "contextual_max_front_ls",
    ):
        cfg[key] = max(0.0, float(cfg.get(key, _DEFAULT_ADAPTIVE_RESPONSE[key])))
    for name in ("predictive", "continuity", "diversity", "front_ls"):
        lo_key = f"contextual_min_{name}"
        hi_key = f"contextual_max_{name}"
        if cfg[hi_key] < cfg[lo_key]:
            cfg[lo_key], cfg[hi_key] = cfg[hi_key], cfg[lo_key]
    return cfg


def _inherited_state_reliability(
    *,
    change_info: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, float | str]:
    cfg = _merge_adaptive_response_config(cfg)
    score = max(0.0, float(change_info.get("score", 0.0)))
    drift_reliability = float(1.0 / (1.0 + score / float(cfg["score_scale"])))

    breakdown = {
        str(k): max(0.0, float(v))
        for k, v in dict(change_info.get("breakdown", {})).items()
    }
    if breakdown:
        component_mean = float(sum(breakdown.values()) / max(1, len(breakdown)))
        similarity = float(np.exp(-float(cfg["similarity_decay"]) * component_mean))
    else:
        component_mean = 0.0
        similarity = drift_reliability

    drift_weight = float(cfg["drift_weight"])
    similarity_weight = float(cfg["similarity_weight"])
    denom = max(1e-12, drift_weight + similarity_weight)
    reliability = float(
        np.clip(
            (drift_weight * drift_reliability + similarity_weight * similarity) / denom,
            0.0,
            1.0,
        )
    )
    if reliability < float(cfg["low_reliability_threshold"]):
        reason = "low_reliability_diversify"
    elif reliability >= float(cfg["high_reliability_threshold"]):
        reason = "high_reliability_exploit"
    else:
        reason = "medium_reliability_mixed"
    return {
        "reliability": reliability,
        "similarity": float(np.clip(similarity, 0.0, 1.0)),
        "drift_reliability": float(np.clip(drift_reliability, 0.0, 1.0)),
        "change_score": float(score),
        "component_mean": float(component_mean),
        "reason": reason,
    }


def _evaluate_adaptive_response_guard(
    *,
    cfg: Dict[str, Any],
    regime: Dict[str, Any],
    candidate_x: np.ndarray,
    incumbent_x: Optional[np.ndarray],
    candidate_score: Optional[float],
    incumbent_score: Optional[float],
) -> Dict[str, Any]:
    cfg = _merge_adaptive_response_config(cfg)
    out: Dict[str, Any] = {
        "enabled": bool(cfg.get("guard_enabled", False)),
        "active": False,
        "accepted": True,
        "reason": "disabled",
        "shift_l1": None,
        "score_gain": None,
        "required_score_gain": float(cfg["guard_min_score_gain"]),
        "debt_signal": False,
        "recent_relocation_cost_eur": float(regime.get("recent_relocation_cost_eur", 0.0)),
        "ops_pressure": float(regime.get("ops_pressure", 0.0)),
        "relocating_pressure": float(regime.get("relocating_pressure", 0.0)),
    }
    if not bool(cfg.get("guard_enabled", False)):
        return out
    out["reason"] = "no_incumbent"
    if incumbent_x is None:
        return out
    if candidate_score is None or incumbent_score is None:
        out["reason"] = "missing_scores"
        return out

    candidate = np.asarray(candidate_x, dtype=float)
    incumbent = np.asarray(incumbent_x, dtype=float)
    if candidate.shape != incumbent.shape:
        out["reason"] = "shape_mismatch"
        return out

    shift_l1 = float(np.sum(np.abs(candidate - incumbent)))
    score_gain = float(candidate_score) - float(incumbent_score)
    out["shift_l1"] = shift_l1
    out["score_gain"] = score_gain

    if shift_l1 < float(cfg["guard_shift_l1_threshold"]):
        out["reason"] = "small_shift"
        return out

    debt_signal = bool(
        float(out["recent_relocation_cost_eur"]) >= float(cfg["guard_recent_relocation_cost_min"])
        or float(out["ops_pressure"]) >= float(cfg["guard_ops_pressure_min"])
        or float(out["relocating_pressure"]) >= float(cfg["guard_relocating_pressure_min"])
    )
    out["debt_signal"] = debt_signal
    if bool(cfg["guard_require_debt_signal"]) and not debt_signal:
        out["reason"] = "large_shift_without_debt"
        return out

    out["active"] = True
    if score_gain + 1e-12 < float(cfg["guard_min_score_gain"]):
        out["accepted"] = False
        out["reason"] = "large_shift_under_operational_debt"
    else:
        out["reason"] = "large_shift_score_gain_ok"
    return out


def _merge_reservation_guard_config(provided: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(_DEFAULT_RESERVATION_GUARD)
    if provided is None:
        return cfg
    if not isinstance(provided, dict):
        raise TypeError(
            f"reservation_guard debe ser un dict, recibido {type(provided).__name__}"
        )

    for key, value in provided.items():
        cfg[str(key)] = value

    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["commitment_horizon_fraction"] = float(
        np.clip(float(cfg.get("commitment_horizon_fraction", 0.75)), 0.10, 1.50)
    )
    cfg["due_soon_window_min"] = max(1.0, float(cfg.get("due_soon_window_min", 25.0)))
    cfg["reservation_pressure_scale"] = max(
        1e-6, float(cfg.get("reservation_pressure_scale", 0.55))
    )
    cfg["due_soon_pressure_scale"] = max(
        1e-6, float(cfg.get("due_soon_pressure_scale", 0.30))
    )
    cfg["preassigned_pressure_scale"] = max(
        1e-6, float(cfg.get("preassigned_pressure_scale", 0.25))
    )
    cfg["commitment_pressure_scale"] = max(
        1e-6, float(cfg.get("commitment_pressure_scale", 0.55))
    )
    cfg["selection_smooth_lambda_gain"] = max(
        0.0, float(cfg.get("selection_smooth_lambda_gain", 0.32))
    )
    cfg["incumbent_min_improvement_gain"] = max(
        0.0, float(cfg.get("incumbent_min_improvement_gain", 0.025))
    )
    cfg["react_strength_damping"] = float(
        np.clip(float(cfg.get("react_strength_damping", 0.18)), 0.0, 1.0)
    )
    cfg["immigrant_ratio_damping"] = float(
        np.clip(float(cfg.get("immigrant_ratio_damping", 0.30)), 0.0, 1.0)
    )
    cfg["react_strength_floor_scale"] = float(
        np.clip(float(cfg.get("react_strength_floor_scale", 0.72)), 0.10, 1.0)
    )
    cfg["immigrant_ratio_floor_scale"] = float(
        np.clip(float(cfg.get("immigrant_ratio_floor_scale", 0.60)), 0.05, 1.0)
    )
    cfg["predictive_disable_threshold"] = float(
        np.clip(float(cfg.get("predictive_disable_threshold", 0.85)), 0.0, 2.0)
    )
    cfg["front_ls_disable_threshold"] = float(
        np.clip(float(cfg.get("front_ls_disable_threshold", 0.60)), 0.0, 2.0)
    )
    cfg["predictive_disable_relocation_threshold"] = float(
        np.clip(float(cfg.get("predictive_disable_relocation_threshold", 2.0)), 0.0, 2.0)
    )
    cfg["front_ls_disable_relocation_threshold"] = float(
        np.clip(float(cfg.get("front_ls_disable_relocation_threshold", 2.0)), 0.0, 2.0)
    )
    cfg["mild_step_budget_scale"] = float(
        np.clip(float(cfg.get("mild_step_budget_scale", 0.85)), 0.10, 1.0)
    )
    cfg["medium_step_budget_scale"] = float(
        np.clip(float(cfg.get("medium_step_budget_scale", 0.95)), 0.10, 1.0)
    )
    cfg["max_interval_step_budget_scale"] = float(
        np.clip(float(cfg.get("max_interval_step_budget_scale", 0.90)), 0.10, 1.0)
    )
    cfg["relocating_pressure_scale"] = max(
        1e-6, float(cfg.get("relocating_pressure_scale", 0.08))
    )
    cfg["recent_deadhead_km_scale"] = max(
        1e-6, float(cfg.get("recent_deadhead_km_scale", 8.0))
    )
    cfg["recent_ops_cost_scale"] = max(
        1e-6, float(cfg.get("recent_ops_cost_scale", 8.0))
    )
    cfg["recent_relocation_cost_scale"] = max(
        1e-6, float(cfg.get("recent_relocation_cost_scale", 8.0))
    )
    cfg["relocating_pressure_weight"] = max(
        0.0, float(cfg.get("relocating_pressure_weight", 0.0))
    )
    cfg["recent_deadhead_km_weight"] = max(
        0.0, float(cfg.get("recent_deadhead_km_weight", 0.0))
    )
    cfg["recent_ops_cost_weight"] = max(
        0.0, float(cfg.get("recent_ops_cost_weight", 0.0))
    )
    cfg["recent_relocation_cost_weight"] = max(
        0.0, float(cfg.get("recent_relocation_cost_weight", 0.0))
    )
    cfg["relocation_debt_signal_weight"] = max(
        0.0, float(cfg.get("relocation_debt_signal_weight", 0.0))
    )
    cfg["demand_pressure_scale"] = max(
        1e-6, float(cfg.get("demand_pressure_scale", 6.0))
    )
    cfg["coverage_gap_overload_scale"] = max(
        1e-6, float(cfg.get("coverage_gap_overload_scale", 0.20))
    )
    cfg["recent_reject_rate_overload_scale"] = max(
        1e-6, float(cfg.get("recent_reject_rate_overload_scale", 0.08))
    )
    cfg["demand_pressure_overload_weight"] = max(
        0.0, float(cfg.get("demand_pressure_overload_weight", 0.0))
    )
    cfg["coverage_gap_overload_weight"] = max(
        0.0, float(cfg.get("coverage_gap_overload_weight", 0.0))
    )
    cfg["recent_reject_rate_overload_weight"] = max(
        0.0, float(cfg.get("recent_reject_rate_overload_weight", 0.0))
    )
    cfg["overload_relief_weight"] = max(
        0.0, float(cfg.get("overload_relief_weight", 0.0))
    )
    cfg["overload_unsuppress_threshold"] = float(
        np.clip(float(cfg.get("overload_unsuppress_threshold", 2.0)), 0.0, 2.0)
    )
    cfg["max_interval_hold_threshold"] = float(
        np.clip(float(cfg.get("max_interval_hold_threshold", 1.1)), 0.0, 2.0)
    )
    cfg["hold_service_reject_rate_max"] = float(
        np.clip(float(cfg.get("hold_service_reject_rate_max", 0.08)), 0.0, 1.0)
    )
    cfg["hold_service_coverage_gap_max"] = float(
        np.clip(float(cfg.get("hold_service_coverage_gap_max", 0.35)), 0.0, 1.0)
    )
    cfg["cost_guard_enabled"] = bool(cfg.get("cost_guard_enabled", False))
    cfg["cost_guard_disable_on_bootstrap"] = bool(
        cfg.get("cost_guard_disable_on_bootstrap", False)
    )
    cfg["cost_guard_activation_signal_min"] = float(
        np.clip(float(cfg.get("cost_guard_activation_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["cost_guard_activation_relocation_signal_min"] = float(
        np.clip(float(cfg.get("cost_guard_activation_relocation_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["cost_guard_activation_demand_signal_min"] = float(
        np.clip(float(cfg.get("cost_guard_activation_demand_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["cost_guard_activation_coverage_gap_signal_min"] = float(
        np.clip(float(cfg.get("cost_guard_activation_coverage_gap_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["cost_guard_activation_recent_reject_signal_min"] = float(
        np.clip(float(cfg.get("cost_guard_activation_recent_reject_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["cost_guard_burst_enabled"] = bool(cfg.get("cost_guard_burst_enabled", False))
    cfg["cost_guard_burst_commitment_signal_min"] = float(
        np.clip(float(cfg.get("cost_guard_burst_commitment_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["cost_guard_burst_relocation_signal_min"] = float(
        np.clip(float(cfg.get("cost_guard_burst_relocation_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["candidate_shift_penalty_burst_max"] = max(
        0.0, float(cfg.get("candidate_shift_penalty_burst_max", 1e9))
    )
    cfg["candidate_min_profit_gain_eur_for_burst_move"] = float(
        cfg.get("candidate_min_profit_gain_eur_for_burst_move", 0.0)
    )
    cfg["candidate_extra_deadhead_km_max"] = max(
        0.0, float(cfg.get("candidate_extra_deadhead_km_max", 1e9))
    )
    cfg["candidate_extra_operational_cost_eur_max"] = max(
        0.0, float(cfg.get("candidate_extra_operational_cost_eur_max", 1e9))
    )
    cfg["candidate_extra_relocation_cost_eur_max"] = max(
        0.0, float(cfg.get("candidate_extra_relocation_cost_eur_max", 1e9))
    )
    cfg["candidate_min_profit_gain_eur_for_costly_move"] = float(
        cfg.get("candidate_min_profit_gain_eur_for_costly_move", 0.0)
    )
    cfg["service_deployment_guard_enabled"] = bool(
        cfg.get("service_deployment_guard_enabled", False)
    )
    cfg["service_guard_disable_on_bootstrap"] = bool(
        cfg.get("service_guard_disable_on_bootstrap", True)
    )
    cfg["service_guard_activation_demand_signal_min"] = float(
        np.clip(float(cfg.get("service_guard_activation_demand_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["service_guard_activation_coverage_gap_signal_min"] = float(
        np.clip(float(cfg.get("service_guard_activation_coverage_gap_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["service_guard_activation_recent_reject_signal_min"] = float(
        np.clip(float(cfg.get("service_guard_activation_recent_reject_signal_min", 0.0)), 0.0, 2.0)
    )
    cfg["service_guard_max_service_drop"] = max(
        0.0, float(cfg.get("service_guard_max_service_drop", 0.0))
    )
    cfg["service_guard_max_extra_rejected"] = max(
        0.0, float(cfg.get("service_guard_max_extra_rejected", 0.0))
    )
    cfg["service_guard_max_extra_deadhead_km"] = max(
        0.0, float(cfg.get("service_guard_max_extra_deadhead_km", 1e9))
    )
    cfg["service_guard_min_profit_gain_eur"] = float(
        cfg.get("service_guard_min_profit_gain_eur", -1e9)
    )
    cfg["service_guard_require_opportunity"] = bool(
        cfg.get("service_guard_require_opportunity", False)
    )
    cfg["service_guard_min_service_gain"] = max(
        0.0, float(cfg.get("service_guard_min_service_gain", 0.0))
    )
    cfg["service_guard_min_rejected_reduction"] = max(
        0.0, float(cfg.get("service_guard_min_rejected_reduction", 0.0))
    )
    cfg["service_guard_neutral_min_profit_gain_eur"] = float(
        cfg.get("service_guard_neutral_min_profit_gain_eur", -1e9)
    )
    return cfg


def _update_regime_reference_ewma(
    reference: Optional[Dict[str, float]],
    current: Dict[str, float],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    cfg = _merge_problem_detector_config(cfg)
    if reference is None:
        return {str(k): float(v) for k, v in current.items()}
    alpha = float(cfg["reference_ewma_alpha"])
    out: Dict[str, float] = {}
    keys = set(reference.keys()) | set(current.keys())
    for key in keys:
        cur = float(current.get(key, reference.get(key, 0.0)))
        ref = float(reference.get(key, cur))
        out[str(key)] = float(alpha * cur + (1.0 - alpha) * ref)
    return out


def _extract_regime_features(
    problem: FreeFloatingRobustDMOOP,
    window: Dict[str, Any],
    carry_state: Optional[SystemState],
    cfg: Optional[Dict[str, Any]] = None,
    reservation_guard_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    state = carry_state if carry_state is not None else problem.state_template
    now_min = float(state.current_time_min)
    detector_cfg = _merge_problem_detector_config(cfg)
    reservation_cfg = _merge_reservation_guard_config(reservation_guard_cfg)
    low_soc_threshold = float(detector_cfg["low_soc_threshold"])

    demand_multiplier = window.get("demand_multiplier")
    if demand_multiplier is None:
        demand_multiplier = window.get("demand_shock")

    demand_forecast = problem.scenario.forecast_demand(
        now_min=now_min,
        horizon_min=float(problem.config.planning_horizon_min),
        external_multiplier=demand_multiplier,
    )

    vehicles = list(state.vehicles.values())
    total_vehicles = max(1, len(vehicles))
    idle_vehicles = [
        v
        for v in vehicles
        if v.status == VehicleStatus.IDLE and float(v.available_at_min) <= now_min + 1e-9
    ]
    idle_count = len(idle_vehicles)

    low_soc_all = sum(1 for v in vehicles if float(v.soc) <= low_soc_threshold)
    charging_count = sum(1 for v in vehicles if v.status == VehicleStatus.CHARGING)
    relocating_count = sum(1 for v in vehicles if v.status == VehicleStatus.RELOCATING)
    pending_reservations = int(len(state.pending_reservations))
    planning_horizon_min = max(1.0, float(problem.config.planning_horizon_min))
    commitment_horizon_min = max(
        float(problem.config.decision_interval_min),
        float(reservation_cfg["commitment_horizon_fraction"]) * planning_horizon_min,
    )
    due_soon_window_min = min(
        commitment_horizon_min,
        float(reservation_cfg["due_soon_window_min"]),
    )
    due_soon_pending = 0
    preassigned_pending = 0
    commitment_load = 0.0
    for req in state.pending_reservations.values():
        pickup_time_min = (
            float(req.pickup_time_min)
            if req.pickup_time_min is not None
            else float(req.request_time_min)
        )
        lead_min = max(0.0, pickup_time_min - now_min)
        lead_signal = float(
            np.clip(1.0 - lead_min / max(commitment_horizon_min, 1e-9), 0.0, 1.0)
        )
        due_soon = bool(lead_min <= due_soon_window_min + 1e-9)
        preassigned = bool(req.preassigned_vehicle_id is not None)
        if due_soon:
            due_soon_pending += 1
        if preassigned:
            preassigned_pending += 1
        # Reservations close in time and already preassigned are harder to disturb safely.
        weight = 0.35 + 0.45 * lead_signal
        if due_soon:
            weight += 0.15
        if preassigned:
            weight += 0.20
        commitment_load += float(weight)
    total_demand = float(sum(float(v) for v in demand_forecast.values()))
    free_charger_slots = int(sum(max(0, int(ch.free_slots)) for ch in state.chargers.values()))
    recent_service_signal = window.get("recent_service_signal", {})
    if not isinstance(recent_service_signal, dict):
        recent_service_signal = {}
    recent_requests = float(recent_service_signal.get("requests", 0.0))
    recent_rejected = float(recent_service_signal.get("rejected", 0.0))
    recent_ops_deadhead_km = float(recent_service_signal.get("ops_deadhead_km", 0.0))
    recent_ops_cost_eur = float(recent_service_signal.get("ops_cost_eur", 0.0))
    recent_relocation_cost_eur = float(recent_service_signal.get("relocation_cost_eur", 0.0))
    recent_reject_rate = (
        0.0 if recent_requests <= 1e-9 else float(np.clip(recent_rejected / max(recent_requests, 1.0), 0.0, 1.0))
    )

    idle_by_zone = {zone_id: 0 for zone_id in state.zones}
    for vehicle in idle_vehicles:
        idle_by_zone[vehicle.zone_id] = int(idle_by_zone.get(vehicle.zone_id, 0) + 1)

    max_walk = float(getattr(problem.config, "max_user_access_walk_min", 0.0))
    walk_speed = float(getattr(problem.config, "user_walk_speed_kmh", 4.6))
    covered_demand = 0.0
    if total_demand > 1e-9:
        idle_zone_ids = [zone_id for zone_id, count in idle_by_zone.items() if int(count) > 0]
        for zone_id in state.zones:
            zone_demand = float(demand_forecast.get(zone_id, 0.0))
            if zone_demand <= 1e-9:
                continue
            covered = False
            if zone_id in idle_zone_ids:
                covered = True
            elif max_walk <= 0.0:
                covered = bool(idle_zone_ids)
            else:
                for idle_zone_id in idle_zone_ids:
                    d_km = zone_distance_km(
                        zone_id,
                        idle_zone_id,
                        state.zones,
                        travel_metrics=state.travel_metrics,
                    )
                    if walk_time_min(d_km, walk_speed) <= max_walk + 1e-9:
                        covered = True
                        break
            if covered:
                covered_demand += zone_demand

    if idle_count <= 0 and total_demand > 1e-9:
        spatial_imbalance = 1.0
    elif total_demand <= 1e-9:
        spatial_imbalance = 0.0
    else:
        demand_share = {
            zone_id: float(demand_forecast.get(zone_id, 0.0)) / max(total_demand, 1e-9)
            for zone_id in state.zones
        }
        idle_share = {
            zone_id: float(idle_by_zone.get(zone_id, 0)) / max(1, idle_count)
            for zone_id in state.zones
        }
        spatial_imbalance = 0.5 * sum(
            abs(float(idle_share.get(zone_id, 0.0)) - float(demand_share.get(zone_id, 0.0)))
            for zone_id in state.zones
        )

    return {
        "time_min": now_min,
        "idle_ratio": float(idle_count) / float(total_vehicles),
        "demand_pressure": float(total_demand) / float(max(1, idle_count)),
        "reservation_pressure": float(pending_reservations) / float(max(1, idle_count)),
        "reservation_due_soon_pressure": float(due_soon_pending) / float(max(1, idle_count)),
        "preassigned_reservation_pressure": float(preassigned_pending) / float(max(1, idle_count)),
        "commitment_pressure": float(commitment_load) / float(max(1, idle_count)),
        "relocating_pressure": float(relocating_count) / float(total_vehicles),
        "recent_ops_deadhead_km": float(max(0.0, recent_ops_deadhead_km)),
        "recent_ops_cost_eur": float(max(0.0, recent_ops_cost_eur)),
        "recent_relocation_cost_eur": float(max(0.0, recent_relocation_cost_eur)),
        "low_soc_share": float(low_soc_all) / float(total_vehicles),
        "charger_pressure": float(low_soc_all + charging_count) / float(max(1, free_charger_slots + charging_count)),
        "ops_pressure": float(
            sum(1 for r in state.ops_resources.values() if r.status != OpsStatus.IDLE)
        )
        / float(max(1, len(state.ops_resources))),
        "spatial_imbalance": float(np.clip(spatial_imbalance, 0.0, 1.0)),
        "coverage_gap": float(
            0.0 if total_demand <= 1e-9 else np.clip(1.0 - covered_demand / max(total_demand, 1e-9), 0.0, 1.0)
        ),
        "recent_request_count": float(max(0.0, recent_requests)),
        "recent_reject_rate": float(recent_reject_rate),
    }


def _compute_reservation_guard_controls(
    *,
    regime: Dict[str, float],
    trigger_info: Dict[str, Any],
    reservation_guard_cfg: Optional[Dict[str, Any]],
    base_selection_smooth_lambda: float,
    base_incumbent_min_improvement: float,
) -> Dict[str, Any]:
    cfg = _merge_reservation_guard_config(reservation_guard_cfg)
    reason = str(trigger_info.get("reason", "always")).strip().lower()
    if not bool(cfg["enabled"]):
        return {
            "enabled": False,
            "signal": 0.0,
            "reservation_signal": 0.0,
            "due_soon_signal": 0.0,
            "preassigned_signal": 0.0,
            "commitment_signal": 0.0,
            "effective_selection_smooth_lambda": float(base_selection_smooth_lambda),
            "effective_incumbent_min_improvement": float(base_incumbent_min_improvement),
            "reaction_scale": 1.0,
            "immigrant_scale": 1.0,
            "predictive_enabled": True,
            "front_ls_enabled": True,
            "step_budget_scale": 1.0,
            "relocating_signal": 0.0,
            "recent_deadhead_signal": 0.0,
            "recent_ops_cost_signal": 0.0,
            "recent_relocation_cost_signal": 0.0,
            "relocation_signal": 0.0,
            "demand_signal": 0.0,
            "coverage_gap_signal": 0.0,
            "recent_reject_signal": 0.0,
            "overload_signal": 0.0,
            "suppress_reopt": False,
            "cost_guard_enabled": False,
            "cost_guard_active": False,
            "cost_guard_bootstrap_ready": False,
            "cost_guard_pressure_active": False,
            "cost_guard_burst_active": False,
            "candidate_shift_penalty_burst_max": 1e9,
            "candidate_min_profit_gain_eur_for_burst_move": 0.0,
            "candidate_extra_deadhead_km_max": 1e9,
            "candidate_extra_operational_cost_eur_max": 1e9,
            "candidate_extra_relocation_cost_eur_max": 1e9,
            "candidate_min_profit_gain_eur_for_costly_move": 0.0,
            "service_deployment_guard_enabled": False,
            "service_deployment_guard_active": False,
            "service_guard_pressure_active": False,
            "service_guard_max_service_drop": 0.0,
            "service_guard_max_extra_rejected": 0.0,
            "service_guard_max_extra_deadhead_km": 1e9,
            "service_guard_min_profit_gain_eur": -1e9,
            "service_guard_require_opportunity": False,
            "service_guard_min_service_gain": 0.0,
            "service_guard_min_rejected_reduction": 0.0,
            "service_guard_neutral_min_profit_gain_eur": -1e9,
        }

    reservation_signal = float(
        np.clip(
            float(regime.get("reservation_pressure", 0.0)) / float(cfg["reservation_pressure_scale"]),
            0.0,
            1.5,
        )
    )
    due_soon_signal = float(
        np.clip(
            float(regime.get("reservation_due_soon_pressure", 0.0)) / float(cfg["due_soon_pressure_scale"]),
            0.0,
            1.5,
        )
    )
    preassigned_signal = float(
        np.clip(
            float(regime.get("preassigned_reservation_pressure", 0.0))
            / float(cfg["preassigned_pressure_scale"]),
            0.0,
            1.5,
        )
    )
    commitment_signal = float(
        np.clip(
            float(regime.get("commitment_pressure", 0.0)) / float(cfg["commitment_pressure_scale"]),
            0.0,
            1.5,
        )
    )
    relocating_signal = float(
        np.clip(
            float(regime.get("relocating_pressure", 0.0)) / float(cfg["relocating_pressure_scale"]),
            0.0,
            1.5,
        )
    )
    recent_deadhead_signal = float(
        np.clip(
            float(regime.get("recent_ops_deadhead_km", 0.0)) / float(cfg["recent_deadhead_km_scale"]),
            0.0,
            1.5,
        )
    )
    recent_ops_cost_signal = float(
        np.clip(
            float(regime.get("recent_ops_cost_eur", 0.0)) / float(cfg["recent_ops_cost_scale"]),
            0.0,
            1.5,
        )
    )
    recent_relocation_cost_signal = float(
        np.clip(
            float(regime.get("recent_relocation_cost_eur", 0.0))
            / float(cfg["recent_relocation_cost_scale"]),
            0.0,
            1.5,
        )
    )
    relocation_weight_sum = (
        float(cfg["relocating_pressure_weight"])
        + float(cfg["recent_deadhead_km_weight"])
        + float(cfg["recent_ops_cost_weight"])
        + float(cfg["recent_relocation_cost_weight"])
    )
    relocation_signal = 0.0
    if relocation_weight_sum > 1e-9:
        relocation_signal = float(
            np.clip(
                (
                    float(cfg["relocating_pressure_weight"]) * relocating_signal
                    + float(cfg["recent_deadhead_km_weight"]) * recent_deadhead_signal
                    + float(cfg["recent_ops_cost_weight"]) * recent_ops_cost_signal
                    + float(cfg["recent_relocation_cost_weight"]) * recent_relocation_cost_signal
                )
                / relocation_weight_sum,
                0.0,
                1.5,
        )
    )
    demand_signal = float(
        np.clip(
            float(regime.get("demand_pressure", 0.0)) / float(cfg["demand_pressure_scale"]),
            0.0,
            1.5,
        )
    )
    coverage_gap_signal = float(
        np.clip(
            float(regime.get("coverage_gap", 0.0)) / float(cfg["coverage_gap_overload_scale"]),
            0.0,
            1.5,
        )
    )
    recent_reject_signal = float(
        np.clip(
            float(regime.get("recent_reject_rate", 0.0))
            / float(cfg["recent_reject_rate_overload_scale"]),
            0.0,
            1.5,
        )
    )
    overload_weight_sum = (
        float(cfg["demand_pressure_overload_weight"])
        + float(cfg["coverage_gap_overload_weight"])
        + float(cfg["recent_reject_rate_overload_weight"])
    )
    overload_signal = 0.0
    if overload_weight_sum > 1e-9:
        overload_signal = float(
            np.clip(
                (
                    float(cfg["demand_pressure_overload_weight"]) * demand_signal
                    + float(cfg["coverage_gap_overload_weight"]) * coverage_gap_signal
                    + float(cfg["recent_reject_rate_overload_weight"]) * recent_reject_signal
                )
                / overload_weight_sum,
                0.0,
                1.5,
            )
        )
    base_signal = float(
        np.clip(
            0.10 * reservation_signal
            + 0.35 * due_soon_signal
            + 0.20 * preassigned_signal
            + 0.35 * commitment_signal,
            0.0,
            1.5,
        )
    )
    signal = float(
        np.clip(
            base_signal
            + float(cfg["relocation_debt_signal_weight"]) * relocation_signal
            - float(cfg["overload_relief_weight"]) * overload_signal,
            0.0,
            1.5,
        )
    )

    reaction_scale = float(
        max(float(cfg["react_strength_floor_scale"]), 1.0 - float(cfg["react_strength_damping"]) * signal)
    )
    immigrant_scale = float(
        max(
            float(cfg["immigrant_ratio_floor_scale"]),
            1.0 - float(cfg["immigrant_ratio_damping"]) * signal,
        )
    )
    step_budget_scale = 1.0
    if reason == "mild":
        floor = float(cfg["mild_step_budget_scale"])
        step_budget_scale = float(max(floor, 1.0 - (1.0 - floor) * min(signal, 1.0)))
    elif reason == "medium":
        floor = float(cfg["medium_step_budget_scale"])
        step_budget_scale = float(max(floor, 1.0 - (1.0 - floor) * min(signal, 1.0)))
    elif reason == "max_interval":
        floor = float(cfg["max_interval_step_budget_scale"])
        step_budget_scale = float(max(floor, 1.0 - (1.0 - floor) * min(signal, 1.0)))

    service_safe_to_hold = bool(
        float(regime.get("recent_reject_rate", 0.0)) <= float(cfg["hold_service_reject_rate_max"])
        and float(regime.get("coverage_gap", 0.0)) <= float(cfg["hold_service_coverage_gap_max"])
    )
    suppress_reopt = bool(
        reason == "max_interval"
        and signal >= float(cfg["max_interval_hold_threshold"])
        and overload_signal < float(cfg["overload_unsuppress_threshold"])
        and service_safe_to_hold
    )
    cost_guard_bootstrap_ready = bool(
        reason != "bootstrap" or not bool(cfg["cost_guard_disable_on_bootstrap"])
    )
    pressure_gate_thresholds = (
        float(cfg["cost_guard_activation_demand_signal_min"]),
        float(cfg["cost_guard_activation_coverage_gap_signal_min"]),
        float(cfg["cost_guard_activation_recent_reject_signal_min"]),
    )
    pressure_gate_configured = any(threshold > 1e-9 for threshold in pressure_gate_thresholds)
    cost_guard_pressure_active = True
    if pressure_gate_configured:
        cost_guard_pressure_active = bool(
            demand_signal >= float(cfg["cost_guard_activation_demand_signal_min"])
            or coverage_gap_signal >= float(cfg["cost_guard_activation_coverage_gap_signal_min"])
            or recent_reject_signal >= float(cfg["cost_guard_activation_recent_reject_signal_min"])
        )
    cost_guard_burst_active = bool(
        bool(cfg["cost_guard_burst_enabled"])
        and commitment_signal >= float(cfg["cost_guard_burst_commitment_signal_min"])
        and relocation_signal >= float(cfg["cost_guard_burst_relocation_signal_min"])
    )
    cost_guard_active = bool(
        bool(cfg["cost_guard_enabled"])
        and cost_guard_bootstrap_ready
        and (cost_guard_pressure_active or cost_guard_burst_active)
        and (
            signal >= float(cfg["cost_guard_activation_signal_min"])
            or relocation_signal >= float(cfg["cost_guard_activation_relocation_signal_min"])
            or cost_guard_burst_active
        )
    )
    service_guard_bootstrap_ready = bool(
        reason != "bootstrap" or not bool(cfg["service_guard_disable_on_bootstrap"])
    )
    service_pressure_thresholds = (
        float(cfg["service_guard_activation_demand_signal_min"]),
        float(cfg["service_guard_activation_coverage_gap_signal_min"]),
        float(cfg["service_guard_activation_recent_reject_signal_min"]),
    )
    service_pressure_configured = any(threshold > 1e-9 for threshold in service_pressure_thresholds)
    service_guard_pressure_active = True
    if service_pressure_configured:
        service_guard_pressure_active = bool(
            demand_signal >= float(cfg["service_guard_activation_demand_signal_min"])
            or coverage_gap_signal >= float(cfg["service_guard_activation_coverage_gap_signal_min"])
            or recent_reject_signal >= float(cfg["service_guard_activation_recent_reject_signal_min"])
        )
    service_deployment_guard_active = bool(
        bool(cfg["service_deployment_guard_enabled"])
        and service_guard_bootstrap_ready
        and service_guard_pressure_active
    )

    return {
        "enabled": True,
        "signal": signal,
        "reservation_signal": reservation_signal,
        "due_soon_signal": due_soon_signal,
        "preassigned_signal": preassigned_signal,
        "commitment_signal": commitment_signal,
        "relocating_signal": relocating_signal,
        "recent_deadhead_signal": recent_deadhead_signal,
        "recent_ops_cost_signal": recent_ops_cost_signal,
        "recent_relocation_cost_signal": recent_relocation_cost_signal,
        "relocation_signal": relocation_signal,
        "demand_signal": demand_signal,
        "coverage_gap_signal": coverage_gap_signal,
        "recent_reject_signal": recent_reject_signal,
        "overload_signal": overload_signal,
        "effective_selection_smooth_lambda": float(
            base_selection_smooth_lambda + float(cfg["selection_smooth_lambda_gain"]) * signal
        ),
        "effective_incumbent_min_improvement": float(
            base_incumbent_min_improvement + float(cfg["incumbent_min_improvement_gain"]) * signal
        ),
        "reaction_scale": reaction_scale,
        "immigrant_scale": immigrant_scale,
        "predictive_enabled": bool(
            signal < float(cfg["predictive_disable_threshold"])
            and relocation_signal < float(cfg["predictive_disable_relocation_threshold"])
        ),
        "front_ls_enabled": bool(
            signal < float(cfg["front_ls_disable_threshold"])
            and relocation_signal < float(cfg["front_ls_disable_relocation_threshold"])
        ),
        "step_budget_scale": step_budget_scale,
        "suppress_reopt": suppress_reopt,
        "suppress_reason": "reservation_guard_max_interval_hold" if suppress_reopt else "",
        "cost_guard_enabled": bool(cfg["cost_guard_enabled"]),
        "cost_guard_active": cost_guard_active,
        "cost_guard_bootstrap_ready": cost_guard_bootstrap_ready,
        "cost_guard_pressure_active": cost_guard_pressure_active,
        "cost_guard_burst_active": cost_guard_burst_active,
        "candidate_shift_penalty_burst_max": float(cfg["candidate_shift_penalty_burst_max"]),
        "candidate_min_profit_gain_eur_for_burst_move": float(
            cfg["candidate_min_profit_gain_eur_for_burst_move"]
        ),
        "candidate_extra_deadhead_km_max": float(cfg["candidate_extra_deadhead_km_max"]),
        "candidate_extra_operational_cost_eur_max": float(
            cfg["candidate_extra_operational_cost_eur_max"]
        ),
        "candidate_extra_relocation_cost_eur_max": float(
            cfg["candidate_extra_relocation_cost_eur_max"]
        ),
        "candidate_min_profit_gain_eur_for_costly_move": float(
            cfg["candidate_min_profit_gain_eur_for_costly_move"]
        ),
        "service_deployment_guard_enabled": bool(cfg["service_deployment_guard_enabled"]),
        "service_deployment_guard_active": service_deployment_guard_active,
        "service_guard_pressure_active": service_guard_pressure_active,
        "service_guard_max_service_drop": float(cfg["service_guard_max_service_drop"]),
        "service_guard_max_extra_rejected": float(cfg["service_guard_max_extra_rejected"]),
        "service_guard_max_extra_deadhead_km": float(cfg["service_guard_max_extra_deadhead_km"]),
        "service_guard_min_profit_gain_eur": float(cfg["service_guard_min_profit_gain_eur"]),
        "service_guard_require_opportunity": bool(cfg["service_guard_require_opportunity"]),
        "service_guard_min_service_gain": float(cfg["service_guard_min_service_gain"]),
        "service_guard_min_rejected_reduction": float(
            cfg["service_guard_min_rejected_reduction"]
        ),
        "service_guard_neutral_min_profit_gain_eur": float(
            cfg["service_guard_neutral_min_profit_gain_eur"]
        ),
    }


def _operational_cost_from_details(details: Dict[str, Any]) -> float:
    return float(details.get("cost_relocation_eur", 0.0)) + float(
        details.get("cost_ops_eur", 0.0)
    ) + float(details.get("cost_ops_deadhead_eur", 0.0))


def _evaluate_candidate_cost_guard(
    *,
    controls: Dict[str, Any],
    candidate_details: Dict[str, Any],
    incumbent_details: Optional[Dict[str, Any]],
    candidate_shift_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    out = {
        "active": bool(
            controls.get("cost_guard_active", False)
            or controls.get("service_deployment_guard_active", False)
        ),
        "cost_guard_active": bool(controls.get("cost_guard_active", False)),
        "service_deployment_guard_active": bool(
            controls.get("service_deployment_guard_active", False)
        ),
        "accepted": True,
        "reasons": [],
        "extra_deadhead_km": 0.0,
        "extra_operational_cost_eur": 0.0,
        "extra_relocation_cost_eur": 0.0,
        "profit_gain_eur": 0.0,
        "service_delta": 0.0,
        "extra_rejected_requests": 0.0,
        "candidate_shift_penalty": None
        if candidate_shift_penalty is None
        else float(candidate_shift_penalty),
    }
    guard_active = bool(
        controls.get("cost_guard_active", False)
        or controls.get("service_deployment_guard_active", False)
    )
    if not guard_active or incumbent_details is None:
        return out

    candidate_profit = float(candidate_details.get("profit_eur", 0.0))
    incumbent_profit = float(incumbent_details.get("profit_eur", 0.0))
    profit_gain = candidate_profit - incumbent_profit
    candidate_deadhead = float(candidate_details.get("ops_deadhead_km_total", 0.0))
    incumbent_deadhead = float(incumbent_details.get("ops_deadhead_km_total", 0.0))
    extra_deadhead = candidate_deadhead - incumbent_deadhead
    candidate_relocation_cost = float(candidate_details.get("cost_relocation_eur", 0.0))
    incumbent_relocation_cost = float(incumbent_details.get("cost_relocation_eur", 0.0))
    extra_relocation_cost = candidate_relocation_cost - incumbent_relocation_cost
    extra_operational_cost = _operational_cost_from_details(candidate_details) - _operational_cost_from_details(
        incumbent_details
    )
    candidate_service = float(
        candidate_details.get("service_rate", candidate_details.get("window_service_rate", 0.0))
    )
    incumbent_service = float(
        incumbent_details.get("service_rate", incumbent_details.get("window_service_rate", 0.0))
    )
    service_delta = candidate_service - incumbent_service
    candidate_rejected = float(candidate_details.get("rejected_requests", 0.0))
    incumbent_rejected = float(incumbent_details.get("rejected_requests", 0.0))
    extra_rejected = candidate_rejected - incumbent_rejected

    out["extra_deadhead_km"] = float(extra_deadhead)
    out["extra_operational_cost_eur"] = float(extra_operational_cost)
    out["extra_relocation_cost_eur"] = float(extra_relocation_cost)
    out["profit_gain_eur"] = float(profit_gain)
    out["service_delta"] = float(service_delta)
    out["extra_rejected_requests"] = float(extra_rejected)

    min_profit_gain = float(controls.get("candidate_min_profit_gain_eur_for_costly_move", 0.0))
    reasons: List[str] = []
    if bool(controls.get("cost_guard_active", False)):
        if extra_deadhead > float(controls.get("candidate_extra_deadhead_km_max", 1e9)) and profit_gain < min_profit_gain:
            reasons.append("extra_deadhead")
        if extra_operational_cost > float(
            controls.get("candidate_extra_operational_cost_eur_max", 1e9)
        ) and profit_gain < min_profit_gain:
            reasons.append("extra_operational_cost")
        if extra_relocation_cost > float(
            controls.get("candidate_extra_relocation_cost_eur_max", 1e9)
        ) and profit_gain < min_profit_gain:
            reasons.append("extra_relocation_cost")
    if (
        bool(controls.get("cost_guard_burst_active", False))
        and candidate_shift_penalty is not None
        and float(candidate_shift_penalty)
        > float(controls.get("candidate_shift_penalty_burst_max", 1e9))
        and (
            extra_deadhead > 1e-9
            or extra_relocation_cost > 1e-9
            or extra_operational_cost > 1e-9
        )
        and profit_gain
        < float(controls.get("candidate_min_profit_gain_eur_for_burst_move", 0.0))
    ):
        reasons.append("burst_shift_move")
    if bool(controls.get("service_deployment_guard_active", False)):
        service_drop = -service_delta
        rejected_reduction = -extra_rejected
        if service_drop > float(controls.get("service_guard_max_service_drop", 0.0)):
            reasons.append("service_drop")
        if extra_rejected > float(controls.get("service_guard_max_extra_rejected", 0.0)):
            reasons.append("extra_rejected")
        if (
            extra_deadhead > float(controls.get("service_guard_max_extra_deadhead_km", 1e9))
            and profit_gain < float(controls.get("service_guard_min_profit_gain_eur", -1e9))
        ):
            reasons.append("service_guard_extra_deadhead")
        if profit_gain < float(controls.get("service_guard_min_profit_gain_eur", -1e9)):
            reasons.append("service_guard_profit_floor")
        if bool(controls.get("service_guard_require_opportunity", False)):
            service_opportunity = bool(
                service_delta > float(controls.get("service_guard_min_service_gain", 0.0))
                or rejected_reduction
                > float(controls.get("service_guard_min_rejected_reduction", 0.0))
            )
            neutral_service_ok = bool(
                service_drop <= float(controls.get("service_guard_max_service_drop", 0.0))
                and extra_rejected
                <= float(controls.get("service_guard_max_extra_rejected", 0.0))
                and extra_deadhead
                <= float(controls.get("service_guard_max_extra_deadhead_km", 1e9))
            )
            neutral_margin_ok = bool(
                profit_gain
                >= float(controls.get("service_guard_neutral_min_profit_gain_eur", -1e9))
            )
            if not service_opportunity and not (neutral_service_ok and neutral_margin_ok):
                reasons.append("no_service_opportunity")
    if reasons:
        out["accepted"] = False
        out["reasons"] = reasons
    return out


def _apply_reservation_guard_controls(
    *,
    optimizer: Any,
    controls: Dict[str, Any],
) -> None:
    base = getattr(optimizer, "_reservation_guard_base", None)
    if not isinstance(base, dict):
        return

    reaction_scale = float(controls.get("reaction_scale", 1.0))
    immigrant_scale = float(controls.get("immigrant_scale", 1.0))

    optimizer.react_strength = float(base["react_strength"]) * reaction_scale
    optimizer.react_strength_mild = float(base["react_strength_mild"]) * reaction_scale
    optimizer.react_strength_medium = float(base["react_strength_medium"]) * reaction_scale
    optimizer.immigrant_ratio = float(np.clip(float(base["immigrant_ratio"]) * immigrant_scale, 0.0, 1.0))
    optimizer.immigrant_ratio_mild = float(
        np.clip(float(base["immigrant_ratio_mild"]) * immigrant_scale, 0.0, 1.0)
    )
    optimizer.immigrant_ratio_medium = float(
        np.clip(float(base["immigrant_ratio_medium"]) * immigrant_scale, 0.0, 1.0)
    )
    optimizer.predictive_block_enabled = bool(base["predictive_block_enabled"]) and bool(
        controls.get("predictive_enabled", True)
    )
    optimizer.continuity_block_enabled = bool(base["continuity_block_enabled"])
    optimizer.adr_front_ls_enabled = bool(base["adr_front_ls_enabled"]) and bool(
        controls.get("front_ls_enabled", True)
    )


def _score_regime_shift(
    current: Dict[str, float],
    reference: Optional[Dict[str, float]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _merge_problem_detector_config(cfg)
    if reference is None:
        return {
            "change_detected": False,
            "score": 0.0,
            "profile_hint": "bootstrap",
            "breakdown": {},
            "hard_trigger": False,
        }

    weights = dict(cfg.get("weights", {}))
    scales = dict(cfg.get("scales", {}))
    total_weight = max(1e-9, float(sum(max(0.0, float(v)) for v in weights.values())))
    breakdown: Dict[str, float] = {}
    score = 0.0
    for name, raw_weight in weights.items():
        weight = max(0.0, float(raw_weight))
        scale = max(1e-9, float(scales.get(name, 1.0)))
        cur = float(current.get(name, 0.0))
        ref = float(reference.get(name, cur))
        component = abs(cur - ref) / scale
        breakdown[str(name)] = float(component)
        score += weight * component
    score = float(score / total_weight)

    hard_component_thresholds = dict(cfg.get("hard_component_thresholds", {}))
    hard_component_trigger = any(
        float(breakdown.get(str(name), 0.0)) >= float(threshold)
        for name, threshold in hard_component_thresholds.items()
    )

    hard_trigger = bool(
        float(current.get("low_soc_share", 0.0)) >= float(cfg["hard_low_soc_share"])
        or float(current.get("charger_pressure", 0.0)) >= float(cfg["hard_charger_pressure"])
        or float(current.get("reservation_pressure", 0.0)) >= float(cfg["hard_reservation_pressure"])
        or float(current.get("coverage_gap", 0.0)) >= float(cfg["hard_coverage_gap"])
        or (
            float(current.get("recent_request_count", 0.0)) >= float(cfg["hard_recent_request_count_min"])
            and float(current.get("recent_reject_rate", 0.0)) >= float(cfg["hard_recent_reject_rate"])
        )
        or hard_component_trigger
    )
    threshold_mild = float(cfg["threshold_mild"])
    threshold_severe = float(cfg["threshold_severe"])
    if hard_trigger or score >= threshold_severe:
        profile_hint = "severe"
    elif score >= threshold_mild:
        profile_hint = "medium"
    else:
        profile_hint = "none"

    return {
        "change_detected": bool(hard_trigger or score >= threshold_mild),
        "score": float(score),
        "profile_hint": profile_hint,
        "breakdown": breakdown,
        "hard_trigger": bool(hard_trigger),
    }


def _select_regime_profile(
    change_info: Dict[str, Any],
    cfg: Dict[str, Any],
) -> tuple[str, float, float]:
    _ = _merge_problem_detector_config(cfg)
    if bool(change_info.get("hard_trigger", False)):
        return "severe", float(change_info.get("score", 0.0)), 0.0
    hint = str(change_info.get("profile_hint", "none")).strip().lower()
    if hint == "severe":
        return "severe", float(change_info.get("score", 0.0)), 0.0
    if hint == "medium":
        return "medium", float(change_info.get("score", 0.0)), 0.0
    return "mild", float(change_info.get("score", 0.0)), 0.0


def _should_trigger_reopt(
    change_info: Dict[str, Any],
    now_min: float,
    last_reopt_min: float,
    incumbent_exists: bool,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _merge_problem_detector_config(cfg)
    if not incumbent_exists:
        return {"reoptimize": True, "reason": "bootstrap", "elapsed_min": None}

    elapsed = float(now_min) - float(last_reopt_min)
    if bool(change_info.get("hard_trigger", False)):
        return {"reoptimize": True, "reason": "hard_trigger", "elapsed_min": float(elapsed)}
    if elapsed >= float(cfg["max_reopt_interval_min"]):
        return {"reoptimize": True, "reason": "max_interval", "elapsed_min": float(elapsed)}
    if elapsed < float(cfg["min_reopt_interval_min"]):
        return {"reoptimize": False, "reason": "min_interval", "elapsed_min": float(elapsed)}
    if bool(change_info.get("change_detected", False)):
        return {
            "reoptimize": True,
            "reason": str(change_info.get("profile_hint", "change_detected")),
            "elapsed_min": float(elapsed),
        }
    return {"reoptimize": False, "reason": "stable_regime", "elapsed_min": float(elapsed)}


def _resolve_event_step_budget(
    base_steps: int,
    trigger_info: Dict[str, Any],
    cfg: "DynamicSPEDARunConfig",
) -> tuple[int, float]:
    base = max(1, int(base_steps))
    reason = str(trigger_info.get("reason", "always")).strip().lower()
    multiplier = 1.0
    if reason == "bootstrap":
        multiplier = float(cfg.bootstrap_step_multiplier)
    elif reason == "hard_trigger":
        multiplier = float(cfg.hard_step_multiplier)
    elif reason == "severe":
        multiplier = float(cfg.severe_step_multiplier)
    elif reason == "medium":
        multiplier = float(cfg.medium_step_multiplier)
    elif reason == "max_interval":
        multiplier = float(cfg.max_interval_step_multiplier)
    effective = max(1, int(np.ceil(base * max(1.0, multiplier))))
    return effective, float(max(1.0, multiplier))


def _augment_step_diagnostics(optimizer: Any, step_info: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(step_info or {})
    front = out.pop("front", None)
    if front is not None:
        try:
            out["front_size_for_metrics"] = int(np.asarray(front).shape[0])
        except Exception:
            out["front_size_for_metrics"] = 0
    change_info = dict(getattr(optimizer, "_problem_change_info", {}) or {})
    out["change_score"] = float(change_info.get("score", out.get("reaction_score", 0.0) or 0.0))
    out["response_level"] = str(out.get("reaction_profile", "none"))
    out["response_weight_predictive"] = float(out.get("reaction_adr_predictive_weight", 0.0) or 0.0)
    out["response_weight_continuity"] = float(out.get("reaction_adr_continuity_weight", 0.0) or 0.0)
    out["response_weight_diversity"] = float(out.get("reaction_adr_diversity_weight", 0.0) or 0.0)
    out["response_weight_front_ls"] = float(out.get("reaction_adr_front_ls_weight", 0.0) or 0.0)
    out["quality_gate_rejected"] = bool(out.get("reaction_quality_reverted", False))
    out["quality_gate_reason"] = str(out.get("reaction_block_reason", "none"))
    if bool(out.get("reaction_applied", False)):
        out["quality_gate_accepted"] = not bool(out["quality_gate_rejected"])
    elif bool(out.get("change_detected", False)):
        out["quality_gate_accepted"] = False
    else:
        out["quality_gate_accepted"] = None
    out["adaptive_response_enabled"] = bool(
        getattr(optimizer, "_adaptive_response_enabled", False)
    )
    out["adaptive_response_reason"] = str(
        getattr(optimizer, "_last_adaptive_response_reason", "disabled")
    )
    out["inherited_state_reliability"] = float(
        getattr(optimizer, "_last_inherited_state_reliability", 1.0)
    )
    out["inherited_state_similarity"] = float(
        getattr(optimizer, "_last_inherited_state_similarity", 1.0)
    )
    out["inherited_state_drift_reliability"] = float(
        getattr(optimizer, "_last_inherited_state_drift_reliability", 1.0)
    )
    out["inherited_state_change_score"] = float(
        getattr(optimizer, "_last_inherited_state_change_score", out["change_score"])
    )
    out["adaptive_scale_predictive"] = float(
        getattr(optimizer, "_last_adaptive_scale_predictive", 1.0)
    )
    out["adaptive_scale_continuity"] = float(
        getattr(optimizer, "_last_adaptive_scale_continuity", 1.0)
    )
    out["adaptive_scale_diversity"] = float(
        getattr(optimizer, "_last_adaptive_scale_diversity", 1.0)
    )
    out["adaptive_scale_front_ls"] = float(
        getattr(optimizer, "_last_adaptive_scale_front_ls", 1.0)
    )
    out["adaptive_weight_policy"] = str(
        getattr(optimizer, "_last_adaptive_weight_policy", "reliability")
    )
    out["contextual_signal_intensity"] = float(
        getattr(optimizer, "_last_contextual_signal_intensity", 0.0)
    )
    out["contextual_signal_abruptness"] = float(
        getattr(optimizer, "_last_contextual_signal_abruptness", 0.0)
    )
    out["contextual_signal_predictability"] = float(
        getattr(optimizer, "_last_contextual_signal_predictability", 0.0)
    )
    out["contextual_signal_diversity_deficit"] = float(
        getattr(optimizer, "_last_contextual_signal_diversity_deficit", 0.0)
    )
    out["contextual_signal_local_search_opportunity"] = float(
        getattr(optimizer, "_last_contextual_signal_local_search_opportunity", 0.0)
    )
    out["contextual_signal_front_quality"] = float(
        getattr(optimizer, "_last_contextual_signal_front_quality", 0.0)
    )
    out["contextual_score_predictive"] = float(
        getattr(optimizer, "_last_contextual_score_predictive", 0.0)
    )
    out["contextual_score_continuity"] = float(
        getattr(optimizer, "_last_contextual_score_continuity", 0.0)
    )
    out["contextual_score_diversity"] = float(
        getattr(optimizer, "_last_contextual_score_diversity", 0.0)
    )
    out["contextual_score_front_ls"] = float(
        getattr(optimizer, "_last_contextual_score_front_ls", 0.0)
    )
    out["contextual_raw_weight_predictive"] = float(
        getattr(optimizer, "_last_contextual_raw_weight_predictive", 0.0)
    )
    out["contextual_raw_weight_continuity"] = float(
        getattr(optimizer, "_last_contextual_raw_weight_continuity", 0.0)
    )
    out["contextual_raw_weight_diversity"] = float(
        getattr(optimizer, "_last_contextual_raw_weight_diversity", 0.0)
    )
    out["contextual_raw_weight_front_ls"] = float(
        getattr(optimizer, "_last_contextual_raw_weight_front_ls", 0.0)
    )
    return out


def _summarize_step_diagnostics(step_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    infos = [dict(info or {}) for info in step_infos if info is not None]
    if not infos:
        return {}

    merged = dict(infos[-1])
    merged["optimizer_steps_recorded"] = int(len(infos))
    merged["reaction_applied_last_step"] = bool(merged.get("reaction_applied", False))
    merged["response_level_last_step"] = str(merged.get("response_level", "none"))

    changed_infos = [info for info in infos if bool(info.get("change_detected", False))]
    reaction_infos = [info for info in infos if bool(info.get("reaction_applied", False))]
    merged["change_detected_any"] = bool(changed_infos)
    merged["change_detected_steps"] = int(len(changed_infos))
    merged["reaction_applied_any"] = bool(reaction_infos)
    merged["reaction_steps"] = int(len(reaction_infos))

    if not reaction_infos:
        return merged

    reaction_info = dict(reaction_infos[-1])
    preserve_from_reaction = [
        "change_detected",
        "reaction_applied",
        "reaction_profile",
        "reaction_change_type",
        "reaction_score",
        "reaction_trend_signal",
        "reaction_diversity_count",
        "reaction_diversity_scale",
        "reaction_predictive_count",
        "reaction_predictive_scale",
        "reaction_continuity_count",
        "reaction_continuity_scale",
        "reaction_front_ls_count",
        "reaction_front_ls_scale",
        "reaction_adr_context",
        "reaction_adr_predictive_weight",
        "reaction_adr_continuity_weight",
        "reaction_adr_diversity_weight",
        "reaction_adr_front_ls_weight",
        "reaction_response_mode",
        "reaction_quality_reverted",
        "reaction_block_reason",
        "response_level",
        "response_weight_predictive",
        "response_weight_continuity",
        "response_weight_diversity",
        "response_weight_front_ls",
        "quality_gate_rejected",
        "quality_gate_reason",
        "quality_gate_accepted",
        "adaptive_response_reason",
        "inherited_state_reliability",
        "inherited_state_similarity",
        "inherited_state_drift_reliability",
        "inherited_state_change_score",
        "adaptive_scale_predictive",
        "adaptive_scale_continuity",
        "adaptive_scale_diversity",
        "adaptive_scale_front_ls",
        "adaptive_weight_policy",
        "contextual_signal_intensity",
        "contextual_signal_abruptness",
        "contextual_signal_predictability",
        "contextual_signal_diversity_deficit",
        "contextual_signal_local_search_opportunity",
        "contextual_signal_front_quality",
        "contextual_score_predictive",
        "contextual_score_continuity",
        "contextual_score_diversity",
        "contextual_score_front_ls",
        "contextual_raw_weight_predictive",
        "contextual_raw_weight_continuity",
        "contextual_raw_weight_diversity",
        "contextual_raw_weight_front_ls",
    ]
    for key in preserve_from_reaction:
        if key in reaction_info:
            merged[key] = reaction_info[key]
    return merged


def _safe_optimizer_step(
    optimizer: Any,
    *,
    cost_function: Callable[[np.ndarray], np.ndarray],
    with_reaction: bool,
    clip_bounds: Optional[tuple[np.ndarray, np.ndarray]],
) -> tuple[Dict[str, Any], List[str], bool]:
    errors: List[str] = []
    try:
        step_info = optimizer.step(
            cost_function=cost_function,
            with_reaction=bool(with_reaction),
            ftol=1e-8,
            clip_bounds=clip_bounds,
        )
        return (_augment_step_diagnostics(optimizer, step_info), errors, False)
    except Exception as exc:
        errors.append(f"with_reaction={bool(with_reaction)}: {type(exc).__name__}: {exc}")
        if bool(with_reaction):
            fallback_info = optimizer.step(
                cost_function=cost_function,
                with_reaction=False,
                ftol=1e-8,
                clip_bounds=clip_bounds,
            )
            return _augment_step_diagnostics(optimizer, fallback_info), errors, True
        raise


def _normalized_front_scores(front: np.ndarray) -> np.ndarray:
    if front.ndim != 2 or front.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    f = np.asarray(front, dtype=float)
    lo = np.min(f, axis=0)
    hi = np.max(f, axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return (f - lo) / span


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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.empty((0,), dtype=float)
    temp = max(1e-6, float(temperature))
    z = x / temp
    z = z - np.max(z)
    exp_z = np.exp(z)
    total = float(np.sum(exp_z))
    if total <= 1e-12:
        return np.full(x.shape, 1.0 / max(1, x.size), dtype=float)
    return exp_z / total


def _bounded_weight_projection(
    raw_weights: Dict[str, float],
    bounds: Dict[str, tuple[float, float]],
) -> Dict[str, float]:
    names = ["predictive", "continuity", "diversity", "front_ls"]
    lower = {name: max(0.0, float(bounds[name][0])) for name in names}
    upper = {
        name: max(lower[name], float(bounds[name][1]))
        for name in names
    }
    lower_sum = float(sum(lower.values()))
    if lower_sum >= 1.0:
        return {name: float(lower[name] / lower_sum) for name in names}

    weights = dict(lower)
    remaining = 1.0 - lower_sum
    free = {name for name in names if upper[name] - lower[name] > 1e-12}
    raw = {name: max(0.0, float(raw_weights.get(name, 0.0))) for name in names}
    while free and remaining > 1e-12:
        raw_sum = float(sum(raw[name] for name in free))
        if raw_sum <= 1e-12:
            raw_sum = float(len(free))
            raw = {**raw, **{name: 1.0 for name in free}}
        fixed_any = False
        for name in list(free):
            allocation = remaining * raw[name] / raw_sum
            capacity = upper[name] - weights[name]
            if allocation >= capacity - 1e-12:
                weights[name] = upper[name]
                remaining -= capacity
                free.remove(name)
                fixed_any = True
        if not fixed_any:
            for name in free:
                weights[name] += remaining * raw[name] / raw_sum
            remaining = 0.0
            break

    total = float(sum(weights.values()))
    if total <= 1e-12:
        return {name: 1.0 / len(names) for name in names}
    return {name: float(weights[name] / total) for name in names}


def _population_diversity_signal(
    *,
    optimizer: Any,
    bounds: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    population = np.asarray(getattr(optimizer, "elite_temp", np.empty((0, 0))), dtype=float)
    if population.ndim != 2 or population.shape[0] < 3:
        population = np.asarray(getattr(optimizer, "generation", np.empty((0, 0))), dtype=float)
    if population.ndim != 2 or population.shape[0] < 3:
        return {"observed": 0.0, "deficit": 1.0}

    lower = np.asarray(bounds[:, 0], dtype=float)
    upper = np.asarray(bounds[:, 1], dtype=float)
    span = np.where((upper - lower) > 1e-12, upper - lower, 1.0)
    normalized = (np.asarray(population, dtype=float) - lower.reshape(1, -1)) / span.reshape(1, -1)
    observed = float(np.mean(np.std(normalized, axis=0)))
    target = max(1e-9, float(cfg["contextual_diversity_target"]))
    deficit = float(np.clip(1.0 - observed / target, 0.0, 1.0))
    return {"observed": max(0.0, observed), "deficit": deficit}


def _contextual_response_policy(
    *,
    base_weights: Dict[str, float],
    change_info: Dict[str, Any],
    reliability_info: Dict[str, Any],
    optimizer: Any,
    bounds: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _merge_adaptive_response_config(cfg)
    names = ["predictive", "continuity", "diversity", "front_ls"]
    base = {name: max(0.0, float(base_weights.get(name, 0.0))) for name in names}
    base_total = float(sum(base.values()))
    if base_total > 1e-12:
        base = {name: float(value / base_total) for name, value in base.items()}

    reliability = float(np.clip(_finite_float(reliability_info.get("reliability"), 1.0), 0.0, 1.0))
    intensity = float(
        np.clip(
            max(0.0, _finite_float(reliability_info.get("change_score"), 0.0))
            / float(cfg["score_scale"]),
            0.0,
            1.0,
        )
    )
    breakdown = {
        str(k): max(0.0, _finite_float(v, 0.0))
        for k, v in dict(change_info.get("breakdown", {}) or {}).items()
    }
    component_sum = float(sum(breakdown.values()))
    component_max = float(max(breakdown.values())) if breakdown else 0.0
    dominance = 0.0 if component_sum <= 1e-12 else float(component_max / component_sum)
    profile_hint = str(change_info.get("profile_hint", "")).strip().lower()
    profile_abruptness = {
        "mild": 0.15,
        "medium": 0.30,
        "severe": 0.45,
        "hard": 0.85,
        "hard_trigger": 0.85,
    }.get(profile_hint, 0.0)
    abruptness = float(
        np.clip(max(profile_abruptness, intensity * (0.55 + 0.45 * dominance)), 0.0, 1.0)
    )
    if bool(change_info.get("hard_trigger", False)):
        abruptness = 1.0

    diversity_info = _population_diversity_signal(optimizer=optimizer, bounds=bounds, cfg=cfg)
    diversity_deficit = float(diversity_info["deficit"])
    observed_diversity = float(diversity_info["observed"])
    predictability = float(np.clip(reliability * (1.0 - abruptness) * (1.0 - 0.50 * intensity), 0.0, 1.0))
    bell_intensity = float(np.clip(4.0 * intensity * (1.0 - intensity), 0.0, 1.0))
    front_size = int(np.asarray(getattr(optimizer, "evaluations_elite", np.empty((0, 0)))).shape[0])
    front_target = max(4.0, float(getattr(optimizer, "size_gen", 1)) * float(getattr(optimizer, "elite_factor", 0.35)))
    front_quality = float(np.clip(front_size / front_target, 0.0, 1.0))
    local_search_opportunity = float(
        np.clip(reliability * (1.0 - abruptness) * bell_intensity * front_quality, 0.0, 1.0)
    )

    formula_scores = {
        "predictive": 0.10 + 1.20 * reliability + 0.90 * predictability - 0.80 * abruptness - 0.50 * intensity,
        "continuity": 0.05 + 1.35 * reliability + 0.55 * (1.0 - intensity) - 0.90 * abruptness,
        "diversity": 0.00 + 1.30 * intensity + 1.10 * abruptness + 0.90 * diversity_deficit - 1.10 * reliability,
        "front_ls": -0.25 + 0.75 * reliability + 1.00 * local_search_opportunity - 0.80 * abruptness + 0.35 * front_quality,
    }
    prior_strength = max(0.0, float(cfg["contextual_base_prior_strength"]))
    logits = {
        name: float(prior_strength * np.log(max(base.get(name, 0.0), 1e-9)) + formula_scores[name])
        for name in names
    }
    raw_array = _softmax(
        np.asarray([logits[name] for name in names], dtype=float),
        temperature=float(cfg["contextual_temperature"]),
    )
    raw_weights = {name: float(raw_array[i]) for i, name in enumerate(names)}

    weight_bounds: Dict[str, tuple[float, float]] = {}
    for name in names:
        lo = float(cfg[f"contextual_min_{name}"])
        hi = float(cfg[f"contextual_max_{name}"])
        if base.get(name, 0.0) <= 1e-12:
            lo, hi = 0.0, 0.0
        weight_bounds[name] = (lo, hi)
    weights = _bounded_weight_projection(raw_weights, weight_bounds)
    return {
        "weights": weights,
        "raw_weights": raw_weights,
        "scores": formula_scores,
        "logits": logits,
        "bounds": {name: [float(v[0]), float(v[1])] for name, v in weight_bounds.items()},
        "signals": {
            "intensity": float(intensity),
            "abruptness": float(abruptness),
            "reliability": float(reliability),
            "predictability": float(predictability),
            "diversity_deficit": float(diversity_deficit),
            "observed_diversity": float(observed_diversity),
            "local_search_opportunity": float(local_search_opportunity),
            "front_quality": float(front_quality),
            "component_dominance": float(dominance),
            "bell_intensity": float(bell_intensity),
        },
        "temperature": float(cfg["contextual_temperature"]),
        "base_prior_strength": float(prior_strength),
    }


def _adaptive_selection_policy_scores(
    *,
    objective_score: np.ndarray,
    shift_penalty: np.ndarray,
    base_smooth_lambda: float,
    response_weights: Optional[Dict[str, Any]],
    adaptive_response_info: Optional[Dict[str, Any]],
    adaptive_response_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = _merge_adaptive_response_config(adaptive_response_cfg)
    weights = dict(response_weights or {})
    info = dict(adaptive_response_info or {})

    reliability = float(np.clip(_finite_float(info.get("reliability"), 1.0), 0.0, 1.0))
    change_score = max(0.0, _finite_float(info.get("change_score"), 0.0))
    severity = float(np.clip(change_score / float(cfg["score_scale"]), 0.0, 1.0))
    strength = max(0.0, float(cfg["selection_policy_strength"]))

    w_predictive = max(0.0, _finite_float(weights.get("predictive"), 0.0))
    w_continuity = max(0.0, _finite_float(weights.get("continuity"), 0.0))
    w_diversity = max(0.0, _finite_float(weights.get("diversity"), 0.0))
    w_front_ls = max(0.0, _finite_float(weights.get("front_ls"), 0.0))

    # The four ADR blocks are translated into the final Pareto-front choice:
    # prediction/front-LS increase trust in the objective score, continuity
    # penalizes policy displacement, and diversity relaxes or reverses that
    # penalty when reliability is low or the detected change is intense.
    objective_gain = (
        strength
        * float(cfg["selection_policy_objective_gain"])
        * (w_predictive * reliability + w_front_ls * (0.50 + 0.50 * reliability))
    )
    continuity_penalty = (
        strength
        * float(cfg["selection_policy_continuity_gain"])
        * w_continuity
        * (0.25 + 0.75 * reliability)
    )
    diversity_bonus = (
        strength
        * float(cfg["selection_policy_diversity_gain"])
        * w_diversity
        * (0.25 + 0.75 * severity)
        * (1.0 - 0.50 * reliability)
    )

    raw_shift_lambda = float(base_smooth_lambda) + continuity_penalty - diversity_bonus
    effective_shift_lambda = float(
        np.clip(
            raw_shift_lambda,
            -float(cfg["selection_policy_max_shift_reward"]),
            float(cfg["selection_policy_max_shift_penalty"]),
        )
    )
    total_score = (1.0 + objective_gain) * objective_score - effective_shift_lambda * shift_penalty
    selected_index = int(np.argmax(total_score)) if total_score.size else None
    return {
        "selected_index": selected_index,
        "total_score": total_score,
        "objective_gain": float(objective_gain),
        "continuity_penalty": float(continuity_penalty),
        "diversity_bonus": float(diversity_bonus),
        "raw_shift_lambda": float(raw_shift_lambda),
        "effective_shift_lambda": float(effective_shift_lambda),
        "reliability": float(reliability),
        "change_score": float(change_score),
        "severity": float(severity),
        "weights": {
            "predictive": float(w_predictive),
            "continuity": float(w_continuity),
            "diversity": float(w_diversity),
            "front_ls": float(w_front_ls),
        },
    }


def _normalized_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, 0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / span


def _detail_float(details: Dict[str, Any], keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key not in details:
            continue
        try:
            value = float(details[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _front_candidate_rollout_details(
    *,
    problem: FreeFloatingRobustDMOOP,
    pset: np.ndarray,
    carry_state: Optional[SystemState],
    demand_multiplier: Optional[Any],
) -> List[Dict[str, float]]:
    details_by_index: List[Dict[str, float]] = []
    for row in np.asarray(pset, dtype=float):
        _, details, _ = problem.rollout(
            x=np.asarray(row, dtype=float),
            base_state=carry_state,
            external_demand_multiplier=demand_multiplier,
        )
        clean: Dict[str, float] = {}
        for key, value in dict(details).items():
            try:
                f_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f_value):
                clean[str(key)] = f_value
        details_by_index.append(clean)
    return details_by_index


def _service_constrained_selection_scores(
    *,
    front: np.ndarray,
    objective_score: np.ndarray,
    shift_penalty: np.ndarray,
    candidate_details: List[Dict[str, Any]],
    baseline_idx: int,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not candidate_details:
        return None
    n = min(int(front.shape[0]), int(objective_score.shape[0]), len(candidate_details))
    if n <= 0:
        return None

    details = candidate_details[:n]
    service = np.asarray(
        [
            _detail_float(row, ("service_rate", "window_service_rate", "f1_service"), float(front[i, 0]))
            for i, row in enumerate(details)
        ],
        dtype=float,
    )
    rejected = np.asarray(
        [_detail_float(row, ("rejected_requests",), 0.0) for row in details],
        dtype=float,
    )
    profit = np.asarray(
        [_detail_float(row, ("profit_eur",), float(front[i, 1])) for i, row in enumerate(details)],
        dtype=float,
    )
    deadhead = np.asarray(
        [_detail_float(row, ("ops_deadhead_km_total",), 0.0) for row in details],
        dtype=float,
    )
    health = np.asarray(
        [_detail_float(row, ("f3_health",), float(front[i, 2])) for i, row in enumerate(details)],
        dtype=float,
    )
    shift = np.asarray(shift_penalty[:n], dtype=float)

    floor = float(cfg["service_constrained_service_floor"])
    max_rejected = float(cfg["service_constrained_max_rejected_requests"])
    feasible = (service + 1e-12 >= floor) & (rejected <= max_rejected + 1e-12)

    utility = (
        float(cfg["service_constrained_profit_weight"]) * _normalized_1d(profit)
        - float(cfg["service_constrained_deadhead_weight"]) * _normalized_1d(deadhead)
        + float(cfg["service_constrained_health_weight"]) * _normalized_1d(health)
        - float(cfg["service_constrained_shift_weight"]) * shift
        - float(cfg["service_constrained_rejected_weight"]) * _normalized_1d(rejected)
    )

    if bool(np.any(feasible)):
        feasible_idx = np.flatnonzero(feasible)
        local_idx = int(feasible_idx[int(np.argmax(utility[feasible_idx]))])
        fallback_used = False
        fallback_score = utility
        reason = "feasible_service_floor"
    else:
        service_violation = np.maximum(0.0, floor - service)
        rejected_violation = np.maximum(0.0, rejected - max_rejected)
        fallback_score = -100.0 * service_violation - 10.0 * rejected_violation + 0.01 * utility
        local_idx = int(np.argmax(fallback_score))
        fallback_used = True
        reason = "fallback_no_feasible_candidate"

    selected_service = float(service[local_idx])
    diagnostics = {
        "service_constrained_enabled": True,
        "service_constrained_active": True,
        "service_constrained_floor": float(floor),
        "service_constrained_max_rejected_requests": float(max_rejected),
        "service_constrained_candidate_count": int(n),
        "service_constrained_feasible_count": int(np.sum(feasible)),
        "service_constrained_selected_index": int(local_idx),
        "service_constrained_selected_service": selected_service,
        "service_constrained_selected_rejected": float(rejected[local_idx]),
        "service_constrained_min_service": float(np.min(service)),
        "service_constrained_max_service": float(np.max(service)),
        "service_constrained_fallback_used": bool(fallback_used),
        "service_constrained_would_change": bool(int(local_idx) != int(baseline_idx)),
        "service_constrained_selected_by_policy": bool(int(local_idx) != int(baseline_idx)),
        "service_constrained_reason": reason,
    }
    total = fallback_score if fallback_used else utility
    return {
        "selected_index": int(local_idx),
        "total_score": np.asarray(total, dtype=float),
        "utility_score": np.asarray(utility, dtype=float),
        "diagnostics": diagnostics,
    }


def _select_front_candidate(
    *,
    front: np.ndarray,
    pset: np.ndarray,
    reference_vector: Optional[np.ndarray],
    bounds: np.ndarray,
    smooth_lambda: float,
    smooth_metric: str,
    response_weights: Optional[Dict[str, Any]] = None,
    adaptive_response_info: Optional[Dict[str, Any]] = None,
    adaptive_response_cfg: Optional[Dict[str, Any]] = None,
    candidate_details: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if front.ndim != 2 or front.shape[0] == 0:
        return {
            "selected_index": None,
            "objective_score": None,
            "shift_penalty": None,
            "total_score": None,
            "selection_strategy": "empty_front",
            "selection_policy_diagnostics": {
                "enabled": False,
                "diagnostics_enabled": False,
                "active": False,
                "baseline_selected_index": None,
                "policy_selected_index": None,
                "selected_by_policy": False,
                "would_change_selection": False,
                "reason": "empty_front",
            },
        }

    objective_score = np.mean(_normalized_front_scores(front), axis=1)
    shift_penalty = _normalized_policy_shift(pset, reference_vector, bounds, smooth_metric)
    lam = max(0.0, float(smooth_lambda))

    if reference_vector is None or lam <= 1e-12:
        baseline_score = objective_score
        baseline_strategy = "normalized_front_mean"
    else:
        baseline_score = objective_score - lam * shift_penalty
        baseline_strategy = "normalized_front_mean_minus_policy_shift"

    baseline_idx = int(np.argmax(baseline_score))
    cfg = _merge_adaptive_response_config(adaptive_response_cfg)
    policy_enabled = bool(cfg.get("selection_policy_enabled", False))
    policy_diagnostics_enabled = bool(
        policy_enabled or cfg.get("selection_policy_diagnostics_enabled", False)
    )
    policy = None
    selected_by_policy = False
    idx = baseline_idx
    total_score = baseline_score
    strategy = baseline_strategy

    if policy_diagnostics_enabled and reference_vector is not None:
        policy = _adaptive_selection_policy_scores(
            objective_score=objective_score,
            shift_penalty=shift_penalty,
            base_smooth_lambda=lam,
            response_weights=response_weights,
            adaptive_response_info=adaptive_response_info,
            adaptive_response_cfg=cfg,
        )
        policy_idx = policy["selected_index"]
        if policy_enabled and policy_idx is not None:
            idx = int(policy_idx)
            total_score = np.asarray(policy["total_score"], dtype=float)
            strategy = "adaptive_response_selection_policy_v2"
            selected_by_policy = True

    service_constrained = None
    service_constrained_enabled = bool(
        cfg.get("service_constrained_selection_enabled", False)
    )
    service_constrained_diagnostics_enabled = bool(
        service_constrained_enabled
        or cfg.get("service_constrained_selection_diagnostics_enabled", False)
    )
    if service_constrained_diagnostics_enabled:
        service_constrained = _service_constrained_selection_scores(
            front=front,
            objective_score=objective_score,
            shift_penalty=shift_penalty,
            candidate_details=list(candidate_details or []),
            baseline_idx=baseline_idx,
            cfg=cfg,
        )
        if service_constrained_enabled and service_constrained is not None:
            idx = int(service_constrained["selected_index"])
            total_score = np.asarray(service_constrained["total_score"], dtype=float)
            strategy = "service_constrained_selector"
            selected_by_policy = True
            if bool(
                service_constrained["diagnostics"].get(
                    "service_constrained_fallback_used", False
                )
            ):
                strategy = "service_constrained_selector_fallback"

    selected_total_score = float(total_score[idx])
    diagnostics: Dict[str, Any] = {
        "enabled": bool(policy_enabled),
        "diagnostics_enabled": bool(policy_diagnostics_enabled),
        "active": bool(policy_diagnostics_enabled and reference_vector is not None),
        "baseline_selected_index": int(baseline_idx),
        "policy_selected_index": None if policy is None else policy["selected_index"],
        "selected_by_policy": bool(selected_by_policy),
        "would_change_selection": bool(
            policy is not None
            and policy.get("selected_index") is not None
            and int(policy["selected_index"]) != int(baseline_idx)
        ),
        "baseline_total_score": float(baseline_score[baseline_idx]),
        "policy_total_score": None
        if policy is None or policy.get("selected_index") is None
        else float(np.asarray(policy["total_score"], dtype=float)[int(policy["selected_index"])]),
        "baseline_smooth_lambda": float(lam),
        "reason": "active" if policy is not None else "disabled",
    }
    if service_constrained_diagnostics_enabled:
        diagnostics.update(
            {
                "service_constrained_enabled": bool(service_constrained_enabled),
                "service_constrained_active": bool(service_constrained is not None),
                "service_constrained_reason": "missing_candidate_details"
                if service_constrained is None
                else str(
                    service_constrained["diagnostics"].get(
                        "service_constrained_reason", "active"
                    )
                ),
            }
        )
    if service_constrained is not None:
        diagnostics.update(dict(service_constrained["diagnostics"]))
    if policy is not None:
        diagnostics.update(
            {
                "objective_gain": float(policy["objective_gain"]),
                "continuity_penalty": float(policy["continuity_penalty"]),
                "diversity_bonus": float(policy["diversity_bonus"]),
                "raw_shift_lambda": float(policy["raw_shift_lambda"]),
                "effective_shift_lambda": float(policy["effective_shift_lambda"]),
                "reliability": float(policy["reliability"]),
                "change_score": float(policy["change_score"]),
                "severity": float(policy["severity"]),
                "weights": dict(policy["weights"]),
            }
        )

    return {
        "selected_index": idx,
        "objective_score": float(objective_score[idx]),
        "shift_penalty": float(shift_penalty[idx]),
        "total_score": selected_total_score,
        "selection_strategy": strategy,
        "selection_policy_diagnostics": diagnostics,
    }


def _load_dynamic_edaspy_classes():
    ensure_edaspy_importable()
    enable_adaptive_bn_folds_patch()
    try:
        from EDAspy.optimization.custom.initialization_models import UniformGenInit
        from EDAspy.optimization.custom.probabilistic_models import GBN, KDEBN, SPBN
        from EDAspy.optimization.eda_dynamic import DynamicEDA
        from EDAspy.optimization.selection import NSGA2Selection
    except Exception as exc:
        raise ImportError(
            "No se pudo importar DynamicEDA/SPBN/KDEBN/GBN/NSGA2Selection desde EDAspy. "
            f"Causa raíz: {type(exc).__name__}: {exc}"
        ) from exc
    return DynamicEDA, UniformGenInit, SPBN, GBN, KDEBN, NSGA2Selection


@dataclass
class DynamicSPEDARunConfig:
    size_gen: int = 96
    max_iter: int = 20
    dead_iter: int = 10
    alpha: float = 0.35
    elite_factor: float = 0.35
    archive_generations: float = 4.0
    w_noise: float = 0.20
    parallelize: bool = False
    disp: bool = True

    pm_family: str = "SPBN"
    num_folds: int = 3
    archive_retention_after_change: float = 0.30
    with_reaction: bool = True
    step_clip_bounds: bool = True
    change_detector_mode: str = "sentinel"
    reopt_trigger_mode: str = "always"

    detector: Dict[str, Any] = field(default_factory=dict)
    reaction: Dict[str, Any] = field(default_factory=dict)
    local_search: Dict[str, Any] = field(default_factory=dict)
    problem_detector: Dict[str, Any] = field(default_factory=dict)
    reservation_guard: Dict[str, Any] = field(default_factory=dict)
    adaptive_response: Dict[str, Any] = field(default_factory=dict)
    bootstrap_step_multiplier: float = 1.0
    medium_step_multiplier: float = 1.0
    severe_step_multiplier: float = 1.0
    hard_step_multiplier: float = 1.0
    max_interval_step_multiplier: float = 1.0
    shadow_update_on_hold: bool = False
    shadow_update_steps: int = 1
    incumbent_gate_enabled: bool = True
    incumbent_min_improvement: float = 0.01
    selection_smooth_lambda: float = 0.0
    selection_smooth_metric: str = "l1"

    def __post_init__(self) -> None:
        self.pm_family = _normalize_pm_family(self.pm_family)
        self.archive_retention_after_change = float(
            np.clip(float(self.archive_retention_after_change), 0.0, 1.0)
        )
        self.num_folds = int(max(2, int(self.num_folds)))
        self.change_detector_mode = _normalize_change_detector_mode(self.change_detector_mode)
        self.reopt_trigger_mode = _normalize_reopt_trigger_mode(self.reopt_trigger_mode)
        self.problem_detector = _merge_problem_detector_config(self.problem_detector)
        self.reservation_guard = _merge_reservation_guard_config(self.reservation_guard)
        self.adaptive_response = _merge_adaptive_response_config(self.adaptive_response)
        self.bootstrap_step_multiplier = max(1.0, float(self.bootstrap_step_multiplier))
        self.medium_step_multiplier = max(1.0, float(self.medium_step_multiplier))
        self.severe_step_multiplier = max(1.0, float(self.severe_step_multiplier))
        self.hard_step_multiplier = max(1.0, float(self.hard_step_multiplier))
        self.max_interval_step_multiplier = max(1.0, float(self.max_interval_step_multiplier))
        self.shadow_update_on_hold = bool(self.shadow_update_on_hold)
        self.shadow_update_steps = max(1, int(self.shadow_update_steps))
        self.incumbent_gate_enabled = bool(self.incumbent_gate_enabled)
        self.incumbent_min_improvement = max(0.0, float(self.incumbent_min_improvement))
        self.selection_smooth_lambda = max(0.0, float(self.selection_smooth_lambda))
        self.selection_smooth_metric = str(self.selection_smooth_metric).strip().lower() or "l1"


@dataclass
class DynamicWindowResult:
    window_id: str
    raw_result: Dict[str, Any]
    selected_index: Optional[int] = None
    selected_objectives: Optional[List[float]] = None
    selected_details: Optional[Dict[str, float]] = None


class StatefulDynamicSPEDA:
    """
    Stateful dynamic EDA over free_floating_robust.

    Key behavior:
    - Keeps a single DynamicEDA instance across windows.
    - At each window, the objective context is switched to the new demand regime.
    - Detection/reaction/memory/local-search happen inside DynamicEDA.
    - State carryover for the simulator is done by rolling out selected Pareto solution.
    """

    def __init__(self, encoding: RobustDecisionEncoding, config: Optional[DynamicSPEDARunConfig] = None):
        self.encoding = encoding
        self.config = config or DynamicSPEDARunConfig()
        (
            self._dynamiceda_cls,
            self._uniform_init_cls,
            self._spbn_cls,
            self._gbn_cls,
            self._kdebn_cls,
            self._nsga2_cls,
        ) = _load_dynamic_edaspy_classes()

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.encoding.bounds[:, 0].astype(float)
        upper = self.encoding.bounds[:, 1].astype(float)
        return lower, upper

    def _build_pm(self, vars_list: List[str]):
        family = self.config.pm_family
        if family == "SPBN":
            return self._spbn_cls(vars_list)
        if family == "GBN":
            return self._gbn_cls(vars_list)
        if family == "KDEBN":
            return self._kdebn_cls(vars_list)
        raise ValueError(f"pm_family='{family}' no soportado.")

    def _build_optimizer(self):
        cfg = self.config
        lower, upper = self._bounds()

        optimizer = self._dynamiceda_cls(
            size_gen=int(cfg.size_gen),
            max_iter=int(cfg.max_iter),
            dead_iter=int(cfg.dead_iter),
            n_variables=int(self.encoding.dimension),
            alpha=float(cfg.alpha),
            elite_factor=float(cfg.elite_factor),
            disp=bool(cfg.disp),
            parallelize=bool(cfg.parallelize),
            w_noise=float(cfg.w_noise),
            selection_strategy=self._nsga2_cls(),
            detector=dict(cfg.detector),
            reaction=dict(cfg.reaction),
            local_search=dict(cfg.local_search),
        )

        vars_list = [str(i) for i in range(int(self.encoding.dimension))]
        optimizer.vars = vars_list
        optimizer.pm = self._build_pm(vars_list)
        optimizer.init = self._uniform_init_cls(int(self.encoding.dimension), lower_bound=lower, upper_bound=upper)
        optimizer.l_len = max(1, int(float(cfg.archive_generations) * int(cfg.size_gen * cfg.alpha)))
        optimizer.archive = np.empty((0, int(self.encoding.dimension)))
        optimizer.num_folds = int(cfg.num_folds)
        optimizer.archive_retention_after_change = float(cfg.archive_retention_after_change)
        base_react = optimizer._react
        base_detect_change = optimizer._detect_change
        base_select_adaptive_profile = optimizer._select_adaptive_profile

        # SPEDA-like archive update.
        def _update_archive() -> None:
            if optimizer.elite_temp is None or len(optimizer.elite_temp) == 0:
                return
            optimizer.archive = np.append(optimizer.archive, optimizer.elite_temp, axis=0)
            optimizer.archive = optimizer.archive[-optimizer.l_len :]

        def _update_pm() -> None:
            _update_archive()
            min_rows = max(10, int(self.encoding.dimension) + 2)
            data = optimizer.archive if len(optimizer.archive) >= min_rows else optimizer.generation
            try:
                optimizer.pm.learn(dataset=data, num_folds=int(optimizer.num_folds))
            except TypeError:
                optimizer.pm.learn(dataset=data)

        def _react(cost_function=None, profile: str = "severe", clip_bounds=None) -> None:
            # Call parent reaction.
            base_react(cost_function=cost_function, profile=profile, clip_bounds=clip_bounds)
            # Keep only a fraction of old archive after change.
            if len(optimizer.archive) > 0:
                keep = int(round(optimizer.l_len * float(optimizer.archive_retention_after_change)))
                if keep <= 0:
                    optimizer.archive = np.empty((0, int(self.encoding.dimension)))
                else:
                    optimizer.archive = optimizer.archive[-keep:]

        optimizer._update_archive = _update_archive
        optimizer._update_pm = _update_pm
        optimizer._react = _react
        optimizer._reservation_guard_base = {
            "react_strength": float(optimizer.react_strength),
            "react_strength_mild": float(optimizer.react_strength_mild),
            "react_strength_medium": float(optimizer.react_strength_medium),
            "immigrant_ratio": float(optimizer.immigrant_ratio),
            "immigrant_ratio_mild": float(optimizer.immigrant_ratio_mild),
            "immigrant_ratio_medium": float(optimizer.immigrant_ratio_medium),
            "predictive_block_enabled": bool(optimizer.predictive_block_enabled),
            "continuity_block_enabled": bool(optimizer.continuity_block_enabled),
            "adr_front_ls_enabled": bool(optimizer.adr_front_ls_enabled),
        }
        adaptive_response_cfg = _merge_adaptive_response_config(cfg.adaptive_response)
        optimizer._adaptive_response_cfg = dict(adaptive_response_cfg)
        optimizer._adaptive_response_enabled = bool(adaptive_response_cfg.get("enabled", False))
        optimizer._last_inherited_state_reliability = 1.0
        optimizer._last_inherited_state_similarity = 1.0
        optimizer._last_inherited_state_drift_reliability = 1.0
        optimizer._last_inherited_state_change_score = 0.0
        optimizer._last_adaptive_response_reason = (
            "enabled_waiting_for_change"
            if bool(adaptive_response_cfg.get("enabled", False))
            else "disabled"
        )
        optimizer._last_adaptive_scale_predictive = 1.0
        optimizer._last_adaptive_scale_continuity = 1.0
        optimizer._last_adaptive_scale_diversity = 1.0
        optimizer._last_adaptive_scale_front_ls = 1.0
        optimizer._last_adaptive_weight_policy = str(adaptive_response_cfg.get("weight_policy", "reliability"))
        optimizer._last_contextual_signal_intensity = 0.0
        optimizer._last_contextual_signal_abruptness = 0.0
        optimizer._last_contextual_signal_predictability = 0.0
        optimizer._last_contextual_signal_diversity_deficit = 0.0
        optimizer._last_contextual_signal_local_search_opportunity = 0.0
        optimizer._last_contextual_signal_front_quality = 0.0
        optimizer._last_contextual_score_predictive = 0.0
        optimizer._last_contextual_score_continuity = 0.0
        optimizer._last_contextual_score_diversity = 0.0
        optimizer._last_contextual_score_front_ls = 0.0
        optimizer._last_contextual_raw_weight_predictive = 0.0
        optimizer._last_contextual_raw_weight_continuity = 0.0
        optimizer._last_contextual_raw_weight_diversity = 0.0
        optimizer._last_contextual_raw_weight_front_ls = 0.0

        base_adr_compute_block_weights = optimizer._adr_compute_block_weights

        def _adr_compute_block_weights_adaptive(context: Dict[str, Any]) -> Dict[str, float]:
            base_weights = dict(base_adr_compute_block_weights(context))
            if not bool(adaptive_response_cfg.get("enabled", False)):
                optimizer._last_adaptive_response_reason = "disabled"
                return base_weights

            change_info = dict(getattr(optimizer, "_problem_change_info", {}) or {})
            if not change_info:
                drift_score = max(0.0, float(getattr(optimizer, "_last_drift_ratio", 1.0)) - 1.0)
                change_info = {
                    "score": drift_score,
                    "breakdown": {},
                }
            reliability_info = _inherited_state_reliability(
                change_info=change_info,
                cfg=adaptive_response_cfg,
            )
            reliability = float(reliability_info["reliability"])
            optimizer._last_adaptive_weight_policy = str(
                adaptive_response_cfg.get("weight_policy", "reliability")
            )

            def _interp(low_key: str, high_key: str) -> float:
                low = float(adaptive_response_cfg[low_key])
                high = float(adaptive_response_cfg[high_key])
                return float(low + reliability * (high - low))

            if str(adaptive_response_cfg.get("weight_policy", "reliability")) == "contextual_formula":
                contextual = _contextual_response_policy(
                    base_weights=base_weights,
                    change_info=change_info,
                    reliability_info=reliability_info,
                    optimizer=optimizer,
                    bounds=self.encoding.bounds,
                    cfg=adaptive_response_cfg,
                )
                weighted = dict(contextual["weights"])
                raw_weights = dict(contextual["raw_weights"])
                contextual_scores = dict(contextual["scores"])
                contextual_signals = dict(contextual["signals"])
                scales = {
                    name: float(weighted.get(name, 0.0))
                    / max(1e-12, float(base_weights.get(name, 0.0)))
                    for name in ("predictive", "continuity", "diversity", "front_ls")
                }
                optimizer._last_contextual_signal_intensity = float(
                    contextual_signals["intensity"]
                )
                optimizer._last_contextual_signal_abruptness = float(
                    contextual_signals["abruptness"]
                )
                optimizer._last_contextual_signal_predictability = float(
                    contextual_signals["predictability"]
                )
                optimizer._last_contextual_signal_diversity_deficit = float(
                    contextual_signals["diversity_deficit"]
                )
                optimizer._last_contextual_signal_local_search_opportunity = float(
                    contextual_signals["local_search_opportunity"]
                )
                optimizer._last_contextual_signal_front_quality = float(
                    contextual_signals["front_quality"]
                )
                optimizer._last_contextual_score_predictive = float(contextual_scores["predictive"])
                optimizer._last_contextual_score_continuity = float(contextual_scores["continuity"])
                optimizer._last_contextual_score_diversity = float(contextual_scores["diversity"])
                optimizer._last_contextual_score_front_ls = float(contextual_scores["front_ls"])
                optimizer._last_contextual_raw_weight_predictive = float(raw_weights["predictive"])
                optimizer._last_contextual_raw_weight_continuity = float(raw_weights["continuity"])
                optimizer._last_contextual_raw_weight_diversity = float(raw_weights["diversity"])
                optimizer._last_contextual_raw_weight_front_ls = float(raw_weights["front_ls"])
            else:
                scales = {
                    "predictive": _interp("predictive_low_scale", "predictive_high_scale"),
                    "continuity": _interp("continuity_low_scale", "continuity_high_scale"),
                    "diversity": _interp("diversity_low_scale", "diversity_high_scale"),
                    "front_ls": _interp("front_ls_low_scale", "front_ls_high_scale"),
                }
                weighted = {
                    name: max(0.0, float(base_weights.get(name, 0.0)) * float(scales[name]))
                    for name in ("predictive", "continuity", "diversity", "front_ls")
                }
                total = float(sum(weighted.values()))
                if total > 1e-12:
                    weighted = {name: float(value / total) for name, value in weighted.items()}
                else:
                    weighted = dict(base_weights)
                optimizer._last_contextual_signal_intensity = 0.0
                optimizer._last_contextual_signal_abruptness = 0.0
                optimizer._last_contextual_signal_predictability = 0.0
                optimizer._last_contextual_signal_diversity_deficit = 0.0
                optimizer._last_contextual_signal_local_search_opportunity = 0.0
                optimizer._last_contextual_signal_front_quality = 0.0
                optimizer._last_contextual_score_predictive = 0.0
                optimizer._last_contextual_score_continuity = 0.0
                optimizer._last_contextual_score_diversity = 0.0
                optimizer._last_contextual_score_front_ls = 0.0
                optimizer._last_contextual_raw_weight_predictive = 0.0
                optimizer._last_contextual_raw_weight_continuity = 0.0
                optimizer._last_contextual_raw_weight_diversity = 0.0
                optimizer._last_contextual_raw_weight_front_ls = 0.0

            optimizer._last_inherited_state_reliability = reliability
            optimizer._last_inherited_state_similarity = float(reliability_info["similarity"])
            optimizer._last_inherited_state_drift_reliability = float(
                reliability_info["drift_reliability"]
            )
            optimizer._last_inherited_state_change_score = float(reliability_info["change_score"])
            optimizer._last_adaptive_response_reason = str(reliability_info["reason"])
            optimizer._last_adaptive_scale_predictive = float(scales["predictive"])
            optimizer._last_adaptive_scale_continuity = float(scales["continuity"])
            optimizer._last_adaptive_scale_diversity = float(scales["diversity"])
            optimizer._last_adaptive_scale_front_ls = float(scales["front_ls"])
            return weighted

        if bool(adaptive_response_cfg.get("enabled", False)) or bool(
            adaptive_response_cfg.get("diagnostics_enabled", False)
        ):
            optimizer._adr_compute_block_weights = _adr_compute_block_weights_adaptive

        if str(cfg.change_detector_mode) == "problem-aware":
            def _detect_change_problem_aware(cost_function) -> bool:
                _ = cost_function
                window_id = getattr(optimizer, "_problem_window_id", None)
                last_window_id = getattr(optimizer, "_problem_last_detect_window_id", None)
                if window_id is not None and window_id == last_window_id:
                    return False

                current = getattr(optimizer, "_problem_regime_current", None)
                reference = getattr(optimizer, "_problem_regime_reference", None)
                if current is None:
                    return False

                precomputed_change_info = getattr(optimizer, "_problem_change_info", None)
                precomputed_window_id = getattr(optimizer, "_problem_change_info_window_id", None)
                use_precomputed = bool(
                    (
                        bool(adaptive_response_cfg.get("enabled", False))
                        or bool(adaptive_response_cfg.get("diagnostics_enabled", False))
                    )
                    and precomputed_change_info
                    and precomputed_window_id == window_id
                )
                if use_precomputed:
                    change_info = dict(precomputed_change_info)
                else:
                    change_info = _score_regime_shift(
                        current=dict(current),
                        reference=None if reference is None else dict(reference),
                        cfg=cfg.problem_detector,
                    )
                optimizer._problem_change_info = dict(change_info)
                optimizer._problem_last_detect_window_id = window_id
                if bool(adaptive_response_cfg.get("enabled", False)) or bool(
                    adaptive_response_cfg.get("diagnostics_enabled", False)
                ):
                    reliability_info = _inherited_state_reliability(
                        change_info=dict(change_info),
                        cfg=adaptive_response_cfg,
                    )
                    optimizer._last_inherited_state_reliability = float(
                        reliability_info["reliability"]
                    )
                    optimizer._last_inherited_state_similarity = float(
                        reliability_info["similarity"]
                    )
                    optimizer._last_inherited_state_drift_reliability = float(
                        reliability_info["drift_reliability"]
                    )
                    optimizer._last_inherited_state_change_score = float(
                        reliability_info["change_score"]
                    )
                    optimizer._last_adaptive_response_reason = str(reliability_info["reason"])

                score = float(change_info.get("score", 0.0))
                optimizer._last_drift_ratio = float(1.0 + score)
                optimizer._last_drift_ratio_mean = float(1.0 + score)
                optimizer._last_drift_ratio_cov = 0.0
                optimizer._last_drift_mean_score = float(score)
                optimizer._last_drift_cov_score = 0.0
                optimizer._last_drift_cov_threshold = np.inf
                optimizer._last_reaction_severity = float(1.0 + score)

                return bool(change_info.get("change_detected", False))

            def _select_adaptive_profile_problem_aware():
                change_info = getattr(optimizer, "_problem_change_info", None) or {}
                profile, score, trend_signal = _select_regime_profile(
                    change_info=change_info,
                    cfg=cfg.problem_detector,
                )
                change_type = optimizer._classify_change_type(profile, trend_signal)
                response_mode = optimizer._select_response_mode(
                    profile,
                    trend_signal,
                    change_type,
                )
                return profile, float(score), float(trend_signal), change_type, response_mode

            optimizer._detect_change = _detect_change_problem_aware
            optimizer._select_adaptive_profile = _select_adaptive_profile_problem_aware
            optimizer._base_detect_change = base_detect_change
            optimizer._base_select_adaptive_profile = base_select_adaptive_profile

        return optimizer

    @staticmethod
    def _best_by_sum_index(front: np.ndarray) -> Optional[int]:
        if front.ndim != 2 or front.shape[0] == 0:
            return None
        f = np.asarray(front, dtype=float)
        lo = np.min(f, axis=0)
        hi = np.max(f, axis=0)
        span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
        f_norm = (f - lo) / span
        return int(np.argmax(np.sum(f_norm, axis=1)))

    def optimize_windows(
        self,
        problem_builder: Callable[[Dict[str, Any], Optional[SystemState]], FreeFloatingRobustDMOOP],
        windows: List[Dict[str, Any]],
    ) -> List[DynamicWindowResult]:
        if len(windows) == 0:
            return []

        optimizer = self._build_optimizer()
        lower, upper = self._bounds()
        clip_bounds = (lower, upper) if self.config.step_clip_bounds else None

        carry_state: Optional[SystemState] = None
        out: List[DynamicWindowResult] = []
        is_initialized = False

        use_event_trigger = bool(
            str(self.config.reopt_trigger_mode) == "event"
            and str(self.config.change_detector_mode) == "problem-aware"
        )
        incumbent_x: Optional[np.ndarray] = None
        incumbent_front: np.ndarray = np.empty((0, 3))
        reference_regime: Optional[Dict[str, float]] = None
        last_reopt_min = -1e18

        for window in windows:
            window_id = str(window.get("id", f"window_{len(out)}"))
            problem = problem_builder(window, carry_state)
            demand_multiplier = window.get("demand_multiplier")
            if demand_multiplier is None:
                demand_multiplier = window.get("demand_shock")
            now_min = float(
                (carry_state.current_time_min if carry_state is not None else problem.state_template.current_time_min)
            )
            regime = _extract_regime_features(
                problem=problem,
                window=window,
                carry_state=carry_state,
                cfg=self.config.problem_detector,
                reservation_guard_cfg=self.config.reservation_guard,
            )
            regime_reference_snapshot = None if reference_regime is None else dict(reference_regime)
            change_info = _score_regime_shift(
                current=regime,
                reference=regime_reference_snapshot,
                cfg=self.config.problem_detector,
            )
            trigger_info = {
                "reoptimize": True,
                "reason": "always",
                "elapsed_min": None if last_reopt_min < -1e17 else float(now_min - last_reopt_min),
            }
            if use_event_trigger:
                trigger_info = _should_trigger_reopt(
                    change_info=change_info,
                    now_min=now_min,
                    last_reopt_min=last_reopt_min,
                    incumbent_exists=incumbent_x is not None,
                    cfg=self.config.problem_detector,
                )

            def cost_function(x: np.ndarray) -> np.ndarray:
                obj, _, _ = problem.rollout(
                    x=x,
                    base_state=carry_state,
                    external_demand_multiplier=demand_multiplier,
                )
                return -np.asarray(obj, dtype=float)

            if not is_initialized:
                optimizer.setup_dynamic(cost_function, output_runtime=bool(self.config.disp))
                is_initialized = True

            base_n_steps = max(1, int(window.get("max_iter", self.config.max_iter)))
            n_steps, step_multiplier = _resolve_event_step_budget(
                base_steps=base_n_steps,
                trigger_info=trigger_info,
                cfg=self.config,
            )
            reservation_guard = _compute_reservation_guard_controls(
                regime=regime,
                trigger_info=trigger_info,
                reservation_guard_cfg=self.config.reservation_guard,
                base_selection_smooth_lambda=float(self.config.selection_smooth_lambda),
                base_incumbent_min_improvement=float(self.config.incumbent_min_improvement),
            )
            if bool(reservation_guard.get("enabled", False)) and bool(
                reservation_guard.get("suppress_reopt", False)
            ):
                trigger_info = dict(trigger_info)
                trigger_info["reoptimize"] = False
                trigger_info["reason"] = str(
                    reservation_guard.get("suppress_reason", "reservation_guard_hold")
                )
                trigger_info["reservation_guard_suppressed"] = True
            if bool(reservation_guard.get("enabled", False)):
                n_steps = max(
                    1,
                    int(
                        np.ceil(
                            float(n_steps)
                            * float(reservation_guard.get("step_budget_scale", 1.0))
                        )
                    ),
                )
            step_info: Dict[str, Any] = {}
            front_min = np.empty((0, 3))
            pset = np.empty((0, int(self.encoding.dimension)))
            front_max = np.empty((0, 3))
            selected_index = None
            selected_objectives = None
            selected_details = None
            reoptimized = False
            shadow_update_executed = False
            optimizer_steps_executed = 0
            incumbent_gate_applied = False
            cost_guard_applied = False
            adaptive_guard_applied = False
            incumbent_score = None
            candidate_score = None
            cost_guard_result: Optional[Dict[str, Any]] = None
            adaptive_guard_result: Optional[Dict[str, Any]] = None
            step_errors: List[str] = []
            step_fallback_used = False
            selection_objective_score = None
            selection_shift_penalty = None
            selection_total_score = None
            selection_strategy = "empty_front"
            selection_policy_diagnostics: Dict[str, Any] = {}

            if bool(trigger_info.get("reoptimize", True)):
                optimizer._problem_window_id = window_id
                optimizer._problem_regime_current = dict(regime)
                optimizer._problem_regime_reference = regime_reference_snapshot
                optimizer._problem_change_info = dict(change_info)
                optimizer._problem_change_info_window_id = window_id
                if bool(self.config.adaptive_response.get("enabled", False)) or bool(
                    self.config.adaptive_response.get("diagnostics_enabled", False)
                ):
                    optimizer._problem_last_detect_window_id = None
                _apply_reservation_guard_controls(optimizer=optimizer, controls=reservation_guard)

                step_infos: List[Dict[str, Any]] = []
                for _ in range(n_steps):
                    step_info, errors_i, fallback_i = _safe_optimizer_step(
                        optimizer,
                        cost_function=cost_function,
                        with_reaction=bool(self.config.with_reaction),
                        clip_bounds=clip_bounds,
                    )
                    step_infos.append(dict(step_info))
                    step_errors.extend(errors_i)
                    step_fallback_used = bool(step_fallback_used or fallback_i)
                step_info = _summarize_step_diagnostics(step_infos)
                optimizer_steps_executed = int(n_steps)

                front_min = np.asarray(getattr(optimizer, "evaluations_elite", np.empty((0, 3))), dtype=float)
                pset = np.asarray(
                    getattr(optimizer, "elite_temp", np.empty((0, int(self.encoding.dimension)))),
                    dtype=float,
                )
                if front_min.ndim != 2 or front_min.shape[0] == 0:
                    front_min = np.asarray(getattr(optimizer, "evaluations", np.empty((0, 3))), dtype=float)
                    pset = np.asarray(
                        getattr(optimizer, "generation", np.empty((0, int(self.encoding.dimension)))),
                        dtype=float,
                    )

                front_max = -front_min if front_min.size > 0 else np.empty((0, 3))
                adaptive_selection_info = dict(
                    _inherited_state_reliability(
                        change_info=change_info,
                        cfg=self.config.adaptive_response,
                    )
                )
                selection_response_weights = {
                    "predictive": step_info.get("response_weight_predictive"),
                    "continuity": step_info.get("response_weight_continuity"),
                    "diversity": step_info.get("response_weight_diversity"),
                    "front_ls": step_info.get("response_weight_front_ls"),
                }
                front_candidate_details = None
                if bool(
                    self.config.adaptive_response.get(
                        "service_constrained_selection_enabled", False
                    )
                ) or bool(
                    self.config.adaptive_response.get(
                        "service_constrained_selection_diagnostics_enabled", False
                    )
                ):
                    front_candidate_details = _front_candidate_rollout_details(
                        problem=problem,
                        pset=pset,
                        carry_state=carry_state,
                        demand_multiplier=demand_multiplier,
                    )
                selection = _select_front_candidate(
                    front=front_max,
                    pset=pset,
                    reference_vector=incumbent_x,
                    bounds=self.encoding.bounds,
                    smooth_lambda=float(
                        reservation_guard.get(
                            "effective_selection_smooth_lambda",
                            self.config.selection_smooth_lambda,
                        )
                    ),
                    smooth_metric=str(self.config.selection_smooth_metric),
                    response_weights=selection_response_weights,
                    adaptive_response_info=adaptive_selection_info,
                    adaptive_response_cfg=self.config.adaptive_response,
                    candidate_details=front_candidate_details,
                )
                selected_index = selection["selected_index"]
                selection_objective_score = selection["objective_score"]
                selection_shift_penalty = selection["shift_penalty"]
                selection_total_score = selection["total_score"]
                selection_strategy = str(selection["selection_strategy"])
                selection_policy_diagnostics = dict(
                    selection.get("selection_policy_diagnostics", {})
                )
                if selected_index is not None and 0 <= selected_index < pset.shape[0]:
                    candidate_x = np.asarray(pset[selected_index], dtype=float).copy()
                    candidate_obj_eval, candidate_details_eval, _ = problem.rollout(
                        x=np.asarray(candidate_x, dtype=float),
                        base_state=carry_state,
                        external_demand_multiplier=demand_multiplier,
                    )
                    candidate_obj = np.asarray(candidate_obj_eval, dtype=float)
                    candidate_score = float(np.mean(candidate_obj))
                    accept_candidate = True
                    incumbent_details_eval = None
                    needs_incumbent_details = bool(
                        incumbent_x is not None
                        and (
                            bool(self.config.incumbent_gate_enabled)
                            or bool(reservation_guard.get("cost_guard_active", False))
                            or bool(reservation_guard.get("service_deployment_guard_active", False))
                            or bool(self.config.adaptive_response.get("guard_enabled", False))
                        )
                    )
                    if needs_incumbent_details:
                        incumbent_obj, incumbent_details_eval, _ = problem.rollout(
                            x=np.asarray(incumbent_x, dtype=float),
                            base_state=carry_state,
                            external_demand_multiplier=demand_multiplier,
                        )
                        incumbent_score = float(np.mean(np.asarray(incumbent_obj, dtype=float)))
                    if bool(self.config.incumbent_gate_enabled) and incumbent_x is not None:
                        if candidate_score < incumbent_score + float(
                            reservation_guard.get(
                                "effective_incumbent_min_improvement",
                                self.config.incumbent_min_improvement,
                            )
                        ):
                            accept_candidate = False
                            incumbent_gate_applied = True
                    if accept_candidate and incumbent_x is not None:
                        adaptive_guard_result = _evaluate_adaptive_response_guard(
                            cfg=self.config.adaptive_response,
                            regime=regime,
                            candidate_x=np.asarray(candidate_x, dtype=float),
                            incumbent_x=np.asarray(incumbent_x, dtype=float),
                            candidate_score=candidate_score,
                            incumbent_score=incumbent_score,
                        )
                        if bool(adaptive_guard_result.get("active", False)):
                            adaptive_guard_applied = True
                        if not bool(adaptive_guard_result.get("accepted", True)):
                            accept_candidate = False
                    if accept_candidate and incumbent_x is not None:
                        cost_guard_result = _evaluate_candidate_cost_guard(
                            controls=reservation_guard,
                            candidate_details={k: float(v) for k, v in candidate_details_eval.items()},
                            incumbent_details=None
                            if incumbent_details_eval is None
                            else {k: float(v) for k, v in incumbent_details_eval.items()},
                            candidate_shift_penalty=None
                            if selection_shift_penalty is None
                            else float(selection_shift_penalty),
                        )
                        if bool(cost_guard_result.get("active", False)):
                            cost_guard_applied = True
                        if not bool(cost_guard_result.get("accepted", True)):
                            accept_candidate = False
                    if accept_candidate:
                        incumbent_x = np.asarray(candidate_x, dtype=float).copy()
                        incumbent_front = np.asarray(front_max, dtype=float)
                        reference_regime = dict(regime)
                        last_reopt_min = float(now_min)
                        reoptimized = True
            elif use_event_trigger and bool(self.config.shadow_update_on_hold):
                optimizer._problem_window_id = window_id
                optimizer._problem_regime_current = dict(regime)
                optimizer._problem_regime_reference = regime_reference_snapshot
                optimizer._problem_change_info = dict(change_info)
                optimizer._problem_change_info_window_id = window_id
                if bool(self.config.adaptive_response.get("enabled", False)) or bool(
                    self.config.adaptive_response.get("diagnostics_enabled", False)
                ):
                    optimizer._problem_last_detect_window_id = None

                shadow_steps = max(1, int(self.config.shadow_update_steps))
                step_infos = []
                for _ in range(shadow_steps):
                    step_info, errors_i, fallback_i = _safe_optimizer_step(
                        optimizer,
                        cost_function=cost_function,
                        with_reaction=False,
                        clip_bounds=clip_bounds,
                    )
                    step_infos.append(dict(step_info))
                    step_errors.extend(errors_i)
                    step_fallback_used = bool(step_fallback_used or fallback_i)
                step_info = _summarize_step_diagnostics(step_infos)
                optimizer_steps_executed = int(shadow_steps)
                shadow_update_executed = True

                front_min = np.asarray(getattr(optimizer, "evaluations_elite", np.empty((0, 3))), dtype=float)
                pset = np.asarray(
                    getattr(optimizer, "elite_temp", np.empty((0, int(self.encoding.dimension)))),
                    dtype=float,
                )
                if front_min.ndim != 2 or front_min.shape[0] == 0:
                    front_min = np.asarray(getattr(optimizer, "evaluations", np.empty((0, 3))), dtype=float)
                    pset = np.asarray(
                        getattr(optimizer, "generation", np.empty((0, int(self.encoding.dimension)))),
                        dtype=float,
                    )
                front_max = -front_min if front_min.size > 0 else np.empty((0, 3))

            if incumbent_x is None:
                raise RuntimeError(
                    f"No hay política incumbente disponible para desplegar en la ventana '{window_id}'."
                )

            if reoptimized and selected_index is not None and 0 <= selected_index < pset.shape[0]:
                x_sel = pset[selected_index]
            else:
                x_sel = np.asarray(incumbent_x, dtype=float)

            if x_sel is not None:
                obj, details, end_state = problem.rollout(
                    x=x_sel,
                    base_state=carry_state,
                    external_demand_multiplier=demand_multiplier,
                )
                carry_state = copy.deepcopy(end_state)
                selected_objectives = [float(obj[0]), float(obj[1]), float(obj[2])]
                selected_details = {k: float(v) for k, v in details.items()}

            reference_regime = _update_regime_reference_ewma(
                reference=reference_regime,
                current=regime,
                cfg=self.config.problem_detector,
            )
            adaptive_response_info = dict(
                _inherited_state_reliability(
                    change_info=change_info,
                    cfg=self.config.adaptive_response,
                )
            )
            adaptive_response_info["enabled"] = bool(
                self.config.adaptive_response.get("enabled", False)
            )
            adaptive_response_info["diagnostics_enabled"] = bool(
                self.config.adaptive_response.get("diagnostics_enabled", False)
            )

            raw_result = {
                "pareto_set": pset,
                "pareto_front": front_min,
                "pareto_front_max": front_max,
                "optimizer_settings": optimizer.export_settings(),
                "step_info_last": step_info,
                "regime": regime,
                "reference_regime": regime_reference_snapshot or {},
                "change_info": change_info,
                "trigger_info": trigger_info,
                "adaptive_response_info": adaptive_response_info,
                "reoptimized": bool(reoptimized),
                "shadow_update_executed": bool(shadow_update_executed),
                "optimizer_steps_executed": int(optimizer_steps_executed),
                "incumbent_gate_applied": bool(incumbent_gate_applied),
                "cost_guard_applied": bool(cost_guard_applied),
                "adaptive_guard_applied": bool(adaptive_guard_applied),
                "incumbent_score": None if incumbent_score is None else float(incumbent_score),
                "candidate_score": None if candidate_score is None else float(candidate_score),
                "cost_guard_result": None if cost_guard_result is None else dict(cost_guard_result),
                "adaptive_guard_result": None
                if adaptive_guard_result is None
                else dict(adaptive_guard_result),
                "step_errors": list(step_errors),
                "step_fallback_used": bool(step_fallback_used),
                "selection_objective_score": selection_objective_score,
                "selection_shift_penalty": selection_shift_penalty,
                "selection_total_score": selection_total_score,
                "selection_strategy": selection_strategy,
                "selection_policy_diagnostics": dict(selection_policy_diagnostics),
                "reservation_guard": dict(reservation_guard),
                "incumbent_vector": np.asarray(incumbent_x, dtype=float).copy(),
                "incumbent_front": np.asarray(incumbent_front, dtype=float),
                "window": {
                    "id": window_id,
                    "parent_window_id": window.get("parent_window_id"),
                    "subwindow_index": window.get("subwindow_index"),
                    "start_min": float(now_min),
                    "horizon_hours": float(window.get("horizon_hours", problem.horizon_hours)),
                    "demand_multiplier": demand_multiplier or {},
                    "max_iter": n_steps,
                    "base_max_iter": base_n_steps,
                    "step_budget_multiplier": float(step_multiplier),
                },
            }

            out.append(
                DynamicWindowResult(
                    window_id=window_id,
                    raw_result=raw_result,
                    selected_index=selected_index,
                    selected_objectives=selected_objectives,
                    selected_details=selected_details,
                )
            )

        return out


class OnlineStatefulDynamicSPEDA(StatefulDynamicSPEDA):
    """
    Persistent DynamicEDA wrapper for online stream reoptimization.

    Unlike `optimize_windows`, this class does not roll the simulator state forward
    internally. Each call receives the current real system state through the
    provided `problem` and keeps only optimizer-side memory between calls.
    """

    def __init__(self, encoding: RobustDecisionEncoding, config: Optional[DynamicSPEDARunConfig] = None):
        super().__init__(encoding=encoding, config=config)
        self.reset()

    def reset(self) -> None:
        self._optimizer = None
        self._is_initialized = False
        self._incumbent_x: Optional[np.ndarray] = None
        self._incumbent_front: np.ndarray = np.empty((0, 3))
        self._reference_regime: Optional[Dict[str, float]] = None
        self._last_reopt_min: float = -1e18

    def optimize_window(
        self,
        problem: FreeFloatingRobustDMOOP,
        window: Dict[str, Any],
        carry_state: Optional[SystemState] = None,
    ) -> DynamicWindowResult:
        if self._optimizer is None:
            self._optimizer = self._build_optimizer()
        optimizer = self._optimizer

        lower, upper = self._bounds()
        clip_bounds = (lower, upper) if self.config.step_clip_bounds else None

        window_id = str(window.get("id", "window"))
        demand_multiplier = window.get("demand_multiplier")
        if demand_multiplier is None:
            demand_multiplier = window.get("demand_shock")

        now_min = float(
            (carry_state.current_time_min if carry_state is not None else problem.state_template.current_time_min)
        )
        regime = _extract_regime_features(
            problem=problem,
            window=window,
            carry_state=carry_state,
            cfg=self.config.problem_detector,
            reservation_guard_cfg=self.config.reservation_guard,
        )
        regime_reference_snapshot = None if self._reference_regime is None else dict(self._reference_regime)
        change_info = _score_regime_shift(
            current=regime,
            reference=regime_reference_snapshot,
            cfg=self.config.problem_detector,
        )

        use_event_trigger = bool(
            str(self.config.reopt_trigger_mode) == "event"
            and str(self.config.change_detector_mode) == "problem-aware"
        )
        trigger_info = {
            "reoptimize": True,
            "reason": "always",
            "elapsed_min": None if self._last_reopt_min < -1e17 else float(now_min - self._last_reopt_min),
        }
        if use_event_trigger:
            trigger_info = _should_trigger_reopt(
                change_info=change_info,
                now_min=now_min,
                last_reopt_min=self._last_reopt_min,
                incumbent_exists=self._incumbent_x is not None,
                cfg=self.config.problem_detector,
            )

        def cost_function(x: np.ndarray) -> np.ndarray:
            obj, _, _ = problem.rollout(
                x=x,
                base_state=carry_state,
                external_demand_multiplier=demand_multiplier,
            )
            return -np.asarray(obj, dtype=float)

        if not self._is_initialized:
            optimizer.setup_dynamic(cost_function, output_runtime=bool(self.config.disp))
            self._is_initialized = True

        base_n_steps = max(1, int(window.get("max_iter", self.config.max_iter)))
        n_steps, step_multiplier = _resolve_event_step_budget(
            base_steps=base_n_steps,
            trigger_info=trigger_info,
            cfg=self.config,
        )
        reservation_guard = _compute_reservation_guard_controls(
            regime=regime,
            trigger_info=trigger_info,
            reservation_guard_cfg=self.config.reservation_guard,
            base_selection_smooth_lambda=float(self.config.selection_smooth_lambda),
            base_incumbent_min_improvement=float(self.config.incumbent_min_improvement),
        )
        if bool(reservation_guard.get("enabled", False)) and bool(
            reservation_guard.get("suppress_reopt", False)
        ):
            trigger_info = dict(trigger_info)
            trigger_info["reoptimize"] = False
            trigger_info["reason"] = str(
                reservation_guard.get("suppress_reason", "reservation_guard_hold")
            )
            trigger_info["reservation_guard_suppressed"] = True
        if bool(reservation_guard.get("enabled", False)):
            n_steps = max(
                1,
                int(
                    np.ceil(
                        float(n_steps)
                        * float(reservation_guard.get("step_budget_scale", 1.0))
                    )
                ),
            )
        step_info: Dict[str, Any] = {}
        front_min = np.empty((0, 3))
        pset = np.empty((0, int(self.encoding.dimension)))
        front_max = np.empty((0, 3))
        selected_index = None
        selected_objectives = None
        selected_details = None
        reoptimized = False
        shadow_update_executed = False
        optimizer_steps_executed = 0
        incumbent_gate_applied = False
        cost_guard_applied = False
        adaptive_guard_applied = False
        incumbent_score = None
        candidate_score = None
        cost_guard_result: Optional[Dict[str, Any]] = None
        adaptive_guard_result: Optional[Dict[str, Any]] = None
        step_errors: List[str] = []
        step_fallback_used = False
        selection_objective_score = None
        selection_shift_penalty = None
        selection_total_score = None
        selection_strategy = "empty_front"
        selection_policy_diagnostics: Dict[str, Any] = {}

        if bool(trigger_info.get("reoptimize", True)):
            optimizer._problem_window_id = window_id
            optimizer._problem_regime_current = dict(regime)
            optimizer._problem_regime_reference = regime_reference_snapshot
            optimizer._problem_change_info = dict(change_info)
            optimizer._problem_change_info_window_id = window_id
            if bool(self.config.adaptive_response.get("enabled", False)) or bool(
                self.config.adaptive_response.get("diagnostics_enabled", False)
            ):
                optimizer._problem_last_detect_window_id = None
            _apply_reservation_guard_controls(optimizer=optimizer, controls=reservation_guard)

            step_infos = []
            for _ in range(n_steps):
                step_info, errors_i, fallback_i = _safe_optimizer_step(
                    optimizer,
                    cost_function=cost_function,
                    with_reaction=bool(self.config.with_reaction),
                    clip_bounds=clip_bounds,
                )
                step_infos.append(dict(step_info))
                step_errors.extend(errors_i)
                step_fallback_used = bool(step_fallback_used or fallback_i)
            step_info = _summarize_step_diagnostics(step_infos)
            optimizer_steps_executed = int(n_steps)

            front_min = np.asarray(getattr(optimizer, "evaluations_elite", np.empty((0, 3))), dtype=float)
            pset = np.asarray(
                getattr(optimizer, "elite_temp", np.empty((0, int(self.encoding.dimension)))),
                dtype=float,
            )
            if front_min.ndim != 2 or front_min.shape[0] == 0:
                front_min = np.asarray(getattr(optimizer, "evaluations", np.empty((0, 3))), dtype=float)
                pset = np.asarray(
                    getattr(optimizer, "generation", np.empty((0, int(self.encoding.dimension)))),
                    dtype=float,
                )

            front_max = -front_min if front_min.size > 0 else np.empty((0, 3))
            adaptive_selection_info = dict(
                _inherited_state_reliability(
                    change_info=change_info,
                    cfg=self.config.adaptive_response,
                )
            )
            selection_response_weights = {
                "predictive": step_info.get("response_weight_predictive"),
                "continuity": step_info.get("response_weight_continuity"),
                "diversity": step_info.get("response_weight_diversity"),
                "front_ls": step_info.get("response_weight_front_ls"),
            }
            front_candidate_details = None
            if bool(
                self.config.adaptive_response.get(
                    "service_constrained_selection_enabled", False
                )
            ) or bool(
                self.config.adaptive_response.get(
                    "service_constrained_selection_diagnostics_enabled", False
                )
            ):
                front_candidate_details = _front_candidate_rollout_details(
                    problem=problem,
                    pset=pset,
                    carry_state=carry_state,
                    demand_multiplier=demand_multiplier,
                )
            selection = _select_front_candidate(
                front=front_max,
                pset=pset,
                reference_vector=self._incumbent_x,
                bounds=self.encoding.bounds,
                smooth_lambda=float(
                    reservation_guard.get(
                        "effective_selection_smooth_lambda",
                        self.config.selection_smooth_lambda,
                    )
                ),
                smooth_metric=str(self.config.selection_smooth_metric),
                response_weights=selection_response_weights,
                adaptive_response_info=adaptive_selection_info,
                adaptive_response_cfg=self.config.adaptive_response,
                candidate_details=front_candidate_details,
            )
            selected_index = selection["selected_index"]
            selection_objective_score = selection["objective_score"]
            selection_shift_penalty = selection["shift_penalty"]
            selection_total_score = selection["total_score"]
            selection_strategy = str(selection["selection_strategy"])
            selection_policy_diagnostics = dict(
                selection.get("selection_policy_diagnostics", {})
            )
            if selected_index is not None and 0 <= selected_index < pset.shape[0]:
                candidate_x = np.asarray(pset[selected_index], dtype=float).copy()
                candidate_obj_eval, candidate_details_eval, _ = problem.rollout(
                    x=np.asarray(candidate_x, dtype=float),
                    base_state=carry_state,
                    external_demand_multiplier=demand_multiplier,
                )
                candidate_obj = np.asarray(candidate_obj_eval, dtype=float)
                candidate_score = float(np.mean(candidate_obj))
                accept_candidate = True
                incumbent_details_eval = None
                needs_incumbent_details = bool(
                    self._incumbent_x is not None
                    and (
                        bool(self.config.incumbent_gate_enabled)
                        or bool(reservation_guard.get("cost_guard_active", False))
                        or bool(reservation_guard.get("service_deployment_guard_active", False))
                        or bool(self.config.adaptive_response.get("guard_enabled", False))
                    )
                )
                if needs_incumbent_details:
                    incumbent_obj, incumbent_details_eval, _ = problem.rollout(
                        x=np.asarray(self._incumbent_x, dtype=float),
                        base_state=carry_state,
                        external_demand_multiplier=demand_multiplier,
                    )
                    incumbent_score = float(np.mean(np.asarray(incumbent_obj, dtype=float)))
                if bool(self.config.incumbent_gate_enabled) and self._incumbent_x is not None:
                    if candidate_score < incumbent_score + float(
                        reservation_guard.get(
                            "effective_incumbent_min_improvement",
                            self.config.incumbent_min_improvement,
                        )
                    ):
                        accept_candidate = False
                        incumbent_gate_applied = True
                if accept_candidate and self._incumbent_x is not None:
                    adaptive_guard_result = _evaluate_adaptive_response_guard(
                        cfg=self.config.adaptive_response,
                        regime=regime,
                        candidate_x=np.asarray(candidate_x, dtype=float),
                        incumbent_x=np.asarray(self._incumbent_x, dtype=float),
                        candidate_score=candidate_score,
                        incumbent_score=incumbent_score,
                    )
                    if bool(adaptive_guard_result.get("active", False)):
                        adaptive_guard_applied = True
                    if not bool(adaptive_guard_result.get("accepted", True)):
                        accept_candidate = False
                if accept_candidate and self._incumbent_x is not None:
                    cost_guard_result = _evaluate_candidate_cost_guard(
                        controls=reservation_guard,
                        candidate_details={k: float(v) for k, v in candidate_details_eval.items()},
                        incumbent_details=None
                        if incumbent_details_eval is None
                        else {k: float(v) for k, v in incumbent_details_eval.items()},
                        candidate_shift_penalty=None
                        if selection_shift_penalty is None
                        else float(selection_shift_penalty),
                    )
                    if bool(cost_guard_result.get("active", False)):
                        cost_guard_applied = True
                    if not bool(cost_guard_result.get("accepted", True)):
                        accept_candidate = False
                if accept_candidate:
                    self._incumbent_x = np.asarray(candidate_x, dtype=float).copy()
                    self._incumbent_front = np.asarray(front_max, dtype=float)
                    self._reference_regime = dict(regime)
                    self._last_reopt_min = float(now_min)
                    reoptimized = True
        elif use_event_trigger and bool(self.config.shadow_update_on_hold):
            optimizer._problem_window_id = window_id
            optimizer._problem_regime_current = dict(regime)
            optimizer._problem_regime_reference = regime_reference_snapshot
            optimizer._problem_change_info = dict(change_info)
            optimizer._problem_change_info_window_id = window_id
            if bool(self.config.adaptive_response.get("enabled", False)) or bool(
                self.config.adaptive_response.get("diagnostics_enabled", False)
            ):
                optimizer._problem_last_detect_window_id = None

            shadow_steps = max(1, int(self.config.shadow_update_steps))
            step_infos = []
            for _ in range(shadow_steps):
                step_info, errors_i, fallback_i = _safe_optimizer_step(
                    optimizer,
                    cost_function=cost_function,
                    with_reaction=False,
                    clip_bounds=clip_bounds,
                )
                step_infos.append(dict(step_info))
                step_errors.extend(errors_i)
                step_fallback_used = bool(step_fallback_used or fallback_i)
            step_info = _summarize_step_diagnostics(step_infos)
            optimizer_steps_executed = int(shadow_steps)
            shadow_update_executed = True

            front_min = np.asarray(getattr(optimizer, "evaluations_elite", np.empty((0, 3))), dtype=float)
            pset = np.asarray(
                getattr(optimizer, "elite_temp", np.empty((0, int(self.encoding.dimension)))),
                dtype=float,
            )
            if front_min.ndim != 2 or front_min.shape[0] == 0:
                front_min = np.asarray(getattr(optimizer, "evaluations", np.empty((0, 3))), dtype=float)
                pset = np.asarray(
                    getattr(optimizer, "generation", np.empty((0, int(self.encoding.dimension)))),
                    dtype=float,
                )
            front_max = -front_min if front_min.size > 0 else np.empty((0, 3))

        if self._incumbent_x is None:
            raise RuntimeError(
                f"No hay política incumbente disponible para desplegar en la ventana online '{window_id}'."
            )

        x_sel = (
            pset[selected_index]
            if reoptimized and selected_index is not None and 0 <= selected_index < pset.shape[0]
            else np.asarray(self._incumbent_x, dtype=float)
        )
        obj, details, _ = problem.rollout(
            x=x_sel,
            base_state=carry_state,
            external_demand_multiplier=demand_multiplier,
        )
        selected_objectives = [float(obj[0]), float(obj[1]), float(obj[2])]
        selected_details = {k: float(v) for k, v in details.items()}

        self._reference_regime = _update_regime_reference_ewma(
            reference=self._reference_regime,
            current=regime,
            cfg=self.config.problem_detector,
        )
        adaptive_response_info = dict(
            _inherited_state_reliability(
                change_info=change_info,
                cfg=self.config.adaptive_response,
            )
        )
        adaptive_response_info["enabled"] = bool(
            self.config.adaptive_response.get("enabled", False)
        )
        adaptive_response_info["diagnostics_enabled"] = bool(
            self.config.adaptive_response.get("diagnostics_enabled", False)
        )

        raw_result = {
            "pareto_set": pset,
            "pareto_front": front_min,
            "pareto_front_max": front_max,
            "optimizer_settings": optimizer.export_settings(),
            "step_info_last": step_info,
            "regime": regime,
            "reference_regime": regime_reference_snapshot or {},
            "change_info": change_info,
            "trigger_info": trigger_info,
            "adaptive_response_info": adaptive_response_info,
            "reoptimized": bool(reoptimized),
            "shadow_update_executed": bool(shadow_update_executed),
            "optimizer_steps_executed": int(optimizer_steps_executed),
            "incumbent_gate_applied": bool(incumbent_gate_applied),
            "cost_guard_applied": bool(cost_guard_applied),
            "adaptive_guard_applied": bool(adaptive_guard_applied),
            "incumbent_score": None if incumbent_score is None else float(incumbent_score),
            "candidate_score": None if candidate_score is None else float(candidate_score),
            "cost_guard_result": None if cost_guard_result is None else dict(cost_guard_result),
            "adaptive_guard_result": None
            if adaptive_guard_result is None
            else dict(adaptive_guard_result),
            "step_errors": list(step_errors),
            "step_fallback_used": bool(step_fallback_used),
            "selection_objective_score": selection_objective_score,
            "selection_shift_penalty": selection_shift_penalty,
            "selection_total_score": selection_total_score,
            "selection_strategy": selection_strategy,
            "selection_policy_diagnostics": dict(selection_policy_diagnostics),
            "reservation_guard": dict(reservation_guard),
            "selected_vector": np.asarray(x_sel, dtype=float).copy(),
            "selected_policy": self.encoding.decode(np.asarray(x_sel, dtype=float)),
            "incumbent_vector": np.asarray(self._incumbent_x, dtype=float).copy(),
            "incumbent_front": np.asarray(self._incumbent_front, dtype=float),
            "window": {
                "id": window_id,
                "start_min": float(now_min),
                "horizon_hours": float(window.get("horizon_hours", problem.horizon_hours)),
                "demand_multiplier": demand_multiplier or {},
                "max_iter": n_steps,
                "base_max_iter": base_n_steps,
                "step_budget_multiplier": float(step_multiplier),
            },
        }
        return DynamicWindowResult(
            window_id=window_id,
            raw_result=raw_result,
            selected_index=selected_index,
            selected_objectives=selected_objectives,
            selected_details=selected_details,
        )
