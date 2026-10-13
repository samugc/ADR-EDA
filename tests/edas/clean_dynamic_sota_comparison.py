#!/usr/bin/env python3
"""
Clean dynamic comparison runner for strong DMOEA baselines vs ADR-EDA.

What this file gives you:
- A single entry point to run a compact but serious benchmark comparison.
- Clean GA/NSGA-II style reference baselines implemented here:
  * DNSGA-II-B (hypermutated immigrants)
  * MNSGA-II memory-based NSGA-II core
  * Memory-Immigrants NSGA-II (provisional, excluded from --algorithms all)
  * Jiang-Yang SGEA 2017 core author-code-audited Python port
  * Jiang et al. 2018 Tr-NSGA-II core implementation with Tr-IPG/TCA
- Dedicated prediction/EDA dynamic baselines:
  * Zhang et al. 2022 PFPA core implementation
- Dedicated type-adaptive dynamic baselines:
  * Cai et al. 2024 TDA-DMOEA core implementation
- Dedicated adaptive-boosting dynamic baselines:
  * Peng et al. 2024 AB-DMOEA author-code-audited Python port
- Dedicated adaptive-response dynamic baselines:
  * Wang et al. 2026 ADR-DMOEA core implementation
- Framework wrappers for the remaining in-repo EDA profile:
  * ADR-EDA

Important note for the paper:
- SGEA and AB-DMOEA have public author code extracted under
  third_party/author_code and are represented here as audited Python ports so
  they can run inside the same benchmark protocol as ADR-EDA.
- The other GA baselines in this file are self-contained reference
  implementations.
- ADR is an explicit wrapper over the DynamicSPEDA/DynamicEDA
  response blocks in your repo. They are much cleaner than the previous
  ablation scripts, but they are still "framework-faithful wrappers", not
  stand-alone reimplementations from scratch of every original paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


def _bootstrap_paths() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        os.path.join(repo_root, "tests", "edas"),
        os.path.join(repo_root, "EDAspy"),
        "/app/tests/edas",
        "/app/EDAspy",
        here,
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_paths()


LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUTHOR_CODE_ROOT = os.path.join(REPO_ROOT, "third_party", "author_code", "extracted")


def repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


@dataclass
class AlgorithmSpec:
    name: str
    runner: str
    description: str
    implementation_kind: str
    faithfulness: str = "unspecified"
    change_protocol: str = "native"
    implementation_source: str = "in_repo"
    author_code_path: str = ""
    author_code_status: str = "not_applicable"
    overrides: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperFidelityTarget:
    algorithm: str
    paper_label: str
    source_url: str
    target_status: str
    required_ingredients: List[str] = field(default_factory=list)
    current_gaps: List[str] = field(default_factory=list)
    frozen_overrides: Optional[Dict[str, object]] = None


ADR_EDA_V2_1_FROZEN_OVERRIDES: Dict[str, object] = {
    "pm_family": "SPBN",
    "local_search_enabled": True,
    "memory_enabled": False,
    "split_drift_signals": True,
    "adaptive_lambda_enabled": True,
    "adaptive_lambda_cov_enabled": True,
    "adaptive_cooldown_enabled": True,
    "adaptive_policy_enabled": True,
    "reaction_quality_gate_enabled": True,
    "cov_threshold_k": 2.8,
    "alert_cooldown": 1,
    "min_change_interval": 1,
    "diversity_block_enabled": True,
    "predictive_block_enabled": True,
    "continuity_block_enabled": True,
    "adr_front_ls_enabled": True,
    "adr_front_ls_context_gate_enabled": True,
    "portfolio_mix_enabled": False,
    "change_type_enabled": True,
    "response_mode_enabled": False,
    "adr_enabled": True,
    "react_strength": 2.8,
    "react_strength_mild": 1.6,
    "immigrant_ratio": 0.28,
    "immigrant_ratio_mild": 0.16,
    "local_search_neighbors": 6,
    "local_search_front_ratio": 0.55,
    "local_search_step": 0.06,
    "local_search_max_points": 36,
    "adr_front_ls_min_count_enabled": True,
    "adr_front_ls_min_weight": 0.04,
    "adr_front_ls_min_budget": 3,
}


ADR_EDA_SGEA_REACTION_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Experimental profile: preserve ADR's detector/model, but make the
    # post-change reaction closer to SGEA's retention/relocation bias.
    "archive_retention_after_change": 0.45,
    "cov_threshold_k": 3.0,
    "react_strength": 2.35,
    "react_strength_medium": 1.95,
    "react_strength_mild": 1.25,
    "immigrant_ratio": 0.22,
    "immigrant_ratio_medium": 0.18,
    "immigrant_ratio_mild": 0.11,
    "reaction_min_drift_ratio": 1.08,
    "reaction_min_interval": 2,
    "diversity_ratio_mild": 0.05,
    "diversity_ratio_medium": 0.09,
    "diversity_ratio_severe": 0.16,
    "diversity_gate_trend_min": 0.38,
    "diversity_gate_cov_ratio_min": 1.25,
    "diversity_gate_severe_ratio": 1.80,
    "diversity_gate_fallback_scale": 0.35,
    "predictive_ratio_mild": 0.08,
    "predictive_ratio_medium": 0.10,
    "predictive_ratio_severe": 0.12,
    "predictive_step_scale": 0.85,
    "predictive_jitter_scale": 0.18,
    "predictive_trend_min": 0.30,
    "continuity_ratio_mild": 0.14,
    "continuity_ratio_medium": 0.18,
    "continuity_ratio_severe": 0.22,
    "continuity_step_scale": 0.42,
    "continuity_jitter_scale": 0.10,
    "adr_stable_predictive_weight": 0.34,
    "adr_stable_continuity_weight": 0.34,
    "adr_stable_diversity_weight": 0.10,
    "adr_stable_front_ls_weight": 0.22,
    "adr_uncertain_predictive_weight": 0.26,
    "adr_uncertain_continuity_weight": 0.36,
    "adr_uncertain_diversity_weight": 0.18,
    "adr_uncertain_front_ls_weight": 0.20,
    "adr_abrupt_predictive_weight": 0.16,
    "adr_abrupt_continuity_weight": 0.36,
    "adr_abrupt_diversity_weight": 0.34,
    "adr_abrupt_front_ls_weight": 0.14,
}


ADR_EDA_SGEA_PREDICTION_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Clean ablation: keep ADR-EDA fixed and replace only the predictive
    # displacement rule with an SGEA-style centroid relocation predictor.
    "predictive_strategy": "sgea_centroid",
    "predictive_sgea_noise_scale": 1.0,
    "predictive_sgea_gate_floor": 1.0,
}


ADR_EDA_FAST_REACTION_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Experimental detector/reaction ablation: keep ADR-EDA's model and block
    # design, but make the internal detector closer to SGEA's frequent
    # re-evaluation style by lowering hysteresis and reaction throttling.
    "drift_alpha_mean": 0.08,
    "lambda_ewma_mean": 0.55,
    "lambda_ewma_cov": 0.55,
    "cov_threshold_k": 2.20,
    "cov_baseline_min_samples": 12,
    "k_consecutive": 1,
    "alert_cooldown": 0,
    "min_change_interval": 0,
    "adaptive_cooldown_enabled": False,
    "detector_rearm_ratio": 0.95,
    "reaction_min_drift_ratio": 1.00,
    "reaction_min_interval": 1,
    "reaction_max_calls_in_window": 8,
    "reaction_max_calls_mild_window": 8,
    "reaction_max_calls_medium_window": 8,
    "reaction_max_calls_severe_window": 8,
}


ADR_EDA_REGIME_UNIFIED_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Experimental regime-aware detector: benchmark-side features are extracted
    # from the active elite/front response, while the score/trigger logic is
    # shared with the problem-aware online bridge used in car-sharing.
    "change_detector_mode": "regime-aware",
    "regime_probe_max_points": 24,
    "problem_detector": {
        "replace_weights": True,
        "replace_scales": True,
        "weights": {
            "quality_best": 0.24,
            "quality_mean": 0.22,
            "quality_ideal": 0.16,
            "centroid_shift": 0.16,
            "front_spread_shift": 0.12,
            "front_size_loss": 0.10,
        },
        "scales": {
            "quality_best": 0.06,
            "quality_mean": 0.05,
            "quality_ideal": 0.06,
            "centroid_shift": 0.12,
            "front_spread_shift": 0.20,
            "front_size_loss": 0.18,
        },
        "threshold_mild": 0.18,
        "threshold_severe": 0.36,
        "reference_ewma_alpha": 1.0,
        "hard_component_thresholds": {
            "quality_best": 2.2,
            "quality_ideal": 2.0,
            "front_size_loss": 1.8,
        },
    },
}


ADR_EDA_REGIME_UNIFIED_BALANCED_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_REGIME_UNIFIED_OVERRIDES,
    # Balanced retuning after the first h60/s4 screen: keep the unified
    # scoring logic, but reduce missed DF-T10 changes by rearming earlier and
    # accepting milder front-quality shifts as actionable regime evidence.
    "detector_rearm_ratio": 0.95,
    "regime_force_rearm_interval": 9,
    "regime_probe_max_points": 24,
    "problem_detector": {
        **ADR_EDA_REGIME_UNIFIED_OVERRIDES["problem_detector"],  # type: ignore[index]
        "threshold_mild": 0.145,
        "threshold_severe": 0.32,
        "hard_component_thresholds": {
            "quality_best": 1.65,
            "quality_ideal": 1.55,
            "quality_mean": 1.80,
            "front_size_loss": 1.45,
        },
    },
}


ADR_EDA_REGIME_UNIFIED_SENSITIVE_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_REGIME_UNIFIED_OVERRIDES,
    # More aggressive screening profile. This is intentionally separated from
    # the balanced profile so that extra recall is not silently promoted if it
    # damages optimization quality or runtime.
    "detector_rearm_ratio": 1.00,
    "alert_cooldown": 0,
    "adaptive_cooldown_enabled": False,
    "regime_force_rearm_interval": 6,
    "regime_probe_max_points": 24,
    "problem_detector": {
        **ADR_EDA_REGIME_UNIFIED_OVERRIDES["problem_detector"],  # type: ignore[index]
        "threshold_mild": 0.115,
        "threshold_severe": 0.27,
        "hard_component_thresholds": {
            "quality_best": 1.35,
            "quality_ideal": 1.30,
            "quality_mean": 1.50,
            "front_size_loss": 1.25,
        },
    },
}


ADR_EDA_CONTEXTUAL_WEIGHTS_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Experimental scheduler: keep the frozen detector and reaction blocks, but
    # replace the discrete regime table with a continuous contextual policy.
    "adr_contextual_weights_enabled": True,
    "adr_contextual_policy_variant": "v1",
    "adr_contextual_temperature": 0.70,
    "adr_contextual_base_prior_strength": 0.35,
    "adr_contextual_diversity_target": 0.18,
    "adr_contextual_min_predictive": 0.05,
    "adr_contextual_max_predictive": 0.45,
    "adr_contextual_min_continuity": 0.10,
    "adr_contextual_max_continuity": 0.50,
    "adr_contextual_min_diversity": 0.15,
    "adr_contextual_max_diversity": 0.70,
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.00,
}


ADR_EDA_CONTEXTUAL_WEIGHTS_V2_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_CONTEXTUAL_WEIGHTS_OVERRIDES,
    # V2 after h60/s5 screen: reduce continuity under abrupt/intense changes
    # and make diversity less automatic unless the signal shows real need.
    "adr_contextual_policy_variant": "v2",
    "adr_contextual_temperature": 0.85,
    "adr_contextual_base_prior_strength": 0.45,
    "adr_contextual_min_continuity": 0.08,
    "adr_contextual_max_continuity": 0.42,
    "adr_contextual_min_diversity": 0.08,
    "adr_contextual_max_diversity": 0.60,
    "adr_contextual_max_predictive": 0.42,
}


ADR_EDA_CONTEXTUAL_WEIGHTS_V3_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_CONTEXTUAL_WEIGHTS_V2_OVERRIDES,
    # V3: keep the v2 contextual signal model but bound the deviation from the
    # frozen ADR prior when recovery-risk proxies suggest noisy adaptation.
    "adr_contextual_policy_variant": "v3",
    "adr_contextual_temperature": 1.00,
    "adr_contextual_base_prior_strength": 0.65,
    "adr_contextual_min_predictive": 0.04,
    "adr_contextual_max_predictive": 0.40,
    "adr_contextual_min_continuity": 0.06,
    "adr_contextual_max_continuity": 0.38,
    "adr_contextual_min_diversity": 0.05,
    "adr_contextual_max_diversity": 0.58,
}


ADR_EDA_GEOMETRIC_RESPONSE_POLICY_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Experimental scheduler: infer the response mix from measured POS/POF
    # movement between detected changes instead of selecting a fixed regime row.
    "adr_contextual_weights_enabled": True,
    "adr_contextual_policy_variant": "geometric",
    "adr_contextual_diversity_target": 0.18,
    "adr_contextual_min_predictive": 0.04,
    "adr_contextual_max_predictive": 0.42,
    "adr_contextual_min_continuity": 0.06,
    "adr_contextual_max_continuity": 0.42,
    "adr_contextual_min_diversity": 0.06,
    "adr_contextual_max_diversity": 0.62,
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.00,
    "adr_geometry_max_points": 48,
    "adr_geometry_min_points": 4,
    "adr_geometry_front_distance_scale": 0.30,
    "adr_geometry_pos_distance_scale": 0.35,
    "adr_geometry_shape_scale": 0.35,
    "adr_geometry_overlap_eps": 0.08,
    "adr_geometry_tau_min": 0.15,
    "adr_geometry_tau_max": 0.62,
}


ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V2_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_OVERRIDES,
    # V2: keep the objective geometry scheduler, but treat high POS movement
    # with low displacement coherence as unreliable inherited state.
    "adr_contextual_policy_variant": "geometric_v2",
    "adr_contextual_max_continuity": 0.36,
    "adr_contextual_min_diversity": 0.08,
    "adr_contextual_max_diversity": 0.64,
    "adr_geometry_continuity_risk_enabled": True,
    "adr_geometry_continuity_risk_strength": 0.80,
    "adr_geometry_low_coherence_threshold": 0.45,
    "adr_geometry_predictive_transfer_floor": 0.15,
    "adr_geometry_predictive_transfer_ceiling": 0.45,
}


ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V3_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V2_OVERRIDES,
    # V3: explicit POS/POF risk matrix. POF stability only licenses inherited
    # continuity when POS transfer risk and displacement incoherence are low.
    "adr_contextual_policy_variant": "geometric_v3",
    "adr_contextual_max_predictive": 0.38,
    "adr_contextual_max_continuity": 0.30,
    "adr_contextual_min_diversity": 0.10,
    "adr_contextual_max_diversity": 0.70,
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.08,
    "adr_front_ls_context_gate_enabled": False,
    "adr_geometry_continuity_risk_strength": 0.95,
    "adr_geometry_low_coherence_threshold": 0.45,
    "adr_geometry_predictive_transfer_floor": 0.05,
    "adr_geometry_predictive_transfer_ceiling": 0.20,
    "adr_front_ls_step_scale": 0.75,
    "adr_front_ls_min_count_enabled": True,
    "adr_front_ls_min_weight": 0.04,
    "adr_front_ls_min_budget": 3,
    "adr_diverse_continuity_enabled": True,
    "adr_reliability_diagnostics_enabled": True,
}


ADR_EDA_POS_POF_RESPONSE_POLICY_V1_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V2_OVERRIDES,
    # Explicit POS/POF response matrix: POS estimates inherited-state trust,
    # POF estimates front repair/exploitation pressure.
    "adr_contextual_policy_variant": "pos_pof_response_v1",
    "adr_contextual_min_predictive": 0.04,
    "adr_contextual_max_predictive": 0.38,
    "adr_contextual_min_continuity": 0.06,
    "adr_contextual_max_continuity": 0.36,
    "adr_contextual_min_diversity": 0.10,
    "adr_contextual_max_diversity": 0.70,
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.12,
    "adr_front_ls_context_gate_enabled": False,
    "adr_front_ls_step_scale": 0.75,
    "adr_front_ls_min_count_enabled": True,
    "adr_front_ls_min_weight": 0.04,
    "adr_front_ls_min_budget": 3,
    "adr_diverse_continuity_enabled": True,
    "adr_reliability_diagnostics_enabled": True,
}


ADR_EDA_POS_POF_RESPONSE_POLICY_V2_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_POS_POF_RESPONSE_POLICY_V1_OVERRIDES,
    # V2: same POS/POF diagnostics, but a regime-aware bounded-prior update.
    # It pushes diversity under hidden POS movement while preserving enough of
    # ADR's fixed prior to avoid over-reacting on noisy small samples.
    "adr_contextual_policy_variant": "pos_pof_response_v2",
    "adr_contextual_max_predictive": 0.34,
    "adr_contextual_max_continuity": 0.32,
    "adr_contextual_min_diversity": 0.12,
    "adr_contextual_max_diversity": 0.72,
    "adr_contextual_max_front_ls": 0.14,
    "adr_pos_pof_v2_shift_budget": 0.32,
}


ADR_EDA_BLOCK_OPTIMIZED_RESPONSE_V1_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_POS_POF_RESPONSE_POLICY_V2_OVERRIDES,
    # Q1 sandbox: modular stronger ADR response while keeping the frozen
    # detector/backbone. The scheduler is POS/POF-aware and bounded by a
    # deviation budget from ADR's fixed prior; block-level improvements reuse
    # safe in-repo mechanisms only.
    "adr_contextual_policy_variant": "block_optimized_v1",
    "predictive_strategy": "sgea_centroid",
    "predictive_sgea_noise_scale": 0.70,
    "predictive_sgea_gate_floor": 0.15,
    "continuity_step_scale": 0.22,
    "continuity_jitter_scale": 0.10,
    "diversity_gate_enabled": False,
    "diversity_ratio_mild": 0.12,
    "diversity_ratio_medium": 0.18,
    "diversity_ratio_severe": 0.28,
    "adr_contextual_min_predictive": 0.04,
    "adr_contextual_max_predictive": 0.35,
    "adr_contextual_min_continuity": 0.08,
    "adr_contextual_max_continuity": 0.45,
    "adr_contextual_min_diversity": 0.20,
    "adr_contextual_max_diversity": 0.70,
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.15,
    "adr_front_ls_step_scale": 0.65,
    "adr_front_ls_min_count_enabled": True,
    "adr_front_ls_min_weight": 0.05,
    "adr_front_ls_min_budget": 4,
    "adr_block_optimized_shift_budget": 0.38,
}


ADR_EDA_GEOMETRIC_TRUST_DELTA_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_OVERRIDES,
    # Conservative paper candidate: keep the frozen ADR response table as a
    # prior and apply only a bounded POS/POF geometry delta.
    "adr_contextual_policy_variant": "geometric_trust_delta",
    "adr_contextual_max_predictive": 0.40,
    "adr_contextual_max_continuity": 0.40,
    "adr_contextual_min_diversity": 0.08,
    "adr_contextual_max_diversity": 0.62,
    "adr_contextual_max_front_ls": 0.00,
    "adr_geometry_trust_delta_enabled": True,
    "adr_geometry_trust_delta_rho_min": 0.05,
    "adr_geometry_trust_delta_rho_max": 0.32,
    "adr_geometry_trust_delta_scale": 0.22,
    "adr_reliability_diagnostics_enabled": True,
}



ADR_EDA_GEOMETRIC_GATED_PRIOR_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_OVERRIDES,
    # Q1 sandbox candidate: ADR-EDA frozen weights are the expert prior. Geometry
    # can only apply a small correction when POS/POF evidence passes a confidence gate.
    'adr_contextual_policy_variant': 'geometric_gated_prior',
    'adr_contextual_max_predictive': 0.40,
    'adr_contextual_max_continuity': 0.42,
    'adr_contextual_min_diversity': 0.08,
    'adr_contextual_max_diversity': 0.62,
    'adr_contextual_min_front_ls': 0.00,
    'adr_contextual_max_front_ls': 0.26,
    'adr_geometry_gated_prior_enabled': True,
    'adr_geometry_gated_prior_evidence_floor': 0.06,
    'adr_geometry_gated_prior_evidence_full': 0.34,
    'adr_geometry_gated_prior_delta_scale': 0.20,
    'adr_geometry_gated_prior_shift_budget': 0.18,
    'adr_geometry_gated_prior_front_ls_min_relative': 0.60,
    'adr_reliability_diagnostics_enabled': True,
}


ADR_EDA_ML_WEIGHTS_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Q1 sandbox: contextual-bandit scheduler over ADR response-weight experts.
    # The frozen ADR table remains the prior and bounds every learned move.
    "adr_contextual_weights_enabled": True,
    "adr_contextual_policy_variant": "ml_bandit",
    "adr_contextual_diversity_target": 0.18,
    "adr_contextual_min_predictive": 0.04,
    "adr_contextual_max_predictive": 0.42,
    "adr_contextual_min_continuity": 0.06,
    "adr_contextual_max_continuity": 0.42,
    "adr_contextual_min_diversity": 0.06,
    "adr_contextual_max_diversity": 0.62,
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.26,
    "adr_ml_weights_enabled": True,
    "adr_ml_ucb_alpha": 0.18,
    "adr_ml_exploration_epsilon": 0.08,
    "adr_ml_prior_blend": 0.42,
    "adr_ml_max_l1_shift": 0.22,
    "adr_ml_reward_scale": 0.04,
    "adr_reliability_diagnostics_enabled": True,
}

ADR_EDA_RELIABILITY_DIAGNOSTICS_V1_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Experimental Q1 sandbox: diagnostics/logging only. The reaction policy is
    # intentionally unchanged from the frozen ADR-EDA profile.
    "adr_reliability_diagnostics_enabled": True,
}


ADR_EDA_ADAPTIVE_WEIGHTS_V1_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_V2_1_FROZEN_OVERRIDES,
    # Experimental Q1 sandbox: reliability-aware continuous weight scheduler
    # plus SGEA-inspired diversity-preserving continuity.
    "adr_reliability_diagnostics_enabled": True,
    "adr_contextual_weights_enabled": True,
    "adr_contextual_policy_variant": "v3",
    "adr_contextual_temperature": 0.95,
    "adr_contextual_base_prior_strength": 0.65,
    "adr_contextual_min_predictive": 0.04,
    "adr_contextual_max_predictive": 0.40,
    "adr_contextual_min_continuity": 0.08,
    "adr_contextual_max_continuity": 0.42,
    "adr_contextual_min_diversity": 0.08,
    "adr_contextual_max_diversity": 0.58,
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.00,
    "adr_diverse_continuity_enabled": True,
}


ADR_EDA_ADAPTIVE_WEIGHTS_FRONT_LS_V1_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_ADAPTIVE_WEIGHTS_V1_OVERRIDES,
    # Conservative exploitation ablation: keep the reliability-aware scheduler,
    # but allow a small front-local-search budget when context suggests a
    # reliable, moderate post-change refinement opportunity.
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.10,
    "adr_front_ls_context_gate_enabled": False,
    "adr_front_ls_step_scale": 0.75,
    "adr_front_ls_min_count_enabled": True,
    "adr_front_ls_min_weight": 0.04,
    "adr_front_ls_min_budget": 3,
}


ADR_EDA_ADAPTIVE_WEIGHTS_SWITCH_GATE_V1_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_ADAPTIVE_WEIGHTS_V1_OVERRIDES,
    # Experimental Q1 sandbox: incumbent/switching-cost-aware acceptance proxy
    # for benchmark runs. Car-sharing uses the operational incumbent gate.
    "adr_switching_gate_enabled": True,
    "adr_switching_gate_min_improvement": 0.0,
    "adr_switching_gate_lambda": 0.015,
    "adr_switching_gate_metric": "l2_centroid",
}


ADR_EDA_GEOMETRIC_RESPONSE_POLICY_FRONT_LS_V1_OVERRIDES: Dict[str, object] = {
    **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V2_OVERRIDES,
    # Geometry-specific exploitation ablation: activate front-LS only as a
    # bounded complement to coherent/moderate POF movement.
    "adr_contextual_min_front_ls": 0.00,
    "adr_contextual_max_front_ls": 0.10,
    "adr_front_ls_context_gate_enabled": False,
    "adr_front_ls_step_scale": 0.75,
    "adr_front_ls_min_count_enabled": True,
    "adr_front_ls_min_weight": 0.04,
    "adr_front_ls_min_budget": 3,
    "adr_diverse_continuity_enabled": True,
    "adr_reliability_diagnostics_enabled": True,
}


PAPER_FIDELITY: Dict[str, PaperFidelityTarget] = {
    "dnsga2_b": PaperFidelityTarget(
        algorithm="dnsga2_b",
        paper_label="Deb et al. 2007 DNSGA-II-B",
        source_url="https://repository.ias.ac.in/83503/",
        target_status="paper_faithful_core_secondary_source",
        required_ingredients=[
            "NSGA-II backbone",
            "paper-style change detection / reevaluation protocol",
            "version-B response using mutated solutions after change",
            "same population and selection semantics as the original baseline",
        ],
        current_gaps=[
            "core DNSGA-II-B response is implemented as approx. 20% mutated immigrants from existing solutions",
            "the original Deb et al. PDF is not local, so exact primary-paper parameter wording remains to be attached",
        ],
    ),
    "mnsga2": PaperFidelityTarget(
        algorithm="mnsga2",
        paper_label="Sahmoud and Topcuoglu 2016 MNSGA-II",
        source_url="https://doi.org/10.1007/978-3-319-31153-1_21",
        target_status="paper_core_memory_based",
        required_ingredients=[
            "NSGA-II backbone",
            "external memory of non-dominated/diverse solutions",
            "memory update and pruning after evolutionary search",
            "partial population reinitialization from memory after environmental change",
            "random diversity guard to reduce stale-memory anchoring",
        ],
        current_gaps=[
            "core memory-based response is implemented in the unified runner",
            "memory sampling uses similarity-ranked snapshots and crowding-pruned elites; exact author-code update details are not reproduced line-by-line",
        ],
    ),
    "sgea": PaperFidelityTarget(
        algorithm="sgea",
        paper_label="Jiang and Yang 2017 SGEA",
        source_url="https://abdn.elsevierpure.com/en/publications/a-steady-state-and-generational-evolutionary-algorithm-for-dynami/",
        target_status="author_code_audited_core_port",
        required_ingredients=[
            "steady-state response after detected change",
            "reuse of outdated solutions with good distribution",
            "relocation of solutions close to the new PF using information from previous and new environments",
            "generational diversity preservation after the response stage",
        ],
        current_gaps=[
            "author-provided Visual Studio/C code is extracted and used as audit reference",
            "unified runner uses a Python port of the core mechanics because the original code has fixed native benchmarks and no Python benchmark adapter",
            "remaining deviations must be restricted to documented benchmark scaling/protocol choices",
        ],
    ),
    "trnsga2": PaperFidelityTarget(
        algorithm="trnsga2",
        paper_label="Jiang et al. 2018 Tr-NSGA-II",
        source_url="https://doi.org/10.1109/TEVC.2017.2771451",
        target_status="paper_faithful_core_optimization_preimage",
        required_ingredients=[
            "latent transfer between consecutive environments",
            "TCA-based source/target alignment",
            "Tr-IPG-style population generation in the new environment",
            "NSGA-II refinement over the transferred population",
        ],
        current_gaps=[
            "Tr-IPG/TCA and change-triggered transfer are integrated in the unified runner",
            "preimage now uses scipy L-BFGS-B optimization seeded by nearest-neighbor/stochastic refinement; exact interior-point semantics remain a documented deviation",
        ],
    ),
    "pfpa": PaperFidelityTarget(
        algorithm="pfpa",
        paper_label="Zhang et al. 2022 PFPA",
        source_url=repo_path("papers", "dinamico", "PFPA.pdf"),
        target_status="paper_core_local_linear_rmeda",
        required_ingredients=[
            "RM-MEDA backbone",
            "subpopulation 1: multi-step movement strategy with two step sizes",
            "subpopulation 2: polynomial fitting-based prediction strategy",
            "subpopulation 3: sampling strategy",
            "merge and non-dominated selection of the mixed population",
        ],
        current_gaps=[
            "PFPA three-subpopulation response is integrated in the unified runner",
            "static optimizer now uses local-linear RM-MEDA-style modelling over selected/nondominated solutions; still not a line-by-line reproduction of the original RM-MEDA package",
        ],
    ),
    "tda_dmoea": PaperFidelityTarget(
        algorithm="tda_dmoea",
        paper_label="TDA-DMOEA 2024",
        source_url="https://www.sciencedirect.com/science/article/abs/pii/S0020025523014524",
        target_status="paper_core_moead_smoa_approx",
        required_ingredients=[
            "explicit change-type detection from the paper",
            "response policies conditioned on the detected change type",
            "paper-level statistics and gating logic, not simplified substitutes",
        ],
        current_gaps=[
            "Wilcoxon/HV type detection and LPS/MA-TL/KPP responses are integrated in the unified runner",
            "SMOA is now a compact MOEA/D-DE implementation in this runner",
            "MA-TL uses a compact multi-view transfer/preimage approximation rather than full GFK closed-form machinery",
        ],
    ),
    "ab_dmoea": PaperFidelityTarget(
        algorithm="ab_dmoea",
        paper_label="AB-DMOEA 2024",
        source_url="https://www.sciencedirect.com/science/article/pii/S2210650224001597",
        target_status="author_code_audited_algorithmic_port",
        required_ingredients=[
            "adaptive boosting response mechanism from the paper",
            "dominated-solution reinforcement strategy",
            "static optimization boosting mechanism",
            "paper-defined collaboration among those components",
        ],
        current_gaps=[
            "author-provided MATLAB/PlatEMO-style code is extracted and used as audit reference",
            "adaptive boosting response, dominated reinforcement and static optimizer boosting are integrated in the unified runner",
            "static boosting selects among compact Python MOEA/D-DE, local-linear RM-MEDA and NSGA-II-DE modules because the original MATLAB code depends on PlatEMO Global/INDIVIDUAL infrastructure",
        ],
    ),
    "adr_dmoea": PaperFidelityTarget(
        algorithm="adr_dmoea",
        paper_label="Wang et al. 2026 ADR-DMOEA",
        source_url=repo_path("papers", "dinamico", "ADR-DMOEA.pdf"),
        target_status="paper_core_adaptive_dynamic_response_candidate",
        required_ingredients=[
            "MOEA/D static optimizer",
            "subpopulation-level adaptive dynamic response mechanism",
            "selection-probability update from static optimization distance",
            "diversity-driven, prediction-driven and memory-driven response strategies",
            "minimum strategy probability safeguard",
        ],
        current_gaps=[
            "core ADRM equations and the three response families are integrated in the unified runner",
            "prediction-driven strategy uses local PCA/MVA/KNN labels reconstructed from the paper text rather than authors' original code",
            "MOEA/D is represented by the compact in-runner MOEA/D-DE step used by other audited baselines",
        ],
    ),
    "adr_eda_v2_1": PaperFidelityTarget(
        algorithm="adr_eda_v2_1",
        paper_label="ADR-EDA benchmark-side profile in the thesis paper",
        source_url=repo_path("Plantilla_TFM_ETSIINF", "secciones", "03_Desarrollo_ADR_EDA.tex"),
        target_status="frozen_profile",
        required_ingredients=[
            "SPBN DynamicEDA backbone",
            "split drift signals",
            "adaptive lambda and cooldown logic",
            "diversity, predictive and continuity blocks",
            "front-local-search with contextual gating",
        ],
        current_gaps=[
            "the paper text also mentions benchmark-side threshold semantics not fully exposed in the generic runner",
        ],
        frozen_overrides=ADR_EDA_V2_1_FROZEN_OVERRIDES,
    ),
    "adr_eda_sgea_reaction": PaperFidelityTarget(
        algorithm="adr_eda_sgea_reaction",
        paper_label="Experimental ADR-EDA profile with SGEA-inspired reaction bias",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_variant_not_primary_baseline",
        required_ingredients=[
            "same ADR-EDA detector and SPBN backbone as the frozen profile",
            "stronger post-change archive retention",
            "continuity/prediction-heavy reaction allocation",
            "reduced random diversity injection unless drift evidence is strong",
        ],
        current_gaps=[
            "this is a candidate improvement profile, not the frozen ADR-EDA v2.1 used for primary claims",
            "must be validated on held-out seeds/problems before being promoted to a new paper version",
        ],
        frozen_overrides=ADR_EDA_SGEA_REACTION_OVERRIDES,
    ),
    "adr_eda_sgea_prediction": PaperFidelityTarget(
        algorithm="adr_eda_sgea_prediction",
        paper_label="ADR-EDA with SGEA-style centroid predictive block",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="clean_predictive_ablation",
        required_ingredients=[
            "same ADR-EDA detector and SPBN backbone as the frozen profile",
            "same ADR block allocation and contextual front-local-search gating",
            "predictive block replaced by SGEA-style centroid relocation",
        ],
        current_gaps=[
            "candidate predictive replacement; it must be validated before being promoted to the final ADR-EDA profile",
        ],
        frozen_overrides=ADR_EDA_SGEA_PREDICTION_OVERRIDES,
    ),
    "adr_eda_fast_reaction": PaperFidelityTarget(
        algorithm="adr_eda_fast_reaction",
        paper_label="ADR-EDA with aggressive internal detector/reaction throttling",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="detector_reaction_ablation",
        required_ingredients=[
            "same ADR-EDA SPBN backbone and reaction blocks as the frozen profile",
            "more sensitive mean/cov EWMA detector",
            "reduced cooldown and reaction throttling",
        ],
        current_gaps=[
            "diagnostic ablation only; must not be promoted unless it improves quality without excessive false positives",
        ],
        frozen_overrides=ADR_EDA_FAST_REACTION_OVERRIDES,
    ),
    "adr_eda_contextual_weights": PaperFidelityTarget(
        algorithm="adr_eda_contextual_weights",
        paper_label="ADR-EDA with contextual adaptive response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_contextual_weight_scheduler",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "same predictive, continuity and diversity response blocks",
            "continuous response-weight policy driven by intensity, abruptness, reliability, predictability and diversity deficit",
            "bounded softmax scheduler with explicit minimum/maximum block weights",
        ],
        current_gaps=[
            "experimental scheduler only; it must beat the frozen profile on matched seeds/problems before promotion",
            "benchmark-side signals are proxies derived from DynamicEDA response context, not the applied car-sharing reliability diagnostics",
        ],
        frozen_overrides=ADR_EDA_CONTEXTUAL_WEIGHTS_OVERRIDES,
    ),
    "adr_eda_contextual_weights_v2": PaperFidelityTarget(
        algorithm="adr_eda_contextual_weights_v2",
        paper_label="ADR-EDA with conservative contextual adaptive response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_contextual_weight_scheduler_v2",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "continuous response-weight policy with stronger abruptness penalty on continuity",
            "lower mandatory diversity floor and higher prior anchoring than contextual v1",
        ],
        current_gaps=[
            "experimental CAEPIA-screen variant only; must beat frozen ADR-EDA on matched h60/s5 before scale-up",
        ],
        frozen_overrides=ADR_EDA_CONTEXTUAL_WEIGHTS_V2_OVERRIDES,
    ),
    "adr_eda_contextual_weights_v3": PaperFidelityTarget(
        algorithm="adr_eda_contextual_weights_v3",
        paper_label="ADR-EDA with recovery-guarded contextual adaptive response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_contextual_weight_scheduler_v3",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "v2 contextual score model",
            "bounded deviation from the frozen ADR prior under recovery-risk proxies",
        ],
        current_gaps=[
            "experimental CAEPIA-screen variant only; must improve recovery/post-change quality before scale-up",
        ],
        frozen_overrides=ADR_EDA_CONTEXTUAL_WEIGHTS_V3_OVERRIDES,
    ),
    "adr_eda_geometric_response_policy": PaperFidelityTarget(
        algorithm="adr_eda_geometric_response_policy",
        paper_label="ADR-EDA with geometry-driven adaptive response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_geometry_response_scheduler",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "same predictive, continuity and diversity response blocks",
            "response weights inferred from measured POS/POF shift, POF overlap, shape change, displacement coherence and diversity deficit",
            "bounded blend with the frozen ADR prior to avoid uncontrolled scheduler drift",
        ],
        current_gaps=[
            "experimental CAEPIA-screen variant only; first detected change warms up the geometry memory",
            "front-local-search remains capped off while the X/F mixing bug is kept outside this line",
        ],
        frozen_overrides=ADR_EDA_GEOMETRIC_RESPONSE_POLICY_OVERRIDES,
    ),
    "adr_eda_geometric_response_policy_v2": PaperFidelityTarget(
        algorithm="adr_eda_geometric_response_policy_v2",
        paper_label="ADR-EDA with reliability-guarded geometry response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_geometry_response_scheduler_v2",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "same predictive, continuity and diversity response blocks",
            "geometry-driven weights from measured POS/POF movement",
            "continuity penalty when POS shift is high and displacement coherence is low",
        ],
        current_gaps=[
            "experimental CAEPIA-screen variant only; must beat frozen ADR-EDA on matched seeds/problems",
            "risk transfer is still formula-based and needs ablation against v1",
        ],
        frozen_overrides=ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V2_OVERRIDES,
    ),
    "adr_eda_geometric_response_policy_v3": PaperFidelityTarget(
        algorithm="adr_eda_geometric_response_policy_v3",
        paper_label="ADR-EDA with POS/POF risk-aware geometry response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_geometry_response_scheduler_v3",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "explicit POF-instability and POS-transfer-risk response matrix",
            "hard continuity/prediction caps under high POS risk and low coherence",
            "bounded front-local-search exploitation",
        ],
        current_gaps=[
            "experimental CAEPIA-screen variant only; must be validated against v2, adaptive v1 and frozen ADR-EDA",
        ],
        frozen_overrides=ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V3_OVERRIDES,
    ),
    "adr_eda_pos_pof_response_policy_v1": PaperFidelityTarget(
        algorithm="adr_eda_pos_pof_response_policy_v1",
        paper_label="ADR-EDA with explicit POS/POF response policy",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_pos_pof_response_policy_v1",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "explicit POS_change and POF_change diagnostics",
            "negative-transfer, coherent-motion and global-change response terms",
            "bounded front-local-search exploitation",
        ],
        current_gaps=[
            "experimental CAEPIA/Q1-screen variant only; must be validated against v3 and frozen ADR-EDA",
        ],
        frozen_overrides=ADR_EDA_POS_POF_RESPONSE_POLICY_V1_OVERRIDES,
    ),
    "adr_eda_pos_pof_response_policy_v2": PaperFidelityTarget(
        algorithm="adr_eda_pos_pof_response_policy_v2",
        paper_label="ADR-EDA with bounded-prior POS/POF response policy",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_pos_pof_response_policy_v2",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "explicit POS_change and POF_change diagnostics",
            "regime-aware hidden-negative-transfer response terms",
            "bounded deviation from the frozen ADR response prior",
        ],
        current_gaps=[
            "experimental CAEPIA/Q1-screen variant only; must be validated against v1, v3 and frozen ADR-EDA",
        ],
        frozen_overrides=ADR_EDA_POS_POF_RESPONSE_POLICY_V2_OVERRIDES,
    ),
    "adr_eda_block_optimized_response_v1": PaperFidelityTarget(
        algorithm="adr_eda_block_optimized_response_v1",
        paper_label="ADR-EDA with block-optimized modular response",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_block_optimized_response_v1",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "POS/POF-aware bounded response-weight scheduler",
            "diversity-preserving continuity and SGEA-style centroid prediction",
            "bounded front-local-search exploitation with safety fallback",
        ],
        current_gaps=[
            "Q1 sandbox variant; requires ablation by block before paper promotion",
        ],
        frozen_overrides=ADR_EDA_BLOCK_OPTIMIZED_RESPONSE_V1_OVERRIDES,
    ),
    "adr_eda_geometric_trust_delta": PaperFidelityTarget(
        algorithm="adr_eda_geometric_trust_delta",
        paper_label="ADR-EDA with conservative POS/POF trust-delta response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_geometry_trust_delta_scheduler",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "frozen ADR response table used as expert prior",
            "bounded POS/POF geometry delta over predictive, continuity and diversity weights",
            "trust coefficient rho based on measurement confidence and adaptation need",
        ],
        current_gaps=[
            "experimental CAEPIA-screen variant only; must beat frozen ADR-EDA on matched h60/s5",
        ],
        frozen_overrides=ADR_EDA_GEOMETRIC_TRUST_DELTA_OVERRIDES,
    ),
    'adr_eda_geometric_gated_prior': PaperFidelityTarget(
        algorithm='adr_eda_geometric_gated_prior',
        paper_label='ADR-EDA with gated prior POS/POF adaptive response weights',
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status='experimental_geometry_gated_prior_scheduler',
        required_ingredients=[
            'same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1',
            'frozen ADR response table used as expert prior',
            'POS/POF correction gated by geometry confidence and adaptation evidence',
            'bounded L1 shift and relative front-local-search floor to avoid overreaction',
        ],
        current_gaps=[
            'experimental CAEPIA-screen variant only; validate first on df1/fda controls before overnight scale-up',
        ],
        frozen_overrides=ADR_EDA_GEOMETRIC_GATED_PRIOR_OVERRIDES,
    ),
    "adr_eda_ml_weights": PaperFidelityTarget(
        algorithm="adr_eda_ml_weights",
        paper_label="ADR-EDA with contextual-bandit ML response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_ml_weight_scheduler",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "frozen ADR response table used as expert prior",
            "contextual LinUCB bandit selecting among interpretable response-weight experts",
            "online credit from reaction reward with bounded deviation from ADR weights",
        ],
        current_gaps=[
            "experimental Q1 sandbox only; validate against frozen ADR-EDA before promotion",
        ],
        frozen_overrides=ADR_EDA_ML_WEIGHTS_OVERRIDES,
    ),
    "adr_eda_reliability_diagnostics_v1": PaperFidelityTarget(
        algorithm="adr_eda_reliability_diagnostics_v1",
        paper_label="ADR-EDA reliability diagnostics sandbox",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_diagnostics_only",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "structured inherited-state reliability and gate diagnostics",
            "no change to response weights or acceptance logic",
        ],
        current_gaps=[
            "benchmark-side reliability is a proxy from drift/gate traces; no transfer block is included",
        ],
        frozen_overrides=ADR_EDA_RELIABILITY_DIAGNOSTICS_V1_OVERRIDES,
    ),
    "adr_eda_adaptive_weights_v1": PaperFidelityTarget(
        algorithm="adr_eda_adaptive_weights_v1",
        paper_label="ADR-EDA reliability-aware adaptive response weights",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_reliability_weight_scheduler",
        required_ingredients=[
            "same frozen ADR-EDA detector and SPBN backbone as ADR-EDA v2.1",
            "adaptive response weights from severity, reliability and gate-reject history",
            "diversity-preserving continuity selection inspired by SGEA",
        ],
        current_gaps=[
            "no Tr-NSGA-II/DANE/TCA transfer is implemented",
            "must be validated outside smoke before any thesis claim",
        ],
        frozen_overrides=ADR_EDA_ADAPTIVE_WEIGHTS_V1_OVERRIDES,
    ),
    "adr_eda_adaptive_weights_front_ls_v1": PaperFidelityTarget(
        algorithm="adr_eda_adaptive_weights_front_ls_v1",
        paper_label="ADR-EDA adaptive weights with bounded front local search",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_front_local_search_ablation",
        required_ingredients=[
            "adaptive weights v1",
            "small bounded front-local-search budget",
            "same frozen ADR detector and response gate",
        ],
        current_gaps=[
            "front-local-search can improve convergence but may add runtime and reduce exploratory recovery after shocks",
        ],
        frozen_overrides=ADR_EDA_ADAPTIVE_WEIGHTS_FRONT_LS_V1_OVERRIDES,
    ),
    "adr_eda_adaptive_weights_switch_gate_v1": PaperFidelityTarget(
        algorithm="adr_eda_adaptive_weights_switch_gate_v1",
        paper_label="ADR-EDA adaptive weights with switching-cost-aware gate",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_switching_cost_gate",
        required_ingredients=[
            "adaptive weights v1",
            "front-quality improvement compared against a switching-cost penalty",
            "structured rejection reason and switch-cost logging",
        ],
        current_gaps=[
            "benchmark switching cost is a centroid-distance proxy; operational cost is handled in car-sharing profiles",
        ],
        frozen_overrides=ADR_EDA_ADAPTIVE_WEIGHTS_SWITCH_GATE_V1_OVERRIDES,
    ),
    "adr_eda_geometric_response_policy_front_ls_v1": PaperFidelityTarget(
        algorithm="adr_eda_geometric_response_policy_front_ls_v1",
        paper_label="ADR-EDA geometry response policy with bounded front local search",
        source_url=repo_path("tests", "edas", "clean_dynamic_sota_comparison.py"),
        target_status="experimental_geometry_front_local_search_ablation",
        required_ingredients=[
            "geometric response policy v2",
            "bounded front-local-search budget under coherent/moderate movement",
            "diversity-preserving continuity and reliability diagnostics",
        ],
        current_gaps=[
            "must be validated against geometric v2 to ensure exploitation does not harm shock recovery",
        ],
        frozen_overrides=ADR_EDA_GEOMETRIC_RESPONSE_POLICY_FRONT_LS_V1_OVERRIDES,
    ),
}


@dataclass
class GAConfig:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 1
    crossover_prob: float = 0.90
    sbx_eta: float = 20.0
    mutation_eta: float = 20.0
    detector_sentinels: int = 20
    detector_alpha: float = 0.10
    detector_lambda: float = 0.75
    detector_k_sigma: float = 2.5
    detector_refresh: int = 8
    detector_cooldown: int = 1
    detector_warmup: int = 6
    detector_history: int = 40
    detector_tolerance: int = 5
    dnsga_immigrant_ratio: float = 0.20
    dnsga_hypermutation_prob: float = 0.80
    dnsga_hypermutation_eta: float = 5.0
    memory_ratio: float = 0.20
    memory_hyper_ratio: float = 0.10
    memory_max_snapshots: int = 10
    memory_snapshot_size: int = 24
    memory_hypermutation_prob: float = 0.70
    memory_hypermutation_eta: float = 8.0
    mnsga_memory_ratio: float = 0.30
    mnsga_memory_mutation_ratio: float = 0.10
    mnsga_random_ratio: float = 0.10
    mnsga_similarity_topk: int = 3
    mnsga_memory_capacity_factor: float = 1.50


@dataclass
class DNSGA2Config:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 1
    crossover_prob: float = 0.90
    sbx_eta: float = 20.0
    mutation_eta: float = 20.0
    response_ratio: float = 0.20
    change_mutation_prob: float = 1.0
    change_mutation_eta: float = 5.0
    detection_mode: str = "oracle"
    detect_tolerance: float = 1e-12
    detect_precision: int = 5
    adr_meta_enabled: bool = False
    adr_meta_min_update_interval: int = 1
    adr_meta_max_update_interval: int = 3
    adr_meta_drift_threshold: float = 0.08
    adr_meta_hard_drift_threshold: float = 0.22


@dataclass
class SGEAConfig:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 1
    crossover_prob: float = 1.0
    sbx_eta: float = 20.0
    mutation_eta: float = 20.0
    archive_size: int = 0
    detection_mode: str = "paper"
    detection_fraction: float = 0.10
    detect_tolerance: float = 1e-12
    detect_precision: int = 5
    adr_meta_enabled: bool = False
    adr_meta_min_update_interval: int = 1
    adr_meta_max_update_interval: int = 3
    adr_meta_drift_threshold: float = 0.08
    adr_meta_hard_drift_threshold: float = 0.22


@dataclass
class TrNSGA2Config:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 50
    crossover_prob: float = 0.90
    sbx_eta: float = 20.0
    mutation_eta: float = 20.0
    transfer_samples: int = 200
    tca_dim: int = 20
    tca_sigma: float = 0.60
    tca_mu: float = 0.50
    preimage_neighbors: int = 8
    preimage_refine_steps: int = 12
    preimage_refine_candidates: int = 8
    refine_sigma_start: float = 0.10
    refine_sigma_decay: float = 0.65
    preimage_use_scipy: bool = True
    preimage_optimizer_maxiter: int = 20
    detection_mode: str = "oracle"
    transfer_when_no_explicit_events: bool = False
    detect_precision: int = 5
    adr_meta_enabled: bool = False
    adr_meta_min_update_interval: int = 1
    adr_meta_max_update_interval: int = 3
    adr_meta_drift_threshold: float = 0.08
    adr_meta_hard_drift_threshold: float = 0.22


@dataclass
class PFPAConfig:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 1
    movement_steps: Tuple[float, float] = (0.3, 1.0)
    compression_ratio: float = 0.10
    polynomial_max_degree: int = 6
    sampling_n2_factor: int = 1
    rmeda_clusters: int = 0
    rmeda_model_noise: float = 0.035
    rmeda_latent_scale: float = 0.18
    detection_mode: str = "oracle"
    detect_precision: int = 5
    adr_meta_enabled: bool = False
    adr_meta_min_update_interval: int = 1
    adr_meta_max_update_interval: int = 3
    adr_meta_drift_threshold: float = 0.08
    adr_meta_hard_drift_threshold: float = 0.22


@dataclass
class TDADMOEAConfig:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 1
    crossover_prob: float = 0.90
    sbx_eta: float = 20.0
    mutation_eta: float = 20.0
    pos_p_threshold: float = 0.05
    pof_threshold_2obj: float = 0.008
    pof_threshold_3obj: float = 0.03
    knee_ratio: float = 0.35
    transfer_pool_samples: int = 160
    ma_tl_noise: float = 0.04
    detection_mode: str = "oracle"
    detect_precision: int = 5
    adr_meta_enabled: bool = False
    adr_meta_min_update_interval: int = 1
    adr_meta_max_update_interval: int = 3
    adr_meta_drift_threshold: float = 0.08
    adr_meta_hard_drift_threshold: float = 0.22


@dataclass
class ABDMOEAConfig:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 1
    crossover_prob: float = 0.90
    sbx_eta: float = 20.0
    mutation_eta: float = 20.0
    reinforcement_threshold: float = 0.70
    boost_alpha: float = 0.50
    min_strategy_prob: float = 0.08
    memory_similarity_eps: float = 0.05
    memory_size: int = 100
    de_f: float = 0.50
    de_cr: float = 0.90
    detection_mode: str = "oracle"
    detect_precision: int = 5
    adr_meta_enabled: bool = False
    adr_meta_min_update_interval: int = 1
    adr_meta_max_update_interval: int = 3
    adr_meta_drift_threshold: float = 0.08
    adr_meta_hard_drift_threshold: float = 0.22


@dataclass
class ADRDMOEAConfig:
    pop_size: int
    n_variables: int
    horizon: int
    generations_per_step: int = 1
    crossover_prob: float = 0.90
    sbx_eta: float = 20.0
    mutation_eta: float = 20.0
    subpop_size: int = 10
    gamma: float = 0.20
    theta_min: float = 0.10
    diversity_poly_prob: float = 0.50
    diversity_gaussian_sigma: float = 0.035
    prediction_neighbors: int = 5
    prediction_random_factor: int = 2
    prediction_pca_dim: int = 0
    memory_similarity_eps: float = 1e-4
    memory_size: int = 50
    memory_noise: float = 0.05
    de_f: float = 0.50
    de_cr: float = 1.0
    moead_neighborhood_ratio: float = 0.15
    moead_max_neighbors: int = 15
    detection_mode: str = "oracle"
    detect_precision: int = 5
    adr_meta_enabled: bool = False
    adr_meta_min_update_interval: int = 1
    adr_meta_max_update_interval: int = 3
    adr_meta_drift_threshold: float = 0.08
    adr_meta_hard_drift_threshold: float = 0.22


ADR_META_DEFAULT_OVERRIDES: Dict[str, object] = {
    "adr_meta_enabled": True,
    "adr_meta_min_update_interval": 0,
    "adr_meta_max_update_interval": 120,
    "adr_meta_drift_threshold": 0.30,
    "adr_meta_hard_drift_threshold": 0.60,
}


def _adr_meta_kwargs(overrides: Dict[str, object]) -> Dict[str, object]:
    return {
        "adr_meta_enabled": bool(overrides.get("adr_meta_enabled", False)),
        "adr_meta_min_update_interval": int(overrides.get("adr_meta_min_update_interval", 1)),
        "adr_meta_max_update_interval": int(overrides.get("adr_meta_max_update_interval", 3)),
        "adr_meta_drift_threshold": float(overrides.get("adr_meta_drift_threshold", 0.08)),
        "adr_meta_hard_drift_threshold": float(overrides.get("adr_meta_hard_drift_threshold", 0.22)),
    }


def _adr_meta_objective_drift(previous_eval: np.ndarray, current_eval: np.ndarray) -> float:
    previous = np.asarray(previous_eval, dtype=float)
    current = np.asarray(current_eval, dtype=float)
    if previous.size == 0 or current.size == 0:
        return float("inf")
    n = min(previous.shape[0], current.shape[0])
    previous = previous[:n]
    current = current[:n]
    if previous.ndim == 2 and previous.shape[1] > 1:
        previous = previous[:, 1:]
        current = current[:, 1:]
    delta = np.linalg.norm(current - previous, axis=1 if current.ndim == 2 else 0)
    scale = np.linalg.norm(previous, axis=1 if previous.ndim == 2 else 0)
    return float(np.mean(delta) / max(float(np.mean(scale)), 1e-12))


def _adr_meta_should_update(
    cfg,
    t: int,
    previous_eval: np.ndarray,
    current_eval: np.ndarray,
    last_update_t: Optional[int],
) -> Tuple[bool, float, str]:
    drift = _adr_meta_objective_drift(previous_eval, current_eval)
    if not bool(getattr(cfg, "adr_meta_enabled", False)):
        return True, drift, "disabled"
    if last_update_t is None:
        return True, drift, "bootstrap"

    elapsed = int(t - last_update_t)
    max_interval = max(1, int(getattr(cfg, "adr_meta_max_update_interval", 3)))
    min_interval = max(0, int(getattr(cfg, "adr_meta_min_update_interval", 1)))
    hard_threshold = float(getattr(cfg, "adr_meta_hard_drift_threshold", 0.22))
    threshold = float(getattr(cfg, "adr_meta_drift_threshold", 0.08))

    if drift >= hard_threshold:
        return True, drift, "hard_drift"
    if elapsed >= max_interval:
        return True, drift, "max_interval"
    if elapsed < min_interval:
        return False, drift, "min_interval"
    if drift >= threshold:
        return True, drift, "drift"
    return False, drift, "reuse"


def _sin_signal(t: int, period: int) -> float:
    return float(np.sin(2.0 * np.pi * t / max(period, 1)))


def _step_signal(t: int, tau: int) -> float:
    return 1.0 if ((t // max(tau, 1)) % 2 == 0) else 0.0


def _piecewise_signal(t: int, change_times: List[int], levels: List[float]) -> float:
    idx = 0
    for i, cp in enumerate(change_times):
        if t >= cp:
            idx = i + 1
        else:
            break
    return float(levels[idx])


class DynamicProblemBase:
    name = "base"
    n_objectives = 2
    use_dynamic_hv_ref = False

    def __init__(self, n_variables: int, horizon: int):
        self.n_variables = n_variables
        self.horizon = horizon
        self.t = 0

    def update_time(self) -> None:
        self.t += 1

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        objs, cv = self.evaluate(x)
        return np.concatenate(([float(cv)], np.asarray(objs, dtype=float)))

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        raise NotImplementedError

    def true_change_points(self, horizon: int) -> List[int]:
        return []


class DZDT1Step(DynamicProblemBase):
    name = "dzdt1_step"

    def __init__(self, n_variables: int, horizon: int, tau: int = None):
        super().__init__(n_variables, horizon)
        self.tau = tau if tau is not None else max(12, horizon // 4)

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        s = _step_signal(self.t, self.tau)
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:]) + 0.4 * s
        g = max(g, 1e-12)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
        return np.array([f1, f2]), 0.0

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        s = _step_signal(self.t, self.tau)
        g = max(1.0 + 0.4 * s, 1e-12)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
        return np.column_stack((f1, f2))

    def true_change_points(self, horizon: int) -> List[int]:
        return [t for t in range(self.tau, horizon, self.tau)]


class DZDT1Sine(DynamicProblemBase):
    name = "dzdt1_sine"

    def __init__(self, n_variables: int, horizon: int, period: int = None):
        super().__init__(n_variables, horizon)
        self.period = period if period is not None else max(40, horizon)

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        s = _sin_signal(self.t, self.period)
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:]) + 0.3 * s
        g = max(g, 1e-12)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
        return np.array([f1, f2]), 0.0

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        s = _sin_signal(self.t, self.period)
        g = max(1.0 + 0.3 * s, 1e-12)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
        return np.column_stack((f1, f2))


class DZDT2Shape(DynamicProblemBase):
    name = "dzdt2_shape"

    def __init__(self, n_variables: int, horizon: int, period: int = None):
        super().__init__(n_variables, horizon)
        self.period = period if period is not None else max(40, horizon)

    def _power(self):
        return 1.2 + 0.8 * (0.5 * (1.0 + _sin_signal(self.t, self.period)))

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        p = self._power()
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:])
        g = max(g, 1e-12)
        f2 = g * (1.0 - np.power(np.clip(f1 / g, 0.0, None), p))
        return np.array([f1, f2]), 0.0

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        p = self._power()
        f2 = 1.0 - np.power(f1, p)
        return np.column_stack((f1, f2))


class DZDT3Shocks(DynamicProblemBase):
    name = "dzdt3_shocks"

    def __init__(self, n_variables: int, horizon: int):
        super().__init__(n_variables, horizon)
        self.change_times = [max(10, horizon // 4), max(20, horizon // 2), max(30, (3 * horizon) // 4)]
        self.levels = [0.0, 0.25, -0.15, 0.30]

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        s = _piecewise_signal(self.t, self.change_times, self.levels)
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:]) + 0.35 * s
        g = max(g, 1e-12)
        z = np.clip(f1 / g, 0.0, None)
        f2 = g * (1.0 - np.sqrt(z) - z * np.sin(10.0 * np.pi * f1))
        return np.array([f1, f2]), 0.0

    def true_pareto(self, n_points: int = 500) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        s = _piecewise_signal(self.t, self.change_times, self.levels)
        g = max(1.0 + 0.35 * s, 1e-12)
        z = np.clip(f1 / g, 0.0, None)
        f2 = g * (1.0 - np.sqrt(z) - z * np.sin(10.0 * np.pi * f1))
        return non_dominated_points(np.column_stack((f1, f2)))

    def true_change_points(self, horizon: int) -> List[int]:
        return [t for t in self.change_times if t < horizon]


class FDA1Lite(DynamicProblemBase):
    name = "fda1_lite"

    def __init__(self, n_variables: int, horizon: int, period: int = None):
        super().__init__(n_variables, horizon)
        self.period = period if period is not None else max(40, horizon)

    def _G(self):
        return 0.5 * (1.0 + _sin_signal(self.t, self.period))

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        g_shift = self._G()
        f1 = x[0]
        g = 1.0 + np.sum((x[1:] - g_shift) ** 2)
        g = max(g, 1e-12)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
        return np.array([f1, f2]), 0.0

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - np.sqrt(f1)))


class FDA2Lite(DynamicProblemBase):
    name = "fda2_lite"

    def __init__(self, n_variables: int, horizon: int, period: int = None):
        super().__init__(n_variables, horizon)
        self.period = period if period is not None else max(40, horizon)

    def _G(self):
        return 0.5 * (1.0 + _sin_signal(self.t, self.period))

    def _H(self):
        return 0.5 + 1.5 * (0.5 * (1.0 + _sin_signal(self.t, self.period)))

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        g_shift = self._G()
        h_pow = self._H()
        f1 = x[0]
        g = 1.0 + np.sum((x[1:] - g_shift) ** 2)
        g = max(g, 1e-12)
        f2 = g * (1.0 - np.power(np.clip(f1 / g, 0.0, None), h_pow))
        return np.array([f1, f2]), 0.0

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        h_pow = self._H()
        return np.column_stack((f1, 1.0 - np.power(f1, h_pow)))


class DConstrainedZDT1(DynamicProblemBase):
    name = "dconstrained_zdt1"

    def __init__(self, n_variables: int, horizon: int, period: int = None):
        super().__init__(n_variables, horizon)
        self.period = period if period is not None else max(40, horizon)
        self.radius = 0.25

    def _center(self):
        return 0.5 + 0.25 * _sin_signal(self.t, self.period)

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        s = _sin_signal(self.t, self.period)
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:]) + 0.2 * s
        g = max(g, 1e-12)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))

        c = self._center()
        viol = (x[0] - c) ** 2 + (x[1] - 0.5) ** 2 - self.radius ** 2
        cv = max(0.0, viol)
        return np.array([f1, f2]), cv

    def true_pareto(self, n_points: int = 250) -> np.ndarray:
        c = self._center()
        xs = np.linspace(0.0, 1.0, n_points)
        pf = []
        for x1 in xs:
            d = self.radius ** 2 - (x1 - c) ** 2
            if d < 0:
                continue
            x2_min = max(0.0, 0.5 - np.sqrt(d))
            x = np.zeros(self.n_variables)
            x[0] = x1
            x[1] = x2_min
            obj, cv = self.evaluate(x)
            if cv <= 1e-12:
                pf.append(obj)
        if not pf:
            return np.empty((0, 2))
        return non_dominated_points(np.asarray(pf, dtype=float))


class DDTLZ2_3Obj(DynamicProblemBase):
    name = "ddtlz2_3obj"
    n_objectives = 3

    def __init__(self, n_variables: int, horizon: int, period: int = None):
        super().__init__(n_variables, horizon)
        self.period = period if period is not None else max(40, horizon)

    def _a(self):
        return 0.5 * (1.0 + _sin_signal(self.t, self.period))

    def evaluate(self, x: np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        a = self._a()
        g = np.sum((x[2:] - a) ** 2)
        theta1 = np.pi * x[0] / 2.0
        theta2 = np.pi * x[1] / 2.0
        f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
        f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
        f3 = (1.0 + g) * np.sin(theta1)
        return np.array([f1, f2, f3]), 0.0

    def true_pareto(self, n_points: int = 400) -> np.ndarray:
        k = max(12, int(np.sqrt(n_points)))
        theta = np.linspace(0.0, np.pi / 2.0, k)
        phi = np.linspace(0.0, np.pi / 2.0, k)
        pts = []
        for th in theta:
            for ph in phi:
                pts.append([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
        return np.asarray(pts, dtype=float)


class CEC2018DFBase(DynamicProblemBase):
    """CEC 2018 DF suite using normalized decision variables in this runner."""

    base_name = "df"
    use_dynamic_hv_ref = True

    def __init__(self, n_variables: int, horizon: int, taut: int = 10, nt: int = 10):
        super().__init__(n_variables, horizon)
        self.taut = int(max(1, taut))
        self.nt = int(max(1, nt))
        self.native_lower, self.native_upper = self._native_bounds()

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_variables, dtype=float), np.ones(self.n_variables, dtype=float)

    def _to_native(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), LOWER_BOUND, UPPER_BOUND)
        return self.native_lower + x * (self.native_upper - self.native_lower)

    def _cec_time(self) -> float:
        return float((self.t // self.taut) / self.nt)

    def evaluate(self, x: np.ndarray):
        return self._evaluate_native(self._to_native(x)), 0.0

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def true_change_points(self, horizon: int) -> List[int]:
        return [t for t in range(self.taut, horizon, self.taut)]

    @staticmethod
    def _pf_from_arrays(arrays: List[np.ndarray], nondominated: bool = False) -> np.ndarray:
        pf = np.column_stack([np.asarray(arr, dtype=float).ravel(order="F") for arr in arrays])
        return non_dominated_points(pf) if nondominated else pf


class CEC2018DF1(CEC2018DFBase):
    base_name = "df1"

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        v = math.sin(0.5 * math.pi * t)
        g_shift = abs(v)
        h_pow = 0.75 * v + 1.25
        g = 1.0 + float(np.sum((x[1:] - g_shift) ** 2))
        f1 = x[0]
        f2 = g * (1.0 - np.power(np.clip(f1 / max(g, 1e-12), 0.0, None), h_pow))
        return np.array([f1, f2], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        h_pow = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - f1 ** h_pow))


class CEC2018DF2(CEC2018DFBase):
    base_name = "df2"

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = abs(math.sin(0.5 * math.pi * t))
        r = int((self.n_variables - 1) * g_shift)
        mask = np.ones(self.n_variables, dtype=bool)
        mask[r] = False
        g = 1.0 + float(np.sum((x[mask] - g_shift) ** 2))
        f1 = x[r]
        f2 = g * (1.0 - math.sqrt(max(f1 / max(g, 1e-12), 0.0)))
        return np.array([f1, f2], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - np.sqrt(f1)))


class CEC2018DF3(CEC2018DFBase):
    base_name = "df3"

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[1:] = -1.0
        ub[1:] = 2.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        h_pow = g_shift + 1.5
        g = 1.0 + float(np.sum((x[1:] - g_shift - x[0] ** h_pow) ** 2))
        f1 = x[0]
        f2 = g * (1.0 - np.power(np.clip(f1 / max(g, 1e-12), 0.0, None), h_pow))
        return np.array([f1, f2], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        h_pow = math.sin(0.5 * math.pi * t) + 1.5
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - f1 ** h_pow))


class CEC2018DF4(CEC2018DFBase):
    base_name = "df4"

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.full(self.n_variables, -2.0, dtype=float), np.full(self.n_variables, 2.0, dtype=float)

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        a = math.sin(0.5 * math.pi * t)
        b = 1.0 + abs(math.cos(0.5 * math.pi * t))
        h_pow = 1.5 + a
        c = max(abs(a), a + b)
        g = 1.0
        for i in range(1, self.n_variables):
            g += float((x[i] - (a * (x[0] / c) ** 2 / (i + 1))) ** 2)
        f1 = g * abs(x[0] - a) ** h_pow
        f2 = g * abs(x[0] - a - b) ** h_pow
        return np.array([f1, f2], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        a = math.sin(0.5 * math.pi * t)
        b = 1.0 + abs(math.cos(0.5 * math.pi * t))
        h_pow = 1.5 + a
        x = np.linspace(a, a + b, n_points)
        return np.column_stack((np.abs(x - a) ** h_pow, np.abs(x - a - b) ** h_pow))


class CEC2018DF5(CEC2018DFBase):
    base_name = "df5"

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[1:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        w = math.floor(10.0 * g_shift)
        g = 1.0 + float(np.sum((x[1:] - g_shift) ** 2))
        bump = 0.02 * math.sin(w * math.pi * x[0])
        return np.array([g * (x[0] + bump), g * (1.0 - x[0] + bump)], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        w = math.floor(10.0 * math.sin(0.5 * math.pi * t))
        x = np.linspace(0.0, 1.0, n_points)
        bump = 0.02 * np.sin(w * math.pi * x)
        return np.column_stack((x + bump, 1.0 - x + bump))


class CEC2018DF6(CEC2018DFBase):
    base_name = "df6"

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[1:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        a = 0.2 + 2.8 * abs(g_shift)
        y = x[1:] - g_shift
        g = 1.0 + float(np.sum(abs(g_shift) * y ** 2 - 10.0 * np.cos(2.0 * math.pi * y) + 10.0))
        f1_base = x[0] + 0.1 * math.sin(3.0 * math.pi * x[0])
        f2_base = 1.0 - x[0] + 0.1 * math.sin(3.0 * math.pi * x[0])
        return np.array([g * max(f1_base, 0.0) ** a, g * max(f2_base, 0.0) ** a], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        a = 0.2 + 2.8 * abs(math.sin(0.5 * math.pi * t))
        x = np.linspace(0.0, 1.0, n_points)
        f1 = np.maximum(x + 0.1 * np.sin(3.0 * math.pi * x), 0.0) ** a
        f2 = np.maximum(1.0 - x + 0.1 * np.sin(3.0 * math.pi * x), 0.0) ** a
        return np.column_stack((f1, f2))


class CEC2018DF7(CEC2018DFBase):
    base_name = "df7"

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[0] = 1.0
        ub[0] = 4.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        a = 5.0 * math.cos(0.5 * math.pi * t)
        target = 1.0 / (1.0 + np.exp(a * (x[0] - 2.5)))
        g = 1.0 + float(np.sum((x[1:] - target) ** 2))
        return np.array([g * (1.0 + t) / x[0], g * x[0] / (1.0 + t)], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        x = np.linspace(1.0, 4.0, n_points)
        pf = np.column_stack(((1.0 + t) / x, x / (1.0 + t)))
        return pf[np.argsort(pf[:, 0])]


class CEC2018DF8(CEC2018DFBase):
    base_name = "df8"

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[1:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        a = 2.25 + 2.0 * math.cos(2.0 * math.pi * t)
        tmp = g_shift * math.sin(4.0 * math.pi * x[0]) / (1.0 + abs(g_shift))
        g = 1.0 + float(np.sum((x[1:] - tmp) ** 2))
        f1_base = x[0] + 0.1 * math.sin(3.0 * math.pi * x[0])
        f2_base = 1.0 - x[0] + 0.1 * math.sin(3.0 * math.pi * x[0])
        return np.array([g * f1_base, g * max(f2_base, 0.0) ** a], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        a = 2.25 + 2.0 * math.cos(2.0 * math.pi * t)
        x = np.linspace(0.0, 1.0, n_points)
        f1 = x + 0.1 * np.sin(3.0 * math.pi * x)
        f2 = np.maximum(1.0 - x + 0.1 * np.sin(3.0 * math.pi * x), 0.0) ** a
        return np.column_stack((f1, f2))


class CEC2018DF9(CEC2018DFBase):
    base_name = "df9"

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[1:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        n_segments = 1.0 + math.floor(10.0 * abs(math.sin(0.5 * math.pi * t)))
        g = 1.0
        for i in range(1, self.n_variables):
            g += float((x[i] - math.cos(4.0 * t + x[0] + x[i - 1])) ** 2)
        bump = max(0.0, (0.1 + 0.5 / n_segments) * math.sin(2.0 * n_segments * math.pi * x[0]))
        return np.array([g * (x[0] + bump), g * (1.0 - x[0] + bump)], dtype=float)

    def true_pareto(self, n_points: int = 300) -> np.ndarray:
        t = self._cec_time()
        n_segments = 1.0 + math.floor(10.0 * abs(math.sin(0.5 * math.pi * t)))
        x = np.linspace(0.0, 1.0, n_points)
        bump = np.maximum(0.0, (0.1 + 0.5 / n_segments) * np.sin(2.0 * n_segments * math.pi * x))
        return non_dominated_points(np.column_stack((x + bump, 1.0 - x + bump)))


class CEC2018DF10(CEC2018DFBase):
    base_name = "df10"
    n_objectives = 3

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[2:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        h_pow = 2.25 + 2.0 * math.cos(0.5 * math.pi * t)
        target = math.sin(2.0 * math.pi * (x[0] + x[1])) / (1.0 + abs(g_shift))
        g = 1.0 + float(np.sum((x[2:] - target) ** 2))
        f1 = g * math.sin(0.5 * math.pi * x[0]) ** h_pow
        f2 = g * (math.sin(0.5 * math.pi * x[1]) * math.cos(0.5 * math.pi * x[0])) ** h_pow
        f3 = g * (math.cos(0.5 * math.pi * x[1]) * math.cos(0.5 * math.pi * x[0])) ** h_pow
        return np.array([f1, f2, f3], dtype=float)

    def true_pareto(self, n_points: int = 400) -> np.ndarray:
        k = max(12, int(np.ceil(np.sqrt(n_points))))
        x1, x2 = np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k), indexing="xy")
        t = self._cec_time()
        h_pow = 2.25 + 2.0 * math.cos(0.5 * math.pi * t)
        f1 = np.sin(0.5 * math.pi * x1) ** h_pow
        f2 = (np.sin(0.5 * math.pi * x2) * np.cos(0.5 * math.pi * x1)) ** h_pow
        f3 = (np.cos(0.5 * math.pi * x2) * np.cos(0.5 * math.pi * x1)) ** h_pow
        return self._pf_from_arrays([f1, f2, f3], nondominated=False)


class CEC2018DF11(CEC2018DFBase):
    base_name = "df11"
    n_objectives = 3

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = abs(math.sin(0.5 * math.pi * t))
        g = 1.0 + g_shift + float(np.sum((x[2:] - 0.5 * g_shift * x[0]) ** 2))
        y1 = math.pi * g_shift / 6.0 + (math.pi / 2.0 - math.pi * g_shift / 3.0) * x[0]
        y2 = math.pi * g_shift / 6.0 + (math.pi / 2.0 - math.pi * g_shift / 3.0) * x[1]
        return np.array([g * math.sin(y1), g * math.sin(y2) * math.cos(y1), g * math.cos(y2) * math.cos(y1)], dtype=float)

    def true_pareto(self, n_points: int = 400) -> np.ndarray:
        k = max(12, int(np.ceil(np.sqrt(n_points))))
        x1, x2 = np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k), indexing="xy")
        t = self._cec_time()
        g_shift = abs(math.sin(0.5 * math.pi * t))
        y1 = math.pi * g_shift / 6.0 + (math.pi / 2.0 - math.pi * g_shift / 3.0) * x1
        y2 = math.pi * g_shift / 6.0 + (math.pi / 2.0 - math.pi * g_shift / 3.0) * x2
        f1 = (1.0 + g_shift) * np.sin(y1)
        f2 = (1.0 + g_shift) * np.sin(y2) * np.cos(y1)
        f3 = (1.0 + g_shift) * np.cos(y2) * np.cos(y1)
        return self._pf_from_arrays([f1, f2, f3], nondominated=False)


class CEC2018DF12(CEC2018DFBase):
    base_name = "df12"
    n_objectives = 3

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[2:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        k_t = 10.0 * math.sin(math.pi * t)
        tmp1 = x[2:] - math.sin(t * x[0])
        holes = np.abs(np.sin(np.floor(k_t * (2.0 * x[:2] - 1.0)) * math.pi / 2.0))
        g = 1.0 + float(np.sum(tmp1 ** 2)) + float(np.prod(holes))
        f1 = g * math.cos(0.5 * math.pi * x[1]) * math.cos(0.5 * math.pi * x[0])
        f2 = g * math.sin(0.5 * math.pi * x[1]) * math.cos(0.5 * math.pi * x[0])
        f3 = g * math.sin(0.5 * math.pi * x[0])
        return np.array([f1, f2, f3], dtype=float)

    def true_pareto(self, n_points: int = 400) -> np.ndarray:
        k = max(12, int(np.ceil(np.sqrt(n_points))))
        x1, x2 = np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k), indexing="xy")
        t = self._cec_time()
        k_t = 10.0 * math.sin(math.pi * t)
        holes = np.abs(
            np.sin(np.floor(k_t * (2.0 * x1 - 1.0)) * math.pi / 2.0)
            * np.sin(np.floor(k_t * (2.0 * x2 - 1.0)) * math.pi / 2.0)
        )
        g = 1.0 + holes
        f1 = g * np.cos(0.5 * math.pi * x2) * np.cos(0.5 * math.pi * x1)
        f2 = g * np.sin(0.5 * math.pi * x2) * np.cos(0.5 * math.pi * x1)
        f3 = g * np.sin(0.5 * math.pi * x1)
        return self._pf_from_arrays([f1, f2, f3], nondominated=True)


class CEC2018DF13(CEC2018DFBase):
    base_name = "df13"
    n_objectives = 3

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[2:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        p_t = math.floor(6.0 * g_shift)
        g = 1.0 + float(np.sum((x[2:] - g_shift) ** 2))
        f1 = g * math.cos(0.5 * math.pi * x[0]) ** 2
        f2 = g * math.cos(0.5 * math.pi * x[1]) ** 2
        f3 = g * (
            math.sin(0.5 * math.pi * x[0]) ** 2
            + math.sin(0.5 * math.pi * x[0]) * math.cos(p_t * math.pi * x[0]) ** 2
            + math.sin(0.5 * math.pi * x[1]) ** 2
            + math.sin(0.5 * math.pi * x[1]) * math.cos(p_t * math.pi * x[1]) ** 2
        )
        return np.array([f1, f2, f3], dtype=float)

    def true_pareto(self, n_points: int = 400) -> np.ndarray:
        k = max(12, int(np.ceil(np.sqrt(n_points))))
        x1, x2 = np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k), indexing="xy")
        t = self._cec_time()
        p_t = math.floor(6.0 * math.sin(0.5 * math.pi * t))
        f1 = np.cos(0.5 * math.pi * x1) ** 2
        f2 = np.cos(0.5 * math.pi * x2) ** 2
        f3 = (
            np.sin(0.5 * math.pi * x1) ** 2
            + np.sin(0.5 * math.pi * x1) * np.cos(p_t * math.pi * x1) ** 2
            + np.sin(0.5 * math.pi * x2) ** 2
            + np.sin(0.5 * math.pi * x2) * np.cos(p_t * math.pi * x2) ** 2
        )
        return self._pf_from_arrays([f1, f2, f3], nondominated=True)


class CEC2018DF14(CEC2018DFBase):
    base_name = "df14"
    n_objectives = 3

    def _native_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_variables, dtype=float)
        ub = np.ones(self.n_variables, dtype=float)
        lb[2:] = -1.0
        return lb, ub

    def _evaluate_native(self, x: np.ndarray) -> np.ndarray:
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        g = 1.0 + float(np.sum((x[2:] - g_shift) ** 2))
        y = 0.5 + g_shift * (x[0] - 0.5)
        f1 = g * (1.0 - y + 0.05 * math.sin(6.0 * math.pi * y))
        y_bump = y + 0.05 * math.sin(6.0 * math.pi * y)
        f2 = g * (1.0 - x[1] + 0.05 * math.sin(6.0 * math.pi * x[1])) * y_bump
        f3 = g * (x[1] + 0.05 * math.sin(6.0 * math.pi * x[1])) * y_bump
        return np.array([f1, f2, f3], dtype=float)

    def true_pareto(self, n_points: int = 400) -> np.ndarray:
        k = max(12, int(np.ceil(np.sqrt(n_points))))
        x1, x2 = np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k), indexing="xy")
        t = self._cec_time()
        g_shift = math.sin(0.5 * math.pi * t)
        y = 0.5 + g_shift * (x1 - 0.5)
        y_bump = y + 0.05 * np.sin(6.0 * math.pi * y)
        f1 = 1.0 - y + 0.05 * np.sin(6.0 * math.pi * y)
        f2 = (1.0 - x2 + 0.05 * np.sin(6.0 * math.pi * x2)) * y_bump
        f3 = (x2 + 0.05 * np.sin(6.0 * math.pi * x2)) * y_bump
        return self._pf_from_arrays([f1, f2, f3], nondominated=False)


CANONICAL_PROBLEM_REGISTRY: Dict[str, object] = {
    DZDT1Step.name: lambda n, h: DZDT1Step(n, h),
    DZDT1Sine.name: lambda n, h: DZDT1Sine(n, h),
    DZDT2Shape.name: lambda n, h: DZDT2Shape(n, h),
    DZDT3Shocks.name: lambda n, h: DZDT3Shocks(n, h),
    FDA1Lite.name: lambda n, h: FDA1Lite(n, h),
    FDA2Lite.name: lambda n, h: FDA2Lite(n, h),
    DConstrainedZDT1.name: lambda n, h: DConstrainedZDT1(n, h),
    DDTLZ2_3Obj.name: lambda n, h: DDTLZ2_3Obj(n, h),
}


CEC2018_DF_CLASSES = [
    CEC2018DF1,
    CEC2018DF2,
    CEC2018DF3,
    CEC2018DF4,
    CEC2018DF5,
    CEC2018DF6,
    CEC2018DF7,
    CEC2018DF8,
    CEC2018DF9,
    CEC2018DF10,
    CEC2018DF11,
    CEC2018DF12,
    CEC2018DF13,
    CEC2018DF14,
]


def _make_cec2018_df_factory(cls, taut: int):
    return lambda n, h, cls=cls, taut=taut: cls(n, h, taut=taut, nt=10)


CEC2018_DF_T10_REGISTRY: Dict[str, object] = {
    f"{cls.base_name}_t10": _make_cec2018_df_factory(cls, 10) for cls in CEC2018_DF_CLASSES
}
CEC2018_DF_T30_REGISTRY: Dict[str, object] = {
    f"{cls.base_name}_t30": _make_cec2018_df_factory(cls, 30) for cls in CEC2018_DF_CLASSES
}
CEC2018_DF_REGISTRY: Dict[str, object] = {
    **CEC2018_DF_T10_REGISTRY,
    **CEC2018_DF_T30_REGISTRY,
}
PROBLEM_REGISTRY: Dict[str, object] = {
    **CANONICAL_PROBLEM_REGISTRY,
    **CEC2018_DF_REGISTRY,
}
CANONICAL_PROBLEM_NAMES = list(CANONICAL_PROBLEM_REGISTRY.keys())
CEC2018_DF_T10_NAMES = list(CEC2018_DF_T10_REGISTRY.keys())
CEC2018_DF_T30_NAMES = list(CEC2018_DF_T30_REGISTRY.keys())
CEC2018_DF_NAMES = CEC2018_DF_T10_NAMES + CEC2018_DF_T30_NAMES
CEC2018_DF_3OBJ_NAMES = [
    name for name in CEC2018_DF_NAMES if any(name.startswith(f"df{i}_") for i in range(10, 15))
]


def non_dominated_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    p = np.asarray(points, dtype=float)
    p = p[np.isfinite(p).all(axis=1)]
    n = p.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(p[j] <= p[i]) and np.any(p[j] < p[i]):
                keep[i] = False
                break
    return p[keep]


def hypervolume_2d(front: np.ndarray, ref: Tuple[float, float]) -> float:
    if front is None or len(front) == 0:
        return 0.0
    points = np.asarray(front, dtype=float)
    points = points[(points[:, 0] <= ref[0]) & (points[:, 1] <= ref[1])]
    if len(points) == 0:
        return 0.0
    nd = non_dominated_points(points)
    nd = nd[np.argsort(nd[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for f1, f2 in nd[::-1]:
        hv += max(prev_x - f1, 0.0) * max(ref[1] - f2, 0.0)
        prev_x = f1
    return float(hv)


def hypervolume_mc(front: np.ndarray, ref: np.ndarray, rng: np.random.Generator, samples: int = 1400) -> float:
    if front is None or len(front) == 0:
        return 0.0
    points = np.asarray(front, dtype=float)
    points = points[np.all(points <= ref, axis=1)]
    if len(points) == 0:
        return 0.0
    random_points = rng.random((samples, ref.shape[0])) * ref[None, :]
    dominated = np.any(np.all(points[:, None, :] <= random_points[None, :, :], axis=2), axis=0)
    return float(np.prod(ref)) * float(np.mean(dominated))


def hypervolume(front: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> float:
    if front is None or len(front) == 0:
        return 0.0
    if ref.shape[0] == 2:
        return hypervolume_2d(front, (ref[0], ref[1]))
    return hypervolume_mc(front, ref, rng)


def igd(front: np.ndarray, true_pf: np.ndarray) -> float:
    if front is None or len(front) == 0 or true_pf is None or len(true_pf) == 0:
        return np.inf
    f = np.asarray(front, dtype=float)
    pf = np.asarray(true_pf, dtype=float)
    distances = np.linalg.norm(pf[:, None, :] - f[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def detection_scores(true_events: List[int], detected_events: List[int], tolerance: int = 5) -> Dict[str, float]:
    if len(true_events) == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "mean_delay": np.nan}
    true_events = sorted(true_events)
    detected_events = sorted(detected_events)
    used = np.zeros(len(detected_events), dtype=bool)
    tp = 0
    delays = []
    for t_true in true_events:
        match_idx = None
        for i, t_det in enumerate(detected_events):
            if used[i]:
                continue
            if t_true <= t_det <= (t_true + tolerance):
                match_idx = i
                break
        if match_idx is not None:
            used[match_idx] = True
            tp += 1
            delays.append(detected_events[match_idx] - t_true)
    fp = len(detected_events) - tp
    fn = len(true_events) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_delay = float(np.mean(delays)) if delays else np.nan
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_delay": mean_delay,
    }


def ref_point_for_problem(
    problem_name: str,
    problem: Optional[DynamicProblemBase] = None,
    true_pf: Optional[np.ndarray] = None,
) -> np.ndarray:
    if (
        problem is not None
        and getattr(problem, "use_dynamic_hv_ref", False)
        and true_pf is not None
        and len(true_pf) > 0
    ):
        pf = np.asarray(true_pf, dtype=float)
        return np.max(pf, axis=0) + 0.5
    if problem_name == DDTLZ2_3Obj.name or problem_name in CEC2018_DF_3OBJ_NAMES:
        return np.array([1.6, 1.6, 1.6], dtype=float)
    return np.array([3.0, 3.0], dtype=float)


def _load_speda_backbone():
    try:
        from test_speda_dynamic_robust_suite import (
            PROBLEM_REGISTRY as speda_problem_registry,
            SpedaRobustConfig,
            _call_react,
            create_speda,
            run_single as run_speda_variant,
            tuned_cfg_for_problem,
        )
        return {
            "PROBLEM_REGISTRY": speda_problem_registry,
            "SpedaRobustConfig": SpedaRobustConfig,
            "_call_react": _call_react,
            "create_speda": create_speda,
            "run_speda_variant": run_speda_variant,
            "tuned_cfg_for_problem": tuned_cfg_for_problem,
        }, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run_speda_variant_with_local_problem(
    problem_name: str,
    cfg,
    seed: int,
    with_reaction: bool,
    create_speda_fn: Callable,
    call_react_fn: Callable,
) -> Dict[str, object]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    try:
        problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        eda = create_speda_fn(cfg)

        eda.generation = np.clip(eda._initialize_generation(), LOWER_BOUND, UPPER_BOUND)
        eda._check_generation(problem)
        eda.evaluations_elite = np.empty((0, eda.evaluations.shape[1]))
        eda._truncation()
        eda._is_multiobjective_dynamic = True
        eda._update_sentinels(force=True)
        if hasattr(eda, "_record_anchor_history"):
            eda._record_anchor_history()

        hv_hist: List[float] = []
        igd_hist: List[float] = []
        detected: List[int] = []
        reaction_quality_reverted_count = 0
        reaction_predictive_samples = 0
        reaction_continuity_samples = 0
        reaction_diversity_samples = 0
        reaction_front_ls_samples = 0
        reaction_adr_weight_count = 0
        reaction_adr_predictive_weight_sum = 0.0
        reaction_adr_continuity_weight_sum = 0.0
        reaction_adr_diversity_weight_sum = 0.0
        reaction_adr_front_ls_weight_sum = 0.0
        inherited_reliability_sum = 0.0
        gate_reject_ratio_sum = 0.0
        switch_cost_sum = 0.0
        front_improvement_sum = 0.0
        required_improvement_sum = 0.0
        switching_gate_rejected_count = 0
        continuity_diverse_count = 0
        reaction_geometry_ready_count = 0
        reaction_geometry_front_shift_sum = 0.0
        reaction_geometry_pos_shift_sum = 0.0
        reaction_geometry_overlap_sum = 0.0
        reaction_geometry_shape_change_sum = 0.0
        reaction_geometry_coherence_sum = 0.0
        reaction_geometry_pos_change_sum = 0.0
        reaction_geometry_pof_change_sum = 0.0
        reaction_geometry_negative_transfer_sum = 0.0
        reaction_geometry_coherent_motion_sum = 0.0
        reaction_geometry_global_change_sum = 0.0
        reaction_geometry_pof_stability_sum = 0.0
        reaction_geometry_tau_sum = 0.0
        reaction_geometry_continuity_risk_sum = 0.0
        reaction_geometry_continuity_removed_sum = 0.0
        reaction_geometry_transfer_predictive_share_sum = 0.0
        true_events = problem.true_change_points(cfg.horizon)

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            if eda._detect_change(problem):
                detected.append(t)
                if with_reaction:
                    quality_reverted = call_react_fn(eda, problem, (LOWER_BOUND, UPPER_BOUND))
                    reaction_quality_reverted_count += int(quality_reverted)
                    reaction_predictive_samples += int(getattr(eda, "_last_reaction_predictive_count", 0))
                    reaction_continuity_samples += int(getattr(eda, "_last_reaction_continuity_count", 0))
                    reaction_diversity_samples += int(getattr(eda, "_last_reaction_diversity_count", 0))
                    reaction_front_ls_samples += int(getattr(eda, "_last_reaction_front_ls_count", 0))
                    reaction_adr_weight_count += 1
                    reaction_adr_predictive_weight_sum += float(getattr(eda, "_last_reaction_adr_predictive_weight", 0.0))
                    reaction_adr_continuity_weight_sum += float(getattr(eda, "_last_reaction_adr_continuity_weight", 0.0))
                    reaction_adr_diversity_weight_sum += float(getattr(eda, "_last_reaction_adr_diversity_weight", 0.0))
                    reaction_adr_front_ls_weight_sum += float(getattr(eda, "_last_reaction_adr_front_ls_weight", 0.0))
                    inherited_reliability_sum += float(getattr(eda, "_last_inherited_state_reliability", 1.0))
                    gate_reject_ratio_sum += float(getattr(eda, "_last_reaction_gate_reject_ratio", 0.0))
                    switch_cost_sum += float(getattr(eda, "_last_reaction_switch_cost", 0.0))
                    front_improvement_sum += float(getattr(eda, "_last_reaction_front_improvement", 0.0))
                    required_improvement_sum += float(getattr(eda, "_last_reaction_required_improvement", 0.0))
                    switching_gate_rejected_count += int(getattr(eda, "_last_reaction_switching_gate_rejected", False))
                    continuity_diverse_count += int(getattr(eda, "_last_reaction_continuity_diverse", False))
                    if float(getattr(eda, "_last_reaction_adr_geometry_ready", 0.0)) >= 0.5:
                        reaction_geometry_ready_count += 1
                        reaction_geometry_front_shift_sum += float(getattr(eda, "_last_reaction_adr_geometry_front_shift", 0.0))
                        reaction_geometry_pos_shift_sum += float(getattr(eda, "_last_reaction_adr_geometry_pos_shift", 0.0))
                        reaction_geometry_overlap_sum += float(getattr(eda, "_last_reaction_adr_geometry_overlap", 0.0))
                        reaction_geometry_shape_change_sum += float(getattr(eda, "_last_reaction_adr_geometry_shape_change", 0.0))
                        reaction_geometry_coherence_sum += float(getattr(eda, "_last_reaction_adr_geometry_coherence", 0.0))
                        reaction_geometry_pos_change_sum += float(getattr(eda, "_last_reaction_adr_geometry_pos_change", 0.0))
                        reaction_geometry_pof_change_sum += float(getattr(eda, "_last_reaction_adr_geometry_pof_change", 0.0))
                        reaction_geometry_negative_transfer_sum += float(getattr(eda, "_last_reaction_adr_geometry_negative_transfer", 0.0))
                        reaction_geometry_coherent_motion_sum += float(getattr(eda, "_last_reaction_adr_geometry_coherent_motion", 0.0))
                        reaction_geometry_global_change_sum += float(getattr(eda, "_last_reaction_adr_geometry_global_change", 0.0))
                        reaction_geometry_pof_stability_sum += float(getattr(eda, "_last_reaction_adr_geometry_pof_stability", 0.0))
                        reaction_geometry_tau_sum += float(getattr(eda, "_last_reaction_adr_geometry_tau", 0.0))
                        reaction_geometry_continuity_risk_sum += float(
                            getattr(eda, "_last_reaction_adr_geometry_continuity_risk", 0.0)
                        )
                        reaction_geometry_continuity_removed_sum += float(
                            getattr(eda, "_last_reaction_adr_geometry_continuity_removed", 0.0)
                        )
                        reaction_geometry_transfer_predictive_share_sum += float(
                            getattr(eda, "_last_reaction_adr_geometry_continuity_transfer_predictive_share", 0.0)
                        )
                    if not quality_reverted:
                        eda.generation = np.clip(eda.generation, LOWER_BOUND, UPPER_BOUND)
                        eda._check_generation(problem)
                        eda._truncation()
                        eda._update_sentinels(force=True)

            eda._truncation()
            front_eval = eda.evaluations_elite if len(eda.evaluations_elite) > 0 else eda.evaluations
            front_obj = non_dominated_points(np.asarray(front_eval[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            if cfg.local_search_enabled and hasattr(eda, "_local_search_over_front"):
                eda._local_search_over_front(problem, (np.zeros(cfg.n_variables), np.ones(cfg.n_variables)))

            eda._update_pm()
            eda._new_generation()
            eda.generation = np.clip(eda.generation, LOWER_BOUND, UPPER_BOUND)
            eda._check_generation(problem)
            eda._update_sentinels()
            if hasattr(eda, "_record_anchor_history"):
                eda._record_anchor_history()

            problem.update_time()

        elapsed = time.perf_counter() - t0
        hv_hist_arr = np.asarray(hv_hist, dtype=float)
        igd_hist_arr = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected, tolerance=cfg.tol_detect)
        weight_den = max(int(reaction_adr_weight_count), 1)
        geom_den = max(int(reaction_geometry_ready_count), 1)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": with_reaction,
            "migd": float(np.mean(igd_hist_arr)),
            "mhv": float(np.mean(hv_hist_arr)),
            "final_igd": float(igd_hist_arr[-1]),
            "final_hv": float(hv_hist_arr[-1]),
            "n_detected": int(len(detected)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(getattr(eda, "env_changes", len(detected))),
            "local_search_calls": int(getattr(eda, "local_search_calls", 0)),
            "memory_hits": int(getattr(eda, "memory_hits", 0)),
            "memory_archive_size": int(len(getattr(eda, "memory_archive", []))),
            "reaction_quality_reverted_count": int(reaction_quality_reverted_count),
            "reaction_predictive_samples": int(reaction_predictive_samples),
            "reaction_continuity_samples": int(reaction_continuity_samples),
            "reaction_diversity_samples": int(reaction_diversity_samples),
            "reaction_front_ls_samples": int(reaction_front_ls_samples),
            "reaction_adr_weight_count": int(reaction_adr_weight_count),
            "reaction_adr_predictive_weight_mean": float(reaction_adr_predictive_weight_sum / weight_den),
            "reaction_adr_continuity_weight_mean": float(reaction_adr_continuity_weight_sum / weight_den),
            "reaction_adr_diversity_weight_mean": float(reaction_adr_diversity_weight_sum / weight_den),
            "reaction_adr_front_ls_weight_mean": float(reaction_adr_front_ls_weight_sum / weight_den),
            "inherited_state_reliability_mean": float(inherited_reliability_sum / weight_den),
            "reaction_gate_reject_ratio_mean": float(gate_reject_ratio_sum / weight_den),
            "reaction_switch_cost_mean": float(switch_cost_sum / weight_den),
            "reaction_front_improvement_mean": float(front_improvement_sum / weight_den),
            "reaction_required_improvement_mean": float(required_improvement_sum / weight_den),
            "reaction_switching_gate_rejected_count": int(switching_gate_rejected_count),
            "reaction_continuity_diverse_count": int(continuity_diverse_count),
            "reaction_geometry_ready_count": int(reaction_geometry_ready_count),
            "reaction_geometry_front_shift_mean": float(reaction_geometry_front_shift_sum / geom_den),
            "reaction_geometry_pos_shift_mean": float(reaction_geometry_pos_shift_sum / geom_den),
            "reaction_geometry_overlap_mean": float(reaction_geometry_overlap_sum / geom_den),
            "reaction_geometry_shape_change_mean": float(reaction_geometry_shape_change_sum / geom_den),
            "reaction_geometry_coherence_mean": float(reaction_geometry_coherence_sum / geom_den),
            "reaction_geometry_pos_change_mean": float(reaction_geometry_pos_change_sum / geom_den),
            "reaction_geometry_pof_change_mean": float(reaction_geometry_pof_change_sum / geom_den),
            "reaction_geometry_negative_transfer_mean": float(reaction_geometry_negative_transfer_sum / geom_den),
            "reaction_geometry_coherent_motion_mean": float(reaction_geometry_coherent_motion_sum / geom_den),
            "reaction_geometry_global_change_mean": float(reaction_geometry_global_change_sum / geom_den),
            "reaction_geometry_pof_stability_mean": float(reaction_geometry_pof_stability_sum / geom_den),
            "reaction_geometry_tau_mean": float(reaction_geometry_tau_sum / geom_den),
            "reaction_geometry_continuity_risk_mean": float(reaction_geometry_continuity_risk_sum / geom_den),
            "reaction_geometry_continuity_removed_mean": float(reaction_geometry_continuity_removed_sum / geom_den),
            "reaction_geometry_transfer_predictive_share_mean": float(
                reaction_geometry_transfer_predictive_share_sum / geom_den
            ),
            "pm_family": str(getattr(eda, "pm_family", cfg.pm_family)),
            "hv_hist": hv_hist_arr,
            "igd_hist": igd_hist_arr,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": with_reaction,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": np.nan,
            "memory_hits": np.nan,
            "memory_archive_size": np.nan,
            "reaction_quality_reverted_count": np.nan,
            "reaction_predictive_samples": np.nan,
            "reaction_continuity_samples": np.nan,
            "reaction_diversity_samples": np.nan,
            "reaction_front_ls_samples": np.nan,
            "reaction_adr_weight_count": np.nan,
            "reaction_adr_predictive_weight_mean": np.nan,
            "reaction_adr_continuity_weight_mean": np.nan,
            "reaction_adr_diversity_weight_mean": np.nan,
            "reaction_adr_front_ls_weight_mean": np.nan,
            "inherited_state_reliability_mean": np.nan,
            "reaction_gate_reject_ratio_mean": np.nan,
            "reaction_switch_cost_mean": np.nan,
            "reaction_front_improvement_mean": np.nan,
            "reaction_required_improvement_mean": np.nan,
            "reaction_switching_gate_rejected_count": np.nan,
            "reaction_continuity_diverse_count": np.nan,
            "reaction_geometry_ready_count": np.nan,
            "reaction_geometry_front_shift_mean": np.nan,
            "reaction_geometry_pos_shift_mean": np.nan,
            "reaction_geometry_overlap_mean": np.nan,
            "reaction_geometry_shape_change_mean": np.nan,
            "reaction_geometry_coherence_mean": np.nan,
            "reaction_geometry_pos_change_mean": np.nan,
            "reaction_geometry_pof_change_mean": np.nan,
            "reaction_geometry_negative_transfer_mean": np.nan,
            "reaction_geometry_coherent_motion_mean": np.nan,
            "reaction_geometry_global_change_mean": np.nan,
            "reaction_geometry_pof_stability_mean": np.nan,
            "reaction_geometry_tau_mean": np.nan,
            "reaction_geometry_continuity_risk_mean": np.nan,
            "reaction_geometry_continuity_removed_mean": np.nan,
            "reaction_geometry_transfer_predictive_share_mean": np.nan,
            "pm_family": cfg.pm_family,
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def _deb_dominates(f_a: np.ndarray, f_b: np.ndarray, eps_feasible: float = 1e-12) -> bool:
    cv_a, cv_b = float(f_a[0]), float(f_b[0])
    obj_a, obj_b = np.asarray(f_a[1:], dtype=float), np.asarray(f_b[1:], dtype=float)
    a_feas = cv_a <= eps_feasible
    b_feas = cv_b <= eps_feasible

    if a_feas and not b_feas:
        return True
    if b_feas and not a_feas:
        return False
    if (not a_feas) and (not b_feas):
        return cv_a < cv_b
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def _fast_non_dominated_sort(f_mat: np.ndarray) -> List[List[int]]:
    n_points = f_mat.shape[0]
    dominates: List[List[int]] = [[] for _ in range(n_points)]
    dominated_count = np.zeros(n_points, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n_points):
        for q in range(n_points):
            if p == q:
                continue
            if _deb_dominates(f_mat[p], f_mat[q]):
                dominates[p].append(q)
            elif _deb_dominates(f_mat[q], f_mat[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    idx = 0
    while idx < len(fronts) and fronts[idx]:
        next_front: List[int] = []
        for p in fronts[idx]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        idx += 1
    return fronts


def _crowding_distance(f_mat: np.ndarray, front: List[int]) -> Dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: np.inf for idx in front}

    obj = np.asarray(f_mat[:, 1:], dtype=float)
    n_obj = obj.shape[1]
    dist = {idx: 0.0 for idx in front}

    for d in range(n_obj):
        sorted_idx = sorted(front, key=lambda i: obj[i, d])
        dist[sorted_idx[0]] = np.inf
        dist[sorted_idx[-1]] = np.inf
        min_v = obj[sorted_idx[0], d]
        max_v = obj[sorted_idx[-1], d]
        denom = max(max_v - min_v, 1e-12)

        for k in range(1, len(sorted_idx) - 1):
            prev_i = sorted_idx[k - 1]
            next_i = sorted_idx[k + 1]
            cur_i = sorted_idx[k]
            dist[cur_i] += (obj[next_i, d] - obj[prev_i, d]) / denom

    return dist


def _rank_and_crowding(f_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ranks = np.full(f_mat.shape[0], np.inf, dtype=float)
    crowd = np.zeros(f_mat.shape[0], dtype=float)
    fronts = _fast_non_dominated_sort(f_mat)
    for rank, front in enumerate(fronts):
        dist = _crowding_distance(f_mat, front)
        for idx in front:
            ranks[idx] = float(rank)
            crowd[idx] = float(dist.get(idx, 0.0))
    return ranks, crowd


def _survivor_indices(f_mat: np.ndarray, n_survivors: int) -> np.ndarray:
    fronts = _fast_non_dominated_sort(f_mat)
    chosen: List[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= n_survivors:
            chosen.extend(front)
            continue
        dist = _crowding_distance(f_mat, front)
        front_sorted = sorted(front, key=lambda i: dist[i], reverse=True)
        needed = max(n_survivors - len(chosen), 0)
        chosen.extend(front_sorted[:needed])
        break
    return np.asarray(chosen, dtype=int)


def _binary_tournament(pop: np.ndarray, f_mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ranks, crowd = _rank_and_crowding(f_mat)
    i, j = rng.integers(0, pop.shape[0], size=2)
    if ranks[i] < ranks[j]:
        return pop[i]
    if ranks[j] < ranks[i]:
        return pop[j]
    if crowd[i] > crowd[j]:
        return pop[i]
    if crowd[j] > crowd[i]:
        return pop[j]
    return pop[i] if rng.random() < 0.5 else pop[j]


def _evaluate_population(problem, pop: np.ndarray) -> np.ndarray:
    rows = []
    for x in np.asarray(pop, dtype=float):
        obj, cv = problem.evaluate(x)
        rows.append(np.concatenate(([float(cv)], np.asarray(obj, dtype=float))))
    return np.asarray(rows, dtype=float)


def _sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    eta: float,
    crossover_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    c1 = np.asarray(p1, dtype=float).copy()
    c2 = np.asarray(p2, dtype=float).copy()
    if rng.random() > crossover_prob:
        return c1, c2

    for i in range(len(c1)):
        if rng.random() > 0.5 or abs(c1[i] - c2[i]) <= 1e-14:
            continue
        x1 = min(c1[i], c2[i])
        x2 = max(c1[i], c2[i])
        rand = rng.random()

        beta = 1.0 + (2.0 * (x1 - LOWER_BOUND) / (x2 - x1))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (UPPER_BOUND - x2) / (x2 - x1))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        child1 = float(np.clip(child1, LOWER_BOUND, UPPER_BOUND))
        child2 = float(np.clip(child2, LOWER_BOUND, UPPER_BOUND))
        if rng.random() < 0.5:
            c1[i], c2[i] = child2, child1
        else:
            c1[i], c2[i] = child1, child2
    return c1, c2


def _poly_mutation(x: np.ndarray, rng: np.random.Generator, eta: float, mutation_prob: float) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    for i in range(len(out)):
        if rng.random() > mutation_prob:
            continue
        delta1 = (out[i] - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND)
        delta2 = (UPPER_BOUND - out[i]) / (UPPER_BOUND - LOWER_BOUND)
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
            deltaq = 1.0 - val ** mut_pow
        out[i] = float(np.clip(out[i] + deltaq * (UPPER_BOUND - LOWER_BOUND), LOWER_BOUND, UPPER_BOUND))
    return out


def _nsga2_offspring(
    pop: np.ndarray,
    f_mat: np.ndarray,
    rng: np.random.Generator,
    crossover_prob: float,
    sbx_eta: float,
    mutation_eta: float,
) -> np.ndarray:
    pop_size, n_variables = pop.shape
    mutation_prob = 1.0 / max(n_variables, 1)
    offspring = []
    while len(offspring) < pop_size:
        p1 = _binary_tournament(pop, f_mat, rng)
        p2 = _binary_tournament(pop, f_mat, rng)
        c1, c2 = _sbx_crossover(p1, p2, rng, eta=sbx_eta, crossover_prob=crossover_prob)
        c1 = _poly_mutation(c1, rng, eta=mutation_eta, mutation_prob=mutation_prob)
        c2 = _poly_mutation(c2, rng, eta=mutation_eta, mutation_prob=mutation_prob)
        offspring.append(c1)
        if len(offspring) < pop_size:
            offspring.append(c2)
    return np.asarray(offspring, dtype=float)


def _nsga2_generation_step(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    offspring = _nsga2_offspring(
        pop=pop,
        f_mat=f_mat,
        rng=rng,
        crossover_prob=cfg.crossover_prob,
        sbx_eta=cfg.sbx_eta,
        mutation_eta=cfg.mutation_eta,
    )
    offspring_eval = _evaluate_population(problem, offspring)
    combined_pop = np.vstack((pop, offspring))
    combined_eval = np.vstack((f_mat, offspring_eval))
    keep = _survivor_indices(combined_eval, n_survivors=pop.shape[0])
    return combined_pop[keep], combined_eval[keep]


def _simplex_lattice_weights(n_weights: int, n_objectives: int) -> np.ndarray:
    n_weights = int(max(n_weights, 1))
    n_objectives = int(max(n_objectives, 1))
    if n_objectives == 1:
        return np.ones((n_weights, 1), dtype=float)
    if n_objectives == 2:
        grid = np.linspace(0.0, 1.0, n_weights)
        return np.column_stack((grid, 1.0 - grid))
    h = 1
    while math.comb(h + n_objectives - 1, n_objectives - 1) < n_weights:
        h += 1
    weights: List[List[float]] = []

    def rec(prefix: List[int], remaining: int, depth: int) -> None:
        if depth == n_objectives - 1:
            weights.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, depth + 1)

    rec([], h, 0)
    arr = np.asarray(weights, dtype=float) / max(float(h), 1.0)
    if len(arr) > n_weights:
        center = np.full(n_objectives, 1.0 / n_objectives, dtype=float)
        order = np.argsort(np.linalg.norm(arr - center[None, :], axis=1))
        extremes = []
        for j in range(n_objectives):
            idx = int(np.argmax(arr[:, j]))
            if idx not in extremes:
                extremes.append(idx)
        selected = extremes + [idx for idx in order if int(idx) not in extremes]
        arr = arr[np.asarray(selected[:n_weights], dtype=int)]
    return np.maximum(arr, 1e-8)


def _moead_scalar_value(eval_row: np.ndarray, weight: np.ndarray, ideal: np.ndarray) -> float:
    cv = max(float(eval_row[0]), 0.0)
    obj = np.asarray(eval_row[1:], dtype=float)
    weight = np.maximum(np.asarray(weight, dtype=float), 1e-8)
    return float(np.max(weight * np.abs(obj - ideal)) + 1e6 * cv)


def _moead_de_step(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    rng: np.random.Generator,
    f: float = 0.5,
    cr: float = 1.0,
    mutation_eta: float = 20.0,
    neighborhood_ratio: float = 0.2,
    max_neighbors: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    pop = np.asarray(pop, dtype=float).copy()
    f_mat = np.asarray(f_mat, dtype=float).copy()
    pop_size, n_variables = pop.shape
    if pop_size <= 1:
        return pop, f_mat
    n_objectives = max(f_mat.shape[1] - 1, 1)
    weights = _simplex_lattice_weights(pop_size, n_objectives)
    if len(weights) != pop_size:
        weights = weights[:pop_size]
    weight_dists = _pairwise_sq_dists(weights, weights)
    n_neighbors = int(max(3, math.ceil(neighborhood_ratio * pop_size)))
    if max_neighbors is not None:
        n_neighbors = min(n_neighbors, int(max_neighbors))
    n_neighbors = int(min(pop_size, n_neighbors))
    neighbors = np.argsort(weight_dists, axis=1)[:, :n_neighbors]
    ideal = np.min(f_mat[:, 1:], axis=0)

    for i in range(pop_size):
        pool = neighbors[i]
        if len(pool) >= 3:
            r1, r2, r3 = rng.choice(pool, size=3, replace=False)
        elif pop_size >= 3:
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
        else:
            r1 = r2 = r3 = int(i)
        mutant = pop[int(r1)] + float(f) * (pop[int(r2)] - pop[int(r3)])
        mask = rng.random(n_variables) < float(np.clip(cr, 0.0, 1.0))
        if not np.any(mask):
            mask[int(rng.integers(0, n_variables))] = True
        child = np.where(mask, mutant, pop[i])
        child = _poly_mutation(
            np.clip(child, LOWER_BOUND, UPPER_BOUND),
            rng=rng,
            eta=mutation_eta,
            mutation_prob=1.0 / max(n_variables, 1),
        )
        child_eval = _evaluate_population(problem, child[None, :])[0]
        ideal = np.minimum(ideal, child_eval[1:])
        for j in pool:
            j = int(j)
            if _moead_scalar_value(child_eval, weights[j], ideal) <= _moead_scalar_value(f_mat[j], weights[j], ideal):
                pop[j] = child
                f_mat[j] = child_eval
    return pop, f_mat


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


class _TCA2018:
    """Transfer Component Analysis as used by Jiang et al. 2018 Tr-IPG."""

    def __init__(self, sigma: float, mu: float, dim: int):
        self.sigma = float(max(sigma, 1e-8))
        self.mu = float(max(mu, 1e-10))
        self.dim = int(max(dim, 1))
        self.w: Optional[np.ndarray] = None
        self.x_train: Optional[np.ndarray] = None

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-_pairwise_sq_dists(x, y) / (2.0 * self.sigma * self.sigma))

    def fit(self, source_obj: np.ndarray, target_obj: np.ndarray) -> "_TCA2018":
        source_obj = np.asarray(source_obj, dtype=float)
        target_obj = np.asarray(target_obj, dtype=float)
        n_source = source_obj.shape[0]
        n_target = target_obj.shape[0]
        self.x_train = np.vstack((source_obj, target_obj))
        n_all = max(n_source + n_target, 1)

        k_mat = self._kernel(self.x_train, self.x_train)
        l_mat = np.zeros((n_all, n_all), dtype=float)
        l_mat[:n_source, :n_source] = 1.0 / max(n_source * n_source, 1)
        l_mat[n_source:, n_source:] = 1.0 / max(n_target * n_target, 1)
        l_mat[:n_source, n_source:] = -1.0 / max(n_source * n_target, 1)
        l_mat[n_source:, :n_source] = -1.0 / max(n_source * n_target, 1)
        h_mat = np.eye(n_all) - np.ones((n_all, n_all), dtype=float) / n_all

        lhs = k_mat @ l_mat @ k_mat + self.mu * np.eye(n_all)
        rhs = k_mat @ h_mat @ k_mat
        eig_mat = np.linalg.pinv(lhs) @ rhs
        evals, evecs = np.linalg.eig(eig_mat)
        order = np.argsort(np.real(evals))[::-1]
        keep = order[: min(self.dim, evecs.shape[1])]
        self.w = np.real(evecs[:, keep])
        return self

    def transform(self, obj: np.ndarray) -> np.ndarray:
        if self.w is None or self.x_train is None:
            raise RuntimeError("TCA must be fitted before transform().")
        k_new = self._kernel(self.x_train, np.asarray(obj, dtype=float))
        return (self.w.T @ k_new).T


def _normalize_triple(
    source_obj: np.ndarray,
    target_obj: np.ndarray,
    front_obj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    all_obj = np.vstack((source_obj, target_obj, front_obj))
    mins = np.min(all_obj, axis=0)
    spans = np.maximum(np.max(all_obj, axis=0) - mins, 1e-12)

    def normalize(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - mins) / spans

    return normalize(source_obj), normalize(target_obj), normalize(front_obj), normalize


def _make_problem_at(problem_name: str, n_variables: int, horizon: int, t: int):
    problem = PROBLEM_REGISTRY[problem_name](n_variables, horizon)
    problem.t = int(t)
    return problem


def _trnsga2_approx_preimage(
    latent_target: np.ndarray,
    target_pool_x: np.ndarray,
    target_pool_latent: np.ndarray,
    target_problem,
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    tca: _TCA2018,
    cfg: TrNSGA2Config,
    rng: np.random.Generator,
) -> np.ndarray:
    distances = np.linalg.norm(target_pool_latent - latent_target[None, :], axis=1)
    n_neighbors = int(max(cfg.preimage_neighbors, 1))
    order = np.argsort(distances)[: min(n_neighbors, len(distances))]
    best_x = np.asarray(target_pool_x[order[0]], dtype=float).copy()
    best_loss = float(distances[order[0]])

    sigma = float(max(cfg.refine_sigma_start, 1e-4))
    n_candidates = int(max(cfg.preimage_refine_candidates, 1))
    for step in range(max(cfg.preimage_refine_steps, 0)):
        noise = rng.normal(0.0, sigma, size=(n_candidates, best_x.shape[0]))
        candidates = np.clip(best_x[None, :] + noise, LOWER_BOUND, UPPER_BOUND)
        cand_eval = _evaluate_population(target_problem, candidates)[:, 1:]
        cand_latent = tca.transform(normalize_fn(cand_eval))
        losses = np.linalg.norm(cand_latent - latent_target[None, :], axis=1)
        best_idx = int(np.argmin(losses))
        if float(losses[best_idx]) < best_loss:
            best_loss = float(losses[best_idx])
            best_x = candidates[best_idx].copy()
        if (step + 1) % 3 == 0:
            sigma *= float(np.clip(cfg.refine_sigma_decay, 0.1, 0.95))

    if cfg.preimage_use_scipy and minimize is not None:
        bounds = [(LOWER_BOUND, UPPER_BOUND)] * int(best_x.shape[0])

        def objective(x: np.ndarray) -> float:
            x = np.clip(np.asarray(x, dtype=float), LOWER_BOUND, UPPER_BOUND)
            obj, _cv = target_problem.evaluate(x)
            latent = tca.transform(normalize_fn(np.asarray(obj, dtype=float)[None, :]))[0]
            diff = latent - latent_target
            return float(np.dot(diff, diff))

        starts = [best_x]
        for idx in order[: min(len(order), 3)]:
            starts.append(np.asarray(target_pool_x[int(idx)], dtype=float))
        for start in starts:
            try:
                res = minimize(
                    objective,
                    np.asarray(start, dtype=float),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": int(max(cfg.preimage_optimizer_maxiter, 1)), "ftol": 1e-10},
                )
            except Exception:
                continue
            if not getattr(res, "success", False) and not np.isfinite(getattr(res, "fun", np.inf)):
                continue
            candidate_x = np.clip(np.asarray(res.x, dtype=float), LOWER_BOUND, UPPER_BOUND)
            candidate_loss = math.sqrt(max(float(objective(candidate_x)), 0.0))
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_x = candidate_x

    return np.asarray(best_x, dtype=float)


def _trnsga2_generate_transfer_population(
    prev_front_obj: np.ndarray,
    source_problem,
    target_problem,
    cfg: TrNSGA2Config,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    prev_front_obj = np.asarray(prev_front_obj, dtype=float)
    if prev_front_obj.size == 0 or len(prev_front_obj) == 0:
        return np.empty((0, cfg.n_variables), dtype=float), {
            "success_rate": 0.0,
            "latent_shift": float("nan"),
            "raw_shift": float("nan"),
        }

    sample_count = int(max(cfg.transfer_samples, cfg.pop_size))
    source_dec = rng.random((sample_count, cfg.n_variables))
    target_dec = rng.random((sample_count, cfg.n_variables))
    source_obj = _evaluate_population(source_problem, source_dec)[:, 1:]
    target_obj = _evaluate_population(target_problem, target_dec)[:, 1:]

    source_norm, target_norm, front_norm, normalize_fn = _normalize_triple(source_obj, target_obj, prev_front_obj)
    tca = _TCA2018(sigma=cfg.tca_sigma, mu=cfg.tca_mu, dim=cfg.tca_dim).fit(source_norm, target_norm)
    source_latent = tca.transform(source_norm)
    target_latent = tca.transform(target_norm)
    front_latent = tca.transform(front_norm)

    predicted = [
        _trnsga2_approx_preimage(
            latent_target=latent_point,
            target_pool_x=target_dec,
            target_pool_latent=target_latent,
            target_problem=target_problem,
            normalize_fn=normalize_fn,
            tca=tca,
            cfg=cfg,
            rng=rng,
        )
        for latent_point in front_latent
    ]
    predicted_pop = np.asarray(predicted, dtype=float) if predicted else np.empty((0, cfg.n_variables), dtype=float)
    raw_shift = float(np.linalg.norm(np.mean(source_norm, axis=0) - np.mean(target_norm, axis=0)))
    latent_shift = float(np.linalg.norm(np.mean(source_latent, axis=0) - np.mean(target_latent, axis=0)))
    return predicted_pop, {
        "success_rate": float(len(predicted_pop) / max(len(prev_front_obj), 1)),
        "latent_shift": latent_shift,
        "raw_shift": raw_shift,
    }


def _trnsga2_initial_population_from_transfer(
    predicted_pop: np.ndarray,
    target_problem,
    cfg: TrNSGA2Config,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    predicted_pop = np.asarray(predicted_pop, dtype=float)
    if predicted_pop.size == 0 or len(predicted_pop) == 0:
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        return pop, _evaluate_population(target_problem, pop)

    if len(predicted_pop) >= cfg.pop_size:
        pred_eval = _evaluate_population(target_problem, predicted_pop)
        keep = _survivor_indices(pred_eval, n_survivors=cfg.pop_size)
        pop = predicted_pop[keep]
        return pop, pred_eval[keep]

    random_block = rng.random((cfg.pop_size - len(predicted_pop), cfg.n_variables))
    pop = np.vstack((predicted_pop, random_block))
    return pop, _evaluate_population(target_problem, pop)


def _trnsga2_should_transfer(t: int, true_events: List[int], cfg: TrNSGA2Config) -> bool:
    if cfg.detection_mode == "every_step":
        return t > 0
    if true_events:
        return t in set(true_events)
    return bool(cfg.transfer_when_no_explicit_events and t > 0)


def run_single_trnsga2_2018(problem_name: str, cfg: TrNSGA2Config, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        template_problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        true_events = list(template_problem.true_change_points(cfg.horizon))
        ref = ref_point_for_problem(problem_name)
        nsga_cfg = GAConfig(
            pop_size=cfg.pop_size,
            n_variables=cfg.n_variables,
            horizon=cfg.horizon,
            generations_per_step=cfg.generations_per_step,
            crossover_prob=cfg.crossover_prob,
            sbx_eta=cfg.sbx_eta,
            mutation_eta=cfg.mutation_eta,
        )

        problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, 0)
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        f_mat = _evaluate_population(problem, pop)
        last_front_obj = np.empty((0, ref.shape[0]), dtype=float)
        detected_events: List[int] = []
        transfer_hits = 0
        preimage_calls = 0
        prev_eval = f_mat.copy()
        adr_meta_updates = 0
        adr_meta_reuses = 0
        adr_meta_drifts: List[float] = []
        adr_meta_reasons: Dict[str, int] = {}
        last_adr_meta_update_t: Optional[int] = None

        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, t)
            if t > 0:
                f_mat = _evaluate_population(problem, pop)
                if _trnsga2_should_transfer(t, true_events, cfg):
                    should_update, drift, reason = _adr_meta_should_update(
                        cfg=cfg,
                        t=t,
                        previous_eval=prev_eval,
                        current_eval=f_mat,
                        last_update_t=last_adr_meta_update_t,
                    )
                    adr_meta_drifts.append(float(drift))
                    adr_meta_reasons[reason] = adr_meta_reasons.get(reason, 0) + 1
                    detected_events.append(t)
                    if should_update:
                        adr_meta_updates += int(bool(getattr(cfg, "adr_meta_enabled", False)))
                        last_adr_meta_update_t = t
                        source_problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, t - 1)
                        predicted_pop, _tr_info = _trnsga2_generate_transfer_population(
                            prev_front_obj=last_front_obj,
                            source_problem=source_problem,
                            target_problem=problem,
                            cfg=cfg,
                            rng=rng,
                        )
                        preimage_calls += int(len(predicted_pop))
                        pop, f_mat = _trnsga2_initial_population_from_transfer(
                            predicted_pop=predicted_pop,
                            target_problem=problem,
                            cfg=cfg,
                            rng=rng,
                        )
                        transfer_hits += int(len(predicted_pop) > 0)
                    else:
                        adr_meta_reuses += 1

            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            for _ in range(max(cfg.generations_per_step, 1)):
                pop, f_mat = _nsga2_generation_step(pop=pop, f_mat=f_mat, problem=problem, cfg=nsga_cfg, rng=rng)

            last_front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            prev_eval = f_mat.copy()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": int(preimage_calls),
            "memory_hits": int(transfer_hits),
            "memory_archive_size": int(len(last_front_obj)),
            "adr_meta_enabled": int(bool(getattr(cfg, "adr_meta_enabled", False))),
            "adr_meta_updates": int(adr_meta_updates),
            "adr_meta_reuses": int(adr_meta_reuses),
            "adr_meta_drift_mean": float(np.mean(adr_meta_drifts)) if adr_meta_drifts else np.nan,
            "adr_meta_reuse_ratio": float(adr_meta_reuses / max(len(detected_events), 1)),
            "adr_meta_reason_summary": json.dumps(adr_meta_reasons, sort_keys=True),
            "pm_family": "Tr-NSGA-II-2018",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "Tr-NSGA-II-2018",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def _front0_indices(f_mat: np.ndarray) -> List[int]:
    fronts = _fast_non_dominated_sort(f_mat)
    return list(fronts[0]) if fronts else []


def _pfpa_nondominated_decisions(pop: np.ndarray, f_mat: np.ndarray, max_keep: int) -> np.ndarray:
    idx = _front0_indices(f_mat)
    if not idx:
        keep = _survivor_indices(f_mat, n_survivors=min(max_keep, pop.shape[0]))
        return np.asarray(pop[keep], dtype=float)
    if len(idx) <= max_keep:
        return np.asarray(pop[idx], dtype=float)
    local_eval = np.asarray(f_mat[idx], dtype=float)
    keep_local = _survivor_indices(local_eval, n_survivors=max_keep)
    keep = np.asarray(idx, dtype=int)[keep_local]
    return np.asarray(pop[keep], dtype=float)


def _pfpa_principal_variable(pop: np.ndarray) -> int:
    pop = np.asarray(pop, dtype=float)
    if pop.ndim != 2 or pop.shape[1] <= 1:
        return 0
    std = np.std(pop, axis=0)
    if pop.shape[0] < 3:
        return int(np.argmax(std))
    corr = np.corrcoef(pop, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    score = np.mean(corr, axis=1) + 0.05 * std
    return int(np.argmax(score))


def _pfpa_multi_step_population(
    prev_pop: np.ndarray,
    last_pop: np.ndarray,
    last_nd_dec: np.ndarray,
    cfg: PFPAConfig,
) -> np.ndarray:
    if last_nd_dec.size == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    center_prev = np.mean(np.asarray(prev_pop, dtype=float), axis=0)
    center_last = np.mean(np.asarray(last_pop, dtype=float), axis=0)
    direction = center_last - center_prev
    if np.linalg.norm(direction) <= 1e-12:
        direction = np.zeros(cfg.n_variables, dtype=float)
    blocks = []
    for step in cfg.movement_steps:
        blocks.append(np.clip(last_nd_dec + float(step) * direction[None, :], LOWER_BOUND, UPPER_BOUND))
    return np.vstack(blocks) if blocks else np.empty((0, cfg.n_variables), dtype=float)


def _pfpa_polyfit_coeffs(x: np.ndarray, y: np.ndarray, max_degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    max_degree = int(max(max_degree, 0))
    if len(x) == 0 or len(y) == 0:
        return np.zeros(max_degree + 1, dtype=float)
    unique_count = len(np.unique(np.round(x, 12)))
    degree = int(min(max_degree, max(unique_count - 1, 0), max(len(x) - 1, 0)))
    if degree <= 0:
        coeff = np.array([float(np.mean(y))], dtype=float)
    else:
        coeff = np.polyfit(x, y, degree)
    padded = np.zeros(max_degree + 1, dtype=float)
    padded[-len(coeff) :] = coeff
    return padded


def _pfpa_polynomial_population(
    prev_pop: np.ndarray,
    last_pop: np.ndarray,
    cfg: PFPAConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    prev_pop = np.asarray(prev_pop, dtype=float)
    last_pop = np.asarray(last_pop, dtype=float)
    if prev_pop.size == 0 or last_pop.size == 0 or n_samples <= 0:
        return np.empty((0, cfg.n_variables), dtype=float)

    principal = _pfpa_principal_variable(last_pop)
    max_degree = int(max(0, cfg.polynomial_max_degree))
    x_prev = prev_pop[:, principal]
    x_last = last_pop[:, principal]
    lo = float(np.clip(np.min(x_last), LOWER_BOUND, UPPER_BOUND))
    hi = float(np.clip(np.max(x_last), LOWER_BOUND, UPPER_BOUND))
    if hi - lo <= 1e-8:
        lo, hi = LOWER_BOUND, UPPER_BOUND

    samples = np.zeros((n_samples, cfg.n_variables), dtype=float)
    samples[:, principal] = rng.uniform(lo, hi, size=n_samples)
    for j in range(cfg.n_variables):
        if j == principal:
            continue
        coeff_prev = _pfpa_polyfit_coeffs(x_prev, prev_pop[:, j], max_degree)
        coeff_last = _pfpa_polyfit_coeffs(x_last, last_pop[:, j], max_degree)
        coeff_pred = coeff_last + (coeff_last - coeff_prev)
        curve = np.polyval(coeff_pred, samples[:, principal])
        noise_scale = float(cfg.compression_ratio) * max(float(np.std(last_pop[:, j])), 0.02)
        samples[:, j] = curve + rng.normal(0.0, noise_scale, size=n_samples)

    return np.clip(samples, LOWER_BOUND, UPPER_BOUND)


def _pfpa_sampling_population(
    last_pop: np.ndarray,
    target_problem,
    cfg: PFPAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    last_pop = np.asarray(last_pop, dtype=float)
    if last_pop.size == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    principal = _pfpa_principal_variable(last_pop)
    non_principal = [j for j in range(cfg.n_variables) if j != principal]
    if not non_principal:
        return rng.random((max(1, cfg.pop_size // 2), cfg.n_variables))

    n2 = int(max(cfg.n_variables * max(cfg.sampling_n2_factor, 1), 1))
    n1 = int(max(1, (cfg.pop_size // 2) // max(n2, 1)))
    center = np.mean(last_pop, axis=0)
    principal_values = np.linspace(LOWER_BOUND, UPPER_BOUND, n1)
    grid_values = np.linspace(LOWER_BOUND, UPPER_BOUND, n2 + 2)[1:-1]

    candidates = []
    for x_principal in principal_values:
        for j, value in enumerate(grid_values):
            x = center.copy()
            x[principal] = x_principal
            x[non_principal[j % len(non_principal)]] = value
            jitter = rng.normal(0.0, 0.015, size=cfg.n_variables)
            jitter[principal] = 0.0
            candidates.append(np.clip(x + jitter, LOWER_BOUND, UPPER_BOUND))
    if not candidates:
        return np.empty((0, cfg.n_variables), dtype=float)

    cand_pop = np.asarray(candidates, dtype=float)
    cand_eval = _evaluate_population(target_problem, cand_pop)
    keep = _survivor_indices(cand_eval, n_survivors=min(max(cfg.pop_size // 2, 1), len(cand_pop)))
    return cand_pop[keep]


def _pfpa_merge_select(
    candidates: List[np.ndarray],
    target_problem,
    cfg: PFPAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    blocks = [np.asarray(block, dtype=float) for block in candidates if np.asarray(block).size > 0]
    if blocks:
        pop = np.vstack(blocks)
    else:
        pop = np.empty((0, cfg.n_variables), dtype=float)
    if len(pop) < cfg.pop_size:
        random_block = rng.random((cfg.pop_size - len(pop), cfg.n_variables))
        pop = np.vstack((pop, random_block)) if len(pop) else random_block
    pop = np.clip(pop, LOWER_BOUND, UPPER_BOUND)
    f_mat = _evaluate_population(target_problem, pop)
    keep = _survivor_indices(f_mat, n_survivors=cfg.pop_size)
    return pop[keep], f_mat[keep]


def _pfpa_change_response(
    prev_pop: Optional[np.ndarray],
    last_pop: Optional[np.ndarray],
    last_eval: Optional[np.ndarray],
    target_problem,
    cfg: PFPAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if prev_pop is None or last_pop is None or last_eval is None:
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        return pop, _evaluate_population(target_problem, pop), {"subpop1": 0, "subpop2": 0, "subpop3": 0}

    last_nd_dec = _pfpa_nondominated_decisions(last_pop, last_eval, max_keep=cfg.pop_size)
    subpop1 = _pfpa_multi_step_population(prev_pop, last_pop, last_nd_dec, cfg)
    subpop2 = _pfpa_polynomial_population(prev_pop, last_pop, cfg, rng, n_samples=cfg.pop_size)
    subpop3 = _pfpa_sampling_population(last_pop, target_problem, cfg, rng)
    pop, f_mat = _pfpa_merge_select([subpop1, subpop2, subpop3], target_problem, cfg, rng)
    return pop, f_mat, {"subpop1": len(subpop1), "subpop2": len(subpop2), "subpop3": len(subpop3)}


def _pfpa_kmeans_labels(pop: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    pop = np.asarray(pop, dtype=float)
    n_points = pop.shape[0]
    n_clusters = int(min(max(n_clusters, 1), n_points))
    init_idx = rng.choice(n_points, size=n_clusters, replace=False)
    centers = pop[init_idx].copy()
    labels = np.zeros(n_points, dtype=int)
    for _ in range(6):
        dists = _pairwise_sq_dists(pop, centers)
        labels = np.argmin(dists, axis=1)
        for k in range(n_clusters):
            members = pop[labels == k]
            if len(members) > 0:
                centers[k] = np.mean(members, axis=0)
            else:
                centers[k] = pop[int(rng.integers(0, n_points))]
    return labels


def _pfpa_rmeda_offspring(
    pop: np.ndarray,
    f_mat: np.ndarray,
    cfg: PFPAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    pop_size, n_variables = pop.shape
    model_idx = _survivor_indices(f_mat, n_survivors=min(pop_size, max(4, pop_size)))
    model_pop = np.asarray(pop[model_idx], dtype=float)
    model_eval = np.asarray(f_mat[model_idx], dtype=float)
    nd_idx = _front0_indices(model_eval)
    if len(nd_idx) >= max(4, pop_size // 4):
        model_pop = model_pop[np.asarray(nd_idx, dtype=int)]
    n_clusters = cfg.rmeda_clusters if cfg.rmeda_clusters > 0 else int(max(2, round(np.sqrt(max(len(model_pop), 1)))))
    labels = _pfpa_kmeans_labels(model_pop, n_clusters, rng)
    objective_count = max(f_mat.shape[1] - 1, 1)
    latent_dim = int(max(1, min(n_variables, max(objective_count - 1, 1))))
    offspring = []

    cluster_ids = np.unique(labels)
    cluster_sizes = np.asarray([np.sum(labels == k) for k in cluster_ids], dtype=float)
    probs = cluster_sizes / max(float(np.sum(cluster_sizes)), 1.0)
    while len(offspring) < pop_size:
        cluster = int(rng.choice(cluster_ids, p=probs))
        members = model_pop[labels == cluster]
        if len(members) < 2:
            base = model_pop[int(rng.integers(0, len(model_pop)))]
            child = _poly_mutation(base, rng, eta=20.0, mutation_prob=1.0 / max(n_variables, 1))
            offspring.append(child)
            continue
        mean = np.mean(members, axis=0)
        centered = members - mean
        try:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
            axes = vt[:latent_dim]
        except np.linalg.LinAlgError:
            axes = np.eye(n_variables)[:latent_dim]
        if axes.ndim == 1:
            axes = axes[None, :]
        projection = centered @ axes.T
        reconstructed = projection @ axes
        residual = centered - reconstructed
        proj_min = np.min(projection, axis=0)
        proj_max = np.max(projection, axis=0)
        if np.any((proj_max - proj_min) <= 1e-10):
            proj_std = np.maximum(np.std(projection, axis=0), cfg.rmeda_latent_scale)
            coeff = rng.normal(0.0, proj_std)
        else:
            coeff = rng.uniform(proj_min, proj_max)
        child = mean + coeff @ axes
        residual_sigma = np.maximum(np.std(residual, axis=0), cfg.rmeda_model_noise)
        child += rng.normal(0.0, residual_sigma, size=n_variables)
        offspring.append(np.clip(child, LOWER_BOUND, UPPER_BOUND))

    return np.asarray(offspring, dtype=float)


def _pfpa_rmeda_step(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: PFPAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    offspring = _pfpa_rmeda_offspring(pop, f_mat, cfg, rng)
    offspring_eval = _evaluate_population(problem, offspring)
    combined_pop = np.vstack((pop, offspring))
    combined_eval = np.vstack((f_mat, offspring_eval))
    keep = _survivor_indices(combined_eval, n_survivors=cfg.pop_size)
    return combined_pop[keep], combined_eval[keep]


def run_single_pfpa_2022(problem_name: str, cfg: PFPAConfig, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        template_problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        true_events = list(template_problem.true_change_points(cfg.horizon))
        ref = ref_point_for_problem(problem_name)

        problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, 0)
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        f_mat = _evaluate_population(problem, pop)
        prev_pop: Optional[np.ndarray] = None
        prev_eval: Optional[np.ndarray] = None
        prev_prev_pop: Optional[np.ndarray] = None
        detected_events: List[int] = []
        response_counts = {"subpop1": 0, "subpop2": 0, "subpop3": 0}
        adr_meta_updates = 0
        adr_meta_reuses = 0
        adr_meta_drifts: List[float] = []
        adr_meta_reasons: Dict[str, int] = {}
        last_adr_meta_update_t: Optional[int] = None

        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, t)
            if t > 0:
                f_mat = _evaluate_population(problem, pop)
                changed = t in set(true_events) if cfg.detection_mode == "oracle" else False
                if changed:
                    previous_for_gate = prev_eval if prev_eval is not None else f_mat
                    should_update, drift, reason = _adr_meta_should_update(
                        cfg=cfg,
                        t=t,
                        previous_eval=previous_for_gate,
                        current_eval=f_mat,
                        last_update_t=last_adr_meta_update_t,
                    )
                    adr_meta_drifts.append(float(drift))
                    adr_meta_reasons[reason] = adr_meta_reasons.get(reason, 0) + 1
                    detected_events.append(t)
                    if should_update:
                        adr_meta_updates += int(bool(getattr(cfg, "adr_meta_enabled", False)))
                        last_adr_meta_update_t = t
                        pop, f_mat, counts = _pfpa_change_response(
                            prev_pop=prev_prev_pop,
                            last_pop=prev_pop,
                            last_eval=prev_eval,
                            target_problem=problem,
                            cfg=cfg,
                            rng=rng,
                        )
                        for key, value in counts.items():
                            response_counts[key] += int(value)
                    else:
                        adr_meta_reuses += 1

            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            for _ in range(max(cfg.generations_per_step, 1)):
                pop, f_mat = _pfpa_rmeda_step(pop=pop, f_mat=f_mat, problem=problem, cfg=cfg, rng=rng)

            prev_prev_pop = None if prev_pop is None else prev_pop.copy()
            prev_pop = pop.copy()
            prev_eval = f_mat.copy()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": 0,
            "memory_hits": int(len(detected_events)),
            "memory_archive_size": int(sum(response_counts.values())),
            "adr_meta_enabled": int(bool(getattr(cfg, "adr_meta_enabled", False))),
            "adr_meta_updates": int(adr_meta_updates),
            "adr_meta_reuses": int(adr_meta_reuses),
            "adr_meta_drift_mean": float(np.mean(adr_meta_drifts)) if adr_meta_drifts else np.nan,
            "adr_meta_reuse_ratio": float(adr_meta_reuses / max(len(detected_events), 1)),
            "adr_meta_reason_summary": json.dumps(adr_meta_reasons, sort_keys=True),
            "pm_family": "PFPA-2022",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "PFPA-2022",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def _tda_nondominated_snapshot(
    pop: np.ndarray,
    f_mat: np.ndarray,
    max_keep: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = _front0_indices(f_mat)
    if not idx:
        keep = _survivor_indices(f_mat, n_survivors=min(max_keep, pop.shape[0]))
    elif len(idx) <= max_keep:
        keep = np.asarray(idx, dtype=int)
    else:
        local_eval = np.asarray(f_mat[idx], dtype=float)
        keep_local = _survivor_indices(local_eval, n_survivors=max_keep)
        keep = np.asarray(idx, dtype=int)[keep_local]
    return np.asarray(pop[keep], dtype=float), np.asarray(f_mat[keep, 1:], dtype=float)


def _tda_wilcoxon_pvalue(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.abs(diff) > 1e-12]
    n = int(len(diff))
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(diff))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    r_pos = float(np.sum(ranks[diff > 0.0]))
    r_neg = float(np.sum(ranks[diff < 0.0]))
    t_stat = min(r_pos, r_neg)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 1e-12:
        return 1.0
    z = abs((t_stat - mean) / math.sqrt(var))
    return float(math.erfc(z / math.sqrt(2.0)))


def _tda_detect_type(
    prev_nd_dec: np.ndarray,
    last_nd_dec: np.ndarray,
    prev_nd_obj: np.ndarray,
    last_nd_obj: np.ndarray,
    ref: np.ndarray,
    cfg: TDADMOEAConfig,
    rng: np.random.Generator,
) -> Tuple[str, Dict[str, float]]:
    if len(prev_nd_dec) == 0 or len(last_nd_dec) == 0 or len(prev_nd_obj) == 0 or len(last_nd_obj) == 0:
        return "none", {"pos_p": 1.0, "hv_ratio": 0.0, "pos_flag": 0.0, "pof_flag": 0.0}

    pos_diff = np.mean(last_nd_dec, axis=0) - np.mean(prev_nd_dec, axis=0)
    pos_p = _tda_wilcoxon_pvalue(pos_diff)
    pos_flag = bool(pos_p < cfg.pos_p_threshold)

    hv_prev = hypervolume(non_dominated_points(prev_nd_obj), ref=ref, rng=rng)
    hv_last = hypervolume(non_dominated_points(last_nd_obj), ref=ref, rng=rng)
    hv_ratio = abs(hv_last - hv_prev) / max(abs(hv_last), 1e-12)
    threshold = cfg.pof_threshold_3obj if last_nd_obj.shape[1] == 3 else cfg.pof_threshold_2obj
    pof_flag = bool(hv_ratio <= threshold)

    if pos_flag and pof_flag:
        change_type = "II"
    elif pos_flag:
        change_type = "I"
    elif pof_flag:
        change_type = "III"
    else:
        change_type = "none"
    return change_type, {
        "pos_p": float(pos_p),
        "hv_ratio": float(hv_ratio),
        "pos_flag": float(pos_flag),
        "pof_flag": float(pof_flag),
    }


def _nearest_rows(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) == 0 or len(target) == 0:
        return np.empty((0, source.shape[1] if source.ndim == 2 else 0), dtype=float)
    dists = _pairwise_sq_dists(target, source)
    idx = np.argmin(dists, axis=1)
    return source[idx]


def _tda_lps_population(
    prev_nd_dec: np.ndarray,
    last_nd_dec: np.ndarray,
    cfg: TDADMOEAConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    if len(prev_nd_dec) == 0 or len(last_nd_dec) == 0 or n_samples <= 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    base_idx = rng.integers(0, len(last_nd_dec), size=n_samples)
    x_t = np.asarray(last_nd_dec[base_idx], dtype=float)
    x_parent = _nearest_rows(prev_nd_dec, x_t)
    delta = x_t - x_parent
    sigma = np.sqrt(np.sum(delta * delta, axis=1) / max(4.0 * cfg.n_variables, 1.0))
    noise = rng.normal(0.0, sigma[:, None], size=x_t.shape)
    return np.clip(x_t + delta + noise, LOWER_BOUND, UPPER_BOUND)


def _tda_knee_indices(obj: np.ndarray, n_keep: int) -> np.ndarray:
    obj = np.asarray(obj, dtype=float)
    if len(obj) <= n_keep:
        return np.arange(len(obj), dtype=int)
    mins = np.min(obj, axis=0)
    spans = np.maximum(np.max(obj, axis=0) - mins, 1e-12)
    norm = (obj - mins) / spans
    if norm.shape[1] == 2:
        left = norm[int(np.argmin(norm[:, 0]))]
        right = norm[int(np.argmin(norm[:, 1]))]
        line = right - left
        denom = max(float(np.linalg.norm(line)), 1e-12)
        scores = np.abs(np.cross(line, norm - left)) / denom
    else:
        centered = norm - np.mean(norm, axis=0)
        try:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
            axis = vt[0]
            projection = centered @ axis[:, None] @ axis[None, :]
            scores = np.linalg.norm(centered - projection, axis=1)
        except np.linalg.LinAlgError:
            scores = np.linalg.norm(centered, axis=1)
    return np.argsort(scores)[-n_keep:]


def _tda_kpp_population(
    last_nd_dec: np.ndarray,
    last_nd_obj: np.ndarray,
    cfg: TDADMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(last_nd_dec) == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    n_knees = int(max(1, round(len(last_nd_dec) * cfg.knee_ratio)))
    knee_idx = _tda_knee_indices(last_nd_obj, n_keep=n_knees)
    knees = last_nd_dec[knee_idx]
    selected = knees[rng.integers(0, len(knees), size=cfg.pop_size)]
    noise = rng.normal(0.0, cfg.ma_tl_noise, size=selected.shape)
    return np.clip(selected + noise, LOWER_BOUND, UPPER_BOUND)


def _tda_objective_preimage_population(
    pred_obj: np.ndarray,
    target_problem,
    cfg: TDADMOEAConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    if len(pred_obj) == 0 or n_samples <= 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    pool_size = int(max(cfg.transfer_pool_samples, cfg.pop_size))
    pool = rng.random((pool_size, cfg.n_variables))
    pool_obj = _evaluate_population(target_problem, pool)[:, 1:]
    idx = rng.integers(0, len(pred_obj), size=n_samples)
    targets = pred_obj[idx]
    dists = _pairwise_sq_dists(targets, pool_obj)
    nearest = pool[np.argmin(dists, axis=1)]
    return np.clip(nearest + rng.normal(0.0, cfg.ma_tl_noise, size=nearest.shape), LOWER_BOUND, UPPER_BOUND)


def _tda_ma_tl_population(
    prev_nd_dec: np.ndarray,
    last_nd_dec: np.ndarray,
    prev_nd_obj: np.ndarray,
    last_nd_obj: np.ndarray,
    target_problem,
    cfg: TDADMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    n_decision = cfg.pop_size // 2
    decision_block = _tda_lps_population(prev_nd_dec, last_nd_dec, cfg, rng, n_samples=n_decision)
    aligned_prev_obj = _nearest_rows(prev_nd_obj, last_nd_obj)
    pred_obj = last_nd_obj + (last_nd_obj - aligned_prev_obj)
    objective_block = _tda_objective_preimage_population(
        pred_obj=pred_obj,
        target_problem=target_problem,
        cfg=cfg,
        rng=rng,
        n_samples=cfg.pop_size - len(decision_block),
    )
    blocks = [b for b in (decision_block, objective_block) if len(b) > 0]
    return np.vstack(blocks) if blocks else np.empty((0, cfg.n_variables), dtype=float)


def _tda_response_population(
    change_type: str,
    prev_nd_dec: np.ndarray,
    last_nd_dec: np.ndarray,
    prev_nd_obj: np.ndarray,
    last_nd_obj: np.ndarray,
    target_problem,
    cfg: TDADMOEAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if change_type == "I":
        candidates = _tda_lps_population(prev_nd_dec, last_nd_dec, cfg, rng, n_samples=cfg.pop_size)
    elif change_type == "II":
        candidates = _tda_ma_tl_population(prev_nd_dec, last_nd_dec, prev_nd_obj, last_nd_obj, target_problem, cfg, rng)
    elif change_type == "III":
        candidates = _tda_kpp_population(last_nd_dec, last_nd_obj, cfg, rng)
    else:
        candidates = np.asarray(last_nd_dec, dtype=float)

    if len(candidates) < cfg.pop_size:
        random_block = rng.random((cfg.pop_size - len(candidates), cfg.n_variables))
        candidates = np.vstack((candidates, random_block)) if len(candidates) else random_block
    candidates = np.clip(candidates, LOWER_BOUND, UPPER_BOUND)
    f_mat = _evaluate_population(target_problem, candidates)
    keep = _survivor_indices(f_mat, n_survivors=cfg.pop_size)
    return candidates[keep], f_mat[keep]


def run_single_tda_dmoea_2024(problem_name: str, cfg: TDADMOEAConfig, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        template_problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        true_events = list(template_problem.true_change_points(cfg.horizon))
        ref = ref_point_for_problem(problem_name)
        nsga_cfg = GAConfig(
            pop_size=cfg.pop_size,
            n_variables=cfg.n_variables,
            horizon=cfg.horizon,
            generations_per_step=cfg.generations_per_step,
            crossover_prob=cfg.crossover_prob,
            sbx_eta=cfg.sbx_eta,
            mutation_eta=cfg.mutation_eta,
        )

        problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, 0)
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        f_mat = _evaluate_population(problem, pop)
        prev_nd_dec: Optional[np.ndarray] = None
        prev_nd_obj: Optional[np.ndarray] = None
        last_nd_dec: Optional[np.ndarray] = None
        last_nd_obj: Optional[np.ndarray] = None
        detected_events: List[int] = []
        type_counts = {"I": 0, "II": 0, "III": 0, "none": 0}
        prev_eval_for_gate = f_mat.copy()
        adr_meta_updates = 0
        adr_meta_reuses = 0
        adr_meta_drifts: List[float] = []
        adr_meta_reasons: Dict[str, int] = {}
        last_adr_meta_update_t: Optional[int] = None

        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, t)
            if t > 0:
                f_mat = _evaluate_population(problem, pop)
                changed = t in set(true_events) if cfg.detection_mode == "oracle" else False
                if changed:
                    should_update, drift, reason = _adr_meta_should_update(
                        cfg=cfg,
                        t=t,
                        previous_eval=prev_eval_for_gate,
                        current_eval=f_mat,
                        last_update_t=last_adr_meta_update_t,
                    )
                    adr_meta_drifts.append(float(drift))
                    adr_meta_reasons[reason] = adr_meta_reasons.get(reason, 0) + 1
                    detected_events.append(t)
                    if should_update:
                        adr_meta_updates += int(bool(getattr(cfg, "adr_meta_enabled", False)))
                        last_adr_meta_update_t = t
                        if prev_nd_dec is not None and last_nd_dec is not None and prev_nd_obj is not None and last_nd_obj is not None:
                            change_type, _info = _tda_detect_type(
                                prev_nd_dec=prev_nd_dec,
                                last_nd_dec=last_nd_dec,
                                prev_nd_obj=prev_nd_obj,
                                last_nd_obj=last_nd_obj,
                                ref=ref,
                                cfg=cfg,
                                rng=rng,
                            )
                        else:
                            change_type = "none"
                        type_counts[change_type] = type_counts.get(change_type, 0) + 1
                        pop, f_mat = _tda_response_population(
                            change_type=change_type,
                            prev_nd_dec=np.empty((0, cfg.n_variables)) if prev_nd_dec is None else prev_nd_dec,
                            last_nd_dec=np.empty((0, cfg.n_variables)) if last_nd_dec is None else last_nd_dec,
                            prev_nd_obj=np.empty((0, ref.shape[0])) if prev_nd_obj is None else prev_nd_obj,
                            last_nd_obj=np.empty((0, ref.shape[0])) if last_nd_obj is None else last_nd_obj,
                            target_problem=problem,
                            cfg=cfg,
                            rng=rng,
                        )
                    else:
                        adr_meta_reuses += 1

            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            for _ in range(max(cfg.generations_per_step, 1)):
                pop, f_mat = _moead_de_step(
                    pop=pop,
                    f_mat=f_mat,
                    problem=problem,
                    rng=rng,
                    f=0.5,
                    cr=1.0,
                    mutation_eta=cfg.mutation_eta,
                )

            prev_nd_dec = None if last_nd_dec is None else last_nd_dec.copy()
            prev_nd_obj = None if last_nd_obj is None else last_nd_obj.copy()
            last_nd_dec, last_nd_obj = _tda_nondominated_snapshot(pop, f_mat, max_keep=cfg.pop_size)
            prev_eval_for_gate = f_mat.copy()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": int(type_counts.get("III", 0)),
            "memory_hits": int(type_counts.get("I", 0) + type_counts.get("II", 0)),
            "memory_archive_size": int(sum(type_counts.values())),
            "adr_meta_enabled": int(bool(getattr(cfg, "adr_meta_enabled", False))),
            "adr_meta_updates": int(adr_meta_updates),
            "adr_meta_reuses": int(adr_meta_reuses),
            "adr_meta_drift_mean": float(np.mean(adr_meta_drifts)) if adr_meta_drifts else np.nan,
            "adr_meta_reuse_ratio": float(adr_meta_reuses / max(len(detected_events), 1)),
            "adr_meta_reason_summary": json.dumps(adr_meta_reasons, sort_keys=True),
            "pm_family": "TDA-DMOEA-MOEAD-DE",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "TDA-DMOEA-MOEAD-DE",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def _ab_med_values(f_mat: np.ndarray) -> np.ndarray:
    obj = np.asarray(f_mat[:, 1:], dtype=float)
    n_points = obj.shape[0]
    if n_points <= 1:
        return np.ones(n_points, dtype=float)
    dists = np.sqrt(_pairwise_sq_dists(obj, obj))
    np.fill_diagonal(dists, np.inf)
    near = np.min(dists, axis=1)
    total = np.sum(np.where(np.isfinite(dists), dists, 0.0), axis=1)
    return near * total


def _ab_survivor_indices_med(f_mat: np.ndarray, n_survivors: int) -> np.ndarray:
    fronts = _fast_non_dominated_sort(f_mat)
    chosen: List[int] = []
    med = _ab_med_values(f_mat)
    for front in fronts:
        if len(chosen) + len(front) <= n_survivors:
            chosen.extend(front)
            continue
        front_sorted = sorted(front, key=lambda i: med[i], reverse=True)
        chosen.extend(front_sorted[: max(n_survivors - len(chosen), 0)])
        break
    return np.asarray(chosen, dtype=int)


def _ab_dominated_reinforcement(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ABDMOEAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    fronts = _fast_non_dominated_sort(f_mat)
    nd_idx = np.asarray(fronts[0] if fronts else [], dtype=int)
    threshold = int(math.floor(cfg.reinforcement_threshold * cfg.pop_size))
    if len(nd_idx) >= threshold or len(nd_idx) == 0:
        return pop, f_mat, False

    dominated_mask = np.ones(pop.shape[0], dtype=bool)
    dominated_mask[nd_idx] = False
    dominated = pop[dominated_mask]
    if len(dominated) == 0:
        return pop, f_mat, False

    c_nd = np.mean(pop[nd_idx], axis=0)
    c_ds = np.mean(dominated, axis=0)
    moved = np.clip(dominated + (c_nd - c_ds)[None, :], LOWER_BOUND, UPPER_BOUND)

    de_block = []
    ga_block = []
    for i, x in enumerate(moved):
        if i % 2 == 0 and pop.shape[0] >= 3:
            ids = rng.choice(pop.shape[0], size=3, replace=False)
            mutant = pop[ids[0]] + cfg.de_f * (pop[ids[1]] - pop[ids[2]])
            mask = rng.random(cfg.n_variables) < cfg.de_cr
            if not np.any(mask):
                mask[int(rng.integers(0, cfg.n_variables))] = True
            child = np.where(mask, mutant, x)
            de_block.append(np.clip(child, LOWER_BOUND, UPPER_BOUND))
        else:
            mate = pop[int(rng.integers(0, pop.shape[0]))]
            c1, _c2 = _sbx_crossover(x, mate, rng, eta=cfg.sbx_eta, crossover_prob=cfg.crossover_prob)
            c1 = _poly_mutation(c1, rng, eta=cfg.mutation_eta, mutation_prob=1.0 / max(cfg.n_variables, 1))
            ga_block.append(c1)

    blocks = [pop[nd_idx], moved]
    if de_block:
        blocks.append(np.asarray(de_block, dtype=float))
    if ga_block:
        blocks.append(np.asarray(ga_block, dtype=float))
    archive = np.vstack(blocks)
    archive_eval = _evaluate_population(problem, archive)
    keep = _ab_survivor_indices_med(archive_eval, n_survivors=cfg.pop_size)
    return archive[keep], archive_eval[keep], True


def _ab_prediction_block(
    block: np.ndarray,
    prev_centroid: Optional[np.ndarray],
    cur_centroid: np.ndarray,
    cfg: ABDMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(block) == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    direction = np.zeros(cfg.n_variables, dtype=float) if prev_centroid is None else cur_centroid - prev_centroid
    sigma = float(np.linalg.norm(direction) / max(len(block), 1))
    out = []
    for i, x in enumerate(block):
        mode = i % 3
        if mode == 0:
            r1, r2 = 0.5, 0.0
        elif mode == 1:
            r1, r2 = 1.0, 1.0
        else:
            r1, r2 = 1.5, 0.0
        noise = rng.normal(0.0, sigma, size=cfg.n_variables) if r2 > 0 else 0.0
        out.append(np.clip(x + r1 * direction + noise, LOWER_BOUND, UPPER_BOUND))
    return np.asarray(out, dtype=float)


def _ab_find_similar_archive(
    mean_obj_archive: List[np.ndarray],
    cur_mean_obj: np.ndarray,
    eps: float,
) -> Tuple[int, int]:
    for idx, archived in enumerate(mean_obj_archive):
        diff = np.abs(np.asarray(archived, dtype=float) - cur_mean_obj)
        strong = np.all(diff < 0.5 * eps)
        weak = np.all((diff >= 0.5 * eps) & (diff < 1.5 * eps))
        if strong:
            return 2, idx
        if weak:
            return 1, idx
    return 0, -1


def _ab_memory_block(
    block: np.ndarray,
    centroid_archive: List[np.ndarray],
    flag: int,
    index: int,
    prev_centroid: Optional[np.ndarray],
    cur_centroid: np.ndarray,
    cfg: ABDMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(block) == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    if flag == 2 and 0 <= index < len(centroid_archive):
        direction = np.asarray(centroid_archive[index], dtype=float) - cur_centroid
        return np.clip(block + direction[None, :], LOWER_BOUND, UPPER_BOUND)
    if flag == 1:
        direction = np.zeros(cfg.n_variables, dtype=float) if prev_centroid is None else cur_centroid - prev_centroid
        return np.clip(block + direction[None, :], LOWER_BOUND, UPPER_BOUND)
    return _ab_prediction_block(block, prev_centroid, cur_centroid, cfg, rng)


def _ab_diversity_block(
    block: np.ndarray,
    nd_pop: np.ndarray,
    prev_centroid: Optional[np.ndarray],
    cur_centroid: np.ndarray,
    cfg: ABDMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(block) == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    if len(nd_pop) == 0:
        nd_pop = block
    direction = np.zeros(cfg.n_variables, dtype=float) if prev_centroid is None else cur_centroid - prev_centroid
    sigma = max(float(np.linalg.norm(direction) / max(len(block), 1)), 0.02)
    out = []
    for x in block:
        choice = int(rng.integers(0, 3))
        if choice == 0:
            mate = nd_pop[int(rng.integers(0, len(nd_pop)))]
            c1, _c2 = _sbx_crossover(x, mate, rng, eta=cfg.sbx_eta, crossover_prob=cfg.crossover_prob)
            out.append(_poly_mutation(c1, rng, eta=cfg.mutation_eta, mutation_prob=1.0 / max(cfg.n_variables, 1)))
        elif choice == 1:
            out.append(_poly_mutation(x, rng, eta=max(cfg.mutation_eta * 0.5, 1.0), mutation_prob=0.5))
        else:
            out.append(np.clip(x + rng.normal(0.0, sigma, size=cfg.n_variables), LOWER_BOUND, UPPER_BOUND))
    return np.asarray(out, dtype=float)


def _ab_assign_strategy_blocks(
    pop: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    labels = rng.choice(np.array(["prediction", "memory", "diversity"], dtype=object), size=pop.shape[0], p=probs)
    blocks = {name: pop[labels == name] for name in ("prediction", "memory", "diversity")}
    return blocks, labels


def _ab_update_strategy_weights(
    candidates: np.ndarray,
    labels: np.ndarray,
    optimized_pop: np.ndarray,
    weights: np.ndarray,
    cfg: ABDMOEAConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(candidates) == 0 or len(optimized_pop) == 0:
        probs = weights / max(float(np.sum(weights)), 1e-12)
        return weights, probs
    dists = np.sqrt(_pairwise_sq_dists(candidates, optimized_pop))
    nearest = np.min(dists, axis=1)
    avg_dist = float(np.mean(nearest))
    names = ["prediction", "memory", "diversity"]
    gamma = np.zeros(3, dtype=float)
    for i, name in enumerate(names):
        mask = labels == name
        if not np.any(mask):
            gamma[i] = 0.0
        else:
            gamma[i] = float(np.mean(nearest[mask] < avg_dist))
    new_weights = (1.0 - cfg.boost_alpha) * weights + cfg.boost_alpha * gamma
    new_weights = np.maximum(new_weights, 1e-6)
    rho_m = float(np.clip(cfg.min_strategy_prob, 1e-6, 1.0 / 3.0))
    probs = rho_m + (1.0 - 3.0 * rho_m) * new_weights / max(float(np.sum(new_weights)), 1e-12)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    return new_weights, probs


def _ab_de_step(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ABDMOEAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    pop_size, n_variables = pop.shape
    offspring = []
    for i in range(pop_size):
        if pop_size >= 4:
            candidates = [idx for idx in range(pop_size) if idx != i]
            ids = rng.choice(candidates, size=3, replace=False)
            mutant = pop[ids[0]] + cfg.de_f * (pop[ids[1]] - pop[ids[2]])
        else:
            mutant = rng.random(n_variables)
        mask = rng.random(n_variables) < cfg.de_cr
        if not np.any(mask):
            mask[int(rng.integers(0, n_variables))] = True
        child = np.where(mask, mutant, pop[i])
        child = _poly_mutation(np.clip(child, LOWER_BOUND, UPPER_BOUND), rng, eta=cfg.mutation_eta, mutation_prob=1.0 / max(n_variables, 1))
        offspring.append(child)
    offspring = np.asarray(offspring, dtype=float)
    offspring_eval = _evaluate_population(problem, offspring)
    combined_pop = np.vstack((pop, offspring))
    combined_eval = np.vstack((f_mat, offspring_eval))
    keep = _survivor_indices(combined_eval, n_survivors=pop_size)
    return combined_pop[keep], combined_eval[keep]


def _ab_static_step(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ABDMOEAConfig,
    nsga_cfg: GAConfig,
    selected_smoa: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if selected_smoa == "rmeda":
        pfpa_cfg = PFPAConfig(
            pop_size=cfg.pop_size,
            n_variables=cfg.n_variables,
            horizon=cfg.horizon,
            generations_per_step=cfg.generations_per_step,
            rmeda_model_noise=0.035,
            rmeda_latent_scale=0.18,
        )
        return _pfpa_rmeda_step(pop, f_mat, problem, pfpa_cfg, rng)
    if selected_smoa == "moead_de":
        return _moead_de_step(
            pop=pop,
            f_mat=f_mat,
            problem=problem,
            rng=rng,
            f=cfg.de_f,
            cr=cfg.de_cr,
            mutation_eta=cfg.mutation_eta,
        )
    if selected_smoa == "nsga2_de":
        return _ab_de_step(pop, f_mat, problem, cfg, rng)
    return _nsga2_generation_step(pop=pop, f_mat=f_mat, problem=problem, cfg=nsga_cfg, rng=rng)


def _ab_population_med_score(f_mat: np.ndarray) -> float:
    return float(np.mean(_ab_med_values(f_mat))) if len(f_mat) else 0.0


def _ab_static_optimizer_boosting(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ABDMOEAConfig,
    nsga_cfg: GAConfig,
    rng: np.random.Generator,
) -> str:
    names = ["moead_de", "rmeda", "nsga2_de"]
    idx = rng.permutation(pop.shape[0])
    chunks = np.array_split(idx, len(names))
    evolved = []
    for name, chunk in zip(names, chunks):
        if len(chunk) == 0:
            continue
        sub_pop = pop[chunk].copy()
        sub_eval = f_mat[chunk].copy()
        local_cfg = replace(cfg, pop_size=len(sub_pop))
        local_nsga_cfg = GAConfig(
            pop_size=len(sub_pop),
            n_variables=cfg.n_variables,
            horizon=cfg.horizon,
            generations_per_step=cfg.generations_per_step,
            crossover_prob=cfg.crossover_prob,
            sbx_eta=cfg.sbx_eta,
            mutation_eta=cfg.mutation_eta,
        )
        for _ in range(max(1, cfg.generations_per_step)):
            sub_pop, sub_eval = _ab_static_step(sub_pop, sub_eval, problem, local_cfg, local_nsga_cfg, name, rng)
        evolved.append((name, sub_pop, sub_eval))

    med_values = np.asarray([_ab_population_med_score(ev[2]) for ev in evolved], dtype=float)
    med_score = med_values / max(float(np.sum(med_values)), 1e-12)
    merged_eval = np.vstack([ev[2] for ev in evolved])
    merged_labels = np.concatenate([[i] * len(ev[2]) for i, ev in enumerate(evolved)])
    nd_idx = np.asarray(_front0_indices(merged_eval), dtype=int)
    nd_score = np.zeros(len(evolved), dtype=float)
    if len(nd_idx) > 0:
        for i in range(len(evolved)):
            nd_score[i] = float(np.sum(merged_labels[nd_idx] == i)) / len(nd_idx)
    score = med_score + nd_score
    return str(evolved[int(np.argmax(score))][0]) if evolved else "nsga2"


def _ab_response(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ABDMOEAConfig,
    nsga_cfg: GAConfig,
    rng: np.random.Generator,
    weights: np.ndarray,
    probs: np.ndarray,
    centroid_archive: List[np.ndarray],
    mean_obj_archive: List[np.ndarray],
    prev_centroid: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    pop, f_mat, reinforced = _ab_dominated_reinforcement(pop, f_mat, problem, cfg, rng)
    cur_centroid = np.mean(pop, axis=0)
    cur_mean_obj = np.mean(f_mat[:, 1:], axis=0)

    centroid_archive.append(cur_centroid.copy())
    mean_obj_archive.append(cur_mean_obj.copy())
    if len(centroid_archive) > cfg.memory_size:
        del centroid_archive[0]
        del mean_obj_archive[0]

    fronts = _fast_non_dominated_sort(f_mat)
    nd_pop = pop[np.asarray(fronts[0], dtype=int)] if fronts and fronts[0] else pop
    flag, index = _ab_find_similar_archive(mean_obj_archive[:-1], cur_mean_obj, cfg.memory_similarity_eps)
    blocks, labels = _ab_assign_strategy_blocks(pop, probs, rng)
    pred = _ab_prediction_block(blocks["prediction"], prev_centroid, cur_centroid, cfg, rng)
    mem = _ab_memory_block(blocks["memory"], centroid_archive[:-1], flag, index, prev_centroid, cur_centroid, cfg, rng)
    div = _ab_diversity_block(blocks["diversity"], nd_pop, prev_centroid, cur_centroid, cfg, rng)
    candidates = np.vstack([b for b in (pred, mem, div) if len(b) > 0])
    candidate_labels = np.concatenate(
        [
            np.array(["prediction"] * len(pred), dtype=object),
            np.array(["memory"] * len(mem), dtype=object),
            np.array(["diversity"] * len(div), dtype=object),
        ]
    )
    if len(candidates) < cfg.pop_size:
        random_block = rng.random((cfg.pop_size - len(candidates), cfg.n_variables))
        candidates = np.vstack((candidates, random_block)) if len(candidates) else random_block
        candidate_labels = np.concatenate((candidate_labels, np.array(["diversity"] * len(random_block), dtype=object)))
    candidate_eval = _evaluate_population(problem, candidates)
    keep = _survivor_indices(candidate_eval, n_survivors=cfg.pop_size)
    response_pop = candidates[keep]
    response_eval = candidate_eval[keep]

    optimized_pop, optimized_eval = response_pop.copy(), response_eval.copy()
    for _ in range(max(1, cfg.generations_per_step)):
        optimized_pop, optimized_eval = _nsga2_generation_step(optimized_pop, optimized_eval, problem, nsga_cfg, rng)
    weights, probs = _ab_update_strategy_weights(candidates, candidate_labels, optimized_pop, weights, cfg)
    return response_pop, response_eval, weights, probs, int(reinforced)


def run_single_ab_dmoea_2024(problem_name: str, cfg: ABDMOEAConfig, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        template_problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        true_events = list(template_problem.true_change_points(cfg.horizon))
        ref = ref_point_for_problem(problem_name)
        nsga_cfg = GAConfig(
            pop_size=cfg.pop_size,
            n_variables=cfg.n_variables,
            horizon=cfg.horizon,
            generations_per_step=cfg.generations_per_step,
            crossover_prob=cfg.crossover_prob,
            sbx_eta=cfg.sbx_eta,
            mutation_eta=cfg.mutation_eta,
        )

        problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, 0)
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        f_mat = _evaluate_population(problem, pop)
        weights = np.ones(3, dtype=float)
        probs = np.ones(3, dtype=float) / 3.0
        centroid_archive: List[np.ndarray] = []
        mean_obj_archive: List[np.ndarray] = []
        prev_centroid: Optional[np.ndarray] = None
        selected_smoa = "moead_de"
        static_boost_done = False
        detected_events: List[int] = []
        reinforcement_hits = 0
        prev_eval_for_gate = f_mat.copy()
        adr_meta_updates = 0
        adr_meta_reuses = 0
        adr_meta_drifts: List[float] = []
        adr_meta_reasons: Dict[str, int] = {}
        last_adr_meta_update_t: Optional[int] = None

        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, t)
            if t > 0:
                f_mat = _evaluate_population(problem, pop)
                changed = t in set(true_events) if cfg.detection_mode == "oracle" else False
                if changed:
                    should_update, drift, reason = _adr_meta_should_update(
                        cfg=cfg,
                        t=t,
                        previous_eval=prev_eval_for_gate,
                        current_eval=f_mat,
                        last_update_t=last_adr_meta_update_t,
                    )
                    adr_meta_drifts.append(float(drift))
                    adr_meta_reasons[reason] = adr_meta_reasons.get(reason, 0) + 1
                    detected_events.append(t)
                    if should_update:
                        adr_meta_updates += int(bool(getattr(cfg, "adr_meta_enabled", False)))
                        last_adr_meta_update_t = t
                        pop, f_mat, weights, probs, reinforced = _ab_response(
                            pop=pop,
                            f_mat=f_mat,
                            problem=problem,
                            cfg=cfg,
                            nsga_cfg=nsga_cfg,
                            rng=rng,
                            weights=weights,
                            probs=probs,
                            centroid_archive=centroid_archive,
                            mean_obj_archive=mean_obj_archive,
                            prev_centroid=prev_centroid,
                        )
                        reinforcement_hits += reinforced
                        if not static_boost_done:
                            selected_smoa = _ab_static_optimizer_boosting(pop, f_mat, problem, cfg, nsga_cfg, rng)
                            static_boost_done = True
                    else:
                        adr_meta_reuses += 1

            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            for _ in range(max(cfg.generations_per_step, 1)):
                pop, f_mat = _ab_static_step(pop, f_mat, problem, cfg, nsga_cfg, selected_smoa, rng)
            prev_centroid = np.mean(pop, axis=0)
            prev_eval_for_gate = f_mat.copy()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": int(reinforcement_hits),
            "memory_hits": int(len(centroid_archive)),
            "memory_archive_size": int(len(centroid_archive)),
            "adr_meta_enabled": int(bool(getattr(cfg, "adr_meta_enabled", False))),
            "adr_meta_updates": int(adr_meta_updates),
            "adr_meta_reuses": int(adr_meta_reuses),
            "adr_meta_drift_mean": float(np.mean(adr_meta_drifts)) if adr_meta_drifts else np.nan,
            "adr_meta_reuse_ratio": float(adr_meta_reuses / max(len(detected_events), 1)),
            "adr_meta_reason_summary": json.dumps(adr_meta_reasons, sort_keys=True),
            "pm_family": f"AB-DMOEA-{selected_smoa}",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "AB-DMOEA",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


ADR_DMOEA_STRATEGIES = ("diversity", "prediction", "memory")


def _adr_dmoea_probabilities(weights: np.ndarray, cfg: ADRDMOEAConfig) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    k = len(ADR_DMOEA_STRATEGIES)
    theta_min = float(np.clip(cfg.theta_min, 1e-9, 1.0 / k))
    probs = theta_min + (1.0 - k * theta_min) * weights / max(float(np.sum(weights)), 1e-12)
    return probs / max(float(np.sum(probs)), 1e-12)


def _adr_dmoea_subpop_indices(pop_size: int, subpop_size: int) -> List[np.ndarray]:
    subpop_size = int(max(subpop_size, 1))
    n_subpops = int(max(1, pop_size // subpop_size))
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(np.arange(pop_size), n_subpops) if len(chunk) > 0]


def _adr_dmoea_reflect_bounds(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    out = np.where(out < LOWER_BOUND, np.minimum(UPPER_BOUND, 2.0 * LOWER_BOUND - out), out)
    out = np.where(out > UPPER_BOUND, np.maximum(LOWER_BOUND, 2.0 * UPPER_BOUND - out), out)
    return np.clip(out, LOWER_BOUND, UPPER_BOUND)


def _adr_dmoea_diversity_strategy(
    block: np.ndarray,
    block_indices: np.ndarray,
    pop: np.ndarray,
    f_mat: np.ndarray,
    cfg: ADRDMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(block) == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    fronts = _fast_non_dominated_sort(f_mat)
    nd_idx = set(fronts[0] if fronts and fronts[0] else [])
    nd_pop = pop[np.asarray(list(nd_idx), dtype=int)] if nd_idx else pop
    out = []
    for local_pos, x in enumerate(np.asarray(block, dtype=float)):
        source_idx = int(block_indices[local_pos])
        if source_idx in nd_idx:
            mask = rng.random(cfg.n_variables) < (1.0 / max(cfg.n_variables, 1))
            noise = rng.normal(0.0, cfg.diversity_gaussian_sigma, size=cfg.n_variables)
            child = np.where(mask, x + noise, x)
        elif rng.random() < cfg.diversity_poly_prob:
            if len(nd_pop) > 0 and rng.random() < 0.5:
                base = np.asarray(nd_pop[int(rng.integers(0, len(nd_pop)))], dtype=float)
            else:
                base = x
            child = _poly_mutation(
                base,
                rng=rng,
                eta=cfg.mutation_eta,
                mutation_prob=1.0 / max(cfg.n_variables, 1),
            )
        elif pop.shape[0] >= 3:
            ids = rng.choice(pop.shape[0], size=3, replace=False)
            rho = rng.uniform(0.0, 2.0)
            child = pop[int(ids[0])] + rho * (pop[int(ids[1])] - pop[int(ids[2])])
        else:
            child = rng.random(cfg.n_variables)
        out.append(_adr_dmoea_reflect_bounds(child))
    return np.asarray(out, dtype=float)


def _adr_dmoea_pca_projection(x: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=0)
    centered = x - mean[None, :]
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        keep = int(min(max(dim, 1), vt.shape[0], x.shape[1]))
        w = vt[:keep].T
    except np.linalg.LinAlgError:
        keep = int(min(max(dim, 1), x.shape[1]))
        w = np.eye(x.shape[1], keep)
    return mean, w


def _adr_dmoea_knn_positive(
    reference_z: np.ndarray,
    reference_labels: np.ndarray,
    test_z: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    if len(test_z) == 0 or len(reference_z) == 0:
        return np.zeros(len(test_z), dtype=bool)
    k = int(min(max(n_neighbors, 1), len(reference_z)))
    dists = _pairwise_sq_dists(test_z, reference_z)
    order = np.argsort(dists, axis=1)[:, :k]
    votes = np.mean(reference_labels[order], axis=1)
    return votes >= 0.5


def _adr_dmoea_static_refine(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ADRDMOEAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    refined_pop = np.asarray(pop, dtype=float).copy()
    refined_eval = np.asarray(f_mat, dtype=float).copy()
    for _ in range(max(1, cfg.generations_per_step)):
        refined_pop, refined_eval = _moead_de_step(
            pop=refined_pop,
            f_mat=refined_eval,
            problem=problem,
            rng=rng,
            f=cfg.de_f,
            cr=cfg.de_cr,
            mutation_eta=cfg.mutation_eta,
            neighborhood_ratio=cfg.moead_neighborhood_ratio,
            max_neighbors=cfg.moead_max_neighbors,
        )
    return refined_pop, refined_eval


def _adr_dmoea_prediction_strategy(
    block: np.ndarray,
    problem,
    cfg: ADRDMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    target_size = int(len(block))
    if target_size <= 0:
        return np.empty((0, cfg.n_variables), dtype=float)

    block = np.asarray(block, dtype=float)
    block_eval = _evaluate_population(problem, block)
    evolved_pop, evolved_eval = _adr_dmoea_static_refine(
        pop=block.copy(),
        f_mat=block_eval.copy(),
        problem=problem,
        cfg=cfg,
        rng=rng,
    )

    rand_prev = rng.random((target_size, cfg.n_variables))
    rand_cur = rng.random((target_size, cfg.n_variables))
    n_unlabeled = int(max(target_size * cfg.prediction_random_factor, cfg.pop_size))
    unlabeled = rng.random((n_unlabeled, cfg.n_variables))

    x_source = np.vstack((rand_prev, block))
    y_source = np.concatenate((np.zeros(target_size, dtype=float), np.ones(target_size, dtype=float)))
    x_labeled_target = np.vstack((rand_cur, evolved_pop))
    y_labeled_target = np.concatenate((np.zeros(target_size, dtype=float), np.ones(target_size, dtype=float)))
    x_all = np.vstack((x_source, x_labeled_target, unlabeled))
    pca_dim = cfg.prediction_pca_dim if cfg.prediction_pca_dim > 0 else min(cfg.n_variables, max(1, target_size))
    mean, w = _adr_dmoea_pca_projection(x_all, dim=pca_dim)

    z_source = (x_source - mean[None, :]) @ w
    z_labeled_target = (x_labeled_target - mean[None, :]) @ w
    z_unlabeled = (unlabeled - mean[None, :]) @ w

    mu_source = np.mean(z_source, axis=0)
    sigma_source = np.maximum(np.std(z_source, axis=0), 1e-12)
    mu_target = np.mean(z_labeled_target, axis=0)
    sigma_target = np.maximum(np.std(z_labeled_target, axis=0), 1e-12)
    z_source_aligned = ((z_source - mu_source[None, :]) / sigma_source[None, :]) * sigma_target[None, :] + mu_target[None, :]

    reference_z = np.vstack((z_source_aligned, z_labeled_target))
    reference_labels = np.concatenate((y_source, y_labeled_target))
    positive = _adr_dmoea_knn_positive(
        reference_z=reference_z,
        reference_labels=reference_labels,
        test_z=z_unlabeled,
        n_neighbors=cfg.prediction_neighbors,
    )
    candidates = unlabeled[positive]
    if len(candidates) < target_size:
        center = np.mean(evolved_pop, axis=0) if len(evolved_pop) else np.mean(block, axis=0)
        sigma = max(float(np.mean(np.std(evolved_pop, axis=0))) if len(evolved_pop) > 1 else 0.0, 0.035)
        fill = np.clip(center[None, :] + rng.normal(0.0, sigma, size=(target_size - len(candidates), cfg.n_variables)), LOWER_BOUND, UPPER_BOUND)
        candidates = np.vstack((candidates, fill)) if len(candidates) else fill

    candidates = np.clip(candidates, LOWER_BOUND, UPPER_BOUND)
    if len(candidates) > target_size:
        cand_eval = _evaluate_population(problem, candidates)
        keep = _survivor_indices(cand_eval, n_survivors=target_size)
        candidates = candidates[keep]
    return np.asarray(candidates[:target_size], dtype=float)


def _adr_dmoea_find_similarity(
    mean_obj_archive: List[np.ndarray],
    cur_mean_obj: np.ndarray,
    rho: float,
) -> Tuple[int, int]:
    rho = float(max(rho, 1e-12))
    best_weak = -1
    for idx, archived in enumerate(mean_obj_archive):
        diff = np.abs(np.asarray(archived, dtype=float) - cur_mean_obj)
        if np.all(diff < 0.5 * rho):
            return 2, idx
        if np.all(diff < 1.5 * rho) and np.any(diff >= 0.5 * rho) and best_weak < 0:
            best_weak = idx
    if best_weak >= 0:
        return 1, best_weak
    return 0, -1


def _adr_dmoea_memory_strategy(
    block: np.ndarray,
    centroid_archive: List[np.ndarray],
    flag: int,
    index: int,
    prev_centroid: Optional[np.ndarray],
    cur_centroid: np.ndarray,
    cfg: ADRDMOEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(block) == 0:
        return np.empty((0, cfg.n_variables), dtype=float)
    block = np.asarray(block, dtype=float)
    if flag == 2 and 0 <= index < len(centroid_archive):
        direction = np.asarray(centroid_archive[index], dtype=float) - cur_centroid
        return np.clip(block + direction[None, :], LOWER_BOUND, UPPER_BOUND)

    direction = np.zeros(cfg.n_variables, dtype=float) if prev_centroid is None else cur_centroid - prev_centroid
    noise_scale = 0.0 if flag == 1 else cfg.memory_noise
    noise = rng.normal(0.0, noise_scale, size=block.shape)
    return np.clip(block + direction[None, :] + noise, LOWER_BOUND, UPPER_BOUND)


def _adr_dmoea_make_response_candidates(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ADRDMOEAConfig,
    rng: np.random.Generator,
    probs: np.ndarray,
    centroid_archive: List[np.ndarray],
    mean_obj_archive: List[np.ndarray],
    prev_centroid: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]], Dict[str, int], int]:
    cur_centroid = np.mean(pop, axis=0)
    cur_mean_obj = np.mean(f_mat[:, 1:], axis=0)
    flag, index = _adr_dmoea_find_similarity(mean_obj_archive, cur_mean_obj, cfg.memory_similarity_eps)
    chunks = _adr_dmoea_subpop_indices(pop.shape[0], cfg.subpop_size)
    out_blocks = []
    out_labels = []
    subpop_records: List[Tuple[int, int, int]] = []
    counts = {name: 0 for name in ADR_DMOEA_STRATEGIES}
    cursor = 0

    for idx in chunks:
        strategy_idx = int(rng.choice(len(ADR_DMOEA_STRATEGIES), p=probs))
        strategy = ADR_DMOEA_STRATEGIES[strategy_idx]
        block = pop[idx]
        if strategy == "diversity":
            updated = _adr_dmoea_diversity_strategy(block, idx, pop, f_mat, cfg, rng)
        elif strategy == "prediction":
            updated = _adr_dmoea_prediction_strategy(block, problem, cfg, rng)
        else:
            updated = _adr_dmoea_memory_strategy(
                block=block,
                centroid_archive=centroid_archive,
                flag=flag,
                index=index,
                prev_centroid=prev_centroid,
                cur_centroid=cur_centroid,
                cfg=cfg,
                rng=rng,
            )
        out_blocks.append(updated)
        out_labels.extend([strategy_idx] * len(updated))
        subpop_records.append((cursor, cursor + len(updated), strategy_idx))
        counts[strategy] += 1
        cursor += len(updated)

    candidate_pop = np.vstack(out_blocks) if out_blocks else rng.random((cfg.pop_size, cfg.n_variables))
    if len(candidate_pop) < cfg.pop_size:
        fill = rng.random((cfg.pop_size - len(candidate_pop), cfg.n_variables))
        candidate_pop = np.vstack((candidate_pop, fill))
        out_labels.extend([0] * len(fill))
        subpop_records.append((cursor, cursor + len(fill), 0))
    candidate_pop = np.asarray(candidate_pop[: cfg.pop_size], dtype=float)
    candidate_labels = np.asarray(out_labels[: cfg.pop_size], dtype=int)
    return candidate_pop, candidate_labels, subpop_records, counts, flag


def _adr_dmoea_update_weights(
    candidate_pop: np.ndarray,
    optimized_pop: np.ndarray,
    subpop_records: List[Tuple[int, int, int]],
    weights: np.ndarray,
    cfg: ADRDMOEAConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(candidate_pop) == 0 or len(optimized_pop) == 0:
        probs = _adr_dmoea_probabilities(weights, cfg)
        return weights, probs
    dists = np.sqrt(_pairwise_sq_dists(candidate_pop, optimized_pop))
    nearest = np.min(dists, axis=1)
    average_dist = float(np.mean(nearest))
    alpha = np.zeros(len(ADR_DMOEA_STRATEGIES), dtype=float)
    beta = np.zeros(len(ADR_DMOEA_STRATEGIES), dtype=float)
    for start, end, strategy_idx in subpop_records:
        if end <= start:
            continue
        sub_avg = float(np.mean(nearest[start:end]))
        alpha[strategy_idx] += 1.0 if sub_avg <= average_dist else 0.0
        beta[strategy_idx] += 1.0
    delta = np.divide(alpha, beta, out=np.zeros_like(alpha), where=beta > 0)
    new_weights = cfg.gamma * np.maximum(weights, 1e-12) + (1.0 - cfg.gamma) * delta
    new_weights = np.maximum(new_weights, 1e-12)
    return new_weights, _adr_dmoea_probabilities(new_weights, cfg)


def _adr_dmoea_response(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: ADRDMOEAConfig,
    rng: np.random.Generator,
    weights: np.ndarray,
    probs: np.ndarray,
    centroid_archive: List[np.ndarray],
    mean_obj_archive: List[np.ndarray],
    prev_centroid: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], int]:
    cur_centroid = np.mean(pop, axis=0)
    cur_mean_obj = np.mean(f_mat[:, 1:], axis=0)
    candidate_pop, _candidate_labels, subpop_records, counts, memory_flag = _adr_dmoea_make_response_candidates(
        pop=pop,
        f_mat=f_mat,
        problem=problem,
        cfg=cfg,
        rng=rng,
        probs=probs,
        centroid_archive=centroid_archive,
        mean_obj_archive=mean_obj_archive,
        prev_centroid=prev_centroid,
    )
    candidate_pop = np.clip(candidate_pop, LOWER_BOUND, UPPER_BOUND)
    candidate_eval = _evaluate_population(problem, candidate_pop)

    optimized_pop, optimized_eval = _adr_dmoea_static_refine(
        pop=candidate_pop.copy(),
        f_mat=candidate_eval.copy(),
        problem=problem,
        cfg=cfg,
        rng=rng,
    )
    weights, probs = _adr_dmoea_update_weights(candidate_pop, optimized_pop, subpop_records, weights, cfg)

    centroid_archive.append(cur_centroid.copy())
    mean_obj_archive.append(cur_mean_obj.copy())
    if len(centroid_archive) > cfg.memory_size:
        del centroid_archive[0]
        del mean_obj_archive[0]

    return optimized_pop, optimized_eval, weights, probs, counts, memory_flag


def run_single_adr_dmoea_2026(problem_name: str, cfg: ADRDMOEAConfig, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        template_problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        true_events = list(template_problem.true_change_points(cfg.horizon))
        true_events_set = set(true_events)
        ref = ref_point_for_problem(problem_name)

        problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, 0)
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        f_mat = _evaluate_population(problem, pop)
        weights = np.ones(len(ADR_DMOEA_STRATEGIES), dtype=float)
        probs = np.ones(len(ADR_DMOEA_STRATEGIES), dtype=float) / len(ADR_DMOEA_STRATEGIES)
        centroid_archive: List[np.ndarray] = []
        mean_obj_archive: List[np.ndarray] = []
        prev_centroid: Optional[np.ndarray] = None
        environment_index = 1
        detected_events: List[int] = []
        strategy_counts = {name: 0 for name in ADR_DMOEA_STRATEGIES}
        memory_similarity_hits = 0

        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            problem = _make_problem_at(problem_name, cfg.n_variables, cfg.horizon, t)
            ran_static_stage = False
            if t == 0:
                pop, f_mat = _adr_dmoea_static_refine(
                    pop=pop,
                    f_mat=f_mat,
                    problem=problem,
                    cfg=cfg,
                    rng=rng,
                )
                ran_static_stage = True
            else:
                f_mat = _evaluate_population(problem, pop)
                changed = t in true_events_set if cfg.detection_mode == "oracle" else False
                if changed:
                    detected_events.append(t)
                    environment_index += 1
                    if environment_index <= 2:
                        pop = rng.random((cfg.pop_size, cfg.n_variables))
                        f_mat = _evaluate_population(problem, pop)
                        pop, f_mat = _adr_dmoea_static_refine(
                            pop=pop,
                            f_mat=f_mat,
                            problem=problem,
                            cfg=cfg,
                            rng=rng,
                        )
                    else:
                        pop, f_mat, weights, probs, counts, memory_flag = _adr_dmoea_response(
                            pop=pop,
                            f_mat=f_mat,
                            problem=problem,
                            cfg=cfg,
                            rng=rng,
                            weights=weights,
                            probs=probs,
                            centroid_archive=centroid_archive,
                            mean_obj_archive=mean_obj_archive,
                            prev_centroid=prev_centroid,
                        )
                        for name, value in counts.items():
                            strategy_counts[name] = strategy_counts.get(name, 0) + int(value)
                        memory_similarity_hits += int(memory_flag > 0)
                    ran_static_stage = True

            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            if not ran_static_stage:
                pop, f_mat = _adr_dmoea_static_refine(
                    pop=pop,
                    f_mat=f_mat,
                    problem=problem,
                    cfg=cfg,
                    rng=rng,
                )
            prev_centroid = np.mean(pop, axis=0)

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": 0,
            "memory_hits": int(memory_similarity_hits),
            "memory_archive_size": int(len(centroid_archive)),
            "pm_family": "ADR-DMOEA-MOEAD-DE",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "ADR-DMOEA-MOEAD-DE",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def _steady_state_insertions(
    pop: np.ndarray,
    f_mat: np.ndarray,
    problem,
    cfg: GAConfig,
    rng: np.random.Generator,
    n_insertions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cur_pop = pop.copy()
    cur_eval = f_mat.copy()
    for _ in range(max(n_insertions, 0)):
        p1 = _binary_tournament(cur_pop, cur_eval, rng)
        p2 = _binary_tournament(cur_pop, cur_eval, rng)
        c1, c2 = _sbx_crossover(p1, p2, rng, eta=cfg.sbx_eta, crossover_prob=cfg.crossover_prob)
        child = c1 if rng.random() < 0.5 else c2
        child = _poly_mutation(child, rng, eta=cfg.mutation_eta, mutation_prob=1.0 / max(cfg.n_variables, 1))
        child_eval = _evaluate_population(problem, child[None, :])
        combined_pop = np.vstack((cur_pop, child[None, :]))
        combined_eval = np.vstack((cur_eval, child_eval))
        keep = _survivor_indices(combined_eval, n_survivors=cur_pop.shape[0])
        cur_pop = combined_pop[keep]
        cur_eval = combined_eval[keep]
    return cur_pop, cur_eval


def _extract_diverse_elites(pop: np.ndarray, f_mat: np.ndarray, n_keep: int) -> np.ndarray:
    if pop.size == 0:
        return np.empty((0, 0), dtype=float)
    idx = _survivor_indices(f_mat, n_survivors=min(n_keep, pop.shape[0]))
    return np.asarray(pop[idx], dtype=float)


def _objective_center(f_mat: np.ndarray) -> np.ndarray:
    if f_mat.size == 0:
        return np.array([], dtype=float)
    idx = _survivor_indices(f_mat, n_survivors=min(max(4, f_mat.shape[0] // 4), f_mat.shape[0]))
    return np.mean(np.asarray(f_mat[idx, 1:], dtype=float), axis=0)


def _hypermutate_block(
    base: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    mutation_prob: float,
    eta: float,
) -> np.ndarray:
    if base.size == 0 or n_samples <= 0:
        return np.empty((0, base.shape[1] if base.ndim == 2 else 0), dtype=float)
    idx = rng.integers(0, base.shape[0], size=n_samples)
    block = []
    for i in idx:
        x = _poly_mutation(np.asarray(base[i], dtype=float), rng, eta=eta, mutation_prob=mutation_prob)
        block.append(x)
    return np.asarray(block, dtype=float)


@dataclass
class MemorySnapshot:
    decisions: np.ndarray
    center_obj: np.ndarray


class SentinelChangeDetector:
    def __init__(
        self,
        n_sentinels: int,
        alpha: float,
        lambda_ewma: float,
        k_sigma: float,
        refresh_period: int,
        cooldown: int,
        warmup: int,
        history_size: int,
    ):
        self.n_sentinels = int(max(n_sentinels, 4))
        self.alpha = float(max(alpha, 1e-8))
        self.lambda_ewma = float(np.clip(lambda_ewma, 0.0, 0.99))
        self.k_sigma = float(max(k_sigma, 0.5))
        self.refresh_period = int(max(refresh_period, 1))
        self.cooldown = int(max(cooldown, 0))
        self.warmup = int(max(warmup, 1))
        self.history_size = int(max(history_size, 8))
        self.sentinel_x: Optional[np.ndarray] = None
        self.prev_eval: Optional[np.ndarray] = None
        self.ewma_mean = 0.0
        self.ewma_var = 0.0
        self.history: List[float] = []
        self.last_detect = -10**9
        self.last_refresh = -10**9

    def refresh(self, pop: np.ndarray, f_mat: np.ndarray, t: int) -> None:
        idx = _survivor_indices(f_mat, n_survivors=min(self.n_sentinels, pop.shape[0]))
        self.sentinel_x = np.asarray(pop[idx], dtype=float).copy()
        self.prev_eval = np.asarray(f_mat[idx], dtype=float).copy()
        self.last_refresh = int(t)

    def detect(self, problem, t: int) -> Dict[str, float]:
        if self.sentinel_x is None or self.prev_eval is None or len(self.sentinel_x) == 0:
            return {
                "detected": False,
                "score": 0.0,
                "threshold": float("nan"),
                "severity": 0.0,
            }

        cur_eval = _evaluate_population(problem, self.sentinel_x)
        prev = np.asarray(self.prev_eval, dtype=float)
        obj_scale = np.maximum(np.mean(np.abs(prev[:, 1:]), axis=0), self.alpha)
        obj_delta = np.mean(np.abs(cur_eval[:, 1:] - prev[:, 1:]) / obj_scale[None, :], axis=1)
        cv_scale = max(float(np.mean(np.abs(prev[:, 0]))), self.alpha)
        cv_delta = np.abs(cur_eval[:, 0] - prev[:, 0]) / cv_scale
        score = float(np.mean(obj_delta + cv_delta))

        threshold = float("inf")
        if len(self.history) >= self.warmup:
            mean = float(np.mean(self.history))
            std = float(np.std(self.history))
            threshold = mean + self.k_sigma * max(std, self.alpha)
            detected = score > threshold and (t - self.last_detect) >= self.cooldown
        else:
            detected = False

        if detected:
            self.last_detect = int(t)
        else:
            if not self.history:
                self.ewma_mean = score
                self.ewma_var = 0.0
            else:
                diff = score - self.ewma_mean
                self.ewma_mean = self.lambda_ewma * self.ewma_mean + (1.0 - self.lambda_ewma) * score
                self.ewma_var = self.lambda_ewma * self.ewma_var + (1.0 - self.lambda_ewma) * (diff * diff)
            self.history.append(score)
            self.history = self.history[-self.history_size :]

        baseline = max(float(np.mean(self.history)) if self.history else score, self.alpha)
        severity = score / baseline
        return {
            "detected": bool(detected),
            "score": score,
            "threshold": threshold,
            "severity": float(severity),
        }


class BaseDynamicNSGA2:
    def __init__(self, cfg: GAConfig):
        self.cfg = cfg
        self.env_changes = 0
        self.memory_hits = 0
        self.memory_archive: List[MemorySnapshot] = []
        self.extra_steady_cycles = 0

    @property
    def name(self) -> str:
        raise NotImplementedError

    def initialize(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random((self.cfg.pop_size, self.cfg.n_variables))

    def _record_memory(self, pop: np.ndarray, f_mat: np.ndarray) -> None:
        if self.cfg.memory_max_snapshots <= 0:
            return
        decisions = _extract_diverse_elites(pop, f_mat, n_keep=self.cfg.memory_snapshot_size)
        if decisions.size == 0:
            return
        center = _objective_center(f_mat)
        self.memory_archive.append(MemorySnapshot(decisions=decisions, center_obj=center))
        self.memory_archive = self.memory_archive[-self.cfg.memory_max_snapshots :]

    def on_change(self, pop: np.ndarray, f_mat: np.ndarray, rng: np.random.Generator, detector_info: Dict[str, float]) -> np.ndarray:
        self.env_changes += 1
        return pop

    def evolve(self, pop: np.ndarray, f_mat: np.ndarray, problem, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        cur_pop = pop
        cur_eval = f_mat
        for _ in range(max(self.cfg.generations_per_step, 1)):
            cur_pop, cur_eval = _nsga2_generation_step(cur_pop, cur_eval, problem, self.cfg, rng)
        return cur_pop, cur_eval


class DNSGAIIB(BaseDynamicNSGA2):
    @property
    def name(self) -> str:
        return "dnsga2_b"

    def on_change(self, pop: np.ndarray, f_mat: np.ndarray, rng: np.random.Generator, detector_info: Dict[str, float]) -> np.ndarray:
        self.env_changes += 1
        n_replace = max(1, int(round(self.cfg.pop_size * self.cfg.dnsga_immigrant_ratio)))
        replace_idx = rng.choice(self.cfg.pop_size, size=n_replace, replace=False)
        immigrants = _hypermutate_block(
            base=pop,
            rng=rng,
            n_samples=n_replace,
            mutation_prob=self.cfg.dnsga_hypermutation_prob,
            eta=self.cfg.dnsga_hypermutation_eta,
        )
        new_pop = pop.copy()
        new_pop[replace_idx] = immigrants
        return np.clip(new_pop, LOWER_BOUND, UPPER_BOUND)


class MemoryImmigrantsNSGA2(BaseDynamicNSGA2):
    @property
    def name(self) -> str:
        return "memory_nsga2"

    def on_change(self, pop: np.ndarray, f_mat: np.ndarray, rng: np.random.Generator, detector_info: Dict[str, float]) -> np.ndarray:
        self.env_changes += 1
        new_pop = pop.copy()
        current_center = _objective_center(f_mat)
        memory_count = max(1, int(round(self.cfg.pop_size * self.cfg.memory_ratio)))
        hyper_count = max(1, int(round(self.cfg.pop_size * self.cfg.memory_hyper_ratio)))

        memory_block = np.empty((0, self.cfg.n_variables), dtype=float)
        if self.memory_archive and current_center.size > 0:
            snapshot = min(
                self.memory_archive,
                key=lambda s: float(np.linalg.norm(current_center - s.center_obj)) if s.center_obj.size else float("inf"),
            )
            if snapshot.decisions.size > 0:
                idx = rng.integers(0, snapshot.decisions.shape[0], size=memory_count)
                memory_block = np.asarray(snapshot.decisions[idx], dtype=float)
                self.memory_hits += 1

        hyper_block = _hypermutate_block(
            base=pop,
            rng=rng,
            n_samples=hyper_count,
            mutation_prob=self.cfg.memory_hypermutation_prob,
            eta=self.cfg.memory_hypermutation_eta,
        )
        random_count = max(self.cfg.pop_size - len(memory_block) - len(hyper_block), 0)
        random_block = rng.random((random_count, self.cfg.n_variables))
        injected = np.vstack([b for b in (memory_block, hyper_block, random_block) if len(b) > 0])
        injected = injected[: self.cfg.pop_size]
        if len(injected) == 0:
            return new_pop

        replace_idx = rng.choice(self.cfg.pop_size, size=len(injected), replace=False)
        new_pop[replace_idx] = injected
        return np.clip(new_pop, LOWER_BOUND, UPPER_BOUND)

    def evolve(self, pop: np.ndarray, f_mat: np.ndarray, problem, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        out_pop, out_eval = super().evolve(pop, f_mat, problem, rng)
        self._record_memory(out_pop, out_eval)
        return out_pop, out_eval


class MNSGAII(BaseDynamicNSGA2):
    @property
    def name(self) -> str:
        return "mnsga2"

    @property
    def _memory_capacity(self) -> int:
        return max(
            self.cfg.memory_snapshot_size,
            int(round(self.cfg.pop_size * self.cfg.mnsga_memory_capacity_factor)),
        )

    def _record_memory(self, pop: np.ndarray, f_mat: np.ndarray) -> None:
        if self.cfg.memory_max_snapshots <= 0:
            return
        n_keep = min(self._memory_capacity, pop.shape[0])
        decisions = _extract_diverse_elites(pop, f_mat, n_keep=n_keep)
        if decisions.size == 0:
            return
        center = _objective_center(f_mat)
        self.memory_archive.append(MemorySnapshot(decisions=decisions, center_obj=center))
        self.memory_archive = self.memory_archive[-self.cfg.memory_max_snapshots :]

    def _sample_memory(self, current_center: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if n_samples <= 0 or not self.memory_archive:
            return np.empty((0, self.cfg.n_variables), dtype=float)
        if current_center.size > 0:
            ranked = sorted(
                self.memory_archive,
                key=lambda s: float(np.linalg.norm(current_center - s.center_obj)) if s.center_obj.size else float("inf"),
            )
        else:
            ranked = list(self.memory_archive)
        topk = ranked[: max(1, min(self.cfg.mnsga_similarity_topk, len(ranked)))]
        blocks = [np.asarray(s.decisions, dtype=float) for s in topk if s.decisions.size > 0]
        if not blocks:
            return np.empty((0, self.cfg.n_variables), dtype=float)
        pool = np.vstack(blocks)
        idx = rng.integers(0, pool.shape[0], size=n_samples)
        self.memory_hits += 1
        return np.asarray(pool[idx], dtype=float)

    def on_change(self, pop: np.ndarray, f_mat: np.ndarray, rng: np.random.Generator, detector_info: Dict[str, float]) -> np.ndarray:
        self.env_changes += 1
        current_center = _objective_center(f_mat)
        memory_count = max(1, int(round(self.cfg.pop_size * self.cfg.mnsga_memory_ratio)))
        mutated_count = max(1, int(round(self.cfg.pop_size * self.cfg.mnsga_memory_mutation_ratio)))
        random_count = max(1, int(round(self.cfg.pop_size * self.cfg.mnsga_random_ratio)))

        memory_block = self._sample_memory(current_center, memory_count, rng)
        if len(memory_block) > 0:
            mutation_base = memory_block
        else:
            mutation_base = pop
        mutated_block = _hypermutate_block(
            base=mutation_base,
            rng=rng,
            n_samples=mutated_count,
            mutation_prob=self.cfg.memory_hypermutation_prob,
            eta=self.cfg.memory_hypermutation_eta,
        )
        random_block = rng.random((random_count, self.cfg.n_variables))
        injected = np.vstack([b for b in (memory_block, mutated_block, random_block) if len(b) > 0])
        injected = injected[: self.cfg.pop_size]
        if len(injected) == 0:
            return pop

        keep_count = max(self.cfg.pop_size - len(injected), 0)
        keep_idx = _survivor_indices(f_mat, n_survivors=keep_count) if keep_count > 0 else np.array([], dtype=int)
        new_pop = np.vstack((pop[keep_idx], injected))
        if new_pop.shape[0] < self.cfg.pop_size:
            fill = rng.random((self.cfg.pop_size - new_pop.shape[0], self.cfg.n_variables))
            new_pop = np.vstack((new_pop, fill))
        return np.clip(new_pop[: self.cfg.pop_size], LOWER_BOUND, UPPER_BOUND)

    def evolve(self, pop: np.ndarray, f_mat: np.ndarray, problem, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        out_pop, out_eval = super().evolve(pop, f_mat, problem, rng)
        self._record_memory(out_pop, out_eval)
        return out_pop, out_eval


def _windowed_post_change_mean(hist: np.ndarray, true_events: List[int], lookahead: int) -> float:
    if hist is None or len(hist) == 0 or len(true_events) == 0:
        return float("nan")
    n = len(hist)
    windows = []
    for cp in true_events:
        if cp >= n:
            continue
        end = min(n, cp + max(1, lookahead))
        if end <= cp:
            continue
        windows.append(float(np.mean(hist[cp:end])))
    return float(np.mean(windows)) if windows else float("nan")


def recovery_time(
    igd_hist: np.ndarray,
    true_events: List[int],
    lookahead: int = 25,
    baseline_window: int = 8,
    tol_ratio: float = 1.10,
) -> float:
    if len(true_events) == 0:
        return float("nan")

    rec = []
    n = len(igd_hist)
    for cp in true_events:
        if cp < baseline_window or cp >= (n - 1):
            continue
        base = float(np.median(igd_hist[cp - baseline_window : cp]))
        base = max(base, 1e-8)
        target = base * tol_ratio
        end = min(n, cp + lookahead + 1)
        rt = lookahead + 1
        for t in range(cp, end):
            if igd_hist[t] <= target:
                rt = t - cp
                break
        rec.append(rt)

    return float(np.mean(rec)) if rec else float("nan")


def compute_temporal_metrics(
    problem_name: str,
    horizon: int,
    n_variables: int,
    result: Dict[str, object],
    post_change_lookahead: int,
    recovery_lookahead: int,
    recovery_baseline_window: int,
    recovery_tol_ratio: float,
) -> Dict[str, float]:
    if result.get("status") != "ok":
        return {
            "post_change_igd": float("nan"),
            "post_change_hv": float("nan"),
            "recovery_time": float("nan"),
        }

    igd_hist = np.asarray(result.get("igd_hist", np.array([])), dtype=float)
    hv_hist = np.asarray(result.get("hv_hist", np.array([])), dtype=float)
    if len(igd_hist) == 0 or len(hv_hist) == 0:
        return {
            "post_change_igd": float("nan"),
            "post_change_hv": float("nan"),
            "recovery_time": float("nan"),
        }

    try:
        problem = PROBLEM_REGISTRY[problem_name](n_variables, horizon)
        true_events = problem.true_change_points(horizon)
    except Exception:
        true_events = []

    return {
        "post_change_igd": _windowed_post_change_mean(igd_hist, true_events, lookahead=post_change_lookahead),
        "post_change_hv": _windowed_post_change_mean(hv_hist, true_events, lookahead=post_change_lookahead),
        "recovery_time": recovery_time(
            igd_hist,
            true_events,
            lookahead=recovery_lookahead,
            baseline_window=recovery_baseline_window,
            tol_ratio=recovery_tol_ratio,
        ),
    }


def _oracle_change(problem_name: str, horizon: int, n_variables: int) -> set[int]:
    try:
        problem = PROBLEM_REGISTRY[problem_name](n_variables, horizon)
        return set(problem.true_change_points(horizon))
    except Exception:
        return set()


def _reevaluate_change(prev_eval: np.ndarray, cur_eval: np.ndarray, tol: float) -> bool:
    if prev_eval.shape != cur_eval.shape:
        return True
    diff = np.max(np.abs(cur_eval - prev_eval))
    return bool(diff > tol)


def _apply_dnsga2_b_response(
    pop: np.ndarray,
    rng: np.random.Generator,
    response_ratio: float,
    mutation_prob: float,
    mutation_eta: float,
) -> np.ndarray:
    pop_size = pop.shape[0]
    n_replace = max(1, int(round(pop_size * response_ratio)))
    source_idx = rng.integers(0, pop_size, size=n_replace)
    replace_idx = rng.choice(pop_size, size=n_replace, replace=False)
    new_pop = pop.copy()
    for src, dst in zip(source_idx, replace_idx):
        child = _poly_mutation(
            pop[src],
            rng=rng,
            eta=mutation_eta,
            mutation_prob=mutation_prob,
        )
        new_pop[dst] = child
    return np.clip(new_pop, LOWER_BOUND, UPPER_BOUND)


def run_single_dnsga2_2007(problem_name: str, cfg: DNSGA2Config, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        ref = ref_point_for_problem(problem_name)
        pop = rng.random((cfg.pop_size, cfg.n_variables))
        f_mat = _evaluate_population(problem, pop)
        oracle_events = _oracle_change(problem_name, cfg.horizon, cfg.n_variables)
        detected_events: List[int] = []
        prev_eval = f_mat.copy()
        adr_meta_updates = 0
        adr_meta_reuses = 0
        adr_meta_drifts: List[float] = []
        adr_meta_reasons: Dict[str, int] = {}
        last_adr_meta_update_t: Optional[int] = None

        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            if t > 0:
                changed = False
                if cfg.detection_mode == "oracle":
                    changed = t in oracle_events
                else:
                    cur_eval = _evaluate_population(problem, pop)
                    changed = _reevaluate_change(prev_eval, cur_eval, tol=cfg.detect_tolerance)
                    f_mat = cur_eval
                if changed:
                    if bool(getattr(cfg, "adr_meta_enabled", False)) and cfg.detection_mode == "oracle":
                        f_mat = _evaluate_population(problem, pop)
                    should_update, drift, reason = _adr_meta_should_update(
                        cfg=cfg,
                        t=t,
                        previous_eval=prev_eval,
                        current_eval=f_mat,
                        last_update_t=last_adr_meta_update_t,
                    )
                    adr_meta_drifts.append(float(drift))
                    adr_meta_reasons[reason] = adr_meta_reasons.get(reason, 0) + 1
                    detected_events.append(t)
                    if should_update:
                        adr_meta_updates += int(bool(getattr(cfg, "adr_meta_enabled", False)))
                        last_adr_meta_update_t = t
                        pop = _apply_dnsga2_b_response(
                            pop=pop,
                            rng=rng,
                            response_ratio=cfg.response_ratio,
                            mutation_prob=cfg.change_mutation_prob,
                            mutation_eta=cfg.change_mutation_eta,
                        )
                        f_mat = _evaluate_population(problem, pop)
                    else:
                        adr_meta_reuses += 1

            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            for _ in range(max(cfg.generations_per_step, 1)):
                pop, f_mat = _nsga2_generation_step(
                    pop=pop,
                    f_mat=f_mat,
                    problem=problem,
                    cfg=GAConfig(
                        pop_size=cfg.pop_size,
                        n_variables=cfg.n_variables,
                        horizon=cfg.horizon,
                        generations_per_step=cfg.generations_per_step,
                        crossover_prob=cfg.crossover_prob,
                        sbx_eta=cfg.sbx_eta,
                        mutation_eta=cfg.mutation_eta,
                    ),
                    rng=rng,
                )

            prev_eval = f_mat.copy()
            problem.update_time()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        true_events = sorted(oracle_events)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "adr_meta_enabled": int(bool(getattr(cfg, "adr_meta_enabled", False))),
            "adr_meta_updates": int(adr_meta_updates),
            "adr_meta_reuses": int(adr_meta_reuses),
            "adr_meta_drift_mean": float(np.mean(adr_meta_drifts)) if adr_meta_drifts else np.nan,
            "adr_meta_reuse_ratio": float(adr_meta_reuses / max(len(detected_events), 1)),
            "adr_meta_reason_summary": json.dumps(adr_meta_reasons, sort_keys=True),
            "pm_family": "NSGA-II",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "NSGA-II",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def _pymoo_dnsga2_failure(seed: int, error: str) -> Dict[str, object]:
    return {
        "status": "failed",
        "error": error,
        "seed": seed,
        "with_reaction": True,
        "migd": np.nan,
        "mhv": np.nan,
        "final_igd": np.nan,
        "final_hv": np.nan,
        "n_detected": 0,
        "n_true_changes": 0,
        "runtime_sec": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "mean_delay": np.nan,
        "env_changes": np.nan,
        "local_search_calls": 0,
        "memory_hits": 0,
        "memory_archive_size": 0,
        "pm_family": "pymoo-DNSGA2-B",
        "hv_hist": np.array([]),
        "igd_hist": np.array([]),
    }


def run_single_pymoo_dnsga2_official(
    problem_name: str,
    cfg: DNSGA2Config,
    seed: int,
    version: str = "B",
    perc_detect_change: float = 0.10,
) -> Dict[str, object]:
    if problem_name == DConstrainedZDT1.name:
        return _pymoo_dnsga2_failure(
            seed=seed,
            error="pymoo DNSGA2 is unconstrained; constrained benchmark intentionally excluded.",
        )

    try:
        cache_dir = os.path.join(REPO_ROOT, ".cache", "matplotlib")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)

        from pymoo.algorithms.moo.dnsga2 import DNSGA2
        from pymoo.core.population import Population
        from pymoo.core.problem import Problem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
    except Exception as exc:
        return _pymoo_dnsga2_failure(seed=seed, error=f"pymoo unavailable: {type(exc).__name__}: {exc}")

    class PymooDynamicProblem(Problem):
        def __init__(self, dynamic_problem):
            self.dynamic_problem = dynamic_problem
            super().__init__(
                n_var=cfg.n_variables,
                n_obj=int(getattr(dynamic_problem, "n_objectives", 2)),
                n_ieq_constr=0,
                xl=LOWER_BOUND,
                xu=UPPER_BOUND,
            )

        @property
        def t(self) -> int:
            return int(getattr(self.dynamic_problem, "t", 0))

        def _evaluate(self, X, out, *args, **kwargs):
            values = []
            for x in np.asarray(X, dtype=float):
                obj, cv = self.dynamic_problem.evaluate(np.clip(x, LOWER_BOUND, UPPER_BOUND))
                if float(cv) > 1e-10:
                    raise ValueError(
                        f"pymoo DNSGA2 wrapper received constrained point for {problem_name}; "
                        "run only unconstrained benchmark problems."
                    )
                values.append(np.asarray(obj, dtype=float))
            out["F"] = np.asarray(values, dtype=float)

    class TracingDNSGA2(DNSGA2):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.detected_change_steps: List[int] = []

        def _advance(self, **kwargs):
            pop = self.pop
            X, F = pop.get("X", "F")

            n_samples = int(np.ceil(len(pop) * self.perc_detect_change))
            I = self.random_state.choice(np.arange(len(pop)), size=n_samples)
            samples = self.evaluator.eval(self.problem, Population.new(X=X[I]))

            delta = ((samples.get("F") - F[I]) ** 2).mean()
            change_detected = delta > self.eps

            if change_detected:
                self.detected_change_steps.append(int(getattr(self.problem, "t", -1)))

                pop = Population.new(X=X)
                I = np.where(self.random_state.random(len(pop)) < self.perc_diversity)[0]

                if self.version == "A":
                    pop[I] = self.initialization.sampling(self.problem, len(I), random_state=self.random_state)
                elif self.version == "B":
                    self.mating.mutation(self.problem, pop[I])
                else:
                    raise ValueError(f"Unknown pymoo D-NSGA-II version: {self.version}")

                self.evaluator.eval(self.problem, pop)
                pop = self.survival.do(self.problem, pop, n_survive=len(pop), random_state=self.random_state)

            off = self.mating.do(self.problem, pop, self.n_offsprings, algorithm=self, random_state=self.random_state)
            self.evaluator.eval(self.problem, off)

            pop = Population.merge(pop, off)
            self.pop = self.survival.do(
                self.problem,
                pop,
                n_survive=self.pop_size,
                algorithm=self,
                random_state=self.random_state,
            )

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        pymoo_problem = PymooDynamicProblem(problem)
        algorithm = TracingDNSGA2(
            pop_size=cfg.pop_size,
            n_offsprings=cfg.pop_size,
            version=version,
            perc_detect_change=perc_detect_change,
            perc_diversity=cfg.response_ratio,
            eps=0.0,
            crossover=SBX(prob=cfg.crossover_prob, eta=cfg.sbx_eta),
            mutation=PM(eta=cfg.mutation_eta),
            eliminate_duplicates=True,
        )
        algorithm.setup(pymoo_problem, seed=seed, verbose=False)

        true_events = problem.true_change_points(cfg.horizon)
        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for _t in range(cfg.horizon):
            algorithm.next()
            f_mat = np.asarray(algorithm.pop.get("F"), dtype=float)
            front_obj = non_dominated_points(f_mat)
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))
            problem.update_time()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        detected_events = list(getattr(algorithm, "detected_change_steps", []))
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "pymoo-DNSGA2-B",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return _pymoo_dnsga2_failure(seed=seed, error=f"{type(exc).__name__}: {exc}")


def _sgea_dominance_fitness(f_mat: np.ndarray) -> np.ndarray:
    """Jiang-Yang SGEA fitness: number of population members dominating i."""
    n_points = int(f_mat.shape[0])
    fitness = np.zeros(n_points, dtype=int)
    for i in range(n_points):
        for j in range(n_points):
            if i != j and _deb_dominates(f_mat[j], f_mat[i]):
                fitness[i] += 1
    return fitness


def _sgea_objective_matrix(f_mat: np.ndarray) -> np.ndarray:
    obj = np.asarray(f_mat[:, 1:], dtype=float)
    if obj.size == 0:
        return obj
    mins = np.min(obj, axis=0)
    spans = np.maximum(np.max(obj, axis=0) - mins, 1e-12)
    return (obj - mins) / spans


def _sgea_spea2_truncate_indices(f_mat: np.ndarray, n_keep: int) -> np.ndarray:
    selected = list(range(f_mat.shape[0]))
    if len(selected) <= n_keep:
        return np.asarray(selected, dtype=int)

    obj = _sgea_objective_matrix(f_mat)
    while len(selected) > n_keep:
        best_pos = 0
        best_key: Tuple[float, ...] | None = None
        for pos, idx in enumerate(selected):
            distances = []
            for other in selected:
                if other != idx:
                    distances.append(float(np.linalg.norm(obj[idx] - obj[other])))
            distances.sort()
            key = tuple(distances)
            if best_key is None or key < best_key:
                best_key = key
                best_pos = pos
        del selected[best_pos]
    return np.asarray(selected, dtype=int)


def _sgea_farthest_first_indices(f_mat: np.ndarray, n_keep: int) -> np.ndarray:
    n_points = int(f_mat.shape[0])
    n_keep = int(min(max(n_keep, 0), n_points))
    if n_keep <= 0:
        return np.asarray([], dtype=int)
    if n_keep >= n_points:
        return np.arange(n_points, dtype=int)

    obj = _sgea_objective_matrix(f_mat)
    selected: List[int] = []
    for d in range(obj.shape[1]):
        candidate = int(np.argmin(obj[:, d]))
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= n_keep:
            return np.asarray(selected[:n_keep], dtype=int)

    if not selected:
        selected.append(0)

    remaining = set(range(n_points)) - set(selected)
    while len(selected) < n_keep and remaining:
        best_idx = None
        best_dist = -np.inf
        for idx in remaining:
            dist = float(np.min(np.linalg.norm(obj[idx][None, :] - obj[selected], axis=1)))
            if dist > best_dist:
                best_dist = dist
                best_idx = idx
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    return np.asarray(selected, dtype=int)


def _sgea_environment_selection(
    q_x: np.ndarray,
    q_f: np.ndarray,
    pop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fitness = _sgea_dominance_fitness(q_f)
    nondom_idx = np.where(fitness < 1)[0]
    archive_x = np.asarray(q_x[nondom_idx], dtype=float)
    archive_f = np.asarray(q_f[nondom_idx], dtype=float)

    if len(archive_x) < pop_size:
        order = np.lexsort((np.arange(len(fitness)), fitness))
        chosen = order[:pop_size]
        pop_x = np.asarray(q_x[chosen], dtype=float)
        pop_f = np.asarray(q_f[chosen], dtype=float)
    elif len(archive_x) == pop_size:
        pop_x = archive_x.copy()
        pop_f = archive_f.copy()
    else:
        keep_archive = _sgea_spea2_truncate_indices(archive_f, n_keep=pop_size)
        archive_x = archive_x[keep_archive]
        archive_f = archive_f[keep_archive]
        pop_x = archive_x.copy()
        pop_f = archive_f.copy()

    pop_fit = _sgea_dominance_fitness(pop_f)
    return pop_x, pop_f, pop_fit, archive_x, archive_f


def _sgea_tournament_index(fitness: np.ndarray, rng: np.random.Generator) -> int:
    n_points = int(len(fitness))
    if n_points <= 1:
        return 0
    i, j = rng.choice(n_points, size=2, replace=False)
    if fitness[i] < fitness[j]:
        return int(i)
    if fitness[j] < fitness[i]:
        return int(j)
    return int(i if rng.random() < 0.5 else j)


def _sgea_generate_offspring(
    pop: np.ndarray,
    pop_fit: np.ndarray,
    archive_x: np.ndarray,
    cfg: SGEAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    parent_1: np.ndarray
    parent_2: np.ndarray
    if len(archive_x) > 0 and rng.random() >= 0.5:
        parent_1 = np.asarray(archive_x[rng.integers(0, len(archive_x))], dtype=float)
        parent_2 = np.asarray(pop[_sgea_tournament_index(pop_fit, rng)], dtype=float)
    else:
        idx_1 = _sgea_tournament_index(pop_fit, rng)
        idx_2 = _sgea_tournament_index(pop_fit, rng)
        if pop.shape[0] > 1:
            attempts = 0
            while idx_2 == idx_1 and attempts < 8:
                idx_2 = _sgea_tournament_index(pop_fit, rng)
                attempts += 1
        parent_1 = np.asarray(pop[idx_1], dtype=float)
        parent_2 = np.asarray(pop[idx_2], dtype=float)

    child_1, child_2 = _sbx_crossover(
        parent_1,
        parent_2,
        rng=rng,
        eta=cfg.sbx_eta,
        crossover_prob=cfg.crossover_prob,
    )
    child = child_1 if rng.random() < 0.5 else child_2
    return _poly_mutation(
        child,
        rng=rng,
        eta=cfg.mutation_eta,
        mutation_prob=1.0 / max(cfg.n_variables, 1),
    )


def _sgea_archive_insert(
    archive_x: np.ndarray,
    archive_f: np.ndarray,
    child: np.ndarray,
    child_eval: np.ndarray,
    archive_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if archive_x.size == 0:
        return child[None, :].copy(), child_eval[None, :].copy()

    duplicate = np.any(np.all(np.isclose(archive_x, child[None, :], atol=1e-12, rtol=0.0), axis=1))
    if duplicate:
        return archive_x, archive_f

    dominated_by_child = np.array([_deb_dominates(child_eval, row) for row in archive_f], dtype=bool)
    child_dominated = any(_deb_dominates(row, child_eval) for row in archive_f)
    kept_x = archive_x[~dominated_by_child]
    kept_f = archive_f[~dominated_by_child]

    if not child_dominated:
        kept_x = np.vstack((kept_x, child[None, :]))
        kept_f = np.vstack((kept_f, child_eval[None, :]))

    if len(kept_x) > archive_size:
        keep = _sgea_spea2_truncate_indices(kept_f, n_keep=archive_size)
        kept_x = kept_x[keep]
        kept_f = kept_f[keep]
    return kept_x, kept_f


def _sgea_update_population(
    pop: np.ndarray,
    f_mat: np.ndarray,
    pop_fit: np.ndarray,
    archive_x: np.ndarray,
    archive_f: np.ndarray,
    child: np.ndarray,
    child_eval: np.ndarray,
    archive_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.any(np.all(np.isclose(pop, child[None, :], atol=1e-12, rtol=0.0), axis=1)):
        return pop, f_mat, pop_fit, archive_x, archive_f

    updated_fit = np.asarray(pop_fit, dtype=int).copy()
    child_fit = 0
    for idx in range(pop.shape[0]):
        if _deb_dominates(child_eval, f_mat[idx]):
            updated_fit[idx] += 1
        if _deb_dominates(f_mat[idx], child_eval):
            child_fit += 1

    worst_value = int(np.max(updated_fit))
    worst_candidates = np.where(updated_fit == worst_value)[0]
    worst_idx = int(rng.choice(worst_candidates))

    if child_fit <= updated_fit[worst_idx]:
        pop = pop.copy()
        f_mat = f_mat.copy()
        pop[worst_idx] = child
        f_mat[worst_idx] = child_eval
        updated_fit[worst_idx] = child_fit
        if child_fit < 1:
            archive_x, archive_f = _sgea_archive_insert(
                archive_x=archive_x,
                archive_f=archive_f,
                child=child,
                child_eval=child_eval,
                archive_size=archive_size,
            )

    return pop, f_mat, updated_fit, archive_x, archive_f


def _sgea_detect_change_by_reevaluation(
    problem,
    pop: np.ndarray,
    f_mat: np.ndarray,
    cfg: SGEAConfig,
    rng: np.random.Generator,
) -> bool:
    n_checks = max(1, int(np.ceil(cfg.detection_fraction * cfg.pop_size)))
    order = rng.permutation(cfg.pop_size)[:n_checks]
    for idx in order:
        cur_eval = _evaluate_population(problem, pop[int(idx)][None, :])[0]
        if _reevaluate_change(f_mat[int(idx)][None, :], cur_eval[None, :], tol=cfg.detect_tolerance):
            return True
    return False


def _sgea_nondominated_subset(pop: np.ndarray, f_mat: np.ndarray, archive_size: int) -> Tuple[np.ndarray, np.ndarray]:
    fitness = _sgea_dominance_fitness(f_mat)
    idx = np.where(fitness < 1)[0]
    out_x = np.asarray(pop[idx], dtype=float)
    out_f = np.asarray(f_mat[idx], dtype=float)
    if len(out_x) > archive_size:
        keep = _sgea_spea2_truncate_indices(out_f, n_keep=archive_size)
        out_x = out_x[keep]
        out_f = out_f[keep]
    return out_x, out_f


def _sgea_change_response_jy2017(
    pop: np.ndarray,
    f_mat: np.ndarray,
    archive_x: np.ndarray,
    problem,
    cfg: SGEAConfig,
    rng: np.random.Generator,
    previous_centroid: Optional[np.ndarray],
    archive_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    pop_size = int(pop.shape[0])
    n_keep = max(1, pop_size // 2)
    keep_idx = _sgea_farthest_first_indices(f_mat, n_keep=n_keep)
    keep_mask = np.zeros(pop_size, dtype=bool)
    keep_mask[keep_idx] = True

    retained_x = np.asarray(pop[keep_idx], dtype=float).copy()
    retained_f = _evaluate_population(problem, retained_x)
    new_archive_x, new_archive_f = _sgea_nondominated_subset(retained_x, retained_f, archive_size=archive_size)

    old_reference_x = archive_x if archive_x.size > 0 else retained_x
    current_centroid = np.mean(old_reference_x, axis=0) if len(old_reference_x) > 0 else np.mean(pop, axis=0)

    response_pop = pop.copy()
    response_eval = f_mat.copy()
    response_pop[keep_idx] = retained_x
    response_eval[keep_idx] = retained_f

    reinit_idx = np.where(~keep_mask)[0]
    first_change = previous_centroid is None or len(new_archive_x) == 0
    if first_change:
        response_pop[reinit_idx] = rng.random((len(reinit_idx), cfg.n_variables))
    else:
        retained_centroid = np.mean(retained_x, axis=0)
        archive_centroid = np.mean(new_archive_x, axis=0)
        direction = archive_centroid - retained_centroid
        norm = float(np.linalg.norm(direction))
        step_size = float(np.linalg.norm(current_centroid - previous_centroid))
        if norm <= 1e-12 or step_size <= 1e-12:
            response_pop[reinit_idx] = rng.random((len(reinit_idx), cfg.n_variables))
        else:
            direction = direction / norm
            sigma = step_size / (2.0 * np.sqrt(max(cfg.n_variables, 1)))
            noise = rng.normal(0.0, sigma, size=(len(reinit_idx), cfg.n_variables))
            response_pop[reinit_idx] = np.clip(pop[reinit_idx] + step_size * direction[None, :] + noise, LOWER_BOUND, UPPER_BOUND)

    if len(reinit_idx) > 0:
        response_eval[reinit_idx] = _evaluate_population(problem, response_pop[reinit_idx])
        for idx in reinit_idx:
            new_archive_x, new_archive_f = _sgea_archive_insert(
                archive_x=new_archive_x,
                archive_f=new_archive_f,
                child=response_pop[int(idx)],
                child_eval=response_eval[int(idx)],
                archive_size=archive_size,
            )

    return response_pop, response_eval, new_archive_x, new_archive_f, current_centroid, bool(first_change)


def run_single_sgea_2017(problem_name: str, cfg: SGEAConfig, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        ref = ref_point_for_problem(problem_name)
        archive_size = cfg.archive_size if cfg.archive_size > 0 else cfg.pop_size
        init_pop = rng.random((cfg.pop_size, cfg.n_variables))
        init_eval = _evaluate_population(problem, init_pop)
        pop, f_mat, pop_fit, archive_x, archive_f = _sgea_environment_selection(
            q_x=init_pop,
            q_f=init_eval,
            pop_size=cfg.pop_size,
        )

        oracle_events = _oracle_change(problem_name, cfg.horizon, cfg.n_variables)
        detected_events: List[int] = []
        previous_centroid: Optional[np.ndarray] = None
        random_first_responses = 0
        prev_eval = f_mat.copy()
        adr_meta_updates = 0
        adr_meta_reuses = 0
        adr_meta_drifts: List[float] = []
        adr_meta_reasons: Dict[str, int] = {}
        last_adr_meta_update_t: Optional[int] = None

        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            if t > 0:
                if cfg.detection_mode == "oracle":
                    changed = t in oracle_events
                elif cfg.detection_mode == "paper":
                    changed = _sgea_detect_change_by_reevaluation(problem, pop, f_mat, cfg, rng)
                else:
                    cur_eval = _evaluate_population(problem, pop)
                    changed = _reevaluate_change(prev_eval, cur_eval, tol=cfg.detect_tolerance)
                    f_mat = cur_eval

                if changed:
                    cur_eval = _evaluate_population(problem, pop)
                    should_update, drift, reason = _adr_meta_should_update(
                        cfg=cfg,
                        t=t,
                        previous_eval=prev_eval,
                        current_eval=cur_eval,
                        last_update_t=last_adr_meta_update_t,
                    )
                    f_mat = cur_eval
                    adr_meta_drifts.append(float(drift))
                    adr_meta_reasons[reason] = adr_meta_reasons.get(reason, 0) + 1
                    detected_events.append(t)
                    if should_update:
                        adr_meta_updates += int(bool(getattr(cfg, "adr_meta_enabled", False)))
                        last_adr_meta_update_t = t
                        pop, f_mat, archive_x, archive_f, previous_centroid, first_change = _sgea_change_response_jy2017(
                            pop=pop,
                            f_mat=f_mat,
                            archive_x=archive_x,
                            problem=problem,
                            cfg=cfg,
                            rng=rng,
                            previous_centroid=previous_centroid,
                            archive_size=archive_size,
                        )
                        random_first_responses += int(first_change)
                        pop_fit = _sgea_dominance_fitness(f_mat)
                    else:
                        adr_meta_reuses += 1
                        pop_fit = _sgea_dominance_fitness(f_mat)

            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            for _ in range(max(cfg.generations_per_step, 1)):
                generation_start_pop = pop.copy()
                generation_start_eval = f_mat.copy()
                for _offspring_idx in range(cfg.pop_size):
                    child = _sgea_generate_offspring(pop, pop_fit, archive_x, cfg, rng)
                    child_eval = _evaluate_population(problem, child[None, :])[0]
                    pop, f_mat, pop_fit, archive_x, archive_f = _sgea_update_population(
                        pop=pop,
                        f_mat=f_mat,
                        pop_fit=pop_fit,
                        archive_x=archive_x,
                        archive_f=archive_f,
                        child=child,
                        child_eval=child_eval,
                        archive_size=archive_size,
                        rng=rng,
                    )
                pop, f_mat, pop_fit, archive_x, archive_f = _sgea_environment_selection(
                    q_x=np.vstack((generation_start_pop, pop)),
                    q_f=np.vstack((generation_start_eval, f_mat)),
                    pop_size=cfg.pop_size,
                )
            prev_eval = f_mat.copy()
            problem.update_time()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        true_events = sorted(oracle_events)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detect_precision)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(len(detected_events)),
            "local_search_calls": 0,
            "memory_hits": int(len(detected_events) - random_first_responses),
            "memory_archive_size": int(len(archive_x)),
            "adr_meta_enabled": int(bool(getattr(cfg, "adr_meta_enabled", False))),
            "adr_meta_updates": int(adr_meta_updates),
            "adr_meta_reuses": int(adr_meta_reuses),
            "adr_meta_drift_mean": float(np.mean(adr_meta_drifts)) if adr_meta_drifts else np.nan,
            "adr_meta_reuse_ratio": float(adr_meta_reuses / max(len(detected_events), 1)),
            "adr_meta_reason_summary": json.dumps(adr_meta_reasons, sort_keys=True),
            "pm_family": "SGEA-JY2017",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "SGEA-JY2017",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def run_ga_baseline(problem_name: str, cfg: GAConfig, seed: int, algorithm_name: str) -> Dict[str, object]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    algorithms = {
        "dnsga2_b": DNSGAIIB,
        "mnsga2": MNSGAII,
        "memory_nsga2": MemoryImmigrantsNSGA2,
    }
    if algorithm_name not in algorithms:
        raise ValueError(f"Unsupported GA baseline '{algorithm_name}'.")

    try:
        algorithm = algorithms[algorithm_name](cfg)
        problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        ref = ref_point_for_problem(problem_name)
        pop = algorithm.initialize(rng)
        f_mat = _evaluate_population(problem, pop)

        detector = SentinelChangeDetector(
            n_sentinels=min(cfg.detector_sentinels, cfg.pop_size),
            alpha=cfg.detector_alpha,
            lambda_ewma=cfg.detector_lambda,
            k_sigma=cfg.detector_k_sigma,
            refresh_period=cfg.detector_refresh,
            cooldown=cfg.detector_cooldown,
            warmup=cfg.detector_warmup,
            history_size=cfg.detector_history,
        )
        detector.refresh(pop, f_mat, t=0)

        true_events = problem.true_change_points(cfg.horizon)
        detected_events: List[int] = []
        hv_hist = []
        igd_hist = []

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            if t > 0:
                det_info = detector.detect(problem, t=t)
                pop = np.clip(pop, LOWER_BOUND, UPPER_BOUND)
                f_mat = _evaluate_population(problem, pop)
                if det_info["detected"]:
                    detected_events.append(t)
                    pop = algorithm.on_change(pop, f_mat, rng, det_info)
                    f_mat = _evaluate_population(problem, pop)
            front_obj = non_dominated_points(np.asarray(f_mat[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)
            hv_hist.append(hypervolume(front_obj, ref=ref_point_for_problem(problem_name, problem, true_pf), rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            pop, f_mat = algorithm.evolve(pop, f_mat, problem, rng)
            if (t - detector.last_refresh) >= detector.refresh_period:
                detector.refresh(pop, f_mat, t=t)
            else:
                detector.refresh(pop, f_mat, t=t)
            problem.update_time()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected_events, tolerance=cfg.detector_tolerance)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": True,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
            "n_detected": int(len(detected_events)),
            "n_true_changes": int(len(true_events)),
            "runtime_sec": float(elapsed),
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "mean_delay": det["mean_delay"],
            "env_changes": int(algorithm.env_changes),
            "local_search_calls": 0,
            "memory_hits": int(algorithm.memory_hits),
            "memory_archive_size": int(sum(len(s.decisions) for s in algorithm.memory_archive)),
            "pm_family": "NSGA-II",
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "with_reaction": True,
            "migd": np.nan,
            "mhv": np.nan,
            "final_igd": np.nan,
            "final_hv": np.nan,
            "n_detected": 0,
            "n_true_changes": 0,
            "runtime_sec": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_delay": np.nan,
            "env_changes": np.nan,
            "local_search_calls": 0,
            "memory_hits": 0,
            "memory_archive_size": 0,
            "pm_family": "NSGA-II",
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


EDA_COMMON_OVERRIDES = {
    "algorithm_family": "speda",
    "pm_family": "SPBN",
    "local_search_enabled": True,
    "split_drift_signals": True,
    "adaptive_lambda_enabled": True,
    "adaptive_lambda_cov_enabled": True,
    "adaptive_cooldown_enabled": True,
    "adaptive_policy_enabled": True,
    "reaction_quality_gate_enabled": True,
    "cov_threshold_k": 2.8,
    "alert_cooldown": 1,
    "min_change_interval": 1,
    "memory_enabled": False,
    "diversity_block_enabled": True,
    "predictive_block_enabled": False,
    "continuity_block_enabled": False,
    "portfolio_mix_enabled": False,
    "change_type_enabled": False,
    "response_mode_enabled": False,
    "adr_enabled": False,
    "adr_front_ls_enabled": False,
    "adr_front_ls_context_gate_enabled": False,
    "adr_credit_enabled": False,
}


ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "dnsga2_b": AlgorithmSpec(
        name="dnsga2_b",
        runner="dnsga2_module",
        description="Dedicated DNSGA-II-B module with explicit version-B mutated-immigrant response after change.",
        implementation_kind="dedicated_module",
        faithfulness="paper_faithful_core_secondary_source",
        change_protocol="oracle_change_notification",
    ),
    "pymoo_dnsga2_b": AlgorithmSpec(
        name="pymoo_dnsga2_b",
        runner="pymoo_dnsga2",
        description="Official pymoo D-NSGA-II version B wrapped in the common dynamic benchmark metrics.",
        implementation_kind="official_library_wrapper",
        faithfulness="official_pymoo_dnsga2_version_b",
        change_protocol="pymoo_native_partial_reevaluation",
        implementation_source="pymoo",
        overrides={
            "version": "B",
            "perc_detect_change": 0.10,
            "perc_diversity": 0.20,
        },
    ),
    "memory_nsga2": AlgorithmSpec(
        name="memory_nsga2",
        runner="ga",
        description="Clean memory-immigrants NSGA-II: archive of diverse elites plus hypermutated immigrants.",
        implementation_kind="reference_impl",
        faithfulness="provisional_reference",
    ),
    "mnsga2": AlgorithmSpec(
        name="mnsga2",
        runner="ga",
        description="Sahmoud-Topcuoglu 2016 memory-based NSGA-II core with external non-dominated memory and memory-guided reinitialization after change.",
        implementation_kind="dedicated_module",
        faithfulness="paper_core_memory_based",
        change_protocol="paper_detection_memory_reinitialization",
    ),
    "sgea": AlgorithmSpec(
        name="sgea",
        runner="sgea_module",
        description="Jiang-Yang 2017 SGEA with author-code-audited environment selection, steady-state update and centroid-guided change response.",
        implementation_kind="author_code_audited_python_port",
        faithfulness="author_code_core_port_unified_protocol",
        change_protocol="paper_detection_reevaluate_fraction",
        implementation_source="author_code_audited_port",
        author_code_path=os.path.join("third_party", "author_code", "extracted", "SGEA"),
        author_code_status="author_provided_c_visual_studio_code_extracted_not_directly_executed",
    ),
    "trnsga2": AlgorithmSpec(
        name="trnsga2",
        runner="trnsga2_module",
        description="Jiang et al. (2018) Tr-NSGA-II with change-triggered Tr-IPG/TCA, optimized preimage and NSGA-II refinement.",
        implementation_kind="dedicated_module",
        faithfulness="paper_faithful_core_optimization_preimage",
        change_protocol="oracle_change_notification",
        overrides={
            "algorithm_family": "trnsga2_jiang2018",
            "transfer_samples": 200,
            "tca_dim": 20,
            "tca_sigma": 0.60,
            "tca_mu": 0.50,
            "preimage_neighbors": 8,
            "preimage_refine_steps": 12,
            "preimage_refine_candidates": 8,
            "refine_sigma_start": 0.10,
            "refine_sigma_decay": 0.65,
            "preimage_use_scipy": True,
            "preimage_optimizer_maxiter": 20,
            "detection_mode": "oracle",
            "transfer_when_no_explicit_events": False,
        },
    ),
    "pfpa": AlgorithmSpec(
        name="pfpa",
        runner="pfpa_module",
        description="Zhang et al. 2022 PFPA with multi-step, polynomial fitting, sampling response and local-linear RM-MEDA refinement.",
        implementation_kind="dedicated_module",
        faithfulness="paper_core_local_linear_rmeda",
        change_protocol="oracle_change_notification",
        overrides={
            "movement_steps": (0.3, 1.0),
            "compression_ratio": 0.10,
            "polynomial_max_degree": 6,
            "sampling_n2_factor": 1,
            "rmeda_model_noise": 0.035,
            "rmeda_latent_scale": 0.18,
            "detection_mode": "oracle",
        },
    ),
    "tda_dmoea": AlgorithmSpec(
        name="tda_dmoea",
        runner="tda_module",
        description="Cai et al. 2024 TDA-DMOEA core with Wilcoxon/HV type detection, LPS/MA-TL/KPP responses and MOEA/D-DE SMOA.",
        implementation_kind="dedicated_module",
        faithfulness="paper_core_moead_smoa_approx",
        change_protocol="oracle_change_notification_plus_type_classifier",
        overrides={
            "pos_p_threshold": 0.05,
            "pof_threshold_2obj": 0.008,
            "pof_threshold_3obj": 0.03,
            "knee_ratio": 0.35,
            "transfer_pool_samples": 160,
            "ma_tl_noise": 0.04,
            "detection_mode": "oracle",
        },
    ),
    "ab_dmoea": AlgorithmSpec(
        name="ab_dmoea",
        runner="ab_module",
        description="Peng et al. 2024 AB-DMOEA with author-code-audited dominated reinforcement, adaptive response boosting and compact MOEA/D-DE/RM-MEDA/NSGA-II-DE static boosting.",
        implementation_kind="author_code_audited_python_port",
        faithfulness="author_code_algorithmic_port_compact_static_boosters",
        change_protocol="oracle_change_notification_plus_adaptive_response",
        implementation_source="author_code_audited_port",
        author_code_path=os.path.join("third_party", "author_code", "extracted", "AB-DMOEA"),
        author_code_status="author_provided_matlab_platemo_code_extracted_not_directly_executed",
        overrides={
            "reinforcement_threshold": 0.70,
            "boost_alpha": 0.50,
            "min_strategy_prob": 0.08,
            "memory_similarity_eps": 0.05,
            "memory_size": 100,
            "de_f": 0.50,
            "de_cr": 0.90,
            "detection_mode": "oracle",
        },
    ),
    "adr_dmoea": AlgorithmSpec(
        name="adr_dmoea",
        runner="adr_dmoea_module",
        description="Wang et al. 2026 ADR-DMOEA with subpopulation-level ADRM, static-optimization-distance feedback and MOEA/D-DE refinement.",
        implementation_kind="dedicated_module",
        faithfulness="paper_core_adaptive_dynamic_response_candidate",
        change_protocol="oracle_change_notification_plus_adaptive_response",
        overrides={
            "subpop_size": 10,
            "gamma": 0.20,
            "theta_min": 0.10,
            "diversity_poly_prob": 0.50,
            "diversity_gaussian_sigma": 0.035,
            "prediction_neighbors": 5,
            "prediction_random_factor": 2,
            "prediction_pca_dim": 0,
            "memory_similarity_eps": 1e-4,
            "memory_size": 50,
            "memory_noise": 0.05,
            "de_f": 0.50,
            "de_cr": 1.0,
            "detection_mode": "oracle",
        },
    ),
    "adr_eda": AlgorithmSpec(
        name="adr_eda",
        runner="speda",
        description="ADR-EDA v2.1 frozen benchmark-side profile with predictive, continuity and diversity blocks plus contextual front-LS gating.",
        implementation_kind="framework_wrapper",
        faithfulness="frozen_profile",
        change_protocol="internal_adr_detector_frozen_profile",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_V2_1_FROZEN_OVERRIDES,
        },
    ),
    "adr_eda_contextual_weights": AlgorithmSpec(
        name="adr_eda_contextual_weights",
        runner="speda",
        description="Experimental ADR-EDA profile with a continuous contextual response-weight scheduler over intensity, abruptness, reliability, predictability and diversity deficit.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_contextual_weight_scheduler",
        change_protocol="internal_adr_detector_continuous_contextual_weight_policy",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_CONTEXTUAL_WEIGHTS_OVERRIDES,
        },
    ),
    "adr_eda_contextual_weights_v2": AlgorithmSpec(
        name="adr_eda_contextual_weights_v2",
        runner="speda",
        description="Conservative contextual ADR-EDA scheduler that penalizes continuity under abrupt/intense changes and lowers mandatory diversity.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_contextual_weight_scheduler_v2",
        change_protocol="internal_adr_detector_conservative_contextual_weight_policy",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_CONTEXTUAL_WEIGHTS_V2_OVERRIDES,
        },
    ),
    "adr_eda_contextual_weights_v3": AlgorithmSpec(
        name="adr_eda_contextual_weights_v3",
        runner="speda",
        description="Recovery-guarded contextual ADR-EDA scheduler with bounded deviation from the frozen ADR prior.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_contextual_weight_scheduler_v3",
        change_protocol="internal_adr_detector_recovery_guarded_contextual_weight_policy",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_CONTEXTUAL_WEIGHTS_V3_OVERRIDES,
        },
    ),
    "adr_eda_geometric_response_policy": AlgorithmSpec(
        name="adr_eda_geometric_response_policy",
        runner="speda",
        description="Experimental ADR-EDA scheduler that derives response weights from measured POS/POF movement, overlap, shape change and displacement coherence.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_geometry_response_scheduler",
        change_protocol="internal_adr_detector_geometry_driven_weight_policy",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_OVERRIDES,
        },
    ),
    "adr_eda_geometric_response_policy_v2": AlgorithmSpec(
        name="adr_eda_geometric_response_policy_v2",
        runner="speda",
        description="Reliability-guarded geometry scheduler that penalizes continuity when POS movement is high and POF displacement coherence is low.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_geometry_response_scheduler_v2",
        change_protocol="internal_adr_detector_geometry_policy_continuity_risk_guard",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V2_OVERRIDES,
        },
    ),
    "adr_eda_geometric_response_policy_v3": AlgorithmSpec(
        name="adr_eda_geometric_response_policy_v3",
        runner="speda",
        description="POS/POF risk-aware geometry scheduler with stronger continuity/prediction caps under high POS transfer risk and bounded front-local-search.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_geometry_response_scheduler_v3",
        change_protocol="internal_adr_detector_geometry_policy_pos_pof_risk_matrix",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_V3_OVERRIDES,
        },
    ),
    "adr_eda_pos_pof_response_policy_v1": AlgorithmSpec(
        name="adr_eda_pos_pof_response_policy_v1",
        runner="speda",
        description="Explicit POS/POF response scheduler: POS controls inherited-state trust, POF controls front repair/exploitation pressure.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_pos_pof_response_policy_v1",
        change_protocol="internal_adr_detector_pos_pof_response_matrix",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_POS_POF_RESPONSE_POLICY_V1_OVERRIDES,
        },
    ),
    "adr_eda_pos_pof_response_policy_v2": AlgorithmSpec(
        name="adr_eda_pos_pof_response_policy_v2",
        runner="speda",
        description="Bounded-prior POS/POF response scheduler: regime-aware hidden-transfer guard with limited deviation from frozen ADR weights.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_pos_pof_response_policy_v2",
        change_protocol="internal_adr_detector_bounded_pos_pof_response_matrix",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_POS_POF_RESPONSE_POLICY_V2_OVERRIDES,
        },
    ),
    "adr_eda_block_optimized_response_v1": AlgorithmSpec(
        name="adr_eda_block_optimized_response_v1",
        runner="speda",
        description="Experimental ADR-EDA with block-optimized response: bounded POS/POF scheduler, SGEA-style prediction, diverse continuity, structured diversity and capped front-LS.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_block_optimized_response_v1",
        change_protocol="internal_adr_detector_block_optimized_pos_pof_response",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_BLOCK_OPTIMIZED_RESPONSE_V1_OVERRIDES,
        },
    ),
    "adr_eda_geometric_trust_delta": AlgorithmSpec(
        name="adr_eda_geometric_trust_delta",
        runner="speda",
        description="Conservative POS/POF geometry scheduler: frozen ADR weights as prior plus a bounded trust-weighted delta.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_geometry_trust_delta_scheduler",
        change_protocol="internal_adr_detector_geometry_trust_delta_weight_policy",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_GEOMETRIC_TRUST_DELTA_OVERRIDES,
        },
    ),
    'adr_eda_geometric_gated_prior': AlgorithmSpec(
        name='adr_eda_geometric_gated_prior',
        runner='speda',
        description='Gated-prior geometry scheduler: frozen ADR weights plus bounded POS/POF correction only under sufficient confidence, preserving front-local-search mass.',
        implementation_kind='framework_wrapper',
        faithfulness='experimental_geometry_gated_prior_scheduler',
        change_protocol='internal_adr_detector_geometry_gated_prior_weight_policy',
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_GEOMETRIC_GATED_PRIOR_OVERRIDES,
        },
    ),
    "adr_eda_ml_weights": AlgorithmSpec(
        name="adr_eda_ml_weights",
        runner="speda",
        description="Experimental ADR-EDA scheduler using a contextual LinUCB bandit to select bounded response-weight experts over the frozen ADR prior.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_ml_weight_scheduler",
        change_protocol="internal_adr_detector_ml_contextual_bandit_weight_policy",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_ML_WEIGHTS_OVERRIDES,
        },
    ),
    "adr_eda_reliability_diagnostics_v1": AlgorithmSpec(
        name="adr_eda_reliability_diagnostics_v1",
        runner="speda",
        description="Experimental diagnostics-only ADR-EDA profile: frozen response with inherited-state reliability, gate and switch-cost trace fields.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_diagnostics_only",
        change_protocol="internal_adr_detector_frozen_profile_diagnostics_only",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_RELIABILITY_DIAGNOSTICS_V1_OVERRIDES,
        },
    ),
    "adr_eda_adaptive_weights_v1": AlgorithmSpec(
        name="adr_eda_adaptive_weights_v1",
        runner="speda",
        description="Experimental reliability-aware ADR-EDA profile with adaptive response weights and diversity-preserving continuity.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_reliability_weight_scheduler",
        change_protocol="internal_adr_detector_reliability_adaptive_weights",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_ADAPTIVE_WEIGHTS_V1_OVERRIDES,
        },
    ),
    "adr_eda_adaptive_weights_front_ls_v1": AlgorithmSpec(
        name="adr_eda_adaptive_weights_front_ls_v1",
        runner="speda",
        description="Experimental reliability-aware ADR-EDA profile with bounded front-local-search exploitation during post-change response.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_front_local_search_ablation",
        change_protocol="internal_adr_detector_reliability_adaptive_weights_front_local_search",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_ADAPTIVE_WEIGHTS_FRONT_LS_V1_OVERRIDES,
        },
    ),
    "adr_eda_adaptive_weights_switch_gate_v1": AlgorithmSpec(
        name="adr_eda_adaptive_weights_switch_gate_v1",
        runner="speda",
        description="Experimental reliability-aware ADR-EDA profile with adaptive weights, diverse continuity and switching-cost-aware reaction gate.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_switching_cost_gate",
        change_protocol="internal_adr_detector_reliability_adaptive_weights_switch_gate",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_ADAPTIVE_WEIGHTS_SWITCH_GATE_V1_OVERRIDES,
        },
    ),
    "adr_eda_geometric_response_policy_front_ls_v1": AlgorithmSpec(
        name="adr_eda_geometric_response_policy_front_ls_v1",
        runner="speda",
        description="Reliability-guarded geometry scheduler with a small bounded front-local-search budget for coherent/moderate post-change refinement.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_geometry_front_local_search_ablation",
        change_protocol="internal_adr_detector_geometry_policy_front_local_search",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_GEOMETRIC_RESPONSE_POLICY_FRONT_LS_V1_OVERRIDES,
        },
    ),
    "adr_eda_sgea_reaction": AlgorithmSpec(
        name="adr_eda_sgea_reaction",
        runner="speda",
        description="Experimental ADR-EDA profile that keeps the frozen detector/model but shifts reaction toward SGEA-like retention, relocation and lower random diversity.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_variant_not_primary_baseline",
        change_protocol="internal_adr_detector_sgea_inspired_profile",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_SGEA_REACTION_OVERRIDES,
        },
    ),
    "adr_eda_sgea_prediction": AlgorithmSpec(
        name="adr_eda_sgea_prediction",
        runner="speda",
        description="ADR-EDA profile that keeps the frozen architecture but replaces the predictive block with SGEA-style centroid relocation.",
        implementation_kind="framework_wrapper",
        faithfulness="clean_predictive_ablation",
        change_protocol="internal_adr_detector_sgea_predictive_block",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_SGEA_PREDICTION_OVERRIDES,
        },
    ),
    "adr_eda_fast_reaction": AlgorithmSpec(
        name="adr_eda_fast_reaction",
        runner="speda",
        description="ADR-EDA diagnostic variant with a more sensitive internal detector and lighter reaction throttling.",
        implementation_kind="framework_wrapper",
        faithfulness="detector_reaction_ablation",
        change_protocol="internal_adr_detector_fast_reaction",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_FAST_REACTION_OVERRIDES,
        },
    ),
    "adr_eda_regime_unified": AlgorithmSpec(
        name="adr_eda_regime_unified",
        runner="speda",
        description="Experimental ADR-EDA variant with a unified regime-aware detector over active-front benchmark features and shared score/trigger logic.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_unified_regime_detector",
        change_protocol="regime_aware_detector_shared_scoring_logic",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_REGIME_UNIFIED_OVERRIDES,
        },
    ),
    "adr_eda_regime_unified_balanced": AlgorithmSpec(
        name="adr_eda_regime_unified_balanced",
        runner="speda",
        description="Balanced retuning of the unified regime-aware ADR-EDA detector, with earlier rearming and lower benchmark-side trigger thresholds.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_unified_regime_detector_tuned_balanced",
        change_protocol="regime_aware_detector_shared_scoring_logic_tuned",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_REGIME_UNIFIED_BALANCED_OVERRIDES,
        },
    ),
    "adr_eda_regime_unified_sensitive": AlgorithmSpec(
        name="adr_eda_regime_unified_sensitive",
        runner="speda",
        description="Sensitive retuning of the unified regime-aware ADR-EDA detector for recall-oriented screening.",
        implementation_kind="framework_wrapper",
        faithfulness="experimental_unified_regime_detector_tuned_sensitive",
        change_protocol="regime_aware_detector_shared_scoring_logic_sensitive",
        overrides={
            **EDA_COMMON_OVERRIDES,
            **ADR_EDA_REGIME_UNIFIED_SENSITIVE_OVERRIDES,
        },
    ),
}


for _adr_meta_base in ("dnsga2_b", "sgea", "trnsga2", "pfpa", "tda_dmoea", "ab_dmoea"):
    _base_spec = ALGORITHMS[_adr_meta_base]
    ALGORITHMS[f"adr_meta_{_adr_meta_base}"] = AlgorithmSpec(
        name=f"adr_meta_{_adr_meta_base}",
        runner=_base_spec.runner,
        description=(
            "ADR meta-controller over "
            + _base_spec.name
            + ": same optimizer and dynamic response, but complete response calls are gated by "
            + "objective-drift, minimum interval and maximum interval reuse logic."
        ),
        implementation_kind="meta_controller_wrapper",
        faithfulness="experimental_adr_meta_control_probe",
        change_protocol=f"adr_meta_control_over_{_base_spec.change_protocol}",
        implementation_source=_base_spec.implementation_source,
        author_code_path=_base_spec.author_code_path,
        author_code_status=_base_spec.author_code_status,
        overrides={
            **_base_spec.overrides,
            **ADR_META_DEFAULT_OVERRIDES,
        },
    )


EXPERIMENTAL_ALGORITHMS = {
    "pymoo_dnsga2_b",
    "adr_meta_dnsga2_b",
    "adr_meta_sgea",
    "adr_meta_trnsga2",
    "adr_meta_pfpa",
    "adr_meta_tda_dmoea",
    "adr_meta_ab_dmoea",
    "adr_dmoea",
    "adr_eda_reliability_diagnostics_v1",
    "adr_eda_ml_weights",
    "adr_eda_adaptive_weights_v1",
    "adr_eda_adaptive_weights_front_ls_v1",
    "adr_eda_adaptive_weights_switch_gate_v1",
    "adr_eda_contextual_weights",
    "adr_eda_contextual_weights_v2",
    "adr_eda_contextual_weights_v3",
    "adr_eda_fast_reaction",
    "adr_eda_geometric_trust_delta",
    'adr_eda_geometric_gated_prior',
    "adr_eda_geometric_response_policy",
    "adr_eda_geometric_response_policy_v2",
    "adr_eda_geometric_response_policy_v3",
    "adr_eda_pos_pof_response_policy_v1",
    "adr_eda_pos_pof_response_policy_v2",
    "adr_eda_block_optimized_response_v1",
    "adr_eda_geometric_response_policy_front_ls_v1",
    "adr_eda_regime_unified",
    "adr_eda_regime_unified_balanced",
    "adr_eda_regime_unified_sensitive",
    "adr_eda_sgea_reaction",
    "adr_eda_sgea_prediction",
    "memory_nsga2",
}


METRICS = [
    "migd",
    "mhv",
    "final_igd",
    "final_hv",
    "post_change_igd",
    "post_change_hv",
    "recovery_time",
    "precision",
    "recall",
    "f1",
    "runtime_sec",
    "n_detected",
    "env_changes",
    "local_search_calls",
    "memory_hits",
    "memory_archive_size",
    "adr_meta_enabled",
    "adr_meta_updates",
    "adr_meta_reuses",
    "adr_meta_reuse_ratio",
    "adr_meta_drift_mean",
    "reaction_quality_reverted_count",
    "reaction_predictive_samples",
    "reaction_continuity_samples",
    "reaction_diversity_samples",
    "reaction_front_ls_samples",
    "reaction_adr_weight_count",
    "reaction_adr_predictive_weight_mean",
    "reaction_adr_continuity_weight_mean",
    "reaction_adr_diversity_weight_mean",
    "reaction_adr_front_ls_weight_mean",
    "inherited_state_reliability_mean",
    "reaction_gate_reject_ratio_mean",
    "reaction_switch_cost_mean",
    "reaction_front_improvement_mean",
    "reaction_required_improvement_mean",
    "reaction_switching_gate_rejected_count",
    "reaction_continuity_diverse_count",
    "reaction_geometry_ready_count",
    "reaction_geometry_front_shift_mean",
    "reaction_geometry_pos_shift_mean",
    "reaction_geometry_overlap_mean",
    "reaction_geometry_shape_change_mean",
    "reaction_geometry_coherence_mean",
    "reaction_geometry_pos_change_mean",
    "reaction_geometry_pof_change_mean",
    "reaction_geometry_negative_transfer_mean",
    "reaction_geometry_coherent_motion_mean",
    "reaction_geometry_global_change_mean",
    "reaction_geometry_pof_stability_mean",
    "reaction_geometry_tau_mean",
    "reaction_geometry_continuity_risk_mean",
    "reaction_geometry_continuity_removed_mean",
    "reaction_geometry_transfer_predictive_share_mean",
]


def parse_algorithm_list(raw: str) -> List[str]:
    requested = [v.strip() for v in raw.split(",") if v.strip()]
    if not requested or (len(requested) == 1 and requested[0].lower() == "all"):
        return [name for name in ALGORITHMS.keys() if name not in EXPERIMENTAL_ALGORITHMS]
    if len(requested) == 1 and requested[0].lower() in {"all_plus_experimental", "all_experimental"}:
        return list(ALGORITHMS.keys())
    selected = []
    for item in requested:
        if item in ALGORITHMS and item not in selected:
            selected.append(item)
    return selected


def parse_problem_list(raw: str) -> List[str]:
    requested = [v.strip() for v in raw.split(",") if v.strip()]
    if not requested or (len(requested) == 1 and requested[0].lower() == "all"):
        return list(CANONICAL_PROBLEM_NAMES)

    groups = {
        "canonical": CANONICAL_PROBLEM_NAMES,
        "df": CEC2018_DF_NAMES,
        "df_all": CEC2018_DF_NAMES,
        "cec2018_df": CEC2018_DF_NAMES,
        "df_t10": CEC2018_DF_T10_NAMES,
        "df_fast": CEC2018_DF_T10_NAMES,
        "df_t30": CEC2018_DF_T30_NAMES,
        "df_slow": CEC2018_DF_T30_NAMES,
        "all_plus_df": CANONICAL_PROBLEM_NAMES + CEC2018_DF_NAMES,
    }
    selected: List[str] = []
    for item in requested:
        key = item.lower()
        names = groups.get(key, [item])
        for name in names:
            if name in PROBLEM_REGISTRY and name not in selected:
                selected.append(name)
    return selected


def _repo_abs_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def algorithm_metadata(spec: AlgorithmSpec) -> Dict[str, object]:
    author_path = _repo_abs_path(spec.author_code_path)
    return {
        "name": spec.name,
        "runner": spec.runner,
        "description": spec.description,
        "implementation_kind": spec.implementation_kind,
        "faithfulness": spec.faithfulness,
        "change_protocol": spec.change_protocol,
        "implementation_source": spec.implementation_source,
        "author_code_path": spec.author_code_path,
        "author_code_abs_path": author_path,
        "author_code_status": spec.author_code_status,
        "author_code_available": bool(author_path and os.path.exists(author_path)),
    }


def run_algorithm(
    algorithm_name: str,
    problem_name: str,
    speda_cfg_base: SimpleNamespace,
    ga_cfg: GAConfig,
    seed: int,
) -> Dict[str, object]:
    spec = ALGORITHMS[algorithm_name]
    overrides = spec.overrides
    adr_meta_kwargs = _adr_meta_kwargs(overrides)
    if spec.runner == "pymoo_dnsga2":
        dnsga_cfg = DNSGA2Config(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            crossover_prob=ga_cfg.crossover_prob,
            sbx_eta=ga_cfg.sbx_eta,
            mutation_eta=ga_cfg.mutation_eta,
            response_ratio=float(overrides.get("perc_diversity", ga_cfg.dnsga_immigrant_ratio)),
            change_mutation_prob=1.0,
            change_mutation_eta=ga_cfg.dnsga_hypermutation_eta,
            detection_mode="native_pymoo",
            detect_precision=ga_cfg.detector_tolerance,
        )
        return run_single_pymoo_dnsga2_official(
            problem_name=problem_name,
            cfg=dnsga_cfg,
            seed=seed,
            version=str(overrides.get("version", "B")),
            perc_detect_change=float(overrides.get("perc_detect_change", 0.10)),
        )
    if spec.runner == "dnsga2_module":
        dnsga_cfg = DNSGA2Config(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            crossover_prob=ga_cfg.crossover_prob,
            sbx_eta=ga_cfg.sbx_eta,
            mutation_eta=ga_cfg.mutation_eta,
            response_ratio=ga_cfg.dnsga_immigrant_ratio,
            change_mutation_prob=1.0,
            change_mutation_eta=ga_cfg.dnsga_hypermutation_eta,
            detection_mode="oracle",
            detect_precision=ga_cfg.detector_tolerance,
            **adr_meta_kwargs,
        )
        return run_single_dnsga2_2007(problem_name=problem_name, cfg=dnsga_cfg, seed=seed)
    if spec.runner == "sgea_module":
        sgea_cfg = SGEAConfig(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            crossover_prob=1.0,
            sbx_eta=ga_cfg.sbx_eta,
            mutation_eta=ga_cfg.mutation_eta,
            archive_size=ga_cfg.pop_size,
            detection_mode="paper",
            detection_fraction=0.10,
            detect_precision=ga_cfg.detector_tolerance,
            **adr_meta_kwargs,
        )
        return run_single_sgea_2017(problem_name=problem_name, cfg=sgea_cfg, seed=seed)
    if spec.runner == "trnsga2_module":
        tr_cfg = TrNSGA2Config(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            crossover_prob=ga_cfg.crossover_prob,
            sbx_eta=ga_cfg.sbx_eta,
            mutation_eta=ga_cfg.mutation_eta,
            transfer_samples=int(overrides.get("transfer_samples", max(200, ga_cfg.pop_size))),
            tca_dim=int(overrides.get("tca_dim", 20)),
            tca_sigma=float(overrides.get("tca_sigma", 0.60)),
            tca_mu=float(overrides.get("tca_mu", 0.50)),
            preimage_neighbors=int(overrides.get("preimage_neighbors", 8)),
            preimage_refine_steps=int(overrides.get("preimage_refine_steps", 12)),
            preimage_refine_candidates=int(overrides.get("preimage_refine_candidates", 8)),
            refine_sigma_start=float(overrides.get("refine_sigma_start", 0.10)),
            refine_sigma_decay=float(overrides.get("refine_sigma_decay", 0.65)),
            preimage_use_scipy=bool(overrides.get("preimage_use_scipy", True)),
            preimage_optimizer_maxiter=int(overrides.get("preimage_optimizer_maxiter", 20)),
            detection_mode=str(overrides.get("detection_mode", "oracle")),
            transfer_when_no_explicit_events=bool(overrides.get("transfer_when_no_explicit_events", False)),
            detect_precision=ga_cfg.detector_tolerance,
            **adr_meta_kwargs,
        )
        return run_single_trnsga2_2018(problem_name=problem_name, cfg=tr_cfg, seed=seed)
    if spec.runner == "pfpa_module":
        pfpa_cfg = PFPAConfig(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            movement_steps=tuple(overrides.get("movement_steps", (0.3, 1.0))),
            compression_ratio=float(overrides.get("compression_ratio", 0.10)),
            polynomial_max_degree=int(overrides.get("polynomial_max_degree", 6)),
            sampling_n2_factor=int(overrides.get("sampling_n2_factor", 1)),
            rmeda_model_noise=float(overrides.get("rmeda_model_noise", 0.035)),
            rmeda_latent_scale=float(overrides.get("rmeda_latent_scale", 0.18)),
            detection_mode=str(overrides.get("detection_mode", "oracle")),
            detect_precision=ga_cfg.detector_tolerance,
            **adr_meta_kwargs,
        )
        return run_single_pfpa_2022(problem_name=problem_name, cfg=pfpa_cfg, seed=seed)
    if spec.runner == "tda_module":
        tda_cfg = TDADMOEAConfig(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            crossover_prob=ga_cfg.crossover_prob,
            sbx_eta=ga_cfg.sbx_eta,
            mutation_eta=ga_cfg.mutation_eta,
            pos_p_threshold=float(overrides.get("pos_p_threshold", 0.05)),
            pof_threshold_2obj=float(overrides.get("pof_threshold_2obj", 0.008)),
            pof_threshold_3obj=float(overrides.get("pof_threshold_3obj", 0.03)),
            knee_ratio=float(overrides.get("knee_ratio", 0.35)),
            transfer_pool_samples=int(overrides.get("transfer_pool_samples", 160)),
            ma_tl_noise=float(overrides.get("ma_tl_noise", 0.04)),
            detection_mode=str(overrides.get("detection_mode", "oracle")),
            detect_precision=ga_cfg.detector_tolerance,
            **adr_meta_kwargs,
        )
        return run_single_tda_dmoea_2024(problem_name=problem_name, cfg=tda_cfg, seed=seed)
    if spec.runner == "ab_module":
        ab_cfg = ABDMOEAConfig(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            crossover_prob=ga_cfg.crossover_prob,
            sbx_eta=ga_cfg.sbx_eta,
            mutation_eta=ga_cfg.mutation_eta,
            reinforcement_threshold=float(overrides.get("reinforcement_threshold", 0.70)),
            boost_alpha=float(overrides.get("boost_alpha", 0.50)),
            min_strategy_prob=float(overrides.get("min_strategy_prob", 0.08)),
            memory_similarity_eps=float(overrides.get("memory_similarity_eps", 0.05)),
            memory_size=int(overrides.get("memory_size", 100)),
            de_f=float(overrides.get("de_f", 0.50)),
            de_cr=float(overrides.get("de_cr", 0.90)),
            detection_mode=str(overrides.get("detection_mode", "oracle")),
            detect_precision=ga_cfg.detector_tolerance,
            **adr_meta_kwargs,
        )
        return run_single_ab_dmoea_2024(problem_name=problem_name, cfg=ab_cfg, seed=seed)
    if spec.runner == "adr_dmoea_module":
        adr_dmoea_cfg = ADRDMOEAConfig(
            pop_size=ga_cfg.pop_size,
            n_variables=ga_cfg.n_variables,
            horizon=ga_cfg.horizon,
            generations_per_step=ga_cfg.generations_per_step,
            crossover_prob=ga_cfg.crossover_prob,
            sbx_eta=ga_cfg.sbx_eta,
            mutation_eta=ga_cfg.mutation_eta,
            subpop_size=int(overrides.get("subpop_size", 10)),
            gamma=float(overrides.get("gamma", 0.20)),
            theta_min=float(overrides.get("theta_min", 0.10)),
            diversity_poly_prob=float(overrides.get("diversity_poly_prob", 0.50)),
            diversity_gaussian_sigma=float(overrides.get("diversity_gaussian_sigma", 0.035)),
            prediction_neighbors=int(overrides.get("prediction_neighbors", 5)),
            prediction_random_factor=int(overrides.get("prediction_random_factor", 2)),
            prediction_pca_dim=int(overrides.get("prediction_pca_dim", 0)),
            memory_similarity_eps=float(overrides.get("memory_similarity_eps", 1e-4)),
            memory_size=int(overrides.get("memory_size", 50)),
            memory_noise=float(overrides.get("memory_noise", 0.05)),
            de_f=float(overrides.get("de_f", 0.50)),
            de_cr=float(overrides.get("de_cr", 1.0)),
            detection_mode=str(overrides.get("detection_mode", "oracle")),
            detect_precision=ga_cfg.detector_tolerance,
            **adr_meta_kwargs,
        )
        return run_single_adr_dmoea_2026(problem_name=problem_name, cfg=adr_dmoea_cfg, seed=seed)
    if spec.runner == "ga":
        return run_ga_baseline(problem_name, cfg=ga_cfg, seed=seed, algorithm_name=algorithm_name)
    if spec.runner == "speda":
        backbone, error = _load_speda_backbone()
        if backbone is None:
            return {
                "status": "failed",
                "error": f"EDA backbone unavailable: {error}",
                "seed": seed,
                "with_reaction": True,
                "migd": np.nan,
                "mhv": np.nan,
                "final_igd": np.nan,
                "final_hv": np.nan,
                "n_detected": 0,
                "n_true_changes": 0,
                "runtime_sec": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "mean_delay": np.nan,
                "env_changes": np.nan,
                "local_search_calls": np.nan,
                "memory_hits": np.nan,
                "memory_archive_size": np.nan,
                "pm_family": spec.overrides.get("pm_family", "SPBN"),
                "hv_hist": np.array([]),
                "igd_hist": np.array([]),
            }
        SpedaRobustConfig = backbone["SpedaRobustConfig"]
        run_speda_variant = backbone["run_speda_variant"]
        tuned_cfg_for_problem = backbone["tuned_cfg_for_problem"]
        speda_problem_registry = backbone["PROBLEM_REGISTRY"]
        cfg = SpedaRobustConfig(
            n_variables=speda_cfg_base.n_variables,
            size_gen=speda_cfg_base.size_gen,
            horizon=speda_cfg_base.horizon,
        )
        cfg = replace(tuned_cfg_for_problem(cfg, problem_name), **spec.overrides)
        if problem_name not in speda_problem_registry:
            return _run_speda_variant_with_local_problem(
                problem_name=problem_name,
                cfg=cfg,
                seed=seed,
                with_reaction=True,
                create_speda_fn=backbone["create_speda"],
                call_react_fn=backbone["_call_react"],
            )
        return run_speda_variant(problem_name, cfg=cfg, seed=seed, with_reaction=True)
    raise ValueError(f"Unsupported runner type '{spec.runner}'.")


def _finite(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = _finite(values)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def summarize_by_algorithm(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(str(row["algorithm"]), []).append(row)

    for algorithm, block in grouped.items():
        summary[algorithm] = {
            "n_ok": int(len(block)),
            **{metric: mean_std([r.get(metric, np.nan) for r in block]) for metric in METRICS},
        }
    return summary


def compare_vs_reference(rows: List[Dict[str, object]], reference_name: str = "adr_eda") -> Dict[str, Dict[str, object]]:
    key = lambda r: (str(r["problem"]), int(r["seed"]))
    ref_rows = {
        key(r): r
        for r in rows
        if r.get("status") == "ok" and str(r.get("algorithm")) == reference_name
    }

    metrics = [
        ("migd", "less"),
        ("mhv", "greater"),
        ("post_change_igd", "less"),
        ("post_change_hv", "greater"),
        ("recovery_time", "less"),
    ]

    out: Dict[str, Dict[str, object]] = {}
    for algorithm in sorted({str(r["algorithm"]) for r in rows if r.get("status") == "ok"}):
        if algorithm == reference_name:
            continue
        algo_rows = {
            key(r): r
            for r in rows
            if r.get("status") == "ok" and str(r.get("algorithm")) == algorithm
        }
        paired = sorted(set(ref_rows.keys()) & set(algo_rows.keys()))
        block: Dict[str, object] = {"n_paired": int(len(paired))}
        if not paired:
            out[algorithm] = block
            continue

        for metric, direction in metrics:
            ref_vals = np.asarray([ref_rows[k][metric] for k in paired], dtype=float)
            alg_vals = np.asarray([algo_rows[k][metric] for k in paired], dtype=float)
            mask = np.isfinite(ref_vals) & np.isfinite(alg_vals)
            ref_vals = ref_vals[mask]
            alg_vals = alg_vals[mask]
            if len(ref_vals) == 0:
                block[metric] = {
                    "delta_mean": float("nan"),
                    "improvement_mean": float("nan"),
                    "pvalue": float("nan"),
                    "p_reference_better": float("nan"),
                    "p_variant_better": float("nan"),
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                    "n_used": 0,
                }
                continue

            delta = ref_vals - alg_vals
            delta_mean = float(np.mean(delta))
            improvement = delta if direction == "less" else -delta
            improvement_mean = float(np.mean(improvement))
            eps = 1.0e-12
            wins = int(np.sum(improvement > eps))
            ties = int(np.sum(np.abs(improvement) <= eps))
            losses = int(np.sum(improvement < -eps))
            p_reference_better = float("nan")
            p_variant_better = float("nan")
            if wilcoxon is not None and len(ref_vals) >= 6:
                try:
                    reference_alt = "less" if direction == "less" else "greater"
                    variant_alt = "greater" if direction == "less" else "less"
                    p_reference_better = float(
                        wilcoxon(ref_vals, alg_vals, alternative=reference_alt, zero_method="wilcox").pvalue
                    )
                    p_variant_better = float(
                        wilcoxon(ref_vals, alg_vals, alternative=variant_alt, zero_method="wilcox").pvalue
                    )
                except Exception:
                    p_reference_better = float("nan")
                    p_variant_better = float("nan")
            block[metric] = {
                "delta_mean": delta_mean,
                "improvement_mean": improvement_mean,
                "pvalue": p_variant_better,
                "p_reference_better": p_reference_better,
                "p_variant_better": p_variant_better,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "n_used": int(len(ref_vals)),
            }
        out[algorithm] = block
    return out


def save_runs_csv(rows: List[Dict[str, object]], out_csv: str) -> None:
    cols = [
        "problem",
        "seed",
        "algorithm",
        "implementation_kind",
        "faithfulness",
        "change_protocol",
        "implementation_source",
        "author_code_status",
        "author_code_path",
        "author_code_available",
        "generations_per_step",
        "status",
        "error",
        "migd",
        "mhv",
        "final_igd",
        "final_hv",
        "post_change_igd",
        "post_change_hv",
        "recovery_time",
        "n_detected",
        "n_true_changes",
        "precision",
        "recall",
        "f1",
        "mean_delay",
        "runtime_sec",
        "env_changes",
        "local_search_calls",
        "memory_hits",
        "memory_archive_size",
        "adr_meta_enabled",
        "adr_meta_updates",
        "adr_meta_reuses",
        "adr_meta_reuse_ratio",
        "adr_meta_drift_mean",
        "adr_meta_reason_summary",
        "reaction_quality_reverted_count",
        "reaction_predictive_samples",
        "reaction_continuity_samples",
        "reaction_diversity_samples",
        "reaction_front_ls_samples",
        "reaction_adr_weight_count",
        "reaction_adr_predictive_weight_mean",
        "reaction_adr_continuity_weight_mean",
        "reaction_adr_diversity_weight_mean",
        "reaction_adr_front_ls_weight_mean",
        "reaction_geometry_ready_count",
        "reaction_geometry_front_shift_mean",
        "reaction_geometry_pos_shift_mean",
        "reaction_geometry_overlap_mean",
        "reaction_geometry_shape_change_mean",
        "reaction_geometry_coherence_mean",
        "reaction_geometry_pos_change_mean",
        "reaction_geometry_pof_change_mean",
        "reaction_geometry_negative_transfer_mean",
        "reaction_geometry_coherent_motion_mean",
        "reaction_geometry_global_change_mean",
        "reaction_geometry_pof_stability_mean",
        "reaction_geometry_tau_mean",
        "reaction_geometry_continuity_risk_mean",
        "reaction_geometry_continuity_removed_mean",
        "reaction_geometry_transfer_predictive_share_mean",
        "pm_family",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def render_report(
    summary: Dict[str, Dict[str, object]],
    pairwise: Dict[str, Dict[str, object]],
    selected_algorithms: List[str],
    selected_problems: List[str],
    args,
) -> str:
    lines = [
        "# Clean Dynamic SOTA Comparison",
        "",
        "## Config",
        f"- Fecha: {time.strftime('%Y-%m-%d')}",
        f"- Algoritmos: {', '.join(selected_algorithms)}",
        f"- Problemas: {', '.join(selected_problems)}",
        f"- Seeds: {args.seeds}",
        f"- Horizonte: {args.horizon}",
        f"- Poblacion: {args.size_gen}",
        f"- Variables: {args.n_variables}",
        f"- Generaciones por step: {args.generations_per_step}",
        "",
        "## Algorithms",
    ]
    for name in selected_algorithms:
        spec = ALGORITHMS[name]
        meta = algorithm_metadata(spec)
        paper_target = PAPER_FIDELITY.get(name) or PAPER_FIDELITY.get("adr_eda_v2_1" if name == "adr_eda" else "")
        suffix = (
            f"{spec.implementation_kind}, {spec.faithfulness}, change={spec.change_protocol}, "
            f"source={spec.implementation_source}"
        )
        if spec.author_code_path:
            available = "yes" if meta["author_code_available"] else "no"
            suffix += f", author_code={spec.author_code_status}, available={available}"
        if paper_target is not None:
            suffix += f", target={paper_target.target_status}"
        lines.append(f"- `{name}`: {spec.description} ({suffix})")

    lines.extend(["", "## Summary"])
    for name in selected_algorithms:
        block = summary.get(name)
        if not block:
            lines.append(f"- `{name}`: sin ejecuciones correctas.")
            continue
        lines.append(
            f"- `{name}`: "
            f"MIGD={block['migd']['mean']:.4f}, "
            f"MHV={block['mhv']['mean']:.4f}, "
            f"postIGD={block['post_change_igd']['mean']:.4f}, "
            f"postHV={block['post_change_hv']['mean']:.4f}, "
            f"RT={block['recovery_time']['mean']:.4f}, "
            f"F1={block['f1']['mean']:.4f}, "
            f"n={block['n_ok']}"
        )

    lines.extend(["", "## ADR Pairwise"])
    if not pairwise:
        lines.append("- No hubo pares suficientes contra `adr_eda`.")
    else:
        lines.append(
            "- `improvement_mean` positivo significa que la variante mejora a `adr_eda` "
            "segun la direccion de la metrica; `p_variant_better` es el Wilcoxon "
            "unilateral en esa direccion."
        )
        for name, block in pairwise.items():
            lines.append(f"- `{name}` vs `adr_eda`: n_paired={block.get('n_paired', 0)}")
            for metric in ["migd", "mhv", "post_change_igd", "post_change_hv", "recovery_time"]:
                if metric not in block:
                    continue
                metric_block = block[metric]
                lines.append(
                    f"  {metric}: n={metric_block.get('n_used', 0)}, "
                    f"improvement_mean={metric_block.get('improvement_mean', float('nan')):.4f}, "
                    f"wins/ties/losses="
                    f"{metric_block.get('wins', 0)}/{metric_block.get('ties', 0)}/{metric_block.get('losses', 0)}, "
                    f"p_variant_better={metric_block.get('p_variant_better', float('nan')):.4g}"
                )

    lines.extend(
        [
            "",
            "## Notes",
            "- `reference_impl`: implementación limpia en este archivo.",
            "- `dedicated_module`: baseline dedicado implementado en este runner unificado.",
            "- `official_library_wrapper`: implementación oficial de librería externa envuelta solo para usar las métricas y el formato del protocolo común.",
            "- `author_code_audited_python_port`: port Python integrado en el protocolo unificado y auditado contra código oficial localizado; no implica ejecución directa del binario/script original.",
            "- `framework_wrapper`: comparación limpia dentro de tu backbone DynamicEDA, pero no reimplementación paper-faithful completa desde cero.",
            "- `frozen_profile`: perfil archivado y congelado para ADR-EDA v2.1 en benchmark-side.",
            "- `provisional_*`: implementación útil para cribado, pero no debe presentarse como reproducción fiel del paper.",
            "- `precision`/`recall`/`f1` se reportan como diagnóstico de detección, pero no entran en el contraste pareado porque varios baselines reciben notificación/oráculo de cambio mientras ADR-EDA usa su detector interno congelado.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean comparison runner for strong dynamic baselines vs ADR-EDA.")
    parser.add_argument("--algorithms", type=str, default="all")
    parser.add_argument("--problems", type=str, default="all")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--size-gen", type=int, default=90)
    parser.add_argument("--n-variables", type=int, default=12)
    parser.add_argument("--generations-per-step", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="clean_dynamic_sota_comparison")
    parser.add_argument("--post-change-lookahead", type=int, default=12)
    parser.add_argument("--recovery-lookahead", type=int, default=25)
    parser.add_argument("--recovery-baseline-window", type=int, default=8)
    parser.add_argument("--recovery-tol-ratio", type=float, default=1.10)
    args = parser.parse_args()

    selected_algorithms = parse_algorithm_list(args.algorithms)
    selected_problems = parse_problem_list(args.problems)
    if not selected_algorithms:
        raise ValueError("No valid algorithms selected.")
    if not selected_problems:
        raise ValueError("No valid problems selected.")

    os.makedirs(args.outdir, exist_ok=True)

    speda_cfg_base = SimpleNamespace(
        n_variables=args.n_variables,
        size_gen=args.size_gen,
        horizon=args.horizon,
    )
    ga_cfg = GAConfig(
        pop_size=args.size_gen,
        n_variables=args.n_variables,
        horizon=args.horizon,
        generations_per_step=args.generations_per_step,
    )

    all_rows: List[Dict[str, object]] = []
    print("Clean dynamic SOTA comparison")
    print(f"Algorithms: {', '.join(selected_algorithms)}")
    print(f"Problems: {', '.join(selected_problems)}")
    print(
        f"Seeds: {args.seeds} | Horizon: {args.horizon} | "
        f"Pop: {args.size_gen} | Gen/step: {args.generations_per_step}"
    )

    for problem_name in selected_problems:
        print(f"\n[Problem] {problem_name}")
        for algorithm_name in selected_algorithms:
            print(f"  - {algorithm_name}")
            for seed in range(args.seeds):
                result = run_algorithm(
                    algorithm_name=algorithm_name,
                    problem_name=problem_name,
                    speda_cfg_base=speda_cfg_base,
                    ga_cfg=ga_cfg,
                    seed=seed,
                )
                temporal = compute_temporal_metrics(
                    problem_name=problem_name,
                    horizon=args.horizon,
                    n_variables=args.n_variables,
                    result=result,
                    post_change_lookahead=args.post_change_lookahead,
                    recovery_lookahead=args.recovery_lookahead,
                    recovery_baseline_window=args.recovery_baseline_window,
                    recovery_tol_ratio=args.recovery_tol_ratio,
                )
                spec = ALGORITHMS[algorithm_name]
                meta = algorithm_metadata(spec)
                all_rows.append(
                    {
                        "problem": problem_name,
                        "seed": seed,
                        "algorithm": algorithm_name,
                        "implementation_kind": spec.implementation_kind,
                        "faithfulness": spec.faithfulness,
                        "change_protocol": spec.change_protocol,
                        "implementation_source": meta["implementation_source"],
                        "author_code_status": meta["author_code_status"],
                        "author_code_path": meta["author_code_path"],
                        "author_code_available": meta["author_code_available"],
                        "generations_per_step": args.generations_per_step,
                        **result,
                        **temporal,
                    }
                )

    summary = summarize_by_algorithm(all_rows)
    pairwise = compare_vs_reference(all_rows, reference_name="adr_eda")
    output = {
        "config": {
            "selected_algorithms": selected_algorithms,
            "selected_problems": selected_problems,
            "seeds": args.seeds,
            "horizon": args.horizon,
            "size_gen": args.size_gen,
            "n_variables": args.n_variables,
            "generations_per_step": args.generations_per_step,
        },
        "algorithm_metadata": {
            name: algorithm_metadata(ALGORITHMS[name])
            for name in selected_algorithms
        },
        "summary": summary,
        "pairwise_vs_adr_eda": pairwise,
    }

    csv_path = os.path.join(args.outdir, "per_run_metrics.csv")
    json_path = os.path.join(args.outdir, "summary.json")
    report_path = os.path.join(args.outdir, "REPORT.md")

    save_runs_csv(all_rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary, pairwise, selected_algorithms, selected_problems, args))

    print("\nCompleted.")
    print(f"- CSV: {csv_path}")
    print(f"- JSON: {json_path}")
    print(f"- REPORT: {report_path}")


if __name__ == "__main__":
    main()
