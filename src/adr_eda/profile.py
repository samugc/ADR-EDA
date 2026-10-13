"""Reference configuration for ADR-EDA."""

ADR_EDA_PROFILE = {
    "pm_family": "SPBN",

    # Dynamic change detection
    "split_drift_signals": True,
    "adaptive_lambda_enabled": True,
    "adaptive_lambda_cov_enabled": True,
    "adaptive_cooldown_enabled": True,
    "cov_threshold_k": 2.8,
    "alert_cooldown": 1,
    "min_change_interval": 1,

    # Adaptive response
    "adaptive_policy_enabled": True,
    "change_type_enabled": True,
    "adr_enabled": True,

    # Response mechanisms
    "diversity_block_enabled": True,
    "predictive_block_enabled": True,
    "continuity_block_enabled": True,
    "adr_front_ls_enabled": True,
    "adr_front_ls_context_gate_enabled": True,

    # Acceptance safeguard
    "reaction_quality_gate_enabled": True,

    # Response strength
    "react_strength": 2.8,
    "react_strength_mild": 1.6,
    "immigrant_ratio": 0.28,
    "immigrant_ratio_mild": 0.16,

    # Pareto-front local search
    "local_search_enabled": True,
    "local_search_neighbors": 6,
    "local_search_front_ratio": 0.55,
    "local_search_step": 0.06,
    "local_search_max_points": 36,
    "adr_front_ls_min_count_enabled": True,
    "adr_front_ls_min_weight": 0.04,
    "adr_front_ls_min_budget": 3,

    # Components not used in the validated profile
    "memory_enabled": False,
    "portfolio_mix_enabled": False,
    "response_mode_enabled": False,
}
