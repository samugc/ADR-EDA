from __future__ import annotations

import argparse
from typing import Dict


OPS_COST_PROFILES: Dict[str, Dict[str, float]] = {
    "legacy": {
        "relocation_cost_per_km": 0.22,
        "ops_deadhead_cost_per_km": 0.22,
        "ops_cost_per_action_eur": 1.20,
        "energy_cost_per_kwh": 0.23,
    },
    "conservador": {
        "relocation_cost_per_km": 0.45,
        "ops_deadhead_cost_per_km": 0.45,
        "ops_cost_per_action_eur": 1.40,
        "energy_cost_per_kwh": 0.18,
    },
    "serio": {
        "relocation_cost_per_km": 0.60,
        "ops_deadhead_cost_per_km": 0.60,
        "ops_cost_per_action_eur": 1.80,
        "energy_cost_per_kwh": 0.22,
    },
    "stress": {
        "relocation_cost_per_km": 0.75,
        "ops_deadhead_cost_per_km": 0.75,
        "ops_cost_per_action_eur": 2.20,
        "energy_cost_per_kwh": 0.28,
    },
}


def resolve_ops_cost_profile(profile: str) -> Dict[str, float]:
    key = str(profile or "legacy").strip().lower()
    if key not in OPS_COST_PROFILES:
        raise ValueError(f"Perfil de costes no soportado: {profile!r}")
    return dict(OPS_COST_PROFILES[key])


def apply_ops_cost_profile(args: argparse.Namespace) -> argparse.Namespace:
    profile = str(getattr(args, "ops_cost_profile", "custom") or "custom").strip().lower()
    if profile == "custom":
        return args

    resolved = resolve_ops_cost_profile(profile)
    args.relocation_cost_per_km = float(resolved["relocation_cost_per_km"])
    args.ops_deadhead_cost_per_km = float(resolved["ops_deadhead_cost_per_km"])
    args.ops_cost_per_action_eur = float(resolved["ops_cost_per_action_eur"])
    args.energy_cost_per_kwh = float(resolved["energy_cost_per_kwh"])
    return args
