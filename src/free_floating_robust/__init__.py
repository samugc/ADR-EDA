"""Official active model surface for the project.

Keep the default import path focused on the operational core:
- scenario generation,
- planning,
- stateful rollout/evaluation.

EDA integrations remain available as optional exports but are loaded lazily so
the core package stays lightweight and easier to reason about during day-to-day
development.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .evaluator import FreeFloatingRobustDMOOP
from .planner import RobustDecisionEncoding, RobustOperationalPolicy, RollingHorizonPlanner
from .scenario import MadridFreeFloatingScenario, ScenarioConfig
from .types import RobustModelConfig

__all__ = [
    "FreeFloatingRobustDMOOP",
    "RobustDecisionEncoding",
    "RobustOperationalPolicy",
    "RollingHorizonPlanner",
    "MadridFreeFloatingScenario",
    "ScenarioConfig",
    "RobustModelConfig",
]

_OPTIONAL_EXPORTS = {
    "SPEDARunConfig": (".edaspy_bridge", "SPEDARunConfig"),
    "SPEDAStaticRunner": (".edaspy_bridge", "SPEDAStaticRunner"),
    "WindowedDynamicSPEDA": (".edaspy_bridge", "WindowedDynamicSPEDA"),
    "WindowedResult": (".edaspy_bridge", "WindowedResult"),
    "DynamicSPEDARunConfig": (".edaspy_dynamic_bridge", "DynamicSPEDARunConfig"),
    "DynamicWindowResult": (".edaspy_dynamic_bridge", "DynamicWindowResult"),
    "StatefulDynamicSPEDA": (".edaspy_dynamic_bridge", "StatefulDynamicSPEDA"),
    "OnlineStatefulDynamicSPEDA": (".edaspy_dynamic_bridge", "OnlineStatefulDynamicSPEDA"),
}


def __getattr__(name: str) -> Any:
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _OPTIONAL_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_OPTIONAL_EXPORTS))
