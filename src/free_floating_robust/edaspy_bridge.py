from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .evaluator import FreeFloatingRobustDMOOP
from .edaspy_runtime import enable_adaptive_bn_folds_patch, ensure_edaspy_importable
from .planner import RobustDecisionEncoding
from .types import SystemState


def _normalize_optimizer_family(raw_family: Optional[str]) -> str:
    aliases = {
        "SPBN": "SPBN",
        "SPEDA": "SPBN",
        "KDEBN": "KDEBN",
        "KDE": "KDEBN",
        "KEDA": "KDEBN",
        "MULTIVARIATEKEDA": "KDEBN",
    }
    family = str(raw_family or "SPBN").strip().upper()
    if family not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"optimizer_family='{raw_family}' no soportado. Usa: {supported}.")
    return aliases[family]


def _load_edaspy_classes(optimizer_family: str):
    ensure_edaspy_importable()
    enable_adaptive_bn_folds_patch()
    family = _normalize_optimizer_family(optimizer_family)
    try:
        from EDAspy.optimization.multivariate.keda import MultivariateKEDA
        from EDAspy.optimization.multivariate.speda import SPEDA
        from EDAspy.optimization.selection import NSGA2Selection
    except Exception as exc:
        raise ImportError(
            "No se pudo importar SPEDA/MultivariateKEDA/NSGA2Selection desde EDAspy. "
            f"Causa raíz: {type(exc).__name__}: {exc}"
        ) from exc
    optimizer_cls = SPEDA if family == "SPBN" else MultivariateKEDA
    return optimizer_cls, NSGA2Selection, family


@dataclass
class SPEDARunConfig:
    size_gen: int = 96
    max_iter: int = 40
    dead_iter: int = 20
    alpha: float = 0.35
    archive_generations: float = 4.0
    w_noise: float = 0.25
    parallelize: bool = False
    disp: bool = True
    optimizer_family: str = "SPBN"

    def __post_init__(self) -> None:
        self.optimizer_family = _normalize_optimizer_family(self.optimizer_family)


class SPEDAStaticRunner:
    """Thin wrapper for EDAspy SPEDA/MultivariateKEDA over free_floating_robust."""

    def __init__(self, encoding: RobustDecisionEncoding, config: Optional[SPEDARunConfig] = None):
        self.encoding = encoding
        self.config = config or SPEDARunConfig()
        self.optimizer_cls, self.nsga2_cls, self.optimizer_family = _load_edaspy_classes(
            self.config.optimizer_family
        )

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.encoding.bounds[:, 0].astype(float)
        upper = self.encoding.bounds[:, 1].astype(float)
        return lower, upper

    def _fit_init_data_shape(self, init_data: np.ndarray) -> np.ndarray:
        data = np.asarray(init_data, dtype=float)
        if data.ndim != 2:
            raise ValueError("init_data debe ser un array 2D.")
        if data.shape[1] != self.encoding.dimension:
            raise ValueError("init_data tiene dimensión distinta al encoding.")

        n = int(self.config.size_gen)
        if data.shape[0] == n:
            return data
        if data.shape[0] > n:
            return data[:n]

        missing = n - data.shape[0]
        rng = np.random.default_rng(1234)
        low = self.encoding.bounds[:, 0]
        high = self.encoding.bounds[:, 1]
        immigrants = rng.uniform(low=low, high=high, size=(missing, self.encoding.dimension))
        return np.vstack([data, immigrants])

    def optimize(
        self,
        problem: FreeFloatingRobustDMOOP,
        init_data: Optional[np.ndarray] = None,
        max_iter: Optional[int] = None,
        objective_min_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        lower, upper = self._bounds()
        cfg = self.config

        effective_init = None
        if init_data is not None:
            effective_init = self._fit_init_data_shape(init_data)

        optimizer = self.optimizer_cls(
            size_gen=int(cfg.size_gen),
            max_iter=int(max_iter or cfg.max_iter),
            dead_iter=int(cfg.dead_iter),
            n_variables=int(self.encoding.dimension),
            lower_bound=lower,
            upper_bound=upper,
            l=float(cfg.archive_generations),
            alpha=float(cfg.alpha),
            disp=bool(cfg.disp),
            parallelize=bool(cfg.parallelize),
            init_data=effective_init,
            w_noise=float(cfg.w_noise),
        )
        optimizer.selection_strategy = self.nsga2_cls()

        objective = objective_min_fn if objective_min_fn is not None else problem.objective_min
        result = optimizer.minimize(objective, output_runtime=cfg.disp)
        result = dict(result)
        result["optimizer_family"] = self.optimizer_family
        if "pareto_front" in result:
            result["pareto_front_max"] = -np.asarray(result["pareto_front"], dtype=float)
        return result


@dataclass
class WindowedResult:
    window_id: str
    raw_result: Dict[str, Any]
    selected_index: Optional[int] = None
    selected_objectives: Optional[List[float]] = None
    selected_details: Optional[Dict[str, float]] = None


class WindowedDynamicSPEDA:
    """
    Minimal dynamic orchestration over static SPEDA runs.

    - Solves one window at a time with static EDA.
    - Transfers previous Pareto set as init_data for next window.
    - Carries forward selected end-state (best-by-sum over Pareto front).
    """

    def __init__(self, static_runner: SPEDAStaticRunner):
        self.static_runner = static_runner

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
    ) -> List[WindowedResult]:
        transfer_pool: Optional[np.ndarray] = None
        carry_state: Optional[SystemState] = None
        out: List[WindowedResult] = []

        for window in windows:
            window_id = str(window.get("id", f"window_{len(out)}"))
            problem = problem_builder(window, carry_state)
            max_iter = window.get("max_iter")
            demand_multiplier = window.get("demand_multiplier")
            if demand_multiplier is None:
                demand_multiplier = window.get("demand_shock")

            def objective_min_fn(x: np.ndarray) -> np.ndarray:
                obj, _, _ = problem.rollout(
                    x=x,
                    base_state=carry_state,
                    external_demand_multiplier=demand_multiplier,
                )
                return -np.asarray(obj, dtype=float)

            raw = self.static_runner.optimize(
                problem=problem,
                init_data=transfer_pool,
                max_iter=int(max_iter) if max_iter is not None else None,
                objective_min_fn=objective_min_fn,
            )
            front = np.asarray(raw.get("pareto_front_max", np.empty((0, 3))), dtype=float)
            pset = np.asarray(
                raw.get("pareto_set", np.empty((0, self.static_runner.encoding.dimension))),
                dtype=float,
            )

            selected_index = self._best_by_sum_index(front)
            selected_objectives = None
            selected_details = None

            if selected_index is not None and selected_index < pset.shape[0]:
                x_sel = pset[selected_index]
                obj, details, end_state = problem.rollout(
                    x=x_sel,
                    base_state=carry_state,
                    external_demand_multiplier=demand_multiplier,
                )
                carry_state = copy.deepcopy(end_state)
                selected_objectives = [float(obj[0]), float(obj[1]), float(obj[2])]
                selected_details = {k: float(v) for k, v in details.items()}

            if pset.ndim == 2 and pset.shape[0] > 0:
                transfer_pool = pset

            out.append(
                WindowedResult(
                    window_id=window_id,
                    raw_result=raw,
                    selected_index=selected_index,
                    selected_objectives=selected_objectives,
                    selected_details=selected_details,
                )
            )
        return out
