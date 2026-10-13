#!/usr/bin/env python3
"""
Robust dynamic multi-objective SPEDA benchmark suite.

What this script does:
1) Runs a robust DynamicSPEDA adaptation (SPEDA + Debski local search).
2) Compares dynamic reaction enabled vs disabled (static baseline).
3) Evaluates a battery of simple dynamic MOO test problems.
4) Exports CSV + JSON + PNG and interactive HTML plots.
"""

import argparse
import csv
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def _bootstrap_edaspy_path():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        # Local execution in the thesis repo.
        os.path.join(repo_root, "tests", "edas"),
        os.path.join(repo_root, "EDAspy"),
        # Typical containerized paths.
        "/app/EDAspy",
        "/app/EDAspy/EDAspy",
        "/app/src",
        # Relative candidates.
        os.path.join(here, "EDAspy"),
        os.path.join(here, "..", "EDAspy"),
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_edaspy_path()

try:
    from EDAspy.optimization.eda_dynamic import DynamicEDA
    from EDAspy.optimization.custom.initialization_models import UniformGenInit
    from EDAspy.optimization.custom.probabilistic_models import GBN, KDEBN, MultiGauss, SPBN
except Exception as exc:  # pragma: no cover - startup guard
    raise ImportError(
        "Could not import EDAspy. Add EDAspy to PYTHONPATH or run from a configured environment."
    ) from exc


class DebConstrainedSelection:
    """Deb constraint handling + NSGA-II style selection."""

    def __init__(self, eps_feasible: float = 1e-12):
        self.eps_feasible = eps_feasible

    def _deb_dominates(self, cv_a: float, obj_a: np.ndarray, cv_b: float, obj_b: np.ndarray) -> bool:
        a_feas = cv_a <= self.eps_feasible
        b_feas = cv_b <= self.eps_feasible

        if a_feas and not b_feas:
            return True
        if b_feas and not a_feas:
            return False
        if (not a_feas) and (not b_feas):
            return cv_a < cv_b

        return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))

    def _fast_non_dominated_sort(self, F: np.ndarray) -> List[List[int]]:
        n_points = F.shape[0]
        cv = F[:, 0]
        obj = F[:, 1:]

        dominates = [[] for _ in range(n_points)]
        dominated_count = np.zeros(n_points, dtype=int)
        fronts: List[List[int]] = [[]]

        for p in range(n_points):
            for q in range(n_points):
                if p == q:
                    continue
                if self._deb_dominates(cv[p], obj[p], cv[q], obj[q]):
                    dominates[p].append(q)
                elif self._deb_dominates(cv[q], obj[q], cv[p], obj[p]):
                    dominated_count[p] += 1

            if dominated_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominates[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            i += 1

        return fronts

    def _crowding_distance(self, obj: np.ndarray, front: List[int]) -> Dict[int, float]:
        if not front:
            return {}
        if len(front) <= 2:
            return {idx: np.inf for idx in front}

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

    def _select_n(self, X: np.ndarray, F: np.ndarray, n_survivors: int) -> np.ndarray:
        fronts = self._fast_non_dominated_sort(F)
        obj = F[:, 1:]

        selected: List[int] = []
        for front in fronts:
            if len(selected) + len(front) <= n_survivors:
                selected.extend(front)
                continue

            dist = self._crowding_distance(obj, front)
            front_sorted = sorted(front, key=lambda i: dist[i], reverse=True)
            needed = n_survivors - len(selected)
            selected.extend(front_sorted[:needed])
            break

        return np.asarray(selected, dtype=int)

    def select(self, X: np.ndarray, F: np.ndarray, elite_size: int, truncation_size: int):
        elite_idx = self._select_n(X, F, elite_size)
        trunc_idx = self._select_n(X, F, truncation_size)
        return X[elite_idx], F[elite_idx], X[trunc_idx], F[trunc_idx]


class DynamicSPEDA(DynamicEDA):
    """DynamicEDA adaptation that preserves SPEDA archive-based PM update."""

    def __init__(
        self,
        size_gen: int,
        max_iter: int,
        dead_iter: int,
        n_variables: int,
        lower_bound=0.0,
        upper_bound=1.0,
        l: float = 6.0,
        alpha: float = 0.5,
        elite_factor: float = 0.4,
        w_noise: float = 0.08,
        parallelize: bool = False,
        disp: bool = False,
        black_list: list = None,
        white_list: list = None,
        pm_family: str = "SPBN",
        num_folds: int = 3,
        archive_retention_after_change: float = 0.25,
        selection_strategy=None,
        pm_multi_gauss_lower: float = 1e-3,
        pm_multi_gauss_upper: float = 1.0,
        **dynamic_kwargs,
    ):
        if selection_strategy is None:
            selection_strategy = DebConstrainedSelection()

        detector_cfg = {}
        reaction_cfg = {}
        local_search_cfg = {}

        explicit_detector = dynamic_kwargs.pop("detector", None)
        explicit_reaction = dynamic_kwargs.pop("reaction", None)
        explicit_local_search = dynamic_kwargs.pop("local_search", None)

        if explicit_detector is not None:
            detector_cfg.update(dict(explicit_detector))
        if explicit_reaction is not None:
            reaction_cfg.update(dict(explicit_reaction))
        if explicit_local_search is not None:
            local_search_cfg.update(dict(explicit_local_search))

        detector_defaults = getattr(DynamicEDA, "_DEFAULT_DETECTOR", {})
        reaction_defaults = getattr(DynamicEDA, "_DEFAULT_REACTION", {})
        local_defaults = getattr(DynamicEDA, "_DEFAULT_LOCAL_SEARCH", {})

        remaining_kwargs = {}
        for key, value in dynamic_kwargs.items():
            if key in detector_defaults:
                detector_cfg[key] = value
            elif key in reaction_defaults:
                reaction_cfg[key] = value
            elif key in local_defaults:
                local_search_cfg[key] = value
            else:
                remaining_kwargs[key] = value

        init_kwargs = {
            "size_gen": size_gen,
            "max_iter": max_iter,
            "dead_iter": dead_iter,
            "n_variables": n_variables,
            "alpha": alpha,
            "elite_factor": elite_factor,
            "disp": disp,
            "parallelize": parallelize,
            "w_noise": w_noise,
            "selection_strategy": selection_strategy,
        }

        # Compatibility across DynamicEDA versions: only pass supported kwargs.
        sig = inspect.signature(DynamicEDA.__init__)
        allowed = set(sig.parameters.keys())

        if "detector" in allowed:
            init_kwargs["detector"] = detector_cfg
            init_kwargs["reaction"] = reaction_cfg
            init_kwargs["local_search"] = local_search_cfg
            init_kwargs.update(remaining_kwargs)
        else:
            # Legacy compatibility path where dynamic params are flat kwargs.
            flat_kwargs = {}
            flat_kwargs.update(detector_cfg)
            flat_kwargs.update(reaction_cfg)
            flat_kwargs.update(local_search_cfg)
            flat_kwargs.update(remaining_kwargs)
            init_kwargs.update(flat_kwargs)

        clean_kwargs = {k: v for k, v in init_kwargs.items() if k in allowed}
        super().__init__(**clean_kwargs)

        self.vars = [str(i) for i in range(n_variables)]
        self.pm_family = str(pm_family).upper()
        self.pm = self._build_pm(
            pm_family=self.pm_family,
            black_list=black_list,
            white_list=white_list,
            multi_gauss_lower=pm_multi_gauss_lower,
            multi_gauss_upper=pm_multi_gauss_upper,
        )
        self.init = UniformGenInit(self.n_variables, lower_bound=lower_bound, upper_bound=upper_bound)

        elite_size = int(size_gen * alpha)
        self.l_len = max(1, int(l * max(1, elite_size)))
        self.archive = np.empty((0, self.n_variables))
        self.num_folds = max(2, int(num_folds))
        self.archive_retention_after_change = float(np.clip(archive_retention_after_change, 0.0, 1.0))

    def _build_pm(
        self,
        pm_family: str,
        black_list: list,
        white_list: list,
        multi_gauss_lower: float,
        multi_gauss_upper: float,
    ):
        family = str(pm_family).strip().upper()
        if family == "SPBN":
            return SPBN(self.vars, black_list=black_list, white_list=white_list)
        if family == "GBN":
            return GBN(self.vars, black_list=black_list, white_list=white_list)
        if family == "KDEBN":
            return KDEBN(self.vars, black_list=black_list, white_list=white_list)
        if family in {"MULTIGAUSS", "MGAUSS", "MULTIVARIATE_GAUSSIAN"}:
            return MultiGauss(
                variables=list(range(self.n_variables)),
                lower_bound=float(multi_gauss_lower),
                upper_bound=float(multi_gauss_upper),
            )
        raise ValueError(f"Unsupported pm_family='{pm_family}'. Use SPBN, GBN, KDEBN or MultiGauss.")

    def _update_archive(self):
        if self.elite_temp is None or len(self.elite_temp) == 0:
            return
        self.archive = np.append(self.archive, self.elite_temp, axis=0)
        self.archive = self.archive[-self.l_len :]

    def _update_pm(self):
        self._update_archive()
        min_rows = max(10, self.n_variables + 2)
        data = self.archive if len(self.archive) >= min_rows else self.generation
        self.pm.learn(dataset=data, num_folds=self.num_folds)

    def _react(self, clip_bounds=None):
        sig = inspect.signature(super()._react)
        if "clip_bounds" in sig.parameters:
            super()._react(clip_bounds=clip_bounds)
        else:
            super()._react()

        if len(self.archive) > 0:
            keep = int(self.l_len * self.archive_retention_after_change)
            self.archive = self.archive[-keep:] if keep > 0 else np.empty((0, self.n_variables))


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

    def __init__(self, n_variables: int, horizon: int):
        self.n_variables = n_variables
        self.horizon = horizon
        self.t = 0

    def update_time(self):
        self.t += 1

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        objs, cv = self.evaluate(x)
        return np.concatenate(([cv], objs))

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


PROBLEM_REGISTRY: Dict[str, Callable[[int, int], DynamicProblemBase]] = {
    DZDT1Step.name: lambda n, h: DZDT1Step(n, h),
    DZDT1Sine.name: lambda n, h: DZDT1Sine(n, h),
    DZDT2Shape.name: lambda n, h: DZDT2Shape(n, h),
    DZDT3Shocks.name: lambda n, h: DZDT3Shocks(n, h),
    FDA1Lite.name: lambda n, h: FDA1Lite(n, h),
    FDA2Lite.name: lambda n, h: FDA2Lite(n, h),
    DConstrainedZDT1.name: lambda n, h: DConstrainedZDT1(n, h),
    DDTLZ2_3Obj.name: lambda n, h: DDTLZ2_3Obj(n, h),
}


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
    volume = float(np.prod(ref))
    return volume * float(np.mean(dominated))


def hypervolume(front: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> float:
    if front is None or len(front) == 0:
        return 0.0
    if ref.shape[0] == 2:
        return hypervolume_2d(front, (ref[0], ref[1]))
    return hypervolume_mc(front, ref, rng=rng)


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


@dataclass
class SpedaRobustConfig:
    n_variables: int = 12
    size_gen: int = 90
    horizon: int = 100
    algorithm_family: str = "speda"

    # SPEDA.
    pm_family: str = "SPBN"
    l: float = 6.0
    num_folds: int = 3
    archive_retention_after_change: float = 0.30

    # Jiang et al. (2018) faithful transfer baseline.
    jiang_transfer_samples: int = 48
    jiang_tca_dim: int = 8
    jiang_tca_sigma: float = 0.60
    jiang_tca_mu: float = 0.50
    jiang_preimage_neighbors: int = 6
    jiang_preimage_refine_steps: int = 6
    jiang_refine_sigma_start: float = 0.08
    jiang_refine_sigma_decay: float = 0.70
    jiang_generations_per_step: int = 2
    jiang_crossover_prob: float = 0.90
    jiang_sbx_eta: float = 20.0
    jiang_mutation_eta: float = 20.0
    jiang_use_transfer_every_step: bool = True

    # Dynamic detector/reaction.
    drift_alpha: float = 0.05
    lambda_ewma: float = 0.70
    n_sentinels: int = 20
    sentinel_refresh: int = 14
    k_consecutive: int = 2
    react_strength: float = 2.7
    immigrant_ratio: float = 0.30
    react_strength_mild: float = 1.5
    immigrant_ratio_mild: float = 0.15
    severity_high: float = 2.0
    normalize_objectives: bool = True
    alert_cooldown: int = 1
    min_change_interval: int = 1
    split_drift_signals: bool = False
    split_drift_hit_mode: str = "or"
    cov_threshold_k: float = 3.0
    cov_baseline_min_samples: int = 30
    cov_baseline_maxlen: int = 200
    adaptive_lambda_enabled: bool = False
    adaptive_lambda_cov_enabled: bool = False
    adaptive_lambda_min: float = 0.05
    adaptive_lambda_max: float = 0.40
    adaptive_lambda_window: int = 20
    adaptive_lambda_mid_relvar: float = 0.05
    adaptive_cooldown_enabled: bool = False
    adaptive_cooldown_low: int = 3
    adaptive_cooldown_high: int = 1
    adaptive_cooldown_severity_ref: float = 2.0
    detector_rearm_ratio: float = 0.70

    react_strength_medium: float = 2.2
    immigrant_ratio_medium: float = 0.22
    adaptive_policy_enabled: bool = False
    profile_threshold_medium: float = 0.45
    profile_threshold_severe: float = 0.75
    adaptive_weight_severity: float = 0.60
    adaptive_weight_recency: float = 0.20
    adaptive_weight_trend: float = 0.20
    adaptive_recency_scale: int = 8
    adaptive_trend_window: int = 6
    reaction_min_drift_ratio: float = 1.10
    reaction_min_interval: int = 3
    reaction_window: int = 20
    reaction_max_calls_in_window: int = 3
    reaction_max_calls_mild_window: int = 2
    reaction_max_calls_medium_window: int = 3
    reaction_max_calls_severe_window: int = 5
    reaction_quality_gate_enabled: bool = True
    reaction_quality_gate_rel_tol: float = 0.02
    reaction_quality_gate_abs_tol: float = 1e-2
    reaction_quality_gate_mo_enabled: bool = True
    reaction_quality_gate_mo_rel_tol: float = 0.02
    reaction_quality_gate_mo_size_tol: float = 0.50
    reaction_quality_gate_mo_strategy: str = "aggregate"
    reaction_quality_gate_mo_dominance_margin: float = 0.0
    memory_enabled: bool = False
    memory_topk: int = 16
    memory_max_size: int = 8
    memory_match_threshold: float = 1.25
    memory_inject_ratio: float = 0.20
    memory_min_archive_size: int = 3
    memory_min_interval: int = 3
    diversity_block_enabled: bool = True
    diversity_ratio_mild: float = 0.08
    diversity_ratio_medium: float = 0.14
    diversity_ratio_severe: float = 0.22
    diversity_gate_enabled: bool = True
    diversity_gate_trend_min: float = 0.30
    diversity_gate_cov_ratio_min: float = 1.10
    diversity_gate_severe_ratio: float = 1.60
    diversity_gate_fallback_scale: float = 0.50
    predictive_block_enabled: bool = False
    predictive_ratio_mild: float = 0.06
    predictive_ratio_medium: float = 0.06
    predictive_ratio_severe: float = 0.08
    predictive_step_scale: float = 0.70
    predictive_jitter_scale: float = 0.25
    predictive_max_step_span_ratio: float = 0.20
    predictive_trend_min: float = 0.40
    predictive_min_history: int = 2
    continuity_block_enabled: bool = True
    continuity_ratio_mild: float = 0.10
    continuity_ratio_medium: float = 0.12
    continuity_ratio_severe: float = 0.14
    continuity_step_scale: float = 0.25
    continuity_jitter_scale: float = 0.15
    adr_front_ls_enabled: bool = False
    adr_front_ls_context_gate_enabled: bool = False
    adr_front_ls_min_count_enabled: bool = False
    adr_front_ls_min_weight: float = 0.04
    adr_front_ls_min_budget: int = 3
    adr_credit_enabled: bool = False
    portfolio_mix_enabled: bool = False
    change_type_enabled: bool = False
    response_mode_enabled: bool = False
    adr_enabled: bool = False
    adr_stable_predictive_weight: float = 0.30
    adr_stable_continuity_weight: float = 0.25
    adr_stable_diversity_weight: float = 0.20
    adr_stable_front_ls_weight: float = 0.25
    adr_uncertain_predictive_weight: float = 0.15
    adr_uncertain_continuity_weight: float = 0.20
    adr_uncertain_diversity_weight: float = 0.40
    adr_uncertain_front_ls_weight: float = 0.25
    adr_abrupt_predictive_weight: float = 0.05
    adr_abrupt_continuity_weight: float = 0.20
    adr_abrupt_diversity_weight: float = 0.60
    adr_abrupt_front_ls_weight: float = 0.15

    # EDA.
    alpha: float = 0.5
    elite_factor: float = 0.4
    w_noise: float = 0.06

    # Debski local search.
    local_search_enabled: bool = True
    local_search_neighbors: int = 6
    local_search_front_ratio: float = 0.55
    local_search_step: float = 0.06
    local_search_max_points: int = 36
    local_search_mode: str = "gaussian"
    local_search_tube_radius: int = 1
    local_search_beam_width: int = 12
    local_search_beam_max_evals: int = 256
    local_search_prune_by_front: bool = True

    # Detection quality metric.
    tol_detect: int = 5


def tuned_cfg_for_problem(cfg: SpedaRobustConfig, problem_name: str) -> SpedaRobustConfig:
    smooth_names = {
        DZDT1Sine.name,
        DZDT2Shape.name,
        FDA1Lite.name,
        FDA2Lite.name,
        DConstrainedZDT1.name,
        DDTLZ2_3Obj.name,
    }
    if problem_name in smooth_names:
        return replace(cfg, drift_alpha=0.08, lambda_ewma=0.5, sentinel_refresh=20, k_consecutive=2)

    if problem_name in {DZDT3Shocks.name, DZDT1Step.name}:
        return replace(cfg, drift_alpha=0.05, lambda_ewma=0.75, sentinel_refresh=10, k_consecutive=2)

    return cfg


def ref_point_for_problem(problem_name: str) -> np.ndarray:
    if problem_name == DDTLZ2_3Obj.name:
        return np.array([1.6, 1.6, 1.6], dtype=float)
    return np.array([3.0, 3.0], dtype=float)


def create_speda(cfg: SpedaRobustConfig) -> DynamicSPEDA:
    dyn_kwargs = {
        "n_sentinels": cfg.n_sentinels,
        "drift_alpha": cfg.drift_alpha,
        "react_strength": cfg.react_strength,
        "immigrant_ratio": cfg.immigrant_ratio,
        "lambda_ewma": cfg.lambda_ewma,
        "sentinel_refresh": cfg.sentinel_refresh,
        "k_consecutive": cfg.k_consecutive,
        "severity_high": cfg.severity_high,
        "react_strength_mild": cfg.react_strength_mild,
        "react_strength_medium": cfg.react_strength_medium,
        "immigrant_ratio_mild": cfg.immigrant_ratio_mild,
        "immigrant_ratio_medium": cfg.immigrant_ratio_medium,
        "adaptive_policy_enabled": cfg.adaptive_policy_enabled,
        "profile_threshold_medium": cfg.profile_threshold_medium,
        "profile_threshold_severe": cfg.profile_threshold_severe,
        "adaptive_weight_severity": cfg.adaptive_weight_severity,
        "adaptive_weight_recency": cfg.adaptive_weight_recency,
        "adaptive_weight_trend": cfg.adaptive_weight_trend,
        "adaptive_recency_scale": cfg.adaptive_recency_scale,
        "adaptive_trend_window": cfg.adaptive_trend_window,
        "reaction_min_drift_ratio": cfg.reaction_min_drift_ratio,
        "reaction_min_interval": cfg.reaction_min_interval,
        "reaction_window": cfg.reaction_window,
        "reaction_max_calls_in_window": cfg.reaction_max_calls_in_window,
        "reaction_max_calls_mild_window": cfg.reaction_max_calls_mild_window,
        "reaction_max_calls_medium_window": cfg.reaction_max_calls_medium_window,
        "reaction_max_calls_severe_window": cfg.reaction_max_calls_severe_window,
        "reaction_quality_gate_enabled": cfg.reaction_quality_gate_enabled,
        "reaction_quality_gate_rel_tol": cfg.reaction_quality_gate_rel_tol,
        "reaction_quality_gate_abs_tol": cfg.reaction_quality_gate_abs_tol,
        "reaction_quality_gate_mo_enabled": cfg.reaction_quality_gate_mo_enabled,
        "reaction_quality_gate_mo_rel_tol": cfg.reaction_quality_gate_mo_rel_tol,
        "reaction_quality_gate_mo_size_tol": cfg.reaction_quality_gate_mo_size_tol,
        "reaction_quality_gate_mo_strategy": cfg.reaction_quality_gate_mo_strategy,
        "reaction_quality_gate_mo_dominance_margin": cfg.reaction_quality_gate_mo_dominance_margin,
        "normalize_objectives": cfg.normalize_objectives,
        "alert_cooldown": cfg.alert_cooldown,
        "min_change_interval": cfg.min_change_interval,
        "split_drift_signals": cfg.split_drift_signals,
        "split_drift_hit_mode": cfg.split_drift_hit_mode,
        "cov_threshold_k": cfg.cov_threshold_k,
        "cov_baseline_min_samples": cfg.cov_baseline_min_samples,
        "cov_baseline_maxlen": cfg.cov_baseline_maxlen,
        "adaptive_lambda_enabled": cfg.adaptive_lambda_enabled,
        "adaptive_lambda_cov_enabled": cfg.adaptive_lambda_cov_enabled,
        "adaptive_lambda_min": cfg.adaptive_lambda_min,
        "adaptive_lambda_max": cfg.adaptive_lambda_max,
        "adaptive_lambda_window": cfg.adaptive_lambda_window,
        "adaptive_lambda_mid_relvar": cfg.adaptive_lambda_mid_relvar,
        "adaptive_cooldown_enabled": cfg.adaptive_cooldown_enabled,
        "adaptive_cooldown_low": cfg.adaptive_cooldown_low,
        "adaptive_cooldown_high": cfg.adaptive_cooldown_high,
        "adaptive_cooldown_severity_ref": cfg.adaptive_cooldown_severity_ref,
        "detector_rearm_ratio": cfg.detector_rearm_ratio,
        "memory_enabled": cfg.memory_enabled,
        "memory_topk": cfg.memory_topk,
        "memory_max_size": cfg.memory_max_size,
        "memory_match_threshold": cfg.memory_match_threshold,
        "memory_inject_ratio": cfg.memory_inject_ratio,
        "memory_min_archive_size": cfg.memory_min_archive_size,
        "memory_min_interval": cfg.memory_min_interval,
        "diversity_block_enabled": cfg.diversity_block_enabled,
        "diversity_ratio_mild": cfg.diversity_ratio_mild,
        "diversity_ratio_medium": cfg.diversity_ratio_medium,
        "diversity_ratio_severe": cfg.diversity_ratio_severe,
        "diversity_gate_enabled": cfg.diversity_gate_enabled,
        "diversity_gate_trend_min": cfg.diversity_gate_trend_min,
        "diversity_gate_cov_ratio_min": cfg.diversity_gate_cov_ratio_min,
        "diversity_gate_severe_ratio": cfg.diversity_gate_severe_ratio,
        "diversity_gate_fallback_scale": cfg.diversity_gate_fallback_scale,
        "predictive_block_enabled": cfg.predictive_block_enabled,
        "predictive_ratio_mild": cfg.predictive_ratio_mild,
        "predictive_ratio_medium": cfg.predictive_ratio_medium,
        "predictive_ratio_severe": cfg.predictive_ratio_severe,
        "predictive_step_scale": cfg.predictive_step_scale,
        "predictive_jitter_scale": cfg.predictive_jitter_scale,
        "predictive_max_step_span_ratio": cfg.predictive_max_step_span_ratio,
        "predictive_trend_min": cfg.predictive_trend_min,
        "predictive_min_history": cfg.predictive_min_history,
        "continuity_block_enabled": cfg.continuity_block_enabled,
        "continuity_ratio_mild": cfg.continuity_ratio_mild,
        "continuity_ratio_medium": cfg.continuity_ratio_medium,
        "continuity_ratio_severe": cfg.continuity_ratio_severe,
        "continuity_step_scale": cfg.continuity_step_scale,
        "continuity_jitter_scale": cfg.continuity_jitter_scale,
        "adr_front_ls_enabled": cfg.adr_front_ls_enabled,
        "adr_front_ls_context_gate_enabled": cfg.adr_front_ls_context_gate_enabled,
        "adr_front_ls_min_count_enabled": cfg.adr_front_ls_min_count_enabled,
        "adr_front_ls_min_weight": cfg.adr_front_ls_min_weight,
        "adr_front_ls_min_budget": cfg.adr_front_ls_min_budget,
        "adr_credit_enabled": cfg.adr_credit_enabled,
        "portfolio_mix_enabled": cfg.portfolio_mix_enabled,
        "change_type_enabled": cfg.change_type_enabled,
        "response_mode_enabled": cfg.response_mode_enabled,
        "adr_enabled": cfg.adr_enabled,
        "adr_stable_predictive_weight": cfg.adr_stable_predictive_weight,
        "adr_stable_continuity_weight": cfg.adr_stable_continuity_weight,
        "adr_stable_diversity_weight": cfg.adr_stable_diversity_weight,
        "adr_stable_front_ls_weight": cfg.adr_stable_front_ls_weight,
        "adr_uncertain_predictive_weight": cfg.adr_uncertain_predictive_weight,
        "adr_uncertain_continuity_weight": cfg.adr_uncertain_continuity_weight,
        "adr_uncertain_diversity_weight": cfg.adr_uncertain_diversity_weight,
        "adr_uncertain_front_ls_weight": cfg.adr_uncertain_front_ls_weight,
        "adr_abrupt_predictive_weight": cfg.adr_abrupt_predictive_weight,
        "adr_abrupt_continuity_weight": cfg.adr_abrupt_continuity_weight,
        "adr_abrupt_diversity_weight": cfg.adr_abrupt_diversity_weight,
        "adr_abrupt_front_ls_weight": cfg.adr_abrupt_front_ls_weight,
        "local_search_enabled": cfg.local_search_enabled,
        "local_search_neighbors": cfg.local_search_neighbors,
        "local_search_front_ratio": cfg.local_search_front_ratio,
        "local_search_step": cfg.local_search_step,
        "local_search_max_points": cfg.local_search_max_points,
        "local_search_mode": cfg.local_search_mode,
        "local_search_tube_radius": cfg.local_search_tube_radius,
        "local_search_beam_width": cfg.local_search_beam_width,
        "local_search_beam_max_evals": cfg.local_search_beam_max_evals,
        "local_search_prune_by_front": cfg.local_search_prune_by_front,
    }
    return DynamicSPEDA(
        size_gen=cfg.size_gen,
        max_iter=2,
        dead_iter=1,
        n_variables=cfg.n_variables,
        lower_bound=0.0,
        upper_bound=1.0,
        l=cfg.l,
        pm_family=cfg.pm_family,
        alpha=cfg.alpha,
        elite_factor=cfg.elite_factor,
        w_noise=cfg.w_noise,
        disp=False,
        num_folds=cfg.num_folds,
        archive_retention_after_change=cfg.archive_retention_after_change,
        **dyn_kwargs,
    )


def _call_react_raw(eda: DynamicSPEDA, clip_bounds):
    sig = inspect.signature(eda._react)
    if "clip_bounds" in sig.parameters:
        eda._react(clip_bounds=clip_bounds)
    else:
        eda._react()


def _call_react(eda: DynamicSPEDA, cost_function, clip_bounds) -> bool:
    if not bool(getattr(eda, "reaction_quality_gate_enabled", False)):
        _call_react_raw(eda, clip_bounds)
        return False

    pre_front = None
    pre_best_cost = None
    if bool(getattr(eda, "_is_multiobjective_dynamic", False)):
        pre_front = eda._get_front_for_metrics().copy()
    else:
        pre_best_cost = float(np.min(eda.evaluations))

    pm_backup = None
    if hasattr(eda.pm, "pm") and isinstance(eda.pm.pm, np.ndarray):
        pm_backup = eda.pm.pm.copy()
    backup = {
        "generation": eda.generation.copy(),
        "evaluations": None if eda.evaluations is None else eda.evaluations.copy(),
        "pm": pm_backup,
        "env_changes": int(getattr(eda, "env_changes", 0)),
        "elite_temp": None if eda.elite_temp is None else eda.elite_temp.copy(),
        "evaluations_elite": None if eda.evaluations_elite is None else eda.evaluations_elite.copy(),
        "front": pre_front,
    }

    _call_react_raw(eda, clip_bounds)
    eda.generation = np.clip(eda.generation, clip_bounds[0], clip_bounds[1])
    eda._check_generation(cost_function)

    quality_reverted = False
    if pre_best_cost is not None:
        post_best_cost = float(np.min(eda.evaluations))
        allowed_delta = max(
            getattr(eda, "reaction_quality_gate_abs_tol", 0.0),
            abs(pre_best_cost) * getattr(eda, "reaction_quality_gate_rel_tol", 0.0),
        )
        quality_reverted = bool(post_best_cost > (pre_best_cost + allowed_delta))
    elif bool(getattr(eda, "_is_multiobjective_dynamic", False)) and backup["front"] is not None:
        eda._truncation()
        post_front = eda._get_front_for_metrics().copy()
        quality_reverted = bool(eda._should_revert_multiobjective_reaction(backup["front"], post_front))

    if quality_reverted:
        eda.generation = backup["generation"]
        eda.evaluations = backup["evaluations"]
        eda.elite_temp = backup["elite_temp"]
        eda.evaluations_elite = backup["evaluations_elite"]
        if backup["pm"] is not None:
            eda.pm.pm = backup["pm"]
        eda.env_changes = int(backup["env_changes"])
        eda._last_memory_reused = False
        eda._last_reaction_quality_reverted = True
        eda._last_reaction_block_reason = "quality_gate_revert"
        if hasattr(eda, "_response_revert_history"):
            eda._response_revert_history.append(1.0)
        if hasattr(eda, "_adr_update_block_credit"):
            eda._adr_update_block_credit(success=False)
        if hasattr(eda, "_stabilize_detector_without_reaction"):
            eda._stabilize_detector_without_reaction()
    else:
        eda._last_reaction_quality_reverted = False
        if hasattr(eda, "_response_revert_history"):
            eda._response_revert_history.append(0.0)

    return bool(quality_reverted)


def run_single(problem_name: str, cfg: SpedaRobustConfig, seed: int, with_reaction: bool) -> Dict[str, object]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    algorithm_family = str(getattr(cfg, "algorithm_family", "speda")).lower()

    if algorithm_family == "jiang2018_trnsga2":
        from jiang2018_transfer_baseline import run_single_jiang2018

        return run_single_jiang2018(
            problem_name=problem_name,
            cfg=cfg,
            seed=seed,
            problem_factory=PROBLEM_REGISTRY[problem_name],
            ref_point=ref_point_for_problem(problem_name),
            hypervolume_fn=hypervolume,
            igd_fn=igd,
            nondominated_fn=non_dominated_points,
        )

    try:
        problem = PROBLEM_REGISTRY[problem_name](cfg.n_variables, cfg.horizon)
        ref = ref_point_for_problem(problem_name)
        eda = create_speda(cfg)

        eda.generation = np.clip(eda._initialize_generation(), 0.0, 1.0)
        eda._check_generation(problem)
        eda.evaluations_elite = np.empty((0, eda.evaluations.shape[1]))
        eda._truncation()
        # Manual execution path: enable multiobjective mode to activate Debski local search.
        eda._is_multiobjective_dynamic = True
        eda._update_sentinels(force=True)

        hv_hist = []
        igd_hist = []
        detected = []
        reaction_quality_reverted_count = 0
        true_events = problem.true_change_points(cfg.horizon)

        t0 = time.perf_counter()
        for t in range(cfg.horizon):
            if eda._detect_change(problem):
                detected.append(t)
                if with_reaction:
                    quality_reverted = _call_react(eda, problem, (0.0, 1.0))
                    reaction_quality_reverted_count += int(quality_reverted)
                    if not quality_reverted:
                        eda.generation = np.clip(eda.generation, 0.0, 1.0)
                        eda._check_generation(problem)
                        eda._truncation()
                        eda._update_sentinels(force=True)

            eda._truncation()
            front_eval = eda.evaluations_elite if len(eda.evaluations_elite) > 0 else eda.evaluations
            front_obj = non_dominated_points(np.asarray(front_eval[:, 1:], dtype=float))
            true_pf = problem.true_pareto(n_points=400)

            hv_hist.append(hypervolume(front_obj, ref=ref, rng=rng))
            igd_hist.append(igd(front_obj, true_pf))

            if cfg.local_search_enabled and hasattr(eda, "_local_search_over_front"):
                eda._local_search_over_front(problem, (np.zeros(cfg.n_variables), np.ones(cfg.n_variables)))

            eda._update_pm()
            eda._new_generation()
            eda.generation = np.clip(eda.generation, 0.0, 1.0)
            eda._check_generation(problem)
            eda._update_sentinels()

            problem.update_time()

        elapsed = time.perf_counter() - t0
        hv_hist = np.asarray(hv_hist, dtype=float)
        igd_hist = np.asarray(igd_hist, dtype=float)
        det = detection_scores(true_events, detected, tolerance=cfg.tol_detect)

        return {
            "status": "ok",
            "error": "",
            "seed": seed,
            "with_reaction": with_reaction,
            "migd": float(np.mean(igd_hist)),
            "mhv": float(np.mean(hv_hist)),
            "final_igd": float(igd_hist[-1]),
            "final_hv": float(hv_hist[-1]),
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
            "pm_family": str(getattr(eda, "pm_family", cfg.pm_family)),
            "hv_hist": hv_hist,
            "igd_hist": igd_hist,
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
            "pm_family": cfg.pm_family,
            "hv_hist": np.array([]),
            "igd_hist": np.array([]),
        }


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "median": float(np.nanmedian(arr)),
    }


def summarize_pair(dynamic_runs: List[Dict[str, object]], static_runs: List[Dict[str, object]]) -> Dict[str, object]:
    keys = [
        "migd",
        "mhv",
        "final_igd",
        "final_hv",
        "precision",
        "recall",
        "f1",
        "runtime_sec",
        "local_search_calls",
    ]
    summary = {}
    for key in keys:
        dvals = [r[key] for r in dynamic_runs]
        svals = [r[key] for r in static_runs]
        summary[key] = {"dynamic": mean_std(dvals), "static": mean_std(svals)}

    if wilcoxon is not None and len(dynamic_runs) >= 6:
        d_migd = np.asarray([r["migd"] for r in dynamic_runs], dtype=float)
        s_migd = np.asarray([r["migd"] for r in static_runs], dtype=float)
        d_mhv = np.asarray([r["mhv"] for r in dynamic_runs], dtype=float)
        s_mhv = np.asarray([r["mhv"] for r in static_runs], dtype=float)

        try:
            p_migd = float(wilcoxon(d_migd, s_migd, alternative="less", zero_method="wilcox").pvalue)
        except Exception:
            p_migd = np.nan
        try:
            p_mhv = float(wilcoxon(d_mhv, s_mhv, alternative="greater", zero_method="wilcox").pvalue)
        except Exception:
            p_mhv = np.nan
    else:
        p_migd = np.nan
        p_mhv = np.nan

    summary["stats"] = {
        "wilcoxon_migd_p": p_migd,
        "wilcoxon_mhv_p": p_mhv,
    }
    return summary


def plot_mean_std_curves(
    dynamic_runs: List[Dict[str, object]],
    static_runs: List[Dict[str, object]],
    metric_key: str,
    title: str,
    out_path: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dyn = np.vstack([r[metric_key] for r in dynamic_runs])
    sta = np.vstack([r[metric_key] for r in static_runs])
    x = np.arange(dyn.shape[1])

    dyn_mean = np.mean(dyn, axis=0)
    dyn_std = np.std(dyn, axis=0)
    sta_mean = np.mean(sta, axis=0)
    sta_std = np.std(sta, axis=0)

    plt.figure(figsize=(11, 6))
    plt.plot(x, dyn_mean, label="DynamicSPEDA (reaccion)", color="#1f77b4")
    plt.fill_between(x, dyn_mean - dyn_std, dyn_mean + dyn_std, color="#1f77b4", alpha=0.20)
    plt.plot(x, sta_mean, label="StaticSPEDA (sin reaccion)", color="#ff7f0e")
    plt.fill_between(x, sta_mean - sta_std, sta_mean + sta_std, color="#ff7f0e", alpha=0.20)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_interactive_problem(
    problem_name: str,
    dynamic_runs: List[Dict[str, object]],
    static_runs: List[Dict[str, object]],
    out_html: str,
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    dyn_hv = np.vstack([r["hv_hist"] for r in dynamic_runs])
    sta_hv = np.vstack([r["hv_hist"] for r in static_runs])
    dyn_igd = np.vstack([r["igd_hist"] for r in dynamic_runs])
    sta_igd = np.vstack([r["igd_hist"] for r in static_runs])
    x = np.arange(dyn_hv.shape[1])

    def _mean_std(a):
        return np.mean(a, axis=0), np.std(a, axis=0)

    d_hv_m, d_hv_s = _mean_std(dyn_hv)
    s_hv_m, s_hv_s = _mean_std(sta_hv)
    d_ig_m, d_ig_s = _mean_std(dyn_igd)
    s_ig_m, s_ig_s = _mean_std(sta_igd)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hypervolume", "IGD"))

    def add_band(fig_obj, xvals, mean, std, name, color, row, col):
        upper = mean + std
        lower = mean - std
        fig_obj.add_trace(
            go.Scatter(
                x=np.concatenate([xvals, xvals[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=color.replace("1.0", "0.18"),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"{name} std",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig_obj.add_trace(
            go.Scatter(x=xvals, y=mean, mode="lines", name=name, line=dict(color=color)),
            row=row,
            col=col,
        )

    add_band(fig, x, d_hv_m, d_hv_s, "Dynamic", "rgba(31,119,180,1.0)", 1, 1)
    add_band(fig, x, s_hv_m, s_hv_s, "Static", "rgba(255,127,14,1.0)", 1, 1)
    add_band(fig, x, d_ig_m, d_ig_s, "Dynamic", "rgba(31,119,180,1.0)", 1, 2)
    add_band(fig, x, s_ig_m, s_ig_s, "Static", "rgba(255,127,14,1.0)", 1, 2)

    fig.update_layout(
        title=f"Interactive results | Robust SPEDA | {problem_name}",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
        width=1200,
        height=520,
    )
    fig.update_xaxes(title_text="Tiempo", row=1, col=1)
    fig.update_xaxes(title_text="Tiempo", row=1, col=2)
    fig.write_html(out_html)
    return True


def save_runs_csv(rows: List[Dict[str, object]], out_csv: str):
    cols = [
        "problem",
        "seed",
        "variant",
        "status",
        "error",
        "migd",
        "mhv",
        "final_igd",
        "final_hv",
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
        "pm_family",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def parse_problem_list(raw: str) -> List[str]:
    requested = [v.strip() for v in raw.split(",") if v.strip()]
    if (not requested) or (len(requested) == 1 and requested[0].lower() == "all"):
        return list(PROBLEM_REGISTRY.keys())

    selected = []
    for item in requested:
        if item in PROBLEM_REGISTRY and item not in selected:
            selected.append(item)
    return selected


def apply_preset(args):
    if args.preset == "quick":
        args.seeds = min(args.seeds, 3)
        args.horizon = min(args.horizon, 60)
        args.size_gen = min(args.size_gen, 60)
        args.n_variables = min(args.n_variables, 10)
    elif args.preset == "overnight":
        args.seeds = max(args.seeds, 16)
        args.horizon = max(args.horizon, 160)
        args.size_gen = max(args.size_gen, 120)
        args.n_variables = max(args.n_variables, 14)


def main():
    parser = argparse.ArgumentParser(description="Robust SPEDA dynamic benchmark suite (8 dynamic MO tests)")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--size-gen", type=int, default=90)
    parser.add_argument("--n-variables", type=int, default=12)
    parser.add_argument("--problems", type=str, default="all")
    parser.add_argument("--preset", choices=["quick", "medium", "overnight"], default="medium")
    parser.add_argument("--quick", action="store_true", help="Alias de --preset quick")
    parser.add_argument("--outdir", type=str, default="speda_dynamic_robust_suite")
    parser.add_argument("--l", type=float, default=6.0)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--pm-family", type=str, default="SPBN", choices=["SPBN", "GBN", "KDEBN", "MultiGauss"])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--disable-debski", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.preset = "quick"

    apply_preset(args)

    selected_problems = parse_problem_list(args.problems)
    if not selected_problems:
        raise ValueError("No valid problems selected. Check --problems values.")

    os.makedirs(args.outdir, exist_ok=True)

    base_cfg = SpedaRobustConfig(
        n_variables=args.n_variables,
        size_gen=args.size_gen,
        horizon=args.horizon,
        pm_family=args.pm_family,
        l=args.l,
        num_folds=args.num_folds,
        local_search_enabled=(not args.disable_debski),
    )

    all_rows = []
    summary = {
        "config": {
            "seeds": args.seeds,
            "run_config": base_cfg.__dict__,
            "selected_problems": selected_problems,
            "preset": args.preset,
            "no_plots": bool(args.no_plots),
            "no_interactive": bool(args.no_interactive),
        },
        "problems": {},
    }

    print("Robust SPEDA dynamic suite")
    print(f"Problems: {', '.join(selected_problems)}")
    print(f"Seeds: {args.seeds} | Horizon: {args.horizon} | Pop: {args.size_gen}")
    print(
        "Modules:",
        f"PM={base_cfg.pm_family}",
        f"Debski={'ON' if base_cfg.local_search_enabled else 'OFF'}",
    )

    for problem_name in selected_problems:
        print(f"\n[Problem] {problem_name}")
        cfg = tuned_cfg_for_problem(base_cfg, problem_name)

        dyn_by_seed = {}
        sta_by_seed = {}

        for seed in range(args.seeds):
            dyn = run_single(problem_name, cfg, seed=seed, with_reaction=True)
            sta = run_single(problem_name, cfg, seed=seed, with_reaction=False)

            all_rows.append({"problem": problem_name, "seed": seed, "variant": "dynamic", **dyn})
            all_rows.append({"problem": problem_name, "seed": seed, "variant": "static", **sta})

            if dyn["status"] == "ok":
                dyn_by_seed[seed] = dyn
            if sta["status"] == "ok":
                sta_by_seed[seed] = sta

        paired = sorted(set(dyn_by_seed.keys()) & set(sta_by_seed.keys()))
        dyn_runs = [dyn_by_seed[s] for s in paired]
        sta_runs = [sta_by_seed[s] for s in paired]

        block = {
            "n_requested": args.seeds,
            "n_success_dynamic": len(dyn_by_seed),
            "n_success_static": len(sta_by_seed),
            "n_paired": len(paired),
        }

        if len(paired) >= 2:
            s = summarize_pair(dyn_runs, sta_runs)
            block["summary"] = s

            print("  MIGD mean (dynamic/static):", f"{s['migd']['dynamic']['mean']:.4f} / {s['migd']['static']['mean']:.4f}")
            print("  MHV  mean (dynamic/static):", f"{s['mhv']['dynamic']['mean']:.4f} / {s['mhv']['static']['mean']:.4f}")
            print("  F1   mean (dynamic/static):", f"{s['f1']['dynamic']['mean']:.4f} / {s['f1']['static']['mean']:.4f}")

            if not args.no_plots:
                plot_mean_std_curves(
                    dyn_runs,
                    sta_runs,
                    metric_key="hv_hist",
                    title=f"HV mean +/- std | Robust SPEDA | {problem_name}",
                    out_path=os.path.join(args.outdir, f"hv_SPEDA_{problem_name}.png"),
                )
                plot_mean_std_curves(
                    dyn_runs,
                    sta_runs,
                    metric_key="igd_hist",
                    title=f"IGD mean +/- std | Robust SPEDA | {problem_name}",
                    out_path=os.path.join(args.outdir, f"igd_SPEDA_{problem_name}.png"),
                )

            if not args.no_interactive:
                ok = plot_interactive_problem(
                    problem_name=problem_name,
                    dynamic_runs=dyn_runs,
                    static_runs=sta_runs,
                    out_html=os.path.join(args.outdir, f"interactive_SPEDA_{problem_name}.html"),
                )
                block["interactive_html"] = bool(ok)
        else:
            block["summary"] = "Insufficient paired runs."
            print("  Insufficient paired runs for statistical summary.")

        summary["problems"][problem_name] = block

    csv_path = os.path.join(args.outdir, "per_run_metrics.csv")
    save_runs_csv(all_rows, csv_path)

    json_path = os.path.join(args.outdir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nRobust SPEDA suite completed.")
    print(f"- CSV:   {csv_path}")
    print(f"- JSON:  {json_path}")
    print(f"- Plots: {args.outdir}/igd_SPEDA_*.png and {args.outdir}/hv_SPEDA_*.png")
    print(f"- HTML:  {args.outdir}/interactive_SPEDA_*.html (if plotly is available)")


if __name__ == "__main__":
    main()
