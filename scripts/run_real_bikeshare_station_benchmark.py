#!/usr/bin/env python3
"""Run a small real station-level dynamic MOO benchmark on bike-share trips.

The benchmark replays hourly OD demand among real stations and evaluates simple
fleet-rebalancing controllers. The station-level ADR-EDA policy keeps the same
conceptual controller used in the thesis--detect change, update when needed,
otherwise reuse inherited EDA state--with either a compact dependency-free EDA
or an EDAspy/SPEDA backend selected by --eda-engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, cast


POLICIES = [
    "no_relocation",
    "static_twice_daily_rebalance",
    "full_rebalance",
    "rolling_horizon_rebalance",
    "selective_rebalance",
    "adr_safe_reuse_proxy",
    "eda_hourly_no_reuse",
    "eda_hourly_mo_no_reuse",
    "pymoo_nsga2_hourly_mo",
    "pymoo_dnsga2_hourly_mo",
    "adr_eda_station",
    "adr_eda_station_mo",
]

DEFAULT_STATION_SERVICE_FLOOR = 0.985
MO_SERVICE_TOLERANCE = 0.02


@dataclass(frozen=True)
class Station:
    name: str
    lat: float
    lng: float
    activity_share: float


@dataclass
class PolicyState:
    target: dict[str, float] | None = None
    eda_center: dict[str, float] | None = None
    eda_sigma: dict[str, float] | None = None
    last_update_hour: int = -10_000
    last_service_rate: float = 1.0
    last_adr_trigger_reason: str = ""
    last_adr_change_score: float = 0.0
    last_adr_change_detected: bool = False
    last_mo_front_size: int = 0
    last_mo_selected_objectives: dict[str, float] | None = None
    last_eda_engine: str = ""
    last_eda_backend_status: str = ""
    last_eda_backend_note: str = ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def haversine_km(a: Station, b: Station) -> float:
    radius = 6371.0
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = lat2 - lat1
    dlng = math.radians(b.lng - a.lng)
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2.0) ** 2
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(h)))


def urban_manhattan_km(a: Station, b: Station) -> float:
    radius = 6371.0
    dlat = math.radians(b.lat - a.lat)
    dlng = math.radians(b.lng - a.lng)
    mean_lat = math.radians((a.lat + b.lat) / 2.0)
    north_south = radius * dlat
    east_west = radius * math.cos(mean_lat) * dlng
    return abs(north_south) + abs(east_west)


def station_distance_km(
    a: Station,
    b: Station,
    *,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None = None,
) -> float:
    if distance_matrix is not None:
        row = distance_matrix.get(a.name, {})
        value = row.get(b.name)
        if value is not None:
            return max(0.0, float(value))
    if distance_model == "haversine":
        return haversine_km(a, b)
    if distance_model == "haversine_detour":
        return haversine_km(a, b) * max(1.0, float(distance_detour_factor))
    if distance_model == "urban_manhattan":
        return urban_manhattan_km(a, b) * max(1.0, float(distance_detour_factor))
    raise ValueError(f"Unknown distance model: {distance_model}")


def load_stations(profile_dir: Path, limit: int) -> list[Station]:
    rows = read_csv_rows(profile_dir / "station_summary.csv")[:limit]
    stations = []
    for row in rows:
        stations.append(
            Station(
                name=str(row["station"]),
                lat=float(row["lat"]),
                lng=float(row["lng"]),
                activity_share=float(row["activity_share"]),
            )
        )
    if not stations:
        raise ValueError(f"No stations found in {profile_dir}")
    return stations


def load_distance_matrix_json(path: Path, stations: list[Station]) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_distance = payload.get("distance_km", {})
    if not isinstance(raw_distance, dict):
        raise ValueError(f"Distance matrix {path} does not contain a distance_km object.")
    station_names = {station.name for station in stations}
    matrix: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for src in station_names:
        raw_row = raw_distance.get(src)
        if not isinstance(raw_row, dict):
            missing.append(src)
            continue
        row: dict[str, float] = {}
        for dst in station_names:
            value = raw_row.get(dst)
            if value is None:
                missing.append(f"{src}->{dst}")
                continue
            row[dst] = max(0.0, float(value))
        matrix[src] = row
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Distance matrix {path} is incomplete for selected stations: {preview}")
    return matrix


def parse_hour_of_day(value: str) -> int | None:
    try:
        return int(str(value).strip()[11:13])
    except (TypeError, ValueError):
        return None


def load_hourly_od(profile_dir: Path, station_names: set[str]) -> dict[str, list[tuple[str, str, int]]]:
    by_hour: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row in read_csv_rows(profile_dir / "hourly_od_top_stations.csv"):
        origin = str(row["origin"])
        dest = str(row["destination"])
        if origin not in station_names or dest not in station_names:
            continue
        by_hour[str(row["hour"])].append((origin, dest, int(row["trips"])))
    return dict(by_hour)


def initial_inventory(stations: list[Station], fleet_size: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    weights = [max(1e-9, s.activity_share) for s in stations]
    total = sum(weights)
    inv = {s.name: max(1, int(round(fleet_size * s.activity_share / total))) for s in stations}
    while sum(inv.values()) > fleet_size:
        candidates = [s.name for s in stations if inv[s.name] > 1]
        inv[rng.choice(candidates)] -= 1
    while sum(inv.values()) < fleet_size:
        pick = rng.choices([s.name for s in stations], weights=weights, k=1)[0]
        inv[pick] += 1
    return inv


def station_capacities(stations: list[Station], fleet_size: int, capacity_multiplier: float) -> dict[str, int]:
    base = max(4, int(math.ceil(float(fleet_size) / max(1, len(stations)))))
    return {s.name: max(base + 1, int(math.ceil(base * capacity_multiplier))) for s in stations}


def forecast_counts(
    hours: list[str],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    start_idx: int,
    lookahead_hours: int,
) -> tuple[Counter[str], Counter[str]]:
    dep: Counter[str] = Counter()
    arr: Counter[str] = Counter()
    for h in hours[start_idx : min(len(hours), start_idx + lookahead_hours)]:
        for origin, dest, trips in od_by_hour.get(h, []):
            dep[origin] += trips
            arr[dest] += trips
    return dep, arr


def od_request_total(
    *,
    hours: list[str],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    start_idx: int,
    lookahead_hours: int,
) -> int:
    total = 0
    for h in hours[start_idx : min(len(hours), start_idx + lookahead_hours)]:
        total += sum(int(trips) for _, _, trips in od_by_hour.get(h, []))
    return total


def averaged_od_rows(
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    source_hours: list[str],
) -> list[tuple[str, str, int]]:
    if not source_hours:
        return []
    counts: Counter[tuple[str, str]] = Counter()
    for hour in source_hours:
        for origin, dest, trips in od_by_hour.get(hour, []):
            counts[(origin, dest)] += int(trips)
    scale = 1.0 / float(len(source_hours))
    rows = []
    for (origin, dest), count in counts.items():
        trips = int(math.floor(float(count) * scale + 0.5))
        if trips > 0:
            rows.append((origin, dest, trips))
    return rows


def moving_average_od_rows(
    *,
    hours: list[str],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    start_idx: int,
    history_window: int,
) -> list[tuple[str, str, int]]:
    history_start = max(0, int(start_idx) - max(1, int(history_window)))
    return averaged_od_rows(od_by_hour, hours[history_start:start_idx])


def forecast_od_for_decision(
    *,
    mode: str,
    hours: list[str],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    start_idx: int,
    lookahead_hours: int,
    history_window: int,
) -> dict[str, list[tuple[str, str, int]]]:
    if mode == "oracle":
        return od_by_hour

    horizon = hours[start_idx : min(len(hours), start_idx + lookahead_hours)]
    forecast: dict[str, list[tuple[str, str, int]]] = {}
    moving_rows = moving_average_od_rows(
        hours=hours,
        od_by_hour=od_by_hour,
        start_idx=start_idx,
        history_window=history_window,
    )

    if mode == "moving_average":
        for hour in horizon:
            forecast[hour] = list(moving_rows)
        return forecast

    if mode == "historical_hour":
        prior_hours = hours[:start_idx]
        for hour in horizon:
            hour_of_day = parse_hour_of_day(hour)
            matching_hours = [
                candidate
                for candidate in prior_hours
                if parse_hour_of_day(candidate) == hour_of_day
            ]
            forecast[hour] = averaged_od_rows(od_by_hour, matching_hours) if matching_hours else list(moving_rows)
        return forecast

    raise ValueError(f"Unknown forecast mode: {mode}")


def desired_target(
    stations: list[Station],
    inventory: dict[str, int],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    total_bikes = float(sum(inventory.values()))
    for station in stations:
        name = station.name
        reserve = 0.20 * capacities[name]
        pressure = float(departures[name]) - 0.35 * float(arrivals[name])
        raw[name] = max(1.0, min(float(capacities[name] - 1), reserve + max(0.0, pressure)))
    raw_sum = sum(raw.values()) or 1.0
    scale = min(1.0, total_bikes / raw_sum)
    target = {name: max(0.0, min(float(capacities[name] - 1), value * scale)) for name, value in raw.items()}
    spare = total_bikes - sum(target.values())
    if spare > 0:
        ordered = sorted(stations, key=lambda s: departures[s.name], reverse=True)
        idx = 0
        while spare > 1e-6 and idx < 10_000:
            name = ordered[idx % len(ordered)].name
            room = float(capacities[name] - 1) - target[name]
            add = min(room, spare, 1.0)
            target[name] += add
            spare -= add
            idx += 1
            if add <= 1e-9 and idx >= len(ordered):
                break
    return target


def rolling_horizon_target(
    *,
    stations: list[Station],
    inventory: dict[str, int],
    capacities: dict[str, int],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    start_idx: int,
    lookahead_hours: int,
) -> dict[str, float]:
    """Fast MPC-style target from projected shortages and dock saturation.

    This is a deterministic operational baseline: every hour it uses the OD
    forecast available to the experiment mode, estimates which stations will
    run short or full under the current inventory, and computes a target
    inventory before the relocation step. It does not learn or reuse
    distributions.
    """
    horizon = hours[start_idx : min(len(hours), start_idx + lookahead_hours)]
    names = [station.name for station in stations]
    projected = {name: float(inventory.get(name, 0)) for name in names}
    dep_total: Counter[str] = Counter()
    arr_total: Counter[str] = Counter()
    shortage_exposure: Counter[str] = Counter()
    fullness_exposure: Counter[str] = Counter()
    volatility: Counter[str] = Counter()

    for hour in horizon:
        dep_h: Counter[str] = Counter()
        arr_h: Counter[str] = Counter()
        for origin, dest, trips in od_by_hour.get(hour, []):
            dep_h[origin] += trips
            arr_h[dest] += trips
        for name in names:
            dep = float(dep_h[name])
            arr = float(arr_h[name])
            dep_total[name] += dep_h[name]
            arr_total[name] += arr_h[name]
            volatility[name] += abs(dep - arr)
            projected[name] -= dep
            if projected[name] < 0.0:
                shortage_exposure[name] += -projected[name]
                projected[name] = 0.0
            projected[name] += arr
            cap = float(capacities[name])
            if projected[name] > cap:
                fullness_exposure[name] += projected[name] - cap
                projected[name] = cap

    base_departures, base_arrivals = forecast_counts(
        hours, od_by_hour, start_idx, lookahead_hours
    )
    base = desired_target(stations, inventory, capacities, base_departures, base_arrivals)
    raw: dict[str, float] = {}
    for station in stations:
        name = station.name
        cap = float(capacities[name])
        net_need = float(dep_total[name]) - 0.70 * float(arr_total[name])
        target = (
            0.58 * float(base[name])
            + 0.42 * max(0.0, 0.22 * cap + net_need)
            + 0.80 * float(shortage_exposure[name])
            + 0.08 * float(volatility[name])
            - 0.65 * float(fullness_exposure[name])
        )
        raw[name] = max(0.0, min(cap - 1.0, target))
    return project_target_to_fleet(
        stations, raw, capacities, total_bikes=float(sum(inventory.values()))
    )


def project_target_to_fleet(
    stations: list[Station],
    raw: dict[str, float],
    capacities: dict[str, int],
    total_bikes: float,
) -> dict[str, float]:
    target = {
        station.name: max(0.0, min(float(capacities[station.name] - 1), float(raw.get(station.name, 0.0))))
        for station in stations
    }
    current = sum(target.values())
    if current > total_bikes and current > 1e-9:
        scale = total_bikes / current
        target = {name: value * scale for name, value in target.items()}
    spare = total_bikes - sum(target.values())
    ordered = sorted(stations, key=lambda station: target[station.name], reverse=True)
    idx = 0
    while spare > 1e-6 and idx < 10_000:
        name = ordered[idx % len(ordered)].name
        room = float(capacities[name] - 1) - target[name]
        add = min(room, spare, 1.0)
        target[name] += add
        spare -= add
        idx += 1
        if add <= 1e-9 and idx >= len(ordered):
            break
    return target


def target_to_vector(stations: list[Station], target: dict[str, float]) -> list[float]:
    return [float(target.get(station.name, 0.0)) for station in stations]


def repair_target_vector(
    values: Iterable[float],
    stations: list[Station],
    capacities: dict[str, int],
    total_bikes: float,
) -> dict[str, float]:
    raw = {
        station.name: max(
            0.0,
            min(float(capacities[station.name] - 1), float(value)),
        )
        for station, value in zip(stations, values)
    }
    return project_target_to_fleet(stations, raw, capacities, total_bikes)


def fit_compact_state_from_elites(
    *,
    state: PolicyState,
    stations: list[Station],
    elite_targets: list[dict[str, float]],
    eda_initial_sigma: float,
) -> None:
    if not elite_targets:
        return
    center: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for station in stations:
        name = station.name
        values = [float(candidate[name]) for candidate in elite_targets]
        center[name] = mean(values)
        sigma[name] = max(0.20, pstdev(values) if len(values) > 1 else max(0.25, eda_initial_sigma))
    state.eda_center = dict(center)
    state.eda_sigma = dict(sigma)


def realbike_edaspy_import_error(exc: Exception) -> ImportError:
    return ImportError(
        "EDAspy engine requested for RealBike, but the EDAspy/SPEDA stack is not importable. "
        "Run from the configured Docker/EDAspy environment, or use --eda-engine compact for "
        f"the dependency-free fallback. Root cause: {type(exc).__name__}: {exc}"
    )


def ensure_realbike_edaspy_importable() -> None:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "EDAspy",
        Path("/app/EDAspy"),
    ]
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def enable_realbike_edaspy_small_sample_patch() -> None:
    try:
        from EDAspy.optimization.custom.probabilistic_models.kde_bayesian_network import KDEBN
        from EDAspy.optimization.custom.probabilistic_models.semiparametric_bayesian_network import SPBN
    except Exception:
        return

    for cls in (SPBN, KDEBN):
        if getattr(cls, "_realbike_adaptive_folds_patch", False):
            continue

        original_learn = cls.learn

        def patched_learn(
            self,
            dataset,
            num_folds: int = 10,
            *args,
            _original_learn=original_learn,
            **kwargs,
        ):
            import numpy as np

            data_array = np.asarray(dataset, dtype=float)
            n_samples = int(data_array.shape[0])
            if data_array.ndim == 2 and 0 < n_samples < 6:
                reps = int(math.ceil(6.0 / float(n_samples)))
                padded = np.vstack([data_array] * reps)[:6].copy()
                scale = np.maximum(np.std(data_array, axis=0), 1.0)
                jitter = np.random.normal(0.0, 1e-6 * scale, size=padded.shape)
                dataset = padded + jitter
                n_samples = int(np.asarray(dataset).shape[0])
            safe_folds = int(num_folds)
            if n_samples >= 3:
                safe_folds = max(2, min(int(num_folds), n_samples - 1))
            return _original_learn(self, dataset, num_folds=safe_folds, *args, **kwargs)

        cls.learn = patched_learn
        cls._realbike_adaptive_folds_patch = True


def load_realbike_edaspy_classes():
    ensure_realbike_edaspy_importable()
    try:
        from EDAspy.optimization.multivariate.speda import SPEDA
        from EDAspy.optimization.selection import NSGA2Selection
    except Exception as exc:
        raise realbike_edaspy_import_error(exc) from exc
    enable_realbike_edaspy_small_sample_patch()
    return SPEDA, NSGA2Selection


def build_edaspy_init_data(
    *,
    rng: random.Random,
    stations: list[Station],
    capacities: dict[str, int],
    total_bikes: float,
    center: dict[str, float],
    sigma: dict[str, float],
    heuristic: dict[str, float],
    incumbent: dict[str, float] | None,
    population_size: int,
) -> object:
    import numpy as np

    rows: list[dict[str, float]] = [
        project_target_to_fleet(stations, center, capacities, total_bikes),
        project_target_to_fleet(stations, heuristic, capacities, total_bikes),
    ]
    if incumbent is not None:
        rows.append(project_target_to_fleet(stations, incumbent, capacities, total_bikes))
    while len(rows) < max(3, int(population_size)):
        raw = {}
        for station in stations:
            name = station.name
            raw[name] = rng.gauss(float(center[name]), max(0.25, float(sigma.get(name, 1.0))))
        rows.append(project_target_to_fleet(stations, raw, capacities, total_bikes))
    return np.asarray([target_to_vector(stations, row) for row in rows[: max(3, int(population_size))]], dtype=float)


def update_state_from_edaspy_elites(
    *,
    state: PolicyState,
    stations: list[Station],
    capacities: dict[str, int],
    total_bikes: float,
    elite_vectors: object,
    eda_initial_sigma: float,
) -> None:
    import numpy as np

    elite_array = np.asarray(elite_vectors, dtype=float)
    if elite_array.ndim != 2 or elite_array.shape[0] == 0:
        return
    elite_targets = [
        repair_target_vector(row, stations, capacities, total_bikes)
        for row in elite_array
    ]
    fit_compact_state_from_elites(
        state=state,
        stations=stations,
        elite_targets=elite_targets,
        eda_initial_sigma=eda_initial_sigma,
    )


def optimize_target_with_edaspy_scalar(
    *,
    state: PolicyState,
    rng: random.Random,
    stations: list[Station],
    inventory: dict[str, int],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    hour_idx: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    service_reward: float,
    rejection_penalty: float,
    relocation_km_cost: float,
    relocation_action_cost: float,
    eda_population: int,
    eda_generations: int,
    eda_elite_frac: float,
    eda_initial_sigma: float,
    eda_imbalance_penalty: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    import numpy as np

    SPEDA, _ = load_realbike_edaspy_classes()
    total_bikes = float(sum(inventory.values()))
    heuristic = desired_target(stations, inventory, capacities, departures, arrivals)
    center = dict(state.eda_center or heuristic)
    if state.eda_center is not None:
        center = {
            station.name: 0.55 * float(state.eda_center.get(station.name, heuristic[station.name]))
            + 0.45 * float(heuristic[station.name])
            for station in stations
        }
    center = project_target_to_fleet(stations, center, capacities, total_bikes)
    sigma = dict(
        state.eda_sigma
        or {station.name: max(0.75, eda_initial_sigma * max(1.0, center[station.name])) for station in stations}
    )
    init_data = build_edaspy_init_data(
        rng=rng,
        stations=stations,
        capacities=capacities,
        total_bikes=total_bikes,
        center=center,
        sigma=sigma,
        heuristic=heuristic,
        incumbent=state.target,
        population_size=eda_population,
    )
    lower = np.zeros(len(stations), dtype=float)
    upper = np.asarray([float(capacities[station.name] - 1) for station in stations], dtype=float)
    np.random.seed(rng.randrange(1, 2**31 - 1))
    optimizer = SPEDA(
        size_gen=max(3, int(eda_population)),
        max_iter=max(1, int(eda_generations)),
        dead_iter=max(1, int(eda_generations)),
        n_variables=len(stations),
        lower_bound=lower,
        upper_bound=upper,
        l=max(1.0, float(eda_generations)),
        alpha=max(0.05, min(0.95, float(eda_elite_frac))),
        disp=False,
        parallelize=False,
        init_data=init_data,
        w_noise=max(0.0, min(1.0, float(eda_initial_sigma) * 0.10)),
    )

    def cost(vector) -> float:
        target = repair_target_vector(vector, stations, capacities, total_bikes)
        return -replay_lookahead_with_target(
            target=target,
            inventory=inventory,
            capacities=capacities,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            service_reward=service_reward,
            rejection_penalty=rejection_penalty,
            relocation_km_cost=relocation_km_cost,
            relocation_action_cost=relocation_action_cost,
            imbalance_penalty=eda_imbalance_penalty,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )

    result = optimizer.minimize(cost, output_runtime=False)
    best_target = repair_target_vector(result.best_ind, stations, capacities, total_bikes)
    update_state_from_edaspy_elites(
        state=state,
        stations=stations,
        capacities=capacities,
        total_bikes=total_bikes,
        elite_vectors=getattr(optimizer, "elite_temp", []),
        eda_initial_sigma=eda_initial_sigma,
    )
    if state.eda_center is None:
        state.eda_center = dict(best_target)
    state.last_eda_engine = "edaspy"
    state.last_eda_backend_status = "SPEDA_scalar_ok"
    state.last_eda_backend_note = f"best_cost={float(result.best_cost):.6f}"
    return best_target


def optimize_target_with_edaspy_mo(
    *,
    state: PolicyState,
    rng: random.Random,
    stations: list[Station],
    inventory: dict[str, int],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    hour_idx: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    eda_population: int,
    eda_generations: int,
    eda_elite_frac: float,
    eda_initial_sigma: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    import numpy as np

    SPEDA, NSGA2Selection = load_realbike_edaspy_classes()
    total_bikes = float(sum(inventory.values()))
    heuristic = desired_target(stations, inventory, capacities, departures, arrivals)
    center = dict(state.eda_center or heuristic)
    if state.eda_center is not None:
        center = {
            station.name: 0.55 * float(state.eda_center.get(station.name, heuristic[station.name]))
            + 0.45 * float(heuristic[station.name])
            for station in stations
        }
    center = project_target_to_fleet(stations, center, capacities, total_bikes)
    sigma = dict(
        state.eda_sigma
        or {station.name: max(0.75, eda_initial_sigma * max(1.0, center[station.name])) for station in stations}
    )
    init_data = build_edaspy_init_data(
        rng=rng,
        stations=stations,
        capacities=capacities,
        total_bikes=total_bikes,
        center=center,
        sigma=sigma,
        heuristic=heuristic,
        incumbent=state.target,
        population_size=eda_population,
    )
    lower = np.zeros(len(stations), dtype=float)
    upper = np.asarray([float(capacities[station.name] - 1) for station in stations], dtype=float)
    np.random.seed(rng.randrange(1, 2**31 - 1))
    optimizer = SPEDA(
        size_gen=max(3, int(eda_population)),
        max_iter=max(1, int(eda_generations)),
        dead_iter=max(1, int(eda_generations)),
        n_variables=len(stations),
        lower_bound=lower,
        upper_bound=upper,
        l=max(1.0, float(eda_generations)),
        alpha=max(0.05, min(0.95, float(eda_elite_frac))),
        disp=False,
        parallelize=False,
        init_data=init_data,
        w_noise=max(0.0, min(1.0, float(eda_initial_sigma) * 0.10)),
    )
    optimizer.selection_strategy = NSGA2Selection()

    def objective_min(vector):
        target = repair_target_vector(vector, stations, capacities, total_bikes)
        objectives = replay_lookahead_objectives_with_target(
            target=target,
            inventory=inventory,
            capacities=capacities,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
        return np.asarray(
            [
                -float(objectives["service_rate"]),
                float(objectives["relocation_km"]),
                float(objectives["relocated_bikes"]),
            ],
            dtype=float,
        )

    result = optimizer.minimize(objective_min, output_runtime=False)
    pareto_set = np.asarray(result.get("pareto_set", np.empty((0, len(stations)))), dtype=float)
    rows: list[dict[str, object]] = []
    for vector in pareto_set:
        candidate = repair_target_vector(vector, stations, capacities, total_bikes)
        objectives = replay_lookahead_objectives_with_target(
            target=candidate,
            inventory=inventory,
            capacities=capacities,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
        rows.append({"candidate": candidate, "objectives": objectives})
    if not rows:
        fallback = repair_target_vector(init_data[0], stations, capacities, total_bikes)
        rows.append(
            {
                "candidate": fallback,
                "objectives": replay_lookahead_objectives_with_target(
                    target=fallback,
                    inventory=inventory,
                    capacities=capacities,
                    stations_by_name=stations_by_name,
                    od_by_hour=od_by_hour,
                    hours=hours,
                    start_idx=hour_idx,
                    lookahead_hours=lookahead_hours,
                    relocation_budget_bikes=relocation_budget_bikes,
                    distance_model=distance_model,
                    distance_detour_factor=distance_detour_factor,
                    distance_matrix=distance_matrix,
                ),
            }
        )
    selected = select_mo_compromise(rows)
    final_front = non_dominated_rows(rows)
    objectives = mo_row_objectives(selected)
    state.last_mo_front_size = len(final_front)
    state.last_mo_selected_objectives = {str(key): float(value) for key, value in objectives.items()}
    elite_count = max(2, int(round(float(eda_population) * float(eda_elite_frac))))
    elites = [mo_row_candidate(row) for row in select_mo_elites(rows, elite_count)]
    fit_compact_state_from_elites(
        state=state,
        stations=stations,
        elite_targets=elites,
        eda_initial_sigma=eda_initial_sigma,
    )
    state.last_eda_engine = "edaspy"
    state.last_eda_backend_status = "SPEDA_mo_ok"
    state.last_eda_backend_note = f"pareto_size={len(rows)}"
    return project_target_to_fleet(stations, mo_row_candidate(selected), capacities, total_bikes)


def replay_lookahead_with_target(
    *,
    target: dict[str, float],
    inventory: dict[str, int],
    capacities: dict[str, int],
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    start_idx: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    service_reward: float,
    rejection_penalty: float,
    relocation_km_cost: float,
    relocation_action_cost: float,
    imbalance_penalty: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
) -> float:
    sim_inventory = dict(inventory)
    served = rejected = relocated = 0
    relocated_km = 0.0
    imbalance = 0.0
    horizon = hours[start_idx : min(len(hours), start_idx + lookahead_hours)]
    for hour in horizon:
        moved, moved_km = rebalance_to_target(
            sim_inventory,
            target,
            stations_by_name,
            capacities,
            budget_bikes=relocation_budget_bikes,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
        relocated += moved
        relocated_km += moved_km
        _, hour_served, hour_empty, hour_full = replay_hour(
            sim_inventory,
            capacities,
            od_by_hour.get(hour, []),
        )
        served += hour_served
        rejected += hour_empty + hour_full
        imbalance += inventory_imbalance(sim_inventory, capacities)
    return (
        service_reward * float(served)
        - rejection_penalty * float(rejected)
        - relocation_km_cost * relocated_km
        - relocation_action_cost * float(relocated)
        - imbalance_penalty * imbalance
    )


def replay_lookahead_objectives_with_target(
    *,
    target: dict[str, float],
    inventory: dict[str, int],
    capacities: dict[str, int],
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    start_idx: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    sim_inventory = dict(inventory)
    total = served = rejected = relocated = 0
    relocated_km = 0.0
    imbalance = 0.0
    horizon = hours[start_idx : min(len(hours), start_idx + lookahead_hours)]
    for hour in horizon:
        moved, moved_km = rebalance_to_target(
            sim_inventory,
            target,
            stations_by_name,
            capacities,
            budget_bikes=relocation_budget_bikes,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
        relocated += moved
        relocated_km += moved_km
        hour_total, hour_served, hour_empty, hour_full = replay_hour(
            sim_inventory,
            capacities,
            od_by_hour.get(hour, []),
        )
        total += hour_total
        served += hour_served
        rejected += hour_empty + hour_full
        imbalance += inventory_imbalance(sim_inventory, capacities)
    service_rate = 1.0 if total == 0 else float(served) / float(total)
    mean_imbalance = imbalance / max(1, len(horizon))
    return {
        "served": float(served),
        "service_rate": service_rate,
        "rejected": float(rejected),
        "relocation_km": relocated_km,
        "relocated_bikes": float(relocated),
        "mean_inventory_imbalance": mean_imbalance,
    }


def mo_dominates(lhs: dict[str, float], rhs: dict[str, float], eps: float = 1e-12) -> bool:
    no_worse = (
        lhs["service_rate"] >= rhs["service_rate"] - eps
        and lhs["relocation_km"] <= rhs["relocation_km"] + eps
        and lhs["relocated_bikes"] <= rhs["relocated_bikes"] + eps
    )
    strictly_better = (
        lhs["service_rate"] > rhs["service_rate"] + eps
        or lhs["relocation_km"] < rhs["relocation_km"] - eps
        or lhs["relocated_bikes"] < rhs["relocated_bikes"] - eps
    )
    return no_worse and strictly_better


def mo_row_objectives(row: dict[str, object]) -> dict[str, float]:
    return cast(dict[str, float], row["objectives"])


def mo_row_candidate(row: dict[str, object]) -> dict[str, float]:
    return cast(dict[str, float], row["candidate"])


def non_dominated_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    front: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        objectives = mo_row_objectives(row)
        dominated = False
        for other_idx, other in enumerate(rows):
            if idx == other_idx:
                continue
            if mo_dominates(mo_row_objectives(other), objectives):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def mo_reference(rows: list[dict[str, object]]) -> dict[str, float]:
    service_values = [float(mo_row_objectives(row)["service_rate"]) for row in rows]
    km_values = [float(mo_row_objectives(row)["relocation_km"]) for row in rows]
    bike_values = [float(mo_row_objectives(row)["relocated_bikes"]) for row in rows]
    return {
        "max_service_rate": max(service_values),
        "min_relocation_km": min(km_values),
        "min_relocated_bikes": min(bike_values),
        "service_range": max(1e-9, max(service_values) - min(service_values)),
        "km_range": max(1e-9, max(km_values) - min(km_values)),
        "bike_range": max(1e-9, max(bike_values) - min(bike_values)),
    }


def mo_ideal_distance(objectives: dict[str, float], reference: dict[str, float]) -> float:
    service_gap = (reference["max_service_rate"] - float(objectives["service_rate"])) / reference["service_range"]
    km_gap = (float(objectives["relocation_km"]) - reference["min_relocation_km"]) / reference["km_range"]
    bike_gap = (float(objectives["relocated_bikes"]) - reference["min_relocated_bikes"]) / reference["bike_range"]
    return math.sqrt(service_gap * service_gap + km_gap * km_gap + bike_gap * bike_gap)


def select_mo_compromise(rows: list[dict[str, object]]) -> dict[str, object]:
    front = non_dominated_rows(rows)
    reference = mo_reference(rows)
    service_guard = [
        row
        for row in front
        if float(mo_row_objectives(row)["service_rate"]) >= reference["max_service_rate"] - MO_SERVICE_TOLERANCE
    ]
    if service_guard:
        front = service_guard
    return min(
        front,
        key=lambda row: (
            mo_ideal_distance(mo_row_objectives(row), reference),
            float(mo_row_objectives(row)["rejected"]),
            float(mo_row_objectives(row)["mean_inventory_imbalance"]),
        ),
    )


def select_mo_elites(rows: list[dict[str, object]], elite_count: int) -> list[dict[str, object]]:
    reference = mo_reference(rows)
    front = sorted(
        non_dominated_rows(rows),
        key=lambda row: (
            mo_ideal_distance(mo_row_objectives(row), reference),
            float(mo_row_objectives(row)["rejected"]),
            float(mo_row_objectives(row)["mean_inventory_imbalance"]),
        ),
    )
    elites = front[:elite_count]
    seen = {id(row) for row in elites}
    if len(elites) < elite_count:
        fallback = sorted(
            (row for row in rows if id(row) not in seen),
            key=lambda row: (
                mo_ideal_distance(mo_row_objectives(row), reference),
                float(mo_row_objectives(row)["rejected"]),
                float(mo_row_objectives(row)["mean_inventory_imbalance"]),
            ),
        )
        elites.extend(fallback[: elite_count - len(elites)])
    return elites


def optimize_target_with_eda(
    *,
    state: PolicyState,
    rng: random.Random,
    stations: list[Station],
    inventory: dict[str, int],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    hour_idx: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    service_reward: float,
    rejection_penalty: float,
    relocation_km_cost: float,
    relocation_action_cost: float,
    eda_population: int,
    eda_generations: int,
    eda_elite_frac: float,
    eda_initial_sigma: float,
    eda_imbalance_penalty: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
    eda_engine: str,
) -> dict[str, float]:
    if eda_engine == "edaspy":
        return optimize_target_with_edaspy_scalar(
            state=state,
            rng=rng,
            stations=stations,
            inventory=inventory,
            capacities=capacities,
            departures=departures,
            arrivals=arrivals,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            hour_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            service_reward=service_reward,
            rejection_penalty=rejection_penalty,
            relocation_km_cost=relocation_km_cost,
            relocation_action_cost=relocation_action_cost,
            eda_population=eda_population,
            eda_generations=eda_generations,
            eda_elite_frac=eda_elite_frac,
            eda_initial_sigma=eda_initial_sigma,
            eda_imbalance_penalty=eda_imbalance_penalty,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
    state.last_eda_engine = "compact"
    state.last_eda_backend_status = "compact_scalar_ok"
    state.last_eda_backend_note = ""
    total_bikes = float(sum(inventory.values()))
    heuristic = desired_target(stations, inventory, capacities, departures, arrivals)
    center = dict(state.eda_center or heuristic)
    if state.eda_center is not None:
        center = {
            station.name: 0.55 * float(state.eda_center.get(station.name, heuristic[station.name]))
            + 0.45 * float(heuristic[station.name])
            for station in stations
        }
    center = project_target_to_fleet(stations, center, capacities, total_bikes)
    sigma = dict(
        state.eda_sigma
        or {station.name: max(0.75, eda_initial_sigma * max(1.0, center[station.name])) for station in stations}
    )
    elite_count = max(2, int(round(float(eda_population) * float(eda_elite_frac))))
    best_target = center
    best_score = -float("inf")

    for _ in range(max(1, int(eda_generations))):
        candidates: list[dict[str, float]] = [center, heuristic]
        if state.target is not None:
            candidates.append(project_target_to_fleet(stations, state.target, capacities, total_bikes))
        while len(candidates) < max(3, int(eda_population)):
            raw = {}
            for station in stations:
                name = station.name
                raw[name] = rng.gauss(float(center[name]), max(0.25, float(sigma.get(name, 1.0))))
            candidates.append(project_target_to_fleet(stations, raw, capacities, total_bikes))

        scored = []
        for candidate in candidates:
            score = replay_lookahead_with_target(
                target=candidate,
                inventory=inventory,
                capacities=capacities,
                stations_by_name=stations_by_name,
                od_by_hour=od_by_hour,
                hours=hours,
                start_idx=hour_idx,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                service_reward=service_reward,
                rejection_penalty=rejection_penalty,
                relocation_km_cost=relocation_km_cost,
                relocation_action_cost=relocation_action_cost,
                imbalance_penalty=eda_imbalance_penalty,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
            )
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > best_score:
            best_score, best_target = scored[0]
        elites = [candidate for _, candidate in scored[:elite_count]]
        center = {}
        sigma = {}
        for station in stations:
            name = station.name
            values = [float(candidate[name]) for candidate in elites]
            center[name] = mean(values)
            sigma[name] = max(0.20, pstdev(values) if len(values) > 1 else max(0.25, eda_initial_sigma))
        center = project_target_to_fleet(stations, center, capacities, total_bikes)

    state.eda_center = dict(center)
    state.eda_sigma = dict(sigma)
    return project_target_to_fleet(stations, best_target, capacities, total_bikes)


def optimize_target_with_eda_mo(
    *,
    state: PolicyState,
    rng: random.Random,
    stations: list[Station],
    inventory: dict[str, int],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    hour_idx: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    eda_population: int,
    eda_generations: int,
    eda_elite_frac: float,
    eda_initial_sigma: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
    eda_engine: str,
) -> dict[str, float]:
    if eda_engine == "edaspy":
        return optimize_target_with_edaspy_mo(
            state=state,
            rng=rng,
            stations=stations,
            inventory=inventory,
            capacities=capacities,
            departures=departures,
            arrivals=arrivals,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            hour_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            eda_population=eda_population,
            eda_generations=eda_generations,
            eda_elite_frac=eda_elite_frac,
            eda_initial_sigma=eda_initial_sigma,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
    state.last_eda_engine = "compact"
    state.last_eda_backend_status = "compact_mo_ok"
    state.last_eda_backend_note = ""
    total_bikes = float(sum(inventory.values()))
    heuristic = desired_target(stations, inventory, capacities, departures, arrivals)
    center = dict(state.eda_center or heuristic)
    if state.eda_center is not None:
        center = {
            station.name: 0.55 * float(state.eda_center.get(station.name, heuristic[station.name]))
            + 0.45 * float(heuristic[station.name])
            for station in stations
        }
    center = project_target_to_fleet(stations, center, capacities, total_bikes)
    sigma = dict(
        state.eda_sigma
        or {station.name: max(0.75, eda_initial_sigma * max(1.0, center[station.name])) for station in stations}
    )
    elite_count = max(2, int(round(float(eda_population) * float(eda_elite_frac))))
    all_rows: list[dict[str, object]] = []

    for _ in range(max(1, int(eda_generations))):
        candidates: list[dict[str, float]] = [center, heuristic]
        if state.target is not None:
            candidates.append(project_target_to_fleet(stations, state.target, capacities, total_bikes))
        while len(candidates) < max(3, int(eda_population)):
            raw = {}
            for station in stations:
                name = station.name
                raw[name] = rng.gauss(float(center[name]), max(0.25, float(sigma.get(name, 1.0))))
            candidates.append(project_target_to_fleet(stations, raw, capacities, total_bikes))

        rows: list[dict[str, object]] = []
        for candidate in candidates:
            objectives = replay_lookahead_objectives_with_target(
                target=candidate,
                inventory=inventory,
                capacities=capacities,
                stations_by_name=stations_by_name,
                od_by_hour=od_by_hour,
                hours=hours,
                start_idx=hour_idx,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
            )
            rows.append({"candidate": candidate, "objectives": objectives})
        all_rows.extend(rows)

        elites = [mo_row_candidate(row) for row in select_mo_elites(rows, elite_count)]
        center = {}
        sigma = {}
        for station in stations:
            name = station.name
            values = [float(candidate[name]) for candidate in elites]
            center[name] = mean(values)
            sigma[name] = max(0.20, pstdev(values) if len(values) > 1 else max(0.25, eda_initial_sigma))
        center = project_target_to_fleet(stations, center, capacities, total_bikes)

    selected = select_mo_compromise(all_rows)
    final_front = non_dominated_rows(all_rows)
    objectives = mo_row_objectives(selected)
    state.last_mo_front_size = len(final_front)
    state.last_mo_selected_objectives = {
        str(key): float(value) for key, value in objectives.items()
    }
    state.eda_center = dict(center)
    state.eda_sigma = dict(sigma)
    return project_target_to_fleet(stations, mo_row_candidate(selected), capacities, total_bikes)


def optimize_target_with_pymoo_mo(
    *,
    state: PolicyState,
    rng: random.Random,
    algorithm_name: str,
    stations: list[Station],
    inventory: dict[str, int],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    hour_idx: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    eda_population: int,
    eda_generations: int,
    eda_initial_sigma: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    import numpy as np

    try:
        from pymoo.algorithms.moo.dnsga2 import DNSGA2
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.sampling import Sampling
        from pymoo.optimize import minimize
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        raise RuntimeError(
            "pymoo baseline requested but pymoo is not importable. "
            "Install pymoo in the configured Docker environment."
        ) from exc

    total_bikes = float(sum(inventory.values()))
    heuristic = desired_target(stations, inventory, capacities, departures, arrivals)
    center = dict(state.eda_center or heuristic)
    if state.eda_center is not None:
        center = {
            station.name: 0.55 * float(state.eda_center.get(station.name, heuristic[station.name]))
            + 0.45 * float(heuristic[station.name])
            for station in stations
        }
    center = project_target_to_fleet(stations, center, capacities, total_bikes)
    sigma = dict(
        state.eda_sigma
        or {station.name: max(0.75, eda_initial_sigma * max(1.0, center[station.name])) for station in stations}
    )
    init_data = build_edaspy_init_data(
        rng=rng,
        stations=stations,
        capacities=capacities,
        total_bikes=total_bikes,
        center=center,
        sigma=sigma,
        heuristic=heuristic,
        incumbent=state.target,
        population_size=eda_population,
    )
    lower = np.zeros(len(stations), dtype=float)
    upper = np.asarray([float(capacities[station.name] - 1) for station in stations], dtype=float)

    class RealBikeSeededSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            rows = np.asarray(init_data, dtype=float)
            if rows.ndim != 2 or rows.shape[1] != len(stations):
                rows = np.empty((0, len(stations)), dtype=float)
            samples = [np.asarray(row, dtype=float) for row in rows[:n_samples]]
            while len(samples) < n_samples:
                raw = []
                for station in stations:
                    name = station.name
                    raw.append(rng.gauss(float(center[name]), max(0.25, float(sigma.get(name, 1.0)))))
                samples.append(np.asarray(raw, dtype=float))
            return np.clip(np.asarray(samples[:n_samples], dtype=float), lower, upper)

    class RealBikeTargetProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(stations), n_obj=3, xl=lower, xu=upper)

        def _evaluate(self, x, out, *args, **kwargs):
            target = repair_target_vector(x, stations, capacities, total_bikes)
            objectives = replay_lookahead_objectives_with_target(
                target=target,
                inventory=inventory,
                capacities=capacities,
                stations_by_name=stations_by_name,
                od_by_hour=od_by_hour,
                hours=hours,
                start_idx=hour_idx,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
            )
            out["F"] = np.asarray(
                [
                    -float(objectives["service_rate"]),
                    float(objectives["relocation_km"]),
                    float(objectives["relocated_bikes"]),
                ],
                dtype=float,
            )

    algo_key = str(algorithm_name).strip().lower()
    common_kwargs = {
        "pop_size": max(4, int(eda_population)),
        "sampling": RealBikeSeededSampling(),
    }
    if algo_key == "dnsga2":
        algorithm = DNSGA2(version="B", **common_kwargs)
        backend_status = "pymoo_DNSGA2_mo_ok"
    elif algo_key == "nsga2":
        algorithm = NSGA2(**common_kwargs)
        backend_status = "pymoo_NSGA2_mo_ok"
    else:
        raise ValueError(f"Unsupported pymoo algorithm: {algorithm_name}")

    np.random.seed(rng.randrange(1, 2**31 - 1))
    result = minimize(
        RealBikeTargetProblem(),
        algorithm,
        termination=("n_gen", max(1, int(eda_generations))),
        seed=rng.randrange(1, 2**31 - 1),
        verbose=False,
        save_history=False,
    )
    candidates = []
    xs = np.asarray(getattr(result, "X", np.empty((0, len(stations)))), dtype=float)
    if xs.ndim == 1 and xs.size:
        xs = xs.reshape(1, -1)
    for vector in xs:
        candidate = repair_target_vector(vector, stations, capacities, total_bikes)
        objectives = replay_lookahead_objectives_with_target(
            target=candidate,
            inventory=inventory,
            capacities=capacities,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
        candidates.append({"candidate": candidate, "objectives": objectives})
    if not candidates:
        fallback = repair_target_vector(init_data[0], stations, capacities, total_bikes)
        candidates.append(
            {
                "candidate": fallback,
                "objectives": replay_lookahead_objectives_with_target(
                    target=fallback,
                    inventory=inventory,
                    capacities=capacities,
                    stations_by_name=stations_by_name,
                    od_by_hour=od_by_hour,
                    hours=hours,
                    start_idx=hour_idx,
                    lookahead_hours=lookahead_hours,
                    relocation_budget_bikes=relocation_budget_bikes,
                    distance_model=distance_model,
                    distance_detour_factor=distance_detour_factor,
                    distance_matrix=distance_matrix,
                ),
            }
        )
    selected = select_mo_compromise(candidates)
    final_front = non_dominated_rows(candidates)
    objectives = mo_row_objectives(selected)
    state.last_mo_front_size = len(final_front)
    state.last_mo_selected_objectives = {str(key): float(value) for key, value in objectives.items()}
    fit_compact_state_from_elites(
        state=state,
        stations=stations,
        elite_targets=[mo_row_candidate(row) for row in select_mo_elites(candidates, max(2, len(candidates) // 4))],
        eda_initial_sigma=eda_initial_sigma,
    )
    state.last_eda_engine = "pymoo"
    state.last_eda_backend_status = backend_status
    state.last_eda_backend_note = f"pareto_size={len(candidates)}"
    return project_target_to_fleet(stations, mo_row_candidate(selected), capacities, total_bikes)


def station_adr_change_info(
    *,
    pressure: float,
    drift: float,
    last_service_rate: float,
    adr_high_pressure: float,
    adr_drift_threshold: float,
    adr_service_floor: float,
) -> dict[str, object]:
    pressure_ratio = float(pressure) / max(1.0, float(adr_high_pressure))
    drift_ratio = float(drift) / max(1e-9, float(adr_drift_threshold))
    service_floor = float(adr_service_floor)
    service_gap = max(0.0, service_floor - float(last_service_rate))
    service_ratio = service_gap / max(1e-9, 1.0 - service_floor)
    change_detected = bool(
        drift >= adr_drift_threshold or last_service_rate < service_floor
    )
    hard_trigger = bool(
        pressure >= adr_high_pressure and last_service_rate < service_floor
    )
    if hard_trigger or service_ratio >= 2.0 or drift_ratio >= 2.0:
        profile_hint = "severe"
    elif change_detected:
        profile_hint = "medium"
    else:
        profile_hint = "none"
    return {
        "change_detected": change_detected,
        "hard_trigger": hard_trigger,
        "profile_hint": profile_hint,
        "score": max(drift_ratio, service_ratio, 0.5 * pressure_ratio),
        "breakdown": {
            "pressure": float(pressure),
            "pressure_ratio": pressure_ratio,
            "target_drift": float(drift),
            "drift_ratio": drift_ratio,
            "last_service_rate": float(last_service_rate),
            "service_floor": service_floor,
            "service_gap": service_gap,
            "service_ratio": service_ratio,
        },
    }


def station_adr_reopt_trigger(
    *,
    change_info: dict[str, object],
    hour_idx: int,
    last_update_hour: int,
    incumbent_exists: bool,
    adr_min_update_interval: int,
    adr_max_update_interval: int,
) -> dict[str, object]:
    if not incumbent_exists:
        return {"reoptimize": True, "reason": "bootstrap", "elapsed_hours": None}

    elapsed = int(hour_idx) - int(last_update_hour)
    if bool(change_info.get("hard_trigger", False)):
        return {"reoptimize": True, "reason": "hard_trigger", "elapsed_hours": elapsed}
    if elapsed >= int(adr_max_update_interval):
        return {"reoptimize": True, "reason": "max_interval", "elapsed_hours": elapsed}
    if elapsed < int(adr_min_update_interval):
        return {"reoptimize": False, "reason": "min_interval", "elapsed_hours": elapsed}
    if bool(change_info.get("change_detected", False)):
        return {
            "reoptimize": True,
            "reason": str(change_info.get("profile_hint", "change_detected")),
            "elapsed_hours": elapsed,
        }
    return {"reoptimize": False, "reason": "stable_regime", "elapsed_hours": elapsed}


def decide_adr_eda_station(
    *,
    state: PolicyState,
    rng: random.Random,
    hour_idx: int,
    inventory: dict[str, int],
    stations: list[Station],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    fleet_size: int,
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    lookahead_hours: int,
    relocation_budget_bikes: int,
    adr_high_pressure: float,
    adr_drift_threshold: float,
    adr_service_floor: float,
    adr_min_update_interval: int,
    adr_max_update_interval: int,
    service_reward: float,
    rejection_penalty: float,
    relocation_km_cost: float,
    relocation_action_cost: float,
    eda_population: int,
    eda_generations: int,
    eda_elite_frac: float,
    eda_initial_sigma: float,
    eda_imbalance_penalty: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
    eda_engine: str,
) -> tuple[str, dict[str, float] | None]:
    heuristic = desired_target(stations, inventory, capacities, departures, arrivals)
    pressure = shortage_pressure(inventory, departures)
    drift = target_drift(state.target, heuristic, fleet_size)
    change_info = station_adr_change_info(
        pressure=pressure,
        drift=drift,
        last_service_rate=state.last_service_rate,
        adr_high_pressure=adr_high_pressure,
        adr_drift_threshold=adr_drift_threshold,
        adr_service_floor=adr_service_floor,
    )
    trigger_info = station_adr_reopt_trigger(
        change_info=change_info,
        hour_idx=hour_idx,
        last_update_hour=state.last_update_hour,
        incumbent_exists=state.target is not None,
        adr_min_update_interval=adr_min_update_interval,
        adr_max_update_interval=adr_max_update_interval,
    )
    state.last_adr_trigger_reason = str(trigger_info["reason"])
    state.last_adr_change_score = float(change_info["score"])
    state.last_adr_change_detected = bool(change_info["change_detected"])
    if bool(trigger_info["reoptimize"]):
        target = optimize_target_with_eda(
            state=state,
            rng=rng,
            stations=stations,
            inventory=inventory,
            capacities=capacities,
            departures=departures,
            arrivals=arrivals,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            hour_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            service_reward=service_reward,
            rejection_penalty=rejection_penalty,
            relocation_km_cost=relocation_km_cost,
            relocation_action_cost=relocation_action_cost,
            eda_population=eda_population,
            eda_generations=eda_generations,
            eda_elite_frac=eda_elite_frac,
            eda_initial_sigma=eda_initial_sigma,
            eda_imbalance_penalty=eda_imbalance_penalty,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
            eda_engine=eda_engine,
        )
        state.target = target
        state.last_update_hour = hour_idx
        return "update", target
    if state.target is not None:
        return "reuse", state.target
    return "hold", None


def decide_adr_eda_station_mo(
    *,
    state: PolicyState,
    rng: random.Random,
    hour_idx: int,
    inventory: dict[str, int],
    stations: list[Station],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    fleet_size: int,
    stations_by_name: dict[str, Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    lookahead_hours: int,
    relocation_budget_bikes: int,
    adr_high_pressure: float,
    adr_drift_threshold: float,
    adr_service_floor: float,
    adr_min_update_interval: int,
    adr_max_update_interval: int,
    eda_population: int,
    eda_generations: int,
    eda_elite_frac: float,
    eda_initial_sigma: float,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
    eda_engine: str,
) -> tuple[str, dict[str, float] | None]:
    heuristic = desired_target(stations, inventory, capacities, departures, arrivals)
    pressure = shortage_pressure(inventory, departures)
    drift = target_drift(state.target, heuristic, fleet_size)
    change_info = station_adr_change_info(
        pressure=pressure,
        drift=drift,
        last_service_rate=state.last_service_rate,
        adr_high_pressure=adr_high_pressure,
        adr_drift_threshold=adr_drift_threshold,
        adr_service_floor=adr_service_floor,
    )
    trigger_info = station_adr_reopt_trigger(
        change_info=change_info,
        hour_idx=hour_idx,
        last_update_hour=state.last_update_hour,
        incumbent_exists=state.target is not None,
        adr_min_update_interval=adr_min_update_interval,
        adr_max_update_interval=adr_max_update_interval,
    )
    state.last_adr_trigger_reason = str(trigger_info["reason"])
    state.last_adr_change_score = float(change_info["score"])
    state.last_adr_change_detected = bool(change_info["change_detected"])
    if bool(trigger_info["reoptimize"]):
        target = optimize_target_with_eda_mo(
            state=state,
            rng=rng,
            stations=stations,
            inventory=inventory,
            capacities=capacities,
            departures=departures,
            arrivals=arrivals,
            stations_by_name=stations_by_name,
            od_by_hour=od_by_hour,
            hours=hours,
            hour_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            eda_population=eda_population,
            eda_generations=eda_generations,
            eda_elite_frac=eda_elite_frac,
            eda_initial_sigma=eda_initial_sigma,
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
            eda_engine=eda_engine,
        )
        state.target = target
        state.last_update_hour = hour_idx
        return "update", target
    if state.target is not None:
        return "reuse", state.target
    return "hold", None


def shortage_pressure(inventory: dict[str, int], departures: Counter[str]) -> float:
    return sum(max(0.0, float(departures[name]) - float(inventory.get(name, 0))) for name in departures)


def target_drift(prev: dict[str, float] | None, new: dict[str, float], fleet_size: int) -> float:
    if prev is None:
        return 1.0
    names = set(prev) | set(new)
    return sum(abs(float(new.get(n, 0.0)) - float(prev.get(n, 0.0))) for n in names) / max(1.0, float(fleet_size))


def rebalance_to_target(
    inventory: dict[str, int],
    target: dict[str, float],
    stations_by_name: dict[str, Station],
    capacities: dict[str, int],
    budget_bikes: int,
    event_rows: list[dict[str, object]] | None = None,
    event_context: dict[str, object] | None = None,
    distance_model: str = "haversine",
    distance_detour_factor: float = 1.0,
    distance_matrix: dict[str, dict[str, float]] | None = None,
) -> tuple[int, float]:
    moved = 0
    moved_km = 0.0
    while moved < budget_bikes:
        donors = [
            name
            for name, bikes in inventory.items()
            if bikes > math.ceil(target.get(name, 0.0)) + 1
        ]
        receivers = [
            name
            for name, bikes in inventory.items()
            if bikes < math.floor(target.get(name, 0.0)) and bikes < capacities[name]
        ]
        if not donors or not receivers:
            break
        receiver = max(receivers, key=lambda name: target.get(name, 0.0) - inventory[name])
        donor = min(
            donors,
            key=lambda name: (
                station_distance_km(
                    stations_by_name[name],
                    stations_by_name[receiver],
                    distance_model=distance_model,
                    distance_detour_factor=distance_detour_factor,
                    distance_matrix=distance_matrix,
                ),
                -(inventory[name] - target.get(name, 0.0)),
            ),
        )
        inventory[donor] -= 1
        inventory[receiver] += 1
        distance_km = station_distance_km(
            stations_by_name[donor],
            stations_by_name[receiver],
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        )
        moved += 1
        moved_km += distance_km
        if event_rows is not None:
            origin_station = stations_by_name[donor]
            destination_station = stations_by_name[receiver]
            row: dict[str, object] = dict(event_context or {})
            row.update(
                {
                    "move_idx": moved,
                    "origin": donor,
                    "destination": receiver,
                    "origin_lat": round(origin_station.lat, 8),
                    "origin_lng": round(origin_station.lng, 8),
                    "destination_lat": round(destination_station.lat, 8),
                    "destination_lng": round(destination_station.lng, 8),
                    "distance_km": round(distance_km, 6),
                }
            )
            event_rows.append(row)
    return moved, moved_km


def decide_policy(
    policy: str,
    state: PolicyState,
    hour_idx: int,
    inventory: dict[str, int],
    stations: list[Station],
    capacities: dict[str, int],
    departures: Counter[str],
    arrivals: Counter[str],
    fleet_size: int,
    selective_threshold: float,
    adr_high_pressure: float,
    adr_drift_threshold: float,
) -> tuple[str, dict[str, float] | None]:
    if policy == "no_relocation":
        return "hold", None

    target = desired_target(stations, inventory, capacities, departures, arrivals)
    pressure = shortage_pressure(inventory, departures)

    if policy == "full_rebalance":
        state.target = target
        state.last_update_hour = hour_idx
        return "update", target

    if policy == "selective_rebalance":
        if state.target is None or pressure >= selective_threshold or hour_idx - state.last_update_hour >= 6:
            state.target = target
            state.last_update_hour = hour_idx
            return "update", target
        return "hold", None

    if policy == "adr_safe_reuse_proxy":
        drift = target_drift(state.target, target, fleet_size)
        if state.target is None:
            state.target = target
            state.last_update_hour = hour_idx
            return "update", target
        min_interval_active = hour_idx - state.last_update_hour < 4
        if min_interval_active and pressure < adr_high_pressure:
            return "hold", None
        if pressure >= adr_high_pressure and (
            drift >= adr_drift_threshold or state.last_service_rate < DEFAULT_STATION_SERVICE_FLOOR
        ):
            state.target = target
            state.last_update_hour = hour_idx
            return "update", target
        if pressure >= max(1.0, selective_threshold * 0.55):
            return "reuse", state.target
        return "hold", None

    raise ValueError(f"Unknown policy: {policy}")


def replay_hour(
    inventory: dict[str, int],
    capacities: dict[str, int],
    od_rows: Iterable[tuple[str, str, int]],
) -> tuple[int, int, int, int]:
    served = 0
    rejected_empty = 0
    rejected_full = 0
    total = 0
    for origin, dest, trips in od_rows:
        for _ in range(int(trips)):
            total += 1
            if inventory.get(origin, 0) <= 0:
                rejected_empty += 1
                continue
            if origin != dest and inventory.get(dest, 0) >= capacities[dest]:
                rejected_full += 1
                continue
            inventory[origin] -= 1
            inventory[dest] += 1
            served += 1
    return total, served, rejected_empty, rejected_full


def inventory_imbalance(inventory: dict[str, int], capacities: dict[str, int]) -> float:
    out = 0.0
    for name, bikes in inventory.items():
        cap = max(1.0, float(capacities[name]))
        target = 0.5 * cap
        out += abs(float(bikes) - target) / cap
    return out / max(1, len(inventory))


def simulate_policy(
    *,
    policy: str,
    seed: int,
    stations: list[Station],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    fleet_size: int,
    capacity_multiplier: float,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    selective_threshold: float,
    adr_high_pressure: float,
    adr_drift_threshold: float,
    adr_service_floor: float,
    adr_min_update_interval: int,
    adr_max_update_interval: int,
    service_reward: float,
    rejection_penalty: float,
    relocation_km_cost: float,
    relocation_action_cost: float,
    eda_population: int,
    eda_generations: int,
    eda_elite_frac: float,
    eda_initial_sigma: float,
    eda_imbalance_penalty: float,
    eda_engine: str,
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
    static_rebalance_hours: set[int],
    static_rebalance_budget_multiplier: float,
    forecast_mode: str,
    forecast_history_window: int,
    warmup_hours: int,
    hourly_trace_rows: list[dict[str, object]] | None = None,
    relocation_event_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    policy_wall_start = time.perf_counter()
    inventory = initial_inventory(stations, fleet_size=fleet_size, seed=seed)
    capacities = station_capacities(stations, fleet_size=fleet_size, capacity_multiplier=capacity_multiplier)
    stations_by_name = {station.name: station for station in stations}
    rng = random.Random(seed * 1009 + 17)
    state = PolicyState()
    total = served = rejected_empty = rejected_full = 0
    relocated_bikes = 0
    relocation_km = 0.0
    updates = reuses = holds = 0
    imbalance_sum = 0.0
    evaluated_hours = max(0, len(hours) - max(0, int(warmup_hours)))
    decision_time_sec = 0.0
    update_time_sec = 0.0
    reuse_time_sec = 0.0
    hold_time_sec = 0.0
    eval_decision_time_sec = 0.0
    eval_update_time_sec = 0.0
    eval_reuse_time_sec = 0.0
    eval_hold_time_sec = 0.0
    all_update_decisions = 0
    all_reuse_decisions = 0
    all_hold_decisions = 0

    for hour_idx, hour in enumerate(hours):
        in_evaluation = hour_idx >= int(warmup_hours)
        decision_od_by_hour = forecast_od_for_decision(
            mode=forecast_mode,
            hours=hours,
            od_by_hour=od_by_hour,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
            history_window=forecast_history_window,
        )
        dep, arr = forecast_counts(hours, decision_od_by_hour, hour_idx, lookahead_hours)
        forecast_lookahead_requests = od_request_total(
            hours=hours,
            od_by_hour=decision_od_by_hour,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
        )
        oracle_lookahead_requests = od_request_total(
            hours=hours,
            od_by_hour=od_by_hour,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
        )
        pre_decision_target = desired_target(stations, inventory, capacities, dep, arr)
        pre_decision_pressure = shortage_pressure(inventory, dep)
        pre_decision_drift = target_drift(state.target, pre_decision_target, fleet_size)
        decision_start = time.perf_counter()
        if policy == "static_twice_daily_rebalance":
            if parse_hour_of_day(hour) in static_rebalance_hours:
                target = pre_decision_target
                state.target = target
                state.last_update_hour = hour_idx
                decision = "update"
            else:
                target = None
                decision = "hold"
        elif policy == "eda_hourly_no_reuse":
            target = optimize_target_with_eda(
                state=state,
                rng=rng,
                stations=stations,
                inventory=inventory,
                capacities=capacities,
                departures=dep,
                arrivals=arr,
                stations_by_name=stations_by_name,
                od_by_hour=decision_od_by_hour,
                hours=hours,
                hour_idx=hour_idx,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                service_reward=service_reward,
                rejection_penalty=rejection_penalty,
                relocation_km_cost=relocation_km_cost,
                relocation_action_cost=relocation_action_cost,
                eda_population=eda_population,
                eda_generations=eda_generations,
                eda_elite_frac=eda_elite_frac,
                eda_initial_sigma=eda_initial_sigma,
                eda_imbalance_penalty=eda_imbalance_penalty,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
                eda_engine=eda_engine,
            )
            state.target = target
            state.last_update_hour = hour_idx
            decision = "update"
        elif policy == "eda_hourly_mo_no_reuse":
            target = optimize_target_with_eda_mo(
                state=state,
                rng=rng,
                stations=stations,
                inventory=inventory,
                capacities=capacities,
                departures=dep,
                arrivals=arr,
                stations_by_name=stations_by_name,
                od_by_hour=decision_od_by_hour,
                hours=hours,
                hour_idx=hour_idx,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                eda_population=eda_population,
                eda_generations=eda_generations,
                eda_elite_frac=eda_elite_frac,
                eda_initial_sigma=eda_initial_sigma,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
                eda_engine=eda_engine,
            )
            state.target = target
            state.last_update_hour = hour_idx
            decision = "update"
        elif policy in {"pymoo_nsga2_hourly_mo", "pymoo_dnsga2_hourly_mo"}:
            target = optimize_target_with_pymoo_mo(
                state=state,
                rng=rng,
                algorithm_name="dnsga2" if policy == "pymoo_dnsga2_hourly_mo" else "nsga2",
                stations=stations,
                inventory=inventory,
                capacities=capacities,
                departures=dep,
                arrivals=arr,
                stations_by_name=stations_by_name,
                od_by_hour=decision_od_by_hour,
                hours=hours,
                hour_idx=hour_idx,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                eda_population=eda_population,
                eda_generations=eda_generations,
                eda_initial_sigma=eda_initial_sigma,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
            )
            state.target = target
            state.last_update_hour = hour_idx
            decision = "update"
        elif policy == "rolling_horizon_rebalance":
            target = rolling_horizon_target(
                stations=stations,
                inventory=inventory,
                capacities=capacities,
                od_by_hour=decision_od_by_hour,
                hours=hours,
                start_idx=hour_idx,
                lookahead_hours=lookahead_hours,
            )
            state.target = target
            state.last_update_hour = hour_idx
            decision = "update"
        elif policy == "adr_eda_station":
            decision, target = decide_adr_eda_station(
                state=state,
                rng=rng,
                hour_idx=hour_idx,
                inventory=inventory,
                stations=stations,
                capacities=capacities,
                departures=dep,
                arrivals=arr,
                fleet_size=fleet_size,
                stations_by_name=stations_by_name,
                od_by_hour=decision_od_by_hour,
                hours=hours,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                adr_high_pressure=adr_high_pressure,
                adr_drift_threshold=adr_drift_threshold,
                adr_service_floor=adr_service_floor,
                adr_min_update_interval=adr_min_update_interval,
                adr_max_update_interval=adr_max_update_interval,
                service_reward=service_reward,
                rejection_penalty=rejection_penalty,
                relocation_km_cost=relocation_km_cost,
                relocation_action_cost=relocation_action_cost,
                eda_population=eda_population,
                eda_generations=eda_generations,
                eda_elite_frac=eda_elite_frac,
                eda_initial_sigma=eda_initial_sigma,
                eda_imbalance_penalty=eda_imbalance_penalty,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
                eda_engine=eda_engine,
            )
        elif policy == "adr_eda_station_mo":
            decision, target = decide_adr_eda_station_mo(
                state=state,
                rng=rng,
                hour_idx=hour_idx,
                inventory=inventory,
                stations=stations,
                capacities=capacities,
                departures=dep,
                arrivals=arr,
                fleet_size=fleet_size,
                stations_by_name=stations_by_name,
                od_by_hour=decision_od_by_hour,
                hours=hours,
                lookahead_hours=lookahead_hours,
                relocation_budget_bikes=relocation_budget_bikes,
                adr_high_pressure=adr_high_pressure,
                adr_drift_threshold=adr_drift_threshold,
                adr_service_floor=adr_service_floor,
                adr_min_update_interval=adr_min_update_interval,
                adr_max_update_interval=adr_max_update_interval,
                eda_population=eda_population,
                eda_generations=eda_generations,
                eda_elite_frac=eda_elite_frac,
                eda_initial_sigma=eda_initial_sigma,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
                eda_engine=eda_engine,
            )
        else:
            decision, target = decide_policy(
                policy=policy,
                state=state,
                hour_idx=hour_idx,
                inventory=inventory,
                stations=stations,
                capacities=capacities,
                departures=dep,
                arrivals=arr,
                fleet_size=fleet_size,
                selective_threshold=selective_threshold,
                adr_high_pressure=adr_high_pressure,
                adr_drift_threshold=adr_drift_threshold,
            )
        decision_elapsed = time.perf_counter() - decision_start
        decision_time_sec += decision_elapsed
        if decision == "update":
            update_time_sec += decision_elapsed
            all_update_decisions += 1
        elif decision == "reuse":
            reuse_time_sec += decision_elapsed
            all_reuse_decisions += 1
        else:
            hold_time_sec += decision_elapsed
            all_hold_decisions += 1
        if in_evaluation:
            eval_decision_time_sec += decision_elapsed
            if decision == "update":
                eval_update_time_sec += decision_elapsed
            elif decision == "reuse":
                eval_reuse_time_sec += decision_elapsed
            else:
                eval_hold_time_sec += decision_elapsed

        moved = 0
        moved_km = 0.0
        if decision in {"update", "reuse"} and target is not None:
            decision_budget_bikes = int(relocation_budget_bikes)
            if policy == "static_twice_daily_rebalance" and decision == "update":
                decision_budget_bikes = max(
                    int(relocation_budget_bikes),
                    int(round(float(relocation_budget_bikes) * float(static_rebalance_budget_multiplier))),
                )
            moved, moved_km = rebalance_to_target(
                inventory,
                target,
                stations_by_name,
                capacities,
                budget_bikes=decision_budget_bikes,
                event_rows=relocation_event_rows if in_evaluation else None,
                event_context={
                    "policy": policy,
                    "seed": seed,
                    "hour_idx": hour_idx,
                    "hour": hour,
                    "decision": decision,
                },
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
            )
            if in_evaluation:
                relocated_bikes += moved
                relocation_km += moved_km
        if in_evaluation:
            if decision == "update":
                updates += 1
            elif decision == "reuse":
                reuses += 1
            else:
                holds += 1

        hour_total, hour_served, hour_empty, hour_full = replay_hour(
            inventory, capacities, od_by_hour.get(hour, [])
        )
        state.last_service_rate = 1.0 if hour_total == 0 else float(hour_served) / float(hour_total)
        hour_imbalance = inventory_imbalance(inventory, capacities)
        if in_evaluation:
            total += hour_total
            served += hour_served
            rejected_empty += hour_empty
            rejected_full += hour_full
            imbalance_sum += hour_imbalance
        if hourly_trace_rows is not None and in_evaluation:
            hour_rejected = hour_empty + hour_full
            mo_objectives = state.last_mo_selected_objectives or {}
            hourly_trace_rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "hour_idx": hour_idx,
                    "hour": hour,
                    "decision": decision,
                    "forecast_mode": forecast_mode,
                    "forecast_lookahead_requests": forecast_lookahead_requests,
                    "oracle_lookahead_requests": oracle_lookahead_requests,
                    "eda_engine": state.last_eda_engine
                    or ("pymoo" if "pymoo" in policy else ("none" if "eda" not in policy else eda_engine)),
                    "eda_backend_status": state.last_eda_backend_status,
                    "eda_backend_note": state.last_eda_backend_note,
                    "decision_time_sec": round(decision_elapsed, 6),
                    "adr_trigger_reason": state.last_adr_trigger_reason,
                    "adr_change_detected": state.last_adr_change_detected,
                    "adr_change_score": round(state.last_adr_change_score, 6),
                    "mo_front_size": state.last_mo_front_size,
                    "mo_selected_service_rate": round(float(mo_objectives.get("service_rate", 0.0)), 6),
                    "mo_selected_relocation_km": round(float(mo_objectives.get("relocation_km", 0.0)), 6),
                    "mo_selected_relocated_bikes": round(float(mo_objectives.get("relocated_bikes", 0.0)), 6),
                    "mo_selected_rejected": round(float(mo_objectives.get("rejected", 0.0)), 6),
                    "pressure": round(pre_decision_pressure, 6),
                    "drift_to_heuristic": round(pre_decision_drift, 6),
                    "moved_bikes": moved,
                    "moved_km": round(moved_km, 6),
                    "hour_requests": hour_total,
                    "hour_served": hour_served,
                    "hour_rejected": hour_rejected,
                    "hour_service_rate": round(
                        1.0 if hour_total == 0 else float(hour_served) / float(hour_total), 6
                    ),
                    "hour_empty_rejections": hour_empty,
                    "hour_full_rejections": hour_full,
                    "inventory_imbalance": round(hour_imbalance, 6),
                }
            )

    rejected = rejected_empty + rejected_full
    service_rate = 1.0 if total == 0 else float(served) / float(total)
    utility = (
        service_reward * float(served)
        - rejection_penalty * float(rejected)
        - relocation_km_cost * relocation_km
        - relocation_action_cost * float(relocated_bikes)
    )
    wall_time_sec = time.perf_counter() - policy_wall_start
    simulated_hours = max(1, len(hours))
    return {
        "policy": policy,
        "seed": seed,
        "eda_engine": "pymoo" if "pymoo" in policy else (eda_engine if "eda" in policy else "none"),
        "hours": evaluated_hours,
        "warmup_hours": int(warmup_hours),
        "wall_time_sec": round(wall_time_sec, 6),
        "decision_time_sec": round(decision_time_sec, 6),
        "update_time_sec": round(update_time_sec, 6),
        "reuse_time_sec": round(reuse_time_sec, 6),
        "hold_time_sec": round(hold_time_sec, 6),
        "eval_decision_time_sec": round(eval_decision_time_sec, 6),
        "eval_update_time_sec": round(eval_update_time_sec, 6),
        "eval_reuse_time_sec": round(eval_reuse_time_sec, 6),
        "eval_hold_time_sec": round(eval_hold_time_sec, 6),
        "mean_decision_time_sec": round(decision_time_sec / simulated_hours, 6),
        "mean_update_time_sec": round(update_time_sec / max(1, all_update_decisions), 6),
        "mean_reuse_time_sec": round(reuse_time_sec / max(1, all_reuse_decisions), 6),
        "mean_hold_time_sec": round(hold_time_sec / max(1, all_hold_decisions), 6),
        "eval_mean_decision_time_sec": round(eval_decision_time_sec / max(1, evaluated_hours), 6),
        "eval_mean_update_time_sec": round(eval_update_time_sec / max(1, updates), 6),
        "eval_mean_reuse_time_sec": round(eval_reuse_time_sec / max(1, reuses), 6),
        "eval_mean_hold_time_sec": round(eval_hold_time_sec / max(1, holds), 6),
        "stations": len(stations),
        "fleet_size": fleet_size,
        "total_requests": total,
        "served": served,
        "rejected": rejected,
        "rejected_empty": rejected_empty,
        "rejected_full": rejected_full,
        "service_rate": round(service_rate, 6),
        "relocated_bikes": relocated_bikes,
        "relocation_km": round(relocation_km, 6),
        "reoptimization_epochs": updates,
        "reuse_epochs": reuses,
        "hold_epochs": holds,
        "mean_inventory_imbalance": round(imbalance_sum / max(1, evaluated_hours), 6),
        "utility_score": round(utility, 6),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        if not group:
            continue
        summary: dict[str, object] = {"policy": policy, "n": len(group)}
        for metric in [
            "utility_score",
            "service_rate",
            "relocation_km",
            "relocated_bikes",
            "reoptimization_epochs",
            "reuse_epochs",
            "hold_epochs",
            "mean_inventory_imbalance",
            "rejected",
            "wall_time_sec",
            "decision_time_sec",
            "update_time_sec",
            "reuse_time_sec",
            "hold_time_sec",
            "eval_decision_time_sec",
            "eval_update_time_sec",
            "eval_reuse_time_sec",
            "eval_hold_time_sec",
            "mean_decision_time_sec",
            "mean_update_time_sec",
            "mean_reuse_time_sec",
            "mean_hold_time_sec",
            "eval_mean_decision_time_sec",
            "eval_mean_update_time_sec",
            "eval_mean_reuse_time_sec",
            "eval_mean_hold_time_sec",
        ]:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = round(mean(values), 6)
            summary[f"{metric}_std"] = round(pstdev(values), 6) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def pairwise(rows: list[dict[str, object]], lhs: str, rhs: str) -> dict[str, object] | None:
    lhs_by_seed = {int(row["seed"]): row for row in rows if row["policy"] == lhs}
    rhs_by_seed = {int(row["seed"]): row for row in rows if row["policy"] == rhs}
    seeds = sorted(set(lhs_by_seed) & set(rhs_by_seed))
    if not seeds:
        return None
    utility_delta = [float(lhs_by_seed[s]["utility_score"]) - float(rhs_by_seed[s]["utility_score"]) for s in seeds]
    service_delta = [float(lhs_by_seed[s]["service_rate"]) - float(rhs_by_seed[s]["service_rate"]) for s in seeds]
    km_delta = [float(lhs_by_seed[s]["relocation_km"]) - float(rhs_by_seed[s]["relocation_km"]) for s in seeds]
    reopt_delta = [
        float(lhs_by_seed[s]["reoptimization_epochs"]) - float(rhs_by_seed[s]["reoptimization_epochs"])
        for s in seeds
    ]
    return {
        "lhs": lhs,
        "rhs": rhs,
        "paired_seeds": len(seeds),
        "utility_delta_mean": round(mean(utility_delta), 6),
        "utility_wins": sum(1 for value in utility_delta if value > 0),
        "service_delta_mean": round(mean(service_delta), 6),
        "service_nonworse": sum(1 for value in service_delta if value >= -1e-12),
        "relocation_km_delta_mean": round(mean(km_delta), 6),
        "relocation_km_wins": sum(1 for value in km_delta if value < 0),
        "reoptimization_delta_mean": round(mean(reopt_delta), 6),
        "reoptimization_wins": sum(1 for value in reopt_delta if value < 0),
    }


def summarize_hourly_traces(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    out = []
    for policy in POLICIES:
        group = grouped.get(policy, [])
        if not group:
            continue
        decisions = Counter(str(row["decision"]) for row in group)
        out.append(
            {
                "policy": policy,
                "trace_rows": len(group),
                "update_epochs": decisions["update"],
                "reuse_epochs": decisions["reuse"],
                "hold_epochs": decisions["hold"],
                "mean_pressure": round(mean(float(row["pressure"]) for row in group), 6),
                "mean_drift_to_heuristic": round(
                    mean(float(row["drift_to_heuristic"]) for row in group), 6
                ),
                "mean_mo_front_size": round(mean(float(row["mo_front_size"]) for row in group), 6),
                "mean_mo_selected_service_rate": round(
                    mean(float(row["mo_selected_service_rate"]) for row in group), 6
                ),
                "mean_mo_selected_relocation_km": round(
                    mean(float(row["mo_selected_relocation_km"]) for row in group), 6
                ),
                "mean_forecast_lookahead_requests": round(
                    mean(float(row["forecast_lookahead_requests"]) for row in group), 6
                ),
                "mean_oracle_lookahead_requests": round(
                    mean(float(row["oracle_lookahead_requests"]) for row in group), 6
                ),
                "total_moved_bikes": sum(int(row["moved_bikes"]) for row in group),
                "total_moved_km": round(sum(float(row["moved_km"]) for row in group), 6),
                "mean_hour_service_rate": round(
                    mean(float(row["hour_service_rate"]) for row in group), 6
                ),
                "total_hour_rejected": sum(int(row["hour_rejected"]) for row in group),
                "low_service_hours": sum(
                    1 for row in group if float(row["hour_service_rate"]) < DEFAULT_STATION_SERVICE_FLOOR
                ),
                "mean_inventory_imbalance": round(
                    mean(float(row["inventory_imbalance"]) for row in group), 6
                ),
            }
        )
    return out


def svg_report(path: Path, summary_rows: list[dict[str, object]]) -> None:
    order = [
        "no_relocation",
        "static_twice_daily_rebalance",
        "full_rebalance",
        "rolling_horizon_rebalance",
        "selective_rebalance",
        "adr_safe_reuse_proxy",
        "eda_hourly_no_reuse",
        "eda_hourly_mo_no_reuse",
        "pymoo_nsga2_hourly_mo",
        "pymoo_dnsga2_hourly_mo",
        "adr_eda_station",
        "adr_eda_station_mo",
    ]
    labels = {
        "no_relocation": "No relocation",
        "static_twice_daily_rebalance": "Static 2/day",
        "full_rebalance": "Full",
        "rolling_horizon_rebalance": "Rolling",
        "selective_rebalance": "Selective",
        "adr_safe_reuse_proxy": "Safe reuse",
        "eda_hourly_no_reuse": "EDA hourly",
        "eda_hourly_mo_no_reuse": "EDA hourly (multicriteria)",
        "pymoo_nsga2_hourly_mo": "pymoo NSGA-II",
        "pymoo_dnsga2_hourly_mo": "pymoo D-NSGA-II",
        "adr_eda_station": "ADR-EDA (utility)",
        "adr_eda_station_mo": "ADR-EDA",
    }
    rows = sorted(summary_rows, key=lambda row: order.index(str(row["policy"])))
    metrics = [
        ("Utility", "utility_score_mean", True),
        ("Service", "service_rate_mean", True),
        ("Relocation km", "relocation_km_mean", False),
        ("Reopt epochs", "reoptimization_epochs_mean", False),
    ]
    colors = [
        "#6b7280",
        "#64748b",
        "#9f1239",
        "#0891b2",
        "#d97706",
        "#0f766e",
        "#7c3aed",
        "#a21caf",
        "#9333ea",
        "#db2777",
        "#2563eb",
        "#16a34a",
    ]
    width, height = 1120, 380
    panel_w, panel_h, gap = 252, 230, 22
    top, left0 = 92, 34
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="560" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#172033">RealBike station-level smoke</text>',
        '<text x="560" y="52" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52606d">Real station coordinates and hourly OD replay</text>',
    ]
    for panel_idx, (title, key, higher) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        min_v, max_v = min(values + [0.0]), max(values + [0.0])
        pad = 0.12 * max(1e-9, max_v - min_v)
        min_axis, max_axis = min_v - pad, max_v + pad

        def y_for(value: float) -> float:
            return top + panel_h - ((value - min_axis) / max(1e-9, max_axis - min_axis)) * panel_h

        left = left0 + panel_idx * (panel_w + gap)
        zero_y = y_for(0.0)
        svg.extend(
            [
                f'<text x="{left + panel_w / 2:.1f}" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#243b53">{title}</text>',
                f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" stroke="#d9e2ec"/>',
                f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + panel_w}" y2="{zero_y:.1f}" stroke="#334e68"/>',
            ]
        )
        if not higher:
            svg.append(
                f'<text x="{left + 4}" y="{top + 14}" font-family="Arial, sans-serif" font-size="10" fill="#627d98">lower is better</text>'
            )
        bar_w = 20
        x_gap = (panel_w - len(rows) * bar_w) / (len(rows) + 1)
        for idx, row in enumerate(rows):
            value = float(row[key])
            x = left + x_gap + idx * (bar_w + x_gap)
            y0, yv = y_for(0.0), y_for(value)
            bar_y = min(y0, yv)
            bar_h = max(2.0, abs(y0 - yv))
            svg.extend(
                [
                    f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colors[idx]}" rx="2"/>',
                    f'<text x="{x + bar_w / 2:.1f}" y="{bar_y - 5 if value >= 0 else bar_y + bar_h + 13:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#1f2933">{value:.2f}</text>',
                    f'<text x="{x + bar_w / 2:.1f}" y="{top + panel_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#334e68">{labels[str(row["policy"])]}</text>',
                ]
            )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def service_km_frontier_svg(path: Path, summary_rows: list[dict[str, object]]) -> None:
    order = [
        "no_relocation",
        "static_twice_daily_rebalance",
        "full_rebalance",
        "rolling_horizon_rebalance",
        "selective_rebalance",
        "adr_safe_reuse_proxy",
        "eda_hourly_no_reuse",
        "eda_hourly_mo_no_reuse",
        "pymoo_nsga2_hourly_mo",
        "pymoo_dnsga2_hourly_mo",
        "adr_eda_station",
        "adr_eda_station_mo",
    ]
    labels = {
        "no_relocation": "No relocation",
        "static_twice_daily_rebalance": "Static 2/day",
        "full_rebalance": "Full hourly",
        "rolling_horizon_rebalance": "Rolling horizon",
        "selective_rebalance": "Selective",
        "adr_safe_reuse_proxy": "Safe reuse",
        "eda_hourly_no_reuse": "EDA hourly",
        "eda_hourly_mo_no_reuse": "EDA hourly (multicriteria)",
        "pymoo_nsga2_hourly_mo": "pymoo NSGA-II",
        "pymoo_dnsga2_hourly_mo": "pymoo D-NSGA-II",
        "adr_eda_station": "ADR-EDA (utility)",
        "adr_eda_station_mo": "ADR-EDA",
    }
    colors = {
        "no_relocation": "#6b7280",
        "static_twice_daily_rebalance": "#64748b",
        "full_rebalance": "#9f1239",
        "rolling_horizon_rebalance": "#0891b2",
        "selective_rebalance": "#d97706",
        "adr_safe_reuse_proxy": "#0f766e",
        "eda_hourly_no_reuse": "#7c3aed",
        "eda_hourly_mo_no_reuse": "#a21caf",
        "pymoo_nsga2_hourly_mo": "#9333ea",
        "pymoo_dnsga2_hourly_mo": "#db2777",
        "adr_eda_station": "#2563eb",
        "adr_eda_station_mo": "#16a34a",
    }
    rows = sorted(summary_rows, key=lambda row: order.index(str(row["policy"])))
    x_values = [float(row["relocation_km_mean"]) for row in rows]
    y_values = [float(row["service_rate_mean"]) for row in rows]
    x_min, x_max = min(x_values + [0.0]), max(x_values + [0.0])
    y_min, y_max = min(y_values), max(y_values)
    x_pad = 0.08 * max(1e-9, x_max - x_min)
    y_pad = 0.18 * max(1e-9, y_max - y_min)
    x_min, x_max = max(0.0, x_min - x_pad), x_max + x_pad
    y_min, y_max = max(0.0, y_min - y_pad), min(1.0, y_max + y_pad)
    if abs(y_max - y_min) < 1e-9:
        y_min = max(0.0, y_min - 0.01)
        y_max = min(1.0, y_max + 0.01)

    width, height = 980, 560
    left, right, top, bottom = 84, 276, 72, 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_for(value: float) -> float:
        return left + ((value - x_min) / max(1e-9, x_max - x_min)) * plot_w

    def y_for(value: float) -> float:
        return top + plot_h - ((value - y_min) / max(1e-9, y_max - y_min)) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#172033">RealBike service-cost frontier</text>',
        '<text x="490" y="52" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52606d">Higher service and lower OSRM relocation distance are better</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#d9e2ec"/>',
    ]
    for tick in range(6):
        frac = tick / 5.0
        x_value = x_min + frac * (x_max - x_min)
        x = x_for(x_value)
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#334e68">{x_value:.0f}</text>',
            ]
        )
    for tick in range(6):
        frac = tick / 5.0
        y_value = y_min + frac * (y_max - y_min)
        y = y_for(y_value)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#334e68">{y_value:.3f}</text>',
            ]
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#243b53" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#243b53" stroke-width="1.2"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#243b53">relocation km</text>',
            f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#243b53">service rate</text>',
        ]
    )
    for row in rows:
        policy = str(row["policy"])
        x = x_for(float(row["relocation_km_mean"]))
        y = y_for(float(row["service_rate_mean"]))
        radius = (
            7
            if policy
            in {
                "adr_eda_station",
                "adr_eda_station_mo",
                "eda_hourly_mo_no_reuse",
                "pymoo_nsga2_hourly_mo",
                "pymoo_dnsga2_hourly_mo",
            }
            else 5
        )
        svg.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{colors[policy]}" stroke="#ffffff" stroke-width="1.5"/>',
                f'<text x="{x + 9:.1f}" y="{y - 9:.1f}" font-family="Arial, sans-serif" font-size="10" fill="#1f2933">{labels[policy]}</text>',
            ]
        )
    legend_x = left + plot_w + 34
    legend_y = top + 8
    svg.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#243b53">Policies</text>'
    )
    for idx, policy in enumerate(order):
        y = legend_y + 24 + idx * 25
        svg.extend(
            [
                f'<circle cx="{legend_x + 6}" cy="{y - 4}" r="5" fill="{colors[policy]}"/>',
                f'<text x="{legend_x + 18}" y="{y}" font-family="Arial, sans-serif" font-size="11" fill="#334e68">{labels[policy]}</text>',
            ]
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def report_md(
    *,
    path: Path,
    profile_dir: Path,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    pairwise_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    effective_distance_model = getattr(args, "effective_distance_model", args.distance_model)
    lines = [
        "# RealBike Station-Level Smoke",
        "",
        "This is a direct station-level replay: real bike-sharing stations, coordinates and hourly OD trips are used without mapping them to Madrid zones.",
        "",
        "## Protocol",
        "",
        f"- Profile dir: `{profile_dir}`.",
        f"- Stations: {args.station_limit}.",
        f"- Seeds: {args.seeds}.",
        f"- Policies run: {getattr(args, 'selected_policies', ','.join(POLICIES))}.",
        f"- Fleet size: {args.fleet_size}.",
        f"- Evaluation hours replayed: {args.max_hours or 'all available after warmup'}.",
        f"- Warmup hours: {args.warmup_hours}.",
        f"- Lookahead hours: {args.lookahead_hours}.",
        f"- Relocation budget: {args.relocation_budget_bikes} bikes/hour.",
        f"- Static rebalance hours: {args.static_rebalance_hours}.",
        f"- Static rebalance budget multiplier: {args.static_rebalance_budget_multiplier}.",
        f"- Effective distance model: {effective_distance_model}.",
        f"- Distance model: {args.distance_model}.",
        f"- Distance detour factor: {args.distance_detour_factor}.",
        f"- Distance matrix: {args.distance_matrix_json or 'none'}.",
        f"- Forecast mode: {args.forecast_mode}.",
        f"- Forecast history window: {args.forecast_history_window} hours.",
        f"- EDA engine: {args.eda_engine}.",
        f"- ADR trigger: bootstrap, hard trigger, max interval, min interval, change detected, stable reuse.",
        f"- ADR update interval: min {args.adr_min_update_interval}, max {args.adr_max_update_interval}.",
        f"- ADR station indicators: target drift threshold {args.adr_drift_threshold}, service floor {args.adr_service_floor}, high pressure {args.adr_high_pressure}.",
        f"- MO service tolerance: {args.mo_service_tolerance}.",
        f"- Hourly trace: {'yes' if args.write_hourly_trace else 'no'}.",
        f"- Relocation events: {'yes' if args.write_relocation_events else 'no'}.",
        "",
        "Policies:",
        "",
        "- `no_relocation`: replay only, no operational control.",
        "- `static_twice_daily_rebalance`: paper-inspired static baseline; it rebalances only at configured daily hours, by default 03:00 and 15:00.",
        "- `full_rebalance`: recompute and rebalance every hour.",
        "- `rolling_horizon_rebalance`: deterministic MPC-style baseline; it forecasts the next hours, estimates shortage/fullness exposure and rebalances every hour toward a forecast-aware target.",
        "- `selective_rebalance`: recompute only under shortage pressure or after a long interval.",
        "- `adr_safe_reuse_proxy`: ADR-inspired ablation without station-level EDA; it uses a simpler operational pressure/drift gate.",
        "- `eda_hourly_no_reuse`: ablation that optimizes a station-level EDA target every hour, without safe temporal reuse.",
        "- `eda_hourly_mo_no_reuse`: hourly EDA with the same multicriteria candidate selection, without the ADR update/reuse gate.",
        "- `pymoo_nsga2_hourly_mo`: external pymoo NSGA-II baseline; it reoptimizes the station target every hour with the same service, relocation-km and relocated-bike objectives.",
        "- `pymoo_dnsga2_hourly_mo`: external pymoo D-NSGA-II baseline; it uses pymoo's dynamic NSGA-II variant as an hourly multicriteria optimizer over the same repaired station-target encoding.",
        "- `adr_eda_station`: ADR-EDA with scalar operational utility selection; useful as a lightweight ablation.",
        "- `adr_eda_station_mo`: ADR-EDA with multicriteria candidate selection; it evaluates each EDA target with explicit service, relocation-km and relocated-bike objectives, keeps the non-dominated candidates and executes a service-compatible compromise closest to the ideal point.",
        "",
        (
            "EDA engine caveat: `compact` is the dependency-free fallback used for fast local reproducibility. "
            "Use `--eda-engine edaspy` in the configured Docker/EDAspy environment for the thesis-unified "
            "SPEDA backend shared with the benchmark and carsharing lines."
            if args.eda_engine == "compact"
            else "EDA engine: this run uses the EDAspy/SPEDA backend. RealBike keeps domain-specific target repair, replay evaluation and Pareto compromise selection outside EDAspy."
        ),
        "External optimizer caveat: pymoo baselines use the external pymoo implementation and the same RealBike repair/replay/objective layer, so differences are due to the optimizer/update policy rather than a different simulator.",
        "",
        "Forecast caveat: `oracle` uses the observed future OD horizon and should be read as a perfect-forecast upper bound. `historical_hour` and `moving_average` use only hours before the current decision time, while the actual observed hour is still used for replay evaluation.",
        "",
        "## Summary",
        "",
        "| policy | n | utility | service | rejected | relocation km | relocated bikes | reopt | reuse | imbalance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['n']} | {float(row['utility_score_mean']):.2f} | "
            f"{float(row['service_rate_mean']):.4f} | {float(row['rejected_mean']):.2f} | "
            f"{float(row['relocation_km_mean']):.2f} | {float(row['relocated_bikes_mean']):.2f} | "
            f"{float(row['reoptimization_epochs_mean']):.2f} | {float(row['reuse_epochs_mean']):.2f} | "
            f"{float(row['mean_inventory_imbalance_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Computational Cost",
            "",
            "| policy | wall sec | decision sec | update sec | reuse sec | mean update sec | mean reuse sec |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {float(row['wall_time_sec_mean']):.3f} | "
            f"{float(row['decision_time_sec_mean']):.3f} | {float(row['update_time_sec_mean']):.3f} | "
            f"{float(row['reuse_time_sec_mean']):.3f} | {float(row['mean_update_time_sec_mean']):.4f} | "
            f"{float(row['mean_reuse_time_sec_mean']):.4f} |"
        )
    lines.extend(["", "## Pairwise Against Baselines", ""])
    lines.extend(
        [
            "| lhs | rhs | seeds | utility delta | utility wins | service delta | km delta | km wins | reopt delta | reopt wins |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            f"| {row['lhs']} | {row['rhs']} | {row['paired_seeds']} | "
            f"{float(row['utility_delta_mean']):.2f} | {row['utility_wins']}/{row['paired_seeds']} | "
            f"{float(row['service_delta_mean']):.4f} | {float(row['relocation_km_delta_mean']):.2f} | "
            f"{row['relocation_km_wins']}/{row['paired_seeds']} | {float(row['reoptimization_delta_mean']):.2f} | "
            f"{row['reoptimization_wins']}/{row['paired_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The Madrid surrogate caveat is removed in this smoke: the replay is station-level on public bike-sharing data.",
            "- `adr_safe_reuse_proxy` is a no-EDA ablation for the safe-reuse idea.",
            "- `eda_hourly_no_reuse` is an upper-bound ablation for aggressive EDA reoptimization without temporal reuse.",
            "- `eda_hourly_mo_no_reuse` isolates the multicriteria target selector from the ADR reuse gate.",
            "- `pymoo_nsga2_hourly_mo` and `pymoo_dnsga2_hourly_mo` add external multiobjective optimizer baselines on the applied RealBike replay.",
            "- `adr_eda_station` is the scalar-utility ablation of the same station-level ADR-EDA controller.",
            "- `adr_eda_station_mo` is the main multicriteria RealBike configuration of ADR-EDA: the gate, memory and EDA backend are unchanged; only candidate selection uses the non-dominated set and compromise choice.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python3 scripts/run_real_bikeshare_station_benchmark.py \\",
            f"  --profile-dir {profile_dir} \\",
            f"  --out-dir {path.parent} \\",
            f"  --seeds {args.seeds} \\",
            (
                f"  --policies {args.policies} \\"
                if str(getattr(args, "policies", "")).strip()
                else ""
            ),
            f"  --station-limit {args.station_limit} \\",
            f"  --fleet-size {args.fleet_size} \\",
            f"  --max-hours {args.max_hours} \\",
            f"  --warmup-hours {args.warmup_hours} \\",
            f"  --lookahead-hours {args.lookahead_hours} \\",
            f"  --relocation-budget-bikes {args.relocation_budget_bikes} \\",
            f"  --static-rebalance-hours {args.static_rebalance_hours} \\",
            f"  --static-rebalance-budget-multiplier {args.static_rebalance_budget_multiplier} \\",
            f"  --distance-model {args.distance_model} \\",
            f"  --distance-detour-factor {args.distance_detour_factor} \\",
            (
                f"  --distance-matrix-json {args.distance_matrix_json} \\"
                if args.distance_matrix_json
                else ""
            ),
            f"  --forecast-mode {args.forecast_mode} \\",
            f"  --forecast-history-window {args.forecast_history_window} \\",
            f"  --adr-high-pressure {args.adr_high_pressure} \\",
            f"  --adr-drift-threshold {args.adr_drift_threshold} \\",
            f"  --adr-service-floor {args.adr_service_floor} \\",
            f"  --mo-service-tolerance {args.mo_service_tolerance} \\",
            f"  --adr-min-update-interval {args.adr_min_update_interval} \\",
            f"  --adr-max-update-interval {args.adr_max_update_interval} \\",
            f"  --eda-engine {args.eda_engine} \\",
            f"  --eda-population {args.eda_population} \\",
            f"  --eda-generations {args.eda_generations}"
            + (" \\" if args.write_hourly_trace or args.write_relocation_events else ""),
            ("  --write-hourly-trace" + (" \\" if args.write_relocation_events else ""))
            if args.write_hourly_trace
            else "",
            "  --write-relocation-events" if args.write_relocation_events else "",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="2026,2027,2028,2029,2030")
    parser.add_argument(
        "--policies",
        type=str,
        default="",
        help=(
            "Optional comma-separated subset of policies to run. Empty means all policies. "
            f"Available: {','.join(POLICIES)}"
        ),
    )
    parser.add_argument("--station-limit", type=int, default=25)
    parser.add_argument("--fleet-size", type=int, default=70)
    parser.add_argument("--capacity-multiplier", type=float, default=2.0)
    parser.add_argument("--max-hours", type=int, default=336)
    parser.add_argument(
        "--warmup-hours",
        type=int,
        default=0,
        help=(
            "Initial simulated hours used to warm up online forecasts, inventory and ADR state. "
            "Warmup hours are not included in summary metrics."
        ),
    )
    parser.add_argument("--lookahead-hours", type=int, default=3)
    parser.add_argument("--relocation-budget-bikes", type=int, default=8)
    parser.add_argument(
        "--static-rebalance-hours",
        type=str,
        default="3,15",
        help="Comma-separated daily hours for the paper-inspired static twice-daily baseline.",
    )
    parser.add_argument(
        "--static-rebalance-budget-multiplier",
        type=float,
        default=12.0,
        help=(
            "Multiplier applied only to static_twice_daily_rebalance update epochs. "
            "This approximates a longer static rebalancing shift while keeping dynamic policies unchanged."
        ),
    )
    parser.add_argument("--selective-threshold", type=float, default=3.0)
    parser.add_argument("--adr-high-pressure", type=float, default=4.0)
    parser.add_argument("--adr-drift-threshold", type=float, default=0.18)
    parser.add_argument("--adr-service-floor", type=float, default=DEFAULT_STATION_SERVICE_FLOOR)
    parser.add_argument(
        "--mo-service-tolerance",
        type=float,
        default=MO_SERVICE_TOLERANCE,
        help=(
            "Maximum service-rate gap from the best sampled candidate allowed when selecting "
            "the Pareto-aware compromise solution."
        ),
    )
    parser.add_argument("--adr-min-update-interval", type=int, default=4)
    parser.add_argument("--adr-max-update-interval", type=int, default=10)
    parser.add_argument("--service-reward", type=float, default=1.0)
    parser.add_argument("--rejection-penalty", type=float, default=2.5)
    parser.add_argument("--relocation-km-cost", type=float, default=0.06)
    parser.add_argument("--relocation-action-cost", type=float, default=0.04)
    parser.add_argument(
        "--distance-model",
        choices=["haversine", "haversine_detour", "urban_manhattan"],
        default="haversine",
        help=(
            "Distance model for relocation cost: straight-line Haversine, "
            "Haversine multiplied by a detour factor, or local urban Manhattan distance."
        ),
    )
    parser.add_argument(
        "--distance-detour-factor",
        type=float,
        default=1.0,
        help="Multiplicative factor applied to haversine_detour and urban_manhattan distances.",
    )
    parser.add_argument(
        "--distance-matrix-json",
        type=Path,
        default=None,
        help=(
            "Optional precomputed station-to-station distance matrix JSON. "
            "When provided, distance_km[src][dst] overrides the analytic distance model."
        ),
    )
    parser.add_argument(
        "--forecast-mode",
        choices=["oracle", "historical_hour", "moving_average"],
        default="oracle",
        help=(
            "Demand information available to decision policies. "
            "oracle uses observed future demand as an upper bound; online modes use only past hours."
        ),
    )
    parser.add_argument(
        "--forecast-history-window",
        type=int,
        default=24,
        help="Number of past hours used by moving_average and as fallback for historical_hour.",
    )
    parser.add_argument("--eda-population", type=int, default=24)
    parser.add_argument("--eda-generations", type=int, default=3)
    parser.add_argument("--eda-elite-frac", type=float, default=0.25)
    parser.add_argument("--eda-initial-sigma", type=float, default=0.35)
    parser.add_argument("--eda-imbalance-penalty", type=float, default=0.5)
    parser.add_argument(
        "--eda-engine",
        choices=["compact", "edaspy"],
        default="compact",
        help=(
            "EDA backend for RealBike station policies. compact preserves the dependency-free "
            "validated runner; edaspy uses the project SPEDA/NSGA-II stack and requires the "
            "configured EDAspy/PyBNesian environment."
        ),
    )
    parser.add_argument(
        "--write-hourly-trace",
        action="store_true",
        help="Write hour_traces.csv with per-hour decisions, pressure, drift, service and relocation.",
    )
    parser.add_argument(
        "--write-relocation-events",
        action="store_true",
        help="Write relocation_events.csv with one row per moved bike, including origin/destination stations.",
    )
    return parser.parse_args()


def main() -> None:
    global MO_SERVICE_TOLERANCE
    args = parse_args()
    MO_SERVICE_TOLERANCE = max(0.0, float(args.mo_service_tolerance))
    seeds = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
    static_rebalance_hours = {
        int(token.strip())
        for token in str(args.static_rebalance_hours).split(",")
        if token.strip()
    }
    if any(hour < 0 or hour > 23 for hour in static_rebalance_hours):
        raise ValueError("--static-rebalance-hours must contain hours between 0 and 23.")
    selected_policies = (
        [token.strip() for token in str(args.policies).split(",") if token.strip()]
        if str(args.policies).strip()
        else list(POLICIES)
    )
    unknown_policies = [policy for policy in selected_policies if policy not in POLICIES]
    if unknown_policies:
        raise ValueError(f"Unknown policies in --policies: {','.join(unknown_policies)}")
    args.selected_policies = ",".join(selected_policies)
    stations = load_stations(args.profile_dir, limit=int(args.station_limit))
    station_names = {station.name for station in stations}
    od_by_hour = load_hourly_od(args.profile_dir, station_names)
    hours = sorted(od_by_hour)
    warmup_hours = max(0, int(args.warmup_hours))
    if int(args.max_hours) > 0:
        hours = hours[: warmup_hours + int(args.max_hours)]
    if not hours:
        raise ValueError("No OD demand hours found for selected stations.")
    if len(hours) <= warmup_hours:
        raise ValueError("--warmup-hours leaves no evaluation hours in the selected horizon.")
    distance_matrix = (
        load_distance_matrix_json(Path(args.distance_matrix_json), stations)
        if args.distance_matrix_json
        else None
    )
    args.effective_distance_model = (
        "precomputed_station_matrix"
        if distance_matrix is not None
        else str(args.distance_model)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hourly_trace_rows: list[dict[str, object]] | None = [] if args.write_hourly_trace else None
    relocation_event_rows: list[dict[str, object]] | None = [] if args.write_relocation_events else None
    for policy in selected_policies:
        for seed in seeds:
            rows.append(
                simulate_policy(
                    policy=policy,
                    seed=seed,
                    stations=stations,
                    od_by_hour=od_by_hour,
                    hours=hours,
                    fleet_size=int(args.fleet_size),
                    capacity_multiplier=float(args.capacity_multiplier),
                    lookahead_hours=int(args.lookahead_hours),
                    relocation_budget_bikes=int(args.relocation_budget_bikes),
                    selective_threshold=float(args.selective_threshold),
                    adr_high_pressure=float(args.adr_high_pressure),
                    adr_drift_threshold=float(args.adr_drift_threshold),
                    adr_service_floor=float(args.adr_service_floor),
                    adr_min_update_interval=max(1, int(args.adr_min_update_interval)),
                    adr_max_update_interval=max(1, int(args.adr_max_update_interval)),
                    service_reward=float(args.service_reward),
                    rejection_penalty=float(args.rejection_penalty),
                    relocation_km_cost=float(args.relocation_km_cost),
                    relocation_action_cost=float(args.relocation_action_cost),
                    eda_population=int(args.eda_population),
                    eda_generations=int(args.eda_generations),
                    eda_elite_frac=float(args.eda_elite_frac),
                    eda_initial_sigma=float(args.eda_initial_sigma),
                    eda_imbalance_penalty=float(args.eda_imbalance_penalty),
                    eda_engine=str(args.eda_engine),
                    distance_model=str(args.distance_model),
                    distance_detour_factor=float(args.distance_detour_factor),
                    distance_matrix=distance_matrix,
                    static_rebalance_hours=static_rebalance_hours,
                    static_rebalance_budget_multiplier=float(args.static_rebalance_budget_multiplier),
                    forecast_mode=str(args.forecast_mode),
                    forecast_history_window=max(1, int(args.forecast_history_window)),
                    warmup_hours=warmup_hours,
                    hourly_trace_rows=hourly_trace_rows,
                    relocation_event_rows=relocation_event_rows,
                )
            )

    summary_rows = summarize(rows)
    pairs = [
        ("adr_eda_station", "no_relocation"),
        ("adr_eda_station", "static_twice_daily_rebalance"),
        ("adr_eda_station", "full_rebalance"),
        ("adr_eda_station", "rolling_horizon_rebalance"),
        ("adr_eda_station", "selective_rebalance"),
        ("adr_eda_station", "adr_safe_reuse_proxy"),
        ("adr_eda_station", "eda_hourly_no_reuse"),
        ("adr_eda_station", "eda_hourly_mo_no_reuse"),
        ("adr_eda_station", "pymoo_nsga2_hourly_mo"),
        ("adr_eda_station", "pymoo_dnsga2_hourly_mo"),
        ("adr_eda_station_mo", "no_relocation"),
        ("adr_eda_station_mo", "static_twice_daily_rebalance"),
        ("adr_eda_station_mo", "full_rebalance"),
        ("adr_eda_station_mo", "rolling_horizon_rebalance"),
        ("adr_eda_station_mo", "selective_rebalance"),
        ("adr_eda_station_mo", "eda_hourly_no_reuse"),
        ("adr_eda_station_mo", "eda_hourly_mo_no_reuse"),
        ("adr_eda_station_mo", "pymoo_nsga2_hourly_mo"),
        ("adr_eda_station_mo", "pymoo_dnsga2_hourly_mo"),
        ("adr_eda_station_mo", "adr_eda_station"),
        ("eda_hourly_no_reuse", "full_rebalance"),
        ("eda_hourly_no_reuse", "rolling_horizon_rebalance"),
        ("eda_hourly_no_reuse", "selective_rebalance"),
        ("eda_hourly_mo_no_reuse", "full_rebalance"),
        ("eda_hourly_mo_no_reuse", "rolling_horizon_rebalance"),
        ("eda_hourly_mo_no_reuse", "selective_rebalance"),
        ("eda_hourly_mo_no_reuse", "eda_hourly_no_reuse"),
        ("pymoo_nsga2_hourly_mo", "full_rebalance"),
        ("pymoo_dnsga2_hourly_mo", "full_rebalance"),
        ("pymoo_nsga2_hourly_mo", "eda_hourly_mo_no_reuse"),
        ("pymoo_dnsga2_hourly_mo", "eda_hourly_mo_no_reuse"),
        ("pymoo_dnsga2_hourly_mo", "pymoo_nsga2_hourly_mo"),
        ("adr_safe_reuse_proxy", "no_relocation"),
        ("adr_safe_reuse_proxy", "full_rebalance"),
        ("adr_safe_reuse_proxy", "rolling_horizon_rebalance"),
        ("adr_safe_reuse_proxy", "selective_rebalance"),
        ("selective_rebalance", "full_rebalance"),
        ("static_twice_daily_rebalance", "no_relocation"),
        ("rolling_horizon_rebalance", "full_rebalance"),
    ]
    pairwise_rows = [row for row in (pairwise(rows, lhs, rhs) for lhs, rhs in pairs) if row is not None]

    write_csv(args.out_dir / "runs.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "pairwise.csv", pairwise_rows)
    if hourly_trace_rows is not None:
        write_csv(args.out_dir / "hour_traces.csv", hourly_trace_rows)
        write_csv(args.out_dir / "trace_summary.csv", summarize_hourly_traces(hourly_trace_rows))
    if relocation_event_rows is not None:
        write_csv(args.out_dir / "relocation_events.csv", relocation_event_rows)
    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary_rows,
        "pairwise": pairwise_rows,
        "runs": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    svg_report(args.out_dir / "realbike_station_tradeoff.svg", summary_rows)
    service_km_frontier_svg(args.out_dir / "realbike_service_km_frontier.svg", summary_rows)
    report_md(
        path=args.out_dir / "REALBIKE_STATION_SMOKE.md",
        profile_dir=args.profile_dir,
        rows=rows,
        summary_rows=summary_rows,
        pairwise_rows=pairwise_rows,
        args=args,
    )
    print(f"Wrote {args.out_dir / 'REALBIKE_STATION_SMOKE.md'}")
    print(f"Wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
