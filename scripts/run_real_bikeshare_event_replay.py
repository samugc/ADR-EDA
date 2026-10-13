#!/usr/bin/env python3
"""Replay real bike-share trips as chronological station events.

This script is a deliberately small bridge between the current hourly RealBike
benchmark and the fine-grained simulation style recommended in recent dynamic
rebalancing literature. Planning decisions are still made at hourly boundaries,
but rentals and returns are applied in their observed chronological order.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from run_real_bikeshare_station_benchmark import (
    PolicyState,
    Station,
    decide_adr_eda_station,
    decide_policy,
    forecast_counts,
    initial_inventory,
    inventory_imbalance,
    load_distance_matrix_json,
    load_hourly_od,
    load_stations,
    parse_hour_of_day,
    rebalance_to_target,
    rolling_horizon_target,
    station_capacities,
    station_distance_km,
    write_csv,
)


EVENT_POLICIES = [
    "no_relocation",
    "static_twice_daily_rebalance",
    "rolling_horizon_rebalance",
    "adr_eda_station",
]


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).strip())


def hour_key(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0).isoformat(sep=" ")


def read_event_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def group_events_by_hour(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_hour: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_hour[hour_key(parse_dt(row["event_time"]))].append(row)
    for hour_rows in by_hour.values():
        hour_rows.sort(key=lambda row: (parse_dt(row["event_time"]), row["event_type"], int(row["trip_id"])))
    return dict(by_hour)


def nearest_station_with_dock(
    *,
    destination: str,
    inventory: dict[str, int],
    capacities: dict[str, int],
    stations_by_name: dict[str, Station],
    distance_model: str,
    distance_detour_factor: float,
    distance_matrix: dict[str, dict[str, float]] | None,
) -> str | None:
    candidates = [name for name, bikes in inventory.items() if bikes < capacities[name]]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda name: station_distance_km(
            stations_by_name[destination],
            stations_by_name[name],
            distance_model=distance_model,
            distance_detour_factor=distance_detour_factor,
            distance_matrix=distance_matrix,
        ),
    )


def decide_hourly_target(
    *,
    policy: str,
    state: PolicyState,
    rng: random.Random,
    hour_idx: int,
    hour: str,
    inventory: dict[str, int],
    stations: list[Station],
    stations_by_name: dict[str, Station],
    capacities: dict[str, int],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    fleet_size: int,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    selective_threshold: float,
    adr_high_pressure: float,
    adr_drift_threshold: float,
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
    static_rebalance_hours: set[int],
) -> tuple[str, dict[str, float] | None]:
    dep, arr = forecast_counts(hours, od_by_hour, hour_idx, lookahead_hours)
    if policy == "static_twice_daily_rebalance":
        if parse_hour_of_day(hour) in static_rebalance_hours:
            decision, target = decide_policy(
                policy="full_rebalance",
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
            return decision, target
        return "hold", None
    if policy == "rolling_horizon_rebalance":
        target = rolling_horizon_target(
            stations=stations,
            inventory=inventory,
            capacities=capacities,
            od_by_hour=od_by_hour,
            hours=hours,
            start_idx=hour_idx,
            lookahead_hours=lookahead_hours,
        )
        state.target = target
        state.last_update_hour = hour_idx
        return "update", target
    if policy == "adr_eda_station":
        return decide_adr_eda_station(
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
            od_by_hour=od_by_hour,
            hours=hours,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            adr_high_pressure=adr_high_pressure,
            adr_drift_threshold=adr_drift_threshold,
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
        )
    return decide_policy(
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


def simulate_event_policy(
    *,
    policy: str,
    seed: int,
    stations: list[Station],
    event_by_hour: dict[str, list[dict[str, str]]],
    od_by_hour: dict[str, list[tuple[str, str, int]]],
    hours: list[str],
    fleet_size: int,
    capacity_multiplier: float,
    lookahead_hours: int,
    relocation_budget_bikes: int,
    selective_threshold: float,
    adr_high_pressure: float,
    adr_drift_threshold: float,
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
    static_rebalance_hours: set[int],
    static_rebalance_budget_multiplier: float,
    hourly_trace_rows: list[dict[str, object]] | None,
) -> dict[str, object]:
    inventory = initial_inventory(stations, fleet_size=fleet_size, seed=seed)
    capacities = station_capacities(stations, fleet_size=fleet_size, capacity_multiplier=capacity_multiplier)
    stations_by_name = {station.name: station for station in stations}
    rng = random.Random(seed * 1009 + 1701)
    state = PolicyState()
    trip_served: dict[int, bool] = {}

    rental_requests = served_rentals = rejected_empty = 0
    return_requests = served_returns = rejected_full = rerouted_returns = 0
    relocated_bikes = 0
    relocation_km = 0.0
    updates = reuses = holds = 0
    imbalance_sum = 0.0

    for hour_idx, hour in enumerate(hours):
        decision, target = decide_hourly_target(
            policy=policy,
            state=state,
            rng=rng,
            hour_idx=hour_idx,
            hour=hour,
            inventory=inventory,
            stations=stations,
            stations_by_name=stations_by_name,
            capacities=capacities,
            od_by_hour=od_by_hour,
            hours=hours,
            fleet_size=fleet_size,
            lookahead_hours=lookahead_hours,
            relocation_budget_bikes=relocation_budget_bikes,
            selective_threshold=selective_threshold,
            adr_high_pressure=adr_high_pressure,
            adr_drift_threshold=adr_drift_threshold,
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
            static_rebalance_hours=static_rebalance_hours,
        )
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
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
            )
            relocated_bikes += moved
            relocation_km += moved_km
        if decision == "update":
            updates += 1
        elif decision == "reuse":
            reuses += 1
        else:
            holds += 1

        hour_rentals = hour_served = hour_empty = hour_returns = hour_returned = hour_full = 0
        for event in event_by_hour.get(hour, []):
            trip_id = int(event["trip_id"])
            station = event["event_station"]
            if event["event_type"] == "rental":
                rental_requests += 1
                hour_rentals += 1
                if inventory.get(station, 0) <= 0:
                    rejected_empty += 1
                    hour_empty += 1
                    trip_served[trip_id] = False
                    continue
                inventory[station] -= 1
                served_rentals += 1
                hour_served += 1
                trip_served[trip_id] = True
                continue

            if not trip_served.get(trip_id, False):
                continue
            return_requests += 1
            hour_returns += 1
            if inventory.get(station, 0) < capacities[station]:
                inventory[station] += 1
                served_returns += 1
                hour_returned += 1
                continue
            rejected_full += 1
            hour_full += 1
            alternative = nearest_station_with_dock(
                destination=station,
                inventory=inventory,
                capacities=capacities,
                stations_by_name=stations_by_name,
                distance_model=distance_model,
                distance_detour_factor=distance_detour_factor,
                distance_matrix=distance_matrix,
            )
            if alternative is not None:
                inventory[alternative] += 1
                rerouted_returns += 1

        state.last_service_rate = (
            1.0 if hour_rentals == 0 else float(hour_served) / float(hour_rentals)
        )
        hour_imbalance = inventory_imbalance(inventory, capacities)
        imbalance_sum += hour_imbalance
        if hourly_trace_rows is not None:
            hourly_trace_rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "hour_idx": hour_idx,
                    "hour": hour,
                    "decision": decision,
                    "adr_trigger_reason": state.last_adr_trigger_reason,
                    "adr_change_detected": state.last_adr_change_detected,
                    "adr_change_score": round(state.last_adr_change_score, 6),
                    "moved_bikes": moved,
                    "moved_km": round(moved_km, 6),
                    "rental_requests": hour_rentals,
                    "served_rentals": hour_served,
                    "empty_rejections": hour_empty,
                    "return_requests": hour_returns,
                    "served_returns": hour_returned,
                    "full_return_rejections": hour_full,
                    "service_rate": round(
                        1.0 if hour_rentals == 0 else float(hour_served) / float(hour_rentals),
                        6,
                    ),
                    "inventory_imbalance": round(hour_imbalance, 6),
                }
            )

    rejected = rejected_empty + rejected_full
    service_rate = 1.0 if rental_requests == 0 else float(served_rentals) / float(rental_requests)
    return_success_rate = 1.0 if return_requests == 0 else float(served_returns) / float(return_requests)
    utility = (
        service_reward * float(served_rentals)
        - rejection_penalty * float(rejected)
        - relocation_km_cost * relocation_km
        - relocation_action_cost * float(relocated_bikes)
    )
    return {
        "policy": policy,
        "seed": seed,
        "hours": len(hours),
        "stations": len(stations),
        "fleet_size": fleet_size,
        "rental_requests": rental_requests,
        "served_rentals": served_rentals,
        "rejected": rejected,
        "rejected_empty": rejected_empty,
        "rejected_full_returns": rejected_full,
        "return_requests": return_requests,
        "served_returns": served_returns,
        "rerouted_returns": rerouted_returns,
        "service_rate": round(service_rate, 6),
        "return_success_rate": round(return_success_rate, 6),
        "relocated_bikes": relocated_bikes,
        "relocation_km": round(relocation_km, 6),
        "reoptimization_epochs": updates,
        "reuse_epochs": reuses,
        "hold_epochs": holds,
        "mean_inventory_imbalance": round(imbalance_sum / max(1, len(hours)), 6),
        "utility_score": round(utility, 6),
    }


def summarize(rows: list[dict[str, object]], policies: list[str]) -> list[dict[str, object]]:
    out = []
    for policy in policies:
        subset = [row for row in rows if row["policy"] == policy]
        if not subset:
            continue
        for metric in [
            "utility_score",
            "service_rate",
            "return_success_rate",
            "relocation_km",
            "reoptimization_epochs",
            "reuse_epochs",
            "rejected",
            "rejected_empty",
            "rejected_full_returns",
        ]:
            values = [float(row[metric]) for row in subset]
            out.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "mean": round(mean(values), 6),
                    "std": round(pstdev(values), 6) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="2026,2027")
    parser.add_argument("--policies", default=",".join(EVENT_POLICIES))
    parser.add_argument("--station-limit", type=int, default=25)
    parser.add_argument("--fleet-size", type=int, default=300)
    parser.add_argument("--capacity-multiplier", type=float, default=2.0)
    parser.add_argument("--max-hours", type=int, default=72)
    parser.add_argument("--lookahead-hours", type=int, default=4)
    parser.add_argument("--relocation-budget-bikes", type=int, default=8)
    parser.add_argument("--selective-threshold", type=float, default=10.0)
    parser.add_argument("--adr-high-pressure", type=float, default=4.0)
    parser.add_argument("--adr-drift-threshold", type=float, default=0.08)
    parser.add_argument("--adr-min-update-interval", type=int, default=1)
    parser.add_argument("--adr-max-update-interval", type=int, default=4)
    parser.add_argument("--service-reward", type=float, default=1.0)
    parser.add_argument("--rejection-penalty", type=float, default=2.0)
    parser.add_argument("--relocation-km-cost", type=float, default=0.035)
    parser.add_argument("--relocation-action-cost", type=float, default=0.02)
    parser.add_argument("--eda-population", type=int, default=20)
    parser.add_argument("--eda-generations", type=int, default=8)
    parser.add_argument("--eda-elite-frac", type=float, default=0.35)
    parser.add_argument("--eda-initial-sigma", type=float, default=0.18)
    parser.add_argument("--eda-imbalance-penalty", type=float, default=0.35)
    parser.add_argument(
        "--distance-model",
        choices=["haversine", "haversine_detour", "urban_manhattan"],
        default="haversine",
    )
    parser.add_argument("--distance-detour-factor", type=float, default=1.0)
    parser.add_argument("--distance-matrix-json", type=Path)
    parser.add_argument("--static-rebalance-hours", default="3,15")
    parser.add_argument("--static-rebalance-budget-multiplier", type=float, default=12.0)
    parser.add_argument("--write-hourly-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = load_stations(args.profile_dir, args.station_limit)
    station_names = {station.name for station in stations}
    od_by_hour = load_hourly_od(args.profile_dir, station_names)
    event_path = args.profile_dir / "event_stream_top_stations.csv"
    if not event_path.exists():
        raise FileNotFoundError(
            f"{event_path} not found. Rebuild the profile with --emit-event-stream first."
        )
    event_rows = read_event_rows(event_path)
    event_by_hour = group_events_by_hour(event_rows)
    hours = sorted(set(event_by_hour) | set(od_by_hour))
    if args.max_hours > 0:
        hours = hours[: args.max_hours]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    invalid = sorted(set(policies) - set(EVENT_POLICIES))
    if invalid:
        raise ValueError(f"Unsupported policies for event replay: {invalid}")
    static_hours = {int(value) for value in args.static_rebalance_hours.split(",") if value.strip()}
    distance_matrix = (
        load_distance_matrix_json(args.distance_matrix_json, stations)
        if args.distance_matrix_json
        else None
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] | None = [] if args.write_hourly_trace else None
    for seed in seeds:
        for policy in policies:
            result_rows.append(
                simulate_event_policy(
                    policy=policy,
                    seed=seed,
                    stations=stations,
                    event_by_hour=event_by_hour,
                    od_by_hour=od_by_hour,
                    hours=hours,
                    fleet_size=args.fleet_size,
                    capacity_multiplier=args.capacity_multiplier,
                    lookahead_hours=args.lookahead_hours,
                    relocation_budget_bikes=args.relocation_budget_bikes,
                    selective_threshold=args.selective_threshold,
                    adr_high_pressure=args.adr_high_pressure,
                    adr_drift_threshold=args.adr_drift_threshold,
                    adr_min_update_interval=args.adr_min_update_interval,
                    adr_max_update_interval=args.adr_max_update_interval,
                    service_reward=args.service_reward,
                    rejection_penalty=args.rejection_penalty,
                    relocation_km_cost=args.relocation_km_cost,
                    relocation_action_cost=args.relocation_action_cost,
                    eda_population=args.eda_population,
                    eda_generations=args.eda_generations,
                    eda_elite_frac=args.eda_elite_frac,
                    eda_initial_sigma=args.eda_initial_sigma,
                    eda_imbalance_penalty=args.eda_imbalance_penalty,
                    distance_model=args.distance_model,
                    distance_detour_factor=args.distance_detour_factor,
                    distance_matrix=distance_matrix,
                    static_rebalance_hours=static_hours,
                    static_rebalance_budget_multiplier=args.static_rebalance_budget_multiplier,
                    hourly_trace_rows=trace_rows,
                )
            )

    summary_rows = summarize(result_rows, policies)
    write_csv(args.out_dir / "event_results.csv", result_rows)
    write_csv(args.out_dir / "event_summary.csv", summary_rows)
    if trace_rows is not None:
        write_csv(args.out_dir / "event_hourly_trace.csv", trace_rows)
    output_files = ["event_results.csv", "event_summary.csv"]
    if trace_rows is not None:
        output_files.append("event_hourly_trace.csv")
    metadata = {
        "profile_dir": str(args.profile_dir),
        "event_stream": str(event_path),
        "hours": len(hours),
        "seeds": seeds,
        "policies": policies,
        "station_limit": args.station_limit,
        "fleet_size": args.fleet_size,
        "distance_model": "precomputed_station_matrix" if distance_matrix else args.distance_model,
        "distance_matrix_json": str(args.distance_matrix_json) if args.distance_matrix_json else None,
        "outputs": output_files,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'event_results.csv'}")
    print(f"Wrote {args.out_dir / 'event_summary.csv'}")
    print(f"Wrote {args.out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
