from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np

from .dynamics import (
    feasible_vehicle_set_for_request,
    choose_user_nearest_vehicle,
    refresh_system_state,
    schedule_trip_for_request,
)
from .planner import EffectiveOperationalPolicy, RobustDecisionEncoding, RollingHorizonPlanner
from .scenario import MadridFreeFloatingScenario
from .travel_metrics import zone_distance_km, zone_drive_time_min_at_time
from .types import Request, RobustModelConfig, SystemMetrics, SystemState, Vehicle, VehicleStatus


@dataclass
class EvaluationSummary:
    objectives: np.ndarray
    details: Dict[str, float]
    end_state: SystemState


class FreeFloatingRobustDMOOP:
    """
    Dynamic stateful evaluator for one-way free-floating EV carsharing in Madrid.

    Objective vector to maximize:
    - f1: service score
    - f2: economic score
    - f3: fleet health score
    """

    def __init__(
        self,
        state_template: SystemState,
        scenario: MadridFreeFloatingScenario,
        encoding: RobustDecisionEncoding,
        config: RobustModelConfig,
        horizon_hours: float = 12,
        random_seed: int = 2026,
        rollout_demand_mode: str = "observed",
    ):
        self.state_template = state_template
        self.scenario = scenario
        self.encoding = encoding
        self.config = config
        self.horizon_hours = float(horizon_hours)
        self.random_seed = int(random_seed)
        self.rollout_demand_mode = str(rollout_demand_mode or "observed").strip().lower()
        self.planner = RollingHorizonPlanner(cfg=config)
        self.reset_evaluation_stats()

    def reset_evaluation_stats(self) -> None:
        self._rollout_call_count = 0
        self._evaluate_call_count = 0
        self._objective_max_call_count = 0
        self._objective_min_call_count = 0
        self._rollout_wall_time_sec = 0.0

    def evaluation_stats(self) -> Dict[str, float]:
        return {
            "rollout_call_count": float(self._rollout_call_count),
            "evaluate_call_count": float(self._evaluate_call_count),
            "objective_max_call_count": float(self._objective_max_call_count),
            "objective_min_call_count": float(self._objective_min_call_count),
            "rollout_wall_time_sec": float(self._rollout_wall_time_sec),
        }

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def _normalized_objective_weights(
        self,
        a: float,
        b: float,
        c: float,
    ) -> tuple[float, float, float]:
        weights = np.array([max(0.0, float(a)), max(0.0, float(b)), max(0.0, float(c))], dtype=float)
        total = float(np.sum(weights))
        if total <= 1e-12:
            return 1.0, 0.0, 0.0
        weights = weights / total
        return float(weights[0]), float(weights[1]), float(weights[2])

    def _profit_score_scale(self, fleet_size: int) -> float:
        scale = float(self.config.profit_score_scale_sqrt) * math.sqrt(
            max(1.0, float(fleet_size) * max(1.0, float(self.horizon_hours)))
        )
        return max(float(self.config.profit_score_min_scale_eur), scale)

    def _tail_soc_score(self, p10_final_soc: float) -> float:
        low = float(self.config.tail_soc_headroom_min)
        high = max(low + 1e-9, float(self.config.tail_soc_headroom_target))
        return self._clamp01((float(p10_final_soc) - low) / (high - low))

    def _soh_loss_score(
        self,
        *,
        soh_loss_total: float,
        fleet_size: int,
        n_epochs: int,
    ) -> tuple[float, float]:
        ref = max(
            1e-12,
            float(self.config.soh_loss_ref_per_vehicle_epoch)
            * max(1.0, float(fleet_size))
            * max(1.0, float(n_epochs)),
        )
        score = 1.0 - min(1.0, float(soh_loss_total) / ref)
        return float(score), float(ref)

    def _seed_from_solution(self, x: np.ndarray) -> int:
        rounded = np.round(np.asarray(x, dtype=float), 4)
        return int((np.sum(np.abs(rounded) * 1000.0) + self.random_seed) % (2**32 - 1))

    def _choose_with_exclusions(
        self,
        request: Request,
        state: SystemState,
        excluded_vehicle_ids: Set[str],
    ) -> Tuple[Optional[Vehicle], float]:
        candidates = [
            (v, walk)
            for v, walk in feasible_vehicle_set_for_request(request=request, state=state, cfg=self.config)
            if v.vehicle_id not in excluded_vehicle_ids
        ]
        if not candidates:
            return choose_user_nearest_vehicle(request=request, state=state, cfg=self.config)
        vehicle, walk_min = min(candidates, key=lambda x: x[1])
        return vehicle, float(walk_min)

    def _try_preassign_reservation(
        self,
        request: Request,
        state: SystemState,
        policy: EffectiveOperationalPolicy,
        event_time_min: float,
    ) -> Request:
        return self.planner.try_preassign_reservation(
            request=request,
            state=state,
            policy=policy,
            event_time_min=event_time_min,
        )

    def _sample_forecast_requests(
        self,
        *,
        state: SystemState,
        start_min: float,
        end_min: float,
        rng: np.random.Generator,
        external_demand_multiplier: Optional[Dict[str, float]],
        origin_zone_filter: Optional[Set[str]],
    ) -> list[Request]:
        demand = self.scenario.forecast_demand(
            now_min=float(start_min),
            horizon_min=max(1e-9, float(end_min) - float(start_min)),
            external_multiplier=external_demand_multiplier,
        )
        zone_ids = list(self.scenario.zones.keys())
        if not zone_ids:
            return []
        weights = np.asarray(
            [max(1e-9, float(self.scenario.zones[z].base_demand_per_hour)) for z in zone_ids],
            dtype=float,
        )
        weights = weights / max(1e-12, float(np.sum(weights)))

        reqs: list[Request] = []
        seq = 0
        for origin, expected in demand.items():
            origin = str(origin)
            if origin_zone_filter is not None and origin not in origin_zone_filter:
                continue
            if origin not in self.scenario.zones:
                continue
            n = int(rng.poisson(max(0.0, float(expected))))
            for _ in range(n):
                t = float(start_min) + float(rng.uniform(0.0, max(1e-9, float(end_min) - float(start_min))))
                dest = str(rng.choice(zone_ids, p=weights))
                if len(zone_ids) > 1 and dest == origin:
                    alternatives = [z for z in zone_ids if z != origin]
                    alt_weights = np.asarray(
                        [max(1e-9, float(self.scenario.zones[z].base_demand_per_hour)) for z in alternatives],
                        dtype=float,
                    )
                    alt_weights = alt_weights / max(1e-12, float(np.sum(alt_weights)))
                    dest = str(rng.choice(alternatives, p=alt_weights))
                d_km = max(
                    float(getattr(self.scenario.cfg, "min_trip_km", 0.65)),
                    zone_distance_km(origin, dest, self.scenario.zones, travel_metrics=state.travel_metrics),
                )
                duration = max(
                    1.0,
                    zone_drive_time_min_at_time(
                        zone_a=origin,
                        zone_b=dest,
                        zones=self.scenario.zones,
                        travel_metrics=state.travel_metrics,
                        fallback_speed_kmh=float(self.config.avg_drive_speed_kmh),
                        departure_time_min=t,
                    ),
                )
                fare_breakdown = self.scenario._trip_revenue_breakdown(
                    origin_zone_id=origin,
                    destination_zone_id=dest,
                    distance_km=float(d_km),
                    duration_min=float(duration),
                )
                reqs.append(
                    Request(
                        request_id=f"FORECAST_{int(start_min):06d}_{seq:04d}",
                        request_time_min=float(t),
                        origin_zone_id=origin,
                        destination_zone_id=dest,
                        distance_km=float(d_km),
                        duration_min=float(duration),
                        revenue_eur=float(fare_breakdown["total_eur"]),
                        pickup_time_min=float(t),
                        is_reservation=False,
                        cancellation_time_min=None,
                        fare_breakdown=fare_breakdown,
                    )
                )
                seq += 1
        return reqs

    @staticmethod
    def _metrics_snapshot(metrics: SystemMetrics) -> Dict[str, float]:
        return {
            name: float(getattr(metrics, name))
            for name in metrics.__dataclass_fields__.keys()
        }

    @staticmethod
    def _metrics_delta(metrics: SystemMetrics, baseline: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in metrics.__dataclass_fields__.keys():
            out[name] = float(getattr(metrics, name)) - float(baseline.get(name, 0.0))
        return out

    def _run_rollout(
        self,
        x: np.ndarray,
        base_state: Optional[SystemState] = None,
        external_demand_multiplier: Optional[Dict[str, float]] = None,
    ) -> EvaluationSummary:
        policy = self.encoding.decode(x)
        seed = self._seed_from_solution(x)
        rng = np.random.default_rng(seed)

        state = copy.deepcopy(base_state if base_state is not None else self.state_template)
        base_time_min = float(state.current_time_min)
        baseline_metrics = self._metrics_snapshot(state.metrics)
        pending_reservations: Dict[str, Request] = dict(state.pending_reservations)

        total_min = int(60 * self.horizon_hours)
        delta = int(self.config.decision_interval_min)
        n_epochs = max(1, int(math.ceil(float(total_min) / max(1.0, float(delta)))))
        twc_raw_total = 0.0
        low_soc_observation_sum = 0.0
        low_soc_observation_count = 0

        def _serve_pickup(req: Request, event_time_min: float) -> None:
            pickup_req = copy.copy(req)
            if pickup_req.pickup_time_min is None:
                pickup_req.pickup_time_min = float(event_time_min)

            if pickup_req.is_reservation:
                state.metrics.reservation_pickups_due += 1
            state.metrics.requests_total += 1
            refresh_system_state(state=state, now_min=event_time_min, cfg=self.config)

            pickup_clock = (
                float(pickup_req.pickup_time_min)
                if pickup_req.pickup_time_min is not None
                else float(pickup_req.request_time_min)
            )
            reserved_vehicle_ids = {
                str(r.preassigned_vehicle_id)
                for r in pending_reservations.values()
                if r.preassigned_vehicle_id is not None
                and (r.pickup_time_min if r.pickup_time_min is not None else r.request_time_min) >= pickup_clock - 1e-9
                and r.request_id != pickup_req.request_id
            }

            vehicle = None
            walk_min = 0.0
            if pickup_req.preassigned_vehicle_id:
                preassigned_id = str(pickup_req.preassigned_vehicle_id)
                for v, w in feasible_vehicle_set_for_request(request=pickup_req, state=state, cfg=self.config):
                    if v.vehicle_id == preassigned_id:
                        vehicle, walk_min = v, float(w)
                        state.metrics.preassigned_pickups_honored += 1
                        break
                if vehicle is None:
                    state.metrics.preassigned_pickups_failed += 1

            if vehicle is None:
                vehicle, walk_min = self._choose_with_exclusions(
                    request=pickup_req,
                    state=state,
                    excluded_vehicle_ids=reserved_vehicle_ids,
                )
            if vehicle is None:
                state.metrics.rejected_requests += 1
                state.metrics.penalty_cost_eur += float(self.config.penalty_unserved_request_eur)
                return
            schedule_trip_for_request(
                request=pickup_req,
                vehicle=vehicle,
                walk_min=walk_min,
                state=state,
            )

        def _record_low_soc_snapshot() -> None:
            nonlocal low_soc_observation_sum, low_soc_observation_count
            if not state.vehicles:
                return
            low_soc_share = float(
                np.mean([1.0 if v.soc < self.config.deep_discharge_soc else 0.0 for v in state.vehicles.values()])
            )
            low_soc_observation_sum += low_soc_share
            low_soc_observation_count += 1

        for epoch_start in range(0, total_min, delta):
            epoch_end = min(total_min, epoch_start + delta)
            now = base_time_min + float(epoch_start)
            end = base_time_min + float(epoch_end)

            refresh_system_state(state=state, now_min=now, cfg=self.config)
            _record_low_soc_snapshot()
            state.pending_reservations = dict(pending_reservations)
            soc_start = {vid: float(np.clip(v.soc, 0.0, 1.0)) for vid, v in state.vehicles.items()}
            soc_last_obs = dict(soc_start)
            gross_discharge = {vid: 0.0 for vid in state.vehicles.keys()}
            gross_charge = {vid: 0.0 for vid in state.vehicles.keys()}

            def _accumulate_soc_deltas() -> None:
                for vid, v in state.vehicles.items():
                    prev = float(soc_last_obs.get(vid, float(np.clip(v.soc, 0.0, 1.0))))
                    cur = float(np.clip(v.soc, 0.0, 1.0))
                    d = cur - prev
                    if d > 0.0:
                        gross_charge[vid] = float(gross_charge.get(vid, 0.0) + d)
                    elif d < 0.0:
                        gross_discharge[vid] = float(gross_discharge.get(vid, 0.0) - d)
                    soc_last_obs[vid] = cur

            demand_forecast = self.scenario.forecast_demand(
                now_min=now,
                horizon_min=float(self.config.planning_horizon_min),
                external_multiplier=external_demand_multiplier,
            )
            # Add already-booked reservation pickups expected in planning horizon.
            horizon_end = now + float(self.config.planning_horizon_min)
            for req in pending_reservations.values():
                p_time = req.pickup_time_min if req.pickup_time_min is not None else req.request_time_min
                if now <= p_time <= horizon_end:
                    demand_forecast[req.origin_zone_id] = demand_forecast.get(req.origin_zone_id, 0.0) + 1.0

            effective_policy = self.planner.materialize_policy(
                state=state,
                demand_forecast=demand_forecast,
                policy=policy,
                at_time_min=now,
            )

            self.planner.plan_epoch(
                state=state,
                epoch_start_min=now,
                epoch_end_min=end,
                demand_forecast=demand_forecast,
                policy=policy,
            )

            origin_zone_filter = (
                {
                    v.zone_id
                    for v in state.vehicles.values()
                    if v.status == VehicleStatus.IDLE and v.available_at_min <= now + 1e-9
                }
                if self.config.demand_only_zones_with_idle
                else None
            )
            if self.rollout_demand_mode == "forecast":
                requests = self._sample_forecast_requests(
                    state=state,
                    start_min=now,
                    end_min=end,
                    rng=rng,
                    external_demand_multiplier=external_demand_multiplier,
                    origin_zone_filter=origin_zone_filter,
                )
            else:
                requests = self.scenario.sample_requests(
                    state=state,
                    start_min=now,
                    end_min=end,
                    rng=rng,
                    external_multiplier=external_demand_multiplier,
                    origin_zone_filter=origin_zone_filter,
                )

            events: list[tuple[float, int, str, str, Request]] = []

            # Events from reservations booked in previous epochs.
            for req in pending_reservations.values():
                p_time = req.pickup_time_min if req.pickup_time_min is not None else req.request_time_min
                c_time = req.cancellation_time_min
                if c_time is not None and now < c_time <= end and c_time < p_time - 1e-9:
                    events.append((float(c_time), 0, "cancel", req.request_id, req))
                if now < p_time <= end:
                    events.append((float(p_time), 2, "pickup", req.request_id, req))

            # New booking arrivals inside current epoch.
            for req in requests:
                events.append((float(req.request_time_min), 1, "booking", req.request_id, req))

                if req.is_reservation:
                    p_time = req.pickup_time_min if req.pickup_time_min is not None else req.request_time_min
                    c_time = req.cancellation_time_min
                    if c_time is not None and now < c_time <= end and c_time < p_time - 1e-9:
                        events.append((float(c_time), 0, "cancel", req.request_id, req))
                    if now < p_time <= end:
                        events.append((float(p_time), 2, "pickup", req.request_id, req))

            events.sort(key=lambda e: (e[0], e[1], e[3]))

            for event_time, _, event_kind, req_id, req in events:
                refresh_system_state(state=state, now_min=event_time, cfg=self.config)
                _accumulate_soc_deltas()
                _record_low_soc_snapshot()

                if event_kind == "booking":
                    state.metrics.bookings_total += 1
                    if req.is_reservation:
                        preassigned = self._try_preassign_reservation(
                            request=req,
                            state=state,
                            policy=effective_policy,
                            event_time_min=event_time,
                        )
                        if preassigned.preassigned_vehicle_id is not None:
                            state.metrics.preassigned_bookings += 1
                        pending_reservations[req.request_id] = preassigned
                        state.pending_reservations = dict(pending_reservations)
                    else:
                        _serve_pickup(req=req, event_time_min=event_time)
                    continue

                if event_kind == "cancel":
                    if req_id in pending_reservations:
                        pending_reservations.pop(req_id, None)
                        state.metrics.canceled_bookings += 1
                        state.pending_reservations = dict(pending_reservations)
                    continue

                if event_kind == "pickup":
                    if req_id in pending_reservations:
                        req_pick = pending_reservations.pop(req_id)
                        state.pending_reservations = dict(pending_reservations)
                        _serve_pickup(req=req_pick, event_time_min=event_time)

            refresh_system_state(state=state, now_min=end, cfg=self.config)
            _accumulate_soc_deltas()
            _record_low_soc_snapshot()
            state.current_time_min = end

            gamma = max(1e-6, float(self.config.gamma_bat))
            kappa = max(0.0, float(self.config.kappa_bat))
            for vid in state.vehicles.keys():
                soc_k = float(np.clip(soc_start.get(vid, 0.0), 0.0, 1.0))
                soc_k1 = float(np.clip(state.vehicles[vid].soc, 0.0, 1.0))
                d_dis = max(0.0, float(gross_discharge.get(vid, 0.0)))
                d_chg = max(0.0, float(gross_charge.get(vid, 0.0)))

                one_minus_soc_k = max(1e-9, 1.0 - soc_k)
                term_dis = max(0.0, ((max(1e-9, one_minus_soc_k + d_dis)) ** gamma) - (one_minus_soc_k ** gamma))
                term_chg = max(
                    0.0,
                    ((max(1e-9, one_minus_soc_k + d_chg)) ** gamma)
                    - ((max(1e-9, 1.0 - soc_k1)) ** gamma),
                )
                term = term_dis + term_chg
                twc_raw_total += kappa * float(term)

        state.pending_reservations = dict(pending_reservations)

        objectives, details = self._compute_objectives(
            state,
            baseline_metrics=baseline_metrics,
            twc_raw_total=float(twc_raw_total),
            n_epochs=int(n_epochs),
            low_soc_exposure_ratio=(
                float(low_soc_observation_sum) / max(1.0, float(low_soc_observation_count))
            ),
        )
        details["window_start_min"] = float(base_time_min)
        details["window_end_min"] = float(base_time_min + total_min)
        details["pending_reservations_end"] = float(len(state.pending_reservations))
        return EvaluationSummary(objectives=objectives, details=details, end_state=state)

    def _compute_objectives(
        self,
        state: SystemState,
        baseline_metrics: Optional[Dict[str, float]] = None,
        twc_raw_total: float = 0.0,
        n_epochs: int = 1,
        low_soc_exposure_ratio: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        metrics = state.metrics
        cumulative = self._metrics_snapshot(metrics)
        window = (
            self._metrics_delta(metrics, baseline_metrics)
            if baseline_metrics is not None
            else dict(cumulative)
        )

        raw_requests = float(window["requests_total"])
        served_requests = float(window["served_requests"])
        n_requests = max(1.0, raw_requests)
        service_rate = served_requests / n_requests if raw_requests > 0.0 else 0.0

        # f1: service quality.
        # Includes served demand, user access time and reservation reliability.
        # Access/reliability act as secondary correctives over actually served demand,
        # avoiding positive credit in windows with no due requests or no service.
        walk_avg = float(window["walked_minutes_total"]) / max(1.0, served_requests)
        walk_ref = max(1.0, float(self.config.service_walk_ref_min))
        walk_score = 1.0 / (1.0 + walk_avg / walk_ref) if served_requests > 0.0 else 0.0
        reservation_due = float(window.get("reservation_pickups_due", 0.0))
        preassigned_resolved = float(window["preassigned_pickups_honored"]) + float(window["preassigned_pickups_failed"])
        reservation_coverage = (
            preassigned_resolved / reservation_due
            if reservation_due > 0.0
            else 1.0
        )
        reservation_honor_rate = (
            float(window["preassigned_pickups_honored"]) / preassigned_resolved
            if preassigned_resolved > 0.0
            else 0.0 if reservation_due > 0.0 else 1.0
        )
        reservation_score = 0.5 * reservation_coverage + 0.5 * reservation_honor_rate
        w_sr, w_walk, w_res = self._normalized_objective_weights(
            self.config.service_rate_weight,
            self.config.walk_score_weight,
            self.config.reservation_reliability_weight,
        )
        if raw_requests <= 0.0:
            f1 = 1.0
        else:
            f1 = self._clamp01(
                service_rate * (w_sr + w_walk * walk_score + w_res * reservation_score)
            )

        # f2: smoothed economic objective in [0, 1] for MO comparability.
        profit = float(window["revenue_eur"]) + float(window["ops_charge_credit_eur"]) - (
            float(window["relocation_cost_eur"])
            + float(window["energy_cost_eur"])
            + float(window["ops_cost_eur"])
            + float(window["ops_deadhead_cost_eur"])
            + float(window["penalty_cost_eur"])
        )
        profit_scale = self._profit_score_scale(fleet_size=len(state.vehicles))
        f2 = 0.5 * (math.tanh(profit / max(1e-9, profit_scale)) + 1.0)

        # f3: Xu2021-inspired degradation core in [0, 1], extended with
        # window-level SOH loss and lower-tail SOC headroom for online control.
        socs = np.asarray([float(v.soc) for v in state.vehicles.values()], dtype=float)
        mean_soh = float(np.mean([v.soh for v in state.vehicles.values()])) if state.vehicles else 0.9
        final_low_soc_ratio = float(np.mean(socs < self.config.deep_discharge_soc)) if socs.size else 0.0
        min_final_soc = float(np.min(socs)) if socs.size else 0.0
        p10_final_soc = float(np.percentile(socs, 10)) if socs.size else 0.0
        low_soc_exposure_ratio = self._clamp01(low_soc_exposure_ratio)
        soc_risk_ratio = max(final_low_soc_ratio, low_soc_exposure_ratio)
        twc_pos = max(0.0, float(twc_raw_total))
        b_ref = max(
            1e-9,
            float(self.config.battery_wear_ref_per_vehicle_epoch)
            * max(1.0, float(len(state.vehicles)))
            * max(1.0, float(n_epochs)),
        )
        twc_score = 1.0 - min(1.0, twc_pos / b_ref)
        mean_soh_score = self._clamp01((mean_soh - 0.80) / 0.20)
        soc_safety_score = self._clamp01(1.0 - soc_risk_ratio)
        soh_loss_total = float(window["soh_loss_total"])
        soh_loss_score, soh_loss_ref = self._soh_loss_score(
            soh_loss_total=soh_loss_total,
            fleet_size=len(state.vehicles),
            n_epochs=int(n_epochs),
        )
        tail_soc_score = self._tail_soc_score(p10_final_soc)
        w_twc, w_soh, w_soc = self._normalized_objective_weights(
            self.config.health_twc_weight,
            self.config.health_soh_weight,
            self.config.health_soc_safety_weight,
        )
        f3 = self._clamp01(
            w_twc * twc_score + w_soh * soh_loss_score + w_soc * tail_soc_score
        )

        details = {
            "window_service_rate": service_rate,
            "service_rate": service_rate,
            "avg_walk_min": walk_avg,
            "walk_score": float(walk_score),
            "profit_eur": profit,
            "profit_score_scale_eur": float(profit_scale),
            "revenue_eur": float(window["revenue_eur"]),
            "trip_fare_revenue_eur": float(window["trip_fare_revenue_eur"]),
            "airport_fee_revenue_eur": float(window["airport_fee_revenue_eur"]),
            "ops_charge_credit_eur": float(window["ops_charge_credit_eur"]),
            "cost_relocation_eur": float(window["relocation_cost_eur"]),
            "cost_energy_eur": float(window["energy_cost_eur"]),
            "cost_ops_eur": float(window["ops_cost_eur"]),
            "cost_ops_deadhead_eur": float(window["ops_deadhead_cost_eur"]),
            "cost_penalty_eur": float(window["penalty_cost_eur"]),
            "served_requests": served_requests,
            "rejected_requests": float(window["rejected_requests"]),
            "total_requests": float(window["requests_total"]),
            "bookings_total": float(window["bookings_total"]),
            "canceled_bookings": float(window["canceled_bookings"]),
            "preassigned_bookings": float(window["preassigned_bookings"]),
            "reservation_pickups_due": float(reservation_due),
            "preassigned_pickups_honored": float(window["preassigned_pickups_honored"]),
            "preassigned_pickups_failed": float(window["preassigned_pickups_failed"]),
            "preassigned_honor_rate": float(reservation_honor_rate),
            "reservation_coverage": float(reservation_coverage),
            "reservation_score": float(reservation_score),
            "mean_soh": mean_soh,
            "mean_soh_score": float(mean_soh_score),
            "soh_loss_total": soh_loss_total,
            "soh_loss_ref": float(soh_loss_ref),
            "soh_loss_score": float(soh_loss_score),
            "final_low_soc_ratio": float(final_low_soc_ratio),
            "min_final_soc": float(min_final_soc),
            "p10_final_soc": float(p10_final_soc),
            "low_soc_exposure_ratio": float(low_soc_exposure_ratio),
            "soc_risk_ratio": soc_risk_ratio,
            "soc_safety_score": float(soc_safety_score),
            "tail_soc_score": float(tail_soc_score),
            "charged_kwh_total": float(window["charged_kwh_total"]),
            "drive_km_total": float(window["drive_km_total"]),
            "ops_deadhead_km_total": float(window["ops_deadhead_km_total"]),
            "cum_requests_total": float(cumulative["requests_total"]),
            "cum_served_requests": float(cumulative["served_requests"]),
            "cum_rejected_requests": float(cumulative["rejected_requests"]),
            "cum_profit_eur": float(
                cumulative["revenue_eur"]
                + cumulative["ops_charge_credit_eur"]
                - (
                    cumulative["relocation_cost_eur"]
                    + cumulative["energy_cost_eur"]
                    + cumulative["ops_cost_eur"]
                    + cumulative["ops_deadhead_cost_eur"]
                    + cumulative["penalty_cost_eur"]
                )
            ),
            "twc_raw": float(twc_raw_total),
            "twc_pos": float(twc_pos),
            "twc_ref": float(b_ref),
            "twc_score": float(twc_score),
            "service_score_rate_weight": float(w_sr),
            "service_score_walk_weight": float(w_walk),
            "service_score_reservation_weight": float(w_res),
            "health_score_twc_weight": float(w_twc),
            "health_score_soh_weight": float(w_soh),
            "health_score_soh_loss_weight": float(w_soh),
            "health_score_soc_weight": float(w_soc),
            "health_score_tail_soc_weight": float(w_soc),
            "f1_service": f1,
            "f2_profit": f2,
            "f3_health": f3,
            "rollout_demand_mode": 1.0 if self.rollout_demand_mode == "forecast" else 0.0,
        }
        return np.array([f1, f2, f3], dtype=float), details

    def rollout(
        self,
        x: np.ndarray,
        base_state: Optional[SystemState] = None,
        external_demand_multiplier: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float], SystemState]:
        self._rollout_call_count += 1
        start = time.monotonic()
        try:
            out = self._run_rollout(
                x=x,
                base_state=base_state,
                external_demand_multiplier=external_demand_multiplier,
            )
        finally:
            self._rollout_wall_time_sec += float(time.monotonic() - start)
        return out.objectives, out.details, out.end_state

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        self._evaluate_call_count += 1
        obj, details, _ = self.rollout(x)
        return obj, details

    def objective_max(self, x: np.ndarray) -> np.ndarray:
        self._objective_max_call_count += 1
        obj, _ = self.evaluate(x)
        return obj

    def objective_min(self, x: np.ndarray) -> np.ndarray:
        # EDAspy NSGA-II works in minimization mode.
        self._objective_min_call_count += 1
        return -self.objective_max(x)
