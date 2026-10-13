from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .travel_metrics import zone_distance_km
from .dynamics import (
    feasible_vehicle_set_for_request,
    find_nearest_available_ops,
    refresh_system_state,
    schedule_charge_with_optional_relocation,
    schedule_relocation,
)
from .types import Request, RobustModelConfig, SystemState, VehicleStatus


@dataclass
class RobustOperationalPolicy:
    charge_trigger: float
    charge_target: float
    relocation_intensity: float
    min_relocation_soc: float
    demand_weight: float
    preassign_intensity: float
    zone_bias: Dict[str, float]


@dataclass
class EffectiveOperationalPolicy:
    charge_trigger_soc: float
    charge_target_soc: float
    relocation_intensity: float
    min_relocation_soc: float
    demand_weight: float
    preassign_intensity: float
    zone_bias: Dict[str, float]


class RobustDecisionEncoding:
    """
    Decision vector layout:
    - 6 operational parameters
    - |Z| spatial biases (one per zone)
    """

    def __init__(self, zone_ids: Sequence[str]):
        self.zone_ids = list(zone_ids)
        lower = [
            0.12,
            0.55,
            0.00,
            0.20,
            0.00,
            0.00,
        ]
        lower += [0.65 for _ in self.zone_ids]
        upper = [
            0.42,
            0.98,
            0.50,
            0.80,
            1.00,
            1.00,
        ]
        upper += [1.35 for _ in self.zone_ids]
        self.bounds = np.column_stack([np.array(lower, dtype=float), np.array(upper, dtype=float)])

    @property
    def dimension(self) -> int:
        return int(self.bounds.shape[0])

    def decode(self, x: np.ndarray) -> RobustOperationalPolicy:
        vec = np.asarray(x, dtype=float)
        vec = np.clip(vec, self.bounds[:, 0], self.bounds[:, 1])

        zone_bias = {
            zone_id: float(vec[6 + i])
            for i, zone_id in enumerate(self.zone_ids)
        }

        charge_trigger = float(vec[0])
        charge_target = max(charge_trigger + 0.08, float(vec[1]))
        charge_target = min(0.99, charge_target)
        min_reloc = min(float(vec[3]), charge_target)

        return RobustOperationalPolicy(
            charge_trigger=charge_trigger,
            charge_target=charge_target,
            relocation_intensity=float(vec[2]),
            min_relocation_soc=min_reloc,
            demand_weight=float(vec[4]),
            preassign_intensity=float(vec[5]),
            zone_bias=zone_bias,
        )


class RollingHorizonPlanner:
    def __init__(self, cfg: RobustModelConfig):
        self.cfg = cfg

    @staticmethod
    def _stable_unit_hash(key: str) -> float:
        h = hashlib.sha1(str(key).encode("utf-8")).digest()
        num = int.from_bytes(h[:8], byteorder="big", signed=False)
        return float(num) / float((1 << 64) - 1)

    @staticmethod
    def _clip(x: float, lower: float, upper: float) -> float:
        return float(np.clip(float(x), float(lower), float(upper)))

    def _context_signals(
        self,
        state: SystemState,
        demand_forecast: Dict[str, float],
        policy: RobustOperationalPolicy,
        at_time_min: float,
    ) -> Dict[str, float]:
        idle_count = sum(
            1
            for v in state.vehicles.values()
            if v.status == VehicleStatus.IDLE and v.available_at_min <= at_time_min + 1e-9
        )
        total_demand = float(sum(max(0.0, float(v)) for v in demand_forecast.values()))
        max_zone_demand = max((float(v) for v in demand_forecast.values()), default=0.0)
        uniform_share = 1.0 / max(1, len(state.zones))
        peak_share = max_zone_demand / max(total_demand, 1e-9) if total_demand > 1e-9 else uniform_share
        demand_pressure = total_demand / max(1.0, float(idle_count))
        reservation_pressure = float(len(state.pending_reservations)) / max(1.0, float(idle_count))
        low_soc_threshold = max(float(self.cfg.deep_discharge_soc), float(policy.charge_trigger) + 0.05)
        low_soc_share = float(
            np.mean([1.0 if v.soc <= low_soc_threshold else 0.0 for v in state.vehicles.values()])
        ) if state.vehicles else 0.0
        concentration_signal = self._clip(
            (peak_share - uniform_share) / max(1e-9, 1.0 - uniform_share),
            0.0,
            1.0,
        )
        hour = int((float(at_time_min) // 60.0) % 24)
        commuter_peak_signal = 1.0 if 7 <= hour < 10 or 17 <= hour < 21 else 0.0
        demand_signal = self._clip((demand_pressure - 0.75) / 1.75, 0.0, 1.0)
        demand_signal = self._clip(max(demand_signal, 0.35 * commuter_peak_signal * concentration_signal), 0.0, 1.0)
        reservation_signal = self._clip(reservation_pressure / 0.60, 0.0, 1.0)
        low_soc_signal = self._clip(low_soc_share / 0.30, 0.0, 1.0)
        return {
            "demand": demand_signal,
            "reservation": reservation_signal,
            "low_soc": low_soc_signal,
            "concentration": concentration_signal,
        }

    def materialize_policy(
        self,
        state: SystemState,
        demand_forecast: Dict[str, float],
        policy: RobustOperationalPolicy,
        at_time_min: float,
    ) -> EffectiveOperationalPolicy:
        signals = self._context_signals(
            state=state,
            demand_forecast=demand_forecast,
            policy=policy,
            at_time_min=at_time_min,
        )

        charge_trigger = self._clip(
            policy.charge_trigger
            + 0.06 * signals["low_soc"]
            + 0.04 * signals["reservation"]
            - 0.04 * signals["demand"],
            0.12,
            0.45,
        )
        charge_target = self._clip(
            max(
                charge_trigger + 0.08,
                policy.charge_target
                + 0.05 * signals["low_soc"]
                + 0.03 * signals["reservation"],
            ),
            0.55,
            0.99,
        )
        min_relocation_soc = self._clip(
            min(
                charge_target,
                policy.min_relocation_soc
                + 0.06 * signals["low_soc"]
                + 0.04 * signals["demand"],
            ),
            0.20,
            0.85,
        )
        zone_bias = {zone_id: float(policy.zone_bias.get(zone_id, 1.0)) for zone_id in state.zones}

        return EffectiveOperationalPolicy(
            charge_trigger_soc=charge_trigger,
            charge_target_soc=charge_target,
            relocation_intensity=self._clip(
                policy.relocation_intensity
                * (
                    1.0
                    + 0.35 * signals["demand"]
                    + 0.30 * signals["concentration"]
                    + 0.20 * signals["reservation"]
                ),
                0.0,
                0.65,
            ),
            min_relocation_soc=min_relocation_soc,
            demand_weight=self._clip(
                policy.demand_weight + 0.10 * signals["demand"] + 0.15 * signals["concentration"],
                0.0,
                1.0,
            ),
            preassign_intensity=self._clip(
                policy.preassign_intensity
                + 0.18 * signals["reservation"]
                - 0.10 * signals["low_soc"],
                0.0,
                1.0,
            ),
            zone_bias=zone_bias,
        )

    def try_preassign_reservation(
        self,
        request: Request,
        state: SystemState,
        policy: EffectiveOperationalPolicy,
        event_time_min: float,
    ) -> Request:
        if not request.is_reservation:
            return request

        p = float(np.clip(policy.preassign_intensity, 0.0, 1.0))
        if p <= 0.0:
            return request
        if self._stable_unit_hash(request.request_id) > p:
            return request

        reserved_vehicle_ids = {
            str(req.preassigned_vehicle_id)
            for req in state.pending_reservations.values()
            if req.preassigned_vehicle_id is not None and req.request_id != request.request_id
        }

        soc_floor = float(self.cfg.min_soc + self.cfg.reserve_soc)
        best = None
        for veh, _ in feasible_vehicle_set_for_request(request=request, state=state, cfg=self.cfg):
            if veh.vehicle_id in reserved_vehicle_ids:
                continue
            e_req = float(request.distance_km) * float(veh.consumption_kwh_km)
            soc_after = float(veh.soc) - e_req / max(1e-9, float(veh.battery_capacity_kwh))
            if soc_after < soc_floor:
                continue
            d_pick_km = float(
                zone_distance_km(
                    request.origin_zone_id,
                    veh.zone_id,
                    state.zones,
                    travel_metrics=state.travel_metrics,
                )
            )
            row = (d_pick_km, -soc_after, veh.vehicle_id)
            if best is None or row < best:
                best = row

        if best is None:
            return request

        out = copy.copy(request)
        out.preassigned_vehicle_id = str(best[2])
        out.preassigned_at_min = float(event_time_min)
        return out

    def _expected_demand_score(
        self,
        state: SystemState,
        demand_forecast: Dict[str, float],
        policy: EffectiveOperationalPolicy,
    ) -> Dict[str, float]:
        score: Dict[str, float] = {}
        for zone_id in state.zones:
            base = float(demand_forecast.get(zone_id, 0.0))
            bias = float(policy.zone_bias.get(zone_id, 1.0))
            score[zone_id] = max(0.0, base * (1.0 + policy.demand_weight * (bias - 1.0)))
        return score

    @staticmethod
    def _rebalance_targets(
        zone_scores: Dict[str, float],
        idle_by_zone: Dict[str, list],
        zone_ids: Sequence[str],
    ) -> tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
        fleet_total = int(sum(len(idle_by_zone[z]) for z in zone_ids))
        total_score = max(1e-9, sum(zone_scores.values()))
        target_by_zone = {
            z: int(round(fleet_total * float(zone_scores[z]) / total_score))
            for z in zone_ids
        }

        diff = fleet_total - sum(target_by_zone.values())
        if diff != 0:
            ranked = sorted(zone_ids, key=lambda z: zone_scores[z], reverse=(diff > 0))
            step = 1 if diff > 0 else -1
            for zone_id in ranked[: abs(diff)]:
                target_by_zone[zone_id] += step

        current_by_zone = {z: len(idle_by_zone[z]) for z in zone_ids}
        deficit_by_zone = {z: max(0, target_by_zone[z] - current_by_zone[z]) for z in zone_ids}
        surplus_by_zone = {z: max(0, current_by_zone[z] - target_by_zone[z]) for z in zone_ids}
        return target_by_zone, current_by_zone, deficit_by_zone, surplus_by_zone

    def _best_relocation_move(
        self,
        state: SystemState,
        epoch_start_min: float,
        zone_scores: Dict[str, float],
        current_by_zone: Dict[str, int],
        target_by_zone: Dict[str, int],
        surplus_by_zone: Dict[str, int],
        deficit_by_zone: Dict[str, int],
        effective_policy: EffectiveOperationalPolicy,
    ) -> tuple[str, str] | None:
        best_move: tuple[float, str, str] | None = None

        for src in state.zones:
            if surplus_by_zone.get(src, 0) <= 0:
                continue
            vehicle_candidates = [
                v
                for v in state.vehicles.values()
                if v.status == VehicleStatus.IDLE
                and v.zone_id == src
                and v.available_at_min <= epoch_start_min + 1e-9
                and v.soc >= effective_policy.min_relocation_soc
            ]
            if not vehicle_candidates:
                continue

            ops = find_nearest_available_ops(
                state=state,
                at_time_min=epoch_start_min,
                target_zone_id=src,
            )
            if ops is None:
                continue

            deadhead_km = float(
                zone_distance_km(
                    ops.zone_id,
                    src,
                    state.zones,
                    travel_metrics=state.travel_metrics,
                )
            )

            for dst in state.zones:
                if src == dst or deficit_by_zone.get(dst, 0) <= 0:
                    continue

                reloc_km = float(
                    zone_distance_km(
                        src,
                        dst,
                        state.zones,
                        travel_metrics=state.travel_metrics,
                    )
                )
                cost_eur = (
                    float(self.cfg.relocation_cost_per_km) * reloc_km
                    + float(self.cfg.ops_deadhead_cost_per_km) * deadhead_km
                    + float(self.cfg.ops_cost_per_action_eur)
                )
                dst_pressure = float(deficit_by_zone[dst]) / max(1.0, float(target_by_zone.get(dst, 0)))
                src_slack = float(surplus_by_zone[src]) / max(1.0, float(current_by_zone.get(src, 0)))
                benefit = float(zone_scores.get(dst, 0.0)) * (1.0 + dst_pressure) * (1.0 + 0.5 * src_slack)
                score = benefit / (1.0 + cost_eur)
                if best_move is None or score > best_move[0]:
                    best_move = (score, str(src), str(dst))

        if best_move is None or best_move[0] <= 0.0:
            return None
        return best_move[1], best_move[2]

    def plan_epoch(
        self,
        state: SystemState,
        epoch_start_min: float,
        epoch_end_min: float,
        demand_forecast: Dict[str, float],
        policy: RobustOperationalPolicy,
    ) -> None:
        refresh_system_state(state=state, now_min=epoch_start_min, cfg=self.cfg)

        effective_policy = self.materialize_policy(
            state=state,
            demand_forecast=demand_forecast,
            policy=policy,
            at_time_min=float(epoch_start_min),
        )

        idle_vehicles = [
            v
            for v in state.vehicles.values()
            if v.status == VehicleStatus.IDLE and v.available_at_min <= epoch_start_min + 1e-9
        ]
        if not idle_vehicles:
            return

        zone_scores = self._expected_demand_score(state, demand_forecast, effective_policy)
        idle_by_zone: Dict[str, list] = {z: [] for z in state.zones}
        for v in idle_vehicles:
            idle_by_zone[v.zone_id].append(v)

        target_by_zone, current_by_zone, deficit_by_zone, surplus_by_zone = self._rebalance_targets(
            zone_scores=zone_scores,
            idle_by_zone=idle_by_zone,
            zone_ids=list(state.zones.keys()),
        )

        low_soc = sorted(
            [v for v in idle_vehicles if v.soc <= effective_policy.charge_trigger_soc],
            key=lambda v: v.soc,
        )
        for vehicle in low_soc:
            ops = find_nearest_available_ops(
                state=state,
                at_time_min=epoch_start_min,
                target_zone_id=vehicle.zone_id,
            )
            if ops is None:
                break

            chargers_sorted = sorted(
                state.chargers.values(),
                key=lambda c: (
                    zone_distance_km(
                        vehicle.zone_id,
                        c.zone_id,
                        state.zones,
                        travel_metrics=state.travel_metrics,
                    ),
                    float(len(c.occupied_vehicle_ids) + len(c.reserved_vehicle_ids))
                    / max(1.0, float(c.capacity_slots)),
                    float(len(c.occupied_vehicle_ids) + len(c.reserved_vehicle_ids)),
                ),
            )

            assigned = False
            for charger in chargers_sorted:
                if charger.free_slots <= 0:
                    continue
                assigned = schedule_charge_with_optional_relocation(
                    vehicle=vehicle,
                    charger=charger,
                    ops=ops,
                    start_time_min=epoch_start_min,
                    target_soc=effective_policy.charge_target_soc,
                    cfg=self.cfg,
                    state=state,
                )
                if assigned:
                    break

        refresh_system_state(state=state, now_min=epoch_start_min, cfg=self.cfg)

        idle_vehicles = [
            v
            for v in state.vehicles.values()
            if v.status == VehicleStatus.IDLE and v.available_at_min <= epoch_start_min + 1e-9
        ]
        if not idle_vehicles:
            return

        idle_by_zone = {z: [] for z in state.zones}
        for v in idle_vehicles:
            idle_by_zone[v.zone_id].append(v)
        target_by_zone, current_by_zone, deficit_by_zone, surplus_by_zone = self._rebalance_targets(
            zone_scores=zone_scores,
            idle_by_zone=idle_by_zone,
            zone_ids=list(state.zones.keys()),
        )

        move_budget = int(round(effective_policy.relocation_intensity * max(0, len(idle_vehicles))))
        if move_budget <= 0:
            return

        for _ in range(move_budget):
            if not any(s > 0 for s in surplus_by_zone.values()) or not any(d > 0 for d in deficit_by_zone.values()):
                break

            refresh_system_state(state=state, now_min=epoch_start_min, cfg=self.cfg)
            idle_by_zone = {z: [] for z in state.zones}
            for v in state.vehicles.values():
                if v.status == VehicleStatus.IDLE and v.available_at_min <= epoch_start_min + 1e-9:
                    idle_by_zone[v.zone_id].append(v)
            target_by_zone, current_by_zone, deficit_by_zone, surplus_by_zone = self._rebalance_targets(
                zone_scores=zone_scores,
                idle_by_zone=idle_by_zone,
                zone_ids=list(state.zones.keys()),
            )

            best_move = self._best_relocation_move(
                state=state,
                epoch_start_min=epoch_start_min,
                zone_scores=zone_scores,
                current_by_zone=current_by_zone,
                target_by_zone=target_by_zone,
                surplus_by_zone=surplus_by_zone,
                deficit_by_zone=deficit_by_zone,
                effective_policy=effective_policy,
            )
            if best_move is None:
                break
            src, dst = best_move

            vehicle_candidates = [
                v
                for v in idle_by_zone.get(src, [])
                if v.soc >= effective_policy.min_relocation_soc
            ]
            if not vehicle_candidates:
                continue

            ops = find_nearest_available_ops(
                state=state,
                at_time_min=epoch_start_min,
                target_zone_id=src,
            )
            if ops is None:
                break

            vehicle = max(vehicle_candidates, key=lambda v: v.soc)
            moved = schedule_relocation(
                vehicle=vehicle,
                destination_zone_id=dst,
                ops=ops,
                start_time_min=epoch_start_min,
                cfg=self.cfg,
                state=state,
            )
            if moved:
                surplus_by_zone[src] = max(0, surplus_by_zone[src] - 1)
                deficit_by_zone[dst] = max(0, deficit_by_zone[dst] - 1)
