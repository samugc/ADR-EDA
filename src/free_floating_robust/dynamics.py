from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, Optional, Tuple

from .travel_metrics import travel_time_min, walk_time_min, zone_distance_km, zone_drive_time_min_at_time
from .types import (
    Charger,
    OpsResource,
    OpsStatus,
    Request,
    RobustModelConfig,
    SystemState,
    TaskType,
    Vehicle,
    VehicleStatus,
    VehicleTask,
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _intra_zone_access_km(
    request_id: str,
    vehicle_id: str,
    cfg: RobustModelConfig,
) -> float:
    """
    Deterministic pseudo-random access distance within a microzone.
    Keeps evaluations reproducible while avoiding unrealistic zero walk.
    """
    lo = max(0.0, float(cfg.intra_zone_walk_km_min))
    hi = max(lo, float(cfg.intra_zone_walk_km_max))
    if hi <= lo + 1e-12:
        return lo
    token = f"{request_id}|{vehicle_id}".encode("utf-8")
    h = hashlib.blake2s(token, digest_size=8).digest()
    u = int.from_bytes(h, byteorder="big", signed=False) / float(2**64 - 1)
    return lo + (hi - lo) * u


def exp_soc_after_minutes(
    soc0: float,
    minutes: float,
    battery_kwh: float,
    power_kw: float,
    cfg: RobustModelConfig,
) -> float:
    soc0 = clamp(float(soc0), cfg.min_soc, 0.99)
    minutes = max(0.0, float(minutes))
    power_eff_kw = max(0.1, power_kw * cfg.charger_efficiency)

    # Constant-current phase until cutoff SOC.
    cutoff = clamp(cfg.charge_cc_cutoff_soc, cfg.min_soc, 0.95)
    if soc0 < cutoff:
        e_to_cutoff = (cutoff - soc0) * battery_kwh
        t_to_cutoff = 60.0 * e_to_cutoff / power_eff_kw
        if minutes <= t_to_cutoff:
            delta = (power_eff_kw * minutes / 60.0) / max(1e-6, battery_kwh)
            return clamp(soc0 + delta, cfg.min_soc, 0.99)

        minutes -= t_to_cutoff
        soc_cc = cutoff
    else:
        soc_cc = soc0

    # Constant-voltage phase (concave saturation).
    soc_max = 0.98
    tau = max(10.0, cfg.charge_cv_tau_min)
    soc = soc_max - (soc_max - soc_cc) * math.exp(-minutes / tau)
    return clamp(soc, cfg.min_soc, soc_max)


def exp_minutes_to_reach_soc(
    soc0: float,
    target_soc: float,
    battery_kwh: float,
    power_kw: float,
    cfg: RobustModelConfig,
) -> float:
    soc0 = clamp(float(soc0), cfg.min_soc, 0.99)
    target = clamp(float(target_soc), soc0, 0.98)
    if target <= soc0 + 1e-9:
        return 0.0

    power_eff_kw = max(0.1, power_kw * cfg.charger_efficiency)
    cutoff = clamp(cfg.charge_cc_cutoff_soc, cfg.min_soc, 0.95)
    minutes = 0.0

    if soc0 < cutoff:
        soc_cc_target = min(target, cutoff)
        e = (soc_cc_target - soc0) * battery_kwh
        minutes += 60.0 * e / power_eff_kw
        if target <= cutoff:
            return max(0.0, minutes)
        soc0 = cutoff

    # Inverse CV curve.
    soc_max = 0.98
    tau = max(10.0, cfg.charge_cv_tau_min)
    ratio = (soc_max - target) / max(1e-9, soc_max - soc0)
    minutes += -tau * math.log(max(1e-9, ratio))
    return max(0.0, minutes)


def _complete_task(vehicle: Vehicle, state: SystemState, cfg: RobustModelConfig) -> None:
    task = vehicle.current_task
    if task is None:
        return

    if task.task_type in (TaskType.TRIP, TaskType.RELOCATION):
        old_soc = vehicle.soc
        vehicle.soc = clamp(
            old_soc - task.energy_kwh / max(1e-6, vehicle.battery_capacity_kwh),
            0.0,
            1.0,
        )
        soh_loss = cfg.soh_loss_per_kwh * max(0.0, task.energy_kwh)
        if vehicle.soc < cfg.deep_discharge_soc:
            soh_loss += cfg.soh_loss_deep_discharge
        vehicle.soh = clamp(vehicle.soh - soh_loss, 0.55, 1.0)
        state.metrics.soh_loss_total += soh_loss
        vehicle.odometer_km += max(0.0, task.distance_km)
        vehicle.zone_id = task.destination_zone_id

        if task.task_type == TaskType.RELOCATION:
            state.metrics.relocation_cost_eur += cfg.relocation_cost_per_km * max(0.0, task.distance_km)
            if task.next_task is not None and task.next_task.task_type == TaskType.CHARGE:
                charger = state.chargers[task.next_task.charger_id]
                charger.reserved_vehicle_ids.discard(vehicle.vehicle_id)
                charger.occupied_vehicle_ids.add(vehicle.vehicle_id)
                state.metrics.ops_charge_credit_eur += max(0.0, float(cfg.ops_charge_action_credit_eur))
        state.metrics.drive_km_total += max(0.0, task.distance_km)

    elif task.task_type == TaskType.CHARGE:
        charger = state.chargers[task.charger_id]
        charger.reserved_vehicle_ids.discard(vehicle.vehicle_id)
        charger.occupied_vehicle_ids.discard(vehicle.vehicle_id)
        if task.charge_start_soc is not None and task.charge_target_soc is not None:
            gained = max(0.0, task.charge_target_soc - task.charge_start_soc)
            charged_kwh = gained * vehicle.battery_capacity_kwh
            state.metrics.charged_kwh_total += charged_kwh
            state.metrics.energy_cost_eur += cfg.energy_cost_per_kwh * charged_kwh
        vehicle.zone_id = task.destination_zone_id

    vehicle.available_at_min = task.end_min

    if task.next_task is not None:
        next_task = task.next_task
        vehicle.current_task = next_task
        vehicle.status = (
            VehicleStatus.CHARGING if next_task.task_type == TaskType.CHARGE else VehicleStatus.RELOCATING
        )
        return

    vehicle.current_task = None
    if vehicle.soc < cfg.min_soc:
        vehicle.status = VehicleStatus.UNAVAILABLE
    else:
        vehicle.status = VehicleStatus.IDLE


def refresh_system_state(state: SystemState, now_min: float, cfg: RobustModelConfig) -> None:
    now = float(now_min)

    for resource in state.ops_resources.values():
        if resource.available_at_min <= now:
            resource.status = OpsStatus.IDLE

    for vehicle in state.vehicles.values():
        task = vehicle.current_task
        if task is None:
            if vehicle.available_at_min <= now and vehicle.soc >= cfg.min_soc:
                vehicle.status = VehicleStatus.IDLE
            elif vehicle.soc < cfg.min_soc:
                vehicle.status = VehicleStatus.UNAVAILABLE
            continue

        # Continuous SOC update while charging.
        if task.task_type == TaskType.CHARGE:
            elapsed = clamp(now - task.start_min, 0.0, task.end_min - task.start_min)
            if task.charge_start_soc is not None and task.charger_id is not None:
                charger = state.chargers[task.charger_id]
                vehicle.soc = exp_soc_after_minutes(
                    soc0=task.charge_start_soc,
                    minutes=elapsed,
                    battery_kwh=vehicle.battery_capacity_kwh,
                    power_kw=charger.power_kw,
                    cfg=cfg,
                )

        # Resolve completed tasks, including chained ones.
        while vehicle.current_task is not None and now >= vehicle.current_task.end_min - 1e-9:
            _complete_task(vehicle=vehicle, state=state, cfg=cfg)


def find_nearest_available_ops(
    state: SystemState,
    at_time_min: float,
    target_zone_id: str,
) -> Optional[OpsResource]:
    candidates = [
        r
        for r in state.ops_resources.values()
        if r.status == OpsStatus.IDLE and r.available_at_min <= at_time_min + 1e-9
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda r: zone_distance_km(
            r.zone_id,
            target_zone_id,
            state.zones,
            travel_metrics=state.travel_metrics,
        ),
    )


def feasible_vehicle_set_for_request(
    request: Request,
    state: SystemState,
    cfg: RobustModelConfig,
) -> Iterable[Tuple[Vehicle, float]]:
    pickup_time = (
        float(request.pickup_time_min)
        if request.pickup_time_min is not None
        else float(request.request_time_min)
    )
    for vehicle in state.vehicles.values():
        if vehicle.status != VehicleStatus.IDLE:
            continue
        if vehicle.available_at_min > pickup_time + 1e-9:
            continue
        if cfg.assign_only_same_origin_zone and vehicle.zone_id != request.origin_zone_id:
            continue

        e_req = request.distance_km * vehicle.consumption_kwh_km
        soc_after = vehicle.soc - e_req / max(1e-6, vehicle.battery_capacity_kwh)
        if soc_after < (cfg.min_soc + cfg.reserve_soc):
            continue

        if cfg.assume_zero_user_access_time:
            walk_min = 0.0
        else:
            if request.origin_zone_id == vehicle.zone_id:
                # Even within same zone, users rarely stand exactly at the parked EV.
                d_km = _intra_zone_access_km(
                    request_id=request.request_id,
                    vehicle_id=vehicle.vehicle_id,
                    cfg=cfg,
                )
            else:
                d_km = zone_distance_km(
                    request.origin_zone_id,
                    vehicle.zone_id,
                    state.zones,
                    travel_metrics=state.travel_metrics,
                )
            walk_min = walk_time_min(d_km, cfg.user_walk_speed_kmh)
            max_walk = float(getattr(cfg, "max_user_access_walk_min", 0.0))
            if max_walk > 0.0 and walk_min > max_walk:
                continue
        yield vehicle, walk_min


def choose_user_nearest_vehicle(
    request: Request,
    state: SystemState,
    cfg: RobustModelConfig,
) -> Tuple[Optional[Vehicle], float]:
    candidates = list(feasible_vehicle_set_for_request(request=request, state=state, cfg=cfg))
    if not candidates:
        return None, 0.0
    vehicle, walk_min = min(candidates, key=lambda x: x[1])
    return vehicle, float(walk_min)


def schedule_trip_for_request(
    request: Request,
    vehicle: Vehicle,
    walk_min: float,
    state: SystemState,
) -> None:
    pickup_time = (
        float(request.pickup_time_min)
        if request.pickup_time_min is not None
        else float(request.request_time_min)
    )
    start_min = max(pickup_time + walk_min, vehicle.available_at_min)
    end_min = start_min + max(1.0, request.duration_min)
    e_trip = request.distance_km * vehicle.consumption_kwh_km

    vehicle.current_task = VehicleTask(
        task_type=TaskType.TRIP,
        start_min=start_min,
        end_min=end_min,
        origin_zone_id=vehicle.zone_id,
        destination_zone_id=request.destination_zone_id,
        energy_kwh=e_trip,
        distance_km=request.distance_km,
        request_id=request.request_id,
    )
    vehicle.status = VehicleStatus.SERVING
    vehicle.available_at_min = end_min

    state.metrics.served_requests += 1
    state.metrics.revenue_eur += max(0.0, request.revenue_eur)
    if request.fare_breakdown:
        airport_fee = max(
            0.0,
            float(request.fare_breakdown.get("airport_pickup_component_eur", 0.0))
            + float(request.fare_breakdown.get("airport_dropoff_component_eur", 0.0)),
        )
        trip_fare = max(0.0, float(request.fare_breakdown.get("trip_fare_eur", request.revenue_eur)))
        state.metrics.airport_fee_revenue_eur += airport_fee
        state.metrics.trip_fare_revenue_eur += trip_fare
    else:
        state.metrics.trip_fare_revenue_eur += max(0.0, request.revenue_eur)
    state.metrics.walked_minutes_total += max(0.0, walk_min)


def schedule_charge_with_optional_relocation(
    vehicle: Vehicle,
    charger: Charger,
    ops: OpsResource,
    start_time_min: float,
    target_soc: float,
    cfg: RobustModelConfig,
    state: SystemState,
) -> bool:
    if charger.free_slots <= 0:
        return False

    ops_deadhead_km = zone_distance_km(
        ops.zone_id,
        vehicle.zone_id,
        state.zones,
        travel_metrics=state.travel_metrics,
    )
    ops_deadhead_min = zone_drive_time_min_at_time(
        zone_a=ops.zone_id,
        zone_b=vehicle.zone_id,
        zones=state.zones,
        travel_metrics=state.travel_metrics,
        fallback_speed_kmh=cfg.avg_drive_speed_kmh,
        departure_time_min=start_time_min,
    )
    vehicle_move_start = start_time_min + ops_deadhead_min
    d_km = zone_distance_km(
        vehicle.zone_id,
        charger.zone_id,
        state.zones,
        travel_metrics=state.travel_metrics,
    )
    t_reloc = zone_drive_time_min_at_time(
        zone_a=vehicle.zone_id,
        zone_b=charger.zone_id,
        zones=state.zones,
        travel_metrics=state.travel_metrics,
        fallback_speed_kmh=cfg.avg_drive_speed_kmh,
        departure_time_min=vehicle_move_start,
    )
    e_reloc = d_km * vehicle.consumption_kwh_km

    soc_after_reloc = vehicle.soc - e_reloc / max(1e-6, vehicle.battery_capacity_kwh)
    if soc_after_reloc < cfg.min_soc:
        return False

    drive_arrival = vehicle_move_start + t_reloc
    handover_min = max(0.0, float(cfg.ops_relocation_handover_min))
    plug_min = max(0.0, float(cfg.ops_charge_plug_min))
    charge_start = drive_arrival + handover_min + plug_min
    t_charge = exp_minutes_to_reach_soc(
        soc0=soc_after_reloc,
        target_soc=target_soc,
        battery_kwh=vehicle.battery_capacity_kwh,
        power_kw=charger.power_kw,
        cfg=cfg,
    )

    charge_task = VehicleTask(
        task_type=TaskType.CHARGE,
        start_min=charge_start,
        end_min=charge_start + t_charge,
        origin_zone_id=charger.zone_id,
        destination_zone_id=charger.zone_id,
        charger_id=charger.charger_id,
        charge_start_soc=soc_after_reloc,
        charge_target_soc=target_soc,
    )

    reloc_task = VehicleTask(
        task_type=TaskType.RELOCATION,
        start_min=vehicle_move_start,
        end_min=charge_start,
        origin_zone_id=vehicle.zone_id,
        destination_zone_id=charger.zone_id,
        energy_kwh=e_reloc,
        distance_km=d_km,
        next_task=charge_task,
    )

    vehicle.current_task = reloc_task
    vehicle.status = VehicleStatus.RELOCATING
    vehicle.available_at_min = charge_task.end_min

    charger.reserved_vehicle_ids.add(vehicle.vehicle_id)
    ops.status = OpsStatus.BUSY
    # Operator remains busy through drive, parking handover and plugging.
    ops.available_at_min = charge_start
    ops.zone_id = charger.zone_id
    state.metrics.ops_cost_eur += cfg.ops_cost_per_action_eur
    state.metrics.ops_deadhead_km_total += max(0.0, ops_deadhead_km)
    state.metrics.ops_deadhead_cost_eur += cfg.ops_deadhead_cost_per_km * max(0.0, ops_deadhead_km)
    return True


def schedule_relocation(
    vehicle: Vehicle,
    destination_zone_id: str,
    ops: OpsResource,
    start_time_min: float,
    cfg: RobustModelConfig,
    state: SystemState,
) -> bool:
    if vehicle.zone_id == destination_zone_id:
        return False

    ops_deadhead_km = zone_distance_km(
        ops.zone_id,
        vehicle.zone_id,
        state.zones,
        travel_metrics=state.travel_metrics,
    )
    ops_deadhead_min = zone_drive_time_min_at_time(
        zone_a=ops.zone_id,
        zone_b=vehicle.zone_id,
        zones=state.zones,
        travel_metrics=state.travel_metrics,
        fallback_speed_kmh=cfg.avg_drive_speed_kmh,
        departure_time_min=start_time_min,
    )
    vehicle_move_start = start_time_min + ops_deadhead_min
    d_km = zone_distance_km(
        vehicle.zone_id,
        destination_zone_id,
        state.zones,
        travel_metrics=state.travel_metrics,
    )
    t_min = zone_drive_time_min_at_time(
        zone_a=vehicle.zone_id,
        zone_b=destination_zone_id,
        zones=state.zones,
        travel_metrics=state.travel_metrics,
        fallback_speed_kmh=cfg.avg_drive_speed_kmh,
        departure_time_min=vehicle_move_start,
    )
    e_kwh = d_km * vehicle.consumption_kwh_km

    soc_after = vehicle.soc - e_kwh / max(1e-6, vehicle.battery_capacity_kwh)
    if soc_after < cfg.min_soc:
        return False

    handover_min = max(0.0, float(cfg.ops_relocation_handover_min))
    vehicle.current_task = VehicleTask(
        task_type=TaskType.RELOCATION,
        start_min=vehicle_move_start,
        end_min=vehicle_move_start + t_min + handover_min,
        origin_zone_id=vehicle.zone_id,
        destination_zone_id=destination_zone_id,
        energy_kwh=e_kwh,
        distance_km=d_km,
    )
    vehicle.status = VehicleStatus.RELOCATING
    vehicle.available_at_min = vehicle_move_start + t_min + handover_min

    ops.status = OpsStatus.BUSY
    # Operator is busy for drive plus parking handover.
    ops.available_at_min = vehicle_move_start + t_min + handover_min
    ops.zone_id = destination_zone_id
    state.metrics.ops_cost_eur += cfg.ops_cost_per_action_eur
    state.metrics.ops_deadhead_km_total += max(0.0, ops_deadhead_km)
    state.metrics.ops_deadhead_cost_eur += cfg.ops_deadhead_cost_per_km * max(0.0, ops_deadhead_km)
    return True
