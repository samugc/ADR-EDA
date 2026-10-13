from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .travel_metrics import (
    build_fallback_travel_metrics,
    load_travel_metrics_json,
    travel_time_min,
    zone_distance_km,
    zone_drive_time_min_at_time,
)
from .types import Charger, MicroZone, OpsResource, Request, RobustModelConfig, SystemState, Vehicle


def default_travel_metrics_json() -> Optional[str]:
    """Return the maintained OSRM travel-matrix path when it exists."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "tests" / "model" / "results" / "madrid_osrm_zone_matrix_congestion_2025_01.json",
        repo_root / "tests" / "model" / "results" / "madrid_osrm_zone_matrix.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@dataclass
class ScenarioConfig:
    fleet_size: int = 220
    charger_count: int = 22
    ops_resources: int = 26
    battery_kwh: float = 48.0
    consumption_kwh_km: float = 0.145
    reservation_share: float = 0.28
    reservation_booking_intensity_mult: float = 1.0
    immediate_request_intensity_mult: float = 1.0
    reservation_min_lead_min: float = 10.0
    reservation_mean_lead_min: float = 35.0
    reservation_max_lead_min: float = 180.0
    reservation_cancel_base_prob: float = 0.06
    reservation_cancel_lead_sensitivity: float = 0.22
    reservation_cancel_cap: float = 0.35
    min_trip_km: float = 0.65
    min_same_zone_trip_km: float = 1.00
    same_zone_trip_km_mean: float = 1.35
    same_zone_trip_km_std: float = 0.45
    destination_same_zone_bonus: float = 0.22
    destination_distance_decay_km: float = 0.045
    destination_duration_decay_min: float = 0.012
    fare_mode: str = "time_only"
    fare_base_eur: float = 0.0
    fare_per_km_eur: float = 0.0
    fare_per_min_eur: float = 0.24
    fare_min_total_eur: float = 0.0
    fare_included_km: float = 200.0
    fare_extra_km_eur: float = 0.39
    fare_airport_pickup_fee_eur: float = 0.0
    fare_airport_dropoff_fee_eur: float = 5.99
    travel_metrics_json: Optional[str] = field(default_factory=default_travel_metrics_json)
    zone_base_demand_json: Optional[str] = None


class MadridFreeFloatingScenario:
    _CORE_ZONES = frozenset({"Centro", "Salamanca", "Chamartin", "Tetuan", "Retiro", "Arganzuela", "Moncloa"})
    _IMMEDIATE_REQUEST_PROFILE = (
        0.38,
        0.33,
        0.30,
        0.28,
        0.31,
        0.50,
        0.84,
        1.10,
        1.26,
        1.18,
        1.02,
        0.96,
        1.00,
        1.08,
        1.14,
        1.08,
        1.04,
        1.14,
        1.32,
        1.36,
        1.18,
        0.92,
        0.70,
        0.54,
    )
    _RESERVATION_BOOKING_PROFILE = (
        0.12,
        0.10,
        0.10,
        0.10,
        0.12,
        0.18,
        0.32,
        0.50,
        0.68,
        0.84,
        0.98,
        1.08,
        1.14,
        1.20,
        1.26,
        1.30,
        1.28,
        1.24,
        1.18,
        1.04,
        0.80,
        0.54,
        0.30,
        0.18,
    )
    _IMMEDIATE_ZONE_CLASS_PROFILE = {
        "core": (
            0.82,
            0.78,
            0.76,
            0.74,
            0.76,
            0.88,
            1.02,
            1.10,
            1.08,
            1.04,
            1.02,
            1.04,
            1.08,
            1.12,
            1.14,
            1.12,
            1.08,
            1.10,
            1.18,
            1.22,
            1.16,
            1.02,
            0.92,
            0.86,
        ),
        "outer": (
            0.88,
            0.84,
            0.82,
            0.80,
            0.86,
            1.02,
            1.20,
            1.28,
            1.16,
            1.02,
            0.94,
            0.92,
            0.94,
            0.98,
            1.02,
            1.04,
            1.08,
            1.16,
            1.24,
            1.18,
            1.06,
            0.94,
            0.88,
            0.84,
        ),
        "airport": (
            0.74,
            0.70,
            0.68,
            0.70,
            0.84,
            1.06,
            1.28,
            1.36,
            1.22,
            1.04,
            0.98,
            0.96,
            0.98,
            1.02,
            1.06,
            1.10,
            1.14,
            1.22,
            1.30,
            1.26,
            1.14,
            1.00,
            0.86,
            0.78,
        ),
    }

    def __init__(self, cfg: ScenarioConfig | None = None):
        self.cfg = cfg or ScenarioConfig()
        self.zones = self._build_default_zones(
            base_demands=self._load_zone_base_demands(self.cfg.zone_base_demand_json)
        )

    @staticmethod
    def _default_zone_base_demands() -> Dict[str, float]:
        return {
            "Centro": 24.5,
            "Salamanca": 18.5,
            "Chamartin": 14.5,
            "Tetuan": 12.8,
            "Retiro": 16.0,
            "Arganzuela": 13.0,
            "Moncloa": 14.5,
            "Latina": 9.8,
            "Carabanchel": 8.9,
            "Usera": 8.1,
            "Vallecas": 8.8,
            "Airport": 9.4,
        }

    @staticmethod
    def _build_default_zones(base_demands: Optional[Dict[str, float]] = None) -> Dict[str, MicroZone]:
        # Centers are synthetic but geographically coherent for Madrid urban area.
        demand = MadridFreeFloatingScenario._default_zone_base_demands()
        if base_demands:
            for zone_id, value in base_demands.items():
                if zone_id in demand:
                    demand[zone_id] = float(value)
        return {
            "Centro": MicroZone("Centro", 40.4168, -3.7038, demand["Centro"]),
            "Salamanca": MicroZone("Salamanca", 40.4302, -3.6774, demand["Salamanca"]),
            "Chamartin": MicroZone("Chamartin", 40.4726, -3.6822, demand["Chamartin"]),
            "Tetuan": MicroZone("Tetuan", 40.4600, -3.6976, demand["Tetuan"]),
            "Retiro": MicroZone("Retiro", 40.4114, -3.6761, demand["Retiro"]),
            "Arganzuela": MicroZone("Arganzuela", 40.3988, -3.6994, demand["Arganzuela"]),
            "Moncloa": MicroZone("Moncloa", 40.4359, -3.7194, demand["Moncloa"]),
            "Latina": MicroZone("Latina", 40.4078, -3.7415, demand["Latina"]),
            "Carabanchel": MicroZone("Carabanchel", 40.3906, -3.7242, demand["Carabanchel"]),
            "Usera": MicroZone("Usera", 40.3840, -3.7002, demand["Usera"]),
            "Vallecas": MicroZone("Vallecas", 40.3815, -3.6215, demand["Vallecas"]),
            "Airport": MicroZone("Airport", 40.4983, -3.5676, demand["Airport"]),
        }

    @staticmethod
    def _load_zone_base_demands(path: Optional[str]) -> Optional[Dict[str, float]]:
        if not path:
            return None
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("zone_base_demand_json debe contener un objeto {zona: valor}.")
        out: Dict[str, float] = {}
        for zone_id, value in raw.items():
            out[str(zone_id)] = float(value)
        return out
    def initial_state(self, seed: int, model_cfg: RobustModelConfig) -> SystemState:
        rng = np.random.default_rng(int(seed))
        zones = self.zones
        zone_ids = list(zones.keys())

        demand_weights = np.array([zones[z].base_demand_per_hour for z in zone_ids], dtype=float)
        demand_weights = demand_weights / np.sum(demand_weights)

        vehicles: Dict[str, Vehicle] = {}
        for i in range(self.cfg.fleet_size):
            zone_id = str(rng.choice(zone_ids, p=demand_weights))
            soc = float(np.clip(rng.normal(0.58, 0.17), 0.18, 0.95))
            soh = float(np.clip(rng.normal(0.93, 0.03), 0.82, 0.995))
            vehicles[f"EV_{i:04d}"] = Vehicle(
                vehicle_id=f"EV_{i:04d}",
                zone_id=zone_id,
                battery_capacity_kwh=self.cfg.battery_kwh,
                consumption_kwh_km=self.cfg.consumption_kwh_km,
                soc=soc,
                soh=soh,
            )

        chargers: Dict[str, Charger] = {}
        charger_zones = rng.choice(zone_ids, size=self.cfg.charger_count, replace=True, p=demand_weights)
        for i, zone_id in enumerate(charger_zones):
            cap = int(rng.integers(2, 7))
            p_kw = float(rng.choice([7.4, 11.0, 22.0], p=[0.35, 0.45, 0.20]))
            chargers[f"CH_{i:03d}"] = Charger(
                charger_id=f"CH_{i:03d}",
                zone_id=str(zone_id),
                capacity_slots=cap,
                power_kw=p_kw,
            )

        resources: Dict[str, OpsResource] = {}
        for i in range(self.cfg.ops_resources):
            zone_id = str(rng.choice(zone_ids, p=demand_weights))
            resources[f"OPS_{i:03d}"] = OpsResource(resource_id=f"OPS_{i:03d}", zone_id=zone_id)

        if self.cfg.travel_metrics_json:
            travel_metrics = load_travel_metrics_json(
                path=str(self.cfg.travel_metrics_json),
                zones=zones,
                avg_drive_speed_kmh=float(model_cfg.avg_drive_speed_kmh),
            )
        else:
            travel_metrics = build_fallback_travel_metrics(
                zones=zones,
                avg_drive_speed_kmh=float(model_cfg.avg_drive_speed_kmh),
            )

        return SystemState(
            current_time_min=0.0,
            zones=zones,
            vehicles=vehicles,
            chargers=chargers,
            ops_resources=resources,
            travel_metrics=travel_metrics,
        )

    def forecast_demand(
        self,
        now_min: float,
        horizon_min: float,
        external_multiplier: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        ext = external_multiplier or {}
        mult_time = horizon_min / 60.0

        out: Dict[str, float] = {}
        for zone_id, zone in self.zones.items():
            m = float(ext.get(zone_id, 1.0))
            h_factor = self._immediate_zone_hour_weight(zone_id=zone_id, t_min=now_min)
            out[zone_id] = max(0.0, zone.base_demand_per_hour * h_factor * mult_time * m)
        return out

    def sample_requests(
        self,
        state: SystemState,
        start_min: float,
        end_min: float,
        rng: np.random.Generator,
        external_multiplier: Dict[str, float] | None = None,
        origin_zone_filter: Optional[Set[str]] = None,
    ) -> List[Request]:
        external_multiplier = external_multiplier or {}
        horizon_h = max(1e-6, (end_min - start_min) / 60.0)

        zone_ids = list(self.zones.keys())
        base_weights = np.array([self.zones[z].base_demand_per_hour for z in zone_ids], dtype=float)
        base_weights = base_weights / np.sum(base_weights)

        reqs: List[Request] = []
        rid = 0
        for zone_id, zone in self.zones.items():
            if origin_zone_filter is not None and zone_id not in origin_zone_filter:
                continue
            multiplier = float(external_multiplier.get(zone_id, 1.0))
            immediate_hour_weight = self._mean_immediate_zone_hour_weight(
                zone_id=zone_id,
                start_min=start_min,
                end_min=end_min,
            )
            reservation_booking_hour_weight = self._mean_hour_profile_weight(
                start_min=start_min,
                end_min=end_min,
                kind="reservation_booking",
            )
            reservation_scale = self._mean_reservation_share_prob(
                origin_zone_id=zone_id,
                start_min=start_min,
                end_min=end_min,
            )
            immediate_scale = max(0.0, 1.0 - reservation_scale)
            lam_immediate = max(
                0.0,
                zone.base_demand_per_hour
                * immediate_hour_weight
                * multiplier
                * horizon_h
                * immediate_scale
                * float(self.cfg.immediate_request_intensity_mult),
            )
            lam_reservation = max(
                0.0,
                zone.base_demand_per_hour
                * reservation_booking_hour_weight
                * multiplier
                * horizon_h
                * reservation_scale
                * float(self.cfg.reservation_booking_intensity_mult)
            )

            request_batches = [
                (False, int(rng.poisson(lam_immediate))),
                (True, int(rng.poisson(lam_reservation))),
            ]

            for is_reservation, n in request_batches:
                if n <= 0:
                    continue
                time_offsets = rng.uniform(0.0, end_min - start_min, size=n)
                for dt in time_offsets:
                # Destination choice should reflect both attraction and travel impedance.
                    w = self._destination_weights(
                        origin_zone_id=zone_id,
                        candidate_zone_ids=zone_ids,
                        base_weights=base_weights,
                        state=state,
                        request_time_min=float(start_min + dt),
                    )
                    dest = str(rng.choice(zone_ids, p=w))

                    if dest == zone_id:
                        d_km = float(
                            np.clip(
                                rng.normal(self.cfg.same_zone_trip_km_mean, self.cfg.same_zone_trip_km_std),
                                0.15,
                                6.0,
                            )
                        )
                        if d_km < float(self.cfg.min_same_zone_trip_km):
                            # Avoid unrealistic ultra-short intra-zone bookings by switching
                            # to an external destination instead of dropping the request.
                            ext_zone_ids = [z for z in zone_ids if z != zone_id]
                            ext_base_weights = np.array(
                                [self.zones[z].base_demand_per_hour for z in ext_zone_ids],
                                dtype=float,
                            )
                            ext_base_weights = ext_base_weights / np.sum(ext_base_weights)
                            ext_weights = self._destination_weights(
                                origin_zone_id=zone_id,
                                candidate_zone_ids=ext_zone_ids,
                                base_weights=ext_base_weights,
                                state=state,
                                request_time_min=float(start_min + dt),
                            )
                            dest = str(rng.choice(ext_zone_ids, p=ext_weights))
                            trip_prof = self._trip_profile(
                                origin_zone_id=zone_id,
                                destination_zone_id=dest,
                                request_time_min=float(start_min + dt),
                            )
                            d_km = max(
                                float(self.cfg.min_trip_km),
                                zone_distance_km(
                                    zone_id,
                                    dest,
                                    self.zones,
                                    travel_metrics=state.travel_metrics,
                                )
                                * float(rng.uniform(trip_prof["distance_mult_low"], trip_prof["distance_mult_high"])),
                            )
                        else:
                            d_km = max(float(self.cfg.min_trip_km), d_km)
                    else:
                        trip_prof = self._trip_profile(
                            origin_zone_id=zone_id,
                            destination_zone_id=dest,
                            request_time_min=float(start_min + dt),
                        )
                        d_km = max(
                            float(self.cfg.min_trip_km),
                            zone_distance_km(
                                zone_id,
                                dest,
                                self.zones,
                                travel_metrics=state.travel_metrics,
                            )
                            * float(rng.uniform(trip_prof["distance_mult_low"], trip_prof["distance_mult_high"])),
                        )
                    base_drive_min = zone_drive_time_min_at_time(
                        zone_a=zone_id,
                        zone_b=dest,
                        zones=self.zones,
                        travel_metrics=state.travel_metrics,
                        fallback_speed_kmh=22.0,
                        departure_time_min=float(start_min + dt),
                    )
                    if dest == zone_id:
                        base_drive_min = travel_time_min(d_km, 22.0)
                    trip_prof = self._trip_profile(
                        origin_zone_id=zone_id,
                        destination_zone_id=dest,
                        request_time_min=float(start_min + dt),
                    )
                    duration = max(
                        4.0,
                        float(base_drive_min)
                        + float(
                            rng.normal(
                                float(trip_prof["duration_buffer_mean_min"]),
                                float(trip_prof["duration_buffer_std_min"]),
                            )
                        ),
                    )
                    fare_breakdown = self._trip_revenue_breakdown(
                        origin_zone_id=zone_id,
                        destination_zone_id=dest,
                        distance_km=float(d_km),
                        duration_min=float(duration),
                    )
                    revenue = float(fare_breakdown["total_eur"])

                    booking_time = float(start_min + dt)
                    pickup_time = booking_time
                    cancellation_time = None
                    if is_reservation:
                        lead_mean = self._reservation_mean_lead_min(
                            origin_zone_id=zone_id,
                            booking_time_min=booking_time,
                        )
                        raw_lead = float(rng.exponential(max(1.0, lead_mean)))
                        lead_min = max(1.0, float(self.cfg.reservation_min_lead_min))
                        lead_max = max(lead_min, float(self.cfg.reservation_max_lead_min))
                        lead = max(lead_min, min(lead_max, raw_lead))
                        pickup_time = booking_time + lead

                        p_cancel = self._reservation_cancel_prob(
                            origin_zone_id=zone_id,
                            booking_time_min=booking_time,
                            pickup_time_min=pickup_time,
                            lead_min=lead,
                        )
                        if rng.random() < p_cancel:
                            cancel_lb = booking_time + 1.0
                            cancel_ub = max(cancel_lb + 0.5, pickup_time - 0.5)
                            cancellation_time = float(rng.uniform(cancel_lb, cancel_ub))

                    reqs.append(
                        Request(
                            request_id=f"R_{int(start_min):04d}_{rid:05d}",
                            request_time_min=booking_time,
                            origin_zone_id=zone_id,
                            destination_zone_id=dest,
                            distance_km=float(d_km),
                            duration_min=float(duration),
                            revenue_eur=float(revenue),
                            pickup_time_min=float(pickup_time),
                            is_reservation=is_reservation,
                            cancellation_time_min=cancellation_time,
                            fare_breakdown=fare_breakdown,
                        )
                    )
                    rid += 1

        reqs.sort(key=lambda r: r.request_time_min)
        return reqs

    def _destination_weights(
        self,
        origin_zone_id: str,
        candidate_zone_ids: List[str],
        base_weights: np.ndarray,
        state: SystemState,
        request_time_min: float,
    ) -> np.ndarray:
        deterrence = []
        for zone_id in candidate_zone_ids:
            if zone_id == origin_zone_id:
                deterrence.append(1.0 + float(self.cfg.destination_same_zone_bonus))
                continue
            d_km = zone_distance_km(
                origin_zone_id,
                zone_id,
                self.zones,
                travel_metrics=state.travel_metrics,
            )
            t_min = zone_drive_time_min_at_time(
                zone_a=origin_zone_id,
                zone_b=zone_id,
                zones=self.zones,
                travel_metrics=state.travel_metrics,
                fallback_speed_kmh=22.0,
                departure_time_min=request_time_min,
            )
            penalty = (
                float(self.cfg.destination_distance_decay_km) * float(d_km)
                + float(self.cfg.destination_duration_decay_min) * float(t_min)
            )
            deterrence.append(float(np.exp(-max(0.0, penalty))))

        weights = np.asarray(base_weights, dtype=float) * np.asarray(deterrence, dtype=float)
        total = float(np.sum(weights))
        if total <= 1e-12:
            return np.full(len(candidate_zone_ids), 1.0 / max(1, len(candidate_zone_ids)), dtype=float)
        return weights / total

    def _zone_class(self, zone_id: str) -> str:
        if zone_id == "Airport":
            return "airport"
        if zone_id in self._CORE_ZONES:
            return "core"
        return "outer"

    def _trip_class(self, origin_zone_id: str, destination_zone_id: str) -> str:
        origin_class = self._zone_class(origin_zone_id)
        dest_class = self._zone_class(destination_zone_id)
        if origin_zone_id == destination_zone_id:
            return f"same_{origin_class}"
        if "airport" in (origin_class, dest_class):
            return "airport_link"
        if origin_class == "core" and dest_class == "core":
            return "core_core"
        if origin_class == "outer" and dest_class == "outer":
            return "outer_outer"
        return "core_outer"

    def _trip_profile(
        self,
        origin_zone_id: str,
        destination_zone_id: str,
        request_time_min: float,
    ) -> Dict[str, float]:
        trip_class = self._trip_class(origin_zone_id, destination_zone_id)
        hour = int((request_time_min // 60.0) % 24)
        if trip_class == "same_core":
            profile = {
                "distance_mult_low": 1.00,
                "distance_mult_high": 1.00,
                "duration_buffer_mean_min": 4.0,
                "duration_buffer_std_min": 1.4,
            }
        elif trip_class == "same_outer":
            profile = {
                "distance_mult_low": 1.00,
                "distance_mult_high": 1.00,
                "duration_buffer_mean_min": 3.0,
                "duration_buffer_std_min": 1.2,
            }
        elif trip_class == "same_airport":
            profile = {
                "distance_mult_low": 1.00,
                "distance_mult_high": 1.00,
                "duration_buffer_mean_min": 5.0,
                "duration_buffer_std_min": 1.6,
            }
        elif trip_class == "core_core":
            profile = {
                "distance_mult_low": 0.90,
                "distance_mult_high": 1.12,
                "duration_buffer_mean_min": 3.5,
                "duration_buffer_std_min": 1.5,
            }
        elif trip_class == "outer_outer":
            profile = {
                "distance_mult_low": 1.02,
                "distance_mult_high": 1.28,
                "duration_buffer_mean_min": 4.5,
                "duration_buffer_std_min": 1.8,
            }
        elif trip_class == "airport_link":
            profile = {
                "distance_mult_low": 1.05,
                "distance_mult_high": 1.25,
                "duration_buffer_mean_min": 6.0,
                "duration_buffer_std_min": 2.0,
            }
        else:
            profile = {
                "distance_mult_low": 0.98,
                "distance_mult_high": 1.20,
                "duration_buffer_mean_min": 4.0,
                "duration_buffer_std_min": 1.7,
            }

        if 17 <= hour < 22:
            profile["duration_buffer_mean_min"] += 0.6
        elif 0 <= hour < 6:
            profile["duration_buffer_mean_min"] -= 0.4

        return profile

    def _reservation_zone_class(self, origin_zone_id: str) -> str:
        return self._zone_class(origin_zone_id)

    def _reservation_hour_profile(self, booking_time_min: float) -> Dict[str, float]:
        hour = int((booking_time_min // 60.0) % 24)
        if 6 <= hour < 10:
            return {"share_mult": 1.10, "lead_mult": 1.20, "cancel_mult": 0.92}
        if 10 <= hour < 17:
            return {"share_mult": 1.00, "lead_mult": 1.00, "cancel_mult": 1.00}
        if 17 <= hour < 22:
            return {"share_mult": 1.15, "lead_mult": 1.10, "cancel_mult": 0.98}
        return {"share_mult": 0.72, "lead_mult": 0.85, "cancel_mult": 1.08}

    def _reservation_pickup_hour_profile(self, pickup_time_min: float) -> Dict[str, float]:
        hour = int((pickup_time_min // 60.0) % 24)
        if 6 <= hour < 10:
            return {"lead_mult": 1.18, "cancel_mult": 0.94}
        if 10 <= hour < 17:
            return {"lead_mult": 1.00, "cancel_mult": 1.00}
        if 17 <= hour < 22:
            return {"lead_mult": 1.22, "cancel_mult": 0.96}
        return {"lead_mult": 0.82, "cancel_mult": 1.10}

    def _reservation_zone_profile(self, origin_zone_id: str) -> Dict[str, float]:
        zone_class = self._reservation_zone_class(origin_zone_id)
        if zone_class == "airport":
            return {"share_mult": 1.70, "lead_mult": 1.90, "cancel_mult": 0.80}
        if zone_class == "core":
            return {"share_mult": 0.88, "lead_mult": 0.85, "cancel_mult": 0.95}
        return {"share_mult": 1.08, "lead_mult": 1.15, "cancel_mult": 1.05}

    def _reservation_share_prob(self, origin_zone_id: str, booking_time_min: float) -> float:
        zone_prof = self._reservation_zone_profile(origin_zone_id)
        hour_prof = self._reservation_hour_profile(booking_time_min)
        share = (
            float(self.cfg.reservation_share)
            * float(zone_prof["share_mult"])
            * float(hour_prof["share_mult"])
        )
        return float(np.clip(share, 0.02, 0.85))

    def _reservation_mean_lead_min(self, origin_zone_id: str, booking_time_min: float) -> float:
        zone_prof = self._reservation_zone_profile(origin_zone_id)
        booking_prof = self._reservation_hour_profile(booking_time_min)
        provisional = (
            float(self.cfg.reservation_mean_lead_min)
            * float(zone_prof["lead_mult"])
            * float(booking_prof["lead_mult"])
        )
        lead_min = max(1.0, float(self.cfg.reservation_min_lead_min))
        lead_max = max(lead_min, float(self.cfg.reservation_max_lead_min))
        est_pickup_time_min = float(booking_time_min) + float(np.clip(provisional, lead_min, lead_max))
        pickup_prof = self._reservation_pickup_hour_profile(est_pickup_time_min)
        lead = (
            float(self.cfg.reservation_mean_lead_min)
            * float(zone_prof["lead_mult"])
            * (0.55 * float(booking_prof["lead_mult"]) + 0.45 * float(pickup_prof["lead_mult"]))
        )
        return float(np.clip(lead, lead_min, lead_max))

    def _reservation_cancel_prob(
        self,
        origin_zone_id: str,
        booking_time_min: float,
        pickup_time_min: float,
        lead_min: float,
    ) -> float:
        lead_max = max(1.0, float(self.cfg.reservation_max_lead_min))
        lead_ratio = min(1.0, float(lead_min) / lead_max)
        zone_prof = self._reservation_zone_profile(origin_zone_id)
        booking_prof = self._reservation_hour_profile(booking_time_min)
        pickup_prof = self._reservation_pickup_hour_profile(pickup_time_min)
        base = (
            float(self.cfg.reservation_cancel_base_prob)
            + float(self.cfg.reservation_cancel_lead_sensitivity) * lead_ratio
        )
        cancel = (
            base
            * float(zone_prof["cancel_mult"])
            * (0.6 * float(booking_prof["cancel_mult"]) + 0.4 * float(pickup_prof["cancel_mult"]))
        )
        return float(np.clip(cancel, 0.0, float(self.cfg.reservation_cancel_cap)))

    def _hour_profile_weight(self, t_min: float, kind: str) -> float:
        hour = int((t_min // 60.0) % 24)
        if kind == "immediate":
            profile = np.asarray(self._IMMEDIATE_REQUEST_PROFILE, dtype=float)
        elif kind == "reservation_booking":
            profile = np.asarray(self._RESERVATION_BOOKING_PROFILE, dtype=float)
        else:
            raise ValueError(f"Unknown hour profile kind={kind!r}")
        norm = float(np.mean(profile))
        return float(profile[hour] / max(1e-9, norm))

    @staticmethod
    def _interval_sample_times(start_min: float, end_min: float, target_step_min: float = 15.0) -> List[float]:
        duration = max(1e-6, float(end_min) - float(start_min))
        n_steps = max(1, int(np.ceil(duration / max(1.0, float(target_step_min)))))
        step = duration / float(n_steps)
        return [float(start_min) + (idx + 0.5) * step for idx in range(n_steps)]

    def _mean_hour_profile_weight(self, start_min: float, end_min: float, kind: str) -> float:
        times = self._interval_sample_times(start_min=start_min, end_min=end_min)
        values = [self._hour_profile_weight(t_min=t, kind=kind) for t in times]
        return float(np.mean(values))

    def _immediate_zone_hour_weight(self, zone_id: str, t_min: float) -> float:
        hour = int((t_min // 60.0) % 24)
        zone_class = self._zone_class(zone_id)
        global_profile = np.asarray(self._IMMEDIATE_REQUEST_PROFILE, dtype=float)
        class_profile = np.asarray(self._IMMEDIATE_ZONE_CLASS_PROFILE[zone_class], dtype=float)
        combined = global_profile * class_profile
        norm = float(np.mean(combined))
        return float(combined[hour] / max(1e-9, norm))

    def _mean_immediate_zone_hour_weight(self, zone_id: str, start_min: float, end_min: float) -> float:
        times = self._interval_sample_times(start_min=start_min, end_min=end_min)
        values = [self._immediate_zone_hour_weight(zone_id=zone_id, t_min=t) for t in times]
        return float(np.mean(values))

    def _mean_reservation_share_prob(self, origin_zone_id: str, start_min: float, end_min: float) -> float:
        times = self._interval_sample_times(start_min=start_min, end_min=end_min)
        values = [self._reservation_share_prob(origin_zone_id=origin_zone_id, booking_time_min=t) for t in times]
        return float(np.mean(values))

    def _trip_revenue_breakdown(
        self,
        origin_zone_id: str,
        destination_zone_id: str,
        distance_km: float,
        duration_min: float,
    ) -> Dict[str, float]:
        mode = str(self.cfg.fare_mode or "time_only").strip().lower()
        base = float(self.cfg.fare_base_eur)
        per_km = float(self.cfg.fare_per_km_eur)
        per_min = float(self.cfg.fare_per_min_eur)
        min_total = float(self.cfg.fare_min_total_eur)
        included_km = max(0.0, float(self.cfg.fare_included_km))
        extra_km = max(0.0, float(distance_km) - included_km)
        extra_km_fee = float(self.cfg.fare_extra_km_eur) * extra_km
        airport_pickup_fee = float(self.cfg.fare_airport_pickup_fee_eur) if origin_zone_id == "Airport" else 0.0
        airport_dropoff_fee = (
            float(self.cfg.fare_airport_dropoff_fee_eur) if destination_zone_id == "Airport" else 0.0
        )

        if mode == "time_only":
            time_component = per_min * float(duration_min)
            distance_component = 0.0
        elif mode == "time_plus_km":
            time_component = per_min * float(duration_min)
            distance_component = per_km * float(distance_km)
        else:
            raise ValueError(f"Unsupported fare_mode={self.cfg.fare_mode!r}")

        trip_fare = max(min_total, float(base + time_component + distance_component + extra_km_fee))
        total = trip_fare + airport_pickup_fee + airport_dropoff_fee
        return {
            "base_component_eur": float(base),
            "time_component_eur": float(time_component),
            "distance_component_eur": float(distance_component),
            "distance_overage_component_eur": float(extra_km_fee),
            "airport_pickup_component_eur": float(airport_pickup_fee),
            "airport_dropoff_component_eur": float(airport_dropoff_fee),
            "trip_fare_eur": float(trip_fare),
            "total_eur": float(total),
        }


class RealTripDemandScenario(MadridFreeFloatingScenario):
    """Free-floating scenario whose demand comes from an observed trip stream.

    The simulator, encoding and three-objective evaluator remain unchanged; this
    subclass only replaces synthetic request generation with observed OD timing.
    """

    demand_source = "real_trip_stream"

    def __init__(
        self,
        cfg: ScenarioConfig | None = None,
        *,
        profile_dir: str,
        forecast_history_min: float = 1440.0,
    ):
        self.profile_dir = Path(profile_dir)
        self.forecast_history_min = max(60.0, float(forecast_history_min))
        self.real_metadata = self._load_metadata(self.profile_dir)
        super().__init__(cfg=cfg)
        self.zones = self._load_profile_zones(self.profile_dir)
        self._real_request_templates = self._load_real_request_templates(self.profile_dir)
        self.real_request_count = len(self._real_request_templates)
        self.real_time_origin = (
            None
            if not self._real_request_templates
            else str(self._real_request_templates[0]["started_at"])
        )
        self._hourly_profile = self._build_hourly_profile()

    @staticmethod
    def _parse_dt(raw: str) -> datetime:
        text = str(raw).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        raise ValueError(f"Fecha no reconocida en demanda real: {raw!r}")

    @staticmethod
    def _load_metadata(profile_dir: Path) -> Dict[str, object]:
        path = profile_dir / "metadata.json"
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _load_profile_zones(profile_dir: Path) -> Dict[str, MicroZone]:
        path = profile_dir / "station_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"No existe station_summary.csv en {profile_dir}")

        meta = RealTripDemandScenario._load_metadata(profile_dir)
        hours = max(1.0, float(meta.get("hours", 1.0)))
        zones: Dict[str, MicroZone] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                zone_id = str(row.get("station", "")).strip()
                if not zone_id:
                    continue
                departures = float(row.get("departures", 0.0) or 0.0)
                arrivals = float(row.get("arrivals", 0.0) or 0.0)
                demand_per_hour = max(1e-6, (departures + 0.10 * arrivals) / hours)
                zones[zone_id] = MicroZone(
                    zone_id=zone_id,
                    lat=float(row["lat"]),
                    lon=float(row["lng"]),
                    base_demand_per_hour=float(demand_per_hour),
                )

        if not zones:
            raise ValueError(f"Perfil real sin zonas válidas: {path}")
        return dict(sorted(zones.items(), key=lambda kv: kv[0]))

    def _load_real_request_templates(self, profile_dir: Path) -> List[Dict[str, object]]:
        path = profile_dir / "event_stream_top_stations.csv"
        if not path.exists():
            raise FileNotFoundError(f"No existe event_stream_top_stations.csv en {profile_dir}")

        rows: List[Dict[str, object]] = []
        origin_dt: Optional[datetime] = None
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("event_type", "")).strip() != "rental":
                    continue
                origin = str(row.get("start_station", "")).strip()
                dest = str(row.get("end_station", "")).strip()
                if origin not in self.zones or dest not in self.zones:
                    continue
                started_at = self._parse_dt(str(row.get("started_at", "")))
                ended_at = self._parse_dt(str(row.get("ended_at", "")))
                if origin_dt is None or started_at < origin_dt:
                    origin_dt = started_at
                rows.append(
                    {
                        "trip_id": str(row.get("trip_id", "")),
                        "vehicle_id": str(row.get("vehicle_id", "")),
                        "origin": origin,
                        "dest": dest,
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "observed_duration_min": max(
                            1.0,
                            float(row.get("duration_min", 0.0) or 0.0),
                            (ended_at - started_at).total_seconds() / 60.0,
                        ),
                    }
                )

        if origin_dt is None:
            raise ValueError(f"No hay eventos rental válidos en {path}")

        templates: List[Dict[str, object]] = []
        for idx, row in enumerate(sorted(rows, key=lambda r: r["started_at"])):
            request_time_min = (row["started_at"] - origin_dt).total_seconds() / 60.0
            templates.append({**row, "request_time_min": float(request_time_min), "idx": int(idx)})
        return templates

    def _build_hourly_profile(self) -> Dict[int, Dict[str, float]]:
        counts: Dict[int, Dict[str, float]] = {h: {} for h in range(24)}
        for row in self._real_request_templates:
            hour = int((float(row["request_time_min"]) // 60.0) % 24)
            origin = str(row["origin"])
            counts[hour][origin] = counts[hour].get(origin, 0.0) + 1.0
        return counts

    def forecast_demand(
        self,
        now_min: float,
        horizon_min: float,
        external_multiplier: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        del external_multiplier
        horizon_h = max(1e-6, float(horizon_min) / 60.0)
        history_start = max(0.0, float(now_min) - self.forecast_history_min)
        counts = {zone_id: 0.0 for zone_id in self.zones}
        for row in self._real_request_templates:
            t = float(row["request_time_min"])
            if history_start <= t < float(now_min):
                counts[str(row["origin"])] += 1.0

        history_h = max(1e-6, (float(now_min) - history_start) / 60.0)
        if sum(counts.values()) > 0.0:
            return {zone_id: float(value) / history_h * horizon_h for zone_id, value in counts.items()}

        # Cold-start fallback: a static historical profile learned before replay.
        hour = int((float(now_min) // 60.0) % 24)
        hour_counts = self._hourly_profile.get(hour, {})
        total_hour = float(sum(hour_counts.values()))
        if total_hour > 0.0:
            base_total = sum(z.base_demand_per_hour for z in self.zones.values())
            return {
                zone_id: max(0.0, float(hour_counts.get(zone_id, 0.0)) / total_hour)
                * base_total
                * horizon_h
                for zone_id in self.zones
            }
        return {
            zone_id: max(0.0, zone.base_demand_per_hour * horizon_h)
            for zone_id, zone in self.zones.items()
        }

    def sample_requests(
        self,
        state: SystemState,
        start_min: float,
        end_min: float,
        rng: np.random.Generator,
        external_multiplier: Dict[str, float] | None = None,
        origin_zone_filter: Optional[Set[str]] = None,
    ) -> List[Request]:
        del rng, external_multiplier
        reqs: List[Request] = []
        for row in self._real_request_templates:
            t = float(row["request_time_min"])
            if not (float(start_min) <= t < float(end_min)):
                continue
            origin = str(row["origin"])
            if origin_zone_filter is not None and origin not in origin_zone_filter:
                continue
            dest = str(row["dest"])
            d_km = max(
                float(self.cfg.min_trip_km),
                zone_distance_km(origin, dest, self.zones, travel_metrics=state.travel_metrics),
            )
            osrm_duration = zone_drive_time_min_at_time(
                zone_a=origin,
                zone_b=dest,
                zones=self.zones,
                travel_metrics=state.travel_metrics,
                fallback_speed_kmh=22.0,
                departure_time_min=t,
            )
            duration = max(float(row["observed_duration_min"]), float(osrm_duration), 1.0)
            fare_breakdown = self._trip_revenue_breakdown(
                origin_zone_id=origin,
                destination_zone_id=dest,
                distance_km=float(d_km),
                duration_min=float(duration),
            )
            reqs.append(
                Request(
                    request_id=f"REAL_{int(row['idx']):06d}",
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
        return reqs


def build_free_floating_scenario(
    *,
    cfg: ScenarioConfig,
    real_demand_profile_dir: str = "",
    real_demand_forecast_history_min: float = 1440.0,
) -> MadridFreeFloatingScenario:
    profile_dir = str(real_demand_profile_dir).strip()
    if profile_dir:
        return RealTripDemandScenario(
            cfg=cfg,
            profile_dir=profile_dir,
            forecast_history_min=float(real_demand_forecast_history_min),
        )
    return MadridFreeFloatingScenario(cfg=cfg)
