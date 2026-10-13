from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class VehicleStatus(str, Enum):
    IDLE = "idle"
    SERVING = "serving"
    RELOCATING = "relocating"
    CHARGING = "charging"
    UNAVAILABLE = "unavailable"


class OpsStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"


class TaskType(str, Enum):
    TRIP = "trip"
    RELOCATION = "relocation"
    CHARGE = "charge"


@dataclass
class MicroZone:
    zone_id: str
    lat: float
    lon: float
    base_demand_per_hour: float


@dataclass
class Charger:
    charger_id: str
    zone_id: str
    capacity_slots: int
    power_kw: float
    occupied_vehicle_ids: set[str] = field(default_factory=set)
    reserved_vehicle_ids: set[str] = field(default_factory=set)

    @property
    def free_slots(self) -> int:
        used = len(self.occupied_vehicle_ids) + len(self.reserved_vehicle_ids)
        return max(0, int(self.capacity_slots) - used)


@dataclass
class TravelMetrics:
    source: str = "geodesic-fallback"
    profile: str = "driving"
    distance_km: Dict[str, Dict[str, float]] = field(default_factory=dict)
    duration_min: Dict[str, Dict[str, float]] = field(default_factory=dict)
    congestion_time_bands: list[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OpsResource:
    resource_id: str
    zone_id: str
    status: OpsStatus = OpsStatus.IDLE
    available_at_min: float = 0.0


@dataclass
class VehicleTask:
    task_type: TaskType
    start_min: float
    end_min: float
    origin_zone_id: str
    destination_zone_id: str
    energy_kwh: float = 0.0
    distance_km: float = 0.0
    request_id: Optional[str] = None
    charger_id: Optional[str] = None
    charge_start_soc: Optional[float] = None
    charge_target_soc: Optional[float] = None
    next_task: Optional["VehicleTask"] = None


@dataclass
class Vehicle:
    vehicle_id: str
    zone_id: str
    battery_capacity_kwh: float
    consumption_kwh_km: float
    soc: float
    soh: float
    status: VehicleStatus = VehicleStatus.IDLE
    available_at_min: float = 0.0
    odometer_km: float = 0.0
    current_task: Optional[VehicleTask] = None


@dataclass
class Request:
    request_id: str
    request_time_min: float
    origin_zone_id: str
    destination_zone_id: str
    distance_km: float
    duration_min: float
    revenue_eur: float
    pickup_time_min: Optional[float] = None
    is_reservation: bool = False
    cancellation_time_min: Optional[float] = None
    preassigned_vehicle_id: Optional[str] = None
    preassigned_at_min: Optional[float] = None
    fare_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    bookings_total: int = 0
    canceled_bookings: int = 0
    preassigned_bookings: int = 0
    preassigned_pickups_honored: int = 0
    preassigned_pickups_failed: int = 0
    reservation_pickups_due: int = 0
    requests_total: int = 0
    served_requests: int = 0
    rejected_requests: int = 0
    revenue_eur: float = 0.0
    trip_fare_revenue_eur: float = 0.0
    airport_fee_revenue_eur: float = 0.0
    ops_charge_credit_eur: float = 0.0
    relocation_cost_eur: float = 0.0
    energy_cost_eur: float = 0.0
    ops_cost_eur: float = 0.0
    ops_deadhead_cost_eur: float = 0.0
    penalty_cost_eur: float = 0.0
    walked_minutes_total: float = 0.0
    charged_kwh_total: float = 0.0
    drive_km_total: float = 0.0
    ops_deadhead_km_total: float = 0.0
    soh_loss_total: float = 0.0

    @property
    def service_rate(self) -> float:
        if self.requests_total <= 0:
            return 0.0
        return float(self.served_requests) / float(self.requests_total)

    @property
    def profit_eur(self) -> float:
        return self.revenue_eur + self.ops_charge_credit_eur - (
            self.relocation_cost_eur
            + self.energy_cost_eur
            + self.ops_cost_eur
            + self.ops_deadhead_cost_eur
            + self.penalty_cost_eur
        )


@dataclass
class RobustModelConfig:
    # Time and vision
    decision_interval_min: int = 5
    planning_horizon_min: int = 60
    # Movement
    avg_drive_speed_kmh: float = 22.0
    user_walk_speed_kmh: float = 4.6
    max_user_access_walk_min: float = 12.0
    #   User-to-vehicle access distance inside the same microzone (km).
    intra_zone_walk_km_min: float = 0.08
    intra_zone_walk_km_max: float = 0.55
    # Batery 
    min_soc: float = 0.08
    reserve_soc: float = 0.10
    deep_discharge_soc: float = 0.12
    charge_cc_cutoff_soc: float = 0.80
    charge_cv_tau_min: float = 42.0
    charger_efficiency: float = 0.92
    # Economy and logistic
    ops_cost_profile: str = "custom"
    relocation_cost_per_km: float = 0.22
    ops_deadhead_cost_per_km: float = 0.22
    ops_cost_per_action_eur: float = 1.20
    ops_relocation_handover_min: float = 1.5
    ops_charge_plug_min: float = 2.0
    ops_charge_action_credit_eur: float = 0.0
    energy_cost_per_kwh: float = 0.23
    penalty_unserved_request_eur: float = 0.0
    # State of health
    soh_loss_per_kwh: float = 0.000035
    soh_loss_deep_discharge: float = 0.00035
    # Xu2021-inspired battery wear scoring constants (model-space adaptation).
    gamma_bat: float = 1.30
    kappa_bat: float = 1.00
    battery_wear_ref_per_vehicle_epoch: float = 0.03
    # Online-health calibration constants for the hardened loss_tail objective.
    soh_loss_ref_per_vehicle_epoch: float = 0.000002
    tail_soc_headroom_min: float = 0.18
    tail_soc_headroom_target: float = 0.45
    # Service-quality scoring:
    # - service_rate remains the dominant term,
    # - walk/access quality refines service only when demand is actually served,
    # - reservation reliability rewards honoring preassignments when used.
    service_rate_weight: float = 0.70
    walk_score_weight: float = 0.15
    reservation_reliability_weight: float = 0.15
    service_walk_ref_min: float = 6.0
    # Profit scoring: smooths raw EUR into [0, 1] with adaptive scale so
    # different horizons/fleet sizes stay comparable in MO optimization.
    profit_score_scale_sqrt: float = 45.0
    profit_score_min_scale_eur: float = 250.0
    # Fleet-health scoring mixes Xu2021-inspired throughput wear with
    # window-level SOH loss and lower-tail SOC headroom for online robustness.
    health_twc_weight: float = 0.50
    health_soh_weight: float = 0.20
    health_soc_safety_weight: float = 0.30
    # Simplified user-access assumptions (v1):
    # - If true, requests are generated only in zones with at least one idle EV.
    demand_only_zones_with_idle: bool = False
    # - If true, an EV is eligible only if it is already in request origin zone.
    assign_only_same_origin_zone: bool = False
    # - If true, user-to-vehicle access time is assumed zero.
    assume_zero_user_access_time: bool = False


@dataclass
class SystemState:
    current_time_min: float
    zones: Dict[str, MicroZone]
    vehicles: Dict[str, Vehicle]
    chargers: Dict[str, Charger]
    ops_resources: Dict[str, OpsResource]
    travel_metrics: Optional[TravelMetrics] = None
    pending_reservations: Dict[str, Request] = field(default_factory=dict)
    metrics: SystemMetrics = field(default_factory=SystemMetrics)
