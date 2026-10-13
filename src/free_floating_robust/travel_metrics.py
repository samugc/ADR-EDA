from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .types import MicroZone, TravelMetrics

_CORE_ZONES = frozenset({"Centro", "Salamanca", "Chamartin", "Tetuan", "Retiro", "Arganzuela", "Moncloa"})


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(max(0.0, a)))


def travel_time_min(distance_km: float, speed_kmh: float) -> float:
    speed = max(5.0, float(speed_kmh))
    return 60.0 * max(0.0, float(distance_km)) / speed


def walk_time_min(distance_km: float, walk_speed_kmh: float) -> float:
    speed = max(2.5, float(walk_speed_kmh))
    return 60.0 * max(0.0, float(distance_km)) / speed


def zone_distance_km(
    zone_a: str,
    zone_b: str,
    zones: Mapping[str, MicroZone],
    travel_metrics: Optional[TravelMetrics] = None,
) -> float:
    if travel_metrics is not None:
        row = travel_metrics.distance_km.get(str(zone_a), {})
        value = row.get(str(zone_b))
        if value is not None:
            return max(0.0, float(value))
    za = zones[zone_a]
    zb = zones[zone_b]
    return haversine_km(za.lat, za.lon, zb.lat, zb.lon)


def zone_drive_time_min(
    zone_a: str,
    zone_b: str,
    zones: Mapping[str, MicroZone],
    travel_metrics: Optional[TravelMetrics],
    fallback_speed_kmh: float,
) -> float:
    if travel_metrics is not None:
        row = travel_metrics.duration_min.get(str(zone_a), {})
        value = row.get(str(zone_b))
        if value is not None:
            return max(0.0, float(value))
    return travel_time_min(
        zone_distance_km(zone_a, zone_b, zones, travel_metrics=travel_metrics),
        fallback_speed_kmh,
    )


def _zone_class(zone_id: str) -> str:
    if str(zone_id) == "Airport":
        return "airport"
    if str(zone_id) in _CORE_ZONES:
        return "core"
    return "outer"


def _flow_class(zone_a: str, zone_b: str) -> str:
    class_a = _zone_class(zone_a)
    class_b = _zone_class(zone_b)
    if zone_a == zone_b:
        return f"same_{class_a}"
    if "airport" in (class_a, class_b):
        return "airport_link"
    if class_a == "core" and class_b == "core":
        return "core_core"
    if class_a == "outer" and class_b == "outer":
        return "outer_outer"
    return "core_outer"


def _normalize_hours(raw: Any) -> set[int]:
    if isinstance(raw, list):
        out: set[int] = set()
        for value in raw:
            try:
                out.add(int(value) % 24)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, dict):
        start = raw.get("start_hour")
        end = raw.get("end_hour")
        if start is None or end is None:
            return set()
        start_i = int(start) % 24
        end_i = int(end) % 24
        if start_i <= end_i:
            return set(range(start_i, end_i + 1))
        return set(range(start_i, 24)) | set(range(0, end_i + 1))
    if isinstance(raw, str):
        text = str(raw).strip().lower()
        if not text:
            return set()
        if "-" in text:
            left, right = text.split("-", 1)
            return _normalize_hours({"start_hour": left, "end_hour": right})
        parts = re.findall(r"\d+", text)
        if len(parts) == 1:
            return {int(parts[0]) % 24}
        if len(parts) >= 2:
            return _normalize_hours({"start_hour": parts[0], "end_hour": parts[1]})
    return set()


def congestion_multiplier(
    zone_a: str,
    zone_b: str,
    departure_time_min: float,
    travel_metrics: Optional[TravelMetrics],
) -> float:
    if travel_metrics is None or not getattr(travel_metrics, "congestion_time_bands", None):
        return 1.0

    hour = int((float(departure_time_min) // 60.0) % 24)
    flow_class = _flow_class(zone_a, zone_b)
    class_a = _zone_class(zone_a)
    class_b = _zone_class(zone_b)

    for band in travel_metrics.congestion_time_bands:
        hours = _normalize_hours(band.get("hours"))
        if hour not in hours:
            continue
        multipliers = band.get("multipliers", {})
        if not isinstance(multipliers, dict):
            break
        for key in (
            flow_class,
            f"origin_{class_a}",
            f"destination_{class_b}",
            "all",
        ):
            value = multipliers.get(str(key))
            if value is not None:
                return max(0.35, float(value))
        break
    return 1.0


def zone_drive_time_min_at_time(
    zone_a: str,
    zone_b: str,
    zones: Mapping[str, MicroZone],
    travel_metrics: Optional[TravelMetrics],
    fallback_speed_kmh: float,
    departure_time_min: float,
) -> float:
    base = zone_drive_time_min(
        zone_a=zone_a,
        zone_b=zone_b,
        zones=zones,
        travel_metrics=travel_metrics,
        fallback_speed_kmh=fallback_speed_kmh,
    )
    return base * congestion_multiplier(
        zone_a=zone_a,
        zone_b=zone_b,
        departure_time_min=departure_time_min,
        travel_metrics=travel_metrics,
    )


def build_fallback_travel_metrics(
    zones: Mapping[str, MicroZone],
    avg_drive_speed_kmh: float,
) -> TravelMetrics:
    distance_km: Dict[str, Dict[str, float]] = {}
    duration_min: Dict[str, Dict[str, float]] = {}
    zone_ids = list(zones.keys())
    for src in zone_ids:
        d_row: Dict[str, float] = {}
        t_row: Dict[str, float] = {}
        for dst in zone_ids:
            d_km = 0.0 if src == dst else zone_distance_km(src, dst, zones, travel_metrics=None)
            d_row[str(dst)] = float(d_km)
            t_row[str(dst)] = float(travel_time_min(d_km, avg_drive_speed_kmh))
        distance_km[str(src)] = d_row
        duration_min[str(src)] = t_row
    return TravelMetrics(
        source="geodesic-fallback",
        profile="driving",
        distance_km=distance_km,
        duration_min=duration_min,
    )


def load_travel_metrics_json(
    path: str,
    zones: Mapping[str, MicroZone],
    avg_drive_speed_kmh: float,
) -> TravelMetrics:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fallback = build_fallback_travel_metrics(zones=zones, avg_drive_speed_kmh=avg_drive_speed_kmh)

    raw_distance = payload.get("distance_km", {})
    raw_duration = payload.get("duration_min", {})
    raw_bands = payload.get("congestion_time_bands", [])
    distance_km: Dict[str, Dict[str, float]] = {}
    duration_min: Dict[str, Dict[str, float]] = {}

    for src in zones.keys():
        fallback_d_row = fallback.distance_km.get(str(src), {})
        fallback_t_row = fallback.duration_min.get(str(src), {})
        src_distance = raw_distance.get(str(src), {}) if isinstance(raw_distance, dict) else {}
        src_duration = raw_duration.get(str(src), {}) if isinstance(raw_duration, dict) else {}
        d_row: Dict[str, float] = {}
        t_row: Dict[str, float] = {}
        for dst in zones.keys():
            d_value = src_distance.get(str(dst), fallback_d_row.get(str(dst), 0.0))
            t_value = src_duration.get(str(dst))
            d_km = max(0.0, float(d_value))
            if t_value is None:
                t_min = float(travel_time_min(d_km, avg_drive_speed_kmh))
            else:
                t_min = max(0.0, float(t_value))
            d_row[str(dst)] = d_km
            t_row[str(dst)] = t_min
        distance_km[str(src)] = d_row
        duration_min[str(src)] = t_row

    return TravelMetrics(
        source=str(payload.get("source", "external-matrix")),
        profile=str(payload.get("profile", "driving")),
        distance_km=distance_km,
        duration_min=duration_min,
        congestion_time_bands=list(raw_bands) if isinstance(raw_bands, list) else [],
    )
