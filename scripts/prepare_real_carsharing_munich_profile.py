#!/usr/bin/env python3
"""Prepare Munich real car-sharing trips as a zone-level dynamic OD profile.

The script converts the Munich shared-mobility Parquet trip files into the
same lightweight profile consumed by ``run_real_bikeshare_station_benchmark``.
For the first RealCar pilot, each high-activity grid cell is treated as a
free-floating inventory zone: cars can be moved between zones, and observed
derived trips provide hourly OD demand.

This is intentionally a Q1/prototype bridge, not a thesis-writing artifact.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


REQUIRED_COLUMNS = {
    "id",
    "startlat",
    "startlon",
    "starttime",
    "endlat",
    "endlon",
    "endtime",
}


@dataclass(frozen=True)
class Trip:
    vehicle_id: str
    started_at: datetime
    ended_at: datetime
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


def unix_to_datetime(value: object) -> datetime | None:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(raw) or raw <= 0:
        return None
    # Dataset documentation says Unix timestamp. Accept ms just in case.
    if raw > 10_000_000_000:
        raw /= 1000.0
    return datetime.fromtimestamp(raw, tz=timezone.utc).replace(tzinfo=None)


def hour_key(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0).isoformat(sep=" ")


def dt_key(dt: datetime) -> str:
    return dt.isoformat(sep=" ")


def parse_date_bound(raw: str | None, *, end: bool = False) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if end and fmt == "%Y-%m-%d":
                return dt.replace(hour=23, minute=59, second=59)
            return dt
        except ValueError:
            continue
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def grid_key(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    return (int(math.floor(lat / cell_deg)), int(math.floor(lon / cell_deg)))


def grid_label(key: tuple[int, int]) -> str:
    return f"cell_{key[0]}_{key[1]}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def iter_parquet_records(
    path: Path,
    *,
    required_columns: set[str] | None = None,
) -> Iterator[dict[str, object]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "This script needs pyarrow to read Parquet. Run it in the project "
            "Docker environment, e.g. docker exec tesis_container python3 ..."
        ) from exc
    if str(path).endswith(".gz"):
        import pyarrow as pa  # type: ignore

        with gzip.open(path, "rb") as fh:
            table = pq.read_table(pa.BufferReader(fh.read()))
    else:
        table = pq.read_table(path)
    if required_columns:
        missing = sorted(required_columns - set(table.column_names))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
    for batch in table.to_batches(max_chunksize=100_000):
        names = batch.schema.names
        columns = {name: batch.column(name).to_pylist() for name in names}
        n = batch.num_rows
        for idx in range(n):
            yield {name: columns[name][idx] for name in names}


def import_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "This script needs pandas + pyarrow. Run it in the project Docker "
            "environment, e.g. docker exec tesis_container python3 ..."
        ) from exc
    return pd


def iter_trips(
    input_paths: Iterable[Path],
    *,
    start_at: datetime | None,
    end_at: datetime | None,
    min_duration_sec: float,
    max_duration_sec: float,
    max_rows: int | None,
    allowed_vehicle_ids: set[str] | None,
) -> Iterable[Trip]:
    yielded = 0
    for path in input_paths:
        for data in iter_parquet_records(path, required_columns=REQUIRED_COLUMNS):
            vehicle_id = str(data.get("id", ""))
            if allowed_vehicle_ids is not None and vehicle_id not in allowed_vehicle_ids:
                continue
            started = unix_to_datetime(data.get("starttime"))
            ended = unix_to_datetime(data.get("endtime"))
            if started is None or ended is None:
                continue
            if start_at is not None and started < start_at:
                continue
            if end_at is not None and started > end_at:
                continue
            duration = (ended - started).total_seconds()
            if duration < min_duration_sec or duration > max_duration_sec:
                continue
            try:
                start_lat = float(data["startlat"])
                start_lon = float(data["startlon"])
                end_lat = float(data["endlat"])
                end_lon = float(data["endlon"])
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (start_lat, start_lon, end_lat, end_lon)):
                continue
            yielded += 1
            yield Trip(
                vehicle_id=vehicle_id,
                started_at=started,
                ended_at=ended,
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
            )
            if max_rows is not None and yielded >= max_rows:
                return


def load_allowed_vehicle_ids(
    *,
    metadata_paths: Iterable[Path],
    fuel_type: str,
) -> set[str]:
    wanted = fuel_type.strip().upper()
    if not wanted:
        return set()
    allowed: set[str] = set()
    for path in metadata_paths:
        for row in iter_parquet_records(path):
            row_fuel = str(row.get("fuel_type", "")).strip().upper()
            if row_fuel == wanted:
                allowed.add(str(row.get("id", "")))
    return {vehicle_id for vehicle_id in allowed if vehicle_id}


def build_profile(args: argparse.Namespace) -> None:
    input_paths = [Path(p) for p in args.input]
    vehicle_metadata_paths = [Path(p) for p in args.vehicle_metadata]
    start_at = parse_date_bound(args.start_date)
    end_at = parse_date_bound(args.end_date, end=True)
    cell_deg = float(args.grid_cell_deg)
    fuel_type = str(args.fuel_type).strip()
    allowed_vehicle_ids: set[str] | None = None
    if fuel_type:
        if not vehicle_metadata_paths:
            raise ValueError("--fuel-type requires --vehicle-metadata.")
        allowed_vehicle_ids = load_allowed_vehicle_ids(
            metadata_paths=vehicle_metadata_paths,
            fuel_type=fuel_type,
        )
        if not allowed_vehicle_ids:
            raise ValueError(f"No vehicle ids found for fuel_type={fuel_type!r}.")

    trips = list(
        iter_trips(
            input_paths,
            start_at=start_at,
            end_at=end_at,
            min_duration_sec=float(args.min_duration_sec),
            max_duration_sec=float(args.max_duration_sec),
            max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
            allowed_vehicle_ids=allowed_vehicle_ids,
        )
    )
    if not trips:
        raise ValueError("No trips left after filtering.")

    departures: Counter[tuple[int, int]] = Counter()
    arrivals: Counter[tuple[int, int]] = Counter()
    point_sums: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    trip_cells: list[tuple[Trip, tuple[int, int], tuple[int, int]]] = []
    for trip in trips:
        origin = grid_key(trip.start_lat, trip.start_lon, cell_deg)
        dest = grid_key(trip.end_lat, trip.end_lon, cell_deg)
        departures[origin] += 1
        arrivals[dest] += 1
        point_sums[origin][0] += trip.start_lat
        point_sums[origin][1] += trip.start_lon
        point_sums[origin][2] += 1.0
        point_sums[dest][0] += trip.end_lat
        point_sums[dest][1] += trip.end_lon
        point_sums[dest][2] += 1.0
        trip_cells.append((trip, origin, dest))

    top_cells = [
        cell for cell, _ in (departures + arrivals).most_common(int(args.top_zones))
    ]
    top_set = set(top_cells)
    cell_names = {cell: f"munich_zone_{rank:02d}" for rank, cell in enumerate(top_cells, start=1)}
    total_activity = sum(departures[c] + arrivals[c] for c in top_cells) or 1

    station_rows: list[dict[str, object]] = []
    for rank, cell in enumerate(top_cells, start=1):
        lat_sum, lon_sum, n = point_sums[cell]
        station_rows.append(
            {
                "rank": rank,
                "station": cell_names[cell],
                "departures": departures[cell],
                "arrivals": arrivals[cell],
                "total_activity": departures[cell] + arrivals[cell],
                "net_arrivals_minus_departures": arrivals[cell] - departures[cell],
                "activity_share": round((departures[cell] + arrivals[cell]) / total_activity, 6),
                "lat": round(lat_sum / max(1.0, n), 7),
                "lng": round(lon_sum / max(1.0, n), 7),
                "munich_grid_cell": grid_label(cell),
            }
        )

    hourly_dep: Counter[tuple[str, str]] = Counter()
    hourly_arr: Counter[tuple[str, str]] = Counter()
    od_hourly: Counter[tuple[str, str, str]] = Counter()
    event_rows: list[dict[str, object]] = []
    retained_trip_count = 0

    for trip, origin_cell, dest_cell in trip_cells:
        if origin_cell not in top_set or dest_cell not in top_set:
            continue
        origin = cell_names[origin_cell]
        dest = cell_names[dest_cell]
        h = hour_key(trip.started_at)
        hourly_dep[(h, origin)] += 1
        hourly_arr[(h, dest)] += 1
        od_hourly[(h, origin, dest)] += 1
        retained_trip_count += 1
        if args.emit_event_stream:
            duration_min = round((trip.ended_at - trip.started_at).total_seconds() / 60.0, 4)
            common = {
                "trip_id": retained_trip_count,
                "vehicle_id": trip.vehicle_id,
                "start_station": origin,
                "end_station": dest,
                "started_at": dt_key(trip.started_at),
                "ended_at": dt_key(trip.ended_at),
                "duration_min": duration_min,
                "start_lat": round(trip.start_lat, 7),
                "start_lng": round(trip.start_lon, 7),
                "end_lat": round(trip.end_lat, 7),
                "end_lng": round(trip.end_lon, 7),
            }
            event_rows.append(
                {
                    "event_time": dt_key(trip.started_at),
                    "event_type": "rental",
                    "event_station": origin,
                    **common,
                }
            )
            event_rows.append(
                {
                    "event_time": dt_key(trip.ended_at),
                    "event_type": "return",
                    "event_station": dest,
                    **common,
                }
            )

    all_hours = sorted({h for h, _ in hourly_dep} | {h for h, _ in hourly_arr})
    station_names = [cell_names[cell] for cell in top_cells]
    hourly_rows: list[dict[str, object]] = []
    for h in all_hours:
        hour_total = sum(hourly_dep[(h, station)] for station in station_names) or 1
        for row in station_rows:
            station = str(row["station"])
            dep = hourly_dep[(h, station)]
            arr = hourly_arr[(h, station)]
            base = max(1.0, float(row["departures"]) / max(1, len(all_hours)))
            hourly_rows.append(
                {
                    "hour": h,
                    "station": station,
                    "departures": dep,
                    "arrivals": arr,
                    "net_arrivals_minus_departures": arr - dep,
                    "departure_share_in_hour": round(dep / hour_total, 6),
                    "departure_multiplier_vs_station_mean": round(dep / base, 6),
                }
            )

    od_rows = [
        {"hour": h, "origin": origin, "destination": dest, "trips": count}
        for (h, origin, dest), count in sorted(od_hourly.items())
    ]
    if not od_rows:
        raise ValueError("Top-zone filter removed all OD rows.")

    event_rows.sort(key=lambda row: (str(row["event_time"]), str(row["event_type"]), int(row["trip_id"])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "station_summary.csv", station_rows)
    write_csv(out_dir / "hourly_station_profile.csv", hourly_rows)
    write_csv(out_dir / "hourly_od_top_stations.csv", od_rows)
    if args.emit_event_stream:
        write_csv(out_dir / "event_stream_top_stations.csv", event_rows)

    metadata = {
        "profile_type": "realcar_munich_zone_level",
        "source_files": [str(p) for p in input_paths],
        "provider_hint": str(args.provider),
        "vehicle_metadata_files": [str(p) for p in vehicle_metadata_paths],
        "fuel_type_filter": fuel_type or None,
        "allowed_vehicle_count": None if allowed_vehicle_ids is None else len(allowed_vehicle_ids),
        "start_date": None if start_at is None else start_at.isoformat(sep=" "),
        "end_date": None if end_at is None else end_at.isoformat(sep=" "),
        "grid_cell_deg": cell_deg,
        "top_zones": int(args.top_zones),
        "valid_trips_after_filters": len(trips),
        "retained_top_zone_trips": retained_trip_count,
        "hours": len(all_hours),
        "min_hour": all_hours[0] if all_hours else None,
        "max_hour": all_hours[-1] if all_hours else None,
        "outputs": [
            "station_summary.csv",
            "hourly_station_profile.csv",
            "hourly_od_top_stations.csv",
        ],
        "interpretation": (
            "Zone-level free-floating carsharing profile derived from Munich "
            "shared-mobility trips. It preserves observed OD timing but does "
            "not include measured battery state, operator relocations, pricing or live traffic."
        ),
    }
    if args.emit_event_stream:
        metadata["outputs"].append("event_stream_top_stations.csv")
        metadata["event_stream_rows"] = len(event_rows)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("PREPARE_REAL_CARSHARING_MUNICH_PROFILE_OK")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Munich car-sharing trips as a RealCar OD profile.")
    parser.add_argument("--input", nargs="+", required=True, help="trips_{provider}.parquet.gz file(s).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--provider", default="unknown")
    parser.add_argument("--vehicle-metadata", nargs="*", default=[], help="Optional vehicle metadata parquet file(s).")
    parser.add_argument("--fuel-type", default="", help="Optional fuel_type filter, e.g. ELECTRIC.")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--top-zones", type=int, default=20)
    parser.add_argument("--grid-cell-deg", type=float, default=0.01)
    parser.add_argument("--min-duration-sec", type=float, default=60.0)
    parser.add_argument("--max-duration-sec", type=float, default=6 * 3600.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--emit-event-stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_profile(parse_args())


if __name__ == "__main__":
    main()
