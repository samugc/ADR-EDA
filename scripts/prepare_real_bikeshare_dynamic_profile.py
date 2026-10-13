#!/usr/bin/env python3
"""Prepare real bike-share trip histories as dynamic OD demand profiles.

The script is intentionally dependency-light: it reads Citi Bike / Divvy style
CSV files directly, including CSVs stored inside ZIP archives, and builds
hourly station-level demand profiles. It does not run ADR-EDA by itself; it
creates the real-data layer needed to plug a consolidated public mobility
dataset into a dynamic multiobjective fleet/rebalancing evaluator.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, TextIO


MADRID_ZONE_ORDER = [
    "Centro",
    "Salamanca",
    "Chamartin",
    "Tetuan",
    "Retiro",
    "Arganzuela",
    "Moncloa",
    "Latina",
    "Carabanchel",
    "Usera",
    "Vallecas",
    "Airport",
]


@dataclass
class Trip:
    started_at: datetime
    ended_at: datetime | None
    start_station: str
    end_station: str
    start_lat: float | None = None
    start_lng: float | None = None
    end_lat: float | None = None
    end_lng: float | None = None


@dataclass(frozen=True)
class StationLookup:
    name: str
    lat: float | None
    lng: float | None


def parse_dt(raw: str) -> datetime | None:
    value = (raw or "").strip()
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def parse_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_geo_point(raw: str) -> tuple[float | None, float | None]:
    value = (raw or "").strip()
    if not value:
        return None, None
    try:
        payload = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    coords = payload.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None, None
    lng = parse_float(str(coords[0]))
    lat = parse_float(str(coords[1]))
    return lat, lng


def first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def is_data_member(member: str, suffix: str) -> bool:
    member_path = Path(member)
    return (
        member.lower().endswith(suffix)
        and not member.endswith("/")
        and "__MACOSX" not in member_path.parts
        and not member_path.name.startswith("._")
    )


def dict_reader(fh: TextIO) -> csv.DictReader:
    header = fh.readline()
    delimiter = ";" if header.count(";") > header.count(",") else ","
    return csv.DictReader(chain([header], fh), delimiter=delimiter)


def iter_csv_handles(path: Path) -> Iterator[tuple[str, TextIO]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if is_data_member(member, ".csv"):
                    with zf.open(member, "r") as raw:
                        yield member, io.TextIOWrapper(
                            raw, encoding="utf-8-sig", errors="replace", newline=""
                        )
                elif is_data_member(member, ".zip"):
                    nested_bytes = io.BytesIO(zf.read(member))
                    with zipfile.ZipFile(nested_bytes) as nested_zf:
                        for nested_member in nested_zf.namelist():
                            if not is_data_member(nested_member, ".csv"):
                                continue
                            with nested_zf.open(nested_member, "r") as raw:
                                yield nested_member, io.TextIOWrapper(
                                    raw, encoding="utf-8-sig", errors="replace", newline=""
                                )
        return
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        yield path.name, fh


def load_station_lookup(paths: list[Path], explicit_paths: list[Path]) -> dict[str, StationLookup]:
    lookup: dict[str, StationLookup] = {}

    def consume(fh: TextIO) -> None:
        reader = dict_reader(fh)
        for row in reader:
            code = first_present(row, ["Code", "code", "station_code", "station_id", "Station ID"])
            name = first_present(row, ["name", "Name", "station", "Station Name"])
            if not code or not name:
                continue
            lookup[code] = StationLookup(
                name=name,
                lat=parse_float(first_present(row, ["latitude", "lat", "Latitude"])),
                lng=parse_float(first_present(row, ["longitude", "lng", "lon", "Longitude"])),
            )

    for path in explicit_paths:
        for _, fh in iter_csv_handles(path):
            consume(fh)
    for path in paths:
        if path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                name = Path(member).name.lower()
                if name.startswith("stations") and name.endswith(".csv"):
                    with zf.open(member, "r") as raw:
                        import io

                        consume(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline=""))
    return lookup


def resolve_station_from_code(
    row: dict[str, str],
    *,
    name_fields: list[str],
    code_fields: list[str],
    lookup: dict[str, StationLookup],
) -> tuple[str, float | None, float | None]:
    name = first_present(row, name_fields)
    code = first_present(row, code_fields)
    if code and code in lookup:
        station = lookup[code]
        return station.name, station.lat, station.lng
    return name or code, None, None


def iter_trips(
    paths: list[Path],
    min_duration_sec: float,
    max_duration_sec: float | None,
    station_lookup: dict[str, StationLookup],
) -> Iterator[Trip]:
    for path in paths:
        for source_name, fh in iter_csv_handles(path):
            if Path(source_name).name.lower().startswith("stations"):
                continue
            reader = dict_reader(fh)
            for row in reader:
                started = parse_dt(
                    first_present(
                        row,
                        [
                            "started_at",
                            "starttime",
                            "start_time",
                            "Start Time",
                            "start_date",
                            "unlock_date",
                        ],
                    )
                )
                if started is None:
                    continue
                ended = parse_dt(
                    first_present(
                        row,
                        ["ended_at", "stoptime", "end_time", "Stop Time", "end_date", "lock_date"],
                    )
                )
                if ended is not None:
                    duration_sec = (ended - started).total_seconds()
                    if duration_sec < min_duration_sec:
                        continue
                    if max_duration_sec is not None and duration_sec > max_duration_sec:
                        continue
                start_station, lookup_start_lat, lookup_start_lng = resolve_station_from_code(
                    row,
                    name_fields=[
                        "start_station_name",
                        "from_station_name",
                        "start station name",
                        "Start Station Name",
                        "unlock_station_name",
                    ],
                    code_fields=[
                        "start_station_code",
                        "start_station_id",
                        "start station id",
                        "station_unlock",
                    ],
                    lookup=station_lookup,
                )
                end_station, lookup_end_lat, lookup_end_lng = resolve_station_from_code(
                    row,
                    name_fields=[
                        "end_station_name",
                        "to_station_name",
                        "end station name",
                        "End Station Name",
                        "lock_station_name",
                    ],
                    code_fields=[
                        "end_station_code",
                        "end_station_id",
                        "end station id",
                        "station_lock",
                    ],
                    lookup=station_lookup,
                )
                if not start_station or not end_station:
                    continue
                geo_start_lat, geo_start_lng = parse_geo_point(
                    first_present(row, ["geolocation_unlock", "start_geolocation"])
                )
                geo_end_lat, geo_end_lng = parse_geo_point(
                    first_present(row, ["geolocation_lock", "end_geolocation"])
                )
                start_lat = parse_float(first_present(row, ["start_lat", "start station latitude"]))
                start_lng = parse_float(first_present(row, ["start_lng", "start station longitude"]))
                end_lat = parse_float(first_present(row, ["end_lat", "end station latitude"]))
                end_lng = parse_float(first_present(row, ["end_lng", "end station longitude"]))
                yield Trip(
                    started_at=started,
                    ended_at=ended,
                    start_station=start_station,
                    end_station=end_station,
                    start_lat=first_not_none(start_lat, geo_start_lat, lookup_start_lat),
                    start_lng=first_not_none(start_lng, geo_start_lng, lookup_start_lng),
                    end_lat=first_not_none(end_lat, geo_end_lat, lookup_end_lat),
                    end_lng=first_not_none(end_lng, geo_end_lng, lookup_end_lng),
                )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def hour_key(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0).isoformat(sep=" ")


def dt_key(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.isoformat(sep=" ")


def write_event_stream(
    *,
    input_paths: list[Path],
    output_path: Path,
    top_set: set[str],
    min_duration_sec: float,
    max_duration_sec: float | None,
    station_lookup: dict[str, StationLookup],
) -> int:
    rows: list[dict[str, object]] = []
    trip_id = 0
    for trip in iter_trips(
        input_paths,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        station_lookup=station_lookup,
    ):
        if (
            trip.ended_at is None
            or trip.start_station not in top_set
            or trip.end_station not in top_set
        ):
            continue
        trip_id += 1
        duration_min = round((trip.ended_at - trip.started_at).total_seconds() / 60.0, 4)
        common = {
            "trip_id": trip_id,
            "start_station": trip.start_station,
            "end_station": trip.end_station,
            "started_at": dt_key(trip.started_at),
            "ended_at": dt_key(trip.ended_at),
            "duration_min": duration_min,
            "start_lat": "" if trip.start_lat is None else round(trip.start_lat, 7),
            "start_lng": "" if trip.start_lng is None else round(trip.start_lng, 7),
            "end_lat": "" if trip.end_lat is None else round(trip.end_lat, 7),
            "end_lng": "" if trip.end_lng is None else round(trip.end_lng, 7),
        }
        rows.append(
            {
                "event_time": dt_key(trip.started_at),
                "event_type": "rental",
                "event_station": trip.start_station,
                **common,
            }
        )
        rows.append(
            {
                "event_time": dt_key(trip.ended_at),
                "event_type": "return",
                "event_station": trip.end_station,
                **common,
            }
        )
    rows.sort(key=lambda row: (str(row["event_time"]), str(row["event_type"]), int(row["trip_id"])))
    write_csv(output_path, rows)
    return trip_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-stations", type=int, default=12)
    parser.add_argument("--min-duration-sec", type=float, default=60.0)
    parser.add_argument("--max-duration-sec", type=float, default=0.0)
    parser.add_argument("--station-lookup", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--emit-madrid-surrogate",
        action="store_true",
        help="Also map the top stations to the existing Madrid zone names for a quick runner-compatible surrogate.",
    )
    parser.add_argument(
        "--emit-event-stream",
        action="store_true",
        help="Also export chronological rental/return events among the selected top stations.",
    )
    args = parser.parse_args()
    max_duration_sec = float(args.max_duration_sec) if float(args.max_duration_sec) > 0 else None
    station_lookup = load_station_lookup(args.input, args.station_lookup)

    departures: Counter[str] = Counter()
    arrivals: Counter[str] = Counter()
    station_geo: dict[str, tuple[float, float]] = {}
    hourly_departures: Counter[tuple[str, str]] = Counter()
    hourly_arrivals: Counter[tuple[str, str]] = Counter()
    od_hourly: Counter[tuple[str, str, str]] = Counter()
    total_trips = 0
    min_hour = None
    max_hour = None

    for trip in iter_trips(
        args.input,
        min_duration_sec=float(args.min_duration_sec),
        max_duration_sec=max_duration_sec,
        station_lookup=station_lookup,
    ):
        total_trips += 1
        h = hour_key(trip.started_at)
        min_hour = h if min_hour is None else min(min_hour, h)
        max_hour = h if max_hour is None else max(max_hour, h)
        departures[trip.start_station] += 1
        arrivals[trip.end_station] += 1
        hourly_departures[(h, trip.start_station)] += 1
        hourly_arrivals[(h, trip.end_station)] += 1
        od_hourly[(h, trip.start_station, trip.end_station)] += 1
        if trip.start_lat is not None and trip.start_lng is not None:
            station_geo.setdefault(trip.start_station, (trip.start_lat, trip.start_lng))
        if trip.end_lat is not None and trip.end_lng is not None:
            station_geo.setdefault(trip.end_station, (trip.end_lat, trip.end_lng))

    top = [
        station
        for station, _ in (departures + arrivals).most_common(int(args.top_stations))
    ]
    top_set = set(top)
    total_top_demand = sum(departures[s] + arrivals[s] for s in top) or 1

    station_rows = []
    for rank, station in enumerate(top, start=1):
        lat, lng = station_geo.get(station, (None, None))
        station_rows.append(
            {
                "rank": rank,
                "station": station,
                "departures": departures[station],
                "arrivals": arrivals[station],
                "total_activity": departures[station] + arrivals[station],
                "net_arrivals_minus_departures": arrivals[station] - departures[station],
                "activity_share": round((departures[station] + arrivals[station]) / total_top_demand, 6),
                "lat": "" if lat is None else round(lat, 7),
                "lng": "" if lng is None else round(lng, 7),
                "madrid_surrogate_zone": MADRID_ZONE_ORDER[rank - 1]
                if rank <= len(MADRID_ZONE_ORDER)
                else "",
            }
        )

    all_hours = sorted({h for h, _ in hourly_departures} | {h for h, _ in hourly_arrivals})
    hourly_rows = []
    for h in all_hours:
        hour_total = sum(hourly_departures[(h, s)] for s in top) or 1
        for station in top:
            dep = hourly_departures[(h, station)]
            arr = hourly_arrivals[(h, station)]
            base = max(1.0, departures[station] / max(1, len(all_hours)))
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

    od_rows = []
    for (h, origin, dest), count in sorted(od_hourly.items()):
        if origin not in top_set or dest not in top_set:
            continue
        od_rows.append({"hour": h, "origin": origin, "destination": dest, "trips": count})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "station_summary.csv", station_rows)
    write_csv(args.out_dir / "hourly_station_profile.csv", hourly_rows)
    write_csv(args.out_dir / "hourly_od_top_stations.csv", od_rows)

    event_stream_trips = None
    if args.emit_event_stream:
        event_stream_trips = write_event_stream(
            input_paths=args.input,
            output_path=args.out_dir / "event_stream_top_stations.csv",
            top_set=top_set,
            min_duration_sec=float(args.min_duration_sec),
            max_duration_sec=max_duration_sec,
            station_lookup=station_lookup,
        )

    metadata = {
        "source_files": [str(p) for p in args.input],
        "total_valid_trips": total_trips,
        "top_stations": int(args.top_stations),
        "min_hour": min_hour,
        "max_hour": max_hour,
        "station_lookup_entries": len(station_lookup),
        "min_duration_sec": float(args.min_duration_sec),
        "max_duration_sec": max_duration_sec,
        "outputs": [
            "station_summary.csv",
            "hourly_station_profile.csv",
            "hourly_od_top_stations.csv",
        ],
    }
    if args.emit_event_stream:
        metadata["outputs"].append("event_stream_top_stations.csv")
        metadata["event_stream_trips_top_stations"] = event_stream_trips
        metadata["event_stream_rows"] = 0 if event_stream_trips is None else 2 * int(event_stream_trips)

    if args.emit_madrid_surrogate:
        zone_base = {}
        for row in station_rows[: len(MADRID_ZONE_ORDER)]:
            zone_base[str(row["madrid_surrogate_zone"])] = round(
                8.0 + 30.0 * float(row["activity_share"]),
                4,
            )
        (args.out_dir / "madrid_surrogate_zone_base_demand.json").write_text(
            json.dumps(zone_base, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metadata["outputs"].append("madrid_surrogate_zone_base_demand.json")
        metadata["surrogate_warning"] = (
            "This maps real bike-share demand intensity onto the existing Madrid "
            "zone labels. It is useful for a quick ADR-EDA stress test, but not a "
            "full geospatial real-world validation."
        )

    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'station_summary.csv'}")
    print(f"Wrote {args.out_dir / 'hourly_station_profile.csv'}")
    print(f"Wrote {args.out_dir / 'hourly_od_top_stations.csv'}")
    if args.emit_event_stream:
        print(f"Wrote {args.out_dir / 'event_stream_top_stations.csv'}")
    print(f"Wrote {args.out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
