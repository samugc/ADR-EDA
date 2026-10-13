#!/usr/bin/env python3
"""Build a station-to-station OSRM matrix for RealBike profiles."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_stations(profile_dir: Path, limit: int) -> list[dict[str, object]]:
    rows = read_csv_rows(profile_dir / "station_summary.csv")[:limit]
    stations: list[dict[str, object]] = []
    for row in rows:
        stations.append(
            {
                "name": str(row["station"]),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
            }
        )
    if not stations:
        raise ValueError(f"No stations found in {profile_dir}")
    return stations


def build_coords_arg(stations: list[dict[str, object]]) -> str:
    return ";".join(
        f"{float(station['lng']):.6f},{float(station['lat']):.6f}"
        for station in stations
    )


def fetch_table(osrm_base_url: str, profile: str, coords_arg: str) -> dict[str, Any]:
    base = str(osrm_base_url).rstrip("/")
    path = f"/table/v1/{urllib.parse.quote(str(profile).strip())}/{coords_arg}"
    query = urllib.parse.urlencode({"annotations": "distance,duration"})
    url = f"{base}{path}?{query}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def convert_payload(
    *,
    raw: dict[str, Any],
    stations: list[dict[str, object]],
    osrm_base_url: str,
    profile: str,
) -> dict[str, object]:
    if str(raw.get("code", "")) != "Ok":
        raise RuntimeError(f"OSRM table returned code={raw.get('code')!r}")
    distances = raw.get("distances")
    durations = raw.get("durations")
    if not isinstance(distances, list) or not isinstance(durations, list):
        raise RuntimeError("OSRM response does not contain valid distances/durations.")
    if len(distances) != len(stations) or len(durations) != len(stations):
        raise RuntimeError("OSRM matrix dimension does not match selected stations.")

    names = [str(station["name"]) for station in stations]
    out_distance: dict[str, dict[str, float]] = {}
    out_duration: dict[str, dict[str, float]] = {}
    for i, src in enumerate(names):
        d_row: dict[str, float] = {}
        t_row: dict[str, float] = {}
        for j, dst in enumerate(names):
            dist_m = distances[i][j]
            dur_s = durations[i][j]
            if dist_m is None or dur_s is None:
                raise RuntimeError(f"OSRM returned null for {src}->{dst}.")
            d_row[dst] = float(dist_m) / 1000.0
            t_row[dst] = float(dur_s) / 60.0
        out_distance[src] = d_row
        out_duration[src] = t_row

    return {
        "source": f"osrm:{osrm_base_url}",
        "profile": str(profile),
        "distance_km": out_distance,
        "duration_min": out_duration,
        "stations": {
            str(station["name"]): {
                "lat": float(station["lat"]),
                "lng": float(station["lng"]),
            }
            for station in stations
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OSRM station matrix JSON for RealBike.")
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--station-limit", type=int, default=25)
    parser.add_argument("--osrm-base-url", type=str, required=True)
    parser.add_argument("--profile", type=str, default="driving")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = load_stations(args.profile_dir, limit=int(args.station_limit))
    coords_arg = build_coords_arg(stations)
    raw = fetch_table(
        osrm_base_url=str(args.osrm_base_url),
        profile=str(args.profile),
        coords_arg=coords_arg,
    )
    payload = convert_payload(
        raw=raw,
        stations=stations,
        osrm_base_url=str(args.osrm_base_url),
        profile=str(args.profile),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("BUILD_REALBIKE_OSRM_STATION_MATRIX_OK")
    print({"output_json": str(args.output_json)})


if __name__ == "__main__":
    main()
