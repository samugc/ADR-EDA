#!/usr/bin/env python3
"""Build a zone-to-zone travel matrix for MadridFreeFloatingScenario using OSRM Table API."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict

from free_floating_robust.scenario import MadridFreeFloatingScenario


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "tests/model/results/madrid_osrm_zone_matrix.json"


def _build_coords_arg(scenario: MadridFreeFloatingScenario) -> str:
    parts = []
    for zone in scenario.zones.values():
        parts.append(f"{float(zone.lon):.6f},{float(zone.lat):.6f}")
    return ";".join(parts)


def _fetch_table(osrm_base_url: str, profile: str, coords_arg: str) -> Dict[str, Any]:
    base = str(osrm_base_url).rstrip("/")
    path = f"/table/v1/{urllib.parse.quote(str(profile).strip())}/{coords_arg}"
    query = urllib.parse.urlencode({"annotations": "distance,duration"})
    url = f"{base}{path}?{query}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _convert_payload(raw: Dict[str, Any], scenario: MadridFreeFloatingScenario, osrm_base_url: str, profile: str) -> Dict[str, Any]:
    if str(raw.get("code", "")) != "Ok":
        raise RuntimeError(f"OSRM table respondió code={raw.get('code')!r}")

    zone_ids = list(scenario.zones.keys())
    distances = raw.get("distances")
    durations = raw.get("durations")
    if not isinstance(distances, list) or not isinstance(durations, list):
        raise RuntimeError("La respuesta OSRM no contiene distances/durations válidas.")
    if len(distances) != len(zone_ids) or len(durations) != len(zone_ids):
        raise RuntimeError("La dimensión de la matriz OSRM no coincide con las zonas del escenario.")

    out_distance: Dict[str, Dict[str, float]] = {}
    out_duration: Dict[str, Dict[str, float]] = {}
    for i, src in enumerate(zone_ids):
        d_row: Dict[str, float] = {}
        t_row: Dict[str, float] = {}
        for j, dst in enumerate(zone_ids):
            dist_m = distances[i][j]
            dur_s = durations[i][j]
            if dist_m is None or dur_s is None:
                raise RuntimeError(f"OSRM devolvió un valor nulo para {src}->{dst}.")
            d_row[str(dst)] = float(dist_m) / 1000.0
            t_row[str(dst)] = float(dur_s) / 60.0
        out_distance[str(src)] = d_row
        out_duration[str(src)] = t_row

    return {
        "source": f"osrm:{osrm_base_url}",
        "profile": str(profile),
        "distance_km": out_distance,
        "duration_min": out_duration,
        "zones": {
            zone_id: {
                "lat": float(zone.lat),
                "lon": float(zone.lon),
            }
            for zone_id, zone in scenario.zones.items()
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OSRM zone matrix JSON for free_floating_robust.")
    p.add_argument("--osrm-base-url", type=str, required=True)
    p.add_argument("--profile", type=str, default="driving")
    p.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenario = MadridFreeFloatingScenario()
    coords_arg = _build_coords_arg(scenario)
    raw = _fetch_table(
        osrm_base_url=str(args.osrm_base_url),
        profile=str(args.profile),
        coords_arg=coords_arg,
    )
    payload = _convert_payload(
        raw=raw,
        scenario=scenario,
        osrm_base_url=str(args.osrm_base_url),
        profile=str(args.profile),
    )
    out_path = Path(str(args.output_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("BUILD_OSRM_ZONE_MATRIX_OK")
    print({"output_json": str(out_path)})


if __name__ == "__main__":
    main()
