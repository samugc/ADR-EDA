# Scripts

## Benchmark and reproducibility

| Entry point | Purpose |
| --- | --- |
| **`./scripts/setup.sh`** | Build and check the canonical Docker runtime |
| **`./scripts/run_benchmark.sh`** | Run a small benchmark through Docker; `--help` lists options |
| **`./scripts/run_full_benchmark.sh`** | Run/resume the full eight-method protocol through Docker |
| `collect_benchmark_chunks.py` | Merge chunk metrics and report collection progress |

Run shell entry points from a host with Docker. See
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for protocol, outputs and collection.
The characterized runner in `tests/edas/` remains unchanged.

## Car sharing

- `prepare_real_carsharing_munich_profile.py`: prepare real-demand OD profiles.
- `run_realcar_freefloating_mini_campaign.py`: compact free-floating experiments.
- `run_realcar_munich_night_campaigns_2026_05_28.sh`: historical Munich campaign
  orchestration; inspect its original assumptions and inputs before reuse.

## Bike sharing

- `prepare_real_bikeshare_dynamic_profile.py`: derive demand profiles from trips.
- `build_realbike_osrm_station_matrix.py`: construct routing-distance matrices.
- `run_real_bikeshare_station_benchmark.py`: hourly station-level benchmark.
- `run_real_bikeshare_event_replay.py`: event-level replay.
- `run_realbike_event_replay_campaign.py`: multicity replay orchestration.

## Artifact generation and analysis

- `build_carsharing_thesis_artifacts.py`: consolidated car-sharing tables.
- `build_realbike_thesis_artifacts.py`: final 336-hour tables and figures.
- `build_realbike_final_package.py`: complementary event-replay package.
- `compare_realbike_hourly_event_gap.py`: hourly/event aggregation comparison.

Application tools require their documented input datasets and matrices; the
public benchmark quick start does not reconstruct missing demand inputs.
Date-specific names and historical artifact-generation scripts are preserved
for scientific provenance. Their outputs are indexed in [results](../docs/RESULTS.md).
