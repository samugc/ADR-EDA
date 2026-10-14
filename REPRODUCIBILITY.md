# Reproducing ADR-EDA

## Canonical environment

Use Docker Desktop or Docker Engine with the Compose plugin. The benchmark
runtime is defined by `docker/Dockerfile.benchmark`,
`docker/benchmark-runtime.txt` and `docker/docker-compose.yml`.
It uses Python 3.10.19 and `platform: linux/amd64` to preserve the compiled
backend environment, including when the host is Apple Silicon.

```bash
./scripts/setup.sh
./scripts/run_benchmark.sh
```

Setup builds the image and runs `pip check`. The first build may take several
minutes and needs network access. The default benchmark runs ADR-EDA on
`df1_t10` with seed 0, horizon 60, population 40, eight variables and one
generation per step. Options are available with `--help`.

Docker installs the compatible EDAspy revision and applies the ADR-EDA patch
automatically. No separate checkout or native macOS installation is needed.
Dependency versions and source revision are maintained in the
[Docker build configuration](docker/Dockerfile.benchmark).
The public configuration is [`ADR_EDA_PROFILE`](src/adr_eda/profile.py).

## Full protocol and resumption

```bash
./scripts/run_full_benchmark.sh
```

The default protocol is 14 problems × 15 seeds (0–14), horizon 60,
population 40, eight variables, one generation per dynamic step.

Problems: `dzdt1_step`, `dzdt1_sine`, `dzdt2_shape`, `dzdt3_shocks`,
`fda1_lite`, `fda2_lite`, `dconstrained_zdt1`, `ddtlz2_3obj`, and
`df1_t10` through `df6_t10`.

Methods: `dnsga2_b`, `mnsga2`, `sgea`, `trnsga2`, `pfpa`, `tda_dmoea`,
`ab_dmoea`, `adr_eda`. ADR-DMOEA's separate consolidated evidence is included
in `docs/benchmark_results/`; it is not part of this launcher.

The host launcher enters Docker automatically. Each problem/algorithm pair is
a chunk. Successful chunks receive a completion marker after all seed rows
have status `ok`; interrupted or failed chunks are retried on the next run.
The launcher refuses to reuse a directory with a different protocol manifest.
Do not run multiple launchers against the same output directory concurrently.

```bash
./scripts/run_full_benchmark.sh results/full_benchmark
# Optional different protocol; always choose a fresh output directory:
SEEDS=2 HORIZON=10 ./scripts/run_full_benchmark.sh results/short_protocol
```

## Outputs and collection

- Default run: `results/benchmark/`, including `summary.json` and
  `per_run_metrics.csv`.
- Full run: `results/full_benchmark/chunks/<problem>/<algorithm>/`, with
  metrics, summary and `run.log` per chunk.
- Campaign progress: `master.log`, `status.tsv` and `protocol.txt`.
- Root `results/` is ignored by Git; archived evidence under `docs/` is tracked.

Collect chunks without installing scientific dependencies on the host:

```bash
docker compose -f docker/docker-compose.yml run --rm benchmark \
  python scripts/collect_benchmark_chunks.py results/full_benchmark
```

The collector writes `per_run_metrics.csv`, `chunk_completion.csv`,
`summary_compact.csv` and `COLLECTION_REPORT.md`. Partial runs may contribute
rows; means over incomplete/unbalanced campaigns are progress diagnostics,
not replacements for the final consolidated tables.

## Characterization and scope

[CHARACTERIZATION.md](docs/reproducibility/CHARACTERIZATION.md) records a
historical-environment comparison using identical source and runner on
`df1_t10`, seed 0, horizon 60, population 40, eight variables, one generation
per step and SPBN. All recorded outputs matched except wall-clock runtime.
Host load and architecture emulation affect timing; functional outputs and
source identity are the reproducibility targets.

The quality safeguard was enabled; no reactions were reverted. The recorded
front-local-search sample count was zero. This check does not establish
coverage of every response branch or every application experiment.

Historical [environment snapshots](docs/reproducibility/environment_snapshots/)
are provenance/debugging material, not additional installation requirements.
The Docker base tag and operating-system package repositories are not immutable
snapshots; a future rebuild can differ even with pinned Python dependencies.

## Application evidence

[Results and sources](docs/RESULTS.md) link the final benchmark, car-sharing,
RealCar-Munich and RealBike packages. Application scripts are indexed in
[scripts/README.md](scripts/README.md). Full application reruns require the
original demand inputs and suitable routing matrices; raw third-party datasets
are not distributed. The benchmark image has been characterized for the dynamic
benchmark, not certified for every historical application pipeline.

Historical filenames, internal labels and scientific outputs are preserved.
The routing fixture at
`tests/model/results/madrid_osrm_zone_matrix_congestion_2025_01.json` is required
input, not disposable generated output. See [THIRD_PARTY.md](THIRD_PARTY.md)
for unresolved rights and provenance.
