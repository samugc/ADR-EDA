#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.yml"

ALGORITHMS="adr_eda"
PROBLEMS="df1_t10"
SEEDS="1"
HORIZON="60"
SIZE_GEN="40"
N_VARIABLES="8"
GENERATIONS_PER_STEP="1"
OUTDIR="results/benchmark"

usage() {
  cat <<USAGE
Usage: ./scripts/run_benchmark.sh [options]

Options:
  --algorithms NAME       Algorithm list passed to the benchmark runner
                          (default: adr_eda)
  --problems NAME         Problem list (default: df1_t10)
  --seeds N               Number of seeds (default: 1)
  --horizon N             Dynamic horizon (default: 60)
  --size-gen N            Population size (default: 40)
  --n-variables N         Decision variables (default: 8)
  --generations N         Generations per dynamic step (default: 1)
  --outdir PATH           Output directory (default: results/benchmark)
  -h, --help              Show this message

Example:
  ./scripts/run_benchmark.sh \
    --problems df1_t10 \
    --seeds 1 \
    --outdir results/df1_smoke
USAGE
}

while [[ $# -gt 0 ]]; do
  if [[ "$1" != "-h" && "$1" != "--help" && $# -lt 2 ]]; then
    echo "Missing value for $1. Use --help." >&2
    exit 2
  fi
  case "$1" in
    --algorithms)
      ALGORITHMS="$2"; shift 2 ;;
    --problems)
      PROBLEMS="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    --horizon)
      HORIZON="$2"; shift 2 ;;
    --size-gen)
      SIZE_GEN="$2"; shift 2 ;;
    --n-variables)
      N_VARIABLES="$2"; shift 2 ;;
    --generations)
      GENERATIONS_PER_STEP="$2"; shift 2 ;;
    --outdir)
      OUTDIR="$2"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

cd "$ROOT_DIR"

docker compose -f "$COMPOSE_FILE" run --rm benchmark \
  python tests/edas/clean_dynamic_sota_comparison.py \
    --algorithms "$ALGORITHMS" \
    --problems "$PROBLEMS" \
    --seeds "$SEEDS" \
    --horizon "$HORIZON" \
    --size-gen "$SIZE_GEN" \
    --n-variables "$N_VARIABLES" \
    --generations-per-step "$GENERATIONS_PER_STEP" \
    --outdir "$OUTDIR"
