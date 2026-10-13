#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: ./scripts/run_full_benchmark.sh [results/OUTPUT_DIRECTORY]

Run/resume 14 problems and 8 algorithms through Docker.
Environment: SEEDS=15 HORIZON=60 POP_SIZE=40 N_VARIABLES=8
One generation per dynamic step. ADR-DMOEA is not included.
Default output: results/full_benchmark
Use a fresh directory when changing the protocol; concurrent runs are unsupported.
USAGE
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo "Expected at most one output directory. Use --help." >&2
  exit 2
fi
OUT="${1:-results/full_benchmark}"
if [[ "$OUT" != results/* || "$OUT" == *".."* || "$OUT" == *$'\n'* ]]; then
  echo "Output must be a relative directory below results/, without '..'." >&2
  exit 2
fi
cd "$ROOT_DIR" || exit 1
if [[ "${ADR_EDA_CONTAINER:-}" != 1 ]]; then
  exec docker compose -f "$ROOT_DIR/docker/docker-compose.yml" run --rm \
    -e ADR_EDA_CONTAINER=1 \
    -e "SEEDS=${SEEDS:-15}" -e "HORIZON=${HORIZON:-60}" \
    -e "POP_SIZE=${POP_SIZE:-40}" -e "N_VARIABLES=${N_VARIABLES:-8}" \
    benchmark bash scripts/run_full_benchmark.sh "$OUT"
fi
SEEDS="${SEEDS:-15}"
HORIZON="${HORIZON:-60}"
POP_SIZE="${POP_SIZE:-40}"
N_VARIABLES="${N_VARIABLES:-8}"

PROBLEMS=(
  dzdt1_step
  dzdt1_sine
  dzdt2_shape
  dzdt3_shocks
  fda1_lite
  fda2_lite
  dconstrained_zdt1
  ddtlz2_3obj
  df1_t10
  df2_t10
  df3_t10
  df4_t10
  df5_t10
  df6_t10
)

ALGORITHMS=(
  dnsga2_b
  mnsga2
  sgea
  trnsga2
  pfpa
  tda_dmoea
  ab_dmoea
  adr_eda
)

for value in "$SEEDS" "$HORIZON" "$POP_SIZE" "$N_VARIABLES"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Protocol values must be positive integers." >&2
    exit 2
  fi
done
mkdir -p "$OUT/chunks" || exit 1
PROTOCOL="seeds=$SEEDS horizon=$HORIZON population=$POP_SIZE variables=$N_VARIABLES generations=1"
PROTOCOL+=" problems=${PROBLEMS[*]} algorithms=${ALGORITHMS[*]}"
if [[ -f "$OUT/protocol.txt" ]]; then
  if [[ "$(cat "$OUT/protocol.txt")" != "$PROTOCOL" ]]; then
    echo "Protocol mismatch: choose a fresh output directory." >&2
    exit 2
  fi
else
  printf '%s\n' "$PROTOCOL" > "$OUT/protocol.txt" || exit 1
fi
FAILED=0

MASTER_LOG="$OUT/master.log"
STATUS_TSV="$OUT/status.tsv"

if [ ! -f "$STATUS_TSV" ]; then
  printf "timestamp\tproblem\talgorithm\texit_code\telapsed_seconds\tchunk_dir\n" > "$STATUS_TSV"
fi

{
  echo "ADR-EDA chunked dynamic benchmark"
  echo "Started: $(date -Iseconds)"
  echo "Outdir: $OUT"
  echo "Seeds: $SEEDS | Horizon: $HORIZON | Pop: $POP_SIZE | Variables: $N_VARIABLES"
  echo "Problems: ${PROBLEMS[*]}"
  echo "Algorithms: ${ALGORITHMS[*]}"
  echo
} >> "$MASTER_LOG"

for problem in "${PROBLEMS[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    CHUNK_DIR="$OUT/chunks/$problem/$algorithm"
    mkdir -p "$CHUNK_DIR"

    if [ -f "$CHUNK_DIR/.complete" ] && [ -s "$CHUNK_DIR/summary.json" ] && [ -s "$CHUNK_DIR/per_run_metrics.csv" ]; then
      echo "[skip] $(date -Iseconds) problem=$problem algorithm=$algorithm existing_outputs=yes" >> "$MASTER_LOG"
      continue
    fi

    start_ts="$(date +%s)"
    echo "[start] $(date -Iseconds) problem=$problem algorithm=$algorithm" >> "$MASTER_LOG"

    python3 -u tests/edas/clean_dynamic_sota_comparison.py \
      --algorithms "$algorithm" \
      --problems "$problem" \
      --seeds "$SEEDS" \
      --horizon "$HORIZON" \
      --size-gen "$POP_SIZE" \
      --n-variables "$N_VARIABLES" \
      --generations-per-step 1 \
      --outdir "$CHUNK_DIR" \
      > "$CHUNK_DIR/run.log" 2>&1

    exit_code="$?"
    if [[ "$exit_code" -eq 0 ]]; then
      python3 - "$CHUNK_DIR" "$SEEDS" <<'CHECK'
import csv
import json
import sys
from pathlib import Path
chunk = Path(sys.argv[1])
json.loads((chunk / 'summary.json').read_text())
with (chunk / 'per_run_metrics.csv').open() as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != int(sys.argv[2]) or any(row.get('status') != 'ok' for row in rows):
    raise SystemExit('Chunk has missing or failed seed rows')
(chunk / '.complete').write_text('ok\n')
CHECK
      exit_code="$?"
    fi
    if [[ "$exit_code" -ne 0 ]]; then FAILED=1; fi
    end_ts="$(date +%s)"
    elapsed="$((end_ts - start_ts))"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date -Iseconds)" "$problem" "$algorithm" "$exit_code" "$elapsed" "$CHUNK_DIR" >> "$STATUS_TSV"

    echo "[end] $(date -Iseconds) problem=$problem algorithm=$algorithm exit=$exit_code elapsed=${elapsed}s" >> "$MASTER_LOG"
  done
done

echo "Finished: $(date -Iseconds)" >> "$MASTER_LOG"

exit "$FAILED"
