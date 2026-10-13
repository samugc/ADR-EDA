#!/usr/bin/env sh
set -eu

export PYTHONPATH=/app/src

ROOT="/app/docs/realcar_q1"
RESULTS="$ROOT/results"
NIGHT_DIR="$RESULTS/realcar_munich_night_2026_05_28"
LOG_DIR="/app/logs"
PROGRESS="$NIGHT_DIR/night_progress.csv"

mkdir -p "$NIGHT_DIR" "$LOG_DIR"

if [ ! -f "$PROGRESS" ]; then
  printf "time,campaign,status,out_dir,log,message\n" > "$PROGRESS"
fi

stamp() {
  date '+%Y-%m-%dT%H:%M:%S'
}

log_row() {
  printf "%s,%s,%s,%s,%s,%s\n" "$(stamp)" "$1" "$2" "$3" "$4" "$5" >> "$PROGRESS"
}

run_campaign() {
  campaign="$1"
  profile_dir="$2"
  travel_json="$3"
  out_dir="$4"
  seeds="$5"
  hours="$6"
  fleet="$7"
  chargers="$8"
  ops="$9"

  log="$LOG_DIR/${campaign}.log"

  if [ -f "$out_dir/summary.csv" ]; then
    log_row "$campaign" "SKIPPED" "$out_dir" "$log" "summary.csv already exists"
    return 0
  fi

  log_row "$campaign" "STARTED" "$out_dir" "$log" "seeds=$seeds hours=$hours fleet=$fleet chargers=$chargers ops=$ops"

  if python3 /app/scripts/run_realcar_freefloating_mini_campaign.py \
    --profile-dir "$profile_dir" \
    --travel-metrics-json "$travel_json" \
    --out-dir "$out_dir" \
    --seeds "$seeds" \
    --hours "$hours" \
    --fleet-size "$fleet" \
    --charger-count "$chargers" \
    --ops-resources "$ops" \
    --size-gen 8 \
    --max-iter 1 > "$log" 2>&1; then
    log_row "$campaign" "OK" "$out_dir" "$log" "completed"
  else
    rc="$?"
    log_row "$campaign" "FAILED" "$out_dir" "$log" "returncode=$rc"
  fi
}

TOP10="$ROOT/sharenow_munich_202306_profile_top10"
TOP20="$ROOT/sharenow_munich_202306_profile_top20"
OSRM_TOP20="$TOP20/zone_osrm_matrix_driving.json"

run_campaign \
  "realcar_top10_h8_s5_2026_05_28" \
  "$TOP10" \
  "$OSRM_TOP20" \
  "$RESULTS/freefloating_real_munich_top10_mini_s5_h8_2026_05_28" \
  "2026,2027,2028,2029,2030" \
  "8" \
  "70" \
  "12" \
  "14"

run_campaign \
  "realcar_top10_h12_s3_2026_05_28" \
  "$TOP10" \
  "$OSRM_TOP20" \
  "$RESULTS/freefloating_real_munich_top10_mini_s3_h12_2026_05_28" \
  "2026,2027,2028" \
  "12" \
  "70" \
  "12" \
  "14"

run_campaign \
  "realcar_top20_h6_s3_2026_05_28" \
  "$TOP20" \
  "$OSRM_TOP20" \
  "$RESULTS/freefloating_real_munich_top20_mini_s3_h6_2026_05_28" \
  "2026,2027,2028" \
  "6" \
  "120" \
  "20" \
  "24"

log_row "realcar_night" "DONE" "$NIGHT_DIR" "$PROGRESS" "night supervisor finished"
