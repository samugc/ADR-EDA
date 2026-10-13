#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "Building the ADR-EDA benchmark environment..."
docker compose -f docker/docker-compose.yml build benchmark

echo
echo "Checking the environment..."
docker compose -f docker/docker-compose.yml run --rm benchmark \
  python -m pip check

echo
echo "ADR-EDA benchmark environment is ready."
