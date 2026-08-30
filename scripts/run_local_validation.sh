#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pytest -q
python tests/fixtures/generate_demo_backtest.py
PYTHONPATH=src python -m graham_nyse.cli backtest \
  --features tests/fixtures/demo_features_10y.csv \
  --prices tests/fixtures/demo_prices_10y.csv \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --transaction-cost-bps 10 \
  --output outputs/local_validation_10y

echo "Validation outputs written under outputs/local_validation_10y (gitignored)."
