#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pytest -q
python tests/fixtures/generate_simulation_smoke_test.py
PYTHONPATH=src python -m graham_nyse.cli backtest \
  --filing-vintages tests/fixtures/smoke_filing_vintages.csv \
  --security-master tests/fixtures/smoke_security_master.csv \
  --prices tests/fixtures/smoke_prices.csv \
  --corporate-actions tests/fixtures/smoke_corporate_actions.csv \
  --start 2016-07-01 \
  --end 2018-06-29 \
  --quiet \
  --output outputs/simulation_smoke_test

echo "Simulation smoke-test outputs written under outputs/simulation_smoke_test (gitignored)."
