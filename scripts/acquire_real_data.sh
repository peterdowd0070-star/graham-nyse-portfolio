#!/usr/bin/env bash
set -euo pipefail

: "${ALPHA_VANTAGE_API_KEY:?Set ALPHA_VANTAGE_API_KEY to a free Alpha Vantage key}"
: "${SEC_USER_AGENT:?Set SEC_USER_AGENT to an app name and contact email}"

START_DATE="${1:-2016-07-01}"
END_DATE="${2:-2026-06-30}"

graham-nyse free-source-plan

graham-nyse acquire-sec-reference \
  --output data/raw/sec/company_tickers_exchange.parquet

graham-nyse acquire-nasdaq-reference \
  --output data/raw/nasdaq_trader/otherlisted.parquet

graham-nyse acquire-open-listings \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --output data/raw/alpha_vantage/listing_snapshots.parquet

echo "Public/free source snapshots acquired. Build and certify the canonical"
echo "security master before downloading prices or publishing a backtest."
