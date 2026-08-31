#!/usr/bin/env bash
set -euo pipefail

: "${WRDS_USERNAME:?Set WRDS_USERNAME to your UCSD WRDS username}"
: "${SEC_USER_AGENT:?Set SEC_USER_AGENT to an app name and contact email}"

START_DATE="${1:-2016-07-01}"
END_DATE="${2:-2026-06-30}"

graham-nyse acquire-wrds \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --output data/derived/wrds

graham-nyse acquire-sec-vintages \
  --identifier-links data/derived/wrds/identifier_links.parquet \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --output data/derived/sec
