# Real historical data acquisition

The empirical backtest uses licensed CRSP observations and public SEC filings.
It never reconstructs prior membership from a current ticker list.

## 1. Obtain UC San Diego WRDS access

Register at <https://wrds-www.wharton.upenn.edu/register/> with the UC San Diego
subscriber and institutional email. Dataset entitlement is controlled by the
university subscription. Confirm that the WRDS web interface exposes:

- CRSP Stock
- CRSP/Compustat Merged
- CRSP Indexes
- CRSP/Ziman Real Estate

Do not place a WRDS password in Git, shell history, a command-line option, or a
configuration file. The official `wrds` client handles its own authenticated
connection and optional user-level PostgreSQL password file.

## 2. Install acquisition dependencies

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,data]'
~~~

## 3. Extract CRSP and identifier history

~~~bash
export WRDS_USERNAME='your_wrds_username'

graham-nyse acquire-wrds \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output data/derived/wrds
~~~

The command requests a 400-day price warm-up before the research start. It
creates:

~~~text
security_master.parquet
raw_prices.parquet
corporate_actions.parquet
benchmark_total_returns.parquet
identifier_links.parquet
source_manifest.json
~~~

The security master contains dated CRSP name intervals. `PERMNO` is permanent;
ticker, exchange, share code, SIC, sector and accounting domain are attributes
of the dated interval. Only CRSP terminal events receive `is_delisted=true`.

The legacy CRSP table defaults are `dsf`, `dsenames`, `dsedelist`, `dsedist`,
and `dsi`. If the UCSD tenancy exposes only CRSP's CIZ schema, inspect the
available WRDS libraries and supply a corresponding `CrspTables` configuration
in Python. Do not silently substitute adjusted Yahoo prices.

## 4. Build SEC filing vintages

Use an identifiable SEC user agent:

~~~bash
export SEC_USER_AGENT='graham-nyse your_email@example.com'

graham-nyse acquire-sec-vintages \
  --identifier-links data/derived/wrds/identifier_links.parquet \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output data/derived/sec
~~~

The client follows the SEC submissions history files, not only the recent
submission block. Each normalized observation is keyed by CRSP security,
accession and `accepted_at`. Later amendments create later vintages and never
overwrite an earlier information state.

For a connectivity smoke test, add `--maximum-issuers 5`. Omit that option for
the empirical panel.

## 5. Run the matrix

~~~bash
graham-nyse experiment-matrix \
  --filing-vintages data/derived/sec/filing_vintages.parquet \
  --security-master data/derived/wrds/security_master.parquet \
  --prices data/derived/wrds/raw_prices.parquet \
  --corporate-actions data/derived/wrds/corporate_actions.parquet \
  --benchmarks data/derived/wrds/benchmark_total_returns.parquet \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output outputs/empirical_matrix
~~~

The four scenarios and six weighting methods produce 24 specifications. The
run must remain unpublished if a held delisting lacks its CRSP delisting return,
if a filing acceptance time is in the future, or if required domain fields are
missing.

## Licensed-data boundary

`data/raw/` and `data/derived/` are gitignored. Commit extraction code,
contracts, tests and manifests only. Do not commit CRSP, Compustat, Ziman or
NAIC observations to a public repository.
