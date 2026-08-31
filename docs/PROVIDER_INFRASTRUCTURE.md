# Provider-neutral research infrastructure

The engine uses canonical historical contracts rather than provider-specific
columns. A run declares exactly one market-data provider. It may combine that
market provider with SEC and domain-regulatory fundamentals, but it may not
splice prices or returns from CRSP, Sharadar and Norgate inside one empirical
specification.

## Install

~~~bash
pip install -e '.[dev,data,infrastructure]'
~~~

OpenBB is isolated because its dependency graph is large:

~~~bash
pip install -e '.[openbb]'
~~~

Check the environment with:

~~~bash
graham-nyse infrastructure-doctor
~~~

## Infrastructure components

| Component | Purpose |
|---|---|
| DuckDB | SQL views over partitioned Parquet without copying licensed files |
| DuckDB lazy scan | Portable lazy Parquet collection used by default |
| Polars | Optional native lazy scans when the installed wheel is CPU-compatible |
| Pandera | Runtime validation of canonical provider contracts |
| exchange-calendars | NYSE sessions and holiday handling |
| Tenacity | Bounded retry policy for transient source failures |
| Structlog | Machine-readable acquisition and run logs |
| CVXPY | Independent convex reference optimizer |
| Statsmodels | HAC-robust factor regression |
| Linearmodels | Panel and cross-sectional research extensions |

`HistoricalLake` registers every matching canonical Parquet file under a root
as a DuckDB view. `scan_prices()` uses a portable DuckDB lazy wrapper;
`scan_prices_polars()` is an explicit opt-in. This avoids making an incompatible
native Polars wheel a hard failure for the historical pipeline.

## Sharadar fallback

The adapter accepts licensed TICKERS, SEP and optional ACTIONS exports:

~~~bash
graham-nyse normalize-sharadar \
  --tickers private/sharadar_tickers.parquet \
  --prices private/sharadar_sep_with_permaticker.parquet \
  --actions private/sharadar_actions.parquet \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output data/derived/sharadar
~~~

SEP must be enriched with `PERMATICKER` before normalization. Ticker-only price
history is rejected because tickers can be renamed or reused. Sharadar does not
supply the CRSP delisting-return field used by the strict terminal-return path;
the normalized master therefore leaves it missing and the empirical run fails
closed if a held delisting cannot be valued from audited event data.

## Norgate fallback

Norgate uses a provider-stable `asset_id` supplied in an audited export:

~~~bash
graham-nyse normalize-norgate \
  --security-history private/norgate_security_history.parquet \
  --prices private/norgate_unadjusted_prices.parquet \
  --actions private/norgate_actions.parquet \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output data/derived/norgate
~~~

Prices must be explicitly unadjusted because dividends and splits are processed
as events by the backtest. The adapter does not reverse-engineer actions from an
adjusted-price series.

## OpenBB SEC reconciliation

Call OpenBB's SEC statements with `pit_mode=True`, then pass each result through
`normalize_openbb_pit_statement`. The adapter rejects rows without both an SEC
accession and an accepted timestamp. OpenBB is a reconciliation route; direct
EDGAR archives remain the canonical filing source.

## Research controls

- Provider observations remain under gitignored data directories.
- Every normalized extraction receives a source manifest and SHA-256 hashes.
- Provider-specific permanent identifiers are namespaced.
- Market data from different vendors cannot be combined within a run.
- Current ticker lists cannot initialize historical membership.
- Adjusted prices cannot be combined with separately processed distributions.
- Missing terminal returns or point-in-time metadata stop publication.

See [OPEN_DATA_RESEARCH.md](OPEN_DATA_RESEARCH.md) for the Alpha Vantage and
Yahoo research adapters and the dataset-level certification command.
