# Open-data research path and certification boundary

An open-source Python package is a client, not a guarantee about the historical
dataset behind it. The engine therefore grades the completed panel rather than
trusting a library name.

## Source findings

| Source/client | Useful role | Empirical status |
|---|---|---|
| Alpha Vantage `LISTING_STATUS` | Dated active and delisted symbol snapshots | Research only: symbol identity and no authoritative delisting return |
| `yfinance` / Yahoo | Raw close, volume, dividend and split observations for known symbols | Research only: no complete historical universe or guaranteed inactive coverage |
| SEC EDGAR | Immutable filings with accession and acceptance timestamps | Required public accounting source; not a market-return source |
| SimFin | Normalized fundamentals and price history | Research fallback until historical NYSE membership and terminal-event coverage are audited |
| Stooq | Historical price downloads | Research only: no permanent-ID security master or terminal-return contract |
| `pandas-datareader` | Macro/factor client | Its maintained API removed the Yahoo, Stooq and other securities readers |
| OpenBB | Provider aggregation and SEC reconciliation | Inherits the selected upstream provider's coverage |
| WRDS/CRSP | PERMNO/PERMCO, active/inactive history, actions and delisting returns | Strict empirical reference; licensed, not open data |

Run the machine-readable capability audit:

~~~bash
graham-nyse open-data-audit
~~~

## Best-effort open stack

The executable default order is role-specific and can be inspected with:

~~~bash
graham-nyse free-source-plan
~~~

It prioritizes SEC EDGAR for filing vintages and terminal-event evidence,
Alpha Vantage for dated active/delisted symbol snapshots, Nasdaq Trader for
current exchange/security-type reconciliation, Yahoo for raw research prices
and actions, and Stooq only as an independent price audit. Commercial sources
are excluded unless `--include-commercial-fallbacks` is passed.

Source priority never overrides certification. Yahoo and Stooq observations
cannot be silently combined or substituted inside one return series.

Archive the free current NYSE reference regularly:

~~~bash
graham-nyse acquire-nasdaq-reference \
  --output data/raw/nasdaq_trader/otherlisted.parquet
~~~

With an Alpha Vantage key, individual dated snapshots can be collected with:

~~~bash
export ALPHA_VANTAGE_API_KEY='...'

graham-nyse acquire-alpha-listings \
  --as-of 2016-06-30 \
  --as-of 2016-12-30 \
  --as-of 2017-06-30 \
  --output data/raw/alpha_vantage/listing_snapshots.parquet
~~~

For the ten-year run, prefer the bounded acquisition command. It captures
active membership before inception and at every June/December reconstruction,
then makes one final delisted-catalog request:

~~~bash
graham-nyse acquire-open-listings \
  --start 2016-07-01 --end 2026-06-30 \
  --output data/raw/alpha_vantage/listing_snapshots.parquet
~~~

The complete public/free bootstrap is:

~~~bash
./scripts/acquire_real_data.sh 2016-07-01 2026-06-30
~~~

The former WRDS-specific workflow is retained only as the explicit commercial
fallback at `scripts/acquire_wrds_data.sh`.

After deterministic domain/sector classification, use
`build_alpha_vantage_research_master`. It deliberately records
`identifier_quality=symbol_interval` and leaves `delisting_return` missing.

Yahoo can then be attempted for each known historical symbol:

~~~bash
graham-nyse acquire-yahoo-research \
  --security-master data/derived/open/security_master.parquet \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output data/derived/yahoo_research
~~~

The adapter requests `auto_adjust=False` and `actions=True`. Missing symbols are
a hard error; they are not dropped. Yahoo rate limits, personal-use terms and
unverified inactive-symbol retention make this a research route only.

Stooq is available strictly as a cross-provider audit. A disagreement is
reported and never patched into the Yahoo series:

~~~bash
graham-nyse audit-stooq-prices \
  --primary-prices data/derived/yahoo_research/raw_prices.parquet \
  --ticker IBM --start 2016-07-01 --end 2026-06-30
~~~

## Certification

Before any result is described as empirical:

~~~bash
graham-nyse certify-historical-data \
  --security-master data/derived/provider/security_master.parquet \
  --prices data/derived/provider/raw_prices.parquet \
  --corporate-actions data/derived/provider/corporate_actions.parquet \
  --filing-vintages data/derived/sec/filing_vintages.parquet \
  --require-empirical
~~~

The command exits with status 2 unless all controls pass:

- one market-data provider per specification;
- dated membership intervals;
- provider-stable permanent security identifiers;
- inactive and delisted securities;
- complete delisting returns;
- raw/unadjusted prices plus explicit corporate actions;
- filing accession and acceptance timestamps.

Backtest manifests always contain `publication_status` and the full
`data_certification` report. A Yahoo/Alpha Vantage panel must remain
`research_only`; generated fixtures remain software validation rather than
investment evidence.

## Historical-result corrections

Taxable runs may not pay tax by creating negative cash. The engine estimates
the current-year liability, makes explicit `tax_funding` sales, processes the
new gains, re-estimates, and then settles. If the portfolio still cannot pay,
the run fails.

Small-trade suppression is also bypassed when it would leave a realized
position or sector above its hard cap. Negative cash and realized cap breaches
are validation errors, not report warnings.
