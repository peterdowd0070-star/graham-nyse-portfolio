# Graham NYSE Portfolio Engine

A deterministic Python implementation of a modernized Benjamin Graham portfolio process over NYSE-listed operating companies.

## Boundary

Python owns all data processing, eligibility rules, valuation signals, scores, weights, shares, trades, and validations. An LLM may consume `report_context.json` only to produce narrative commentary. It must not change the portfolio.

## Components

1. `data/sec.py` — obtains the SEC exchange/ticker universe and Company Facts.
2. `data/market.py` — obtains prices, market capitalization, and liquidity observations.
3. `data/fundamentals.py` — maps XBRL concepts into normalized annual fundamentals.
4. `portfolio/scoring.py` — applies hard safety gates and computes value/quality scores.
5. `portfolio/construction.py` — selects securities, caps weights, supports fractional shares, and calculates rebalance orders.
6. `validation.py` — fails closed when portfolio invariants are violated.
7. `reporting/payload.py` — emits a schema-limited JSON payload for narrative generation.
8. `pipeline.py` — orchestrates monthly monitoring, quarterly rebalancing, and semiannual reconstruction modes.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run

The SEC requires a descriptive user agent with contact information.

```bash
export SEC_USER_AGENT="graham-portfolio your_email@example.com"
graham-nyse run --config config/strategy.yaml --output outputs/latest
```

## Outputs

- `universe.csv`
- `fundamental_dataset.parquet`
- `scored_universe.csv`
- `target_portfolio.csv`
- `report_context.json`
- `run_audit.json`

## Known limitations before live use

- Yahoo Finance is used only as a prototype market-data adapter and should be replaced or snapshotted for production reproducibility.
- XBRL concept mapping is conservative but not complete; banks, insurers, REITs, foreign private issuers, and industry-specific accounting require separate models.
- The current name-based security-type filter should be supplemented with a licensed or exchange-native security master.
- Intrinsic-value Monte Carlo and sector-specific models are not yet implemented in version 0.1.0.
- This code must be backtested with point-in-time membership, delisting returns, filing lags, and transaction costs before capital is allocated.

## Ten-year point-in-time backtest

The backtest consumes two explicit datasets:

1. A point-in-time feature panel with one row per `as_of_date` and `ticker`.
2. A long-form adjusted-price panel with `date`, `ticker`, and `adjusted_close`.

Run:

```bash
graham-nyse backtest \
  --features data/features_point_in_time.parquet \
  --prices data/prices_adjusted.parquet \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --transaction-cost-bps 10 \
  --output outputs/backtest_10y
```

The engine performs initial construction on the first trading day, weight-only rebalances in March and September, and full reconstructions in June and December. It produces daily NAV, historical holdings, trades, turnover, transaction costs, CAGR, volatility, Sharpe ratio, and maximum drawdown.

A credible historical result requires a survivorship-bias-free security master, delisted-security returns, and fundamentals keyed to the date they became public. The current NYSE listing file and Yahoo adapter are suitable for live prototyping, not for claiming an unbiased historical return.

## Local validation

Run the deterministic tests and generated 10-year fixture with:

```bash
./scripts/run_local_validation.sh
```

Only a rounded, aggregate validation record is retained in [`docs/BACKTEST_VALIDATION.md`](docs/BACKTEST_VALIDATION.md). Raw fixture holdings, trades, prices, features, and NAV outputs are gitignored. The recorded figures validate software mechanics only and are not historical performance claims.
