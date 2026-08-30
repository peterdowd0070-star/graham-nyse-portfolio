# Backtest Validation Record

## Scope

This record documents a local software-validation run of the deterministic backtest engine. It does **not** report an investable historical strategy result.

- Validation period: July 2016 through June 2026
- Starting capital: approximately $5,000
- Transaction-cost assumption: 10 basis points per one-way trade
- Initial construction: first trading day
- Weight-only rebalances: March and September
- Full reconstructions: June and December
- Fractional shares: enabled

## Data classification

The validation dataset is generated and contains no real issuer fundamentals, historical NYSE membership, or real security returns. Security-level holdings, trades, and daily NAV files are intentionally excluded from version control. This avoids presenting synthetic constituents as recommendations and avoids publishing implementation-level fixture output as investment evidence.

## Aggregate validation result

The local run completed without an exception and produced:

| Check | Rounded result |
|---|---:|
| Ending value | approximately $15.5k |
| Compound annual growth | approximately 12.0% |
| Annualized volatility | approximately 3.2% |
| Maximum drawdown | approximately -2.4% |
| Annualized one-way turnover | approximately 16.9% |

These values are properties of the generated fixture. They must not be used to estimate expected returns, risk, or alpha.

## Tests

The local test suite completed with three passing tests. The tests cover portfolio caps and normalization, fractional-share allocation, initial construction, quarterly rebalancing, semiannual reconstruction, and backtest output generation.

## Requirements for an empirical backtest

A publishable historical analysis requires all of the following:

1. Survivorship-bias-free NYSE membership and security identifiers.
2. Delisted-security prices and delisting returns.
3. Corporate-action-adjusted total-return series.
4. Point-in-time fundamentals keyed to public filing dates.
5. Historical security classifications for common shares, ADRs, REITs, banks, insurers, funds, preferreds, warrants, and acquisition vehicles.
6. Reproducible market-capitalization and liquidity observations at every decision date.
7. Benchmark data, transaction costs, and execution assumptions fixed before evaluation.

Until those inputs are available, the backtest is a verification of software mechanics only.
