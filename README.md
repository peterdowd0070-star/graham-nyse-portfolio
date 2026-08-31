# Graham NYSE Historical Portfolio Engine

A deterministic Python research engine for Graham-style portfolio construction over the historical NYSE common-equity universe.

## Control boundary

Python owns data timing, security eligibility, accounting normalization, valuation, scoring, constituent selection, weights, trades, taxes, validation, benchmarks, and factor attribution. An LLM may explain a validated report payload; it cannot add or modify a security, weight, trade, number, or calculation.

## Historical model

The backtest does not apply today's portfolio to prior returns. It reconstructs the information set through time:

1. Determine the common stocks actively listed on the NYSE at the historical decision date, including issuers that later became inactive or delisted.
2. Expose only filings whose SEC acceptance timestamp is at or before the decision cutoff.
3. Merge raw prices and liquidity observations available by that date.
4. Apply the accounting model for the company's domain and sector.
5. Recalculate the selected strategy scenario and portfolio weights.
6. Monitor every month and reconstruct the investable portfolio from the full dated NYSE universe in March, June, September, and December.
7. Permit securities to enter, remain, or exit as listing status, filing vintages, eligibility, ranks, and portfolio constraints change.
8. Process dividends, splits, delistings, transaction costs, tax lots, and annual tax settlements explicitly.

No ticker list is hard-coded into portfolio construction. Holdings are outputs of the historical universe, information set, scenario, and weighting model at each reconstruction date.

The old as_of_date feature-panel and adjusted-price interfaces have been removed because they could not prove filing availability, historical membership, or corporate-action treatment.

## Strategy scenarios

- defensive: strict history and quality requirements.
- enterprising: broader eligibility with greater valuation emphasis.
- deep_value: strongest relative-value emphasis with a lower quality floor.
- quality_value: balanced relative value and quality.

Every scenario uses domain-relative and, where configured, sector-relative gates and factors. Ordinary companies, banks, insurers, and REITs have separate accounting models.

## Weighting strategies

Every scenario is tested with equal, score-proportional, inverse-volatility, score/volatility, minimum-variance, and liquidity-adjusted equal weighting. All methods use the same position and sector constraints.

## Tax modes

- tax_deferred
- taxable_fifo_no_liquidation
- taxable_hifo_no_liquidation
- taxable_hifo_terminal_liquidation

The taxable modes maintain lots, holding periods, qualified and ordinary dividends, short- and long-term netting, loss carryforwards, and wash-sale basis adjustments. Tax rates are configurable assumptions, not individualized tax advice.
Tax payment source is independently configurable as portfolio cash or external cash; external payments are deducted in the reported after-tax total-wealth metric.
Portfolio-funded taxes are raised through explicit recorded sales. Negative
cash is not treated as free borrowing.

## Install

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
~~~

## Run one historical specification

~~~bash
graham-nyse backtest \
  --filing-vintages data/filing_vintages.parquet \
  --security-master data/security_master.parquet \
  --prices data/raw_prices.parquet \
  --corporate-actions data/corporate_actions.parquet \
  --benchmarks data/benchmark_total_returns.parquet \
  --factors data/factor_returns.parquet \
  --scenario quality_value \
  --weighting-strategy equal \
  --tax-mode tax_deferred \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output outputs/historical_quality_value_equal
~~~

## Run the 24-cell research matrix

~~~bash
graham-nyse experiment-matrix \
  --filing-vintages data/filing_vintages.parquet \
  --security-master data/security_master.parquet \
  --prices data/raw_prices.parquet \
  --corporate-actions data/corporate_actions.parquet \
  --start 2016-07-01 \
  --end 2026-06-30 \
  --output outputs/scenario_weight_matrix
~~~

## Run the self-contained evolving-universe matrix

This command generates a deterministic ten-year point-in-time universe with new listings, delistings, delisting returns, dated filings, dividends, and changing ranks. It then runs every scenario and weighting strategy through the production historical engine.

~~~bash
PYTHONPATH=src:. python scripts/run_evolving_validation.py \
  --tax-mode tax_deferred \
  --output outputs/evolving_validation_10y
~~~

The command writes:

- `scenario_weight_matrix.csv` for all 24 variants;
- one complete result directory per scenario and weighting strategy;
- generated input tables under `data/`;
- `validation_manifest.json` with the evidence classification.

This mode validates evolving portfolio behavior without requiring external licensed data. Its prices and fundamentals are generated and must not be presented as historical NYSE performance.

## Validation

~~~bash
./scripts/run_local_validation.sh
~~~

The generated fixture is named a **simulation smoke test**. It verifies event timing, constraints, tax lots, delisting treatment, and output generation. It does not publish CAGR, volatility, drawdown, alpha, or benchmark comparisons because generated returns are not investment evidence.

See [DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md) and [SIMULATION_SMOKE_TEST.md](docs/SIMULATION_SMOKE_TEST.md).

For the authenticated CRSP/WRDS and public SEC pipeline, see
[REAL_DATA_ACQUISITION.md](docs/REAL_DATA_ACQUISITION.md). Licensed observations
are never committed to this repository.

Alternative provider adapters and the DuckDB/Polars/Pandera research layer are
documented in [PROVIDER_INFRASTRUCTURE.md](docs/PROVIDER_INFRASTRUCTURE.md).

The audited Alpha Vantage/Yahoo/SEC research path, source comparison and hard
empirical certification gate are documented in
[OPEN_DATA_RESEARCH.md](docs/OPEN_DATA_RESEARCH.md).

## Evidence status

The engine can produce an empirical result only when supplied with a survivorship-free historical security master, delisting returns, raw security returns and corporate actions, immutable SEC filing vintages, historical classifications, benchmarks, and factor returns. The included fixtures validate software only.
Every run manifest records `publication_status`; open-data fallbacks are not
silently promoted to empirical evidence.
