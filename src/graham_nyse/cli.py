from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer

from graham_nyse.backtest.data import load_table
from graham_nyse.backtest.engine import run_experiment_matrix, run_historical_backtest
from graham_nyse.config import WeightStrategy, load_config

app = typer.Typer(no_args_is_help=True)


@app.command("acquire-wrds")
def acquire_wrds_command(
    start: str = typer.Option(..., help="Research start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Research end date YYYY-MM-DD"),
    output: str = typer.Option("data/derived/wrds", help="Gitignored output path"),
    username: str | None = typer.Option(
        None, envvar="WRDS_USERNAME", help="WRDS username; password is never a CLI option"
    ),
) -> None:
    from graham_nyse.data.wrds import connect_wrds, extract_crsp

    connection = connect_wrds(username)
    paths = extract_crsp(connection, start, end, output)
    typer.echo(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


@app.command("acquire-sec-vintages")
def acquire_sec_vintages_command(
    identifier_links: str = typer.Option(..., help="WRDS CRSP/Compustat link history"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    output: str = typer.Option("data/derived/sec"),
    user_agent: str = typer.Option(..., envvar="SEC_USER_AGENT"),
    refresh: bool = typer.Option(False),
    maximum_issuers: int | None = typer.Option(
        None, help="Optional smoke-test limit; omit for the full universe"
    ),
) -> None:
    from graham_nyse.data.sec_vintages import acquire_sec_filing_vintages

    target = acquire_sec_filing_vintages(
        load_table(identifier_links), output, user_agent=user_agent,
        start=start, end=end, refresh=refresh, maximum_issuers=maximum_issuers,
    )
    typer.echo(str(target))


@app.command("run")
def run_command(
    config: str = typer.Option("config/strategy.yaml", help="Strategy YAML path"),
    output: str = typer.Option("outputs/latest", help="Output directory"),
    user_agent: str = typer.Option(
        ..., envvar="SEC_USER_AGENT", help="App name and contact email"
    ),
    classifications: str = typer.Option(
        ..., help="Ticker/domain/sector classification CSV"
    ),
    refresh: bool = typer.Option(False, help="Refresh cached SEC responses"),
    scenario: str = typer.Option("quality_value", help="Strategy scenario"),
    weighting_strategy: str = typer.Option(
        "equal", help="Portfolio weighting strategy"
    ),
) -> None:
    from graham_nyse.pipeline import run

    run(
        config,
        output,
        user_agent,
        classifications,
        refresh,
        scenario,
        weighting_strategy,
    )


@app.command("backtest")
def backtest_command(
    filing_vintages: str = typer.Option(
        ..., help="Immutable filing-vintage CSV or Parquet"
    ),
    security_master: str = typer.Option(
        ..., help="Dated security-master CSV or Parquet"
    ),
    prices: str = typer.Option(..., help="Long-form raw-price CSV or Parquet"),
    corporate_actions: str | None = typer.Option(
        None, help="Dividends and splits CSV or Parquet"
    ),
    start: str = typer.Option(..., help="Backtest start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Backtest end date YYYY-MM-DD"),
    config: str = typer.Option("config/strategy.yaml", help="Strategy YAML path"),
    output: str = typer.Option("outputs/backtest", help="Output directory"),
    scenario: str = typer.Option("quality_value", help="Strategy scenario"),
    weighting_strategy: str = typer.Option("equal", help="Weight strategy"),
    tax_mode: str = typer.Option("tax_deferred", help="Tax/account scenario"),
    tax_payment_source: str | None = typer.Option(None, help="portfolio or external"),
    benchmarks: str | None = typer.Option(
        None, help="Optional long-form benchmark total returns"
    ),
    factors: str | None = typer.Option(None, help="Optional daily factor returns"),
    quiet: bool = typer.Option(False, help="Suppress metric printing"),
) -> None:
    cfg = load_config(config)
    result = run_historical_backtest(
        load_table(filing_vintages),
        load_table(security_master),
        load_table(prices),
        cfg,
        start=start,
        end=end,
        scenario=scenario,
        weighting_strategy=cast(WeightStrategy, weighting_strategy),
        tax_mode=tax_mode,
        tax_payment_source=tax_payment_source,
        corporate_actions=load_table(corporate_actions) if corporate_actions else None,
        benchmark_returns=load_table(benchmarks) if benchmarks else None,
        factor_returns=load_table(factors) if factors else None,
    )
    result.write(output)
    Path(output, "historical_summary.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    if not quiet:
        typer.echo(json.dumps(result.metrics, indent=2))


@app.command("experiment-matrix")
def experiment_matrix_command(
    filing_vintages: str = typer.Option(...),
    security_master: str = typer.Option(...),
    prices: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    config: str = typer.Option("config/strategy.yaml"),
    output: str = typer.Option("outputs/experiment_matrix"),
    corporate_actions: str | None = typer.Option(None),
    tax_mode: str = typer.Option("tax_deferred"),
    tax_payment_source: str | None = typer.Option(None),
    benchmarks: str | None = typer.Option(None),
    factors: str | None = typer.Option(None),
) -> None:
    cfg = load_config(config)
    matrix, results = run_experiment_matrix(
        load_table(filing_vintages),
        load_table(security_master),
        load_table(prices),
        cfg,
        start,
        end,
        corporate_actions=load_table(corporate_actions) if corporate_actions else None,
        tax_mode=tax_mode,
        tax_payment_source=tax_payment_source,
        benchmark_returns=load_table(benchmarks) if benchmarks else None,
        factor_returns=load_table(factors) if factors else None,
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out / "scenario_weight_matrix.csv", index=False)
    for (scenario, strategy), result in results.items():
        result.write(out / scenario / strategy)
    typer.echo(f"Wrote {len(matrix)} scenario/weight combinations to {out}")


if __name__ == "__main__":
    app()
