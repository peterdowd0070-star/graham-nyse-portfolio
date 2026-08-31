from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import typer

from graham_nyse.backtest.data import load_table
from graham_nyse.backtest.engine import run_experiment_matrix, run_historical_backtest
from graham_nyse.config import WeightStrategy, load_config

app = typer.Typer(no_args_is_help=True)


@app.command("open-data-audit")
def open_data_audit_command() -> None:
    import importlib.util

    from graham_nyse.data.open_data import OPEN_DATA_SOURCE_AUDIT

    libraries = {
        name: "available" if importlib.util.find_spec(name) else "not installed"
        for name in ("yfinance", "simfin", "openbb", "pandas_datareader")
    }
    typer.echo(
        json.dumps(
            {"libraries": libraries, "sources": OPEN_DATA_SOURCE_AUDIT}, indent=2
        )
    )


@app.command("free-source-plan")
def free_source_plan_command(
    include_commercial_fallbacks: bool = typer.Option(
        False, help="Show paid fallbacks after the free/public sources"
    ),
) -> None:
    from graham_nyse.data.source_priority import source_plan_payload

    typer.echo(
        json.dumps(
            source_plan_payload(no_commercial=not include_commercial_fallbacks),
            indent=2,
        )
    )


@app.command("acquire-nasdaq-reference")
def acquire_nasdaq_reference_command(
    output: str = typer.Option("data/raw/nasdaq_trader/otherlisted.parquet"),
    refresh: bool = typer.Option(False),
) -> None:
    from graham_nyse.data.nasdaq_trader import NasdaqTraderClient

    frame = NasdaqTraderClient().fetch_other_listed(refresh=refresh)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    typer.echo(str(target))


@app.command("audit-stooq-prices")
def audit_stooq_prices_command(
    primary_prices: str = typer.Option(...),
    ticker: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    relative_tolerance: float = typer.Option(0.02),
    refresh: bool = typer.Option(False),
) -> None:
    from graham_nyse.data.stooq import StooqPriceAuditClient, audit_close_prices

    primary = load_table(primary_prices)
    audit = StooqPriceAuditClient().fetch_daily(ticker, start, end, refresh=refresh)
    typer.echo(
        json.dumps(
            audit_close_prices(primary, audit, relative_tolerance=relative_tolerance),
            indent=2,
        )
    )


@app.command("acquire-alpha-listings")
def acquire_alpha_listings_command(
    as_of: list[str] = typer.Option(  # noqa: B008
        ..., "--as-of", help="Repeatable YYYY-MM-DD"
    ),
    output: str = typer.Option("data/raw/alpha_vantage/listing_snapshots.parquet"),
    api_key: str | None = typer.Option(None, envvar="ALPHA_VANTAGE_API_KEY"),
    refresh: bool = typer.Option(False),
) -> None:
    from graham_nyse.data.open_data import AlphaVantageListingClient

    client = AlphaVantageListingClient(api_key=api_key)
    frames = [
        client.fetch_snapshot(day, state, refresh)
        for day in as_of
        for state in ("active", "delisted")
    ]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd_frame = pd.concat(frames, ignore_index=True)
    pd_frame.to_parquet(target, index=False)
    typer.echo(str(target))


@app.command("acquire-open-listings")
def acquire_open_listings_command(
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    output: str = typer.Option("data/raw/alpha_vantage/listing_snapshots.parquet"),
    api_key: str | None = typer.Option(None, envvar="ALPHA_VANTAGE_API_KEY"),
    refresh: bool = typer.Option(False),
) -> None:
    from graham_nyse.data.open_data import (
        AlphaVantageListingClient,
        open_listing_snapshot_requests,
    )

    client = AlphaVantageListingClient(api_key=api_key)
    requests = open_listing_snapshot_requests(start, end)
    frames = [client.fetch_snapshot(day, state, refresh) for day, state in requests]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
    typer.echo(json.dumps({"path": str(target), "requests": len(requests)}, indent=2))


@app.command("acquire-sec-reference")
def acquire_sec_reference_command(
    output: str = typer.Option("data/raw/sec/company_tickers_exchange.parquet"),
    user_agent: str = typer.Option(..., envvar="SEC_USER_AGENT"),
    refresh: bool = typer.Option(False),
) -> None:
    from graham_nyse.data.sec import SecClient

    frame = SecClient(user_agent, cache_dir=Path(output).parent).exchange_tickers(
        refresh=refresh
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    typer.echo(str(target))


@app.command("acquire-yahoo-research")
def acquire_yahoo_research_command(
    security_master: str = typer.Option(..., help="Research security master"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    output: str = typer.Option("data/derived/yahoo_research"),
) -> None:
    from graham_nyse.data.certification import certify_historical_data
    from graham_nyse.data.open_data import YahooResearchProvider

    bundle = YahooResearchProvider(load_table(security_master)).load(start, end)
    paths = bundle.write(output)
    report = certify_historical_data(
        bundle.security_master, bundle.prices, bundle.corporate_actions
    )
    target = Path(output, "data_certification.json")
    target.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "paths": {key: str(value) for key, value in paths.items()},
                "certification": report.as_dict(),
            },
            indent=2,
        )
    )


@app.command("certify-historical-data")
def certify_historical_data_command(
    security_master: str = typer.Option(...),
    prices: str = typer.Option(...),
    corporate_actions: str = typer.Option(...),
    filing_vintages: str | None = typer.Option(None),
    require_empirical: bool = typer.Option(
        False, help="Exit non-zero unless every empirical control passes"
    ),
) -> None:
    from graham_nyse.data.certification import certify_historical_data

    report = certify_historical_data(
        load_table(security_master),
        load_table(prices),
        load_table(corporate_actions),
        load_table(filing_vintages) if filing_vintages else None,
    )
    typer.echo(json.dumps(report.as_dict(), indent=2))
    if require_empirical and not report.empirical_results_allowed:
        raise typer.Exit(code=2)


@app.command("infrastructure-doctor")
def infrastructure_doctor_command() -> None:
    import importlib.util

    modules = [
        "cvxpy",
        "duckdb",
        "exchange_calendars",
        "linearmodels",
        "pandera",
        "polars",
        "statsmodels",
        "structlog",
        "tenacity",
        "openbb",
        "yfinance",
        "simfin",
        "pandas_datareader",
    ]
    status = {
        module: "available" if importlib.util.find_spec(module) else "not installed"
        for module in modules
    }
    typer.echo(json.dumps(status, indent=2))


@app.command("normalize-sharadar")
def normalize_sharadar_command(
    tickers: str = typer.Option(..., help="Sharadar TICKERS CSV/Parquet"),
    prices: str = typer.Option(..., help="SEP export enriched with PERMATICKER"),
    actions: str | None = typer.Option(None, help="Optional ACTIONS export"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    output: str = typer.Option("data/derived/sharadar"),
) -> None:
    from graham_nyse.data.contracts import validate_canonical_bundle
    from graham_nyse.data.provenance import write_source_manifest
    from graham_nyse.data.sharadar import SharadarExportProvider

    bundle = SharadarExportProvider(tickers, prices, actions).load(start, end)
    validate_canonical_bundle(
        bundle.security_master, bundle.prices, bundle.corporate_actions
    )
    paths = bundle.write(output)
    write_source_manifest(
        output,
        source=bundle.provider,
        parameters={"start": start, "end": end},
        files=list(paths.values()),
    )
    typer.echo(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


@app.command("normalize-norgate")
def normalize_norgate_command(
    security_history: str = typer.Option(...),
    prices: str = typer.Option(...),
    actions: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    output: str = typer.Option("data/derived/norgate"),
) -> None:
    from graham_nyse.data.contracts import validate_canonical_bundle
    from graham_nyse.data.norgate import NorgateExportProvider
    from graham_nyse.data.provenance import write_source_manifest

    bundle = NorgateExportProvider(security_history, prices, actions).load(start, end)
    validate_canonical_bundle(
        bundle.security_master, bundle.prices, bundle.corporate_actions
    )
    paths = bundle.write(output)
    write_source_manifest(
        output,
        source=bundle.provider,
        parameters={"start": start, "end": end},
        files=list(paths.values()),
    )
    typer.echo(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


@app.command("acquire-wrds")
def acquire_wrds_command(
    start: str = typer.Option(..., help="Research start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Research end date YYYY-MM-DD"),
    output: str = typer.Option("data/derived/wrds", help="Gitignored output path"),
    username: str | None = typer.Option(
        None,
        envvar="WRDS_USERNAME",
        help="WRDS username; password is never a CLI option",
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
        load_table(identifier_links),
        output,
        user_agent=user_agent,
        start=start,
        end=end,
        refresh=refresh,
        maximum_issuers=maximum_issuers,
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
