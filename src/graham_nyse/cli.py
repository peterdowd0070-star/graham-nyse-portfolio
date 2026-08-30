from __future__ import annotations

import json
from pathlib import Path

import typer

from graham_nyse.backtest.data import load_feature_panel, load_price_panel
from graham_nyse.backtest.engine import run_backtest
from graham_nyse.config import load_config

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run_command(
    config: str = typer.Option("config/strategy.yaml", help="Strategy YAML path"),
    output: str = typer.Option("outputs/latest", help="Output directory"),
    user_agent: str = typer.Option(..., envvar="SEC_USER_AGENT", help="App name and contact email"),
    refresh: bool = typer.Option(False, help="Refresh cached SEC responses"),
) -> None:
    from graham_nyse.pipeline import run

    run(config, output, user_agent, refresh)


@app.command("backtest")
def backtest_command(
    features: str = typer.Option(..., help="Point-in-time feature panel CSV or Parquet"),
    prices: str = typer.Option(..., help="Long-form adjusted-price CSV or Parquet"),
    start: str = typer.Option(..., help="Backtest start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Backtest end date YYYY-MM-DD"),
    config: str = typer.Option("config/strategy.yaml", help="Strategy YAML path"),
    output: str = typer.Option("outputs/backtest", help="Output directory"),
    transaction_cost_bps: float = typer.Option(10.0, min=0.0, help="One-way transaction costs in basis points"),
    benchmark: str | None = typer.Option(None, help="Optional benchmark adjusted-price CSV or Parquet"),
) -> None:
    cfg = load_config(config)
    feature_panel = load_feature_panel(features)
    price_panel = load_price_panel(prices)
    benchmark_panel = load_price_panel(benchmark) if benchmark else None
    result = run_backtest(feature_panel, price_panel, cfg, start=start, end=end, transaction_cost_bps=transaction_cost_bps, benchmark_prices=benchmark_panel)
    result.write(output)
    Path(output, "backtest_summary.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    typer.echo(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    app()
