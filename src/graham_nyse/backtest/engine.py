from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from graham_nyse.config import StrategyConfig
from graham_nyse.portfolio.construction import construct_portfolio, select_constituents
from graham_nyse.portfolio.scoring import apply_eligibility, calculate_scores
from graham_nyse.backtest.metrics import benchmark_comparison, performance_metrics


REQUIRED_FEATURE_COLUMNS = {
    "as_of_date", "ticker", "price", "market_cap", "median_dollar_volume_60d",
    "positive_earnings_years", "earnings_history_years", "interest_coverage", "cfo",
    "normalized_net_income", "fcf", "net_debt", "equity", "assets", "roa",
    "accruals", "earnings_stability",
}


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]

    def write(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.nav.to_csv(out / "backtest_nav.csv", index=False)
        self.holdings.to_csv(out / "backtest_holdings.csv", index=False)
        self.trades.to_csv(out / "backtest_trades.csv", index=False)
        pd.Series(self.metrics, name="value").rename_axis("metric").to_csv(out / "backtest_metrics.csv")


def _validate_inputs(features: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_FEATURE_COLUMNS - set(features.columns)
    if missing:
        raise ValueError(f"Feature panel is missing required columns: {sorted(missing)}")
    f = features.copy()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"]).dt.normalize()
    if f.duplicated(["as_of_date", "ticker"]).any():
        raise ValueError("Feature panel contains duplicate date/ticker rows")
    p = prices.copy()
    if not {"date", "ticker", "adjusted_close"}.issubset(p.columns):
        raise ValueError("Prices must contain date, ticker, and adjusted_close")
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p.drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"])
    return f.sort_values(["as_of_date", "ticker"]), p


def _latest_snapshot(features: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    available = features.loc[features["as_of_date"] <= decision_date]
    if available.empty:
        return available
    latest = available.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
    return latest.copy()


def _weights_for_snapshot(
    snapshot: pd.DataFrame,
    cfg: StrategyConfig,
    incumbents: set[str] | None,
    full_reconstruction: bool,
) -> pd.DataFrame:
    eligible = apply_eligibility(snapshot, cfg)
    scored = calculate_scores(eligible, cfg)
    if full_reconstruction or not incumbents:
        selected = select_constituents(scored, cfg)
    else:
        selected = scored.loc[scored["ticker"].isin(incumbents) & scored["eligible"]].copy()
    if selected.empty:
        return selected
    return construct_portfolio(selected, cfg)


def _decision_kind(month: int, cfg: StrategyConfig) -> str | None:
    if month in cfg.schedule.full_reconstruction_months:
        return "full_reconstruction"
    if month in cfg.schedule.quarterly_rebalance_months:
        return "quarterly_rebalance"
    return None


def run_backtest(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    benchmark_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    features, prices = _validate_inputs(features, prices)
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    px = prices.loc[prices["date"].between(start_ts, end_ts)].pivot(index="date", columns="ticker", values="adjusted_close").sort_index().ffill()
    if px.empty:
        raise ValueError("No prices in requested backtest range")

    month_ends = px.index.to_series().groupby(px.index.to_period("M")).max()
    decision_dates = [d for d in month_ends if _decision_kind(d.month, cfg) is not None]
    decision_dates = [d for d in decision_dates if d >= start_ts]
    first_trading_day = px.index[0]
    if first_trading_day not in decision_dates:
        decision_dates.insert(0, first_trading_day)
    if not decision_dates:
        raise ValueError("No configured rebalance dates in requested range")

    nav_value = float(cfg.portfolio.capital)
    shares: dict[str, float] = {}
    cash = nav_value
    nav_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    turnover_rows: list[tuple[pd.Timestamp, float]] = []
    decision_set = set(decision_dates)

    for day in px.index:
        row = px.loc[day]
        position_value = sum(qty * float(row.get(ticker, np.nan)) for ticker, qty in shares.items() if pd.notna(row.get(ticker, np.nan)))
        nav_value = cash + position_value

        if day in decision_set:
            kind = "initial_construction" if day == first_trading_day else _decision_kind(day.month, cfg)
            snapshot = _latest_snapshot(features, day)
            incumbents = set(shares)
            target = _weights_for_snapshot(snapshot, cfg, incumbents, kind in {"initial_construction", "full_reconstruction"})
            if target.empty:
                raise RuntimeError(f"No eligible holdings on {day.date()}")
            target = target.loc[target["ticker"].isin(row.dropna().index)].copy()
            target["live_price"] = target["ticker"].map(row.to_dict())
            target = target.loc[target["live_price"].gt(0)].copy()
            target["target_dollars_live"] = target["target_weight"] / target["target_weight"].sum() * nav_value
            target["target_shares_live"] = target["target_dollars_live"] / target["live_price"]
            target_shares = dict(zip(target["ticker"], target["target_shares_live"], strict=True))

            all_names = sorted(set(shares) | set(target_shares))
            gross_traded = 0.0
            for ticker in all_names:
                current_qty = shares.get(ticker, 0.0)
                desired_qty = target_shares.get(ticker, 0.0)
                live_price = float(row.get(ticker, np.nan))
                if not np.isfinite(live_price):
                    continue
                delta = desired_qty - current_qty
                dollars = delta * live_price
                if abs(dollars) < cfg.portfolio.minimum_trade_dollars:
                    desired_qty = current_qty
                    delta = 0.0
                    dollars = 0.0
                gross_traded += abs(dollars)
                if delta != 0:
                    trade_rows.append({"date": day, "run_type": kind, "ticker": ticker, "trade_shares": delta, "trade_dollars": dollars, "side": "BUY" if delta > 0 else "SELL"})
                if desired_qty > 1e-12:
                    shares[ticker] = desired_qty
                else:
                    shares.pop(ticker, None)

            cost = gross_traded * transaction_cost_bps / 10_000.0
            invested = sum(qty * float(row[ticker]) for ticker, qty in shares.items())
            cash = nav_value - invested - cost
            nav_value = cash + invested
            turnover_rows.append((day, gross_traded / (2.0 * max(nav_value, 1e-12))))
            for ticker, qty in shares.items():
                holding_rows.append({"date": day, "run_type": kind, "ticker": ticker, "shares": qty, "price": float(row[ticker]), "market_value": qty * float(row[ticker]), "weight": qty * float(row[ticker]) / nav_value})

        nav_rows.append({"date": day, "nav": nav_value, "cash": cash})

    nav = pd.DataFrame(nav_rows)
    holdings = pd.DataFrame(holding_rows)
    trades = pd.DataFrame(trade_rows)
    turnover = pd.Series({d: value for d, value in turnover_rows}, name="turnover")
    nav_series = nav.set_index("date")["nav"]
    metrics = performance_metrics(nav_series, turnover)
    if benchmark_prices is not None and not benchmark_prices.empty:
        b = benchmark_prices.copy()
        b["date"] = pd.to_datetime(b["date"]).dt.normalize()
        b = b.set_index("date")["adjusted_close"].sort_index().reindex(nav_series.index).ffill().dropna()
        if not b.empty:
            benchmark_nav = b / b.iloc[0] * cfg.portfolio.capital
            metrics.update(benchmark_comparison(nav_series, benchmark_nav))
    return BacktestResult(nav=nav, holdings=holdings, trades=trades, metrics=metrics)
