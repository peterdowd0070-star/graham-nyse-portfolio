from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from graham_nyse.backtest.metrics import (
    compare_benchmarks,
    factor_attribution,
    performance_metrics,
)
from graham_nyse.backtest.tax import TaxLedger
from graham_nyse.config import StrategyConfig, WeightStrategy
from graham_nyse.data.security_master import SecurityMaster
from graham_nyse.data.vintages import FilingVintageStore
from graham_nyse.portfolio.construction import construct_portfolio, select_constituents
from graham_nyse.portfolio.scoring import calculate_scores
from graham_nyse.validation import validate_historical_run


@dataclass(frozen=True)
class PendingDecision:
    decision_at: pd.Timestamp
    run_type: str
    selected: pd.DataFrame
    return_history: pd.DataFrame


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    snapshots: pd.DataFrame
    scores: pd.DataFrame
    tax_lots: pd.DataFrame
    tax_events: pd.DataFrame
    metrics: dict[str, float]
    audit: dict[str, object]
    metadata: dict[str, object]

    def write(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.nav.to_csv(out / "historical_nav.csv", index=False)
        self.holdings.to_csv(out / "historical_holdings.csv", index=False)
        self.trades.to_csv(out / "historical_trades.csv", index=False)
        self.snapshots.to_csv(out / "vintage_snapshot_audit.csv", index=False)
        self.scores.to_parquet(out / "historical_scores.parquet", index=False)
        self.tax_lots.to_csv(out / "tax_lots.csv", index=False)
        self.tax_events.to_csv(out / "tax_events.csv", index=False)
        pd.Series(self.metrics, name="value").rename_axis("metric").to_csv(
            out / "historical_metrics.csv"
        )
        (out / "historical_run_audit.json").write_text(
            json.dumps(self.audit, indent=2, default=str), encoding="utf-8"
        )
        (out / "historical_run_manifest.json").write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "security_id", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Raw price panel is missing columns: {sorted(missing)}")
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    out["security_id"] = out["security_id"].astype(str)
    if out.duplicated(["date", "security_id"]).any():
        raise ValueError("Raw price panel contains duplicate date/security rows")
    if (pd.to_numeric(out["close"], errors="coerce") <= 0).any():
        raise ValueError("Raw closing prices must be positive")
    return out.sort_values(["date", "security_id"]).reset_index(drop=True)


def _validate_actions(actions: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["date", "security_id", "action_type", "value", "qualified"]
    if actions is None or actions.empty:
        return pd.DataFrame(columns=columns)
    missing = {"date", "security_id", "action_type", "value"} - set(actions.columns)
    if missing:
        raise ValueError(f"Corporate actions are missing columns: {sorted(missing)}")
    out = actions.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    out["security_id"] = out["security_id"].astype(str)
    out["action_type"] = out["action_type"].str.upper()
    out["qualified"] = out.get("qualified", False).fillna(False).astype(bool)
    unsupported = set(out["action_type"]) - {"DIVIDEND", "SPLIT"}
    if unsupported:
        raise ValueError(f"Unsupported corporate actions: {sorted(unsupported)}")
    return out[columns].sort_values(["date", "security_id"]).reset_index(drop=True)


def _decision_kind(month: int, cfg: StrategyConfig) -> str:
    if month in cfg.schedule.full_reconstruction_months:
        return "full_reconstruction"
    if month in cfg.schedule.quarterly_rebalance_months:
        return "quarterly_rebalance"
    return "monthly_monitor"


def _cutoff(day: pd.Timestamp, cfg: StrategyConfig) -> pd.Timestamp:
    local = pd.Timestamp(
        f"{day.date()} {cfg.execution.decision_time}", tz="America/New_York"
    )
    return local.tz_convert("UTC")


def _market_snapshot(
    prices: pd.DataFrame, decision_date: pd.Timestamp, vintage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = prices.loc[prices["date"].le(decision_date)].copy()
    if history.empty:
        raise ValueError(f"No price history is available at {decision_date.date()}")
    latest = history.groupby("security_id", as_index=False).tail(1)
    latest = latest.rename(columns={"close": "price"})
    trailing = history.loc[history["date"].ge(decision_date - pd.Timedelta(days=400))]
    trailing = trailing.copy()
    trailing["dollar_volume"] = trailing["close"] * trailing["volume"]
    liquidity = trailing.groupby("security_id")["dollar_volume"].apply(
        lambda s: float(s.tail(60).median())
    )
    close_panel = trailing.pivot(
        index="date", columns="security_id", values="close"
    ).sort_index()
    returns = close_panel.pct_change(fill_method=None).tail(252)
    volatility = returns.std(ddof=1) * np.sqrt(252)
    market = latest[["security_id", "price"]].copy()
    market["median_dollar_volume_60d"] = market["security_id"].map(liquidity)
    market["volatility_252d"] = market["security_id"].map(volatility)
    if "shares_outstanding" in latest:
        market["market_cap"] = latest["price"] * latest["shares_outstanding"]
    elif "market_cap" in vintage:
        market = market.merge(
            vintage[["security_id", "market_cap"]], on="security_id", how="left"
        )
    else:
        market["market_cap"] = np.nan
    return market, returns


def _build_decision(
    store: FilingVintageStore,
    master: SecurityMaster,
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    decision_date: pd.Timestamp,
    scenario: str,
    run_type: str,
    incumbents: set[str],
    previous_cutoff: pd.Timestamp | None,
) -> tuple[PendingDecision | None, dict[str, object], pd.DataFrame]:
    cutoff = _cutoff(decision_date, cfg)
    vintage = store.snapshot(cutoff)
    active = master.active_as_of(decision_date, cfg.universe.exchange)
    snapshot = active.merge(
        vintage, on="security_id", how="inner", suffixes=("", "_filing")
    )
    if snapshot.empty:
        raise RuntimeError(
            f"No point-in-time filings for active securities at {cutoff}"
        )
    for authoritative in ("ticker", "issuer_id", "company_domain", "sector"):
        filing_column = f"{authoritative}_filing"
        if filing_column in snapshot:
            snapshot = snapshot.drop(columns=[filing_column])
    market, returns = _market_snapshot(prices, decision_date, snapshot)
    snapshot = snapshot.drop(
        columns=[
            c
            for c in (
                "price",
                "market_cap",
                "median_dollar_volume_60d",
                "volatility_252d",
            )
            if c in snapshot
        ]
    )
    snapshot = snapshot.merge(market, on="security_id", how="inner")
    scored = calculate_scores(snapshot, cfg, scenario)
    score_audit = scored.copy()
    score_audit["decision_at"] = cutoff
    score_audit["run_type"] = run_type
    new_filings = store.frame.loc[store.frame["accepted_at"].le(cutoff)]
    if previous_cutoff is not None:
        new_filings = new_filings.loc[new_filings["accepted_at"].gt(previous_cutoff)]
    snapshot_audit = {
        "decision_at": cutoff,
        "run_type": run_type,
        "scenario": scenario,
        "active_security_count": len(active),
        "available_filing_count": len(vintage),
        "new_filing_count": len(new_filings),
        "eligible_count": int(scored["eligible"].sum()),
        "maximum_accepted_at": vintage["accepted_at"].max(),
    }
    if run_type == "monthly_monitor":
        return None, snapshot_audit, score_audit
    if run_type in {"initial_construction", "full_reconstruction"} or not incumbents:
        selected = select_constituents(scored, cfg)
    else:
        selected = scored.loc[
            scored["security_id"].isin(incumbents) & scored["base_eligible"]
        ].copy()
    if selected.empty:
        raise RuntimeError(f"No eligible holdings at {cutoff}")
    if len(selected) * cfg.portfolio.max_position_weight < 1.0 - 1e-9:
        raise RuntimeError(
            f"Too few eligible incumbents for the position cap at {cutoff}"
        )
    return (
        PendingDecision(cutoff, run_type, selected, returns),
        snapshot_audit,
        score_audit,
    )


def _execution_schedule(
    dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[dict[pd.Timestamp, str], dict[pd.Timestamp, pd.Timestamp]]:
    month_ends = dates.to_series().groupby(dates.to_period("M")).max()
    decisions = {day: "scheduled" for day in month_ends if start <= day <= end}
    next_session: dict[pd.Timestamp, pd.Timestamp] = {}
    for day in decisions:
        later = dates[dates > day]
        if len(later):
            next_session[day] = later[0]
    return decisions, next_session


def run_historical_backtest(
    filing_vintages: pd.DataFrame,
    security_master: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    scenario: str = "quality_value",
    weighting_strategy: WeightStrategy = "equal",
    tax_mode: str = "tax_deferred",
    tax_payment_source: str | None = None,
    corporate_actions: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | None = None,
    factor_returns: pd.DataFrame | None = None,
) -> BacktestResult:
    store = FilingVintageStore.from_frame(filing_vintages)
    master = SecurityMaster.from_frame(security_master)
    price_frame = _validate_prices(prices)
    actions = _validate_actions(corporate_actions)
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    price_frame = price_frame.loc[
        price_frame["date"].between(start_ts - pd.Timedelta(days=400), end_ts)
    ].copy()
    dates = pd.DatetimeIndex(
        sorted(
            price_frame.loc[
                price_frame["date"].between(start_ts, end_ts), "date"
            ].unique()
        )
    )
    if len(dates) < 2:
        raise ValueError("At least two trading sessions are required")
    decision_days, next_sessions = _execution_schedule(dates, start_ts, end_ts)
    final_session_by_year = (
        pd.Series(dates, index=dates).groupby(dates.year).max().to_dict()
    )

    tax = TaxLedger(cfg.tax, tax_mode)
    payment_source = (
        cfg.tax.payment_source if tax_payment_source is None else tax_payment_source
    )
    if payment_source not in {"portfolio", "external"}:
        raise ValueError("tax_payment_source must be 'portfolio' or 'external'")
    external_tax_paid = 0.0
    shares: dict[str, float] = {}
    cash = float(cfg.portfolio.capital)
    last_prices: dict[str, float] = {}
    last_price_dates: dict[str, pd.Timestamp] = {}
    pending: dict[pd.Timestamp, PendingDecision] = {}
    nav_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    turnover_rows: list[tuple[pd.Timestamp, float]] = []
    previous_cutoff: pd.Timestamp | None = None

    # Initial construction uses only information public before the first session.
    initial_decision_date = dates[0] - pd.Timedelta(days=1)
    initial, audit_row, score_frame = _build_decision(
        store,
        master,
        price_frame,
        cfg,
        initial_decision_date,
        scenario,
        "initial_construction",
        set(),
        previous_cutoff,
    )
    if initial is None:
        raise RuntimeError("Initial construction did not produce a decision")
    pending[dates[0]] = initial
    snapshot_rows.append(audit_row)
    score_frames.append(score_frame)
    previous_cutoff = initial.decision_at

    delisting_schedule: dict[pd.Timestamp, list[pd.Series]] = {}
    terminal_rows = master.frame.loc[master.frame["is_delisted"]]
    for _, record in terminal_rows.iterrows():
        candidates = dates[dates >= record["listing_end"]]
        if len(candidates):
            delisting_schedule.setdefault(candidates[0], []).append(record)

    for day in dates:
        day_prices = price_frame.loc[price_frame["date"].eq(day)].set_index(
            "security_id"
        )
        for security_id, row in day_prices.iterrows():
            last_prices[str(security_id)] = float(row["close"])
            last_price_dates[str(security_id)] = day

        for action in actions.loc[actions["date"].eq(day)].itertuples(index=False):
            security_id = str(action.security_id)
            quantity = shares.get(security_id, 0.0)
            if quantity <= 0:
                continue
            if action.action_type == "SPLIT":
                ratio = float(action.value)
                shares[security_id] *= ratio
                for lot in tax.lots.get(security_id, []):
                    lot.quantity *= ratio
                    lot.basis_per_share /= ratio
            elif action.action_type == "DIVIDEND":
                amount = quantity * float(action.value)
                cash += amount
                tax.dividend(security_id, amount, day, bool(action.qualified))

        for record in delisting_schedule.get(day, []):
            security_id = str(record["security_id"])
            quantity = shares.get(security_id, 0.0)
            if quantity <= 0:
                continue
            delisting_return = record["delisting_return"]
            if not np.isfinite(delisting_return):
                if cfg.universe.require_delisting_returns:
                    raise RuntimeError(
                        f"Missing delisting return for held security {security_id}"
                    )
                delisting_return = -1.0
            reference_price = last_prices.get(security_id)
            if reference_price is None:
                raise RuntimeError(f"No final price for delisted holding {security_id}")
            proceeds_price = reference_price * (1.0 + float(delisting_return))
            proceeds = quantity * proceeds_price
            realized = tax.sell(security_id, quantity, proceeds_price, day)
            cash += proceeds
            shares.pop(security_id, None)
            trade_rows.append(
                {
                    "date": day,
                    "decision_at": pd.NaT,
                    "run_type": "delisting",
                    "security_id": security_id,
                    "trade_shares": -quantity,
                    "trade_price": proceeds_price,
                    "trade_dollars": -proceeds,
                    "transaction_cost": 0.0,
                    "realized_gain": realized,
                    "side": "SELL",
                }
            )

        def marked_nav(current_day: pd.Timestamp, cash_balance: float) -> float:
            value = cash_balance
            for security_id, quantity in shares.items():
                price = last_prices.get(security_id)
                price_day = last_price_dates.get(security_id)
                if (
                    price is None
                    or price_day is None
                    or (current_day - price_day).days
                    > cfg.universe.maximum_price_staleness_days
                ):
                    raise RuntimeError(
                        f"Stale or missing price for held security {security_id} on {current_day.date()}"
                    )
                value += quantity * price
            return float(value)

        if day in pending:
            decision = pending.pop(day)
            pre_trade_nav = marked_nav(day, cash)
            selected = decision.selected.loc[
                decision.selected["security_id"].isin(day_prices.index.astype(str))
            ].copy()
            selected["price"] = selected["security_id"].map(
                day_prices["close"].to_dict()
            )
            investable = pre_trade_nav * (
                1.0 - cfg.execution.transaction_cost_bps / 10_000.0
            )
            target = construct_portfolio(
                selected,
                cfg,
                weighting_strategy,
                decision.return_history,
                capital=investable,
            )
            desired = dict(
                zip(target["security_id"], target["target_shares"], strict=True)
            )
            all_ids = set(shares) | set(desired)
            orders: list[tuple[str, float, float]] = []
            for security_id in all_ids:
                price = last_prices.get(security_id)
                if price is None:
                    raise RuntimeError(
                        f"No execution price for {security_id} on {day.date()}"
                    )
                delta = desired.get(security_id, 0.0) - shares.get(security_id, 0.0)
                if abs(delta * price) < cfg.portfolio.minimum_trade_dollars:
                    delta = 0.0
                orders.append((security_id, delta, price))
            orders.sort(key=lambda order: order[1])  # sells fund buys
            gross = 0.0
            for security_id, delta, price in orders:
                if abs(delta) <= 1e-12:
                    continue
                rate = cfg.execution.transaction_cost_bps / 10_000.0
                if delta < 0:
                    quantity = -delta
                    dollars = quantity * price
                    cost = dollars * rate
                    realized = tax.sell(security_id, quantity, price, day)
                    cash += dollars - cost
                    shares[security_id] = shares.get(security_id, 0.0) - quantity
                    if shares[security_id] <= 1e-12:
                        shares.pop(security_id, None)
                    signed_dollars, signed_quantity, side = -dollars, -quantity, "SELL"
                else:
                    affordable = max(0.0, cash) / (price * (1.0 + rate))
                    quantity = min(delta, affordable)
                    if quantity <= 1e-12:
                        continue
                    dollars = quantity * price
                    cost = dollars * rate
                    cash -= dollars + cost
                    shares[security_id] = shares.get(security_id, 0.0) + quantity
                    tax.buy(security_id, quantity, price, day)
                    realized = 0.0
                    signed_dollars, signed_quantity, side = dollars, quantity, "BUY"
                gross += dollars
                trade_rows.append(
                    {
                        "date": day,
                        "decision_at": decision.decision_at,
                        "run_type": decision.run_type,
                        "security_id": security_id,
                        "trade_shares": signed_quantity,
                        "trade_price": price,
                        "trade_dollars": signed_dollars,
                        "transaction_cost": cost,
                        "realized_gain": realized,
                        "side": side,
                    }
                )
            turnover_rows.append((day, gross / (2.0 * max(pre_trade_nav, 1e-12))))
            post_nav = marked_nav(day, cash)
            for security_id, quantity in shares.items():
                security = master.frame.loc[
                    master.frame["security_id"].eq(security_id)
                ].iloc[0]
                value = quantity * last_prices[security_id]
                holding_rows.append(
                    {
                        "date": day,
                        "decision_at": decision.decision_at,
                        "run_type": decision.run_type,
                        "security_id": security_id,
                        "ticker": security["ticker"],
                        "scenario": scenario,
                        "weighting_strategy": weighting_strategy,
                        "shares": quantity,
                        "price": last_prices[security_id],
                        "market_value": value,
                        "weight": value / post_nav,
                    }
                )

        if final_session_by_year.get(day.year) == day:
            tax_due = tax.settle_year(day.year)
            if payment_source == "portfolio":
                cash -= tax_due
            else:
                external_tax_paid += tax_due

        if day in decision_days:
            run_type = _decision_kind(day.month, cfg)
            scheduled_decision, audit_row, score_frame = _build_decision(
                store,
                master,
                price_frame,
                cfg,
                day,
                scenario,
                run_type,
                set(shares),
                previous_cutoff,
            )
            snapshot_rows.append(audit_row)
            score_frames.append(score_frame)
            previous_cutoff = _cutoff(day, cfg)
            execution_day = next_sessions.get(day)
            if scheduled_decision is not None and execution_day is not None:
                pending[execution_day] = scheduled_decision

        nav_rows.append(
            {
                "date": day,
                "nav": marked_nav(day, cash),
                "cash": cash,
                "scenario": scenario,
                "weighting_strategy": weighting_strategy,
                "tax_mode": tax_mode,
            }
        )

    if tax.terminal_liquidation and shares:
        day = dates[-1]
        rate = cfg.execution.transaction_cost_bps / 10_000.0
        for security_id, quantity in list(shares.items()):
            price = last_prices[security_id]
            proceeds = quantity * price
            cost = proceeds * rate
            realized = tax.sell(security_id, quantity, price, day)
            cash += proceeds - cost
            trade_rows.append(
                {
                    "date": day,
                    "decision_at": pd.NaT,
                    "run_type": "terminal_liquidation",
                    "security_id": security_id,
                    "trade_shares": -quantity,
                    "trade_price": price,
                    "trade_dollars": -proceeds,
                    "transaction_cost": cost,
                    "realized_gain": realized,
                    "side": "SELL",
                }
            )
            shares.pop(security_id)
        tax_due = tax.settle_year(day.year)
        if payment_source == "portfolio":
            cash -= tax_due
        else:
            external_tax_paid += tax_due
        nav_rows[-1]["nav"] = cash
        nav_rows[-1]["cash"] = cash

    nav = pd.DataFrame(nav_rows)
    holdings = pd.DataFrame(holding_rows)
    trades = pd.DataFrame(trade_rows)
    snapshots = pd.DataFrame(snapshot_rows)
    scores = (
        pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    )
    nav_series = nav.set_index("date")["nav"]
    turnover = pd.Series(
        {date: value for date, value in turnover_rows}, name="turnover"
    )
    metrics = performance_metrics(nav_series, turnover)
    metrics.update(compare_benchmarks(nav_series, benchmark_returns))
    metrics.update(factor_attribution(nav_series, factor_returns))
    metrics["tax_paid"] = float(tax.tax_paid)
    metrics["external_tax_paid"] = float(external_tax_paid)
    metrics["after_tax_end_wealth"] = float(nav_series.iloc[-1] - external_tax_paid)
    audit = validate_historical_run(
        store.frame,
        master.frame,
        price_frame,
        nav,
        holdings,
        trades,
        snapshots,
        cfg,
    )
    metadata = {
        "engine": "historical_point_in_time",
        "scenario": scenario,
        "weighting_strategy": weighting_strategy,
        "tax_mode": tax_mode,
        "tax_payment_source": payment_source,
        "start": start_ts,
        "end": end_ts,
        "uses_current_constituent_list": False,
        "uses_adjusted_prices": False,
        "filing_availability_field": "accepted_at",
        "methodology_version": "1.0.0",
    }
    if not audit["passed"]:
        raise RuntimeError(f"Historical validation failed: {audit['errors']}")
    return BacktestResult(
        nav,
        holdings,
        trades,
        snapshots,
        scores,
        tax.lot_frame(),
        tax.event_frame(),
        metrics,
        audit,
        metadata,
    )


def run_experiment_matrix(
    filing_vintages: pd.DataFrame,
    security_master: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    corporate_actions: pd.DataFrame | None = None,
    tax_mode: str = "tax_deferred",
    tax_payment_source: str | None = None,
    benchmark_returns: pd.DataFrame | None = None,
    factor_returns: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], BacktestResult]]:
    rows: list[dict[str, object]] = []
    results: dict[tuple[str, str], BacktestResult] = {}
    for scenario in cfg.valuation.scenarios:
        for weighting_strategy in cfg.portfolio.weighting_strategies:
            result = run_historical_backtest(
                filing_vintages,
                security_master,
                prices,
                cfg,
                start,
                end,
                scenario=scenario,
                weighting_strategy=weighting_strategy,
                tax_mode=tax_mode,
                tax_payment_source=tax_payment_source,
                corporate_actions=corporate_actions,
                benchmark_returns=benchmark_returns,
                factor_returns=factor_returns,
            )
            results[(scenario, weighting_strategy)] = result
            rows.append(
                {
                    "scenario": scenario,
                    "weighting_strategy": weighting_strategy,
                    **result.metrics,
                }
            )
    return pd.DataFrame(rows), results


# Intentional breaking alias: old feature-panel backtests must migrate to filing vintages.
def run_backtest(*args: object, **kwargs: object) -> BacktestResult:
    raise RuntimeError(
        "run_backtest(feature_panel, adjusted_prices, ...) was removed. "
        "Use run_historical_backtest with filing vintages, a dated security master, raw prices, and corporate actions."
    )
