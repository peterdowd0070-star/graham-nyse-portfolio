from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from graham_nyse.config import StrategyConfig


def dataframe_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    digest = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    return hashlib.sha256(digest).hexdigest()


def validate_portfolio(portfolio: pd.DataFrame, cfg: StrategyConfig) -> dict[str, Any]:
    errors: list[str] = []
    if portfolio.empty:
        errors.append("portfolio_is_empty")
        return {"passed": False, "errors": errors}
    if not math.isclose(
        float(portfolio["target_weight"].sum()), 1.0, rel_tol=0, abs_tol=1e-8
    ):
        errors.append("weights_do_not_sum_to_one")
    if (portfolio["target_weight"] > cfg.portfolio.max_position_weight + 1e-8).any():
        errors.append("position_cap_exceeded")
    if (
        "sector" in portfolio
        and (
            portfolio.groupby("sector")["target_weight"].sum()
            > cfg.portfolio.max_sector_weight + 1e-8
        ).any()
    ):
        errors.append("sector_cap_exceeded")
    if portfolio[["price", "target_weight", "target_shares"]].isna().any().any():
        errors.append("missing_portfolio_values")
    return {"passed": not errors, "errors": errors}


def _accounting_errors(vintages: pd.DataFrame, tolerance: float) -> list[str]:
    required = {"assets", "liabilities", "equity"}
    if not required.issubset(vintages):
        return []
    sample = vintages.dropna(subset=list(required)).copy()
    if sample.empty:
        return []
    denominator = sample["assets"].abs().clip(lower=1.0)
    gap = (
        sample["assets"] - sample["liabilities"] - sample["equity"]
    ).abs() / denominator
    return ["accounting_equation_failure"] if gap.gt(tolerance).any() else []


def _trade_inventory_errors(
    trades: pd.DataFrame, corporate_actions: pd.DataFrame | None = None
) -> list[str]:
    if trades.empty:
        return []
    inventory: dict[str, float] = {}
    events: list[tuple[pd.Timestamp, int, str, float, str]] = []
    if corporate_actions is not None and not corporate_actions.empty:
        splits = corporate_actions.loc[corporate_actions["action_type"].eq("SPLIT")]
        for row in splits.itertuples(index=False):
            events.append(
                (
                    pd.Timestamp(row.date),
                    0,
                    str(row.security_id),
                    float(row.value),
                    "SPLIT",
                )
            )
    for row in trades.itertuples(index=False):
        # Corporate actions are applied before execution.  Within a session,
        # buys precede sells for reconciliation so an initial purchase and a
        # same-day terminal liquidation remain valid.
        priority = 1 if str(row.side) == "BUY" else 2
        events.append(
            (
                pd.Timestamp(row.date),
                priority,
                str(row.security_id),
                float(row.trade_shares),
                "TRADE",
            )
        )
    for _, _, security_id, value, event_type in sorted(events):
        if event_type == "SPLIT":
            inventory[security_id] = inventory.get(security_id, 0.0) * value
        else:
            inventory[security_id] = inventory.get(security_id, 0.0) + value
            if inventory[security_id] < -1e-8:
                return ["trade_inventory_below_zero"]
    return []


def validate_historical_run(
    vintages: pd.DataFrame,
    security_master: pd.DataFrame,
    prices: pd.DataFrame,
    nav: pd.DataFrame,
    holdings: pd.DataFrame,
    trades: pd.DataFrame,
    snapshots: pd.DataFrame,
    cfg: StrategyConfig,
    corporate_actions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if nav.empty or not np.isfinite(nav["nav"]).all() or (nav["nav"] <= 0).any():
        errors.append("invalid_nav")
    if prices.duplicated(["date", "security_id"]).any():
        errors.append("duplicate_price_observation")
    if vintages.duplicated(["security_id", "accession_number"]).any():
        errors.append("mutable_or_duplicate_filing_vintage")
    if not snapshots.empty:
        future = pd.to_datetime(
            snapshots["maximum_accepted_at"], utc=True
        ) > pd.to_datetime(snapshots["decision_at"], utc=True)
        if future.fillna(False).any():
            errors.append("lookahead_filing_detected")
    if cfg.universe.require_delisting_returns:
        incomplete = (
            security_master["listing_end"].notna()
            & security_master["delisting_return"].isna()
        )
        if incomplete.any():
            errors.append("missing_delisting_returns")
    errors.extend(_accounting_errors(vintages, cfg.validation.accounting_tolerance))
    errors.extend(_trade_inventory_errors(trades, corporate_actions))
    if not holdings.empty:
        grouped = holdings.groupby("date")["weight"].sum()
        if grouped.gt(1.0 + cfg.validation.nav_tolerance).any():
            errors.append("holding_weights_exceed_nav")
        if holdings[["shares", "price", "market_value", "weight"]].isna().any().any():
            errors.append("missing_holding_values")
    if not trades.empty and (trades["transaction_cost"] < 0).any():
        errors.append("negative_transaction_cost")
    if (nav["cash"] < -cfg.validation.nav_tolerance).any():
        warnings.append("negative_cash_from_tax_or_execution")
    decision_times = pd.to_datetime(
        snapshots.get("decision_at", pd.Series(dtype=str)), utc=True
    ).dt.tz_convert(None)
    monitor_months = set(decision_times.dt.to_period("M").astype(str))
    expected_months = set(
        pd.period_range(nav["date"].min(), nav["date"].max(), freq="M").astype(str)
    )
    if cfg.schedule.monitor_monthly and len(expected_months - monitor_months) > 1:
        errors.append("missing_monthly_vintage_snapshots")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "checks": {
            "temporal_integrity": "passed"
            if "lookahead_filing_detected" not in errors
            else "failed",
            "security_master_integrity": "passed"
            if "missing_delisting_returns" not in errors
            else "failed",
            "accounting_integrity": "passed"
            if "accounting_equation_failure" not in errors
            else "failed",
            "portfolio_reconciliation": "passed"
            if not {
                "invalid_nav",
                "holding_weights_exceed_nav",
                "trade_inventory_below_zero",
            }
            & set(errors)
            else "failed",
        },
        "lineage": {
            "filing_vintages_sha256": dataframe_hash(vintages),
            "security_master_sha256": dataframe_hash(security_master),
            "prices_sha256": dataframe_hash(prices),
        },
    }
