from __future__ import annotations

import numpy as np
import pandas as pd

from graham_nyse.config import StrategyConfig


def select_constituents(scored: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return scored.loc[scored["eligible"]].nsmallest(cfg.portfolio.target_positions, "rank").copy()


def cap_and_redistribute(weights: pd.Series, cap: float, tolerance: float = 1e-12) -> pd.Series:
    w = weights / weights.sum()
    for _ in range(100):
        over = w > cap + tolerance
        if not over.any():
            return w / w.sum()
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = ~over
        if not under.any():
            raise ValueError("Position cap is infeasible")
        w.loc[under] += excess * w.loc[under] / w.loc[under].sum()
    raise RuntimeError("Weight capping did not converge")


def construct_portfolio(selected: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    result = selected.copy()
    raw = np.sqrt(result["market_cap"].clip(lower=1)) * result["graham_score"].clip(lower=1e-6)
    result["target_weight"] = cap_and_redistribute(raw, cfg.portfolio.max_position_weight)
    result["target_dollars"] = result["target_weight"] * cfg.portfolio.capital
    if cfg.portfolio.fractional_shares:
        result["target_shares"] = result["target_dollars"] / result["price"]
    else:
        result["target_shares"] = np.floor(result["target_dollars"] / result["price"])
        result["target_dollars"] = result["target_shares"] * result["price"]
        result["target_weight"] = result["target_dollars"] / cfg.portfolio.capital
    return result


def rebalance_orders(target: pd.DataFrame, current: pd.DataFrame, minimum_trade_dollars: float) -> pd.DataFrame:
    cur = current[["ticker", "shares"]].rename(columns={"shares": "current_shares"})
    merged = target.merge(cur, on="ticker", how="outer")
    merged["current_shares"] = merged["current_shares"].fillna(0.0)
    merged["target_shares"] = merged["target_shares"].fillna(0.0)
    merged["price"] = merged["price"].ffill().bfill()
    merged["trade_shares"] = merged["target_shares"] - merged["current_shares"]
    merged["trade_dollars"] = merged["trade_shares"] * merged["price"]
    merged.loc[merged["trade_dollars"].abs() < minimum_trade_dollars, ["trade_shares", "trade_dollars"]] = 0.0
    merged["side"] = np.select([merged["trade_dollars"] > 0, merged["trade_dollars"] < 0], ["BUY", "SELL"], default="HOLD")
    return merged
