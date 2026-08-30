from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from graham_nyse.config import StrategyConfig, WeightStrategy


def select_constituents(scored: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return (
        scored.loc[scored["eligible"]]
        .nsmallest(cfg.portfolio.target_positions, "rank")
        .copy()
    )


def cap_and_redistribute(
    weights: pd.Series, cap: float, tolerance: float = 1e-12
) -> pd.Series:
    if len(weights) * cap < 1.0 - tolerance:
        raise ValueError("Position cap is infeasible")
    sectors = pd.Series("__all__", index=weights.index)
    return _project_to_constraints(weights, sectors, cap, 1.0)


def _constraints(sectors: pd.Series, sector_cap: float) -> list[dict[str, object]]:
    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    values = sectors.astype(str).to_numpy()
    for sector in sorted(set(values)):
        mask = values == sector
        constraints.append(
            {"type": "ineq", "fun": lambda w, m=mask: sector_cap - np.sum(w[m])}
        )
    return constraints


def _project_to_constraints(
    raw: pd.Series,
    sectors: pd.Series,
    position_cap: float,
    sector_cap: float,
) -> pd.Series:
    clean = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if clean.sum() <= 0:
        clean[:] = 1.0
    target = (clean / clean.sum()).to_numpy()
    n = len(target)
    if n * position_cap < 1.0 - 1e-9:
        raise ValueError("Position cap is infeasible for selected count")
    result = minimize(
        lambda w: float(np.sum((w - target) ** 2)),
        x0=np.repeat(1.0 / n, n),
        bounds=[(0.0, position_cap)] * n,
        constraints=_constraints(sectors.reindex(raw.index), sector_cap),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise ValueError(f"Portfolio constraints are infeasible: {result.message}")
    return pd.Series(result.x / result.x.sum(), index=raw.index)


def _risk_inputs(
    selected: pd.DataFrame, return_history: pd.DataFrame | None
) -> tuple[pd.Series, np.ndarray]:
    id_column = "security_id" if "security_id" in selected else "ticker"
    ids = selected[id_column].astype(str).tolist()
    if return_history is not None and not return_history.empty:
        available = return_history.reindex(columns=ids).dropna(how="all")
        vol = available.std(ddof=1) * np.sqrt(252)
        covariance = available.cov(
            min_periods=max(10, min(60, len(available) // 2))
        ).reindex(index=ids, columns=ids)
        diagonal = (
            np.square(vol.reindex(ids).fillna(vol.median()).fillna(0.30).to_numpy())
            / 252
        )
        matrix = covariance.fillna(0.0).to_numpy()
        matrix = 0.8 * matrix + 0.2 * np.diag(diagonal)
    else:
        source = (
            selected["volatility_252d"]
            if "volatility_252d" in selected
            else pd.Series(0.30, index=selected.index)
        )
        vol = pd.Series(source.to_numpy(), index=ids, dtype=float)
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(vol.median()).fillna(0.30)
        matrix = np.diag(np.square(vol.to_numpy()) / 252)
    return vol.reindex(ids), matrix + np.eye(len(ids)) * 1e-10


def _raw_weights(
    selected: pd.DataFrame,
    strategy: WeightStrategy,
    return_history: pd.DataFrame | None,
    cfg: StrategyConfig,
) -> pd.Series:
    indexes = selected.index
    if strategy == "equal":
        return pd.Series(1.0, index=indexes)
    if strategy == "score_proportional":
        return selected["graham_score"].clip(lower=1e-8)
    if strategy == "liquidity_adjusted_equal":
        return np.sqrt(selected["median_dollar_volume_60d"].clip(lower=1.0))
    volatility, covariance = _risk_inputs(selected, return_history)
    id_column = "security_id" if "security_id" in selected else "ticker"
    vol_by_row = selected[id_column].astype(str).map(volatility)
    if strategy == "inverse_volatility":
        return pd.Series(1.0 / vol_by_row.clip(lower=1e-4).to_numpy(), index=indexes)
    if strategy == "score_over_volatility":
        return pd.Series(
            selected["graham_score"].to_numpy()
            / vol_by_row.clip(lower=1e-4).to_numpy(),
            index=indexes,
        )
    if strategy == "minimum_variance":
        sectors = selected["sector"].astype(str)
        n = len(selected)
        result = minimize(
            lambda w: float(w @ covariance @ w),
            x0=np.repeat(1.0 / n, n),
            bounds=[(0.0, cfg.portfolio.max_position_weight)] * n,
            constraints=_constraints(sectors, cfg.portfolio.max_sector_weight),
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise ValueError(f"Minimum-variance optimization failed: {result.message}")
        return pd.Series(result.x, index=indexes)
    raise ValueError(f"Unknown weighting strategy: {strategy}")


def construct_portfolio(
    selected: pd.DataFrame,
    cfg: StrategyConfig,
    weighting_strategy: WeightStrategy = "equal",
    return_history: pd.DataFrame | None = None,
    capital: float | None = None,
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    result = selected.copy()
    raw = _raw_weights(result, weighting_strategy, return_history, cfg)
    result["target_weight"] = _project_to_constraints(
        raw,
        result["sector"].astype(str),
        cfg.portfolio.max_position_weight,
        cfg.portfolio.max_sector_weight,
    )
    target_capital = cfg.portfolio.capital if capital is None else float(capital)
    result["target_dollars"] = result["target_weight"] * target_capital
    if cfg.portfolio.fractional_shares:
        result["target_shares"] = result["target_dollars"] / result["price"]
    else:
        result["target_shares"] = np.floor(result["target_dollars"] / result["price"])
        result["target_dollars"] = result["target_shares"] * result["price"]
        result["target_weight"] = result["target_dollars"] / target_capital
    result["weighting_strategy"] = weighting_strategy
    return result


def rebalance_orders(
    target: pd.DataFrame, current: pd.DataFrame, minimum_trade_dollars: float
) -> pd.DataFrame:
    cur = current[["security_id", "shares"]].rename(
        columns={"shares": "current_shares"}
    )
    merged = target.merge(cur, on="security_id", how="outer")
    merged["current_shares"] = merged["current_shares"].fillna(0.0)
    merged["target_shares"] = merged["target_shares"].fillna(0.0)
    merged["trade_shares"] = merged["target_shares"] - merged["current_shares"]
    merged["trade_dollars"] = merged["trade_shares"] * merged["price"]
    merged.loc[
        merged["trade_dollars"].abs() < minimum_trade_dollars,
        ["trade_shares", "trade_dollars"],
    ] = 0.0
    merged["side"] = np.select(
        [merged["trade_dollars"] > 0, merged["trade_dollars"] < 0],
        ["BUY", "SELL"],
        default="HOLD",
    )
    return merged
