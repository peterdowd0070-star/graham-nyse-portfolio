from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance_metrics(nav: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    """Calculate standard annualized performance statistics from a dated NAV series."""
    clean = nav.dropna().astype(float)
    if len(clean) < 2:
        raise ValueError("At least two NAV observations are required")
    daily = clean.pct_change().dropna()
    years = max((clean.index[-1] - clean.index[0]).days / 365.25, 1 / 365.25)
    total_return = clean.iloc[-1] / clean.iloc[0] - 1.0
    cagr = (clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0
    volatility = daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 else 0.0
    sharpe = daily.mean() / daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 and daily.std(ddof=1) > 0 else 0.0
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    max_drawdown = float(drawdown.min())
    result = {
        "start_nav": float(clean.iloc[0]),
        "end_nav": float(clean.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe_zero_rf": float(sharpe),
        "max_drawdown": max_drawdown,
    }
    if turnover is not None and not turnover.empty:
        result["average_one_way_turnover"] = float(turnover.mean())
        result["annualized_one_way_turnover"] = float(turnover.sum() / years)
    return result


def benchmark_comparison(portfolio_nav: pd.Series, benchmark_nav: pd.Series) -> dict[str, float]:
    aligned = pd.concat([portfolio_nav.rename("portfolio"), benchmark_nav.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 2:
        return {}
    p = aligned["portfolio"].pct_change().dropna()
    b = aligned["benchmark"].pct_change().dropna()
    active = p - b
    tracking_error = active.std(ddof=1) * np.sqrt(252) if len(active) > 1 else 0.0
    information_ratio = active.mean() / active.std(ddof=1) * np.sqrt(252) if len(active) > 1 and active.std(ddof=1) > 0 else 0.0
    beta = p.cov(b) / b.var(ddof=1) if len(p) > 1 and b.var(ddof=1) > 0 else np.nan
    return {
        "benchmark_total_return": float(aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1.0),
        "active_total_return": float(
            aligned["portfolio"].iloc[-1] / aligned["portfolio"].iloc[0]
            - aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0]
        ),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
        "beta": float(beta) if np.isfinite(beta) else float("nan"),
    }
