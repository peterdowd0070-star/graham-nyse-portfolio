from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance_metrics(
    nav: pd.Series, turnover: pd.Series | None = None
) -> dict[str, float]:
    clean = nav.dropna().astype(float)
    if len(clean) < 2:
        raise ValueError("At least two NAV observations are required")
    daily = clean.pct_change().dropna()
    years = max((clean.index[-1] - clean.index[0]).days / 365.25, 1 / 365.25)
    total_return = clean.iloc[-1] / clean.iloc[0] - 1.0
    cagr = (clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0
    volatility = daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 else 0.0
    sharpe = (
        daily.mean() / daily.std(ddof=1) * math.sqrt(252)
        if len(daily) > 1 and daily.std(ddof=1) > 0
        else 0.0
    )
    drawdown = clean / clean.cummax() - 1.0
    result = {
        "start_nav": float(clean.iloc[0]),
        "end_nav": float(clean.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe_zero_rf": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
    if turnover is not None and not turnover.empty:
        result["average_one_way_turnover"] = float(turnover.mean())
        result["annualized_one_way_turnover"] = float(turnover.sum() / years)
    return result


def benchmark_comparison(
    portfolio_nav: pd.Series, benchmark_nav: pd.Series, prefix: str = "benchmark"
) -> dict[str, float]:
    aligned = pd.concat(
        [portfolio_nav.rename("portfolio"), benchmark_nav.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return {}
    p = aligned["portfolio"].pct_change().dropna()
    b = aligned["benchmark"].pct_change().dropna()
    active = p - b
    tracking_error = active.std(ddof=1) * np.sqrt(252) if len(active) > 1 else 0.0
    information_ratio = (
        active.mean() / active.std(ddof=1) * np.sqrt(252)
        if len(active) > 1 and active.std(ddof=1) > 0
        else 0.0
    )
    beta = p.cov(b) / b.var(ddof=1) if len(p) > 1 and b.var(ddof=1) > 0 else np.nan
    return {
        f"{prefix}_total_return": float(
            aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1.0
        ),
        f"{prefix}_active_total_return": float(
            aligned["portfolio"].iloc[-1] / aligned["portfolio"].iloc[0]
            - aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0]
        ),
        f"{prefix}_tracking_error": float(tracking_error),
        f"{prefix}_information_ratio": float(information_ratio),
        f"{prefix}_beta": float(beta) if np.isfinite(beta) else float("nan"),
    }


def compare_benchmarks(
    portfolio_nav: pd.Series, benchmark_returns: pd.DataFrame | None
) -> dict[str, float]:
    if benchmark_returns is None or benchmark_returns.empty:
        return {}
    required = {"date", "benchmark", "total_return"}
    if not required.issubset(benchmark_returns):
        raise ValueError(f"Benchmark returns require columns {sorted(required)}")
    frame = benchmark_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    result: dict[str, float] = {}
    for name, group in frame.groupby("benchmark"):
        returns = (
            group.set_index("date")["total_return"]
            .sort_index()
            .reindex(portfolio_nav.index)
            .fillna(0.0)
        )
        benchmark_nav = (1.0 + returns).cumprod() * float(portfolio_nav.iloc[0])
        prefix = "benchmark_" + "".join(
            ch.lower() if ch.isalnum() else "_" for ch in str(name)
        ).strip("_")
        result.update(benchmark_comparison(portfolio_nav, benchmark_nav, prefix))
    return result


def factor_attribution(
    portfolio_nav: pd.Series, factor_returns: pd.DataFrame | None
) -> dict[str, float]:
    if factor_returns is None or factor_returns.empty:
        return {}
    required = {"date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"}
    if not required.issubset(factor_returns):
        raise ValueError(f"Factor returns require columns {sorted(required)}")
    factors = factor_returns.copy()
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    factors = factors.set_index("date").sort_index()
    portfolio = portfolio_nav.pct_change().rename("portfolio")
    joined = factors.join(portfolio, how="inner").dropna()
    if len(joined) < 30:
        return {"factor_observation_count": float(len(joined))}
    y = joined["portfolio"].to_numpy() - joined["RF"].to_numpy()
    names = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    x = np.column_stack([np.ones(len(joined)), joined[names].to_numpy()])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residual = y - fitted
    ss_total = float(np.sum((y - y.mean()) ** 2))
    result = {
        "factor_alpha_annualized": float(coefficients[0] * 252),
        "factor_r_squared": float(1.0 - np.sum(residual**2) / ss_total)
        if ss_total > 0
        else 0.0,
        "factor_observation_count": float(len(joined)),
    }
    result.update(
        {
            f"factor_beta_{name.lower()}": float(value)
            for name, value in zip(names, coefficients[1:], strict=True)
        }
    )
    return result
