from __future__ import annotations

import numpy as np
import pandas as pd

from graham_nyse.config import StrategyConfig


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.replace([np.inf, -np.inf], np.nan).rank(pct=True, method="average")
    return ranked if higher_is_better else 1.0 - ranked


def apply_eligibility(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    f, u = cfg.fundamentals, cfg.universe
    mask = (
        df["market_cap"].ge(u.min_market_cap)
        & df["price"].ge(u.min_price)
        & df["median_dollar_volume_60d"].ge(u.min_median_dollar_volume_60d)
        & df["positive_earnings_years"].ge(f.min_positive_years)
        & df["earnings_history_years"].ge(f.history_years)
        & df["interest_coverage"].ge(f.min_interest_coverage)
    )
    if f.require_positive_cfo:
        mask &= df["cfo"].gt(0)
    result = df.copy()
    result["eligible"] = mask.fillna(False)
    return result


def calculate_scores(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["earnings_yield"] = out["normalized_net_income"] / out["market_cap"]
    out["cfo_yield"] = out["cfo"] / out["market_cap"]
    out["fcf_yield"] = out["fcf"] / (out["market_cap"] + out["net_debt"].clip(lower=0))
    out["book_to_market"] = out["equity"] / out["market_cap"]
    out["leverage"] = out["net_debt"] / out["assets"]

    v = cfg.valuation.value_weights
    q = cfg.valuation.quality_weights
    out["value_score"] = (
        v["earnings_yield"] * percentile(out["earnings_yield"])
        + v["cfo_yield"] * percentile(out["cfo_yield"])
        + v["fcf_yield"] * percentile(out["fcf_yield"])
        + v["book_to_market"] * percentile(out["book_to_market"])
    )
    out["quality_score"] = (
        q["roa"] * percentile(out["roa"])
        + q["accrual_quality"] * percentile(out["accruals"], higher_is_better=False)
        + q["leverage"] * percentile(out["leverage"], higher_is_better=False)
        + q["interest_coverage"] * percentile(out["interest_coverage"])
        + q["earnings_stability"] * percentile(out["earnings_stability"], higher_is_better=False)
    )
    out["stability_score"] = percentile(out["earnings_stability"], higher_is_better=False)
    w = cfg.valuation.score_weights
    eps = 1e-6
    out["graham_score"] = np.exp(
        w["value"] * np.log(out["value_score"].clip(lower=eps))
        + w["quality"] * np.log(out["quality_score"].clip(lower=eps))
        + w["stability"] * np.log(out["stability_score"].clip(lower=eps))
    )
    out.loc[~out["eligible"], "graham_score"] = np.nan
    out["rank"] = out["graham_score"].rank(ascending=False, method="first")
    return out.sort_values("rank")
