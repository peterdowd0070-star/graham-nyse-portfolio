from __future__ import annotations

import math
from typing import Any

import pandas as pd

from graham_nyse.config import StrategyConfig


def validate_portfolio(portfolio: pd.DataFrame, cfg: StrategyConfig) -> dict[str, Any]:
    errors: list[str] = []
    if portfolio.empty:
        errors.append("portfolio_is_empty")
    if not math.isclose(float(portfolio["target_weight"].sum()), 1.0, rel_tol=0, abs_tol=1e-8):
        errors.append("weights_do_not_sum_to_one")
    if (portfolio["target_weight"] > cfg.portfolio.max_position_weight + 1e-8).any():
        errors.append("position_cap_exceeded")
    if portfolio[["price", "target_weight", "target_shares"]].isna().any().any():
        errors.append("missing_portfolio_values")
    return {"passed": not errors, "errors": errors}
