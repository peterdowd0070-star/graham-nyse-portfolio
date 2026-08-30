import numpy as np
import pandas as pd

from graham_nyse.config import load_config
from graham_nyse.portfolio.construction import cap_and_redistribute, construct_portfolio


def make_selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": [f"SEC{i:03d}" for i in range(30)],
            "ticker": [f"T{i}" for i in range(30)],
            "sector": [f"S{i % 6}" for i in range(30)],
            "market_cap": [1e9 + i * 1e7 for i in range(30)],
            "median_dollar_volume_60d": [2e6 + i * 1e5 for i in range(30)],
            "volatility_252d": [0.20 + i / 1000 for i in range(30)],
            "graham_score": [0.3 + i / 100 for i in range(30)],
            "price": [100.0] * 30,
        }
    )


def test_cap_and_redistribute_sums_to_one():
    weights = pd.Series([10.0] * 30)
    result = cap_and_redistribute(weights, 0.06)
    assert abs(result.sum() - 1.0) < 1e-10
    assert result.max() <= 0.06 + 1e-10


def test_every_weight_strategy_respects_constraints():
    cfg = load_config("config/strategy.yaml")
    selected = make_selected()
    returns = pd.DataFrame(
        np.random.default_rng(1).normal(0, 0.01, (260, 30)),
        columns=selected["security_id"],
    )
    for strategy in cfg.portfolio.weighting_strategies:
        result = construct_portfolio(selected, cfg, strategy, returns)
        assert abs(result["target_weight"].sum() - 1.0) < 1e-8
        assert result["target_weight"].max() <= cfg.portfolio.max_position_weight + 1e-8
        assert (
            result.groupby("sector")["target_weight"].sum().max()
            <= cfg.portfolio.max_sector_weight + 1e-8
        )
        assert abs(result["target_dollars"].sum() - cfg.portfolio.capital) < 1e-6
