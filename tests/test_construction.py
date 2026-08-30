import pandas as pd

from graham_nyse.config import load_config
from graham_nyse.portfolio.construction import cap_and_redistribute, construct_portfolio


def test_cap_and_redistribute_sums_to_one():
    weights = pd.Series([10.0] * 30)
    result = cap_and_redistribute(weights, 0.06)
    assert abs(result.sum() - 1.0) < 1e-10
    assert result.max() <= 0.06 + 1e-10


def test_fractional_shares_use_all_capital():
    cfg = load_config("config/strategy.yaml")
    frame = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(30)],
        "market_cap": [1e9] * 30,
        "graham_score": [0.5] * 30,
        "price": [100.0] * 30,
    })
    result = construct_portfolio(frame, cfg)
    assert abs(result["target_dollars"].sum() - cfg.portfolio.capital) < 1e-8
