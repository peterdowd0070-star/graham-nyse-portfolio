import numpy as np
import pandas as pd

from graham_nyse.backtest.engine import run_backtest
from graham_nyse.config import load_config


def make_features() -> pd.DataFrame:
    dates = pd.to_datetime(["2016-01-04", "2016-06-30", "2016-12-30", "2017-06-30"])
    rows = []
    for date in dates:
        for i in range(30):
            rows.append({
                "as_of_date": date,
                "ticker": f"T{i:02d}",
                "price": 20.0 + i,
                "market_cap": 1e9 + i * 1e7,
                "median_dollar_volume_60d": 2e6,
                "positive_earnings_years": 10,
                "earnings_history_years": 10,
                "interest_coverage": 5.0,
                "cfo": 1e8 + i * 1e6,
                "normalized_net_income": 8e7 + i * 1e6,
                "fcf": 7e7 + i * 1e6,
                "net_debt": 1e8,
                "equity": 5e8,
                "assets": 1e9,
                "roa": 0.08 + i / 10000,
                "accruals": 0.01,
                "earnings_stability": 0.02,
            })
    return pd.DataFrame(rows)


def make_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-04", "2017-12-29")
    rows = []
    for i in range(30):
        base = 20.0 + i
        growth = np.linspace(1.0, 1.2 + i / 1000, len(dates))
        rows.extend({"date": d, "ticker": f"T{i:02d}", "adjusted_close": base * g} for d, g in zip(dates, growth, strict=True))
    return pd.DataFrame(rows)


def test_backtest_constructs_and_rebalances():
    cfg = load_config("config/strategy.yaml")
    result = run_backtest(make_features(), make_prices(), cfg, "2016-01-04", "2017-12-29", transaction_cost_bps=0)
    assert not result.nav.empty
    assert not result.holdings.empty
    assert result.nav.iloc[-1]["nav"] > cfg.portfolio.capital
    assert {"initial_construction", "quarterly_rebalance", "full_reconstruction"}.issubset(set(result.holdings["run_type"]))
    assert abs(result.holdings.groupby("date")["weight"].sum().iloc[-1] - 1.0) < 1e-8
    assert "cagr" in result.metrics
