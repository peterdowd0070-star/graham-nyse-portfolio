import numpy as np
import pandas as pd

from graham_nyse.backtest.metrics import compare_benchmarks, factor_attribution


def test_multiple_benchmarks_and_factor_attribution():
    dates = pd.bdate_range("2020-01-01", periods=80)
    portfolio = pd.Series((1.001 ** np.arange(80)) * 100, index=dates)
    benchmarks = pd.DataFrame(
        [
            {"date": day, "benchmark": name, "total_return": daily}
            for name, daily in [("TOTAL_US", 0.0008), ("VALUE", 0.0009)]
            for day in dates
        ]
    )
    factors = pd.DataFrame(
        {
            "date": dates,
            "MKT_RF": 0.0007,
            "SMB": 0.0001,
            "HML": 0.0002,
            "RMW": 0.0001,
            "CMA": 0.0001,
            "RF": 0.00005,
        }
    )
    comparison = compare_benchmarks(portfolio, benchmarks)
    attribution = factor_attribution(portfolio, factors)
    assert "benchmark_total_us_tracking_error" in comparison
    assert "benchmark_value_tracking_error" in comparison
    assert attribution["factor_observation_count"] >= 30
    assert "factor_alpha_annualized" in attribution
