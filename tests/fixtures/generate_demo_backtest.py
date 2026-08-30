from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
out = Path(__file__).parent
start, end = "2016-07-01", "2026-06-30"
dates = pd.bdate_range(start, end)
tickers = [f"DEMO{i:02d}" for i in range(45)]

price_rows = []
for i, ticker in enumerate(tickers):
    annual_mu = 0.045 + 0.0022 * i
    annual_sigma = 0.16 + 0.001 * (i % 8)
    daily = rng.normal(annual_mu / 252, annual_sigma / np.sqrt(252), len(dates))
    series = (18 + i) * np.exp(np.cumsum(daily))
    price_rows.extend({"date": d, "ticker": ticker, "adjusted_close": p} for d, p in zip(dates, series, strict=True))
prices = pd.DataFrame(price_rows)
prices.to_csv(out / "demo_prices_10y.csv", index=False)

as_of_dates = pd.DatetimeIndex([pd.Timestamp(start)]).append(pd.date_range(start, end, freq="QE")).unique().sort_values()
feature_rows = []
for d in as_of_dates:
    live = prices.loc[prices["date"] <= d].groupby("ticker", as_index=False).tail(1).set_index("ticker")["adjusted_close"]
    for i, ticker in enumerate(tickers):
        quality = 0.05 + i / 1000 + rng.normal(0, 0.003)
        market_cap = (0.6e9 + i * 0.11e9) * (1 + (d.year - 2016) * 0.035)
        cfo = market_cap * (0.055 + i / 5000)
        ni = market_cap * (0.04 + i / 7000)
        feature_rows.append({
            "as_of_date": d,
            "ticker": ticker,
            "price": float(live.get(ticker, 20 + i)),
            "market_cap": market_cap,
            "median_dollar_volume_60d": 2e6 + i * 1e5,
            "positive_earnings_years": 10 if i != 0 else 7,
            "earnings_history_years": 10,
            "interest_coverage": 4.0 + i / 10,
            "cfo": cfo,
            "normalized_net_income": ni,
            "fcf": cfo * 0.72,
            "net_debt": market_cap * max(0.05, 0.25 - i / 500),
            "equity": market_cap * (0.5 + i / 1000),
            "assets": market_cap * 1.3,
            "roa": quality,
            "accruals": 0.03 - i / 5000,
            "earnings_stability": 0.08 - min(i, 40) / 1000,
        })
pd.DataFrame(feature_rows).to_csv(out / "demo_features_10y.csv", index=False)
