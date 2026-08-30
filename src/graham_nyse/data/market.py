from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def normalize_yahoo_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")


def fetch_market_snapshot(tickers: Iterable[str], period: str = "6mo") -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for ticker in tickers:
        symbol = normalize_yahoo_ticker(ticker)
        obj = yf.Ticker(symbol)
        hist = obj.history(period=period, auto_adjust=False)
        if hist.empty:
            continue
        close = float(hist["Close"].dropna().iloc[-1])
        dollar_volume = (hist["Close"] * hist["Volume"]).dropna().tail(60)
        fast = obj.fast_info
        market_cap = fast.get("market_cap")
        rows.append({"ticker": ticker, "price": close, "market_cap": float(market_cap) if market_cap is not None else np.nan, "median_dollar_volume_60d": float(dollar_volume.median())})
    return pd.DataFrame(rows)
