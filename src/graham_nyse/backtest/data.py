from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_feature_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_price_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def download_adjusted_prices(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download adjusted historical prices in batches.

    This adapter is suitable for research prototyping. A production backtest should
    use a licensed point-in-time source that includes delisted securities.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required only for live price downloads; install project dependencies") from exc

    frames: list[pd.DataFrame] = []
    clean = sorted({ticker.replace(".", "-") for ticker in tickers})
    for offset in range(0, len(clean), batch_size):
        batch = clean[offset : offset + batch_size]
        raw = yf.download(batch, start=start, end=end, auto_adjust=False, actions=False, progress=False, group_by="column")
        if raw.empty:
            continue
        adjusted = raw["Adj Close"] if "Adj Close" in raw else raw["Close"]
        if isinstance(adjusted, pd.Series):
            adjusted = adjusted.to_frame(batch[0])
        long = adjusted.rename_axis("date").stack(future_stack=True).rename("adjusted_close").reset_index()
        long.columns = ["date", "ticker", "adjusted_close"]
        long["ticker"] = long["ticker"].str.replace("-", ".", regex=False)
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "adjusted_close"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["adjusted_close"])
