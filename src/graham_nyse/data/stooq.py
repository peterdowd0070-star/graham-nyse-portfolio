from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


@dataclass
class StooqPriceAuditClient:
    cache_dir: str | Path = "data/raw/stooq"
    base_url: str = "https://stooq.com/q/d/l/"

    def fetch_daily(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        symbol = ticker.lower().replace(".", "-") + ".us"
        start_token = pd.Timestamp(start).strftime("%Y%m%d")
        end_token = pd.Timestamp(end).strftime("%Y%m%d")
        target = Path(self.cache_dir) / f"{symbol}_{start_token}_{end_token}.csv"
        if target.exists() and not refresh:
            text = target.read_text(encoding="utf-8")
        else:
            response = httpx.get(
                self.base_url,
                params={"s": symbol, "d1": start_token, "d2": end_token, "i": "d"},
                timeout=60.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            text = response.text
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        frame = pd.read_csv(io.StringIO(text))
        frame.columns = [str(column).lower() for column in frame.columns]
        if missing := {"date", "close"} - set(frame):
            raise ValueError(f"Stooq response is missing: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["ticker"] = ticker
        frame["audit_provider"] = "stooq"
        return frame


def audit_close_prices(
    primary: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    relative_tolerance: float = 0.02,
) -> dict[str, object]:
    for name, frame in (("Primary", primary), ("Audit", audit)):
        if missing := {"date", "close"} - set(frame):
            raise ValueError(f"{name} prices are missing: {sorted(missing)}")
    left = primary[["date", "close"]].rename(columns={"close": "primary_close"})
    right = audit[["date", "close"]].rename(columns={"close": "audit_close"})
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    joined = left.merge(right, on="date", how="inner", validate="one_to_one")
    if joined.empty:
        return {"overlap_rows": 0, "mismatch_rows": 0, "mismatch_rate": None}
    denominator = joined["audit_close"].abs().replace(0, np.nan)
    relative_error = (joined["primary_close"] - joined["audit_close"]).abs() / denominator
    mismatch = relative_error.gt(relative_tolerance) | relative_error.isna()
    return {
        "overlap_rows": len(joined),
        "mismatch_rows": int(mismatch.sum()),
        "mismatch_rate": float(mismatch.mean()),
        "maximum_relative_error": float(relative_error.max()),
        "relative_tolerance": relative_tolerance,
    }
