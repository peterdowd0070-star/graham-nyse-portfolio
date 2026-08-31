from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCHANGE_CODES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}
NON_COMMON_NAME_PATTERN = (
    r"\b(?:ETF|ETN|FUND|PREFERRED|PFD|WARRANT|RIGHT|UNIT|DEPOSITARY|ADS|ADR|"
    r"NOTE|BOND|DEBENTURE|BENEFICIAL INTEREST)\b"
)


@dataclass
class NasdaqTraderClient:
    cache_dir: str | Path = "data/raw/nasdaq_trader"
    url: str = OTHER_LISTED_URL

    def fetch_other_listed(self, *, refresh: bool = False) -> pd.DataFrame:
        target = Path(self.cache_dir) / "otherlisted.txt"
        if target.exists() and not refresh:
            text = target.read_text(encoding="utf-8")
        else:
            response = httpx.get(self.url, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
            text = response.text
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return normalize_other_listed(text)


def normalize_other_listed(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text), sep="|")
    frame.columns = [str(column).strip() for column in frame.columns]
    symbol_column = "ACT Symbol"
    required = {symbol_column, "Security Name", "Exchange", "ETF", "Test Issue"}
    if missing := required - set(frame):
        raise ValueError(f"Nasdaq Trader otherlisted file is missing: {sorted(missing)}")
    frame = frame.loc[
        ~frame[symbol_column].astype(str).str.startswith("File Creation Time:")
    ].copy()
    frame["ticker"] = frame[symbol_column].astype(str).str.strip()
    frame["exchange"] = frame["Exchange"].map(EXCHANGE_CODES)
    frame["is_etf"] = frame["ETF"].astype(str).str.upper().eq("Y")
    frame["is_test_issue"] = frame["Test Issue"].astype(str).str.upper().eq("Y")
    non_common_name = frame["Security Name"].fillna("").str.contains(
        NON_COMMON_NAME_PATTERN, case=False, regex=True
    )
    frame["is_operating_common_candidate"] = (
        frame["exchange"].eq("NYSE")
        & ~frame["is_etf"]
        & ~frame["is_test_issue"]
        & ~non_common_name
    )
    frame["reference_source"] = "nasdaq_trader_otherlisted"
    return frame[
        [
            "ticker",
            "Security Name",
            "exchange",
            "is_etf",
            "is_test_issue",
            "is_operating_common_candidate",
            "reference_source",
        ]
    ].rename(columns={"Security Name": "security_name"})
