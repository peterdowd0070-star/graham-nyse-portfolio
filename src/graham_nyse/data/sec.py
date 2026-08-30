from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


class SecClient:
    def __init__(self, user_agent: str, cache_dir: str | Path = ".cache/sec", delay: float = 0.12):
        if "@" not in user_agent:
            raise ValueError("SEC user_agent must identify an application and contact email")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.client = httpx.Client(timeout=45, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, follow_redirects=True)

    def _get_json(self, url: str, cache_name: str, refresh: bool = False) -> dict[str, Any]:
        path = self.cache_dir / cache_name
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        response = self.client.get(url)
        response.raise_for_status()
        payload = response.json()
        path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(self.delay)
        return payload

    def exchange_tickers(self, refresh: bool = False) -> pd.DataFrame:
        raw = self._get_json(SEC_TICKERS_URL, "company_tickers_exchange.json", refresh)
        return pd.DataFrame(raw["data"], columns=raw["fields"])

    def company_facts(self, cik: int, refresh: bool = False) -> dict[str, Any]:
        return self._get_json(SEC_FACTS_URL.format(cik=int(cik)), f"companyfacts_{int(cik):010d}.json", refresh)


def build_nyse_universe(tickers: pd.DataFrame, exchange: str, exclude_name_patterns: list[str], include_adrs: bool = False) -> pd.DataFrame:
    df = tickers.loc[tickers["exchange"].str.upper().eq(exchange.upper())].copy()
    pattern = re.compile("|".join(exclude_name_patterns), flags=re.IGNORECASE) if exclude_name_patterns else None
    if pattern:
        df = df.loc[~df["name"].fillna("").str.contains(pattern)]
    if not include_adrs:
        df = df.loc[~df["name"].fillna("").str.contains(r"ADR|ADS|DEPOSITARY", case=False, regex=True)]
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    return df.dropna(subset=["ticker", "cik"]).drop_duplicates("ticker").reset_index(drop=True)
