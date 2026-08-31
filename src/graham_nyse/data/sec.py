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
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSION_FILE_URL = "https://data.sec.gov/submissions/{name}"


class SecClient:
    def __init__(
        self, user_agent: str, cache_dir: str | Path = ".cache/sec", delay: float = 0.12
    ):
        if "@" not in user_agent:
            raise ValueError(
                "SEC user_agent must identify an application and contact email"
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.client = httpx.Client(
            timeout=45,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
        )

    def _get_json(
        self, url: str, cache_name: str, refresh: bool = False
    ) -> dict[str, Any]:
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
        return self._get_json(
            SEC_FACTS_URL.format(cik=int(cik)),
            f"companyfacts_{int(cik):010d}.json",
            refresh,
        )

    def submissions(self, cik: int, refresh: bool = False) -> dict[str, Any]:
        return self._get_json(
            SEC_SUBMISSIONS_URL.format(cik=int(cik)),
            f"submissions_{int(cik):010d}.json",
            refresh,
        )

    def submissions_all(self, cik: int, refresh: bool = False) -> dict[str, Any]:
        """Return recent and archived submission metadata as one immutable list."""
        payload = self.submissions(cik, refresh)
        recent = pd.DataFrame(payload.get("filings", {}).get("recent", {}))
        pieces = [recent] if not recent.empty else []
        for descriptor in payload.get("filings", {}).get("files", []):
            name = descriptor.get("name")
            if not name:
                continue
            archive = self._get_json(
                SEC_SUBMISSION_FILE_URL.format(name=name), name, refresh
            )
            frame = pd.DataFrame(archive)
            if not frame.empty:
                pieces.append(frame)
        combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        if not combined.empty and "accessionNumber" in combined:
            combined = combined.drop_duplicates("accessionNumber", keep="first")
        result = dict(payload)
        result["filings"] = {"recent": combined.to_dict(orient="list"), "files": []}
        return result


def build_nyse_universe(
    tickers: pd.DataFrame,
    exchange: str,
    exclude_name_patterns: list[str],
    include_adrs: bool = False,
) -> pd.DataFrame:
    df = tickers.loc[tickers["exchange"].str.upper().eq(exchange.upper())].copy()
    pattern = (
        re.compile("|".join(exclude_name_patterns), flags=re.IGNORECASE)
        if exclude_name_patterns
        else None
    )
    if pattern:
        df = df.loc[~df["name"].fillna("").str.contains(pattern)]
    if not include_adrs:
        df = df.loc[
            ~df["name"]
            .fillna("")
            .str.contains(r"ADR|ADS|DEPOSITARY", case=False, regex=True)
        ]
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    return (
        df.dropna(subset=["ticker", "cik"])
        .drop_duplicates("ticker")
        .reset_index(drop=True)
    )
