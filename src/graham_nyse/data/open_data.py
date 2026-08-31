from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from graham_nyse.data.providers import HistoricalBundle

OPEN_DATA_SOURCE_AUDIT = (
    {
        "source": "SEC EDGAR",
        "library": "httpx/direct API and bulk archives",
        "role": "canonical filing vintages, issuer identity evidence and terminal-event filings",
        "priority": 10,
        "survivorship_free": False,
        "reason": "authoritative filings do not themselves provide security returns",
    },
    {
        "source": "Alpha Vantage LISTING_STATUS",
        "library": "httpx/direct API",
        "role": "dated active and delisted symbol snapshots",
        "priority": 10,
        "survivorship_free": False,
        "reason": "no CRSP-style permanent security ID or authoritative delisting return",
    },
    {
        "source": "Nasdaq Trader Symbol Directory",
        "library": "httpx/direct public file",
        "role": "current NYSE exchange and security-type reference",
        "priority": 10,
        "survivorship_free": False,
        "reason": "current reference only; snapshots must be archived prospectively",
    },
    {
        "source": "Yahoo Finance",
        "library": "yfinance",
        "role": "raw research prices, dividends and splits for a supplied symbol list",
        "priority": 10,
        "survivorship_free": False,
        "reason": "not a historical universe/security master and inactive coverage is not guaranteed",
    },
    {
        "source": "SimFin",
        "library": "simfin",
        "role": "normalized fundamentals and share prices",
        "priority": 30,
        "survivorship_free": False,
        "reason": "public documentation does not certify complete historical NYSE membership and terminal returns",
    },
    {
        "source": "Stooq",
        "library": "direct download",
        "role": "independent price audit, never silent price replacement",
        "priority": 10,
        "survivorship_free": False,
        "reason": "no audited permanent-ID security master or delisting-return table",
    },
    {
        "source": "pandas-datareader",
        "library": "pandas-datareader",
        "role": "client library",
        "priority": 99,
        "survivorship_free": False,
        "reason": "current maintained surface removed Yahoo, Stooq and other securities readers",
    },
    {
        "source": "OpenBB",
        "library": "openbb",
        "role": "provider aggregation and SEC reconciliation",
        "priority": 40,
        "survivorship_free": False,
        "reason": "coverage and identity quality inherit from the selected upstream provider",
    },
    {
        "source": "WRDS/CRSP",
        "library": "wrds",
        "role": "permanent IDs, active/inactive histories, returns and delisting returns",
        "priority": 90,
        "survivorship_free": True,
        "reason": "strict reference dataset; licensed rather than open",
    },
)


@dataclass
class AlphaVantageListingClient:
    api_key: str | None = None
    cache_dir: str | Path = "data/raw/alpha_vantage"
    base_url: str = "https://www.alphavantage.co/query"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Set ALPHA_VANTAGE_API_KEY for LISTING_STATUS snapshots")
        self.cache_dir = Path(self.cache_dir)

    def fetch_snapshot(
        self, as_of: str | pd.Timestamp, state: str = "active", refresh: bool = False
    ) -> pd.DataFrame:
        if state not in {"active", "delisted"}:
            raise ValueError("state must be active or delisted")
        day = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        target = Path(self.cache_dir) / f"listing_status_{state}_{day}.csv"
        if target.exists() and not refresh:
            frame = pd.read_csv(target)
        else:
            response = httpx.get(
                self.base_url,
                params={
                    "function": "LISTING_STATUS",
                    "date": day,
                    "state": state,
                    "apikey": self.api_key,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            text = response.text
            if text.lstrip().startswith("{"):
                raise RuntimeError(f"Alpha Vantage returned an API error: {text[:300]}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            frame = pd.read_csv(io.StringIO(text))
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["snapshot_date"] = pd.Timestamp(day)
        frame["requested_state"] = state
        return frame


def open_listing_snapshot_requests(
    start: str | pd.Timestamp, end: str | pd.Timestamp
) -> list[tuple[pd.Timestamp, str]]:
    """Use only the snapshots needed by the agreed semiannual reconstruction.

    Active membership is captured immediately before the first session and at
    each June/December reconstruction. One final delisted query supplies the
    provider's exit catalog without doubling every historical request.
    """

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")
    initial = start_ts - pd.offsets.Day(1)
    semiannual = pd.date_range(initial, end_ts, freq="2QE-DEC").normalize()
    active_dates = sorted({initial, *semiannual.tolist()})
    return [(day, "active") for day in active_dates] + [(end_ts, "delisted")]


def build_alpha_vantage_research_master(
    snapshots: pd.DataFrame,
    classifications: pd.DataFrame,
    exchange: str = "NYSE",
) -> pd.DataFrame:
    """Build a ticker-interval research master, never a certified permanent-ID master."""

    required = {"symbol", "name", "exchange", "assetType", "ipoDate", "delistingDate"}
    if missing := required - set(snapshots):
        raise ValueError(f"Alpha Vantage snapshots are missing: {sorted(missing)}")
    if missing := {"symbol", "company_domain", "sector"} - set(classifications):
        raise ValueError(f"Classifications are missing: {sorted(missing)}")
    rows = snapshots.copy()
    rows = rows.loc[
        rows["exchange"].astype(str).str.upper().eq(exchange.upper())
        & rows["assetType"].astype(str).str.lower().eq("stock")
    ].copy()
    rows["listing_start"] = pd.to_datetime(rows["ipoDate"], errors="coerce")
    rows["listing_end"] = pd.to_datetime(rows["delistingDate"], errors="coerce")
    rows = rows.sort_values(["symbol", "listing_start", "listing_end"]).drop_duplicates(
        ["symbol", "listing_start", "listing_end"], keep="last"
    )
    rows = rows.merge(classifications, on="symbol", how="left", validate="many_to_one")
    if rows[["company_domain", "sector"]].isna().any().any():
        missing_symbols = rows.loc[
            rows[["company_domain", "sector"]].isna().any(axis=1), "symbol"
        ].unique()
        raise ValueError(
            f"Domain/sector classification is missing for {len(missing_symbols)} symbols"
        )
    start_token = rows["listing_start"].dt.strftime("%Y%m%d").fillna("unknown")
    rows["security_id"] = (
        "AV:" + exchange.upper() + ":" + rows["symbol"].astype(str) + ":" + start_token
    )
    rows["issuer_id"] = rows["security_id"]
    rows["ticker"] = rows["symbol"]
    rows["security_type"] = "common_stock"
    rows["is_delisted"] = rows["listing_end"].notna()
    rows["delisting_return"] = np.nan
    rows["identifier_type"] = "ticker_plus_listing_date"
    rows["identifier_quality"] = "symbol_interval"
    rows["universe_source"] = "alpha_vantage_listing_status"
    rows["data_provider"] = "yahoo_research"
    columns = [
        "security_id",
        "issuer_id",
        "ticker",
        "exchange",
        "security_type",
        "company_domain",
        "sector",
        "listing_start",
        "listing_end",
        "is_delisted",
        "delisting_return",
        "identifier_type",
        "identifier_quality",
        "universe_source",
        "data_provider",
    ]
    return rows[columns].reset_index(drop=True)


@dataclass
class YahooResearchProvider:
    security_master: pd.DataFrame
    threads: bool = False

    name: str = "yahoo_research"

    @staticmethod
    def _yahoo_symbol(ticker: str) -> str:
        return ticker.replace(".", "-")

    def load(self, start: str, end: str) -> HistoricalBundle:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance to use YahooResearchProvider") from exc
        price_rows: list[pd.DataFrame] = []
        action_rows: list[pd.DataFrame] = []
        failures: list[str] = []
        for record in self.security_master.itertuples(index=False):
            symbol = self._yahoo_symbol(str(record.ticker))
            history = yf.download(
                symbol,
                start=start,
                end=(pd.Timestamp(end) + pd.offsets.Day(1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=True,
                repair=False,
                keepna=True,
                progress=False,
                threads=self.threads,
                multi_level_index=False,
            )
            if history.empty:
                failures.append(str(record.security_id))
                continue
            history = history.reset_index().rename(
                columns={"Date": "date", "Close": "close", "Volume": "volume"}
            )
            history["security_id"] = str(record.security_id)
            prices = history[["date", "security_id", "close", "volume"]].copy()
            prices = prices.dropna(subset=["close"])
            prices["price_adjustment"] = "raw"
            prices["data_provider"] = self.name
            price_rows.append(prices)
            for source, action_type in (
                ("Dividends", "DIVIDEND"),
                ("Stock Splits", "SPLIT"),
            ):
                if source not in history:
                    continue
                events = history.loc[
                    pd.to_numeric(history[source], errors="coerce").ne(0),
                    ["date", source],
                ].copy()
                if events.empty:
                    continue
                events["security_id"] = str(record.security_id)
                events["action_type"] = action_type
                events["value"] = pd.to_numeric(events[source], errors="raise")
                events["qualified"] = False
                events["data_provider"] = self.name
                action_rows.append(
                    events[
                        [
                            "date",
                            "security_id",
                            "action_type",
                            "value",
                            "qualified",
                            "data_provider",
                        ]
                    ]
                )
        if failures:
            raise RuntimeError(
                "Yahoo returned no history for security IDs: "
                + ", ".join(failures[:20])
            )
        prices = (
            pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()
        )
        actions = (
            pd.concat(action_rows, ignore_index=True)
            if action_rows
            else pd.DataFrame(
                columns=[
                    "date",
                    "security_id",
                    "action_type",
                    "value",
                    "qualified",
                    "data_provider",
                ]
            )
        )
        master = self.security_master.copy()
        master["data_provider"] = self.name
        return HistoricalBundle(self.name, master, prices, actions)
