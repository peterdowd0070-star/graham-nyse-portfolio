from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from graham_nyse.backtest.data import load_table
from graham_nyse.data.classification import (
    broad_sector_from_sic,
    company_domain_from_sic,
)
from graham_nyse.data.providers import HistoricalBundle


@dataclass(frozen=True)
class SharadarExportProvider:
    """Normalize licensed Sharadar exports without placing an API key in code.

    The ticker reference export must contain PERMATICKER. This adapter refuses
    ticker-only identity because tickers can be renamed or reused.
    """

    tickers_path: str | Path
    prices_path: str | Path
    actions_path: str | Path | None = None
    name: str = "sharadar"

    def load(self, start: str, end: str) -> HistoricalBundle:
        tickers = load_table(self.tickers_path)
        prices = load_table(self.prices_path)
        required_tickers = {
            "permaticker",
            "ticker",
            "name",
            "exchange",
            "category",
            "siccode",
            "firstpricedate",
            "lastpricedate",
            "isdelisted",
        }
        if missing := required_tickers - set(tickers.columns.str.lower()):
            raise ValueError(f"Sharadar TICKERS export is missing: {sorted(missing)}")
        tickers.columns = tickers.columns.str.lower()
        prices.columns = prices.columns.str.lower()
        if missing := {"permaticker", "date", "close", "volume"} - set(prices):
            raise ValueError(
                "Sharadar prices must be enriched with permanent identity; missing "
                f"{sorted(missing)}"
            )
        common = (
            tickers["category"]
            .astype(str)
            .str.contains(r"Domestic Common Stock", case=False, regex=True)
        )
        reference = tickers.loc[common].copy()
        reference["security_id"] = "SHARADAR:" + reference["permaticker"].astype(str)
        reference["issuer_id"] = reference["security_id"]
        reference["security_type"] = "common_stock"
        reference["identifier_type"] = "permaticker"
        reference["identifier_quality"] = "provider_permanent"
        reference["listing_start"] = pd.to_datetime(reference["firstpricedate"])
        reference["listing_end"] = pd.to_datetime(
            reference["lastpricedate"], errors="coerce"
        )
        reference["is_delisted"] = (
            reference["isdelisted"].astype(str).str.upper().isin(["Y", "TRUE", "1"])
        )
        reference["delisting_return"] = np.nan
        reference["company_domain"] = reference["siccode"].map(company_domain_from_sic)
        reference["sector"] = reference["siccode"].map(broad_sector_from_sic)
        master_columns = [
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
        ]
        master = reference[master_columns]

        price_frame = prices.copy()
        price_frame["date"] = pd.to_datetime(price_frame["date"])
        price_frame = price_frame.loc[
            price_frame["date"].between(
                pd.Timestamp(start) - pd.Timedelta(days=400), pd.Timestamp(end)
            )
        ]
        price_frame["security_id"] = "SHARADAR:" + price_frame["permaticker"].astype(
            str
        )
        price_frame = price_frame[
            ["date", "security_id", "close", "volume"]
            + (["sharesbas"] if "sharesbas" in price_frame else [])
        ].rename(columns={"sharesbas": "shares_outstanding"})
        price_frame["price_adjustment"] = "unadjusted"

        actions = pd.DataFrame(
            columns=["date", "security_id", "action_type", "value", "qualified"]
        )
        if self.actions_path is not None:
            raw = load_table(self.actions_path)
            raw.columns = raw.columns.str.lower()
            required = {"permaticker", "date", "action", "value"}
            if missing := required - set(raw):
                raise ValueError(f"Sharadar actions are missing: {sorted(missing)}")
            raw["security_id"] = "SHARADAR:" + raw["permaticker"].astype(str)
            raw["action_type"] = (
                raw["action"]
                .astype(str)
                .str.upper()
                .replace({"DIVIDENDS": "DIVIDEND", "SPLITS": "SPLIT"})
            )
            raw["qualified"] = False
            actions = raw.loc[
                raw["action_type"].isin(["DIVIDEND", "SPLIT"]),
                ["date", "security_id", "action_type", "value", "qualified"],
            ]
            actions["date"] = pd.to_datetime(actions["date"])
        return HistoricalBundle(self.name, master, price_frame, actions)
