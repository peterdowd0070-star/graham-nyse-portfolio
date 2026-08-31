from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from graham_nyse.backtest.data import load_table
from graham_nyse.data.providers import HistoricalBundle


@dataclass(frozen=True)
class NorgateExportProvider:
    """Load audited Norgate exports while preserving provider-stable asset IDs."""

    security_history_path: str | Path
    prices_path: str | Path
    actions_path: str | Path
    name: str = "norgate"

    def load(self, start: str, end: str) -> HistoricalBundle:
        master = load_table(self.security_history_path)
        prices = load_table(self.prices_path)
        actions = load_table(self.actions_path)
        master.columns = master.columns.str.lower()
        prices.columns = prices.columns.str.lower()
        actions.columns = actions.columns.str.lower()
        master_required = {
            "asset_id",
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
        }
        if missing := master_required - set(master):
            raise ValueError(f"Norgate security history is missing: {sorted(missing)}")
        if missing := {"date", "asset_id", "close_unadjusted", "volume"} - set(prices):
            raise ValueError(f"Norgate price export is missing: {sorted(missing)}")
        if missing := {"date", "asset_id", "action_type", "value"} - set(actions):
            raise ValueError(f"Norgate action export is missing: {sorted(missing)}")

        master["security_id"] = "NORGATE:" + master["asset_id"].astype(str)
        master = master.drop(columns=["asset_id"])
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.loc[
            prices["date"].between(
                pd.Timestamp(start) - pd.Timedelta("400D"), pd.Timestamp(end)
            )
        ].copy()
        prices["security_id"] = "NORGATE:" + prices["asset_id"].astype(str)
        prices = prices.rename(columns={"close_unadjusted": "close"})[
            ["date", "security_id", "close", "volume"]
            + (["shares_outstanding"] if "shares_outstanding" in prices else [])
        ]
        actions["security_id"] = "NORGATE:" + actions["asset_id"].astype(str)
        actions["action_type"] = actions["action_type"].astype(str).str.upper()
        actions["qualified"] = actions.get("qualified", False)
        actions = actions[["date", "security_id", "action_type", "value", "qualified"]]
        return HistoricalBundle(self.name, master, prices, actions)
