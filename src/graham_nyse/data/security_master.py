from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_SECURITY_COLUMNS = {
    "security_id",
    "issuer_id",
    "ticker",
    "exchange",
    "security_type",
    "company_domain",
    "sector",
    "listing_start",
    "listing_end",
    "delisting_return",
}
VALID_DOMAINS = {"ordinary", "bank", "insurer", "reit"}


@dataclass(frozen=True)
class SecurityMaster:
    frame: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> SecurityMaster:
        missing = REQUIRED_SECURITY_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Security master is missing columns: {sorted(missing)}")
        out = frame.copy()
        out["security_id"] = out["security_id"].astype(str)
        out["issuer_id"] = out["issuer_id"].astype(str)
        out["listing_start"] = pd.to_datetime(
            out["listing_start"], errors="raise"
        ).dt.normalize()
        out["listing_end"] = pd.to_datetime(
            out["listing_end"], errors="coerce"
        ).dt.normalize()
        out["delisting_return"] = pd.to_numeric(
            out["delisting_return"], errors="coerce"
        )
        if out["security_id"].duplicated().any():
            raise ValueError("security_id must be unique and permanent")
        invalid_domains = set(out["company_domain"].dropna()) - VALID_DOMAINS
        if invalid_domains:
            raise ValueError(f"Unsupported company domains: {sorted(invalid_domains)}")
        if (
            out["listing_end"].notna() & (out["listing_end"] < out["listing_start"])
        ).any():
            raise ValueError("listing_end precedes listing_start")
        return cls(out.sort_values("security_id").reset_index(drop=True))

    def active_as_of(
        self, as_of: str | pd.Timestamp, exchange: str = "NYSE"
    ) -> pd.DataFrame:
        day = pd.Timestamp(as_of).normalize()
        active = (
            self.frame["listing_start"].le(day)
            & (self.frame["listing_end"].isna() | self.frame["listing_end"].ge(day))
            & self.frame["exchange"].str.upper().eq(exchange.upper())
            & self.frame["security_type"].eq("common_stock")
        )
        return self.frame.loc[active].copy()

    def delistings_on(self, day: str | pd.Timestamp) -> pd.DataFrame:
        target = pd.Timestamp(day).normalize()
        return self.frame.loc[self.frame["listing_end"].eq(target)].copy()

    def require_complete_delisting_returns(self) -> list[str]:
        inactive = self.frame["listing_end"].notna()
        missing = self.frame.loc[
            inactive & ~np.isfinite(self.frame["delisting_return"]), "security_id"
        ]
        return missing.astype(str).tolist()


def security_master_from_listing_history(
    listing_history: pd.DataFrame,
    classifications: pd.DataFrame,
    delistings: pd.DataFrame,
) -> SecurityMaster:
    """Build a dated master without using today's ticker list as historical membership."""
    base = listing_history.merge(
        classifications, on="security_id", how="left", validate="one_to_one"
    )
    base = base.merge(
        delistings[["security_id", "listing_end", "delisting_return"]],
        on="security_id",
        how="left",
        suffixes=("", "_delisting"),
        validate="one_to_one",
    )
    if "listing_end_delisting" in base:
        base["listing_end"] = base["listing_end_delisting"].combine_first(
            base.get("listing_end")
        )
        base = base.drop(columns=["listing_end_delisting"])
    return SecurityMaster.from_frame(base)


def security_master_from_crsp(
    stock_names: pd.DataFrame,
    delistings: pd.DataFrame,
    classifications: pd.DataFrame,
) -> SecurityMaster:
    """Normalize CRSP-style name history and delisting files.

    Expected CRSP columns are PERMNO, PERMCO, TICKER, EXCHCD, SHRCD, NAMEDT,
    NAMEENDT and, for delistings, DLSTDT and DLRET. Classification rows are
    keyed by PERMNO and provide company_domain and sector.
    """
    name_required = {
        "PERMNO",
        "PERMCO",
        "TICKER",
        "EXCHCD",
        "SHRCD",
        "NAMEDT",
        "NAMEENDT",
    }
    delist_required = {"PERMNO", "DLSTDT", "DLRET"}
    class_required = {"PERMNO", "company_domain", "sector"}
    if missing := name_required - set(stock_names):
        raise ValueError(f"CRSP stock-name history is missing: {sorted(missing)}")
    if missing := delist_required - set(delistings):
        raise ValueError(f"CRSP delistings are missing: {sorted(missing)}")
    if missing := class_required - set(classifications):
        raise ValueError(f"CRSP classifications are missing: {sorted(missing)}")
    names = stock_names.loc[
        stock_names["EXCHCD"].eq(1) & stock_names["SHRCD"].isin([10, 11])
    ].copy()
    names = (
        names.sort_values(["PERMNO", "NAMEENDT"])
        .groupby("PERMNO", as_index=False)
        .agg(
            PERMCO=("PERMCO", "last"),
            TICKER=("TICKER", "last"),
            listing_start=("NAMEDT", "min"),
            listing_end_name=("NAMEENDT", "max"),
        )
    )
    latest_delisting = (
        delistings.sort_values(["PERMNO", "DLSTDT"])
        .groupby("PERMNO", as_index=False)
        .tail(1)
    )
    out = names.merge(
        latest_delisting[["PERMNO", "DLSTDT", "DLRET"]], on="PERMNO", how="left"
    )
    out = out.merge(
        classifications[list(class_required)],
        on="PERMNO",
        how="left",
        validate="one_to_one",
    )
    out["security_id"] = "CRSP:" + out["PERMNO"].astype(int).astype(str)
    out["issuer_id"] = "CRSPCO:" + out["PERMCO"].astype(int).astype(str)
    out["ticker"] = out["TICKER"]
    out["exchange"] = "NYSE"
    out["security_type"] = "common_stock"
    out["listing_end"] = pd.to_datetime(out["DLSTDT"], errors="coerce")
    out["delisting_return"] = pd.to_numeric(out["DLRET"], errors="coerce")
    # A CRSP name interval ending is not itself a delisting; only DLSTDT closes
    # the permanent security record.
    columns = sorted(REQUIRED_SECURITY_COLUMNS)
    return SecurityMaster.from_frame(out[columns])
