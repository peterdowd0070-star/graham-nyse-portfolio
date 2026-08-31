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
        interval_key = ["security_id", "listing_start", "listing_end"]
        if out.duplicated(interval_key).any():
            raise ValueError("security history contains duplicate dated intervals")
        for security_id, history in out.groupby("security_id"):
            ordered = history.sort_values("listing_start")
            previous_end: pd.Timestamp | None = None
            for row in ordered.itertuples(index=False):
                if previous_end is not None and row.listing_start <= previous_end:
                    raise ValueError(
                        f"security history intervals overlap for {security_id}"
                    )
                previous_end = row.listing_end if pd.notna(row.listing_end) else None
        invalid_domains = set(out["company_domain"].dropna()) - VALID_DOMAINS
        if invalid_domains:
            raise ValueError(f"Unsupported company domains: {sorted(invalid_domains)}")
        if (
            out["listing_end"].notna() & (out["listing_end"] < out["listing_start"])
        ).any():
            raise ValueError("listing_end precedes listing_start")
        if "is_delisted" not in out:
            # Backward compatibility for the original one-row security contract.
            out["is_delisted"] = out["listing_end"].notna()
        out["is_delisted"] = out["is_delisted"].fillna(False).astype(bool)
        return cls(
            out.sort_values(["security_id", "listing_start"]).reset_index(drop=True)
        )

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
        inactive = self.frame["is_delisted"]
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
    has_dated_classification = {"company_domain", "sector"}.issubset(stock_names)
    if not has_dated_classification and (
        missing := class_required - set(classifications)
    ):
        raise ValueError(f"CRSP classifications are missing: {sorted(missing)}")
    names = stock_names.loc[
        stock_names["EXCHCD"].eq(1) & stock_names["SHRCD"].isin([10, 11])
    ].copy()
    names["NAMEDT"] = pd.to_datetime(names["NAMEDT"], errors="raise")
    names["NAMEENDT"] = pd.to_datetime(names["NAMEENDT"], errors="coerce")
    latest_delisting = (
        delistings.sort_values(["PERMNO", "DLSTDT"])
        .groupby("PERMNO", as_index=False)
        .tail(1)
    )
    out = names.merge(
        latest_delisting[["PERMNO", "DLSTDT", "DLRET"]], on="PERMNO", how="left"
    )
    if not has_dated_classification:
        out = out.merge(
            classifications[list(class_required)],
            on="PERMNO",
            how="left",
            validate="many_to_one",
        )
    out["security_id"] = "CRSP:" + out["PERMNO"].astype(int).astype(str)
    out["issuer_id"] = "CRSPCO:" + out["PERMCO"].astype(int).astype(str)
    out["ticker"] = out["TICKER"]
    out["exchange"] = "NYSE"
    out["security_type"] = "common_stock"
    out["identifier_type"] = "permno"
    out["identifier_quality"] = "provider_permanent"
    out["listing_start"] = out["NAMEDT"]
    out["listing_end"] = out["NAMEENDT"]
    delisting_date = pd.to_datetime(out["DLSTDT"], errors="coerce")
    out["is_delisted"] = delisting_date.eq(out["listing_end"])
    out["delisting_return"] = pd.to_numeric(out["DLRET"], errors="coerce")
    # A CRSP name interval ending is not itself a delisting; only DLSTDT closes
    # the permanent security record.
    columns = sorted(REQUIRED_SECURITY_COLUMNS) + [
        "is_delisted",
        "identifier_type",
        "identifier_quality",
    ]
    return SecurityMaster.from_frame(out[columns])
