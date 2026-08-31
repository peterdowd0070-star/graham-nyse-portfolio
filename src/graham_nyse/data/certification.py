from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

CertificationStatus = Literal["empirical_certified", "research_only", "simulation_only"]


@dataclass(frozen=True)
class CertificationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class HistoricalDataCertification:
    status: CertificationStatus
    provider: str
    checks: tuple[CertificationCheck, ...]

    @property
    def empirical_results_allowed(self) -> bool:
        return self.status == "empirical_certified"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "provider": self.provider,
            "empirical_results_allowed": self.empirical_results_allowed,
            "checks": [asdict(check) for check in self.checks],
        }


def _provider_values(*frames: pd.DataFrame | None) -> set[str]:
    values: set[str] = set()
    for frame in frames:
        if frame is None or frame.empty or "data_provider" not in frame:
            continue
        values.update(frame["data_provider"].dropna().astype(str).str.lower())
    return values


def certify_historical_data(
    security_master: pd.DataFrame,
    prices: pd.DataFrame,
    corporate_actions: pd.DataFrame | None,
    filing_vintages: pd.DataFrame | None = None,
) -> HistoricalDataCertification:
    """Grade a historical panel before its performance can be described as empirical.

    This deliberately tests dataset properties, not which Python package downloaded
    the rows.  An open-source client does not make its upstream data survivorship-free.
    """

    checks: list[CertificationCheck] = []
    providers = _provider_values(security_master, prices, corporate_actions)
    provider = next(iter(providers)) if len(providers) == 1 else "mixed_or_unspecified"
    checks.append(
        CertificationCheck(
            "single_market_provider",
            len(providers) == 1,
            f"observed providers: {sorted(providers) or ['unspecified']}",
        )
    )

    required_master = {
        "security_id",
        "listing_start",
        "listing_end",
        "is_delisted",
        "delisting_return",
    }
    has_dated_master = required_master.issubset(security_master)
    checks.append(
        CertificationCheck(
            "dated_security_master",
            has_dated_master,
            "dated listing intervals and terminal fields are required",
        )
    )

    identity_quality = set(
        security_master.get("identifier_quality", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.lower()
    )
    permanent_identity = bool(identity_quality) and identity_quality <= {
        "provider_permanent"
    }
    checks.append(
        CertificationCheck(
            "permanent_security_identity",
            permanent_identity,
            "ticker-derived identities fail; provider-stable security IDs are required",
        )
    )

    inactive_count = 0
    missing_terminal = 0
    if has_dated_master:
        inactive = security_master["is_delisted"].fillna(False).astype(bool)
        inactive_count = int(inactive.sum())
        terminal = pd.to_numeric(
            security_master.loc[inactive, "delisting_return"], errors="coerce"
        )
        missing_terminal = int((~np.isfinite(terminal)).sum())
    checks.append(
        CertificationCheck(
            "inactive_securities_included",
            inactive_count > 0,
            f"inactive/delisted rows: {inactive_count}",
        )
    )
    checks.append(
        CertificationCheck(
            "complete_delisting_returns",
            inactive_count > 0 and missing_terminal == 0,
            f"missing delisting returns among inactive rows: {missing_terminal}",
        )
    )

    adjustment = set(
        prices.get("price_adjustment", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.lower()
    )
    raw_prices = bool(adjustment) and adjustment <= {"raw", "unadjusted"}
    checks.append(
        CertificationCheck(
            "raw_prices_with_explicit_actions",
            raw_prices and corporate_actions is not None,
            "unadjusted closes and a separate corporate-action table are required",
        )
    )

    filing_ok = False
    if filing_vintages is not None and not filing_vintages.empty:
        filing_ok = {"accepted_at", "accession_number", "security_id"}.issubset(
            filing_vintages
        ) and pd.to_datetime(filing_vintages["accepted_at"], utc=True).notna().all()
    checks.append(
        CertificationCheck(
            "point_in_time_filing_availability",
            filing_ok,
            "SEC accession and accepted_at timestamp are required for each vintage",
        )
    )

    status: CertificationStatus
    simulation_provider = any(
        token in provider_name
        for provider_name in providers
        for token in ("simulation", "synthetic", "fixture", "generated")
    )
    if simulation_provider:
        status = "simulation_only"
    elif all(check.passed for check in checks):
        status = "empirical_certified"
    elif not providers or security_master.empty or prices.empty:
        status = "simulation_only"
    else:
        status = "research_only"
    return HistoricalDataCertification(status, provider, tuple(checks))
