from __future__ import annotations

from typing import Any

import pandas as pd


def company_domain_from_sic(sic: Any) -> str:
    """Classify accounting domain from the SIC known on the observation date."""
    try:
        value = int(float(sic))
    except (TypeError, ValueError):
        return "ordinary"
    if value == 6798:
        return "reit"
    if 6020 <= value <= 6099 or 6100 <= value <= 6199:
        return "bank"
    if 6310 <= value <= 6411:
        return "insurer"
    return "ordinary"


def broad_sector_from_sic(sic: Any) -> str:
    """Deterministic broad peer group based only on dated SIC information."""
    try:
        value = int(float(sic))
    except (TypeError, ValueError):
        return "Unclassified"
    if 100 <= value <= 999:
        return "Agriculture"
    if 1000 <= value <= 1499:
        return "Materials"
    if 1500 <= value <= 1799:
        return "Industrials"
    if 2000 <= value <= 3999:
        if 2830 <= value <= 2839 or 3840 <= value <= 3851:
            return "Health Care"
        if 3570 <= value <= 3579 or 3660 <= value <= 3699:
            return "Technology"
        return "Manufacturing"
    if 4000 <= value <= 4999:
        return "Utilities" if 4900 <= value <= 4999 else "Transportation"
    if 5000 <= value <= 5999:
        return "Consumer"
    if 6000 <= value <= 6799:
        return "Real Estate" if value == 6798 else "Financials"
    if 7000 <= value <= 8999:
        return "Health Care" if 8000 <= value <= 8099 else "Services"
    if 9000 <= value <= 9999:
        return "Public Administration"
    return "Unclassified"


def classify_crsp_name_history(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"PERMNO", "SICCD"}
    if missing := required - set(frame):
        raise ValueError(f"CRSP name history is missing: {sorted(missing)}")
    out = frame.copy()
    out["company_domain"] = out["SICCD"].map(company_domain_from_sic)
    out["sector"] = out["SICCD"].map(broad_sector_from_sic)
    return out
