from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Concept:
    names: tuple[str, ...]
    unit: str = "USD"


CONCEPTS = {
    "revenue": Concept(("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")),
    "net_income": Concept(("NetIncomeLoss",)),
    "cfo": Concept(("NetCashProvidedByUsedInOperatingActivities",)),
    "capex": Concept(("PaymentsToAcquirePropertyPlantAndEquipment",)),
    "assets": Concept(("Assets",)),
    "liabilities": Concept(("Liabilities",)),
    "equity": Concept(("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    "cash": Concept(("CashAndCashEquivalentsAtCarryingValue",)),
    "debt_current": Concept(("ShortTermBorrowings", "LongTermDebtCurrent")),
    "debt_long": Concept(("LongTermDebtNoncurrent",)),
    "operating_income": Concept(("OperatingIncomeLoss",)),
    "interest_expense": Concept(("InterestExpenseNonOperating", "InterestAndDebtExpense")),
    "current_assets": Concept(("AssetsCurrent",)),
    "current_liabilities": Concept(("LiabilitiesCurrent",)),
}


def _facts_namespace(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("facts", {}).get("us-gaap", {})


def annual_series(payload: dict[str, Any], concept: Concept) -> pd.Series:
    facts = _facts_namespace(payload)
    candidates: list[pd.DataFrame] = []
    for name in concept.names:
        units = facts.get(name, {}).get("units", {})
        records = units.get(concept.unit, [])
        if not records:
            continue
        frame = pd.DataFrame(records)
        if frame.empty or "fy" not in frame:
            continue
        frame = frame.loc[frame.get("fp", pd.Series(index=frame.index)).eq("FY")]
        forms = frame.get("form", pd.Series(index=frame.index, dtype=str))
        frame = frame.loc[forms.isin(["10-K", "10-K/A", "20-F", "20-F/A"])]
        if frame.empty:
            continue
        frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
        frame = frame.sort_values(["fy", "filed"]).drop_duplicates("fy", keep="last")
        candidates.append(frame[["fy", "val"]])
    if not candidates:
        return pd.Series(dtype=float)
    combined = pd.concat(candidates).sort_values("fy").drop_duplicates("fy", keep="first")
    return combined.set_index("fy")["val"].astype(float)


def latest_annual(payload: dict[str, Any], key: str) -> float:
    series = annual_series(payload, CONCEPTS[key])
    return float(series.iloc[-1]) if not series.empty else np.nan


def extract_fundamental_row(ticker: str, cik: int, payload: dict[str, Any], history_years: int = 10) -> dict[str, Any]:
    net_income = annual_series(payload, CONCEPTS["net_income"]).tail(history_years)
    assets = annual_series(payload, CONCEPTS["assets"])
    cfo = annual_series(payload, CONCEPTS["cfo"])
    capex = annual_series(payload, CONCEPTS["capex"])

    ni = float(net_income.iloc[-1]) if not net_income.empty else np.nan
    latest_cfo = float(cfo.iloc[-1]) if not cfo.empty else np.nan
    latest_capex = float(capex.iloc[-1]) if not capex.empty else np.nan
    latest_assets = float(assets.iloc[-1]) if not assets.empty else np.nan
    prev_assets = float(assets.iloc[-2]) if len(assets) >= 2 else latest_assets
    avg_assets = np.nanmean([latest_assets, prev_assets])

    debt = np.nansum([latest_annual(payload, "debt_current"), latest_annual(payload, "debt_long")])
    cash = latest_annual(payload, "cash")
    op_income = latest_annual(payload, "operating_income")
    interest = latest_annual(payload, "interest_expense")
    equity = latest_annual(payload, "equity")
    current_assets = latest_annual(payload, "current_assets")
    current_liabilities = latest_annual(payload, "current_liabilities")

    return {
        "ticker": ticker,
        "cik": int(cik),
        "positive_earnings_years": int((net_income > 0).sum()),
        "earnings_history_years": int(net_income.notna().sum()),
        "normalized_net_income": float(net_income.median()) if not net_income.empty else np.nan,
        "net_income": ni,
        "cfo": latest_cfo,
        "capex": latest_capex,
        "fcf": latest_cfo - latest_capex if np.isfinite(latest_cfo) and np.isfinite(latest_capex) else np.nan,
        "assets": latest_assets,
        "equity": equity,
        "cash": cash,
        "debt": debt,
        "net_debt": debt - cash if np.isfinite(debt) and np.isfinite(cash) else np.nan,
        "operating_income": op_income,
        "interest_expense": interest,
        "interest_coverage": op_income / interest if np.isfinite(op_income) and np.isfinite(interest) and interest > 0 else np.nan,
        "current_ratio": current_assets / current_liabilities if np.isfinite(current_assets) and np.isfinite(current_liabilities) and current_liabilities > 0 else np.nan,
        "roa": ni / avg_assets if np.isfinite(ni) and avg_assets > 0 else np.nan,
        "accruals": (ni - latest_cfo) / avg_assets if np.isfinite(ni) and np.isfinite(latest_cfo) and avg_assets > 0 else np.nan,
        "earnings_stability": float(net_income.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).std()),
    }
