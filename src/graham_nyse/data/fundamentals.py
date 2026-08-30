from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from graham_nyse.data.vintages import acceptance_metadata_from_submissions


@dataclass(frozen=True)
class Concept:
    names: tuple[str, ...]
    unit: str = "USD"


CONCEPTS = {
    "revenue": Concept(
        ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
    ),
    "net_income": Concept(("NetIncomeLoss",)),
    "cfo": Concept(("NetCashProvidedByUsedInOperatingActivities",)),
    "capex": Concept(("PaymentsToAcquirePropertyPlantAndEquipment",)),
    "assets": Concept(("Assets",)),
    "liabilities": Concept(("Liabilities",)),
    "equity": Concept(
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        )
    ),
    "cash": Concept(("CashAndCashEquivalentsAtCarryingValue",)),
    "debt_current": Concept(("ShortTermBorrowings", "LongTermDebtCurrent")),
    "debt_long": Concept(("LongTermDebtNoncurrent",)),
    "operating_income": Concept(("OperatingIncomeLoss",)),
    "interest_expense": Concept(
        ("InterestExpenseNonOperating", "InterestAndDebtExpense")
    ),
    "current_assets": Concept(("AssetsCurrent",)),
    "current_liabilities": Concept(("LiabilitiesCurrent",)),
}


def _facts_namespace(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("facts", {}).get("us-gaap", {})


def annual_series(
    payload: dict[str, Any],
    concept: Concept,
    accepted_by_accession: dict[str, pd.Timestamp] | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
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
        if accepted_by_accession is not None:
            frame["accepted_at"] = frame.get(
                "accn", pd.Series(index=frame.index, dtype=str)
            ).map(accepted_by_accession)
            frame["accepted_at"] = pd.to_datetime(
                frame["accepted_at"], utc=True, errors="coerce"
            )
            if as_of is not None:
                cutoff = pd.Timestamp(as_of)
                cutoff = (
                    cutoff.tz_localize("UTC")
                    if cutoff.tzinfo is None
                    else cutoff.tz_convert("UTC")
                )
                frame = frame.loc[frame["accepted_at"].le(cutoff)]
            ordering = "accepted_at"
        else:
            frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
            ordering = "filed"
        frame = frame.sort_values(["fy", ordering]).drop_duplicates("fy", keep="last")
        candidates.append(frame[["fy", "val"]])
    if not candidates:
        return pd.Series(dtype=float)
    combined = (
        pd.concat(candidates).sort_values("fy").drop_duplicates("fy", keep="first")
    )
    return combined.set_index("fy")["val"].astype(float)


def latest_annual(
    payload: dict[str, Any],
    key: str,
    accepted_by_accession: dict[str, pd.Timestamp] | None = None,
    as_of: pd.Timestamp | None = None,
) -> float:
    series = annual_series(payload, CONCEPTS[key], accepted_by_accession, as_of)
    return float(series.iloc[-1]) if not series.empty else np.nan


def extract_fundamental_row(
    ticker: str,
    cik: int,
    payload: dict[str, Any],
    history_years: int = 10,
    accepted_by_accession: dict[str, pd.Timestamp] | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    net_income = annual_series(
        payload, CONCEPTS["net_income"], accepted_by_accession, as_of
    ).tail(history_years)
    assets = annual_series(payload, CONCEPTS["assets"], accepted_by_accession, as_of)
    cfo = annual_series(payload, CONCEPTS["cfo"], accepted_by_accession, as_of)
    capex = annual_series(payload, CONCEPTS["capex"], accepted_by_accession, as_of)

    ni = float(net_income.iloc[-1]) if not net_income.empty else np.nan
    latest_cfo = float(cfo.iloc[-1]) if not cfo.empty else np.nan
    latest_capex = float(capex.iloc[-1]) if not capex.empty else np.nan
    latest_assets = float(assets.iloc[-1]) if not assets.empty else np.nan
    prev_assets = float(assets.iloc[-2]) if len(assets) >= 2 else latest_assets
    avg_assets = np.nanmean([latest_assets, prev_assets])

    latest = lambda key: latest_annual(payload, key, accepted_by_accession, as_of)
    debt = np.nansum([latest("debt_current"), latest("debt_long")])
    cash = latest("cash")
    op_income = latest("operating_income")
    interest = latest("interest_expense")
    equity = latest("equity")
    liabilities = latest("liabilities")
    current_assets = latest("current_assets")
    current_liabilities = latest("current_liabilities")

    return {
        "ticker": ticker,
        "cik": int(cik),
        "positive_earnings_years": int((net_income > 0).sum()),
        "earnings_history_years": int(net_income.notna().sum()),
        "normalized_net_income": float(net_income.median())
        if not net_income.empty
        else np.nan,
        "net_income": ni,
        "cfo": latest_cfo,
        "capex": latest_capex,
        "fcf": latest_cfo - latest_capex
        if np.isfinite(latest_cfo) and np.isfinite(latest_capex)
        else np.nan,
        "assets": latest_assets,
        "liabilities": liabilities,
        "equity": equity,
        "cash": cash,
        "debt": debt,
        "net_debt": debt - cash if np.isfinite(debt) and np.isfinite(cash) else np.nan,
        "operating_income": op_income,
        "interest_expense": interest,
        "interest_coverage": op_income / interest
        if np.isfinite(op_income) and np.isfinite(interest) and interest > 0
        else np.nan,
        "current_ratio": current_assets / current_liabilities
        if np.isfinite(current_assets)
        and np.isfinite(current_liabilities)
        and current_liabilities > 0
        else np.nan,
        "roa": ni / avg_assets if np.isfinite(ni) and avg_assets > 0 else np.nan,
        "accruals": (ni - latest_cfo) / avg_assets
        if np.isfinite(ni) and np.isfinite(latest_cfo) and avg_assets > 0
        else np.nan,
        "earnings_stability": float(
            net_income.pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .std()
        ),
    }


def extract_fundamental_vintages(
    security_id: str,
    ticker: str,
    cik: int,
    company_facts: dict[str, Any],
    submissions: dict[str, Any],
    history_years: int = 10,
) -> pd.DataFrame:
    """Recompute normalized fundamentals at each historical SEC acceptance cutoff."""
    metadata = acceptance_metadata_from_submissions(submissions)
    metadata = metadata.loc[
        metadata["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A"])
    ]
    accepted = dict(
        zip(
            metadata["accession_number"].astype(str),
            metadata["accepted_at"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    for record in metadata.sort_values("accepted_at").itertuples(index=False):
        row = extract_fundamental_row(
            ticker,
            cik,
            company_facts,
            history_years,
            accepted_by_accession=accepted,
            as_of=record.accepted_at,
        )
        row.update(
            {
                "security_id": str(security_id),
                "accession_number": str(record.accession_number),
                "accepted_at": record.accepted_at,
                "period_end": record.report_date,
                "form": record.form,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
