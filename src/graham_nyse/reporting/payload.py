from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def build_report_payload(as_of: date, run_type: str, portfolio: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    columns = ["ticker", "name", "rank", "graham_score", "value_score", "quality_score", "target_weight", "target_dollars", "target_shares", "price"]
    available = [c for c in columns if c in portfolio.columns]
    return {
        "as_of_date": as_of.isoformat(),
        "run_type": run_type,
        "methodology_version": "0.1.0",
        "portfolio_summary": {
            "position_count": int(len(portfolio)),
            "target_capital": float(portfolio["target_dollars"].sum()),
            "largest_weight": float(portfolio["target_weight"].max()),
        },
        "positions": portfolio[available].to_dict(orient="records"),
        "audit": audit,
        "llm_policy": "Narrative only. Do not alter or invent securities, weights, trades, or calculations.",
    }
