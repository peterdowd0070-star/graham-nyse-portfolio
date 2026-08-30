from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd


def _claim(
    claim_id: str, value: object, source: str, display: str | None = None
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "value": value,
        "display_value": str(value) if display is None else display,
        "source_artifact": source,
        "claim_type": "DETERMINISTIC_RESULT",
    }


def build_report_payload(
    as_of: date,
    run_type: str,
    portfolio: pd.DataFrame,
    audit: dict[str, Any],
    methodology_version: str = "1.0.0",
) -> dict[str, Any]:
    columns = [
        "security_id",
        "ticker",
        "name",
        "company_domain",
        "sector",
        "scenario",
        "weighting_strategy",
        "rank",
        "graham_score",
        "value_score",
        "quality_score",
        "target_weight",
        "target_dollars",
        "target_shares",
        "price",
    ]
    available = [column for column in columns if column in portfolio.columns]
    capital = float(portfolio["target_dollars"].sum()) if not portfolio.empty else 0.0
    largest = float(portfolio["target_weight"].max()) if not portfolio.empty else 0.0
    claims = [
        _claim("position_count", len(portfolio), "target_portfolio.csv"),
        _claim(
            "target_capital", capital, "target_portfolio.csv", "$" + f"{capital:,.2f}"
        ),
        _claim("largest_weight", largest, "target_portfolio.csv", f"{largest:.2%}"),
        _claim("audit_passed", bool(audit.get("passed")), "run_audit.json"),
    ]
    return {
        "as_of_date": as_of.isoformat(),
        "run_type": run_type,
        "methodology_version": methodology_version,
        "portfolio_summary": {
            "position_count": len(portfolio),
            "target_capital": capital,
            "largest_weight": largest,
        },
        "positions": portfolio[available].to_dict(orient="records"),
        "claims": claims,
        "audit": audit,
        "llm_policy": {
            "role": "Narrative only",
            "may_not": [
                "alter securities",
                "alter weights",
                "alter trades",
                "recalculate results",
                "fill missing data",
                "state a number without a supplied claim_id",
            ],
            "required_output_schema": {
                "sections": [
                    {"heading": "string", "text": "string", "claim_ids": ["string"]}
                ],
                "warnings": ["string"],
            },
        },
    }


def build_historical_report_payload(
    metadata: dict[str, object],
    metrics: dict[str, float],
    audit: dict[str, object],
) -> dict[str, object]:
    claims = []
    for metric, value in sorted(metrics.items()):
        is_rate = any(
            token in metric
            for token in (
                "return",
                "cagr",
                "volatility",
                "drawdown",
                "turnover",
                "alpha",
            )
        )
        display = f"{value:.2%}" if is_rate else f"{value:.6g}"
        claims.append(
            _claim(f"metric_{metric}", value, "historical_metrics.csv", display)
        )
    return {
        "methodology_version": metadata.get("methodology_version"),
        "run_metadata": metadata,
        "claims": claims,
        "audit": audit,
        "llm_policy": "Narrative only; every numerical statement must cite a supplied claim_id.",
    }


def validate_report_document(
    document: dict[str, Any], context: dict[str, Any]
) -> dict[str, object]:
    allowed = {str(claim["claim_id"]): claim for claim in context.get("claims", [])}
    errors: list[str] = []
    for section in document.get("sections", []):
        claim_ids = [str(value) for value in section.get("claim_ids", [])]
        unknown = set(claim_ids) - set(allowed)
        if unknown:
            errors.append(f"unknown_claim_ids:{','.join(sorted(unknown))}")
        text = str(section.get("text", ""))
        numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?%?", text))
        permitted: set[str] = set()
        for claim_id in claim_ids:
            claim = allowed.get(claim_id)
            if not claim:
                continue
            permitted.add(str(claim["value"]))
            permitted.add(str(claim["display_value"]))
            permitted.add(str(claim["display_value"]).replace(",", ""))
        for number in numbers:
            normalized = number.replace(",", "")
            if number not in permitted and normalized not in permitted:
                errors.append(f"unsupported_number:{number}")
    return {"passed": not errors, "errors": sorted(set(errors))}
