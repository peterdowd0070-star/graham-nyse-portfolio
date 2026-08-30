from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import httpx
import pandas as pd

from graham_nyse.config import WeightStrategy, load_config
from graham_nyse.data.fundamentals import extract_fundamental_row
from graham_nyse.data.market import fetch_market_snapshot
from graham_nyse.data.sec import SecClient, build_nyse_universe
from graham_nyse.portfolio.construction import construct_portfolio, select_constituents
from graham_nyse.portfolio.scoring import apply_eligibility, calculate_scores
from graham_nyse.reporting.payload import build_report_payload
from graham_nyse.validation import validate_portfolio


def determine_run_type(
    as_of: date, quarterly: list[int], reconstruction: list[int]
) -> str:
    if as_of.month in reconstruction:
        return "full_reconstruction"
    if as_of.month in quarterly:
        return "quarterly_rebalance"
    return "monthly_monitor"


def run(
    config_path: str,
    output_dir: str,
    user_agent: str,
    classification_path: str,
    refresh: bool = False,
    scenario: str = "quality_value",
    weighting_strategy: str = "equal",
) -> None:
    cfg = load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sec = SecClient(user_agent=user_agent)

    tickers = sec.exchange_tickers(refresh=refresh)
    universe = build_nyse_universe(
        tickers,
        cfg.universe.exchange,
        cfg.universe.exclude_name_patterns,
        cfg.universe.include_adrs,
    )
    classifications = pd.read_csv(classification_path)
    required_classifications = {"ticker", "company_domain", "sector"}
    if not required_classifications.issubset(classifications):
        raise ValueError(
            f"Classification file requires {sorted(required_classifications)}"
        )
    universe = universe.merge(
        classifications[list(required_classifications)], on="ticker", how="inner"
    )
    universe["security_id"] = universe["cik"].astype(str) + ":" + universe["ticker"]
    universe["issuer_id"] = universe["cik"].astype(str)
    universe.to_csv(out / "universe.csv", index=False)

    market = fetch_market_snapshot(universe["ticker"])
    screened_universe = universe.merge(market, on="ticker", how="inner")
    liquid = screened_universe.loc[
        screened_universe["market_cap"].ge(cfg.universe.min_market_cap)
        & screened_universe["price"].ge(cfg.universe.min_price)
        & screened_universe["median_dollar_volume_60d"].ge(
            cfg.universe.min_median_dollar_volume_60d
        )
    ]

    rows = []
    for record in liquid.itertuples(index=False):
        try:
            payload = sec.company_facts(int(record.cik), refresh=refresh)
            rows.append(
                extract_fundamental_row(
                    record.ticker,
                    int(record.cik),
                    payload,
                    cfg.fundamentals.history_years,
                )
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
            rows.append(
                {
                    "ticker": record.ticker,
                    "cik": int(record.cik),
                    "data_error": str(exc),
                }
            )

    fundamentals = pd.DataFrame(rows)
    dataset = liquid.merge(fundamentals, on=["ticker", "cik"], how="left")
    eligible = apply_eligibility(dataset, cfg, scenario)
    scored = calculate_scores(eligible, cfg, scenario)
    selected = select_constituents(scored, cfg)
    portfolio = construct_portfolio(
        selected, cfg, weighting_strategy=cast(WeightStrategy, weighting_strategy)
    )
    audit = validate_portfolio(portfolio, cfg)

    as_of = datetime.now(UTC).date()
    run_type = determine_run_type(
        as_of,
        cfg.schedule.quarterly_rebalance_months,
        cfg.schedule.full_reconstruction_months,
    )
    payload = build_report_payload(as_of, run_type, portfolio, audit)

    dataset.to_parquet(out / "fundamental_dataset.parquet", index=False)
    scored.to_csv(out / "scored_universe.csv", index=False)
    portfolio.to_csv(out / "target_portfolio.csv", index=False)
    (out / "report_context.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (out / "run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"Portfolio validation failed: {audit['errors']}")
