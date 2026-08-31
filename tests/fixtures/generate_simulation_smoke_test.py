from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_smoke_frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    securities = [f"SEC{i:03d}" for i in range(48)]
    tickers = [f"FIX{i:03d}" for i in range(48)]
    domains = ["ordinary", "bank", "insurer", "reit"]
    sectors = [
        "Industrials",
        "Energy",
        "Utilities",
        "Technology",
        "Banks",
        "Insurance",
        "Hotels",
        "Data Centers",
    ]
    master_rows = []
    for i, (security_id, ticker) in enumerate(zip(securities, tickers, strict=True)):
        listing_end = pd.Timestamp("2017-06-30") if i == 0 else pd.NaT
        master_rows.append(
            {
                "security_id": security_id,
                "issuer_id": f"ISS{i:03d}",
                "ticker": ticker,
                "exchange": "NYSE",
                "security_type": "common_stock",
                "company_domain": domains[i % 4],
                "sector": sectors[i % 8],
                "listing_start": pd.Timestamp("2010-01-01"),
                "listing_end": listing_end,
                "delisting_return": -0.55 if i == 0 else np.nan,
            }
        )
    master = pd.DataFrame(master_rows)

    dates = pd.bdate_range("2015-01-02", "2018-07-03")
    price_rows = []
    for i, security_id in enumerate(securities):
        base = 18.0 + i
        daily = np.full(len(dates), 0.00018 + i / 10_000_000)
        daily += rng.normal(0.0, 0.008 + (i % 5) * 0.0005, len(dates))
        series = base * np.exp(np.cumsum(daily))
        end = pd.Timestamp("2017-06-30") if i == 0 else dates[-1]
        for day, price in zip(dates, series, strict=True):
            if day <= end:
                price_rows.append(
                    {
                        "date": day,
                        "security_id": security_id,
                        "close": price,
                        "volume": 250_000 + i * 10_000,
                        "shares_outstanding": 50_000_000 + i * 500_000,
                    }
                )
    prices = pd.DataFrame(price_rows)

    acceptance_dates = pd.to_datetime(
        [
            "2015-12-15 21:00:00+00:00",
            "2016-03-15 21:00:00+00:00",
            "2016-08-01 21:00:00+00:00",
            "2017-03-15 21:00:00+00:00",
            "2017-08-01 21:00:00+00:00",
            "2018-03-15 21:00:00+00:00",
        ]
    )
    vintage_rows = []
    for filing_index, accepted_at in enumerate(acceptance_dates):
        for i, security_id in enumerate(securities):
            assets = 1.0e9 + i * 20e6 + filing_index * 10e6
            equity = assets * (0.38 + (i % 4) * 0.02)
            market_cap = 1.2e9 + i * 30e6
            ni = market_cap * (0.035 + i / 10_000)
            cfo = ni * 1.25
            row = {
                "security_id": security_id,
                "accession_number": f"000000-{filing_index:02d}-{i:04d}",
                "accepted_at": accepted_at,
                "period_end": accepted_at.tz_localize(None).normalize()
                - pd.Timedelta(days=45),
                "earnings_history_years": 10,
                "positive_earnings_years": 9,
                "normalized_net_income": ni,
                "cfo": cfo,
                "fcf": cfo * 0.72,
                "assets": assets,
                "liabilities": assets - equity,
                "equity": equity,
                "tangible_equity": equity * 0.9,
                "net_debt": assets * 0.12,
                "interest_coverage": 4.0 + (i % 6) * 0.4,
                "current_ratio": 1.4,
                "roa": ni / assets,
                "roe": ni / equity,
                "accruals": 0.02 - i / 20_000,
                "earnings_stability": 0.08 - min(i, 40) / 2000,
                "cet1_ratio": 0.11 + (i % 3) * 0.005,
                "nonperforming_assets_ratio": 0.015 + (i % 3) * 0.002,
                "deposit_funding_ratio": 0.72,
                "statutory_capital_ratio": 1.8,
                "combined_ratio": 0.94 + (i % 3) * 0.01,
                "reserve_development_ratio": 0.01,
                "free_surplus": market_cap * 0.06,
                "ffo": market_cap * 0.065,
                "affo": market_cap * 0.055,
                "nav_discount": 0.12 - (i % 4) * 0.01,
                "annual_dividend": market_cap * 0.035,
                "fixed_charge_coverage": 2.4,
                "net_debt_to_ebitda": 4.5,
                "occupancy": 0.91,
                "dividend_coverage": 1.35,
                "market_cap": market_cap,
            }
            vintage_rows.append(row)
    vintages = pd.DataFrame(vintage_rows)
    # SEC000 is intentionally attractive so the engine must process its 2017 delisting.
    special = vintages["security_id"].eq("SEC000")
    vintages.loc[special, "normalized_net_income"] = (
        vintages.loc[special, "market_cap"] * 0.12
    )
    vintages.loc[special, "cfo"] = vintages.loc[special, "market_cap"] * 0.14
    vintages.loc[special, "fcf"] = vintages.loc[special, "market_cap"] * 0.11
    vintages.loc[special, "roa"] = 0.14
    vintages.loc[special, "accruals"] = -0.03
    vintages.loc[special, "earnings_stability"] = 0.01
    vintages.loc[special, "interest_coverage"] = 12.0
    actions = pd.DataFrame(
        [
            {
                "date": "2017-02-15",
                "security_id": "SEC010",
                "action_type": "DIVIDEND",
                "value": 0.20,
                "qualified": True,
            },
            {
                "date": "2017-05-15",
                "security_id": "SEC011",
                "action_type": "SPLIT",
                "value": 2.0,
                "qualified": False,
            },
        ]
    )
    return {
        "security_master": master,
        "filing_vintages": vintages,
        "prices": prices,
        "corporate_actions": actions,
    }


if __name__ == "__main__":
    output = Path(__file__).parent
    for name, frame in build_smoke_frames().items():
        frame.to_csv(output / f"smoke_{name}.csv", index=False)
