from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2016-07-01")
END = pd.Timestamp("2026-06-30")
HISTORY_START = pd.Timestamp("2015-01-02")
SEED = 20260830


def build_evolving_frames() -> dict[str, pd.DataFrame]:
    """Build deterministic point-in-time data with listings, delistings and new filings.

    This is a software-validation fixture, not historical market evidence.
    """
    rng = np.random.default_rng(SEED)
    n = 72
    ids = [f"EV{i:03d}" for i in range(n)]
    tickers = [f"EVO{i:03d}" for i in range(n)]
    domains = ["ordinary", "bank", "insurer", "reit"]
    sectors = [
        "Industrials", "Energy", "Utilities", "Technology",
        "Banks", "Insurance", "Hotels", "Data Centers",
    ]

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp | pd.NaT] = []
    for i in range(n):
        if i < 48:
            listing_start = pd.Timestamp("2010-01-01")
        else:
            listing_start = pd.Timestamp(f"{2017 + (i - 48) // 4}-01-03")
        starts.append(listing_start)
        if i in {2, 7, 13, 18, 25, 31, 39, 44}:
            ends.append(pd.Timestamp(f"{2018 + i % 8}-06-30"))
        else:
            ends.append(pd.NaT)

    master_rows: list[dict[str, object]] = []
    for i, security_id in enumerate(ids):
        master_rows.append(
            {
                "security_id": security_id,
                "issuer_id": f"ISS{i:03d}",
                "ticker": tickers[i],
                "exchange": "NYSE",
                "security_type": "common_stock",
                "company_domain": domains[i % 4],
                "sector": sectors[i % len(sectors)],
                "listing_start": starts[i],
                "listing_end": ends[i],
                "delisting_return": -0.30 - 0.05 * (i % 5) if pd.notna(ends[i]) else np.nan,
                "data_provider": "deterministic_evolving_fixture",
            }
        )
    master = pd.DataFrame(master_rows)

    dates = pd.bdate_range(HISTORY_START, END + pd.offsets.Day(2))
    market_factor = rng.normal(0.00028, 0.0080, len(dates))
    price_rows: list[dict[str, object]] = []
    for i, security_id in enumerate(ids):
        idio = rng.normal(0.0, 0.006 + (i % 7) * 0.00045, len(dates))
        value_premium = (0.00005 * ((i % 9) - 4))
        quality_premium = 0.00004 * (i % 5)
        daily = market_factor * (0.70 + (i % 6) * 0.08) + idio + value_premium + quality_premium
        series = (14.0 + i * 0.65) * np.exp(np.cumsum(daily))
        for day, close in zip(dates, series, strict=True):
            if day < starts[i] or (pd.notna(ends[i]) and day > ends[i]):
                continue
            price_rows.append(
                {
                    "date": day,
                    "security_id": security_id,
                    "close": float(close),
                    "volume": int(180_000 + i * 12_000 + rng.integers(0, 40_000)),
                    "shares_outstanding": float(35_000_000 + i * 700_000),
                    "data_provider": "deterministic_evolving_fixture",
                }
            )
    prices = pd.DataFrame(price_rows)

    filing_rows: list[dict[str, object]] = []
    for year in range(2015, 2027):
        accepted_dates = [
            pd.Timestamp(f"{year}-03-15 21:00:00", tz="UTC"),
            pd.Timestamp(f"{year}-08-01 21:00:00", tz="UTC"),
        ]
        for filing_number, accepted_at in enumerate(accepted_dates):
            if accepted_at > pd.Timestamp(END, tz="UTC"):
                continue
            for i, security_id in enumerate(ids):
                day = accepted_at.tz_localize(None).normalize()
                if day < starts[i] or (pd.notna(ends[i]) and day > ends[i]):
                    continue
                age = max(0, year - starts[i].year)
                cycle = np.sin((year - 2015) / 2.2 + i / 8.0)
                market_cap = 650e6 + i * 42e6 + age * 28e6
                earnings_yield = 0.035 + (i % 12) * 0.003 + cycle * 0.006
                quality = 0.045 + (i % 8) * 0.004 - max(0.0, -cycle) * 0.005
                ni = market_cap * earnings_yield
                cfo = ni * (1.12 + (i % 4) * 0.04)
                assets = 900e6 + i * 23e6 + age * 18e6
                equity = assets * (0.34 + (i % 5) * 0.025)
                row = {
                    "security_id": security_id,
                    "accession_number": f"EV-{year}-{filing_number}-{i:04d}",
                    "accepted_at": accepted_at,
                    "period_end": day - pd.offsets.Day(45),
                    "earnings_history_years": min(10, max(5, age)),
                    "positive_earnings_years": min(9, max(4, age - (i % 3))),
                    "normalized_net_income": ni,
                    "cfo": cfo,
                    "fcf": cfo * (0.62 + (i % 6) * 0.035),
                    "assets": assets,
                    "liabilities": assets - equity,
                    "equity": equity,
                    "tangible_equity": equity * 0.90,
                    "net_debt": assets * (0.08 + (i % 6) * 0.018),
                    "interest_coverage": 3.2 + (i % 8) * 0.65 + quality * 8,
                    "current_ratio": 1.25 + (i % 5) * 0.10,
                    "roa": ni / assets,
                    "roe": ni / equity,
                    "accruals": 0.035 - quality * 0.45,
                    "earnings_stability": 0.13 - quality,
                    "cet1_ratio": 0.09 + (i % 5) * 0.008,
                    "nonperforming_assets_ratio": 0.012 + (i % 4) * 0.004,
                    "deposit_funding_ratio": 0.65 + (i % 5) * 0.045,
                    "statutory_capital_ratio": 1.45 + (i % 6) * 0.12,
                    "combined_ratio": 0.91 + (i % 5) * 0.012,
                    "reserve_development_ratio": 0.005 + (i % 4) * 0.006,
                    "free_surplus": market_cap * (0.04 + (i % 5) * 0.01),
                    "ffo": market_cap * (0.045 + (i % 7) * 0.006),
                    "affo": market_cap * (0.038 + (i % 7) * 0.0055),
                    "nav_discount": 0.04 + (i % 8) * 0.018,
                    "annual_dividend": market_cap * (0.018 + (i % 6) * 0.004),
                    "fixed_charge_coverage": 1.7 + (i % 7) * 0.24,
                    "net_debt_to_ebitda": 3.0 + (i % 7) * 0.35,
                    "occupancy": 0.76 + (i % 8) * 0.025,
                    "dividend_coverage": 1.05 + (i % 6) * 0.12,
                    "market_cap": market_cap,
                }
                filing_rows.append(row)
    vintages = pd.DataFrame(filing_rows)

    action_rows: list[dict[str, object]] = []
    for year in range(2017, 2027):
        for i in range(0, n, 9):
            security_id = ids[i]
            day = pd.Timestamp(f"{year}-05-15")
            if day >= starts[i] and (pd.isna(ends[i]) or day <= ends[i]):
                action_rows.append(
                    {
                        "date": day,
                        "security_id": security_id,
                        "action_type": "DIVIDEND",
                        "value": 0.10 + (i % 4) * 0.03,
                        "qualified": True,
                        "data_provider": "deterministic_evolving_fixture",
                    }
                )
    actions = pd.DataFrame(action_rows)
    return {
        "security_master": master,
        "filing_vintages": vintages,
        "prices": prices,
        "corporate_actions": actions,
    }


def write_frames(output: str | Path) -> dict[str, Path]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in build_evolving_frames().items():
        path = target / f"evolving_{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


if __name__ == "__main__":
    written = write_frames(Path("outputs/evolving_fixture/data"))
    for key, value in written.items():
        print(f"{key}: {value}")
