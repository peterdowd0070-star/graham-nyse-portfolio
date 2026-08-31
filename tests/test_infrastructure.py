from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graham_nyse.backtest.engine import _validate_market_provider_consistency
from graham_nyse.data.openbb_sec import normalize_openbb_pit_statement
from graham_nyse.data.providers import HistoricalBundle, require_single_provider
from graham_nyse.data.sharadar import SharadarExportProvider
from graham_nyse.infrastructure import HistoricalLake, nyse_sessions
from graham_nyse.portfolio.optimization import cvxpy_minimum_variance


def _empty_bundle(provider: str) -> HistoricalBundle:
    empty = pd.DataFrame()
    return HistoricalBundle(provider, empty, empty, empty)


def test_empirical_run_refuses_cross_provider_splicing():
    with pytest.raises(ValueError, match="may not splice"):
        require_single_provider(_empty_bundle("crsp"), _empty_bundle("sharadar"))
    with pytest.raises(ValueError, match="may not splice"):
        _validate_market_provider_consistency(
            pd.DataFrame({"data_provider": ["crsp"]}),
            pd.DataFrame({"data_provider": ["sharadar"]}),
        )


def test_openbb_adapter_requires_acceptance_time():
    with pytest.raises(ValueError, match="point-in-time auditable"):
        normalize_openbb_pit_statement(
            pd.DataFrame([{"period_ending": "2024-12-31", "accession_number": "x"}]),
            security_id="SEC:1",
        )


def test_openbb_adapter_normalizes_pit_result():
    result = normalize_openbb_pit_statement(
        pd.DataFrame(
            [
                {
                    "period_ending": "2024-12-31",
                    "accepted_date": "2025-02-01T21:00:00Z",
                    "accession_number": "0001-25-000001",
                    "assets": 10.0,
                }
            ]
        ),
        security_id="SEC:1",
    )
    assert result.iloc[0]["security_id"] == "SEC:1"
    assert str(result["accepted_at"].dtype) == "datetime64[ns, UTC]"


def test_sharadar_adapter_uses_permanent_identity(tmp_path: Path):
    tickers = pd.DataFrame(
        [
            {
                "permaticker": 101,
                "ticker": "OLD",
                "name": "Old Co",
                "exchange": "NYSE",
                "category": "Domestic Common Stock",
                "siccode": 4911,
                "firstpricedate": "2010-01-01",
                "lastpricedate": "2018-01-31",
                "isdelisted": "Y",
            }
        ]
    )
    prices = pd.DataFrame(
        [{"permaticker": 101, "date": "2017-01-03", "close": 10.0, "volume": 1000}]
    )
    ticker_path, price_path = tmp_path / "tickers.csv", tmp_path / "prices.csv"
    tickers.to_csv(ticker_path, index=False)
    prices.to_csv(price_path, index=False)
    bundle = SharadarExportProvider(ticker_path, price_path).load(
        "2017-01-01", "2018-01-01"
    )
    assert bundle.security_master.iloc[0]["security_id"] == "SHARADAR:101"
    assert bundle.prices.iloc[0]["security_id"] == "SHARADAR:101"


def test_duckdb_polars_lake_and_calendar(tmp_path: Path):
    prices = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "security_id": "X",
                "close": 10.0,
                "volume": 5.0,
            }
        ]
    )
    prices.to_parquet(tmp_path / "raw_prices.parquet", index=False)
    lake = HistoricalLake(tmp_path)
    connection = lake.connect()
    assert connection.execute("select count(*) from raw_prices").fetchone()[0] == 1
    assert lake.scan_prices().collect().height == 1
    sessions = nyse_sessions("2024-01-01", "2024-01-03")
    assert pd.Timestamp("2024-01-02") in sessions


def test_cvxpy_minimum_variance_respects_caps():
    covariance = np.diag([0.01, 0.02, 0.03, 0.04])
    sectors = pd.Series(["A", "A", "B", "B"])
    weights = cvxpy_minimum_variance(covariance, sectors, 0.4, 0.7)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.4 + 1e-7
    assert weights[:2].sum() <= 0.7 + 1e-7
