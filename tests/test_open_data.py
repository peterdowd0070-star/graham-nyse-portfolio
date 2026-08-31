from __future__ import annotations

import numpy as np
import pandas as pd

from graham_nyse.data.certification import certify_historical_data
from graham_nyse.data.open_data import build_alpha_vantage_research_master


def test_alpha_vantage_master_is_explicitly_research_only():
    snapshots = pd.DataFrame(
        [
            {
                "symbol": "OLD",
                "name": "Old Corp",
                "exchange": "NYSE",
                "assetType": "Stock",
                "ipoDate": "2010-01-01",
                "delistingDate": "2018-02-01",
            }
        ]
    )
    classifications = pd.DataFrame(
        [{"symbol": "OLD", "company_domain": "ordinary", "sector": "Industrials"}]
    )
    master = build_alpha_vantage_research_master(snapshots, classifications)
    prices = pd.DataFrame(
        [
            {
                "date": "2018-01-31",
                "security_id": master.iloc[0]["security_id"],
                "close": 10.0,
                "volume": 1000.0,
                "price_adjustment": "raw",
                "data_provider": "yahoo_research",
            }
        ]
    )
    actions = pd.DataFrame(
        columns=[
            "date",
            "security_id",
            "action_type",
            "value",
            "qualified",
            "data_provider",
        ]
    )
    report = certify_historical_data(master, prices, actions)
    assert report.status == "research_only"
    assert not report.empirical_results_allowed
    failed = {check.name for check in report.checks if not check.passed}
    assert "permanent_security_identity" in failed
    assert "complete_delisting_returns" in failed


def test_crsp_style_bundle_can_pass_empirical_certification():
    master = pd.DataFrame(
        [
            {
                "security_id": "CRSP:10001",
                "listing_start": "2010-01-01",
                "listing_end": "2018-02-01",
                "is_delisted": True,
                "delisting_return": -0.5,
                "identifier_quality": "provider_permanent",
                "data_provider": "crsp",
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "date": "2018-01-31",
                "security_id": "CRSP:10001",
                "close": 10.0,
                "volume": 1000.0,
                "price_adjustment": "unadjusted",
                "data_provider": "crsp",
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "date": "2017-12-01",
                "security_id": "CRSP:10001",
                "action_type": "DIVIDEND",
                "value": 0.1,
                "qualified": False,
                "data_provider": "crsp",
            }
        ]
    )
    filings = pd.DataFrame(
        [
            {
                "security_id": "CRSP:10001",
                "accession_number": "0001",
                "accepted_at": "2017-10-01T12:00:00Z",
            }
        ]
    )
    report = certify_historical_data(master, prices, actions, filings)
    assert report.status == "empirical_certified"
    assert report.empirical_results_allowed
    assert np.isfinite(master["delisting_return"]).all()


def test_generated_provider_can_never_be_empirical():
    master = pd.DataFrame(
        [
            {
                "security_id": "SIM:1",
                "listing_start": "2010-01-01",
                "listing_end": "2018-01-01",
                "is_delisted": True,
                "delisting_return": -0.5,
                "identifier_quality": "provider_permanent",
                "data_provider": "deterministic_evolving_fixture",
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "security_id": "SIM:1",
                "price_adjustment": "raw",
                "data_provider": "deterministic_evolving_fixture",
            }
        ]
    )
    actions = pd.DataFrame([{"data_provider": "deterministic_evolving_fixture"}])
    filings = pd.DataFrame(
        [
            {
                "security_id": "SIM:1",
                "accession_number": "sim-1",
                "accepted_at": "2017-01-01T00:00:00Z",
            }
        ]
    )
    report = certify_historical_data(master, prices, actions, filings)
    assert report.status == "simulation_only"
    assert not report.empirical_results_allowed
