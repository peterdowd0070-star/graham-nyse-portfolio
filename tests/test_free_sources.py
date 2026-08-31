from __future__ import annotations

import pandas as pd
import pytest

from graham_nyse.data.nasdaq_trader import normalize_other_listed
from graham_nyse.data.open_data import open_listing_snapshot_requests
from graham_nyse.data.source_priority import source_plan
from graham_nyse.data.stooq import audit_close_prices


def test_free_plan_prioritizes_public_sources_and_excludes_wrds():
    plan = source_plan(no_commercial=True)
    assert plan["filing_vintages"][0].source == "sec_edgar"
    assert plan["historical_membership"][0].source == (
        "alpha_vantage_listing_status"
    )
    assert all(
        choice.access != "commercial"
        for choices in plan.values()
        for choice in choices
    )


def test_open_listing_requests_match_semiannual_reconstruction():
    requests = open_listing_snapshot_requests("2016-07-01", "2026-06-30")
    active = [day for day, state in requests if state == "active"]
    assert active[0] == pd.Timestamp("2016-06-30")
    assert active[-1] == pd.Timestamp("2026-06-30")
    assert len(active) == 21
    assert requests[-1] == (pd.Timestamp("2026-06-30"), "delisted")


def test_nasdaq_reference_keeps_only_nyse_common_candidates():
    raw = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
        "Test Issue|NASDAQ Symbol\n"
        "ABC|ABC Corp Common Stock|N|ABC|N|100|N|ABC\n"
        "PFD|PFD Corp Preferred Stock|N|PFD|N|100|N|PFD\n"
        "ETF1|Example ETF|N|ETF1|Y|100|N|ETF1\n"
        "ARC|Arca Corp Common Stock|P|ARC|N|100|N|ARC\n"
        "File Creation Time: 0831202612:00|||||||"
    )
    result = normalize_other_listed(raw)
    candidates = result.loc[result["is_operating_common_candidate"], "ticker"]
    assert candidates.tolist() == ["ABC"]


def test_stooq_audit_reports_mismatches_without_splicing():
    primary = pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-03"], "close": [100.0, 102.0]}
    )
    audit = pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-03"], "close": [100.0, 100.0]}
    )
    result = audit_close_prices(primary, audit, relative_tolerance=0.01)
    assert result["overlap_rows"] == 2
    assert result["mismatch_rows"] == 1
    assert result["mismatch_rate"] == pytest.approx(0.5)
