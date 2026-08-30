import pandas as pd

from graham_nyse.data.security_master import SecurityMaster, security_master_from_crsp
from graham_nyse.data.vintages import FilingVintageStore
from tests.fixtures.generate_simulation_smoke_test import build_smoke_frames


def test_filing_snapshot_never_uses_future_acceptance():
    frames = build_smoke_frames()
    store = FilingVintageStore.from_frame(frames["filing_vintages"])
    cutoff = pd.Timestamp("2016-06-30 20:00:00+00:00")
    snapshot = store.snapshot(cutoff)
    assert not snapshot.empty
    assert snapshot["accepted_at"].max() <= cutoff
    assert snapshot["accession_number"].str.contains("01-").all()


def test_inactive_security_is_present_before_delisting_and_absent_after():
    frames = build_smoke_frames()
    master = SecurityMaster.from_frame(frames["security_master"])
    assert "SEC000" in set(master.active_as_of("2017-06-30")["security_id"])
    assert "SEC000" not in set(master.active_as_of("2017-07-03")["security_id"])
    assert master.require_complete_delisting_returns() == []


def test_crsp_adapter_keeps_inactive_nyse_common_stock():
    names = pd.DataFrame(
        [
            {
                "PERMNO": 10001,
                "PERMCO": 9001,
                "TICKER": "OLD",
                "EXCHCD": 1,
                "SHRCD": 10,
                "NAMEDT": "2010-01-01",
                "NAMEENDT": "2018-03-01",
            }
        ]
    )
    delistings = pd.DataFrame(
        [{"PERMNO": 10001, "DLSTDT": "2018-03-01", "DLRET": -0.7}]
    )
    classifications = pd.DataFrame(
        [{"PERMNO": 10001, "company_domain": "ordinary", "sector": "Industrials"}]
    )
    master = security_master_from_crsp(names, delistings, classifications)
    assert "CRSP:10001" in set(master.active_as_of("2018-02-28")["security_id"])
    assert master.active_as_of("2018-03-02").empty
