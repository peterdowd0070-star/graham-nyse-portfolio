from __future__ import annotations

import pandas as pd

from tests.fixtures.generate_evolving_10y import END, START, build_evolving_frames


def test_evolving_fixture_contains_entries_exits_and_point_in_time_filings() -> None:
    frames = build_evolving_frames()
    master = frames["security_master"]
    vintages = frames["filing_vintages"]
    prices = frames["prices"]

    assert master["listing_start"].gt(START).any()
    assert master["listing_end"].between(START, END).any()
    delisted = master.loc[master["listing_end"].notna()]
    assert delisted["delisting_return"].notna().all()

    accepted = pd.to_datetime(vintages["accepted_at"], utc=True)
    assert accepted.is_monotonic_increasing is False or accepted.nunique() > 10
    assert accepted.min() < pd.Timestamp(START, tz="UTC")
    assert accepted.max() <= pd.Timestamp("2026-08-01", tz="UTC")

    price_dates = pd.to_datetime(prices["date"])
    assert price_dates.min() < START
    assert price_dates.max() >= END
    assert prices["security_id"].nunique() == len(master)


def test_evolving_fixture_is_deterministic() -> None:
    first = build_evolving_frames()
    second = build_evolving_frames()
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
