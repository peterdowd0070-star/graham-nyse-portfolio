import pandas as pd

from graham_nyse.backtest.engine import _market_snapshot
from graham_nyse.config import load_config
from graham_nyse.data.security_master import SecurityMaster
from graham_nyse.data.vintages import FilingVintageStore
from graham_nyse.portfolio.scoring import calculate_scores, rules_for
from tests.fixtures.generate_simulation_smoke_test import build_smoke_frames


def test_domain_specific_gate_does_not_leak_across_domains():
    cfg = load_config("config/strategy.yaml")
    frames = build_smoke_frames()
    store = FilingVintageStore.from_frame(frames["filing_vintages"])
    master = SecurityMaster.from_frame(frames["security_master"])
    cutoff = pd.Timestamp("2016-06-30 20:00:00+00:00")
    snapshot = master.active_as_of("2016-06-30").merge(
        store.snapshot(cutoff), on="security_id", how="inner"
    )
    market, _ = _market_snapshot(frames["prices"], pd.Timestamp("2016-06-30"), snapshot)
    snapshot = snapshot.drop(columns=["market_cap"]).merge(
        market, on="security_id", how="inner"
    )
    ordinary = snapshot["company_domain"].eq("ordinary")
    bank = snapshot["company_domain"].eq("bank")
    snapshot.loc[ordinary | bank, "cet1_ratio"] = 0.01
    scored = calculate_scores(snapshot, cfg, "enterprising")
    assert scored.loc[scored["company_domain"].eq("ordinary"), "base_eligible"].all()
    assert not scored.loc[scored["company_domain"].eq("bank"), "base_eligible"].any()


def test_sector_override_replaces_global_value_factor_set():
    cfg = load_config("config/strategy.yaml")
    ordinary = rules_for(cfg, "ordinary", "Industrials")
    utilities = rules_for(cfg, "ordinary", "Utilities")
    assert "cfo_yield" in ordinary.value_factors
    assert "cfo_yield" not in utilities.value_factors
    assert "dividend_yield" in utilities.value_factors
