import pandas as pd

from graham_nyse.backtest.engine import run_experiment_matrix, run_historical_backtest
from graham_nyse.config import load_config
from tests.fixtures.generate_simulation_smoke_test import build_smoke_frames


def test_historical_engine_uses_vintages_and_monthly_updates():
    cfg = load_config("config/strategy.yaml")
    frames = build_smoke_frames()
    result = run_historical_backtest(
        frames["filing_vintages"],
        frames["security_master"],
        frames["prices"],
        cfg,
        "2016-07-01",
        "2018-06-29",
        scenario="quality_value",
        weighting_strategy="equal",
        corporate_actions=frames["corporate_actions"],
    )
    assert result.audit["passed"]
    assert not result.nav.empty
    assert {
        "initial_construction",
        "quarterly_rebalance",
        "full_reconstruction",
    }.issubset(set(result.holdings["run_type"]))
    assert result.snapshots["new_filing_count"].gt(0).any()
    assert (
        pd.to_datetime(result.snapshots["maximum_accepted_at"], utc=True)
        <= pd.to_datetime(result.snapshots["decision_at"], utc=True)
    ).all()
    assert result.metadata["uses_current_constituent_list"] is False
    assert result.metadata["uses_adjusted_prices"] is False
    assert result.metadata["publication_status"] == "simulation_only"
    assert "delisting" in set(result.trades["run_type"])


def test_all_four_scenarios_by_all_six_weight_strategies():
    cfg = load_config("config/strategy.yaml")
    frames = build_smoke_frames()
    matrix, _ = run_experiment_matrix(
        frames["filing_vintages"],
        frames["security_master"],
        frames["prices"],
        cfg,
        "2016-07-01",
        "2016-12-30",
        corporate_actions=frames["corporate_actions"],
    )
    assert len(matrix) == 24
    assert set(matrix["scenario"]) == {
        "defensive",
        "enterprising",
        "deep_value",
        "quality_value",
    }
    assert set(matrix["weighting_strategy"]) == set(cfg.portfolio.weighting_strategies)


def test_portfolio_funded_taxes_never_create_implicit_borrowing():
    cfg = load_config("config/strategy.yaml")
    frames = build_smoke_frames()
    result = run_historical_backtest(
        frames["filing_vintages"],
        frames["security_master"],
        frames["prices"],
        cfg,
        "2016-07-01",
        "2018-06-29",
        scenario="quality_value",
        weighting_strategy="equal",
        tax_mode="taxable_fifo_no_liquidation",
        tax_payment_source="portfolio",
        corporate_actions=frames["corporate_actions"],
    )
    assert result.nav["cash"].ge(-cfg.validation.nav_tolerance).all()
    assert "negative_cash_from_tax_or_execution" not in result.audit["errors"]
