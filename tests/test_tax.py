import pandas as pd

from graham_nyse.backtest.tax import TaxLedger
from graham_nyse.config import load_config


def test_long_term_gain_and_terminal_modes():
    cfg = load_config("config/strategy.yaml")
    ledger = TaxLedger(cfg.tax, "taxable_hifo_terminal_liquidation")
    ledger.buy("SEC1", 10, 100, pd.Timestamp("2016-01-01"))
    gain = ledger.sell("SEC1", 5, 120, pd.Timestamp("2017-02-01"))
    assert gain == 100
    assert abs(ledger.settle_year(2017) - 20) < 1e-9
    assert ledger.terminal_liquidation


def test_prospective_wash_sale_adjusts_replacement_basis():
    cfg = load_config("config/strategy.yaml")
    ledger = TaxLedger(cfg.tax, "taxable_fifo_no_liquidation")
    ledger.buy("SEC1", 10, 100, pd.Timestamp("2017-01-01"))
    ledger.sell("SEC1", 10, 90, pd.Timestamp("2017-02-01"))
    ledger.buy("SEC1", 10, 92, pd.Timestamp("2017-02-15"))
    assert ledger.lots["SEC1"][0].basis_per_share == 102
