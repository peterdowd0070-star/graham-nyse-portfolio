from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target)
    return pd.read_csv(target)


load_filing_vintages = load_table
load_security_master = load_table
load_price_panel = load_table
load_corporate_actions = load_table
load_benchmark_returns = load_table
load_factor_returns = load_table


def load_feature_panel(path: str | Path) -> pd.DataFrame:
    raise RuntimeError(
        "Feature panels with as_of_date are no longer accepted. "
        "Provide immutable filing vintages keyed by accession_number and accepted_at."
    )


def download_adjusted_prices(*args: object, **kwargs: object) -> pd.DataFrame:
    raise RuntimeError(
        "Adjusted-price downloads were removed from the historical engine. "
        "Provide raw closes plus explicit dividends, splits, inactive securities, and delisting returns."
    )
