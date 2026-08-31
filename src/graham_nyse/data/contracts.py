from __future__ import annotations

from typing import Any

import pandas as pd


def _pandera() -> Any:
    try:
        import pandera.pandas as pa
    except ImportError as exc:
        raise RuntimeError(
            "Install canonical contract checks with pip install -e '.[infrastructure]'"
        ) from exc
    return pa


def validate_canonical_bundle(
    security_master: pd.DataFrame,
    prices: pd.DataFrame,
    corporate_actions: pd.DataFrame,
) -> None:
    pa = _pandera()
    master_schema = pa.DataFrameSchema(
        {
            "security_id": pa.Column(str, nullable=False),
            "issuer_id": pa.Column(str, nullable=False),
            "ticker": pa.Column(str, nullable=True),
            "exchange": pa.Column(str, nullable=False),
            "security_type": pa.Column(str, nullable=False),
            "company_domain": pa.Column(
                str, checks=pa.Check.isin(["ordinary", "bank", "insurer", "reit"])
            ),
            "sector": pa.Column(str, nullable=False),
            "listing_start": pa.Column(pa.DateTime, nullable=False),
            "listing_end": pa.Column(pa.DateTime, nullable=True),
            "is_delisted": pa.Column(bool, nullable=False),
            "delisting_return": pa.Column(float, nullable=True, coerce=True),
        },
        strict=False,
        coerce=True,
    )
    price_schema = pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, nullable=False),
            "security_id": pa.Column(str, nullable=False),
            "close": pa.Column(
                float, checks=pa.Check.gt(0), nullable=False, coerce=True
            ),
            "volume": pa.Column(
                float, checks=pa.Check.ge(0), nullable=False, coerce=True
            ),
        },
        strict=False,
        coerce=True,
        unique=["date", "security_id"],
    )
    action_schema = pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, nullable=False),
            "security_id": pa.Column(str, nullable=False),
            "action_type": pa.Column(str, checks=pa.Check.isin(["DIVIDEND", "SPLIT"])),
            "value": pa.Column(float, nullable=False, coerce=True),
            "qualified": pa.Column(bool, nullable=False, coerce=True),
        },
        strict=False,
        coerce=True,
    )
    master_schema.validate(security_master, lazy=True)
    price_schema.validate(prices, lazy=True)
    action_schema.validate(corporate_actions, lazy=True)
