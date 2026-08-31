from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_openbb_pit_statement(
    result: Any,
    *,
    security_id: str,
    accession_column: str = "accession_number",
) -> pd.DataFrame:
    """Normalize an OpenBB SEC `pit_mode=True` result to filing vintages.

    This adapter requires accepted timestamps and accession identifiers. It
    refuses provider output that cannot prove when information became public.
    """
    if isinstance(result, pd.DataFrame):
        frame = result.copy()
    elif hasattr(result, "to_dataframe"):
        frame = result.to_dataframe().copy()
    elif hasattr(result, "to_df"):
        frame = result.to_df().copy()
    else:
        frame = pd.DataFrame(result)
    aliases = {
        "accepted_date": "accepted_at",
        "period_ending": "period_end",
        "filing_date": "filed_at",
    }
    frame = frame.rename(columns=aliases)
    required = {"accepted_at", "period_end", accession_column}
    if missing := required - set(frame):
        raise ValueError(
            "OpenBB SEC output is not point-in-time auditable; missing "
            f"{sorted(missing)}"
        )
    frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], utc=True)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    frame["security_id"] = str(security_id)
    if accession_column != "accession_number":
        frame = frame.rename(columns={accession_column: "accession_number"})
    if frame.duplicated(["security_id", "accession_number"]).any():
        raise ValueError("OpenBB SEC result contains duplicate filing accessions")
    return frame
