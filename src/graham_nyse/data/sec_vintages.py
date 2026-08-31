from __future__ import annotations

from pathlib import Path

import pandas as pd

from graham_nyse.data.fundamentals import extract_fundamental_vintages
from graham_nyse.data.provenance import write_source_manifest
from graham_nyse.data.sec import SecClient


def acquire_sec_filing_vintages(
    links: pd.DataFrame,
    output_dir: str | Path,
    *,
    user_agent: str,
    start: str,
    end: str,
    refresh: bool = False,
    maximum_issuers: int | None = None,
) -> Path:
    """Create acceptance-time fundamental vintages for dated CRSP-CIK links."""
    required = {"security_id", "cik", "valid_from", "valid_to"}
    if missing := required - set(links):
        raise ValueError(f"Identifier link history is missing: {sorted(missing)}")
    frame = links.copy()
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce")
    frame["valid_from"] = pd.to_datetime(frame["valid_from"], errors="coerce")
    frame["valid_to"] = pd.to_datetime(frame["valid_to"], errors="coerce")
    frame = frame.dropna(subset=["security_id", "cik", "valid_from"])
    ciks = sorted(frame["cik"].astype(int).unique())
    if maximum_issuers is not None:
        ciks = ciks[:maximum_issuers]

    client = SecClient(user_agent, cache_dir=Path(output_dir) / "raw_sec")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    pieces: list[pd.DataFrame] = []
    for cik in ciks:
        submissions = client.submissions_all(cik, refresh)
        facts = client.company_facts(cik, refresh)
        linked = frame.loc[frame["cik"].eq(cik)]
        ticker = str(facts.get("entityName", cik))
        for link in linked.itertuples(index=False):
            vintages = extract_fundamental_vintages(
                str(link.security_id), ticker, cik, facts, submissions
            )
            if vintages.empty:
                continue
            accepted_naive = vintages["accepted_at"].dt.tz_convert(None)
            link_end = link.valid_to if pd.notna(link.valid_to) else pd.Timestamp.max
            keep = accepted_naive.between(
                max(start_ts - pd.Timedelta("3660D"), link.valid_from),
                min(end_ts, link_end),
            )
            pieces.append(vintages.loc[keep])

    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "filing_vintages.parquet"
    result.to_parquet(target, index=False)
    write_source_manifest(
        out,
        source="SEC EDGAR submissions and Company Facts",
        parameters={
            "start": str(start_ts.date()),
            "end": str(end_ts.date()),
            "issuer_count": len(ciks),
        },
        files=[target],
    )
    return target
