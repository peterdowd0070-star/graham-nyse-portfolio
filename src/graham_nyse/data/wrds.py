from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from graham_nyse.data.classification import classify_crsp_name_history
from graham_nyse.data.provenance import write_source_manifest
from graham_nyse.data.security_master import security_master_from_crsp


class SqlConnection(Protocol):
    def raw_sql(self, sql: str, date_cols: list[str] | None = None) -> pd.DataFrame: ...


@dataclass(frozen=True)
class CrspTables:
    monthly_stock: str = "crsp.dsf"
    stock_names: str = "crsp.dsenames"
    delistings: str = "crsp.dsedelist"
    distributions: str = "crsp.dsedist"
    indexes: str = "crsp.dsi"
    ccm_links: str = "crsp.ccmxpf_linktable"
    compustat_company: str = "comp.company"


def _iso_date(value: str) -> str:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.strftime("%Y-%m-%d")


def _table(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*", value):
        raise ValueError(f"Unsafe WRDS table identifier: {value!r}")
    return value


def extract_crsp(
    connection: SqlConnection,
    start: str,
    end: str,
    output_dir: str | Path,
    tables: CrspTables | None = None,
) -> dict[str, Path]:
    """Extract survivorship-free CRSP inputs without embedding credentials.

    Table names are configurable because some WRDS subscriptions expose CRSP's
    newer CIZ schema rather than the legacy SIZ names used by default here.
    """
    tables = tables or CrspTables()
    start_date, end_date = _iso_date(start), _iso_date(end)
    warmup = (pd.Timestamp(start_date) - pd.Timedelta("400D")).strftime("%Y-%m-%d")
    msf = _table(tables.monthly_stock)
    names_table = _table(tables.stock_names)
    delist_table = _table(tables.delistings)
    dist_table = _table(tables.distributions)
    index_table = _table(tables.indexes)
    link_table = _table(tables.ccm_links)
    company_table = _table(tables.compustat_company)

    names = connection.raw_sql(
        f"""
        select permno, permco, ticker, comnam, exchcd, shrcd, siccd,
               namedt, nameendt
        from {names_table}
        where nameendt >= '{warmup}' and namedt <= '{end_date}'
        """,
        date_cols=["namedt", "nameendt"],
    )
    delistings = connection.raw_sql(
        f"""
        select permno, dlstdt, dlstcd, dlprc, dlret, dlretx
        from {delist_table}
        where dlstdt between '{warmup}' and '{end_date}'
        """,
        date_cols=["dlstdt"],
    )
    prices = connection.raw_sql(
        f"""
        select date, permno, abs(prc) as close, vol as volume,
               shrout * 1000.0 as shares_outstanding, ret, retx
        from {msf}
        where date between '{warmup}' and '{end_date}'
        """,
        date_cols=["date"],
    )
    distributions = connection.raw_sql(
        f"""
        select permno, exdt, distcd, divamt, facpr, facshr, dclrdt, rcrddt, paydt
        from {dist_table}
        where exdt between '{warmup}' and '{end_date}'
        """,
        date_cols=["exdt", "dclrdt", "rcrddt", "paydt"],
    )
    indexes = connection.raw_sql(
        f"""
        select date, vwretd, ewretd, sprtrn
        from {index_table}
        where date between '{start_date}' and '{end_date}'
        """,
        date_cols=["date"],
    )
    identifier_links = connection.raw_sql(
        f"""
        select l.lpermno as permno, l.gvkey, c.cik, l.linkdt, l.linkenddt,
               l.linktype, l.linkprim
        from {link_table} l
        join {company_table} c on c.gvkey = l.gvkey
        where coalesce(l.linkenddt, '9999-12-31') >= '{warmup}'
          and l.linkdt <= '{end_date}'
          and l.linktype in ('LC', 'LU')
          and l.linkprim in ('P', 'C')
          and c.cik is not null
        """,
        date_cols=["linkdt", "linkenddt"],
    )

    for frame in (names, delistings, prices, distributions, indexes, identifier_links):
        frame.columns = frame.columns.str.upper()
    classified_names = classify_crsp_name_history(names)
    classifications = classified_names[["PERMNO", "company_domain", "sector"]]
    classifications = classifications.drop_duplicates("PERMNO", keep="last")
    master = security_master_from_crsp(
        classified_names, delistings, classifications
    ).frame

    prices = prices.rename(columns={"PERMNO": "permno", "DATE": "date"})
    prices["security_id"] = "CRSP:" + prices["permno"].astype(int).astype(str)
    raw_prices = prices.rename(
        columns={
            "CLOSE": "close",
            "VOLUME": "volume",
            "SHARES_OUTSTANDING": "shares_outstanding",
            "RET": "vendor_total_return",
            "RETX": "vendor_price_return",
        }
    )[
        [
            "date",
            "security_id",
            "close",
            "volume",
            "shares_outstanding",
            "vendor_total_return",
            "vendor_price_return",
        ]
    ]

    action_rows: list[dict[str, Any]] = []
    for row in distributions.itertuples(index=False):
        security_id = f"CRSP:{int(row.PERMNO)}"
        if pd.notna(row.DIVAMT) and float(row.DIVAMT) != 0:
            action_rows.append(
                {
                    "date": row.EXDT,
                    "security_id": security_id,
                    "action_type": "DIVIDEND",
                    "value": float(row.DIVAMT),
                    "qualified": False,
                    "distcd": row.DISTCD,
                }
            )
        split_factor = row.FACSHR if pd.notna(row.FACSHR) else row.FACPR
        if pd.notna(split_factor) and float(split_factor) != 0:
            action_rows.append(
                {
                    "date": row.EXDT,
                    "security_id": security_id,
                    "action_type": "SPLIT",
                    "value": 1.0 + float(split_factor),
                    "qualified": False,
                    "distcd": row.DISTCD,
                }
            )
    actions = pd.DataFrame(
        action_rows,
        columns=["date", "security_id", "action_type", "value", "qualified", "distcd"],
    )
    benchmark = indexes.rename(
        columns={
            "DATE": "date",
            "VWRETD": "CRSP_VW",
            "EWRETD": "CRSP_EW",
            "SPRTRN": "SP500",
        }
    ).melt(id_vars="date", var_name="benchmark", value_name="total_return")
    identifier_links = identifier_links.rename(
        columns={
            "PERMNO": "permno",
            "GVKEY": "gvkey",
            "CIK": "cik",
            "LINKDT": "valid_from",
            "LINKENDDT": "valid_to",
            "LINKTYPE": "link_type",
            "LINKPRIM": "link_primary",
        }
    )
    identifier_links["security_id"] = "CRSP:" + identifier_links["permno"].astype(
        int
    ).astype(str)
    for frame in (master, raw_prices, actions, benchmark, identifier_links):
        frame["data_provider"] = "crsp"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "security_master": out / "security_master.parquet",
        "prices": out / "raw_prices.parquet",
        "corporate_actions": out / "corporate_actions.parquet",
        "benchmarks": out / "benchmark_total_returns.parquet",
        "identifier_links": out / "identifier_links.parquet",
    }
    master.to_parquet(paths["security_master"], index=False)
    raw_prices.to_parquet(paths["prices"], index=False)
    actions.to_parquet(paths["corporate_actions"], index=False)
    benchmark.to_parquet(paths["benchmarks"], index=False)
    identifier_links.to_parquet(paths["identifier_links"], index=False)
    write_source_manifest(
        out,
        source="WRDS/CRSP",
        parameters={"start": start_date, "end": end_date, "tables": tables.__dict__},
        files=list(paths.values()),
    )
    return paths


def connect_wrds(username: str | None = None) -> SqlConnection:
    try:
        import wrds  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    return wrds.Connection(wrds_username=username)
