from __future__ import annotations

from pathlib import Path

import pandas as pd

from graham_nyse.data.classification import (
    broad_sector_from_sic,
    company_domain_from_sic,
)
from graham_nyse.data.wrds import extract_crsp


class FakeWrds:
    def raw_sql(self, sql: str, date_cols: list[str] | None = None) -> pd.DataFrame:
        del date_cols
        if "dsenames" in sql:
            return pd.DataFrame(
                [
                    [
                        10001,
                        9001,
                        "OLD",
                        "Old Bank",
                        1,
                        10,
                        6021,
                        "2010-01-01",
                        "2018-03-01",
                    ],
                    [
                        10002,
                        9002,
                        "REIT",
                        "Old REIT",
                        1,
                        11,
                        6798,
                        "2011-01-01",
                        "2026-12-31",
                    ],
                ],
                columns=[
                    "permno",
                    "permco",
                    "ticker",
                    "comnam",
                    "exchcd",
                    "shrcd",
                    "siccd",
                    "namedt",
                    "nameendt",
                ],
            )
        if "dsedelist" in sql:
            return pd.DataFrame(
                [[10001, "2018-03-01", 500, 4.0, -0.7, -0.7]],
                columns=["permno", "dlstdt", "dlstcd", "dlprc", "dlret", "dlretx"],
            )
        if "from crsp.dsf" in sql:
            return pd.DataFrame(
                [["2017-12-29", 10001, 10.0, 1000, 1_000_000, 0.01, 0.01]],
                columns=[
                    "date",
                    "permno",
                    "close",
                    "volume",
                    "shares_outstanding",
                    "ret",
                    "retx",
                ],
            )
        if "dsedist" in sql:
            return pd.DataFrame(
                [[10001, "2017-12-29", 1232, 0.25, None, None, None, None, None]],
                columns=[
                    "permno",
                    "exdt",
                    "distcd",
                    "divamt",
                    "facpr",
                    "facshr",
                    "dclrdt",
                    "rcrddt",
                    "paydt",
                ],
            )
        if "from crsp.dsi" in sql:
            return pd.DataFrame(
                [["2017-12-29", 0.01, 0.02, 0.009]],
                columns=["date", "vwretd", "ewretd", "sprtrn"],
            )
        if "ccmxpf_linktable" in sql:
            return pd.DataFrame(
                [[10001, "001000", 1234, "2010-01-01", None, "LC", "P"]],
                columns=[
                    "permno",
                    "gvkey",
                    "cik",
                    "linkdt",
                    "linkenddt",
                    "linktype",
                    "linkprim",
                ],
            )
        raise AssertionError(sql)


def test_dated_sic_classification_is_domain_specific():
    assert company_domain_from_sic(6021) == "bank"
    assert company_domain_from_sic(6331) == "insurer"
    assert company_domain_from_sic(6798) == "reit"
    assert broad_sector_from_sic(4911) == "Utilities"


def test_wrds_extract_keeps_delisted_security_and_provenance(tmp_path: Path):
    paths = extract_crsp(FakeWrds(), "2017-01-01", "2018-12-31", tmp_path)
    master = pd.read_parquet(paths["security_master"])
    old = master.loc[master["security_id"].eq("CRSP:10001")].iloc[0]
    assert old["company_domain"] == "bank"
    assert bool(old["is_delisted"])
    assert old["delisting_return"] == -0.7
    assert (tmp_path / "source_manifest.json").exists()
