from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

T = TypeVar("T")


def configure_logging(json_output: bool = True) -> Any:
    try:
        import structlog
    except ImportError as exc:
        raise RuntimeError(
            "Install logging with pip install -e '.[infrastructure]'"
        ) from exc
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(processors=processors)
    return structlog.get_logger("graham_nyse")


def retry_data_request(function: Callable[..., T]) -> Callable[..., T]:
    try:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential_jitter,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install retries with pip install -e '.[infrastructure]'"
        ) from exc
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=0.5, max=20),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )(function)


def nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise RuntimeError(
            "Install exchange calendars with pip install -e '.[infrastructure]'"
        ) from exc
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return pd.DatetimeIndex(sessions).tz_localize(None).normalize()


@dataclass
class HistoricalLake:
    root: Path
    database_path: Path | None = None

    def __init__(self, root: str | Path, database_path: str | Path | None = None):
        self.root = Path(root)
        self.database_path = Path(database_path) if database_path else None

    def connect(self) -> Any:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "Install DuckDB with pip install -e '.[infrastructure]'"
            ) from exc
        database = str(self.database_path) if self.database_path else ":memory:"
        connection = duckdb.connect(database)
        for name in (
            "security_master",
            "raw_prices",
            "corporate_actions",
            "identifier_links",
            "filing_vintages",
            "benchmark_total_returns",
        ):
            matches = list(self.root.rglob(f"{name}.parquet"))
            if not matches:
                continue
            paths = ", ".join(
                "'" + str(path).replace("'", "''") + "'" for path in matches
            )
            connection.execute(
                f'create or replace view "{name}" as select * from read_parquet([{paths}], union_by_name=true)'
            )
        return connection

    def scan_prices(self) -> Any:
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError(
                "Install Polars with pip install -e '.[infrastructure]'"
            ) from exc
        paths = [str(path) for path in self.root.rglob("raw_prices.parquet")]
        if not paths:
            raise FileNotFoundError(f"No raw_prices.parquet under {self.root}")
        return pl.scan_parquet(paths)
