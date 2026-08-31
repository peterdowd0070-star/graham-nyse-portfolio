from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class HistoricalBundle:
    provider: str
    security_master: pd.DataFrame
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    identifier_links: pd.DataFrame | None = None
    benchmarks: pd.DataFrame | None = None

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        frames = {
            "security_master": self.security_master,
            "raw_prices": self.prices,
            "corporate_actions": self.corporate_actions,
        }
        if self.identifier_links is not None:
            frames["identifier_links"] = self.identifier_links
        if self.benchmarks is not None:
            frames["benchmark_total_returns"] = self.benchmarks
        paths: dict[str, Path] = {}
        for name, frame in frames.items():
            frame = frame.copy()
            frame["data_provider"] = self.provider
            paths[name] = out / f"{name}.parquet"
            frame.to_parquet(paths[name], index=False)
        return paths


class HistoricalProvider(Protocol):
    name: str

    def load(self, start: str, end: str) -> HistoricalBundle: ...


def require_single_provider(*bundles: HistoricalBundle) -> str:
    providers = {bundle.provider for bundle in bundles}
    if len(providers) != 1:
        raise ValueError(
            "An empirical run may not splice market observations across providers: "
            f"{sorted(providers)}"
        )
    return next(iter(providers))
