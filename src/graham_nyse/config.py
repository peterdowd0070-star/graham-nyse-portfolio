from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class UniverseConfig(BaseModel):
    exchange: str = "NYSE"
    include_adrs: bool = False
    min_market_cap: float = 250_000_000
    min_price: float = 3.0
    min_median_dollar_volume_60d: float = 1_000_000
    exclude_name_patterns: list[str] = Field(default_factory=list)


class FundamentalConfig(BaseModel):
    min_positive_years: int = 8
    history_years: int = 10
    min_interest_coverage: float = 3.0
    max_net_debt_to_ebitda: float = 3.0
    require_positive_cfo: bool = True
    winsor_lower: float = 0.025
    winsor_upper: float = 0.975


class ValuationConfig(BaseModel):
    value_weights: dict[str, float]
    quality_weights: dict[str, float]
    score_weights: dict[str, float]


class PortfolioConfig(BaseModel):
    capital: float = 5_000.0
    target_positions: int = 30
    max_position_weight: float = 0.06
    max_sector_weight: float = 0.25
    fractional_shares: bool = True
    minimum_trade_dollars: float = 10.0
    entry_rank: int = 25
    exit_rank: int = 40
    turnover_limit: float = 0.25


class ScheduleConfig(BaseModel):
    quarterly_rebalance_months: list[int] = Field(default_factory=lambda: [3, 9])
    full_reconstruction_months: list[int] = Field(default_factory=lambda: [6, 12])


class StrategyConfig(BaseModel):
    universe: UniverseConfig
    fundamentals: FundamentalConfig
    valuation: ValuationConfig
    portfolio: PortfolioConfig
    schedule: ScheduleConfig


def load_config(path: str | Path) -> StrategyConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return StrategyConfig.model_validate(raw)
