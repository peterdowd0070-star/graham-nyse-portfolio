from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class UniverseConfig(BaseModel):
    exchange: str = "NYSE"
    include_adrs: bool = False
    min_market_cap: float = 250_000_000
    min_price: float = 3.0
    min_median_dollar_volume_60d: float = 1_000_000
    exclude_name_patterns: list[str] = Field(default_factory=list)
    require_delisting_returns: bool = True
    maximum_price_staleness_days: int = 5


class FundamentalConfig(BaseModel):
    history_years: int = 10
    winsor_lower: float = 0.025
    winsor_upper: float = 0.975
    minimum_group_size: int = 5


class FactorRule(BaseModel):
    weight: float
    higher_is_better: bool = True


class GateRule(BaseModel):
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True


class DomainModelConfig(BaseModel):
    hard_gates: dict[str, GateRule]
    value_factors: dict[str, FactorRule]
    quality_factors: dict[str, FactorRule]
    sector_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    value_weight: float
    quality_weight: float
    stability_weight: float
    confidence_weight: float
    minimum_positive_year_ratio: float = 0.0
    minimum_history_years: int = 1
    minimum_value_percentile: float = 0.0
    minimum_quality_percentile: float = 0.0

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> ScenarioConfig:
        total = (
            self.value_weight
            + self.quality_weight
            + self.stability_weight
            + self.confidence_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Scenario score weights must sum to one")
        return self


class ValuationConfig(BaseModel):
    domains: dict[str, DomainModelConfig]
    scenarios: dict[str, ScenarioConfig]


WeightStrategy = Literal[
    "equal",
    "score_proportional",
    "inverse_volatility",
    "score_over_volatility",
    "minimum_variance",
    "liquidity_adjusted_equal",
]


def default_weighting_strategies() -> list[WeightStrategy]:
    return [
        "equal",
        "score_proportional",
        "inverse_volatility",
        "score_over_volatility",
        "minimum_variance",
        "liquidity_adjusted_equal",
    ]


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
    weighting_strategies: list[WeightStrategy] = Field(
        default_factory=default_weighting_strategies
    )


class ScheduleConfig(BaseModel):
    quarterly_rebalance_months: list[int] = Field(default_factory=lambda: [3, 9])
    full_reconstruction_months: list[int] = Field(default_factory=lambda: [6, 12])
    monitor_monthly: bool = True


class ExecutionConfig(BaseModel):
    transaction_cost_bps: float = 10.0
    decision_time: str = "16:00:00"
    execute_next_session: bool = True
    minimum_history_days_for_risk: int = 60
    covariance_lookback_days: int = 252


class TaxRatesConfig(BaseModel):
    short_term: float = 0.37
    long_term: float = 0.20
    qualified_dividend: float = 0.20
    ordinary_dividend: float = 0.37


class TaxConfig(BaseModel):
    rates: TaxRatesConfig = Field(default_factory=TaxRatesConfig)
    wash_sale_days: int = 30
    long_term_days: int = 365
    payment_source: Literal["portfolio", "external"] = "portfolio"
    modes: list[str] = Field(
        default_factory=lambda: [
            "tax_deferred",
            "taxable_fifo_no_liquidation",
            "taxable_hifo_no_liquidation",
            "taxable_hifo_terminal_liquidation",
        ]
    )


class ValidationConfig(BaseModel):
    maximum_missing_required_rate: float = 0.0
    accounting_tolerance: float = 0.02
    nav_tolerance: float = 1e-6
    fail_on_temporal_violation: bool = True


class StrategyConfig(BaseModel):
    universe: UniverseConfig
    fundamentals: FundamentalConfig
    valuation: ValuationConfig
    portfolio: PortfolioConfig
    schedule: ScheduleConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    tax: TaxConfig = Field(default_factory=TaxConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


def load_config(path: str | Path) -> StrategyConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return StrategyConfig.model_validate(raw)
