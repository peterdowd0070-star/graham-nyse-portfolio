from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from graham_nyse.config import DomainModelConfig, StrategyConfig


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    ranked = clean.rank(pct=True, method="average")
    return ranked if higher_is_better else 1.0 - ranked


def _merge_domain_rules(
    base: DomainModelConfig, override: dict[str, Any]
) -> DomainModelConfig:
    raw = deepcopy(base.model_dump())
    raw["hard_gates"].update(override.get("hard_gates", {}))
    for section in ("value_factors", "quality_factors"):
        if section in override:
            raw[section] = override[section]
    raw["sector_overrides"] = base.sector_overrides
    return DomainModelConfig.model_validate(raw)


def rules_for(cfg: StrategyConfig, domain: str, sector: str) -> DomainModelConfig:
    if domain not in cfg.valuation.domains:
        raise ValueError(f"No valuation model for company domain {domain!r}")
    base = cfg.valuation.domains[domain]
    return _merge_domain_rules(base, base.sector_overrides.get(sector, {}))


def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    market_cap = pd.to_numeric(out.get("market_cap"), errors="coerce")
    enterprise_value = market_cap + pd.to_numeric(
        out.get("net_debt", 0.0), errors="coerce"
    ).clip(lower=0)
    out["earnings_yield"] = (
        pd.to_numeric(out.get("normalized_net_income"), errors="coerce") / market_cap
    )
    out["normalized_earnings_yield"] = out["earnings_yield"]
    out["cfo_yield"] = pd.to_numeric(out.get("cfo"), errors="coerce") / market_cap
    out["fcf_yield"] = pd.to_numeric(out.get("fcf"), errors="coerce") / enterprise_value
    out["book_to_market"] = (
        pd.to_numeric(out.get("equity"), errors="coerce") / market_cap
    )
    out["tangible_book_to_market"] = (
        pd.to_numeric(out.get("tangible_equity"), errors="coerce") / market_cap
    )
    out["leverage"] = pd.to_numeric(
        out.get("net_debt"), errors="coerce"
    ) / pd.to_numeric(out.get("assets"), errors="coerce")
    out["normalized_roe_yield"] = pd.to_numeric(
        out.get("normalized_net_income"), errors="coerce"
    ) / pd.to_numeric(out.get("tangible_equity"), errors="coerce")
    for numerator, result in [
        ("free_surplus", "free_surplus_yield"),
        ("ffo", "ffo_yield"),
        ("affo", "affo_yield"),
        ("annual_dividend", "dividend_yield"),
    ]:
        if result not in out:
            out[result] = (
                pd.to_numeric(out.get(numerator), errors="coerce") / market_cap
            )
    return out.replace([np.inf, -np.inf], np.nan)


def apply_eligibility(
    df: pd.DataFrame, cfg: StrategyConfig, scenario: str = "quality_value"
) -> pd.DataFrame:
    if scenario not in cfg.valuation.scenarios:
        raise ValueError(f"Unknown strategy scenario: {scenario}")
    out = calculate_derived_metrics(df)
    sc = cfg.valuation.scenarios[scenario]
    universal = (
        out["market_cap"].ge(cfg.universe.min_market_cap)
        & out["price"].ge(cfg.universe.min_price)
        & out["median_dollar_volume_60d"].ge(cfg.universe.min_median_dollar_volume_60d)
        & out["earnings_history_years"].ge(sc.minimum_history_years)
        & (
            out["positive_earnings_years"]
            / out["earnings_history_years"].replace(0, np.nan)
        ).ge(sc.minimum_positive_year_ratio)
    )
    gate_pass = pd.Series(True, index=out.index)
    gate_reasons: dict[int, list[str]] = {int(i): [] for i in out.index}
    for (domain, sector), indexes in out.groupby(
        ["company_domain", "sector"], dropna=False
    ).groups.items():
        rules = rules_for(cfg, str(domain), str(sector))
        for metric, gate in rules.hard_gates.items():
            values = (
                pd.to_numeric(out.loc[indexes, metric], errors="coerce")
                if metric in out
                else pd.Series(np.nan, index=indexes)
            )
            passed = pd.Series(True, index=indexes)
            if gate.required:
                passed &= values.notna()
            if gate.minimum is not None:
                passed &= values.ge(gate.minimum) | (
                    (not gate.required) & values.isna()
                )
            if gate.maximum is not None:
                passed &= values.le(gate.maximum) | (
                    (not gate.required) & values.isna()
                )
            gate_pass.loc[indexes] &= passed
            for idx in passed.index[~passed]:
                gate_reasons[int(idx)].append(metric)
    out["eligible"] = (universal & gate_pass).fillna(False)
    out["base_eligible"] = out["eligible"]
    out["gate_failures"] = [gate_reasons[int(i)] for i in out.index]
    return out


def _factor_score(
    out: pd.DataFrame,
    indexes: pd.Index,
    peers: pd.Index,
    factors: dict[str, Any],
    winsor_lower: float,
    winsor_upper: float,
) -> tuple[pd.Series, pd.Series]:
    numerator = pd.Series(0.0, index=indexes)
    used_weight = pd.Series(0.0, index=indexes)
    total_weight = sum(rule.weight for rule in factors.values())
    for metric, rule in factors.items():
        if metric not in out:
            continue
        peer_values = pd.to_numeric(out.loc[peers, metric], errors="coerce")
        lo, hi = (
            peer_values.quantile([winsor_lower, winsor_upper])
            if peer_values.notna().any()
            else (np.nan, np.nan)
        )
        ranked = percentile(peer_values.clip(lower=lo, upper=hi), rule.higher_is_better)
        local = ranked.reindex(indexes)
        present = local.notna()
        numerator.loc[present] += rule.weight * local.loc[present]
        used_weight.loc[present] += rule.weight
    score = numerator / used_weight.replace(0, np.nan)
    confidence = used_weight / max(total_weight, 1e-12)
    return score, confidence


def calculate_scores(
    df: pd.DataFrame, cfg: StrategyConfig, scenario: str = "quality_value"
) -> pd.DataFrame:
    out = apply_eligibility(df, cfg, scenario)
    out["value_score"] = np.nan
    out["quality_score"] = np.nan
    out["data_confidence"] = 0.0
    for (domain, sector), indexes_raw in out.groupby(
        ["company_domain", "sector"], dropna=False
    ).groups.items():
        indexes = pd.Index(indexes_raw)
        domain_peers = out.index[out["company_domain"].eq(domain)]
        peers = (
            indexes
            if len(indexes) >= cfg.fundamentals.minimum_group_size
            else domain_peers
        )
        rules = rules_for(cfg, str(domain), str(sector))
        value, value_confidence = _factor_score(
            out,
            indexes,
            peers,
            rules.value_factors,
            cfg.fundamentals.winsor_lower,
            cfg.fundamentals.winsor_upper,
        )
        quality, quality_confidence = _factor_score(
            out,
            indexes,
            peers,
            rules.quality_factors,
            cfg.fundamentals.winsor_lower,
            cfg.fundamentals.winsor_upper,
        )
        out.loc[indexes, "value_score"] = value
        out.loc[indexes, "quality_score"] = quality
        out.loc[indexes, "data_confidence"] = (
            value_confidence + quality_confidence
        ) / 2.0

    out["stability_score"] = 0.5
    if "earnings_stability" in out:
        for indexes in out.groupby("company_domain").groups.values():
            out.loc[indexes, "stability_score"] = percentile(
                out.loc[indexes, "earnings_stability"], False
            )
    out["value_percentile"] = out.groupby("company_domain")["value_score"].rank(
        pct=True
    )
    out["quality_percentile"] = out.groupby("company_domain")["quality_score"].rank(
        pct=True
    )
    sc = cfg.valuation.scenarios[scenario]
    out["eligible"] = out["base_eligible"] & (
        out["value_percentile"].ge(sc.minimum_value_percentile)
        & out["quality_percentile"].ge(sc.minimum_quality_percentile)
    ).fillna(False)
    eps = 1e-6
    out["graham_score"] = np.exp(
        sc.value_weight * np.log(out["value_score"].clip(lower=eps))
        + sc.quality_weight * np.log(out["quality_score"].clip(lower=eps))
        + sc.stability_weight * np.log(out["stability_score"].clip(lower=eps))
        + sc.confidence_weight * np.log(out["data_confidence"].clip(lower=eps))
    )
    out.loc[~out["base_eligible"], "graham_score"] = np.nan
    out["scenario"] = scenario
    out["rank"] = out["graham_score"].rank(ascending=False, method="first")
    return out.sort_values("rank", na_position="last")
