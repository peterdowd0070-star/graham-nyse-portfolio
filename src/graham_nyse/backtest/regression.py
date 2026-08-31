from __future__ import annotations

from typing import Any

import pandas as pd


def robust_factor_regression(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor_columns: list[str],
    risk_free_column: str = "RF",
) -> dict[str, float]:
    """Statsmodels OLS with HAC errors for research inference."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError(
            "Install regression tooling with pip install -e '.[infrastructure]'"
        ) from exc
    joined = (
        factor_returns[factor_columns + [risk_free_column]]
        .join(portfolio_returns.rename("portfolio"), how="inner")
        .dropna()
    )
    if len(joined) < max(30, len(factor_columns) * 5):
        raise ValueError("Insufficient aligned observations for factor regression")
    dependent = joined["portfolio"] - joined[risk_free_column]
    design = sm.add_constant(joined[factor_columns])
    fitted: Any = sm.OLS(dependent, design).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    output = {
        "alpha_daily": float(fitted.params["const"]),
        "alpha_pvalue": float(fitted.pvalues["const"]),
        "r_squared": float(fitted.rsquared),
        "observations": float(fitted.nobs),
    }
    for factor in factor_columns:
        output[f"beta_{factor.lower()}"] = float(fitted.params[factor])
        output[f"pvalue_{factor.lower()}"] = float(fitted.pvalues[factor])
    return output
