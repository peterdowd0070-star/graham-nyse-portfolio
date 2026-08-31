from __future__ import annotations

import numpy as np
import pandas as pd


def cvxpy_minimum_variance(
    covariance: np.ndarray,
    sectors: pd.Series,
    position_cap: float,
    sector_cap: float,
) -> np.ndarray:
    """Convex reference implementation for constrained long-only weights."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "Install CVXPY with pip install -e '.[infrastructure]'"
        ) from exc
    matrix = np.asarray(covariance, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    if minimum_eigenvalue < 0:
        matrix += np.eye(len(matrix)) * (-minimum_eigenvalue + 1e-10)
    weights = cp.Variable(len(matrix))
    constraints = [weights >= 0, weights <= position_cap, cp.sum(weights) == 1]
    sector_values = sectors.astype(str).to_numpy()
    for sector in sorted(set(sector_values)):
        constraints.append(cp.sum(weights[sector_values == sector]) <= sector_cap)
    problem = cp.Problem(cp.Minimize(cp.quad_form(weights, matrix)), constraints)
    problem.solve(solver=cp.CLARABEL)
    if (
        problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        or weights.value is None
    ):
        raise ValueError(
            f"CVXPY minimum-variance optimization failed: {problem.status}"
        )
    result = np.asarray(weights.value, dtype=float)
    return result / result.sum()
