from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graham_nyse.backtest.engine import run_experiment_matrix
from graham_nyse.config import load_config
from tests.fixtures.generate_evolving_10y import START, END, write_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all scenario/weighting variants on the deterministic evolving-universe fixture."
    )
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--output", default="outputs/evolving_validation_10y")
    parser.add_argument(
        "--tax-mode",
        default="tax_deferred",
        choices=[
            "tax_deferred",
            "taxable_fifo_no_liquidation",
            "taxable_hifo_no_liquidation",
            "taxable_hifo_terminal_liquidation",
        ],
    )
    args = parser.parse_args()

    output = Path(args.output)
    data_dir = output / "data"
    paths = write_frames(data_dir)
    cfg = load_config(args.config)

    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    matrix, results = run_experiment_matrix(
        frames["filing_vintages"],
        frames["security_master"],
        frames["prices"],
        cfg,
        START,
        END,
        corporate_actions=frames["corporate_actions"],
        tax_mode=args.tax_mode,
    )

    output.mkdir(parents=True, exist_ok=True)
    matrix = matrix.sort_values(
        ["cagr", "sharpe_zero_rf"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    matrix.to_csv(output / "scenario_weight_matrix.csv", index=False)

    for (scenario, strategy), result in results.items():
        result.write(output / "runs" / scenario / strategy)

    summary = {
        "evidence_status": "deterministic evolving-universe software validation",
        "historical_market_claim": False,
        "start": str(START.date()),
        "end": str(END.date()),
        "scenario_count": len(cfg.valuation.scenarios),
        "weighting_strategy_count": len(cfg.portfolio.weighting_strategies),
        "run_count": len(matrix),
        "tax_mode": args.tax_mode,
        "best_by_cagr": matrix.iloc[0].to_dict() if not matrix.empty else {},
    }
    (output / "validation_manifest.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(matrix.to_string(index=False))
    print(f"\nWrote {len(matrix)} evolving-universe runs to {output}")


if __name__ == "__main__":
    main()
