"""Compare matched runs from the original and equal-budget protocols."""
import argparse
from pathlib import Path

import pandas as pd


def compare(original, corrected, output):
    keys = ["dataset", "split", "method", "optimizer_seed"]
    old = pd.read_csv(Path(original) / "raw_results.csv")
    new = pd.read_csv(Path(corrected) / "raw_results.csv")
    matched = old.merge(new, on=keys, suffixes=("_original", "_equal"), validate="one_to_one")
    for metric in ("MAE", "RMSE", "R2"):
        matched[f"delta_{metric}"] = matched[f"{metric}_equal"] - matched[f"{metric}_original"]
    matched.to_csv(output, index=False)
    return matched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/equal_budget/budget_comparison.csv"))
    args = parser.parse_args()
    compared = compare(args.original, args.corrected, args.output)
    print(f"Compared {len(compared)} matched runs in {args.output}")
