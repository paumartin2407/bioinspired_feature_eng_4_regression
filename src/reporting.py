"""Descriptive summaries only; this module never chooses a pipeline."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(runs):
    frame = pd.read_csv(runs)
    frame = frame.loc[frame.arm != "no_fe"]
    measures = ["mae", "mae_reduction_pct", "rmse", "r2", "selected_k", "transformed_dim",
                "candidate_evaluations", "fe_model_fits", "failed_candidates", "outer_failure",
                "runtime_seconds"]
    measures = [name for name in measures if name in frame]
    split = frame.groupby(["dataset", "learner", "arm", "split"], as_index=False)[measures].mean()
    dataset = split.groupby(["dataset", "learner", "arm"], as_index=False)[measures].mean()
    rows = []
    for (model, arm), group in dataset.groupby(["learner", "arm"], sort=False):
        reductions = group.mae_reduction_pct.dropna().to_numpy()
        if len(reductions):
            rng = np.random.default_rng(20261014)
            boot = rng.choice(reductions, size=(2000, len(reductions)), replace=True).mean(axis=1)
            interval = np.percentile(boot, [2.5, 97.5])
        else:
            interval = [float("nan"), float("nan")]
        rows.append({"learner": model, "arm": arm, "datasets": len(group),
                     "mean_dataset_mae_reduction_pct": float(np.mean(reductions)) if len(reductions) else float("nan"),
                     "bootstrap_low_pct": float(interval[0]), "bootstrap_high_pct": float(interval[1]),
                     "fraction_datasets_helped": float(np.mean(reductions > 0)) if len(reductions) else float("nan"),
                     "mean_outer_failure_fraction": float(group.outer_failure.mean()),
                     "mean_selected_k": float(group.selected_k.mean()),
                     "mean_runtime_seconds": float(group.runtime_seconds.mean())})
    return split, dataset, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    split, dataset, aggregate = summarize(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output / "split_summary.csv", index=False)
    dataset.to_csv(args.output / "dataset_summary.csv", index=False)
    aggregate.to_csv(args.output / "aggregate_summary.csv", index=False)


if __name__ == "__main__":
    main()
