"""Milestone 2: python benchmark_continuous.py [--output results/milestone2]."""

import argparse
import json
import platform
import time
from itertools import combinations
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from benchmark import OPTIMIZER_SEEDS, OUTER_SEEDS
from continuous_features import LOWER_BOUND, UPPER_BOUND, construct, make_fitness
from continuous_optimizers import METHODS, optimize
from data import DATASETS, load_dataset, split_and_preprocess
from evaluation import regression_metrics


BUDGET = 1000
FEATURE_NAMES = ("squared_1", "squared_2")


def run_one(dataset, split_number, outer_seed, optimizer_seed, method, parts, budget):
    train, val, test, y_train, y_val, y_test = parts
    started = time.perf_counter()
    result = optimize(method, make_fitness(train, val, y_train, y_val), train.shape[1],
                      optimizer_seed, budget)
    X_train = construct(train, result.weights)
    X_val = construct(val, result.weights)
    X_test = construct(test, result.weights)
    model = LinearRegression().fit(X_train, y_train)
    predicted = model.predict(X_test)
    residuals = np.asarray(y_test) - predicted
    coefficients = np.ravel(model.coef_)
    names = list(train.columns) + list(FEATURE_NAMES)
    key = {"dataset": dataset, "split": split_number, "outer_seed": outer_seed,
           "optimizer_seed": optimizer_seed, "method": method}
    row = {**key, **regression_metrics(y_test, predicted),
           "validation_MAE": float(mean_absolute_error(y_val, model.predict(X_val))),
           "runtime_seconds": time.perf_counter() - started,
           "fitness_evaluations": result.evaluations,
           "max_abs_test_residual": float(np.abs(residuals).max())}
    trace = [{**key, "evaluations": count, "best_validation_MAE": score}
             for count, score in result.history]
    definition = {**key, "input_columns": list(train.columns),
                  "weights": result.weights.tolist(), "downstream_coefficients": coefficients.tolist(),
                  "downstream_intercept": float(model.intercept_)}
    worst = []
    for position in np.argsort(np.abs(residuals))[-5:][::-1]:
        values = X_test[position]
        largest = int(np.argmax(np.abs(values * coefficients)))
        worst.append({**key, "test_row_index": int(test.index[position]),
                      "actual": float(np.asarray(y_test)[position]),
                      "predicted": float(predicted[position]),
                      "residual": float(residuals[position]),
                      "abs_residual": float(abs(residuals[position])),
                      "largest_contribution_feature": names[largest],
                      "feature_value": float(values[largest]),
                      "coefficient": float(coefficients[largest]),
                      "contribution": float(values[largest] * coefficients[largest])})
    magnitudes = []
    for column, name in enumerate(FEATURE_NAMES, start=train.shape[1]):
        train_abs = np.abs(X_train[:, column])
        test_abs = np.abs(X_test[:, column])
        train_p99 = float(np.percentile(train_abs, 99))
        magnitudes.append({**key, "feature": name, "train_abs_p99": train_p99,
                           "train_abs_max": float(train_abs.max()),
                           "test_abs_p99": float(np.percentile(test_abs, 99)),
                           "test_abs_max": float(test_abs.max()),
                           "test_max_to_train_p99": float(test_abs.max() / max(train_p99, 1e-12)),
                           "coefficient": float(coefficients[column])})
    signature = {**key, "values": construct(val, result.weights)[:, -2:].T}
    return row, trace, definition, worst, magnitudes, signature


def save_results(output, rows, traces, definitions, residuals, magnitudes, signatures, splits):
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "raw_results.csv", index=False)
    pd.DataFrame(traces).to_csv(output / "convergence.csv", index=False)
    plot_convergence(output, traces)
    pd.DataFrame(residuals).to_csv(output / "extreme_residuals.csv", index=False)
    pd.DataFrame(magnitudes).to_csv(output / "feature_diagnostics.csv", index=False)
    with (output / "feature_definitions.jsonl").open("w", encoding="utf-8") as handle:
        for item in definitions:
            handle.write(json.dumps(item) + "\n")
    (output / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    summary = raw.groupby(["dataset", "method"])[
        ["MAE", "RMSE", "R2", "runtime_seconds", "max_abs_test_residual", "fitness_evaluations"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "performance.csv", index=False)
    by_split = raw.groupby(["dataset", "split", "method"])[["MAE", "RMSE", "R2"]].mean().reset_index()
    paired = []
    for dataset, frame in by_split.groupby("dataset"):
        methods = {name: values.set_index("split") for name, values in frame.groupby("method")}
        for first, second in combinations(methods, 2):
            aligned = methods[first].join(methods[second], lsuffix="_first", rsuffix="_second")
            paired.append({"dataset": dataset, "first": first, "second": second,
                           "n_splits": len(aligned),
                           **{f"delta_{metric}_{stat}": float(getattr(
                               aligned[f"{metric}_first"] - aligned[f"{metric}_second"], stat)())
                              for metric in ("MAE", "RMSE", "R2") for stat in ("mean", "std")}})
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    stability = []
    grouped = {}
    for signature in signatures:
        grouped.setdefault((signature["dataset"], signature["split"], signature["method"]), []).append(signature)
    for (dataset, split, method), group in grouped.items():
        for first, second in combinations(group, 2):
            similarity = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    a, b = first["values"][i], second["values"][j]
                    if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                        similarity[i, j] = abs(np.corrcoef(a, b)[0, 1])
            matching = linear_sum_assignment(-similarity)
            stability.append({"dataset": dataset, "split": split, "method": method,
                              "seed_a": first["optimizer_seed"], "seed_b": second["optimizer_seed"],
                              "average_matched_similarity": float(similarity[matching].mean())})
    pd.DataFrame(stability).to_csv(output / "stability.csv", index=False)


def plot_convergence(output, traces):
    frame = pd.DataFrame(traces)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, (dataset, group) in zip(axes, frame.groupby("dataset")):
        for method, values in group.groupby("method"):
            curve = values.groupby("evaluations").best_validation_MAE.mean()
            axis.plot(curve.index, curve.values, label=method)
        axis.set(title=dataset, xlabel="Fitness evaluations", ylabel="Mean best validation MAE")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One split and seed per dataset, 200 rows")
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("results/milestone2_smoke" if args.smoke else "results/milestone2")
    if args.output.resolve() in (Path("results/milestone1").resolve(), Path("results/equal_budget").resolve()):
        parser.error("Choose a separate Milestone 2 output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"protocol": "milestone2_controlled_continuous", "datasets": DATASETS,
              "methods": METHODS, "outer_seeds": OUTER_SEEDS, "optimizer_seeds": OPTIMIZER_SEEDS,
              "optimizer_settings": {
                  "GA": {"population": 40, "tournament": 2, "per_coefficient_mutation_probability": "1/dimension", "mutation_sd": 0.15, "elite": 1},
                  "PSO": {"particles": 40, "inertia": 0.7, "cognitive": 1.4, "social": 1.4, "initial_velocity": [-0.1, 0.1]},
                  "ES": {"parents": 20, "offspring": 40, "selection": "mu+lambda", "mutation_sd": 0.15},
                  "DE": {"population": 40, "strategy": "rand/1/bin", "differential_weight": 0.8, "crossover_probability": 0.9},
                  "CMA-ES": {"population": 20, "initial_sigma": 0.3, "backend": "pycma ask/tell"}},
              "budget": args.budget, "outer_test_fraction": 0.2,
              "inner_validation_fraction": 0.25, "input_scaling": "training-only StandardScaler",
              "representation": "all original features plus two squared linear combinations",
              "coefficient_bounds": [LOWER_BOUND, UPPER_BOUND],
              "initialization": "uniform [-1,1] (CMA-ES initial mean only)",
              "objective": "inner validation MAE", "downstream_model": "LinearRegression fitted on inner training set",
              "smoke": args.smoke,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__,
                           "scikit_learn": sklearn.__version__, "cma": cma.__version__}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, traces, definitions, residuals, magnitudes, signatures, splits = [], [], [], [], [], [], {}
    for dataset, (path, target) in DATASETS.items():
        X, y = load_dataset(path, target)
        if args.smoke:
            X, y = X.iloc[:200], y.iloc[:200]
        splits[dataset] = []
        for split_number, outer_seed in enumerate(OUTER_SEEDS[:1] if args.smoke else OUTER_SEEDS):
            *parts, indices = split_and_preprocess(X, y, outer_seed, return_indices=True)
            splits[dataset].append({"outer_seed": outer_seed, **indices})
            for optimizer_seed in OPTIMIZER_SEEDS[:1] if args.smoke else OPTIMIZER_SEEDS:
                for method in METHODS:
                    print(f"{dataset} split {split_number + 1} seed {optimizer_seed} {method}", flush=True)
                    row, trace, definition, worst, feature, signature = run_one(
                        dataset, split_number, outer_seed, optimizer_seed, method, parts, args.budget)
                    rows.append(row)
                    traces.extend(trace)
                    definitions.append(definition)
                    residuals.extend(worst)
                    magnitudes.extend(feature)
                    signatures.append(signature)
            save_results(args.output, rows, traces, definitions, residuals, magnitudes, signatures, splits)
    print(f"Saved {len(rows)} controlled optimizer runs to {args.output}")


if __name__ == "__main__":
    main()
