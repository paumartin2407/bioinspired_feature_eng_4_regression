"""Exploratory downstream-model probe on saved, fixed feature definitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data import DATASETS, load_dataset, split_and_preprocess
from evaluation import regression_metrics
from replay_diagnostics import replay_features


ALPHAS = (0.01, 0.1, 1., 10., 100., 1000., 10000., 100000.)


def fit_probe(train, val, test, y_train, y_val, clipped=False):
    train, val, test = (frame.to_numpy().copy() for frame in (train, val, test))
    if clipped:
        bound = np.quantile(np.abs(train), 0.995, axis=0)
        bound = np.maximum(bound, 1e-12)
        train, val, test = (np.clip(frame, -bound, bound) for frame in (train, val, test))
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val, test = scaler.transform(val), scaler.transform(test)
    best_alpha = min(ALPHAS, key=lambda alpha: mean_absolute_error(
        y_val, Ridge(alpha=alpha).fit(train, y_train).predict(val)))
    model = Ridge(alpha=best_alpha).fit(train, y_train)
    return model.predict(test), best_alpha


def probe(directory, method="PSO construction", dataset="California Housing"):
    directory = Path(directory)
    X, y = load_dataset(*DATASETS[dataset])
    config = json.loads((directory / "config.json").read_text())
    if config.get("smoke"):
        X, y = X.iloc[:200], y.iloc[:200]
    raw = pd.read_csv(directory / "raw_results.csv")
    definitions = [json.loads(line) for line in (directory / "feature_definitions.jsonl").read_text().splitlines()]
    splits, rows = {}, []
    for record in definitions:
        if record["dataset"] != dataset or record["method"] != method:
            continue
        split = record["split"]
        if split not in splits:
            seed = int(raw.loc[(raw.dataset == dataset) & (raw.split == split), "outer_seed"].iloc[0])
            splits[split] = split_and_preprocess(X, y, seed)
        train, val, test, y_train, y_val, y_test = splits[split]
        features = replay_features(record, train, val, test)
        original = raw.loc[(raw.dataset == dataset) & (raw.split == split) &
                           (raw.method == method) &
                           (raw.optimizer_seed == record["optimizer_seed"])].iloc[0]
        for clipped in (False, True):
            prediction, alpha = fit_probe(*features, y_train, y_val, clipped=clipped)
            rows.append({"dataset": dataset, "method": method, "split": split,
                         "optimizer_seed": record["optimizer_seed"],
                         "variant": "Ridge + scaling + clipping" if clipped else "Ridge + scaling",
                         "alpha": alpha, **regression_metrics(y_test, prediction),
                         "max_abs_residual": float(np.abs(np.asarray(y_test) - prediction).max()),
                         "original_MAE": original.MAE, "original_RMSE": original.RMSE,
                         "original_R2": original.R2})
    output = directory / "ridge_probe.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output, pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--method", default="PSO construction")
    parser.add_argument("--dataset", default="California Housing")
    args = parser.parse_args()
    output, results = probe(args.directory, args.method, args.dataset)
    print(output)
    print(results.groupby("variant")[["MAE", "RMSE", "R2", "max_abs_residual"]].mean().to_string())
