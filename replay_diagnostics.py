"""Reconstruct saved feature definitions and inspect held-out errors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from data import DATASETS, load_dataset, split_and_preprocess
from features import add_gp_feature, candidate_frame
from optimizers import (BinaryOpNode, ConstantNode, FeatureNode, UnaryOpNode,
                        protected_div, protected_sqrt)


def restore_tree(item):
    if "feature" in item:
        return FeatureNode(item["feature"], item["name"])
    if "constant" in item:
        return ConstantNode(item["constant"])
    operator = item["operator"]
    if "child" in item:
        return UnaryOpNode(restore_tree(item["child"]), operator,
                           {"sin": np.sin, "cos": np.cos, "sqrt": protected_sqrt}[operator])
    return BinaryOpNode(restore_tree(item["left"]), restore_tree(item["right"]),
                        operator, {"+": np.add, "-": np.subtract, "*": np.multiply,
                                   "/": protected_div}[operator])


def replay_features(record, train, val, test):
    method = record["method"]
    definition = record["definition"]
    if method.startswith("GP"):
        tree_item = definition.get("gp_feature", definition)
        tree = restore_tree(tree_item["tree"])
        train, val, test = add_gp_feature(tree, train, val, test)
        if method == "GP":
            return train, val, test
    kind = method.split()[-1] if method.startswith("GP +") else method.split()[0]
    mode = "combined" if method.startswith("GP +") else definition["mode"]
    individual = definition["individual"]
    return tuple(candidate_frame(individual, frame, kind, reference=train, mode=mode)
                 for frame in (train, val, test))


def replay(saved):
    saved = Path(saved)
    config = json.loads((saved / "config.json").read_text())
    definitions = [json.loads(line) for line in (saved / "feature_definitions.jsonl").read_text().splitlines()]
    original = pd.read_csv(saved / "raw_results.csv")
    residual_rows, feature_rows = [], []
    datasets = {}
    splits = {}
    for item in definitions:
        method = item["method"]
        if method in {"Linear Regression", "Ridge", "Lasso", "Elastic Net", "Polynomial + Ridge"}:
            continue
        dataset, split = item["dataset"], item["split"]
        if dataset not in datasets:
            X, y = load_dataset(*DATASETS[dataset])
            datasets[dataset] = (X.iloc[:200], y.iloc[:200]) if config.get("smoke") else (X, y)
        if (dataset, split) not in splits:
            X, y = datasets[dataset]
            seed = int(original.loc[(original.dataset == dataset) & (original.split == split),
                                    "outer_seed"].iloc[0])
            splits[dataset, split] = split_and_preprocess(X, y, seed)
        train, val, test, y_train, y_val, y_test = splits[dataset, split]
        train_features, _, test_features = replay_features(item, train, val, test)
        model = LinearRegression().fit(train_features, y_train)
        predictions = model.predict(test_features)
        row = original.loc[(original.dataset == dataset) & (original.split == split) &
                           (original.method == method) &
                           (original.optimizer_seed == item["optimizer_seed"])].iloc[0]
        replayed_mae = mean_absolute_error(y_test, predictions)
        if abs(replayed_mae - row.MAE) > 1e-6:
            raise AssertionError(f"Replayed MAE differs for {dataset}, split {split}, {method}")
        coefficients = np.ravel(model.coef_)
        residuals = np.asarray(y_test) - predictions
        for position in np.argsort(np.abs(residuals))[-5:][::-1]:
            values = test_features.iloc[position].to_numpy()
            largest = int(np.argmax(np.abs(values * coefficients)))
            residual_rows.append({"dataset": dataset, "split": split,
                "optimizer_seed": item["optimizer_seed"], "method": method,
                "test_row_index": int(test.index[position]), "actual": float(np.asarray(y_test)[position]),
                "predicted": float(predictions[position]), "residual": float(residuals[position]),
                "abs_residual": float(abs(residuals[position])),
                "largest_contribution_feature": test_features.columns[largest],
                "feature_value": float(values[largest]), "coefficient": float(coefficients[largest]),
                "contribution": float(values[largest] * coefficients[largest])})
        for name in (c for c in test_features if str(c).startswith(("synth_", "GP_feature"))):
            train_abs, test_abs = np.abs(train_features[name].to_numpy()), np.abs(test_features[name].to_numpy())
            train_p99 = float(np.percentile(train_abs, 99))
            feature_rows.append({"dataset": dataset, "split": split,
                "optimizer_seed": item["optimizer_seed"], "method": method, "feature": name,
                "train_abs_p99": train_p99, "train_abs_max": float(train_abs.max()),
                "test_abs_max": float(test_abs.max()), "test_abs_p99": float(np.percentile(test_abs, 99)),
                "test_to_train_p99_ratio": float(test_abs.max() / max(train_p99, 1e-12)),
                "coefficient": float(coefficients[list(test_features.columns).index(name)])})
    pd.DataFrame(residual_rows).to_csv(saved / "extreme_residuals.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(saved / "feature_diagnostics.csv", index=False)
    return len(residual_rows), len(feature_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(replay(args.directory))
