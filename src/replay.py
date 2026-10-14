"""Rebuild one saved FE pipeline from source rows and verify final predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import fit_preprocessor, load_dataset, transformed
from .evaluation import learner, metrics
from .experiments import ROOT, _apply_pair, array_digest


def replay(output, dataset_name, split_number, model_name, arm_name, search_seed, data_root=ROOT):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    dataset = next(item for item in protocol["datasets"] if item["name"] == dataset_name)
    source = Path(data_root) / dataset["path"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != protocol["source_sha256"][dataset_name]:
        raise ValueError("Source data changed after the experiment")
    X, y = load_dataset(dataset, data_root)
    if protocol["smoke"]:
        X, y = X.iloc[:protocol["smoke_row_cap"]], y[:protocol["smoke_row_cap"]]
    partition = next(item for item in json.loads((output / "splits.json").read_text(encoding="utf-8"))
                     if item["dataset"] == dataset_name and item["split"] == split_number)
    definition = next(item for line in (output / "definitions.jsonl").read_text(encoding="utf-8").splitlines()
                      if (item := json.loads(line))["dataset"] == dataset_name and item["split"] == split_number
                      and item["learner"] == model_name and item["arm"] == arm_name
                      and item["search_seed"] == search_seed)
    arm = next(item for item in protocol["config"]["arms"] if item["name"] == arm_name)
    dev_index, test_index = partition["development"], partition["test"]
    dev, held = X.iloc[dev_index], X.iloc[test_index]
    preprocessor = fit_preprocessor(dev)
    columns = definition["columns"]
    train = transformed(preprocessor, dev, columns).to_numpy()
    test = transformed(preprocessor, held, columns).to_numpy()
    if definition["fallback"]:
        train_features, test_features = train, test
    else:
        train_features, test_features = _apply_pair(train, test, arm, definition["candidate"], definition["k"])
    if array_digest(train_features) != definition["dev_features_sha256"]:
        raise AssertionError("Final development transformation did not replay")
    if array_digest(test_features) != definition["test_features_sha256"]:
        raise AssertionError("Final test transformation did not replay")
    prediction = learner(model_name, definition["learner_parameter"], search_seed).fit(
        train_features, y[dev_index]).predict(test_features)
    if array_digest(prediction) != definition["prediction_sha256"]:
        raise AssertionError("Final predictions did not replay")
    saved = pd.read_csv(output / "runs.csv")
    row = saved.loc[(saved.dataset == dataset_name) & (saved.split == split_number) &
                    (saved.learner == model_name) & (saved.arm == arm_name) &
                    (saved.search_seed == search_seed)].iloc[0]
    observed = metrics(y[test_index], prediction)
    if not np.isclose(observed["mae"], row.mae, rtol=0, atol=1e-12):
        raise AssertionError("Saved MAE did not replay")
    return {"dataset": dataset_name, "split": split_number, "learner": model_name,
            "arm": arm_name, "search_seed": search_seed, "mae": observed["mae"],
            "prediction_sha256": definition["prediction_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, type=int)
    parser.add_argument("--learner", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(replay(args.output, args.dataset, args.split, args.learner, args.arm, args.seed), indent=2))


if __name__ == "__main__":
    main()
