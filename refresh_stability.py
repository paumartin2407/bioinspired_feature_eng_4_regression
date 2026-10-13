"""Recompute feature-selection similarity from saved fitted definitions."""
import argparse
import json
from pathlib import Path

import pandas as pd


def selected_features(record):
    method = record["method"]
    columns = record["input_columns"]
    if method == "GP" or method.endswith(" construction"):
        return set(columns)
    mask = record["definition"]["individual"]["mask"]
    return {column for column, keep in zip(columns, mask)
            if keep and column != "GP_feature"}


def refresh(directory):
    directory = Path(directory)
    definitions = {}
    for line in (directory / "feature_definitions.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "optimizer_seed" in record:
            key = (record["dataset"], record["split"], record["method"], record["optimizer_seed"])
            definitions[key] = selected_features(record)
    path = directory / "stability.csv"
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        first = definitions[row.dataset, row.split, row.method, int(row.seed_a)]
        second = definitions[row.dataset, row.split, row.method, int(row.seed_b)]
        frame.at[index, "selection_jaccard"] = len(first & second) / len(first | second) if first | second else float("nan")
    frame.to_csv(path, index=False)
    return len(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Updated {refresh(args.directory)} stability pairs")
