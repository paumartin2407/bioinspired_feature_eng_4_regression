"""Fetch prespecified UCI source data; never fetch a dataset during an experiment."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .experiments import ROOT, read_config


def fetch(stage, manifest, root=ROOT):
    from ucimlrepo import fetch_ucirepo

    for spec in manifest["datasets"]:
        if spec["stage"] != stage or "uci_id" not in spec:
            continue
        path = Path(root) / spec["path"]
        if path.exists():
            print(f"Already present: {path}")
            continue
        source = fetch_ucirepo(id=spec["uci_id"])
        features = source.data.features.reset_index(drop=True)
        targets = source.data.targets.reset_index(drop=True)
        if features.empty or targets.shape[1] < 1:
            raise ValueError(f"UCI {spec['uci_id']} returned no usable data")
        frame = features.copy()
        frame["target"] = pd.to_numeric(targets.iloc[:, spec.get("target_index", 0)], errors="raise")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (path.with_suffix(".sha256")).write_text(digest + "\n", encoding="ascii")
        print(f"Fetched {spec['name']}: {path} sha256={digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("exploration", "final"), default="exploration")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/datasets.yaml")
    args = parser.parse_args()
    fetch(args.stage, read_config(args.manifest))


if __name__ == "__main__":
    main()
