"""Check pilot health without ranking arms by predictive performance."""

import argparse
import json
from pathlib import Path

import pandas as pd


def check(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if not protocol["config"].get("pilot") or protocol["smoke"]:
        raise ValueError("This checker only accepts a complete pilot run")
    runs = pd.read_csv(output / "runs.csv")
    searched = runs.loc[runs.arm != "no_fe"]
    expected = len(protocol["datasets"]) * len(protocol["config"]["outer_seeds"]) * len(
        protocol["config"]["search_seeds"]) * len(protocol["config"]["learners"]) * len(protocol["config"]["arms"])
    if len(searched) != expected:
        raise AssertionError(f"Incomplete pilot: {len(searched)} of {expected} FE rows")
    definitions = [json.loads(line) for line in (output / "definitions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(definitions) != expected:
        raise AssertionError("Missing saved feature definitions")
    for item in definitions:
        row = searched.loc[(searched.dataset == item["dataset"]) & (searched.split == item["split"]) &
                            (searched.learner == item["learner"]) & (searched.arm == item["arm"]) &
                            (searched.search_seed == item["search_seed"])].iloc[0]
        if sum(item["evaluation_allocation"].values()) != row.candidate_evaluations:
            raise AssertionError("Candidate budget accounting mismatch")
    groups = []
    for (dataset, arm), part in searched.groupby(["dataset", "arm"], sort=True):
        groups.append({"dataset": dataset, "arm": arm, "runs": len(part),
                       "failed_candidates": int(part.failed_candidates.sum()),
                       "outer_failures": int(part.outer_failure.sum()),
                       "selected_k_counts": {str(int(k)): int(count) for k, count in
                                             part.selected_k.value_counts().sort_index().items()},
                       "runtime_median_seconds": float(part.runtime_seconds.median()),
                       "runtime_max_seconds": float(part.runtime_seconds.max())})
    result = {"pilot_complete": True, "fe_runs": expected, "diagnostics": groups}
    (output / "pilot_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.output), indent=2))


if __name__ == "__main__":
    main()
