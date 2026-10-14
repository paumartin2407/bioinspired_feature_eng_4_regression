# Bio-inspired feature engineering for regression

The prospective protocol is [docs/research_plan.md](docs/research_plan.md). This branch contains experiment code and configurations only; no new study has been run or interpreted. Old milestone results are excluded from this study.

## Setup and data

```powershell
python -m pip install -r requirements.txt
python -m src.datasets --stage exploration
```

The manifest at `configs/datasets.yaml` fixes seven public regression datasets: five for RQ1–RQ4 and two reserved for independent final confirmation. Existing California and Diabetes files are exploration sources; the others are fetched from UCI by ID and stored with SHA-256 checksums. The final panel is Yacht Hydrodynamics and Abalone Rings; neither has been fetched for this branch. Fetching `--stage final` should wait until the final protocol is locked. The runner never downloads data. Cite original datasets when publishing results.

## Checks and later experiments

```powershell
python -m unittest discover -s tests -q
python -m src.experiments --config configs/rq1.yaml --output results/rq1_smoke --smoke
```

The smoke path uses one dataset, one outer split, one search seed, up to 120 source rows, and a reduced `k` grid with 40 optimizer evaluations per searched arm. It verifies wiring only; its metrics are not research evidence. To run a full study later, omit `--smoke` and give a fresh output directory:

```powershell
python -m src.experiments --config configs/rq1.yaml --output results/rq1
python -m src.experiments --config configs/rq1_fixed.yaml --output results/rq1_fixed
python -m src.experiments --config configs/rq2.yaml --output results/rq2
python -m src.experiments --config configs/rq3.yaml --output results/rq3
python -m src.experiments --config configs/rq4.yaml --output results/rq4
python -m src.reporting --runs results/rq1/runs.csv --output results/rq1_summary
```

RQ4's two initial pipelines should be replaced with a development-evidence shortlist before that study. `configs/final.yaml` has no shortlisted arms and `locked: false`; the runner refuses to run it until the procedure is fixed. Final runs only select `stage: final` datasets.

## Protocol contract

Each dataset uses five shared 80/20 outer splits and three shuffled inner folds. Rows with missing targets are dropped; other nonnumeric or nonfinite targets are rejected. Numeric predictors receive median imputation and standard scaling; categorical predictors receive most-frequent imputation and one-hot encoding with unknown categories ignored. These operations are fitted in each inner training fold. The same candidate formula or vector is scored across all inner folds, then replayed after a full development refit. Outer-test rows are only predicted after model, representation and `k` selection. Ridge and forest parameters are pre-tuned on original development features once per outer split, then frozen for all FE arms; these tuning fits are recorded. OLS uses fixed parameters.

An adaptive construction arm divides its **total** candidate budget among eligible positive `k` values, with `k=0` using the already computed no-FE inner score. Selection-only has no synthetic features. The combined arm constructs from all original inputs even when some are dropped as direct learner inputs. `runs.csv` records candidate evaluations, FE fits, shared learner-tuning fits, baseline fit, failures, runtime, chosen `k`, transformed dimension, MAE difference and percent reduction. `definitions.jsonl` contains replayable genes or GP trees and scores for every tested `k`; `splits.json` records row indices. `stability.csv` compares seed pairs within each outer split. A polynomial plus Ridge reference is included in RQ2 and is not capacity matched to the optimized families.

The reporting helper averages seeds inside outer splits, then splits inside datasets, and assigns equal weight to datasets. Its bootstrap interval resamples datasets and is descriptive with a small panel. Neither the runner nor reporting tool reads previous milestone outputs or makes a final recommendation.
