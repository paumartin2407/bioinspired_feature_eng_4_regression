# Bio-inspired feature engineering for regression

The prospective protocol is [docs/research_plan.md](docs/research_plan.md). This branch contains experiment code and configurations, with generated outputs ignored by Git. No full study has been run or interpreted; local pilot diagnostics are not research evidence. Old milestone results are excluded from this study.

## Setup and data

```powershell
python -m pip install -r requirements.txt
```

The manifest at `configs/datasets.yaml` contains only California Housing and Diabetes Progression, using the existing source files. The runner never downloads data. Both datasets have been examined before, so results from this two-dataset study are exploratory and cannot serve as independent confirmation. Cite the original datasets when publishing results.

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
python -m src.experiments --config configs/rq2_fixed.yaml --output results/rq2_fixed
python -m src.experiments --config configs/rq3.yaml --output results/rq3
python -m src.experiments --config configs/rq4.yaml --output results/rq4
python -m src.reporting --runs results/rq1/runs.csv --output results/rq1_summary
```

RQ4's two initial pipelines should be replaced with a development-evidence shortlist before that study. `configs/final.yaml` remains disabled and the manifest has no final-stage datasets; no independent confirmation run is available under this two-dataset scope.

`configs/rq2.yaml` is the primary adaptive-`k` pipeline comparison. `configs/rq2_fixed.yaml` is the secondary capacity check at matched `k=1,2,4`. GP changes both feature language and search mechanism, so its comparison is at pipeline level. The fixed-`k` arms spend 1,000 evaluations at one dimension; adaptive arms divide 1,000 evaluations among eligible searched `k` values. Their difference cannot be interpreted as a feature-count effect alone.

## Protocol contract

Each dataset uses five shared 80/20 outer splits and three shuffled inner folds. Rows with missing targets are dropped; other nonnumeric or nonfinite targets are rejected. Numeric predictors receive median imputation and standard scaling; categorical predictors receive most-frequent imputation and one-hot encoding with unknown categories ignored. These operations are fitted in each inner training fold. The same candidate formula or vector is scored across all inner folds, then replayed after preprocessing is refitted on all development rows. Coefficients therefore operate in fold-standardized coordinates during selection and in development-standardized coordinates at final fit; their raw-input interpretation changes with that refit. Outer-test rows are only predicted after model, representation and `k` selection. Ridge and forest parameters are pre-tuned on original development features once per outer split, then frozen for all FE arms; these tuning fits are recorded. RQ4 addresses FE with a pre-tuned fixed learner, not joint learner-and-FE optimization. OLS uses fixed parameters.

An adaptive construction-only arm divides its **total** candidate budget among eligible positive `k` values, with `k=0` using the same learner's pre-tuned no-FE inner score. Selection-only has no synthetic features. For the combined arm, `k=0` still searches a selection mask and therefore receives part of the budget. The combined arm constructs from all original inputs even when some are dropped as direct learner inputs. Its mask describes direct retention only; `selected_originals`, `construction_input_count` and `effective_original_count` distinguish direct and indirect use. `runs.csv` records candidate evaluations, FE fits, shared learner-tuning fits, baseline fit, failures, runtime, chosen `k`, transformed dimension, MAE difference and percent reduction. `definitions.jsonl` records per-`k` evaluation allocations and scores, direct and constructed input names, replayable genes or GP trees, and hashes of final features and predictions. `splits.json` records row indices. `stability.csv` compares seed pairs within each outer split. A polynomial plus Ridge reference is included in primary RQ2 and is not capacity matched to the optimized families.

| Run field | Meaning |
| --- | --- |
| `selected_k` | Number of constructed features. |
| `selected_originals` | Number of original features directly retained. |
| `construction_input_count` | Number of original inputs used in constructed features. |
| `effective_original_count` | Number of distinct original inputs used directly or indirectly. |
| `transformed_dim` | Number of columns given to the regression model. |

For quadratic features, an original input counts as used in construction when any corresponding coefficient has absolute value greater than `1e-12`. With continuous weights, nearly every nonzero coefficient may meet this threshold. Thus `effective_original_count` measures algebraic input use, not feature importance or effective sparsity.

To verify a saved FE run exactly, use `python -m src.replay --output results/rq1 --dataset "California Housing" --split 0 --learner ols --arm construction --seed 11`. Replay verifies the source checksum, refits preprocessing and the learner on development rows, and compares development features, test features, predictions and MAE with saved hashes and metrics.

The earlier reduced RQ1 pilot used five datasets; [pilot readiness](docs/pilot_readiness.md) preserves its health check as historical diagnostics. With the current manifest, `configs/rq1_pilot.yaml` selects only California Housing and Diabetes Progression. The checker reports only completeness, numerical failures, selected-`k` counts and runtime. Pilot predictive scores must not select or alter the full-study configurations.

The reporting helper averages seeds inside outer splits, then splits inside datasets, and assigns equal weight to datasets. Its bootstrap interval resamples datasets and is descriptive with a small panel. Neither the runner nor reporting tool reads previous milestone outputs or makes a final recommendation.
