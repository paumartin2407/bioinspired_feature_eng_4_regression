# Bio-inspired feature engineering for regression

The [project report](docs/followup_project_report.md) describes the six-phase exploratory study. Configurations and retained results are organized under `configs/phase_1` through `configs/phase_6` and `results/phase_1` through `results/phase_6`. Phase 1 includes adaptive and fixed-capacity studies; Phase 5 includes equal-candidate and 30-second equal-time studies. Generated outputs are ignored by Git.

## Setup and data

```powershell
python -m pip install -r requirements.txt
```

The manifest at `configs/datasets.yaml` contains only California Housing and Diabetes Progression, using the existing source files. The runner never downloads data. Both datasets have been examined before, so results from this two-dataset study are exploratory and cannot serve as independent confirmation. Cite the original datasets when publishing results.

## Checks and experiments

```powershell
python -m unittest discover -s tests -q
python -m src.experiments --config configs/phase_1/adaptive.yaml --output smoke-results/phase_1 --smoke
```

The smoke run uses a reduced dataset and search budget. To reproduce a reported run, use the corresponding configuration in the phase folder and select a new output directory; completed result directories are preserved. For example:

```powershell
python -m src.experiments --config configs/phase_2/individual.yaml --output results/new_phase_2
python -m src.reporting --runs results/new_phase_2/runs.csv --output results/new_phase_2_summary
```

## Protocol contract

Each dataset uses five shared 80/20 outer splits and three shuffled inner folds. Rows with missing targets are dropped; other nonnumeric or nonfinite targets are rejected. Numeric predictors receive median imputation and standard scaling; categorical predictors receive most-frequent imputation and one-hot encoding with unknown categories ignored. These operations are fitted in each inner training fold. The same candidate vector is scored across all inner folds, then replayed after preprocessing is refitted on all development rows. Coefficients therefore operate in fold-standardized coordinates during selection and in development-standardized coordinates at final fit; their raw-input interpretation changes with that refit. Outer-test rows are only predicted after model, representation and `k` selection. Learner settings are fixed in the retained experiments; Phase 6 additionally scales the constructed feature matrix and target within each training fold.

Phase 6's model settings are implemented in code and recorded in its configuration. The runner checks that the configured values match those fixed implementations; changing a value in the configuration alone is rejected rather than silently ignored.

An adaptive construction-only arm divides its **total** candidate budget among eligible positive `k` values, with `k=0` using the same learner's no-FE inner score. Selection-only has no synthetic features. For the combined arm, `k=0` still searches a selection mask and therefore receives part of the budget. The combined arm constructs from all original inputs even when some are dropped as direct learner inputs. Its mask describes direct retention only; `selected_originals`, `construction_input_count` and `effective_original_count` distinguish direct and indirect use. `runs.csv` records candidate evaluations, FE fits, baseline scoring and fit counts, failures, runtime, chosen `k`, transformed dimension, MAE difference and percent reduction. The historical `tuning_model_fits` field counts no-FE inner-fold scoring even though the retained learners are not tuned. `definitions.jsonl` records per-`k` evaluation allocations and scores, input names, replayable genes, and hashes of final features and predictions. `splits.json` records row indices. `stability.csv` compares seed pairs within each outer split.

| Run field | Meaning |
| --- | --- |
| `selected_k` | Number of constructed features. |
| `selected_originals` | Number of original features directly retained. |
| `construction_input_count` | Number of original inputs used in constructed features. |
| `effective_original_count` | Number of distinct original inputs used directly or indirectly. |
| `transformed_dim` | Number of columns given to the regression model. |

For quadratic features, an original input counts as used in construction when any corresponding coefficient has absolute value greater than `1e-12`. With continuous weights, nearly every nonzero coefficient may meet this threshold. Thus `effective_original_count` measures algebraic input use, not feature importance or effective sparsity.

To verify a saved FE run exactly, use `python -m src.replay --output results/phase_2/individual --dataset "Diabetes Progression" --split 0 --learner ols --arm quadratic --seed 11`. Replay verifies the source checksum, refits preprocessing and the learner on development rows, and compares development features, test features, predictions and MAE with saved hashes and metrics. Exact hashes can differ across numerical environments even when results agree to floating-point precision. For example, checked Phase 1 runs matched saved MAE within `1e-14` but did not reproduce all hashes; their original thread settings were not recorded. Replay deliberately reports such differences rather than weakening its exact verification.

The reporting helper averages seeds inside outer splits, then splits inside datasets, and assigns equal weight to datasets. Its bootstrap interval resamples datasets and is descriptive with a small panel. Neither the runner nor reporting tool reads previous milestone outputs or makes a final recommendation.
