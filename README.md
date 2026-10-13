# Regression feature engineering benchmark

Install `requirements.txt`, then run:

```powershell
python benchmark.py --smoke --output results/equal_smoke
python benchmark.py --output results/equal_budget
python compare_budgets.py results/milestone1 results/equal_budget
python probe_regularization.py results/equal_budget
```

The second command runs both full datasets with five outer splits and five optimizer seeds per split. Each split uses 80% development data and 20% held-out test data; the development portion has a fixed 75/25 training/validation split. All methods within an outer split share those rows. Every optimizer run consumes exactly 1,000 fitness evaluations, including its initial population and any partial final generation. The full run can take a long time on California Housing.

Outputs include `config.json`, `splits.json`, `feature_definitions.jsonl`, `raw_results.csv`, `performance.csv`, `paired_comparisons.csv`, `ablation.csv`, `stability.csv`, `convergence.csv`, `convergence.png`, `extreme_residuals.csv`, and `feature_diagnostics.csv`. Positive MAE and RMSE differences in the paired file mean the first method is worse. Stability matches constructed features one to one and averages absolute correlations; unmatched features count as zero similarity. The old unequal-budget results remain in `results/milestone1`. Run `python replay_diagnostics.py results/milestone1` to reproduce and inspect their held-out outliers without rerunning optimization.

`compare_budgets.py` writes `results/equal_budget/budget_comparison.csv` with matched old and corrected runs; a negative `delta_MAE` means the equal-budget run improved. Baselines retain zero optimizer evaluations and are included as unchanged references.

`probe_regularization.py` reuses fixed saved features to compare training-only scaling plus Ridge, with and without training-derived 99.5th-percentile clipping. It tunes Ridge alpha on the inner validation rows and writes `ridge_probe.csv`. It is exploratory: the optimizers still selected these features under the original linear regression objective.

Run `python -m unittest -q test_milestone1.py` for preprocessing, feature scaling, GP mutation, exact budget, decoder, test isolation, and stability checks. The notebook is an entry point to the same benchmark code. Dependency versions are pinned in `requirements.txt`; CI runs the tests and validates the notebook.

## Milestone 2: controlled continuous comparison

```powershell
python benchmark_continuous.py --smoke --budget 80 --output results/milestone2_smoke
python benchmark_continuous.py --output results/milestone2
python -m unittest -q test_milestone1.py test_milestone2.py
```

The full command runs GA, PSO, ES, DE/rand/1/bin, and CMA-ES on California Housing and Diabetes: five outer splits, five optimizer seeds, and 1,000 validation-MAE evaluations per run (250 runs total). These are **controlled continuous variants**, separate from the original mixed-variable methods and GP. Every candidate is a 2-by-d matrix of coefficients in [-1, 1]. The decoder retains all training-standardized original inputs and adds `(X @ W.T) ** 2` as two features. All five methods use the same decoder, inner training/validation split, `LinearRegression` fitness, and held-out test protocol. GA, PSO, ES, and DE start populations uniformly in the bounds; CMA-ES starts with a uniformly sampled mean and draws its own Gaussian population. Each method gets a reproducible stream from its optimizer seed. Equal evaluations control search effort, while runtime is reported separately.

`results/milestone2/` contains `raw_results.csv` (test MAE, RMSE, R², maximum absolute residual, runtime, count), `performance.csv`, split-paired comparisons, per-evaluation convergence and its plot, coefficients, worst five test residuals per run, synthetic-feature magnitudes, one-to-one matched stability, split indices, and a versioned configuration. The inner training data alone determines preprocessing and the downstream fit. This experiment does not add clipping or regularization; those changes belong to a separately defined robustness experiment. The Milestone 1 results remain under their existing directories.
