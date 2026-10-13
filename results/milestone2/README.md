# Milestone 2: controlled continuous optimizer comparison

The benchmark completed 250 optimizer runs: two datasets × five outer splits × five optimizer seeds × five methods. Every run used exactly 1,000 validation-MAE evaluations, including initialization. `splits.json` exactly matches the prior equal-budget study. GA, PSO, and ES here are continuous variants, separate from the Milestone 1 mixed-variable methods; GP is outside this representation-controlled comparison.

Every method searched a 2 × d coefficient matrix in [-1, 1]. Training-only standardized inputs were retained, and two features `(X @ W.T) ** 2` were added. The inner-training `LinearRegression` fit and inner-validation MAE were identical across methods. CMA-ES used the `pycma` ask/tell implementation. The optimizer settings, random seeds, dataset splits, and package versions are recorded in `config.json`.

## Held-out test results

Means over 25 runs per dataset and method (five splits × five seeds):

| Dataset | Method | MAE | RMSE | R² | Mean runtime (s) | Mean maximum absolute residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| California | CMA-ES | 0.518 | 1.100 | -0.287 | 2.635 | 44.397 |
| California | DE | 0.527 | 1.133 | -0.394 | 2.687 | 45.404 |
| California | ES | 0.524 | 1.172 | -0.723 | 2.676 | 48.421 |
| California | GA | 0.526 | 1.183 | -0.713 | 2.753 | 49.441 |
| California | PSO | 0.525 | 1.310 | -1.363 | 2.707 | 57.493 |
| Diabetes | CMA-ES | 43.677 | 54.249 | 0.494 | 0.717 | 142.840 |
| Diabetes | DE | 43.464 | 53.908 | 0.501 | 0.704 | 140.304 |
| Diabetes | ES | 43.470 | 53.966 | 0.499 | 0.642 | 141.513 |
| Diabetes | GA | 43.176 | 53.691 | 0.505 | 0.661 | 141.232 |
| Diabetes | PSO | 43.640 | 54.190 | 0.495 | 0.643 | 141.979 |

CMA-ES had the lowest mean California test MAE, 0.006–0.008 below the other methods, and had the lowest split-averaged MAE in four of five outer splits. GA had the lowest mean Diabetes test MAE, 0.288–0.500 below the other methods, and led in four of five splits. CMA-ES reached the lowest mean Diabetes *validation* MAE (39.302), so the validation and held-out rankings differ. The `paired_comparisons.csv` file first averages the five optimizer seeds within each outer split, then computes method differences across the five matched splits. These descriptive differences do not establish a general ranking beyond these settings and datasets.

## Robustness finding

California remains sensitive to rare input extremes. In outer split 2 (seed 23), test row 19006 has `AveOccup = 1243.33`, while the inner training set's 99.9th percentile is about 14.83. For one PSO run, the observed target is 1.375 and the prediction is 319.770, giving an absolute residual of 318.395 and test R² of -20.168. One constructed feature reaches 111,522.65 on this row versus a training absolute 99th percentile of 8.19; its fitted coefficient is 0.103699. Large, partly offsetting feature contributions make the prediction unstable. Extreme errors on this same row appear with all five methods. MAE alone hides this behavior, so RMSE, R², maximum residuals, and `feature_diagnostics.csv` should be read with the primary MAE results.

The downstream fit was deliberately kept as ordinary `LinearRegression`. Clipping or regularization would change the experimental protocol and should be tested as a separate prespecified robustness study. Equal fitness budgets control objective-call counts; observed runtimes still differ. The fixed operator settings and five outer splits limit how broadly these findings generalize.

## Artifacts

- `raw_results.csv`: one row per optimizer run.
- `performance.csv` and `paired_comparisons.csv`: descriptive aggregate and split-paired results.
- `convergence.csv` and `convergence.png`: best validation MAE by evaluation count.
- `feature_definitions.jsonl`: search coefficients and fitted linear coefficients.
- `extreme_residuals.csv` and `feature_diagnostics.csv`: worst five test residuals per run and constructed-feature magnitudes.
- `stability.csv`: mean one-to-one matched absolute correlation of constructed validation features across seeds.
- `config.json` and `splits.json`: protocol, versions, optimizer settings, and exact row partitions.

Reproduce with `python benchmark_continuous.py --output results/milestone2`. Run `python -m unittest -q test_milestone1.py test_milestone2.py` to validate the implementation and synthetic optimizer checks.
