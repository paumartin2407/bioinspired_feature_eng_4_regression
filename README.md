# Milestone 1 benchmark

Install `requirements.txt`, then run:

```powershell
python benchmark.py --smoke --output results/smoke
python benchmark.py --output results/milestone1
```

The second command runs both full datasets with five outer splits and five optimizer seeds per split. Each split uses 80% development data and 20% held-out test data; the development portion has a fixed 75/25 training/validation split. All methods within an outer split share those rows. The full run can take a long time on California Housing.

Outputs include `config.json`, `splits.json`, `feature_definitions.jsonl`, `raw_results.csv`, `performance.csv`, `paired_comparisons.csv`, `ablation.csv`, `stability.csv`, `convergence.csv`, and `convergence.png`. Positive MAE and RMSE differences in the paired file mean the first method is worse. The old root-level CSV files are from the earlier notebook workflow.

Run `python -m unittest -q test_milestone1.py` for the preprocessing, feature scaling, GP mutation, budget and stability checks. The notebook is an entry point to the same benchmark code.
