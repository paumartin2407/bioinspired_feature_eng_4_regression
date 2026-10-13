# Milestone 1 results

The full benchmark contains five outer 80/20 splits and five optimizer seeds per split on each dataset. Metaheuristics use validation MAE for search and at most 1,000 candidate evaluations. The numbers below are means of held-out test scores; the CSV files contain standard deviations and all run-level data.

| Dataset | Lowest mean MAE | MAE | RMSE | R² | Linear baseline MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| California Housing | GP + GA | 0.493 | 0.706 | 0.620 | 0.534 |
| Diabetes Progression | GA construction | 43.287 | 53.824 | 0.502 | 43.557 |

On California Housing, GP alone has nearly the same MAE as GP + GA (0.494 versus 0.493) and better mean RMSE and R² (0.681 and 0.651 versus 0.706 and 0.620). Paired outer-split MAE differences between GP and its hybrids are at most 0.0012 in magnitude, smaller than their across-split variation. The hybrid stage does not show a clear material gain here.

On Diabetes Progression, construction-only GA, ES, and PSO have lower MAE than their selection-only and combined variants. The construction-only mean MAEs are 43.287, 43.348, and 43.423 respectively; selection-only MAEs are 45.077, 45.089, and 44.888. GP and all GP hybrids trail the linear and Lasso baselines on this dataset.

On California Housing, construction-only variants have slightly lower MAE than selection-only variants, but substantially worse RMSE and R². This indicates large errors on some test cases. Selection-only variants have mean RMSE about 0.776, while construction-only variants range from 1.186 to 1.343. The degree-2 polynomial Ridge baseline is also unstable here, with mean RMSE 6.191. These results should be compared across all three metrics, not MAE alone.

Stability is reported in `stability.csv` as Jaccard similarity of selected original features and absolute correlation of the most similar pair of constructed features across seeds within each outer split. `paired_comparisons.csv` averages optimizer seeds within each split before computing paired differences, so its sample size is five outer splits. It is descriptive rather than a claim of statistical significance.
