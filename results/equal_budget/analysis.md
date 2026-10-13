# Equal-budget results and diagnostics

This run uses five outer splits and five optimizer seeds per split for each dataset. Every GA, PSO, ES, GP, and hybrid run consumed exactly 1,000 candidate fitness evaluations, including the initial population and a partial final generation when needed. The original unequal-budget results remain in `../milestone1/`; `budget_comparison.csv` joins all 700 old and corrected runs by dataset, split, method, and seed. Baseline and GP scores are unchanged because their evaluation counts did not change.

| Dataset | Lowest mean test MAE | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| California Housing | GP + GA | 0.493 | 0.706 | 0.620 |
| Diabetes Progression | GA construction | 43.269 | 53.853 | 0.502 |

The extra evaluations produce mixed effects. On California Housing, GA construction improves mean MAE by 0.0064 relative to the original run, while PSO construction improves MAE by 0.0050 but its mean R² falls by 0.951. On Diabetes Progression, PSO combined becomes worse by 0.207 mean MAE. Equalizing search effort therefore does not make one method uniformly better. See `performance.csv` for means and standard deviations and `paired_comparisons.csv` for split-paired differences.

The California construction outliers persist. In the corrected run, a PSO construction model predicts 663.289 for a held-out target of 1.375. Its largest reported feature contribution comes from a synthetic feature valued at 24,866.262 with coefficient 0.427424. `extreme_residuals.csv` records the five worst residuals per optimizer run; `feature_diagnostics.csv` contains training and test feature magnitudes and fitted coefficients. Both were computed after optimizer selection. Replaying every saved feature definition reproduced the recorded held-out MAE within 1e-6.

`ridge_probe.csv` is an exploratory downstream-model check on the fixed California PSO construction features. Training-only scaling plus inner-tuned Ridge has mean MAE 0.522 and RMSE 1.332, close to the original linear fit (0.522 and 1.324). Adding a training-derived 99.5th-percentile clip before scaling and Ridge gives mean MAE 0.478 and RMSE 0.657. This probe was chosen after inspecting the outliers and does not re-optimize features under the new downstream model. It motivates a separate, prespecified robustness experiment; it is not an unbiased claim that clipping is the best method.

`stability.csv` matches constructed features one to one across seeds and averages their absolute correlations, counting unmatched features as zero. Construction-only runs correctly report all original inputs as selected. Comparing GP with GA, PSO, or ES still mixes symbolic trees with linear and squared combinations. `../../comparison_design.md` specifies a shared-representation protocol for the next optimizer study.

This stricter stability measure lowers mean constructed-feature similarity for GA combined from 0.640 under the old maximum-pair measure to 0.479, and for PSO combined from 0.620 to 0.452. Construction-only selection similarity is 1.0 because every original input is included by definition.
