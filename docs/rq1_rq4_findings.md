# Findings from RQ1–RQ4

**Date:** 29 October 2026  
**Datasets:** California Housing and Diabetes Progression  
**Status:** Completed exploratory analysis; no independent confirmation set

## Executive summary

Feature construction improved mean outer-test MAE more consistently than feature selection in these two datasets. The preferred construction depends on the data and learner. Radial basis features (RBF) were particularly effective on California Housing; quadratic projections were more effective for Diabetes Progression with OLS and Ridge. In the reduced RQ4 study, RBF also improved Random Forest against its own baseline on all five splits of both datasets, although the gains were modest relative to its large search cost. No optimizer won across both datasets and all fixed feature counts. These are exploratory findings from reused datasets and overlapping outer splits, not independent confirmation of a universally best pipeline.

## Design and interpretation

All studies use five shared 80/20 outer train/test splits and three inner folds. Candidate transformations are scored by inner-fold MAE; the selected transformation and learner are refitted on the outer development data and measured on the held-out outer test. RQ1–RQ3 use five stochastic search seeds per split, while the compute-limited RQ4 uses two. RQ4 tunes Ridge and Random Forest on the original development features, then freezes their parameters during feature search. The Random Forest has 100 trees.

The primary result below is **mean percentage reduction in outer-test MAE against the same learner's no-feature-engineering baseline**. Positive values mean lower error. For each arm, search-seed results are averaged *within each outer split*, then the five split percentages are averaged. A “positive splits” count is based on those five split means; search seeds are not treated as independent test sets. Mean MAEs are shown in each dataset's native target units. Percentage changes need not equal ratios of the displayed mean MAEs, because they are calculated on each split before averaging.

| Study | Main contrast | Candidate evaluations per FE arm/run | Search seeds | Result rows |
| --- | --- | ---: | ---: | ---: |
| RQ1 | Selection, quadratic construction, combined | 1,000 total across eligible \(k\) | 5 | 160 |
| RQ1 fixed \(k\) | Construction versus combined at \(k=1,2,4\) | 1,000 at one \(k\) | 5 | 310 |
| RQ2 | Quadratic, RBF, interactions, symbolic GP | 1,000 total across eligible \(k\) | 5 | 220 |
| RQ2 fixed \(k\) | Four families at \(k=1,2,4\) | 1,000 at one \(k\) | 5 | 610 |
| RQ3 | GA, PSO, ES, DE, CMA-ES at \(k=1,2,4\) | 1,000 at one \(k\) | 5 | 760 |
| RQ4 reduced | Quadratic and RBF across OLS, Ridge, forest | 500 total across eligible \(k\) | 2 | 150 |

The adaptive construction grid is \(k\in\{0,1,2,4,8\}\). For construction-only, \(k=0\) uses the original-feature baseline inner score and consumes no candidate evaluations; the 1,000- or 500-candidate allowance is divided across the four positive values. Thus fixed-\(k\) versus adaptive-\(k\) contrasts also change the search effort *per* \(k\). Repeated deterministic arms across RQ1, RQ2 and RQ3 are sensitivity checks, not independent replications.

## RQ1 — Which feature-engineering strategy helps?

OLS, a genetic algorithm (GA), and quadratic projection features were held fixed. Selection retains a subset of original columns; construction appends \(k\) squared projections to all originals; combined retains a subset directly and appends the projections.

| Dataset | Strategy | Mean test MAE | Mean MAE reduction | Positive splits |
| --- | --- | ---: | ---: | ---: |
| California | Original features | 0.5359 | — | — |
| California | Selection | 0.5360 | −0.01% | 3/5 |
| California | Construction | 0.5241 | **+2.21%** | 4/5 |
| California | Combined | 0.5314 | +0.85% | 4/5 |
| Diabetes | Original features | 45.1491 | — | — |
| Diabetes | Selection | 45.4591 | −0.71% | 1/5 |
| Diabetes | Construction | 43.7882 | **+2.96%** | 5/5 |
| Diabetes | Combined | 44.6023 | +1.15% | 4/5 |

Construction gave the largest average MAE gain on both datasets. Selection alone added no practical value here, and the combined search generally underperformed construction alone. The fixed-\(k\) study supports the same direction: California construction improved MAE by 1.34%, 2.35% and 2.36% at \(k=1,2,4\), versus 0.87%, 0.49% and 1.68% for combined. Diabetes construction improved by 2.90%, 2.75% and 2.68% at those counts; combined improved by 1.67%, 1.26% and 0.23%.

Across the five split means, adaptive construction's California gains ranged from −0.54% to +4.49%; Diabetes gains ranged from +0.50% to +5.80%. These ranges describe split variation, not confidence intervals from independent samples.

The capacity choice is data dependent. Adaptive construction selected \(k=8\) in 13/25 California runs, but Diabetes used \(k=1,2,4,8\) in 8, 4, 8 and 5 runs. Frequent selection of the grid edge does not establish that a larger grid would generalize better. California quadratic construction's mean RMSE rose from 0.7455 at baseline to 0.9032 despite its lower MAE, so a few large errors offset its typical-error improvement. Its highest recorded absolute test residual was 73.36 versus 11.81 for the baseline.

**Interpretation of original-feature counts:** Combined construction may use a column inside a new feature even when the direct-input mask drops it. The combined arm directly retained a mean of 7.00/8 originals in California and 7.08/10 in Diabetes, but its mean count of originals used directly *or indirectly* was 7.92/8 and 10/10. `effective_original_count` is an algebraic usage count: quadratic coefficients with absolute value above \(10^{-12}\) count as active. It does not estimate feature importance or effective sparsity. `selected_k` counts constructed columns; `transformed_dim` counts all columns presented to the learner.

## RQ2 — Which representation family helps?

All arms append constructed features to the original inputs and use OLS. The parametric families are squared projections \((w_j^\top x)^2\), localized RBFs \(\exp[-\gamma_j\lVert x-c_j\rVert^2]\), and explicit products \(x_{a_j}x_{b_j}\). Symbolic GP evolves bounded trees of variables, constants and `+`, `−`, `×`, protected division, with maximum depth three. GP changes both representation and search operators, so its comparison is an **end-to-end pipeline** contrast rather than an isolated family or optimizer effect.

### Adaptive \(k\), 1,000 total evaluations per arm

| Dataset | Family | Mean test MAE | Mean MAE reduction | Positive splits | Median selected \(k\) |
| --- | --- | ---: | ---: | ---: | ---: |
| California | Quadratic | 0.5241 | +2.21% | 4/5 | 8 |
| California | RBF | 0.4805 | **+10.34%** | 5/5 | 8 |
| California | Interactions | 0.5208 | +2.82% | 4/5 | 8 |
| California | GP | 0.5254 | +1.98% | 4/5 | 4 |
| Diabetes | Quadratic | 43.7882 | **+2.96%** | 5/5 | 4 |
| Diabetes | RBF | 44.3587 | +1.80% | 4/5 | 4 |
| Diabetes | Interactions | 44.5904 | +1.20% | 3/5 | 4 |
| Diabetes | GP | 45.3640 | −0.57% | 3/5 | 4 |

California RBF improved the five split means by 8.83%–12.31%. It also improved mean RMSE from 0.7455 to 0.6707 and mean \(R^2\) from 0.5847 to 0.6637. The other three adaptive families lowered mean MAE but worsened California RMSE: quadratic 0.9032, interactions 1.2488, and GP 1.2035. Interactions and GP had negative mean \(R^2\) (−0.7740 and −0.6400), reflecting occasional severe errors. On Diabetes, quadratic improved mean RMSE from 55.5852 to 54.3991, while GP worsened it to 55.7581.

The fixed-\(k\) comparison supports the family distinction while exposing search-budget sensitivity. With 1,000 evaluations concentrated at \(k=4\), California RBF improved MAE by **12.24%** and Diabetes RBF by **3.05%**; the adaptive RBF improvements were 10.34% and 1.80%. Diabetes quadratic at fixed \(k=1\) improved by 2.90%. GP at fixed \(k=4\) improved California by 3.44% but worsened Diabetes by 0.36%. These are not pure capacity effects, because a fixed arm gives all evaluations to one \(k\).

The GP expressions were less reproducible across search seeds: mean matched constructed-feature similarity was 0.225 in California and 0.149 in Diabetes, versus 0.630/0.301 for RBF and 0.648/0.368 for quadratic. Six GP adaptive runs contained a constant constructed feature. The deterministic polynomial-plus-Ridge reference is not capacity matched and has a different learner; its California mean MAE of 0.5327 came with mean RMSE 3.5297 and mean \(R^2=-32.33\), another warning against choosing by MAE alone without tail diagnostics.

## RQ3 — Which optimizer helps on a fixed task?

Here the task is held fixed: quadratic construction, OLS, one of \(k=1,2,4\), common bounds and 1,000 candidate evaluations. Table entries are mean outer-test MAE reductions (%) against OLS on original features. Higher is better. Each cell summarizes five split means, each averaged over five search seeds.

| Dataset | \(k\) | GA | PSO | ES | DE | CMA-ES |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| California | 1 | +1.34 | +1.61 | +2.00 | +1.21 | **+2.30** |
| California | 2 | +2.35 | +2.36 | +2.49 | +1.66 | **+2.98** |
| California | 4 | +2.36 | +2.11 | +2.75 | +1.71 | **+2.87** |
| Diabetes | 1 | +2.90 | +2.92 | **+3.22** | +3.03 | +2.41 |
| Diabetes | 2 | +2.75 | +2.68 | +2.23 | **+2.91** | +1.94 |
| Diabetes | 4 | **+2.68** | +1.90 | +2.62 | +2.20 | +1.36 |

CMA-ES had the largest mean MAE gain at all three California feature counts, but not at any Diabetes count. The spread between the best and worst optimizer within a dataset/\(k\) cell was about 0.8–1.3 percentage points. Median run times were similar within a fixed task: roughly 6 seconds at California \(k=1\), 10 seconds at \(k=2\), 12–13 seconds at \(k=4\), and about 2 seconds for Diabetes. All five methods completed their equal candidate budgets without recorded failed candidates or outer fits. Thus there is a California-specific CMA-ES signal, but no universal optimizer replacement. RQ3 did not run a separate adaptive-\(k\) optimizer comparison.

## RQ4 — Does the benefit transfer across learners?

The completed reduced RQ4 used GA and the two shortlisted continuous families, each with 500 total evaluations over the adaptive \(k\) grid and two seeds per split. Each learner is compared with **its own** no-FE baseline within RQ4. The table gives mean test MAE, mean percentage reduction, and number of positive split means.

| Dataset | Learner | No-FE MAE | Quadratic MAE (gain; splits) | RBF MAE (gain; splits) |
| --- | --- | ---: | ---: | ---: |
| California | OLS | 0.5359 | 0.5273 (+1.62%; 4/5) | **0.4929 (+8.03%; 5/5)** |
| California | Ridge | 0.5359 | 0.5272 (+1.62%; 3/5) | **0.5074 (+5.32%; 5/5)** |
| California | Forest | 0.3304 | 0.3309 (−0.14%; 2/5) | **0.3251 (+1.61%; 5/5)** |
| Diabetes | OLS | 45.1491 | **43.9043 (+2.78%; 4/5)** | 44.6232 (+1.20%; 4/5) |
| Diabetes | Ridge | 45.3939 | **44.2909 (+2.44%; 4/5)** | 44.5177 (+1.93%; 5/5) |
| Diabetes | Forest | 47.6986 | 47.1772 (+0.89%; 3/5) | **46.2997 (+2.85%; 5/5)** |

RBF improved average MAE for every learner/dataset combination and every forest split. Its observed forest gains are small: California mean MAE fell by 0.0053 (1.61%) and Diabetes by 1.3989 (2.85%). California forest RBF also lowered mean RMSE from 0.5107 to 0.5050 and raised mean \(R^2\) from 0.8051 to 0.8094. Diabetes forest RBF lowered mean RMSE from 58.9110 to 56.7263 and raised mean \(R^2\) from 0.4690 to 0.5061. On Diabetes, the best *absolute* MAE among the tested learners was OLS with quadratic construction (43.9043), not the forest.

The five California forest RBF split means gained between +0.71% and +3.46%; the Diabetes forest split means gained between +1.03% and +6.96%. California OLS RBF gains ranged from +6.28% to +10.68%. These descriptive ranges reflect overlapping splits and do not establish population-level uncertainty.

The selected capacity changes by learner. In California, OLS/Ridge generally chose larger \(k\) for both families (median 8), whereas the forest chose median \(k=1\) for quadratic and \(k=4\) for RBF. On Diabetes, median selected \(k\) ranged from 2.5 to 8 across learner/family combinations. This supports reselecting \(k\) for each learner rather than transferring an OLS-selected feature count unchanged. California quadratic OLS again improved MAE while worsening mean RMSE (0.8808 versus 0.7455 at baseline); California RBF OLS improved both (RMSE 0.6832, \(R^2=0.6510\)).

**Compute trade-off:** RQ4 completed all 60,000 candidate evaluations and 150 result rows in 62.77 recorded active hours, below its 118-hour cap. Its 180,120 recorded FE fits include candidate scoring and final refits. California forest searches accounted for approximately 59.5 hours of summed arm runtime, about 95% of total active time. The reduced design used two rather than five search seeds and 500 rather than 1,000 candidates per arm; its smaller search budget can understate achievable gains and makes stochastic variability less precise. Compare RQ4 arms *within* RQ4, not their values directly with the larger-budget RQ1–RQ3 arms as if only the learner changed.

## Conclusions and limits

1. **RQ1:** Construction is the supported strategy on these datasets; selection alone does not help, and combining a direct-input mask with construction usually helps less than construction alone.
2. **RQ2:** RBF is the strongest tested family for California, with improvements in MAE and RMSE. Quadratic construction is the more reliable option for Diabetes with a linear learner. Explicit interactions and GP do not justify replacing these families in this study, particularly given the observed tail errors and GP instability.
3. **RQ3:** Optimizer preference depends on dataset and fixed \(k\). CMA-ES leads the California quadratic comparisons, but that result does not transfer to Diabetes. GA remains a defensible common search method for RQ4 rather than a claimed universal winner.
4. **RQ4:** Engineered features can help OLS, Ridge, and Random Forest, but the benefit's size and best family depend on the learner and dataset. RBF's forest gains are consistent across the five splits yet modest relative to the computation required. A no-FE baseline remains essential when deploying a new learner or dataset.

No run recorded failed candidate folds, failed candidates, fallback final models, or failed outer fits. Nevertheless, the five outer splits overlap, and both datasets have been examined during the research programme. These results support **exploratory, dataset-specific** conclusions, not a formal generalization claim or an independently confirmed winning pipeline. A future confirmation study would need a prospectively locked pipeline and genuinely untouched data or additional datasets.

**Source artifacts:** [RQ1](../results/rq1/runs.csv), [RQ1 fixed](../results/rq1_fixed/runs.csv), [RQ2](../results/rq2/runs.csv), [RQ2 fixed](../results/rq2_fixed/runs.csv), [RQ3](../results/rq3/runs.csv), and [RQ4 reduced](../results/rq4_120h/runs.csv). The corresponding `protocol.json`, `definitions.jsonl`, `convergence.csv`, `stability.csv`, and RQ4 `progress.json` files are in the same result directories. Methods and prior constraints are recorded in [the research plan](research_plan.md) and [the RQ4 reduction note](rq4_120h.md).
