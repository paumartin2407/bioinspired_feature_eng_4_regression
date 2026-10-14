# Research refocus on the bio-inspired feature engineering for regression project

**Status:** Revised research specification — 14 October 2026
**Repository inspected:** [paumartin2407/bioinspired_feature_eng_4_regression](https://github.com/paumartin2407/bioinspired_feature_eng_4_regression)
**Repository state:** Default-branch tree `e694a045905bd1cd274f7edeb3775c977dee7a6e`; latest checked GitHub Actions run passed.
**Scope:** A *new*, coherent experimental programme. **Reuse useful code, not prior experimental results.** Old results do not form part of the new study or its evidence base.

> **Central objective:** Given a regression learning procedure \(M\) and a dataset \((X,y)\), discover a bio-inspired feature transformation \(T\) such that a new instance of \(M\), fitted on \((T(X_{\mathrm{train}}),y_{\mathrm{train}})\), predicts unseen observations better than the same learning procedure fitted on the original features.

The project studies *feature engineering*, not the proliferation of metaheuristic algorithms. It has four research questions: **RQ1 strategy**, **RQ2 representation**, **RQ3 optimizer**, and **RQ4 model generality**. It ends with a prospectively evaluated recommendation of the best-supported configuration(s).

---

## 1. Define the learning problem precisely

For dataset \(D=(X,y)\), let:

- \(S\) be the **feature-engineering strategy**: no transformation, feature selection (S), feature construction (C), or selection + construction (S+C).
- \(R\) be a **representation family** for constructed features (quadratic projections, radial basis functions, sparse interactions, GP expressions, etc.).
- \(A\) be a **bio-inspired optimizer** (GA, PSO, ES, DE, CMA-ES, or structural GP where applicable).
- \(k\) be the **number of constructed features**, which must be chosen rather than assumed to equal 2.
- \(M\) be the **downstream regression learning procedure**, including the estimator and a specified hyperparameter-tuning rule.

A learned feature pipeline is

\[
T_{S,R,\theta,k}: \mathbb{R}^d \rightarrow \mathbb{R}^{p(S,\theta,k)}.
\]

For each candidate \((\theta,k)\), fit \(M\) using **training data only**, then calculate the prespecified validation error. The preferred pipeline minimizes validation MAE (or mean inner-fold MAE), and is finally measured on unseen test data.

The relevant effect is **improvement relative to the same learner without feature engineering**:

\[
\Delta \mathrm{MAE}_{D,M}=
\mathrm{MAE}_{\mathrm{test}}(M\circ T)-
\mathrm{MAE}_{\mathrm{test}}(M\circ \mathrm{Id}).
\]

Negative values mean the FE pipeline helped. Report percentage reduction as well as absolute differences. The original target \(y\) remains unchanged.

**Retraining is essential:** The project changes the feature space passed to the learning algorithm. It is not trying to modify the inputs of an immutable, already-trained predictor whose input shape cannot change.

**What “arbitrary model” means here:** the framework should accommodate a standard sklearn-like regressor, but cross-model benefit is an *empirical question* tested in RQ4, not a property to assume.

---

## 2. Clean break from the previous project: reuse implementation, discard old experiments

### 2.1 Non-negotiable decision

Treat the revised programme as a **fresh study**, not as an additional milestone appended to the previous ones.

- **Do not import, rank, summarize, combine, or cite old experiment results** from `results/milestone1/`, `results/equal_budget/`, or `results/milestone2/` in any results table, conclusion, or model-selection procedure for the new research.
- **Do not retain an archive or “historical results” area as part of the refocused project.** Remove obsolete generated outputs from the *new project working tree* rather than copying them forward.
- **Reuse code, tests and ideas only after checking they implement the new research contract.** Existing tests are infrastructure, not scientific evidence.
- Do not use previous apparent “winners,” specific test outliers, or post-hoc clipping probes as selection rules. All new candidate choices and preprocessing policies must be prespecified or selected using data confined to the new study's development folds.
- Keep the new study's data partitions, results and configuration in clean directories, with no dependency on old result files. Git history may exist outside the new study; it is not part of the research deliverable. Rewriting remote history is **not required**.

### 2.2 File-level disposition (based on the inspected GitHub tree)

| Current file/area | Decision | Specific action |
|---|---|---|
| `data.py` | **Reuse/adapt** | Generalize loaders; implement fitted, leak-free preprocessing inside inner folds and final refit. |
| `evaluation.py` | **Reuse/adapt** | Preserve metrics and exact candidate-evaluation counter; add model factories and per-fold fit tracking. |
| `features.py` | **Partially reuse** | Extract original-feature masks and construction decoders into unified strategies/representation modules. |
| `optimizers.py` | **Salvage useful operators** | Reuse GA/PSO/ES logic only after separating optimizer mechanics from hardcoded data/feature types. |
| `continuous_optimizers.py` | **Reuse/adapt** | Use tested GA, PSO, ES, DE and CMA-ES components for genuinely continuous problems. |
| `continuous_features.py` | **Refactor, do not retain as universal API** | Replace fixed `N_CONSTRUCTED=2` with variable \(k\), supporting \(k=0\) when appropriate. |
| GP tree generation, mutation, crossover and serialization inside `optimizers.py` | **Reuse selectively** | Move to a GP feature-construction family; evolve \(k\) expressions or a bounded variable-length set. |
| `benchmark.py`, `benchmark_continuous.py` | **Replace as active entry points** | Extract runner/recording utilities; create one new experimental runner with RQ configuration files. |
| `test_milestone1.py`, `test_milestone2.py` | **Harvest and replace** | Move reusable tests to `tests/`; add variable-\(k\), decoding, budgeting, model, fold-isolation and replay tests. Remove tests tied solely to retired scripts. |
| `.github/workflows/tests.yml`, `requirements.txt` | **Keep/adapt** | Switch CI to new tests and smoke experiments; keep a reproducible environment. |
| `dataset/` | **Reuse source files** | Use California Housing and Diabetes Progression only; do not reuse old output-driven choices. |
| `results/milestone1/`, `results/equal_budget/`, `results/milestone2/`, all smoke/legacy result directories | **Discard from refocused repository** | Delete obsolete result artefacts from the new active tree. Generate only new `results/rq*/` outputs. |
| `compare_budgets.py`, `refresh_stability.py`, `replay_diagnostics.py`, `probe_regularization.py` | **Remove if not needed** | Their old-result comparison workflows are out of scope. Salvage generic functions only if required in the new design. |
| `comparison_design.md` and prior research narratives | **Replace** | This document is the new single research specification. |
| GP→GA/PSO/ES hybrids, NSGA-II, additional novelty metaheuristics | **Exclude from main scope** | Do not build or keep these as active experiment arms without a newly stated RQ. |

**Repository cleanup should be explicit:** make a new branch, extract tested reusable code, move the new research plan into `docs/research_plan.md`, then delete obsolete scripts/results from the active branch. Verify the new benchmark does not read any old `results/` file. **Do not automatically delete local-only work or rewrite Git history.**

### 2.3 A minimal target layout

```text
src/
  data.py                  # datasets, fold construction, fitted preprocessing
  strategies.py            # none / selection / construction / both
  representations.py       # parametric feature families; supports variable k
  gp_features.py           # expression-tree feature generator
  optimizers.py            # GA / PSO / ES / DE / CMA-ES adapters
  evaluation.py            # learner factory, nested validation, budgets, metrics
  experiments.py           # protocol-driven runner and output recording
configs/
  datasets.yaml
  rq1.yaml
  rq2.yaml
  rq3.yaml
  rq4.yaml
  final.yaml              # disabled without an independent evaluation panel
tests/
docs/
  research_plan.md
results/
  rq1/
  rq2/
  rq3/
  rq4/
```

These names are recommendations, not a requirement to reorganize every source file immediately. Keep the implementation small and testable.

---

## 3. A crucial design variable: the number of new features \(k\)

### 3.1 Why \(k\) must not be fixed to 2

The earlier continuous feature representation generated exactly two features:

\[
T_{W,2}(x)=[x,(w_1^\top x)^2,(w_2^\top x)^2].
\]

That was useful for a controlled *optimizer* comparison, but it is an artificial restriction for a feature-engineering study. Different datasets and learners may benefit from **zero, one, several, or many** constructed features.

More features also mean greater flexibility, larger search spaces, added runtime, redundancy, and increased overfitting risk. Therefore \(k\) is a **capacity hyperparameter of the feature pipeline** and must be selected on development data, never on a held-out test set.

Define \(k=0\) as **no constructed features**. Note that `selection-only` can still reduce the original feature count; `construction-only` with \(k=0\) is the unchanged-input baseline.

### 3.2 Recommended primary solution: nested choice of \(k\)

Start with a bounded, prespecified grid such as:

\[
\mathcal{K}=\{0,1,2,4,8\}
\]

but **cap it based on input dimension, dataset size, representation family and compute** before running any experiments. This is an illustrative starting grid, not a claim that 8 is always sensible. A small dataset may require a smaller upper bound. For RBF, GP and polynomial interactions, comparable numbers of features may have very different effective complexity.

For every dataset, outer split, representation \(R\), optimizer \(A\), learner \(M\), and applicable FE strategy:

1. Evaluate each eligible \(k\) using **only inner-training / inner-validation data**; learn a fresh transformation for each \(k\).
2. Use an **equal total computational selection allowance across strategies/families** when comparing their search efficiency. If running every \(k\) with 1,000 evaluations, log the true combined cost; to match total cost, divide a prespecified search allowance among candidate \(k\) values.
3. Select \(k^*\) by the lowest mean inner-validation MAE, with a prespecified tolerance-based tie-break toward **smaller \(k\)**; never choose by outer-test score.
4. Refit the selected FE procedure and learner using the permitted development data under a consistent refitting policy, and evaluate **once** on that outer test partition.
5. Save the validation scores for all \(k\), selected \(k^*\), candidate/fit counts, actual transformed dimension and runtime.

This makes **automatic feature-count selection part of the feature-engineering pipeline**.

### 3.3 How to interpret \(k\) without confounding the RQs

**Two complementary analyses are needed:**

- **Fixed-\(k\) controlled contrasts:** Compare strategies or representation families at matched \(k\in\{1,2,4\}\) (or another prespecified common feasible subset). This isolates a question while holding the *nominal number of new features* fixed. It does **not** equalize true expressive complexity across feature families.
- **Adaptive-\(k\) practical contrasts:** Allow each method to choose its own \(k^*\) from the same prespecified grid using inner data. This measures the effectiveness of a **complete usable pipeline**, including complexity selection. It is the primary basis for the final recommendation.

RQ3 needs an additional control: compare optimizers **at a fixed shared \(k\)**, or give every optimizer the **same inner-\(k\)-selection procedure and total resource budget**. Otherwise better results might come from choosing a different feature count rather than from a better search mechanism.

### 3.4 Alternatives if grid search becomes too expensive

A variable-length chromosome or an overcomplete representation with active/inactive feature slots can optimize \(k\) jointly with feature coefficients, but it introduces new discrete operations and complicates comparison with DE/CMA-ES. Do **not** make that the primary design.

Use the bounded inner-validation grid first. Consider joint \(k\)-optimization only after the main experiments if it becomes a distinct research question.

**Avoid outer-test-driven tuning of \(k\).** The test data may be used to report the distribution of \(k^*\) selected using development data, but never to determine its value.

---

## 4. Common experimental contract

The following applies to *all* four RQs.

### 4.1 Datasets and evaluation data

- Use only **California Housing and Diabetes Progression** for RQ1–RQ4. None of the older saved experiments or selected configurations enters the new study. Past test observations from these sources have already been examined, so new splits of them cannot provide independent confirmation.
- Treat findings as specific to these two datasets; do not infer broad performance across regression problems from this panel.
- Define dataset inclusion criteria, target handling, missing-value policies, train/test partition rules and random seeds **before** inspecting new results.
- Keep the outer-test rows isolated during all transformation/optimizer/learner selection. All experiments within a dataset share the same locked outer splits for paired comparisons.

### 4.2 Validation and retraining

Use 5 prespecified outer 80/20 splits and 3 fixed inner folds over each outer-development partition as an initial practical protocol. Use a reasonable alternative for very small datasets, but define it in advance.

For every inner fold:

1. Fit imputation, one-hot encoding, scaling and any bounds/normalization **only on the inner-training fold**.
2. Fit the feature engineer on inner-training rows and evaluate its candidates on inner-validation rows.
3. Fit the downstream learner with fixed, declared hyperparameters, or use a properly nested and budgeted learner-tuning policy. Do not tune on the outer test partition.
4. Do not let a transform trained on the whole development partition leak into the inner-fold fitness evaluation.

**Important for GP and stochastic FE:** define whether an individual consists of a reusable formula/vector evaluated across folds, or whether independent FE optimization occurs per fold. For candidate selection across folds, use the same mathematical candidate and fold-specific fitted preprocessing; the objective is mean validation MAE across folds.

After inner selection, refit transforms and the learner on the outer-development partition using the same specified procedure; evaluate on the outer test. If the FE optimization itself is validation-driven, reserve an internal validation set or cross-fitted fitness inside outer development rather than fitting and scoring on the same observations.

### 4.3 Objectives and metrics

**Primary:** paired outer-test MAE reduction relative to the same model's no-FE baseline. Also report:

- Relative MAE reduction (%) on each dataset, not pooled raw MAE across target scales.
- RMSE, R², 95th/99th percentile absolute residual and maximum absolute residual.
- Mean and distribution of selected \(k^*\), final original-feature count and constructed-feature count.
- Number of fitness candidates evaluated, actual model fits, runtime and failed/nonfinite candidates.
- Stability of selected originals and constructed features (one-to-one matched correlations for numerical features; expression complexity for GP).

Only **inner-development evidence** can choose a strategy, family, optimizer, hyperparameters or \(k\). Extra diagnostics on the outer test are for reporting, not for making an adjustment and reporting the same test as confirmation.

### 4.4 Fair comparisons and statistical reporting

- **Matched data:** same datasets/splits and comparable seed schedules for paired experiments.
- **Matched evaluation objective:** same learner, metric and validation folds when isolating strategy, representation or optimizer.
- **Matched resource accounting:** report *total* optimizer evaluations across all tested \(k\), inner folds, model hyperparameter candidates and multistage procedures, as well as wall-clock time. A candidate evaluated on 3 folds should count as one candidate and **three fitted models**; clarify any additional fits.
- **Seed variability:** use at least 5 stochastic search seeds for full configurations, but average seeds within outer split before constructing paired dataset-level contrasts; avoid treating all seeds as independent data samples.
- **Avoid trivial baselines:** include each learner's own no-FE baseline, regularized original-feature baselines where relevant, and deterministic representation-specific alternatives such as polynomial + Ridge if informative.
- **Selection bias:** RQ1–RQ4 exploration and pipeline choice share two previously examined datasets. Report their comparisons as exploratory; no independent final evaluation is available under this scope.

---

## 5. RQ1 — Which feature-engineering strategy helps?

**Question:** Does *selection*, *construction* or *selection + construction* improve regression more than the original features?

**What varies:** Only the strategy \(S\). **Fixed:** learner = OLS LinearRegression; optimizer = GA; constructed-feature family = squared projections; preprocessing, folds, budgets and candidate \(k\) grid consistent.

### Arms

| Arm | Transformed input | Search |
|---|---|---|
| No FE | \(X\) | None; learner baseline |
| Selection-only (S) | \(X_{\mathcal I}\) | Binary mask of retained original variables |
| Construction-only (C) | \([X,z_1,\ldots,z_k]\) | Construct \(k\) features; retain original variables |
| Selection + construction (S+C) | \([X_{\mathcal I},z_1,\ldots,z_k]\) | Original-feature mask + new-feature parameters |

**Primary pragmatic comparison:** every construction arm chooses \(k^*\) from its **own inner validation** using the **same prespecified grid and total selection budget**. Selection-only has no \(k\) of synthetic variables.

**Diagnostic matched-capacity comparison:** use the same fixed \(k\) for C and S+C (e.g., \(k=1,2,4\)); compare S separately by selected-original count. The combined arm has a larger search space—do not attribute its advantage or disadvantage solely to the strategy without noting search difficulty.

**Important design nuance:** in S+C, synthetic features may use **all** original variables even if some are dropped from the direct regression input, unless you explicitly constrain construction to selected inputs. Select and document one interpretation, as it changes the hypothesis. A clean default is to constrain constructed features to the selected subset if interpretability is central, and then document that the effective search space differs.

**Outputs:** paired MAE improvements against no FE; C vs S vs S+C contrasts; selected \(k\); selected-original count; runtimes; stability and failure rates. The answer should be conditional on dataset and learner, not an unconditional claim that construction always wins.

**Gate:** the same runner reproduces no FE, S, C and S+C without changing model/fold/fitness code, and selects \(k\) exclusively from inner data.

---

## 6. RQ2 — Which feature representations are useful?

**Question:** Do genuinely different mathematical families of constructed features change the utility of bio-inspired FE?

**What varies:** representation family \(R\). **Fixed for the core comparison:** construction-only; GA or a declared common compatible optimizer; OLS; inner metric, partitions and accounting.

### Proposed representation families

| Family | Form | Why meaningfully different |
|---|---|---|
| **Quadratic projection** | \(z_j=(w_j^\top x)^2\) | Global low-rank quadratic interactions, current implementation can be generalized to variable \(k\) |
| **Radial basis (RBF)** | \(z_j=\exp(-\gamma_j\lVert x-c_j\rVert^2)\) | Localized nonlinear effects, unlike global quadratic projections |
| **Sparse explicit interactions** | \(z_j=x_{a_j}x_{b_j}\) or \(x_i^2\) | Discrete selection of individual polynomial interactions rather than dense coupled projection terms |
| **Symbolic GP feature construction** | \(z_j=f_{\mathrm{tree},j}(x)\) | Evolves mathematical structure and operator compositions, not just predefined coefficients |

Keep a deterministic polynomial baseline as a *reference*, but do not treat its vastly different feature count as a fair capacity-matched counterpart without qualification.

### Two separate experiments inside RQ2

**RQ2-A: parametric / explicit families.** Run quadratic, RBF and sparse interactions with a common GA framework and similar evaluation rules. Compare (i) **matched fixed \(k\)** and (ii) **independently inner-selected \(k^*\)**. Because the genes and search operators may differ (continuous versus integer index), this isolates complete *representation-compatible GA pipelines*, not identical continuous search dynamics.

**RQ2-B: GP versus parametric pipelines.** GP should evolve **a set of up to \(k\) expressions**, not a predictive equation replacing \(M\). Compare GP's symbolic feature generation with selected fixed-form construction pipelines using matched data, learner and *total* search budgets. Label the contrast a **pipeline comparison**: symbolic tree evolution changes both representation and search mechanics. Compare fixed \(k\) as well as development-selected \(k^*\). Control maximum tree depth, grammar, nonfinite behavior, and expression complexity.

**Capacity selection:** all families use an eligible \(k\) grid chosen *before* evaluation. For GP additionally constrain max depth/node count; \(k\) alone does not measure symbolic complexity. Track redundant/constant features and effective rank. If \(k=0\), route to no-FE rather than running a feature optimizer.

**Outputs:** family-wise paired gains, chosen \(k\), cost, expressiveness diagnostics, complexity, feature stability and robustness.

**Gate:** determine whether at least one genuinely distinct family provides repeatable benefit, and carry **no more than two** relevant continuous families into RQ3. If none helps, keep the baseline as a legitimate result.

---

## 7. RQ3 — Which bio-inspired optimizer is better for a fixed FE task?

**Question:** With feature strategy, family, \(k\), learner and validation objective controlled, does the choice of search algorithm improve predictive performance or efficiency?

**Optimizers:** GA, PSO, ES, DE/rand/1/bin and CMA-ES.

**Fixed core task:** construction-only, a selected continuous parametric family (initially quadratic projections, optionally RBF), OLS LinearRegression, common decoder/bounds, identical validation folds and 1,000 candidate evaluations **per fixed-\(k\) optimization run**.

### Separate optimizer quality from \(k\)-selection quality

1. **Primary mechanistic contrast:** Compare all five algorithms at the **same prespecified fixed** \(k\) values (e.g. \(1,2,4\)). Each method solves the same \(k\)-specific continuous search space. Results are grouped by \(k\); do not average them into a single unqualified ranking.
2. **Practical adaptive-\(k\) contrast:** Optionally run the exact same outer-inner \(k\)-selection grid for all optimizers, with the **same total evaluation allocation across \(k\)**. Each method may then select a different \(k^*\), which is part of its pipeline. Report separately from the fixed-\(k\) result.
3. Preserve algorithm-specific update mechanisms and their justified initial sampling distributions; disclose that a common domain is not an identical probability distribution.
4. Count candidate evaluations, inner-fold fits, training runtime, numerical errors and best validation MAE versus evaluation count.

**Do not include GP as a sixth method in this comparison.** GP optimizes structures in a different search domain and belongs to RQ2 as a distinct feature-construction pipeline.

**Outputs:** split-paired test changes against no FE and against other optimizers, stability, runtimes, convergence and failure rates, conditioned on family and \(k\).

**Gate:** identify practical ties and context-dependent gains, not declare a universal metaheuristic champion.

---

## 8. RQ4 — Does feature engineering transfer across regression learners?

**Question:** Is the FE framework effective for models beyond ordinary linear regression, and does the preferred transformation change with \(M\)?

**Learner panel:**

1. **Ordinary LinearRegression:** strong baseline for testing whether new features recover missing nonlinear structure.
2. **Ridge:** tests the value of regularization when synthetic features are correlated or unstable.
3. **RandomForestRegressor:** can already model nonlinear effects; tests whether engineered features add measurable value.

**Design:** Compare each learner's own tuned/no-FE baseline against **2–3 shortlisted** strategy–family–optimizer pipelines, with their *\(k\) chosen inside that learner's development data*. Do not assume an OLS-optimal transformation or feature count works for Ridge or Random Forest. Re-optimize the FE pipeline for each learner.

Fix a reproducible hyperparameter-selection protocol. Either pre-tune each learner on development data and hold its configuration fixed during FE search, or use a carefully nested tuning procedure that charges and reports all model fits. Avoid hidden extra tuning for some arms.

**Optional secondary question:** learn \(T\) using OLS and transfer it without modification to a newly fitted Ridge or Random Forest. This measures transfer and is **not** the main model-agnostic FE objective.

**Outputs:** per-learner paired gain, selected \(k^*\), proportion of datasets helped, RMSE/R² robustness, fit-time overhead and simplicity. Non-improvement for a powerful learner is a meaningful negative result.

**Gate:** state where a learned feature-engineering transformation contributes beyond a learner's built-in inductive bias.

---

## 9. Synthesis within the two-dataset scope

**Goal:** Identify an effective and reproducible choice of **strategy + representation + optimizer + selected \(k\) + learner**, not merely report the smallest MAE from all attempted runs.

### Selection and reporting

1. Shortlist **at most 2–3 practical pipelines** using development-only evidence from RQ1–RQ4. Always include the applicable **no-FE baseline**.
2. Fix their entire procedures: preprocessing, \(k\)-selection grids, feature family, optimizer settings, learner tuning policy, evaluation budget, stopping conditions and rules for ties/failures.
3. Summarize the prespecified comparisons on California Housing and Diabetes Progression. Since both datasets have been examined before and the same panel informs the shortlist, these results are exploratory, not independent confirmation.
4. Primary selection metric: **dataset-level percentage reduction in MAE** versus that learner's baseline, aggregated with equal dataset weight; supplement with uncertainty, fraction of datasets helped, RMSE, tail residuals, actual selected \(k\), runtime and transformation complexity.
5. Require **a meaningful and stable improvement** (predeclare a minimum practical effect); in a practical tie choose fewer features, lower training/prediction overhead and better stability. If no combination consistently improves on no FE, explicitly recommend **no feature engineering** in the relevant setting.
6. If data do not support a single universal pipeline, report **conditional recommendations** (e.g. by learner family or dataset characteristics), with their limitations and verification requirements.

**Final output:** a reproducible comparison answering RQ1–RQ4 and a transparent, dataset-specific recommendation with the lack of independent confirmation stated. It is acceptable—and scientifically stronger—to conclude that no universal winner exists.

---

## 10. Minimal controlled experiment matrix

| Study | One primary variable | Hold fixed | Capacity handling | Interpretation |
|---|---|---|---|---|
| **RQ1** | None / S / C / S+C | GA, squared-projection representation, OLS, data folds | Inner-select \(k\) for C and S+C; secondary fixed-\(k\) comparison | Strategy effect on learner improvement |
| **RQ2-A** | Quadratic / RBF / explicit interactions | Construction-only, compatible GA, OLS, folds | Compare shared fixed \(k\) **and** inner-selected \(k^*\) | Parametric family/pipeline effect |
| **RQ2-B** | Symbolic GP vs fixed-form pipelines | Construction-only, OLS, folds, total budget | GP evolves \(k\) trees; same \(k\) grid where feasible | End-to-end pipeline comparison, not optimizer isolation |
| **RQ3** | GA / PSO / ES / DE / CMA-ES | Continuous representation, construction-only, OLS, folds | **Fixed common \(k\)** primary; equal-budget adaptive-\(k\) secondary | Optimizer/search mechanism effect |
| **RQ4** | OLS / Ridge / Random Forest | Small shortlist, matched folds and baseline | Re-select \(k\) per learner using inner data | Model specificity/generalizability |
| **Synthesis** | Shortlisted procedures | California Housing and Diabetes Progression | All \(k\)-selection remains within development folds | Exploratory, dataset-specific recommendation |

**Compute discipline:** begin with unit/smoke tests, a small pilot (e.g. 300 candidate evaluations, 3 optimizer seeds), then run the agreed full-study configuration (e.g. 1,000 evaluations, 5 optimizer seeds). For adaptive-\(k\), set a **total budget across values of \(k\)** before running. Never hide a fivefold increase in compute by calling each \(k\) run “equal-budget.”

---

## 11. Implementation phases and acceptance criteria

### Phase 0 — Cleanup and shared framework (start here)

- [ ] Put this document under `docs/research_plan.md` as the new specification.
- [ ] Extract useful existing implementations; remove obsolete results, comparison scripts and GP hybrids from the active refocused project.
- [ ] Create reusable `Strategy`, `Representation`, `Optimizer`, `LearnerFactory`, and `ExperimentRunner` interfaces.
- [ ] Make `k` a parameter of every construction decoder, GP feature generator, configuration and saved result. Remove any unconditional hardcoded two-feature assumption.
- [ ] Define an explicit `k` grid and total capacity-selection budget; create `k=0` no-construction behavior.
- [ ] Add tests for \(k=0,1,2,4\), output dimensions, selected original features, construction replay, budget counts across \(k\), leakage, fold-local preprocessing and model-specific tuning.
- [ ] Fix the two-dataset manifest, prospective splits and baseline policy before comparing methods; record that independent confirmation is unavailable.
- [ ] Update CI to execute only current tests and short smoke experiments.

**Exit criterion:** Running one configuration produces an unmodified baseline and all three FE strategy outputs, without reading old results and without revealing outer-test data to any selection routine.

### Phase 1 — RQ1: strategy

- [ ] GA-based no-FE, selection-only, construction-only and combined pipelines.
- [ ] Nested validation selects \(k^*\) only for construction arms under the same declared total resource budget.
- [ ] Produce paired improvements, chosen \(k\), selected-original counts, stability and runtime.
- [ ] Run fixed-\(k\) sensitivity as a controlled auxiliary result.

**Exit criterion:** Determine under which observed dataset conditions adding, dropping or combining variables helps the same regressor.

### Phase 2 — RQ2: representation

- [ ] Implement RBF and sparse explicit-interaction families alongside variable-\(k\) quadratic projections.
- [ ] Adapt GP to emit a set of \(k\) symbolic features; lock its grammar, depth and complexity limits.
- [ ] Compare matched fixed-\(k\) pipelines, then adaptive-\(k\) pipelines using inner-data selection.
- [ ] Record effective feature diversity and approximation/extrapolation diagnostics.

**Exit criterion:** Distinguish families that genuinely expand useful hypothesis classes from those adding redundant transformations.

### Phase 3 — RQ3: optimization

- [ ] Reuse GA, PSO, ES, DE and CMA-ES on one or two compatible continuous families.
- [ ] Run fixed-\(k\), equal-objective, equal-candidate-budget comparisons; optionally run equal-*total*-budget adaptive-\(k\) comparisons.
- [ ] Compare paired generalization gains, convergence and computational costs.

**Exit criterion:** Attribute method differences to the search mechanism under a shared feature task rather than to differing representations.

### Phase 4 — RQ4: model generality

- [ ] Implement OLS, tuned Ridge and tuned Random Forest model factories.
- [ ] Retrain/optimize candidate transformations separately for each learner, selecting \(k\) only in its inner validation.
- [ ] Compare each learner against its own no-FE baseline, including runtime and robustness.

**Exit criterion:** Determine whether and where feature engineering benefits different regression learning algorithms.

### Phase 5 — Two-dataset synthesis

- [ ] Lock candidate shortlist and all tuning/capacity-selection rules.
- [ ] Summarize the prespecified comparisons on California Housing and Diabetes Progression without calling them independent confirmation.
- [ ] Report selected \(k\) distributions, paired relative MAE benefits, uncertainty, failure rates and compute.
- [ ] Recommend a pipeline only where supported, including “none” when appropriate.

**Exit criterion:** The two-dataset conclusions are reproducible without old result artefacts and state their exploratory scope.

---

## 12. Threats to validity worth documenting

- **Capacity bias:** families differ in expressivity even at equal \(k\). Nominal feature-count matching is not complete model-complexity matching.
- **Adaptive search-cost bias:** varying \(k\) introduces another hyperparameter-selection problem; compare both total compute and performance.
- **GP representation/search confounding:** GP and parametric GA differ in both feature language and operators; GP comparisons belong at pipeline level.
- **Outer-test reuse:** old observations from both datasets have influenced discussion; these data cannot support an independent confirmation claim.
- **Nested hyperparameters:** feature selection, \(k\), optimizer settings and learner tuning must not all be optimized against the final test.
- **Small datasets and large \(k\):** avoid producing too many constructed features for the sample size; define caps before experiments.
- **Numerical extrapolation:** quadratic, RBF and GP constructions may behave badly out of distribution; monitor error tails without post-hoc test-driven fixes.
- **No free lunch across learners:** a useful feature for OLS may be redundant for a nonlinear model.
- **Fitting-cost differences:** equal feature-candidate evaluations do not imply equal fitted-model counts or runtime.
- **Practical interpretability:** GP node count, sparse-interaction count and quadratic weight density are different complexity notions; report them explicitly.

---

## 13. Locked implementation decisions before the pilot

- **Combined strategy:** The mask controls which originals enter the learner directly. Constructed features may use all originals, including masked ones. Record direct originals, construction inputs and their union separately; the mask alone does not describe total input use. For quadratic features, an input counts as used when a coefficient has absolute value greater than `1e-12`. This is algebraic use, not an estimate of importance or effective sparsity.
- **Capacity search:** Each adaptive arm divides its total candidate allowance across eligible searched `k` values. Construction-only `k=0` uses the same learner's pre-tuned no-FE inner-fold score without optimizer calls. Combined `k=0` searches a selection mask and consumes its allocated evaluations. Save every allocation and validation score. Fixed-`k` arms use the full allowance at one dimension, so fixed-versus-adaptive differences reflect search allocation as well as capacity.
- **Refit:** Candidate genes or GP expressions are shared across inner folds, while preprocessing is fitted within each fold. The selected candidate is replayed after preprocessing is refitted on all development rows. Coefficients are defined in standardized coordinates and therefore acquire a different raw-input interpretation after the refit. Saved source, split and feature definitions must reproduce final transformed arrays and predictions exactly.
- **RQ2 order:** `configs/rq2.yaml` compares four adaptive pipelines first; `configs/rq2_fixed.yaml` is a secondary matched-`k` capacity study. GP remains a pipeline comparison because its representation and search both differ.
- **RQ4 scope:** Ridge and Random Forest parameters are pre-tuned on original development features and frozen during FE search. This tests FE with a pre-tuned fixed learner, not joint learner-and-FE optimization.
- **Pilot:** The earlier five-dataset pilot used one outer split, three optimizer seeds and 300 candidate evaluations per arm. Its health record is historical; the current manifest contains only California Housing and Diabetes Progression. Pilot predictive scores must not choose methods or change full-study configurations.
