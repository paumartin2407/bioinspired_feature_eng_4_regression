"""Run the Milestone 1 comparison: python benchmark.py --output results/milestone1."""
import argparse
import json
import random
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

from data import DATASETS, load_dataset, split_and_preprocess
from evaluation import EvaluationCounter, regression_metrics
from features import add_gp_feature, candidate_frame
from optimizers import EvolutionStrategy, GeneticAlgorithm, GeneticProgrammingFeatureEngineer, PSO


OUTER_SEEDS = [11, 23, 42, 67, 89]
OPTIMIZER_SEEDS = [101, 211, 307, 401, 503]
BUDGET = 1000
GA_CONFIG = {"pop_size": 24, "n_gens": 20, "r_synth": 2, "verbose": False}
PSO_CONFIG = {"n_particles": 24, "n_iters": 20, "r_synth": 2, "verbose": False}
ES_CONFIG = {"mu": 12, "lambda_": 24, "n_gens": 20, "r_synth": 2, "verbose": False}
GP_CONFIG = {"pop_size": 40, "n_gens": 24, "max_depth": 4}
HYBRID_GP_CONFIG = {**GP_CONFIG, "n_gens": 9}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def tree_definition(node):
    if hasattr(node, "idx"):
        return {"feature": int(node.idx), "name": node.name}
    if hasattr(node, "val"):
        return {"constant": float(node.val)}
    if hasattr(node, "left"):
        return {"operator": node.symbol, "left": tree_definition(node.left), "right": tree_definition(node.right)}
    return {"operator": node.symbol, "child": tree_definition(node.child)}


def es_normalization(individual, train):
    stats = []
    for coefficients, squared in zip(individual["coefs"], individual["type_mask"]):
        values = train.to_numpy() @ coefficients
        if squared:
            values = values ** 2
        stats.append({"mean": float(values.mean()), "std": float(values.std() + 1e-6)})
    return stats


def tuned_baseline(name, train, val, test, y_train, y_val):
    if name == "Linear Regression":
        model = LinearRegression().fit(train, y_train)
        return model, train, val, test, {}
    if name == "Polynomial + Ridge":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        train, val, test = poly.fit_transform(train), poly.transform(val), poly.transform(test)
    else:
        train, val, test = train.to_numpy(), val.to_numpy(), test.to_numpy()
    grid = [{"alpha": alpha} for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if name == "Elastic Net":
        grid = [{**params, "l1_ratio": ratio} for params in grid for ratio in (0.2, 0.5, 0.8)]
    model_class = {"Ridge": Ridge, "Lasso": Lasso, "Elastic Net": ElasticNet,
                   "Polynomial + Ridge": Ridge}[name]
    best = min(grid, key=lambda params: mean_absolute_error(
        y_val, model_class(**params, max_iter=10000).fit(train, y_train).predict(val))
        if name in ("Lasso", "Elastic Net") else mean_absolute_error(
            y_val, model_class(**params).fit(train, y_train).predict(val)))
    kwargs = {**best, **({"max_iter": 10000} if name in ("Lasso", "Elastic Net") else {})}
    return model_class(**kwargs).fit(train, y_train), train, val, test, best


def instrument(optimizer, counter, method):
    original = getattr(optimizer, method)
    setattr(optimizer, method, lambda candidate: counter.evaluate(original, candidate))


def optimize(name, train, val, y_train, y_val, seed, counter, mode="combined", hybrid=False):
    np.random.seed(seed)
    random.seed(seed)
    if name == "GA":
        config = {**GA_CONFIG, "n_gens": 24} if hybrid else GA_CONFIG
        optimizer = GeneticAlgorithm(train, val, y_train, y_val, seed=seed, mode=mode, **config)
        instrument(optimizer, counter, "fitness")
        frame = optimizer.individual_to_dataframe
    elif name == "PSO":
        config = {**PSO_CONFIG, "n_iters": 24} if hybrid else PSO_CONFIG
        optimizer = PSO(train, val, y_train, y_val, seed=seed, mode=mode, **config)
        instrument(optimizer, counter, "fitness_particle")
        frame = optimizer.particle_df
    else:
        config = {**ES_CONFIG, "mu": 10, "lambda_": 20} if hybrid else ES_CONFIG
        optimizer = EvolutionStrategy(train, val, y_train, y_val, seed=seed, mode=mode, **config)
        instrument(optimizer, counter, "fitness")
        frame = optimizer.individual_to_df
    result = optimizer.run()
    individual = result["best_individual"]
    if name == "PSO":
        # The PSO result stores a decoded dictionary; its vector is the actual search state.
        vector = np.asarray(result["best_expr"])
        return individual, lambda X: frame(vector, X), plain(individual)
    if name == "ES":
        return individual, lambda X: candidate_frame(individual, X, "ES", reference=train, mode=mode), plain(individual)
    return individual, lambda X: frame(individual, X), plain(individual)


def run_split(dataset, X, y, split_number, outer_seed, optimizer_seeds, limit=BUDGET):
    train, val, test, y_train, y_val, y_test, indices = split_and_preprocess(
        X, y, outer_seed, return_indices=True)
    rows, traces, definitions, signatures = [], [], [], []

    def record(method, seed, started, train_features, val_features, test_features,
               counter=None, definition=None, selected=None):
        model = LinearRegression().fit(train_features, y_train) if counter else None
        if model is None:
            raise ValueError("Baseline model must be recorded separately")
        row = {"dataset": dataset, "split": split_number, "outer_seed": outer_seed,
               "optimizer_seed": seed, "method": method, **regression_metrics(y_test, model.predict(test_features)),
               "validation_MAE": float(mean_absolute_error(y_val, model.predict(val_features))),
               "runtime_seconds": time.perf_counter() - started,
               "feature_count": int(train_features.shape[1]), "fitness_evaluations": counter.count}
        rows.append(row)
        definitions.append({"dataset": dataset, "split": split_number, "optimizer_seed": seed,
                            "method": method, "input_columns": list(train.columns), "definition": definition})
        traces.extend({"dataset": dataset, "split": split_number, "optimizer_seed": seed,
                       "method": method, "evaluations": count, "validation_MAE": score}
                      for count, score in counter.history)
        synthetic = [c for c in val_features.columns if str(c).startswith(("synth_", "GP_feature"))]
        signatures.append({"dataset": dataset, "split": split_number, "method": method,
                           "seed": seed, "selected": selected or [],
                           "constructed": val_features[synthetic].to_numpy().T.tolist()})

    for name in ("Linear Regression", "Ridge", "Lasso", "Elastic Net", "Polynomial + Ridge"):
        started = time.perf_counter()
        model, train_b, val_b, test_b, params = tuned_baseline(name, train, val, test, y_train, y_val)
        rows.append({"dataset": dataset, "split": split_number, "outer_seed": outer_seed,
                     "optimizer_seed": None, "method": name,
                     **regression_metrics(y_test, model.predict(test_b)),
                     "validation_MAE": float(mean_absolute_error(y_val, model.predict(val_b))),
                     "runtime_seconds": time.perf_counter() - started,
                     "feature_count": int(train_b.shape[1]), "fitness_evaluations": 0})
        definitions.append({"dataset": dataset, "split": split_number, "method": name,
                            "input_columns": list(train.columns), "definition": params})

    for seed in optimizer_seeds:
        for name in ("GA", "PSO", "ES"):
            for mode in ("selection", "construction", "combined"):
                started, counter = time.perf_counter(), EvaluationCounter(limit)
                individual, apply, definition = optimize(name, train, val, y_train, y_val, seed, counter, mode)
                train_f, val_f, test_f = (apply(part) for part in (train, val, test))
                selected = [c for c, keep in zip(train.columns, individual["mask"]) if keep]
                normalization = es_normalization(individual, train) if name == "ES" and mode != "selection" else []
                record(f"{name} {mode}", seed, started, train_f, val_f, test_f, counter,
                       {"kind": name, "mode": mode, "individual": definition,
                        "normalization": normalization}, selected)
        for second in (None, "GA", "PSO", "ES"):
            started, counter = time.perf_counter(), EvaluationCounter(limit)
            gp_cfg = GP_CONFIG if second is None else HYBRID_GP_CONFIG
            gp = GeneticProgrammingFeatureEngineer(train, val, y_train, y_val, seed=seed, **gp_cfg)
            instrument(gp, counter, "_evaluate_individual")
            tree = gp.run()["best_individual"]
            gp_values = tree.evaluate(train.to_numpy())
            gp_definition = {"tree": tree_definition(tree), "mean": float(gp_values.mean()),
                             "std": float(gp_values.std())}
            gp_train, gp_val, gp_test = add_gp_feature(tree, train, val, test)
            if second is None:
                record("GP", seed, started, gp_train, gp_val, gp_test, counter,
                       {"kind": "GP", **gp_definition}, list(train.columns))
            else:
                individual, apply, definition = optimize(second, gp_train, gp_val, y_train, y_val,
                                                         seed, counter, hybrid=True)
                train_f, val_f, test_f = (apply(part) for part in (gp_train, gp_val, gp_test))
                selected = [c for c, keep in zip(gp_train.columns, individual["mask"]) if keep]
                record(f"GP + {second}", seed, started, train_f, val_f, test_f, counter,
                       {"kind": f"GP + {second}", "gp_feature": gp_definition,
                        "individual": definition,
                        "normalization": es_normalization(individual, gp_train) if second == "ES" else []}, selected)
    return rows, traces, definitions, signatures, indices


def stability(signatures):
    rows = []
    grouped = {}
    for signature in signatures:
        key = signature["dataset"], signature["split"], signature["method"]
        grouped.setdefault(key, []).append(signature)
    for key, group in grouped.items():
        for a, b in combinations(group, 2):
            sa, sb = set(a["selected"]), set(b["selected"])
            jaccard = len(sa & sb) / len(sa | sb) if sa | sb else np.nan
            correlations = []
            for x in a["constructed"]:
                for y in b["constructed"]:
                    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                        correlations.append(abs(np.corrcoef(x, y)[0, 1]))
            rows.append({"dataset": key[0], "split": key[1], "method": key[2],
                         "seed_a": a["seed"], "seed_b": b["seed"],
                         "selection_jaccard": jaccard,
                         "constructed_similarity": max(correlations) if correlations else np.nan})
    return pd.DataFrame(rows)


def save_analysis(rows, traces, definitions, signatures, splits, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "raw_results.csv", index=False)
    pd.DataFrame(traces).to_csv(output / "convergence.csv", index=False)
    with (output / "feature_definitions.jsonl").open("w", encoding="utf-8") as handle:
        for item in definitions:
            handle.write(json.dumps(plain(item)) + "\n")
    (output / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    summary = raw.groupby(["dataset", "method"])[["MAE", "RMSE", "R2", "runtime_seconds", "feature_count", "fitness_evaluations"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "performance.csv", index=False)
    # Average optimizer seeds within each split before pairing methods.
    by_split = raw.groupby(["dataset", "split", "method"])[["MAE", "RMSE", "R2"]].mean().reset_index()
    paired = []
    for dataset, frame in by_split.groupby("dataset"):
        methods = {name: values.set_index("split") for name, values in frame.groupby("method")}
        for first, second in combinations(methods, 2):
            aligned = methods[first].join(methods[second], lsuffix="_first", rsuffix="_second")
            differences = {metric: aligned[f"{metric}_first"] - aligned[f"{metric}_second"]
                           for metric in ("MAE", "RMSE", "R2")}
            paired.append({"dataset": dataset, "first": first, "second": second,
                           "n_splits": len(aligned),
                           **{f"delta_{metric}_{stat}": float(getattr(diff, stat)())
                              for metric, diff in differences.items() for stat in ("mean", "std")}})
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    ablation = raw[raw.method.str.match(r"^(GA|PSO|ES) (selection|construction|combined)$")].groupby(
        ["dataset", "method"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    ablation.columns = [f"{metric}_{stat}" for metric, stat in ablation.columns]
    ablation.reset_index().to_csv(output / "ablation.csv", index=False)
    stability(signatures).to_csv(output / "stability.csv", index=False)
    traces_frame = pd.DataFrame(traces)
    if not traces_frame.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for axis, (dataset, group) in zip(axes, traces_frame.groupby("dataset")):
            for method, values in group.groupby("method"):
                curve = values.groupby("evaluations").validation_MAE.mean()
                axis.plot(curve.index, curve.values, label=method)
            axis.set(title=dataset, xlabel="Fitness evaluations", ylabel="Best validation MAE")
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(output / "convergence.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/milestone1"))
    parser.add_argument("--smoke", action="store_true", help="One split, one optimizer seed, first 200 rows")
    args = parser.parse_args()
    config = {"outer_seeds": OUTER_SEEDS, "optimizer_seeds": OPTIMIZER_SEEDS,
              "budget": BUDGET, "outer_test_fraction": 0.2, "inner_validation_fraction": 0.25,
              "datasets": DATASETS, "smoke": args.smoke,
              "optimizer_config": {"GA": GA_CONFIG, "PSO": PSO_CONFIG, "ES": ES_CONFIG,
                                   "GP": GP_CONFIG, "hybrid_GP": HYBRID_GP_CONFIG,
                                   "hybrid_GA_generations": 24, "hybrid_PSO_iterations": 24,
                                   "hybrid_ES": {"mu": 10, "lambda": 20, "n_gens": 20}},
              "baseline_alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
              "elastic_net_l1_ratio_grid": [0.2, 0.5, 0.8]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, traces, definitions, signatures, splits = [], [], [], [], {}
    for dataset, (path, target) in DATASETS.items():
        X, y = load_dataset(path, target)
        if args.smoke:
            X, y = X.iloc[:200], y.iloc[:200]
        splits[dataset] = []
        for number, seed in enumerate(OUTER_SEEDS[:1] if args.smoke else OUTER_SEEDS):
            print(f"{dataset}, split {number + 1}", flush=True)
            result = run_split(dataset, X, y, number, seed,
                               OPTIMIZER_SEEDS[:1] if args.smoke else OPTIMIZER_SEEDS)
            for target_list, values in zip((rows, traces, definitions, signatures), result[:4]):
                target_list.extend(values)
            splits[dataset].append({"outer_seed": seed, **result[4]})
            save_analysis(rows, traces, definitions, signatures, splits, args.output)
    print(f"Saved {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
