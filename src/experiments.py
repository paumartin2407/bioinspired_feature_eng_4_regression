"""Run one prespecified research protocol. No result file is used as input."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

from .data import fit_preprocessor, load_dataset, make_outer_split, prepared_folds, transformed
from .evaluation import candidate_fitness, choose_learner, learner, metrics
from .gp_features import feature_pair, node_count, optimize_gp
from .optimizers import METHODS, optimize
from .strategies import decode, gene_count, transform


ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    # JSON is a YAML subset; these .yaml files need no additional parser dependency.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def allocated_budgets(ks, budget, strategy):
    active = [k for k in ks if strategy == "selection" or k > 0]
    if not active:
        return {}
    if budget < 40 * len(active):
        raise ValueError("Total budget must supply at least 40 evaluations per searched k")
    return {k: budget // len(active) + (index < budget % len(active))
            for index, k in enumerate(active)}


def eligible_ks(arm, n_dev, d):
    if arm["strategy"] in ("none", "selection"):
        return [0]
    cap = min(d, max(1, n_dev // 20), arm.get("max_k", 8))
    ks = sorted(set(int(k) for k in arm["k_grid"] if 0 <= k <= cap))
    if not ks:
        raise ValueError(f"No eligible k for {arm['name']}")
    return ks


def _apply_pair(train, other, arm, candidate, k):
    if arm["family"] == "gp":
        return feature_pair(train, other, candidate)
    return (transform(train, arm["strategy"], arm["family"], candidate, k),
            transform(other, arm["strategy"], arm["family"], candidate, k))


def matched_stability(first, second):
    a, b = first["constructed"], second["constructed"]
    selected_a, selected_b = first["selected"], second["selected"]
    union = selected_a | selected_b
    jaccard = len(selected_a & selected_b) / len(union) if union else 1.0
    if a.shape[1] == b.shape[1] == 0:
        return jaccard, 1.0
    if a.shape[1] == 0 or b.shape[1] == 0:
        return jaccard, 0.0
    similarity = np.zeros((a.shape[1], b.shape[1]))
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            if np.std(a[:, i]) > 1e-9 and np.std(b[:, j]) > 1e-9:
                correlation = np.corrcoef(a[:, i], b[:, j])[0, 1]
                similarity[i, j] = abs(correlation) if np.isfinite(correlation) else 0.0
    row, col = linear_sum_assignment(-similarity)
    return jaccard, float(similarity[row, col].sum() / max(a.shape[1], b.shape[1]))


def run_arm(arm, folds, dev, test, y_dev, y_test, columns, model_name, parameter,
            seed, baseline_inner, baseline_test_mae, smoke=False):
    start = time.perf_counter()
    strategy, family = arm["strategy"], arm.get("family", "quadratic")
    ks = eligible_ks(arm, len(dev), len(columns))
    if smoke and len(ks) > 1:
        ks = [1] if strategy == "combined" and 1 in ks else ks[:2]
    budget = 40 if smoke else arm.get("budget", 1000)
    allocation = allocated_budgets(ks, budget, strategy)
    scores, candidates, traces = {}, {}, []
    counts = {"fits": 0, "failed_folds": 0, "failed_candidates": 0}
    evaluations = 0
    for k in ks:
        if strategy == "construction" and k == 0:
            scores[k], candidates[k] = baseline_inner, []
            continue
        objective = candidate_fitness(folds, strategy, family, k, model_name, parameter, seed, counts,
                                      feature_pair if family == "gp" else None)
        if family == "gp":
            candidate, score, spent, history = optimize_gp(objective, len(columns), k, seed + 1009 * k,
                                                            allocation[k])
        else:
            result = optimize(arm["optimizer"], objective,
                              gene_count(strategy, family, len(columns), k), seed + 1009 * k,
                              allocation[k])
            candidate, score, spent, history = result.genes, result.score, result.evaluations, result.history
        offset = evaluations
        evaluations += spent
        scores[k], candidates[k] = score, candidate
        previous_best = min((value for other_k, value in scores.items() if other_k != k), default=float("inf"))
        traces.extend({"k": k, "evaluation": count, "best_inner_mae": best,
                       "global_evaluation": offset + count,
                       "global_best_inner_mae": min(previous_best, best)} for count, best in history)
    best_score = min(scores.values())
    tolerance = arm.get("k_tolerance", 1e-4)
    selected_k = min(k for k in ks if scores[k] <= best_score + tolerance)
    candidate = candidates[selected_k]
    fallback = not np.isfinite(scores[selected_k])
    if fallback:
        selected_k, candidate = 0, []
    preprocessor = fit_preprocessor(dev)
    train = transformed(preprocessor, dev, columns).to_numpy()
    held = transformed(preprocessor, test, columns).to_numpy()
    if fallback:
        train_features, test_features = train, held
    else:
        train_features, test_features = _apply_pair(train, held, arm, candidate, selected_k)
    outer_failure = False
    try:
        if not np.isfinite(train_features).all() or not np.isfinite(test_features).all():
            raise ValueError("Nonfinite final features")
        model = learner(model_name, parameter, seed)
        counts["fits"] += 1
        model.fit(train_features, y_dev)
        prediction = model.predict(test_features)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite final prediction")
        report = metrics(y_test, prediction)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        outer_failure = True
        report = {name: float("nan") for name in
                  ("mae", "rmse", "r2", "p95_abs_residual", "p99_abs_residual", "max_abs_residual")}
    report.update({"arm": arm["name"], "strategy": strategy, "family": family,
                   "optimizer": arm.get("optimizer", "GP" if family == "gp" else "none"),
                   "learner": model_name, "learner_parameter": parameter,
                   "selected_k": selected_k, "selected_originals": len(columns) if fallback or
                   strategy in ("none", "construction") else int(decode(strategy, family, candidate, len(columns), selected_k)[0].sum()),
                   "transformed_dim": train_features.shape[1], "candidate_evaluations": evaluations,
                   "fe_model_fits": counts["fits"], "failed_folds": counts["failed_folds"],
                   "failed_candidates": counts["failed_candidates"], "outer_failure": outer_failure,
                   "runtime_seconds": time.perf_counter() - start, "fallback": fallback,
                   "delta_mae": report["mae"] - baseline_test_mae,
                   "mae_reduction_pct": 100 * (baseline_test_mae - report["mae"]) / baseline_test_mae
                   if baseline_test_mae else float("nan")})
    definition = {"arm": arm["name"], "k": selected_k, "columns": columns,
                  "candidate": _jsonable(candidate), "inner_scores": {str(k): scores[k] for k in ks},
                  "gp_nodes": sum(node_count(tree) for tree in candidate) if family == "gp" and not fallback else 0}
    report["gp_nodes"] = definition["gp_nodes"]
    selected = set(columns) if fallback or strategy in ("none", "construction") else set(
        name for name, keep in zip(columns, decode(strategy, family, candidate, len(columns), selected_k)[0]) if keep)
    constructed = train_features[:, len(selected):]
    finite_construction = np.isfinite(constructed).all()
    report["constant_constructed"] = int(sum(np.std(constructed[:, j]) < 1e-9 for j in range(constructed.shape[1]))) if finite_construction else float("nan")
    report["constructed_rank"] = int(np.linalg.matrix_rank(constructed)) if constructed.size and finite_construction else 0
    report["max_abs_constructed"] = float(np.max(np.abs(constructed))) if constructed.size and finite_construction else float("nan")
    signature = {"selected": selected, "constructed": constructed}
    return report, definition, traces, signature


def polynomial_reference(folds, dev_matrix, test_matrix, y_dev, y_test, seed):
    start = time.perf_counter()
    from sklearn.linear_model import Ridge
    scores = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        fold_scores = []
        for train, val, yt, yv, *_ in folds:
            polynomial = PolynomialFeatures(degree=2, include_bias=False)
            built = polynomial.fit_transform(train)
            model = Ridge(alpha=alpha).fit(built, yt)
            fold_scores.append(mean_absolute_error(yv, model.predict(polynomial.transform(val))))
        scores.append(float(np.mean(fold_scores)))
    alpha = (0.1, 1.0, 10.0, 100.0)[int(np.argmin(scores))]
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    model = Ridge(alpha=alpha).fit(polynomial.fit_transform(dev_matrix), y_dev)
    report = metrics(y_test, model.predict(polynomial.transform(test_matrix)))
    report.update({"arm": "polynomial_ridge_reference", "strategy": "deterministic_reference",
                   "family": "polynomial", "optimizer": "none", "learner": "ridge",
                   "learner_parameter": alpha, "selected_k": polynomial.n_output_features_ - dev_matrix.shape[1],
                   "selected_originals": dev_matrix.shape[1], "transformed_dim": polynomial.n_output_features_,
                   "candidate_evaluations": 0, "fe_model_fits": 0, "tuning_model_fits": 12,
                   "baseline_model_fits": 1, "failed_folds": 0, "failed_candidates": 0,
                   "outer_failure": False,
                   "runtime_seconds": time.perf_counter() - start, "fallback": False})
    return report


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def run(config, datasets, output, smoke=False, data_root=ROOT):
    if config["study"] == "final" and not config.get("locked", False):
        raise ValueError("Lock the final shortlist and full protocol before running final evaluation")
    if config["study"] == "final" and (not 1 <= len(config["arms"]) <= 3 or not config["learners"]):
        raise ValueError("Final protocol requires one to three FE arms and at least one learner")
    if not config["arms"]:
        raise ValueError("Protocol needs at least one FE arm")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results in {output}")
    selected = [item for item in datasets if item["stage"] == ("final" if config["study"] == "final" else "exploration")]
    if smoke:
        selected = selected[:1]
    if not selected:
        raise ValueError("No datasets for this study stage")
    rows, definitions, traces, splits, stability_rows, source_hashes = [], [], [], [], [], {}
    for dataset in selected:
        source_path = Path(data_root) / dataset["path"]
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        checksum_path = source_path.with_suffix(".sha256")
        if checksum_path.exists() and checksum_path.read_text(encoding="ascii").strip() != digest:
            raise ValueError(f"Source checksum mismatch: {source_path}")
        source_hashes[dataset["name"]] = digest
        X, y = load_dataset(dataset, data_root)
        if smoke:
            X, y = X.iloc[:min(120, len(X))], y[:min(120, len(y))]
        for split_number, outer_seed in enumerate(config["outer_seeds"][:1] if smoke else config["outer_seeds"]):
            dev_idx, test_idx = make_outer_split(len(X), outer_seed)
            dev, test = X.iloc[dev_idx], X.iloc[test_idx]
            y_dev, y_test = y[dev_idx], y[test_idx]
            folds, columns = prepared_folds(dev, y_dev, outer_seed, config.get("inner_folds", 3))
            splits.append({"dataset": dataset["name"], "split": split_number,
                           "development": dev_idx.tolist(), "test": test_idx.tolist(),
                           "inner": [{"train": dev_idx[ti].tolist(), "validation": dev_idx[vi].tolist()}
                                     for *_, ti, vi in folds]})
            preprocessor = fit_preprocessor(dev)
            dev_matrix = transformed(preprocessor, dev, columns).to_numpy()
            test_matrix = transformed(preprocessor, test, columns).to_numpy()
            if "polynomial_ridge" in config.get("references", []):
                reference = polynomial_reference(folds, dev_matrix, test_matrix, y_dev, y_test, outer_seed)
                reference.update({"dataset": dataset["name"], "split": split_number,
                                  "outer_seed": outer_seed, "search_seed": None})
                rows.append(reference)
            for model_name in config["learners"]:
                parameter, tune_fits, tune_scores = choose_learner(model_name, folds, outer_seed)
                baseline_inner = min(tune_scores)
                baseline_start = time.perf_counter()
                baseline_model = learner(model_name, parameter, outer_seed).fit(dev_matrix, y_dev)
                baseline_report = metrics(y_test, baseline_model.predict(test_matrix))
                baseline_report.update({"dataset": dataset["name"], "split": split_number,
                                        "outer_seed": outer_seed, "search_seed": None, "arm": "no_fe",
                                        "strategy": "none", "family": "none", "optimizer": "none",
                                        "learner": model_name, "learner_parameter": parameter,
                                        "selected_k": 0, "selected_originals": len(columns),
                                        "transformed_dim": len(columns), "candidate_evaluations": 0,
                                        "fe_model_fits": 0, "tuning_model_fits": tune_fits,
                                        "baseline_model_fits": 1, "failed_folds": 0,
                                        "failed_candidates": 0, "outer_failure": False,
                                        "runtime_seconds": time.perf_counter() - baseline_start,
                                        "delta_mae": 0.0, "mae_reduction_pct": 0.0,
                                        "fallback": False})
                rows.append(baseline_report)
                signatures = {}
                for search_seed in config["search_seeds"][:1] if smoke else config["search_seeds"]:
                    for arm in config["arms"]:
                        report, definition, history, signature = run_arm(
                            arm, folds, dev, test, y_dev, y_test, columns, model_name, parameter,
                            search_seed, baseline_inner, baseline_report["mae"], smoke)
                        key = {"dataset": dataset["name"], "split": split_number,
                               "outer_seed": outer_seed, "search_seed": search_seed}
                        report.update(key)
                        report["tuning_model_fits"] = tune_fits
                        report["baseline_model_fits"] = 1
                        rows.append(report)
                        definitions.append({**key, "learner": model_name, **definition})
                        signatures.setdefault(arm["name"], []).append((search_seed, signature))
                        traces.extend({**key, "learner": model_name, "arm": arm["name"], **point}
                                      for point in history)
                for arm_name, items in signatures.items():
                    for left in range(len(items)):
                        for right in range(left + 1, len(items)):
                            selection_score, construction_score = matched_stability(items[left][1], items[right][1])
                            stability_rows.append({"dataset": dataset["name"], "split": split_number,
                                                   "learner": model_name, "arm": arm_name,
                                                   "seed_a": items[left][0], "seed_b": items[right][0],
                                                   "selection_jaccard": selection_score,
                                                   "constructed_similarity": construction_score})
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps({"config": config, "datasets": selected,
        "smoke": smoke, "smoke_row_cap": 120 if smoke else None,
        "source_sha256": source_hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__}}, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (output / "definitions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in definitions), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
    pd.DataFrame(traces, columns=["dataset", "split", "outer_seed", "search_seed", "learner", "arm",
                                 "k", "evaluation", "best_inner_mae", "global_evaluation",
                                 "global_best_inner_mae"]).to_csv(output / "convergence.csv", index=False)
    pd.DataFrame(stability_rows, columns=["dataset", "split", "learner", "arm", "seed_a", "seed_b",
                                          "selection_jaccard", "constructed_similarity"]).to_csv(output / "stability.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--datasets", type=Path, default=ROOT / "configs/datasets.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="One small split, one seed and 40 evaluations per arm")
    args = parser.parse_args()
    rows = run(read_config(args.config), read_config(args.datasets)["datasets"], args.output, args.smoke)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
