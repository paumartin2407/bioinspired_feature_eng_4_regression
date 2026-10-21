"""Run one prespecified research protocol. No result file is used as input."""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
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
    active = [k for k in ks if strategy in ("selection", "combined") or k > 0]
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


def construction_inputs(family, candidate, d, k):
    if k == 0:
        return set()
    if family == "quadratic":
        return set(np.flatnonzero(np.any(np.abs(np.asarray(candidate).reshape(k, d)) > 1e-12, axis=0)))
    if family == "rbf":
        return set(range(d))  # Every input contributes to each squared distance.
    if family == "interactions":
        pairs = np.minimum(((np.asarray(candidate).reshape(k, 2) + 1) * d / 2).astype(int), d - 1)
        return set(pairs.ravel())
    if family == "gp":
        def leaves(tree):
            if tree[0] == "x":
                return {int(tree[1])}
            if tree[0] == "c":
                return set()
            return leaves(tree[1]) | leaves(tree[2])
        return set().union(*(leaves(tree) for tree in candidate))
    raise ValueError(f"Unknown feature family: {family}")


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


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
            seed, baseline_inner, baseline_test_mae, smoke=False, resume=None,
            checkpoint=None, progress=None, forest_jobs=1):
    start = time.perf_counter()
    strategy, family = arm["strategy"], arm.get("family", "quadratic")
    ks = eligible_ks(arm, len(dev), len(columns))
    if smoke and len(ks) > 1:
        ks = [1] if strategy == "combined" and 1 in ks else ks[:2]
    budget = 40 if smoke else arm.get("budget", 1000)
    allocation = allocated_budgets(ks, budget, strategy)
    scores = {int(k): v for k, v in resume["scores"].items()} if resume else {}
    candidates = {int(k): v for k, v in resume["candidates"].items()} if resume else {}
    traces = resume["traces"] if resume else []
    counts = resume["counts"] if resume else {"fits": 0, "failed_folds": 0, "failed_candidates": 0}
    evaluations = resume["evaluations"] if resume else 0
    prior_seconds = resume["elapsed_seconds"] if resume else 0.0
    for k in ks:
        if k in scores:
            continue
        if strategy == "construction" and k == 0:
            scores[k], candidates[k] = baseline_inner, []
            continue
        objective = candidate_fitness(folds, strategy, family, k, model_name, parameter, seed, counts,
                                      feature_pair if family == "gp" else None, forest_jobs)
        if progress is not None:
            original_objective = objective
            current_count = 0
            def objective(genes):
                nonlocal current_count
                score = original_objective(genes)
                current_count += 1
                if current_count % 25 == 0:
                    progress(evaluations + current_count, k)
                return score
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
        if checkpoint is not None:
            checkpoint({"kind": "partial", "scores": {str(key): value for key, value in scores.items()},
                        "candidates": {str(key): _jsonable(value) for key, value in candidates.items()},
                        "traces": traces, "counts": counts, "evaluations": evaluations,
                        "elapsed_seconds": prior_seconds + time.perf_counter() - start})
        if progress is not None:
            progress(evaluations, k)
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
                   "runtime_seconds": prior_seconds + time.perf_counter() - start, "fallback": fallback,
                   "delta_mae": report["mae"] - baseline_test_mae,
                   "mae_reduction_pct": 100 * (baseline_test_mae - report["mae"]) / baseline_test_mae
                   if baseline_test_mae else float("nan")})
    direct_mask = np.ones(len(columns), dtype=bool) if fallback or strategy == "construction" else decode(
        strategy, family, candidate, len(columns), selected_k)[0]
    direct_names = [name for name, keep in zip(columns, direct_mask) if keep]
    construction_genes = candidate if family == "gp" else decode(
        strategy, family, candidate, len(columns), selected_k)[1]
    constructed_indices = set() if fallback else construction_inputs(family, construction_genes, len(columns), selected_k)
    constructed_names = [columns[index] for index in sorted(constructed_indices)]
    report["construction_input_count"] = len(constructed_names)
    report["effective_original_count"] = len(set(direct_names) | set(constructed_names))
    definition = {"arm": arm["name"], "k": selected_k, "columns": columns,
                  "candidate": _jsonable(candidate), "inner_scores": {str(k): scores[k] for k in ks},
                  "evaluation_allocation": {str(k): allocation.get(k, 0) for k in ks},
                  "learner_parameter": parameter, "fallback": fallback,
                  "direct_originals": direct_names,
                  "construction_inputs": constructed_names,
                  "dev_features_sha256": array_digest(train_features) if np.isfinite(train_features).all() else None,
                  "test_features_sha256": array_digest(test_features) if np.isfinite(test_features).all() else None,
                  "prediction_sha256": array_digest(prediction) if not outer_failure else None,
                  "gp_nodes": sum(node_count(tree) for tree in candidate) if family == "gp" and not fallback else 0}
    report["gp_nodes"] = definition["gp_nodes"]
    selected = set(direct_names)
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


def _atomic_text(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _save_tables(output, rows, definitions, traces, splits, stability_rows):
    _atomic_text(output / "splits.json", json.dumps(splits, indent=2))
    _atomic_text(output / "definitions.jsonl", "".join(json.dumps(item) + "\n" for item in definitions))
    for name, frame in (
        ("runs.csv", pd.DataFrame(rows)),
        ("convergence.csv", pd.DataFrame(traces, columns=["dataset", "split", "outer_seed",
            "search_seed", "learner", "arm", "k", "evaluation", "best_inner_mae",
            "global_evaluation", "global_best_inner_mae"])),
        ("stability.csv", pd.DataFrame(stability_rows, columns=["dataset", "split", "learner",
            "arm", "seed_a", "seed_b", "selection_jaccard", "constructed_similarity"])),
    ):
        temporary = output / (name + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output / name)


def _saved_signature(definition, arm, dev_matrix):
    candidate = definition["candidate"]
    selected = set(definition["direct_originals"])
    if definition["fallback"]:
        constructed = dev_matrix[:, :0]
    else:
        features, _ = _apply_pair(dev_matrix, dev_matrix, arm, candidate, definition["k"])
        constructed = features[:, len(selected):]
    return {"selected": selected, "constructed": constructed}


def run(config, datasets, output, smoke=False, data_root=ROOT):
    if config["study"] == "final" and not config.get("locked", False):
        raise ValueError("Lock the final shortlist and full protocol before running final evaluation")
    if config["study"] == "final" and (not 1 <= len(config["arms"]) <= 3 or not config["learners"]):
        raise ValueError("Final protocol requires one to three FE arms and at least one learner")
    if not config["arms"]:
        raise ValueError("Protocol needs at least one FE arm")
    forest_jobs = config.get("forest_jobs", 1)
    if not isinstance(forest_jobs, int) or forest_jobs < 1:
        raise ValueError("forest_jobs must be a positive integer")
    max_active_hours = config.get("max_active_hours")
    if max_active_hours is not None and (not isinstance(max_active_hours, (int, float)) or
                                         not np.isfinite(max_active_hours) or max_active_hours <= 0):
        raise ValueError("max_active_hours must be positive")
    output = Path(output)
    selected = [item for item in datasets if item["stage"] == ("final" if config["study"] == "final" else "exploration")]
    if smoke:
        selected = selected[:1]
    if not selected:
        raise ValueError("No datasets for this study stage")
    source_hashes = {}
    for dataset in selected:
        source_path = Path(data_root) / dataset["path"]
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        checksum_path = source_path.with_suffix(".sha256")
        if checksum_path.exists() and checksum_path.read_text(encoding="ascii").strip() != digest:
            raise ValueError(f"Source checksum mismatch: {source_path}")
        source_hashes[dataset["name"]] = digest
    protocol = {"config": config, "datasets": selected, "smoke": smoke,
                "smoke_row_cap": 120 if smoke else None, "source_sha256": source_hashes,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    progress_path = output / "progress.json"
    if output.exists() and any(output.iterdir()):
        if not progress_path.exists() or not (output / "checkpoints").is_dir():
            raise FileExistsError(f"Refusing to overwrite existing results in {output}")
        previous = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        last_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if last_progress["status"] == "complete":
            raise FileExistsError(f"Refusing to overwrite completed results in {output}")
        if previous != protocol:
            raise ValueError(f"Cannot resume changed protocol in {output}")
        previous_seconds = last_progress["active_seconds"]
        started_at = last_progress["started_at"]
        print(f"Resuming checkpoints in {output}", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir()
        _atomic_text(output / "protocol.json", json.dumps(protocol, indent=2))
        previous_seconds = 0.0
        started_at = datetime.now(timezone.utc).isoformat()
    session_start = time.perf_counter()
    total_evaluations = (len(selected) * (1 if smoke else len(config["outer_seeds"])) *
                         len(config["learners"]) * (1 if smoke else len(config["search_seeds"])) *
                         sum(40 if smoke else arm.get("budget", 1000) for arm in config["arms"]))
    rows, definitions, traces, splits, stability_rows = [], [], [], [], []

    def save_progress(count, active, status="running"):
        active_seconds = previous_seconds + time.perf_counter() - session_start
        limit_reached = (status == "running" and max_active_hours is not None and
                         active_seconds >= 3600 * max_active_hours)
        state = {"status": "compute_limit_reached" if limit_reached else status,
                 "started_at": started_at,
                 "updated_at": datetime.now(timezone.utc).isoformat(),
                 "active_seconds": active_seconds, "completed_candidate_evaluations": count,
                 "total_candidate_evaluations": total_evaluations, "active": active,
                 "completed_result_rows": len(rows)}
        _atomic_text(progress_path, json.dumps(state, indent=2))
        print(f"Progress {count}/{total_evaluations} evaluations; {active}", flush=True)
        if limit_reached:
            raise RuntimeError(f"Active compute limit reached in {output}; completed checkpoints are saved")

    def checkpoint_path(label):
        return output / "checkpoints" / (label + ".json")

    def save_checkpoint(path, value):
        _atomic_text(path, json.dumps(value))

    def snapshot():
        _save_tables(output, rows, definitions, traces, splits, stability_rows)

    saved_count = 0
    for path in (output / "checkpoints").glob("arm-*.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved_count += (saved["report"]["candidate_evaluations"] if saved["kind"] == "complete"
                        else saved["evaluations"])
    save_progress(saved_count, "resuming" if saved_count else "starting")
    for dataset_index, dataset in enumerate(selected):
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
                path = checkpoint_path(f"reference-{dataset_index}-{split_number}")
                if path.exists():
                    reference = json.loads(path.read_text(encoding="utf-8"))
                else:
                    reference = polynomial_reference(folds, dev_matrix, test_matrix, y_dev, y_test, outer_seed)
                    reference.update({"dataset": dataset["name"], "split": split_number,
                                      "outer_seed": outer_seed, "search_seed": None})
                    save_checkpoint(path, reference)
                rows.append(reference)
                snapshot()
            for model_index, model_name in enumerate(config["learners"]):
                path = checkpoint_path(f"baseline-{dataset_index}-{split_number}-{model_index}")
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    parameter, baseline_inner, baseline_report = saved["parameter"], saved["inner"], saved["report"]
                    tune_fits = baseline_report["tuning_model_fits"]
                else:
                    parameter, tune_fits, tune_scores = choose_learner(model_name, folds, outer_seed,
                                                                         forest_jobs)
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
                    save_checkpoint(path, {"parameter": parameter, "inner": baseline_inner,
                                           "report": baseline_report})
                rows.append(baseline_report)
                snapshot()
                signatures = {}
                for seed_index, search_seed in enumerate(config["search_seeds"][:1] if smoke else config["search_seeds"]):
                    for arm_index, arm in enumerate(config["arms"]):
                        path = checkpoint_path(f"arm-{dataset_index}-{split_number}-{model_index}-{seed_index}-{arm_index}")
                        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
                        if saved and saved["kind"] == "complete":
                            report, definition, history = saved["report"], saved["definition"], saved["history"]
                            signature = _saved_signature(definition, arm, dev_matrix)
                        else:
                            base_count = sum(row["candidate_evaluations"] for row in rows)
                            active = f"{dataset['name']} split={split_number} learner={model_name} seed={search_seed} arm={arm['name']}"
                            def arm_progress(count, k):
                                save_progress(base_count + count, f"{active} k={k}")
                            report, definition, history, signature = run_arm(
                                arm, folds, dev, test, y_dev, y_test, columns, model_name, parameter,
                                search_seed, baseline_inner, baseline_report["mae"], smoke,
                                resume=saved if saved else None,
                                checkpoint=lambda state: save_checkpoint(path, state),
                                progress=arm_progress, forest_jobs=forest_jobs)
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
                        if not saved or saved["kind"] != "complete":
                            save_checkpoint(path, {"kind": "complete", "report": report,
                                                   "definition": definitions[-1], "history": history})
                            snapshot()
                            save_progress(sum(row["candidate_evaluations"] for row in rows), active)
                for arm_name, items in signatures.items():
                    for left in range(len(items)):
                        for right in range(left + 1, len(items)):
                            selection_score, construction_score = matched_stability(items[left][1], items[right][1])
                            stability_rows.append({"dataset": dataset["name"], "split": split_number,
                                                   "learner": model_name, "arm": arm_name,
                                                   "seed_a": items[left][0], "seed_b": items[right][0],
                                                   "selection_jaccard": selection_score,
                                                   "constructed_similarity": construction_score})
                snapshot()
    snapshot()
    save_progress(sum(row["candidate_evaluations"] for row in rows), "finished", "complete")
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
