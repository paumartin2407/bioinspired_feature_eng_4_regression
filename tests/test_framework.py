import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import fit_preprocessor, prepared_folds, transformed
from src.evaluation import candidate_fitness, choose_learner
from src.experiments import _atomic_text, allocated_budgets, matched_stability, read_config, run, run_arm
from src.optimizers import METHODS, optimize, optimize_until
from src.replay import replay
from src.representations import construct
from src.strategies import transform


class RepresentationTests(unittest.TestCase):
    def test_atomic_progress_write_survives_temporary_file_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text("old", encoding="utf-8")
            replace = os.replace
            attempts = 0

            def locked_then_replace(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError(5, "temporary lock")
                replace(source, destination)

            with patch("src.experiments.os.replace", side_effect=locked_then_replace), \
                    patch("src.experiments.time.sleep") as sleep:
                _atomic_text(path, "new")
            self.assertEqual(path.read_text(encoding="utf-8"), "new")
            self.assertEqual(attempts, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_variable_k_and_replay(self):
        X = np.array([[1., 2.], [3., 4.]])
        for family in ("quadratic", "rbf", "interactions"):
            for k in (0, 1, 2, 4):
                genes = np.zeros({"quadratic": 2, "rbf": 3, "interactions": 2}[family] * k)
                built = construct(X, genes, family, k)
                self.assertEqual(built.shape, (2, k))
                np.testing.assert_allclose(built, construct(X.copy(), genes.tolist(), family, k))
        np.testing.assert_allclose(construct(X, [1., 0.], "quadratic", 1).ravel(), [1., 9.])

    def test_strategy_masks_and_combined_construction(self):
        X = np.array([[2., 3.], [4., 5.]])
        np.testing.assert_allclose(transform(X, "selection", "quadratic", [1., -1.], 0), X[:, :1])
        combined = transform(X, "combined", "quadratic", [1., -1., 0., 1.], 1)
        self.assertEqual(combined.shape, (2, 2))
        np.testing.assert_allclose(combined[:, 1], X[:, 1] ** 2)
        self.assertEqual(transform(X, "construction", "quadratic", [], 0).shape, X.shape)
        np.testing.assert_allclose(transform(X, "selection", "quadratic", [-0.9, -0.3], 0), X[:, 1:2])

    def test_numerical_edges_and_duplicate_interactions(self):
        distant = construct(np.array([[1e6, -1e6], [2e6, 2e6]]), [0., 0., 0.], "rbf", 1)
        self.assertTrue(np.isfinite(distant).all())
        np.testing.assert_allclose(distant, 0.)
        X = np.array([[2., 3.], [4., 5.]])
        repeated = construct(X, [-1., -1., -1., -1.], "interactions", 2)
        np.testing.assert_allclose(repeated[:, 0], repeated[:, 1])
        self.assertEqual(np.linalg.matrix_rank(repeated), 1)
        accounting = {"fits": 0, "failed_folds": 0, "failed_candidates": 0}
        fold = [(np.array([[1e200], [2e200]]), np.array([[3e200]]),
                 np.array([0., 1.]), np.array([2.]), None, None)]
        score = candidate_fitness(fold, "construction", "quadratic", 1, "ols", None, 1, accounting)([1.])
        self.assertEqual(score, float("inf"))
        self.assertEqual(accounting["failed_candidates"], 1)


class ProtocolTests(unittest.TestCase):
    def test_mismatched_fixed_learner_settings_are_rejected(self):
        config = {"learners": ["linear_panel_ridge"],
                  "fixed_model_settings": {"ridge": {"alpha": 10.0}}}
        with self.assertRaisesRegex(ValueError, "do not match the fixed learner"):
            run(config, [], "unused")

    def test_removed_reference_models_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Reference models are not supported"):
            run({"references": ["polynomial_ridge"]}, [], "unused")

    def test_time_limited_optimizers_retain_best_completed_candidate(self):
        for method in METHODS:
            with self.subTest(method=method):
                now = [0.0]
                seen = []
                def objective(genes):
                    seen.append(genes.copy())
                    now[0] += 0.6
                    return float(np.square(genes).sum())
                result = optimize_until(method, objective, 3, 7, 1.0, 100, clock=lambda: now[0])
                self.assertEqual(result.evaluations, 2)
                self.assertEqual(len(result.history), 2)
                self.assertAlmostEqual(result.score, min(float(np.square(v).sum()) for v in seen))
                np.testing.assert_allclose(result.genes, min(seen, key=lambda v: float(np.square(v).sum())))
                self.assertGreaterEqual(now[0], 1.0)
                self.assertLess(now[0], 1.6)
                empty = optimize_until(method, objective, 3, 7, 1.0, 100, clock=lambda: 1.0)
                self.assertEqual(empty.evaluations, 0)
                self.assertIsNone(empty.genes)

    def test_timed_arm_accounts_for_k_and_candidate_fits(self):
        X = pd.DataFrame(np.random.default_rng(12).normal(size=(75, 3)), columns=list("abc"))
        y = (X.a ** 2 + X.b).to_numpy()
        folds, columns = prepared_folds(X, y, 4)
        arm = {"name": "timed", "strategy": "construction", "family": "quadratic", "optimizer": "GA",
               "k_grid": [0, 1, 2], "budget": 2000, "record_search_timing": True, "time_budget_seconds": 0.3}
        report, definition, trace, _ = run_arm(arm, folds, X, X.iloc[:5], y, y[:5], columns, "ols", None, 4, 1.0, 1.0)
        self.assertGreater(report["candidate_evaluations"], 0)
        self.assertEqual(sum(definition["actual_evaluations_by_k"].values()), report["candidate_evaluations"])
        self.assertEqual(len(trace), report["candidate_evaluations"])
        self.assertEqual(report["fe_model_fits"], 3 * report["candidate_evaluations"] + 1)
        self.assertEqual(definition["time_allocation_seconds"], {"0": 0, "1": 0.15, "2": 0.15})
        self.assertTrue(np.isfinite(report["selected_inner_mae"]))

    def test_budget_allocation_and_optimizer_count(self):
        self.assertEqual(allocated_budgets([0, 1, 2, 4], 121, "construction"), {1: 41, 2: 40, 4: 40})
        self.assertEqual(allocated_budgets([0, 1, 2, 4], 160, "combined"),
                         {0: 40, 1: 40, 2: 40, 4: 40})
        for method in METHODS:
            with self.subTest(method=method):
                result = optimize(method, lambda genes: float(np.square(genes).sum()), 3, 3, 41)
                self.assertEqual(result.evaluations, 41)
                self.assertEqual(len(result.history), 41)

    def test_shared_k_budget_and_stability(self):
        X = pd.DataFrame(np.random.default_rng(4).normal(size=(75, 3)), columns=list("abc"))
        y = (X.a ** 2 + X.b).to_numpy()
        folds, columns = prepared_folds(X, y, 4)
        arm = {"name": "two_k", "strategy": "construction", "family": "quadratic",
               "optimizer": "GA", "k_grid": [0, 1, 2], "budget": 81, "record_search_timing": True}
        report, definition, trace, signature = run_arm(
            arm, folds, X, X.iloc[:5], y, y[:5], columns, "ols", None, 4, 1.0, 1.0)
        self.assertEqual(report["candidate_evaluations"], 81)
        self.assertEqual(report["fe_model_fits"], 244)
        self.assertEqual(set(definition["inner_scores"]), {"0", "1", "2"})
        self.assertEqual(trace[-1]["global_evaluation"], 81)
        self.assertTrue(all(t["elapsed_wall_seconds"] >= 0 and t["elapsed_cpu_seconds"] >= 0 for t in trace))
        self.assertTrue(all(a["elapsed_wall_seconds"] <= b["elapsed_wall_seconds"] and
                            a["elapsed_cpu_seconds"] <= b["elapsed_cpu_seconds"]
                            for a, b in zip(trace, trace[1:])))
        self.assertGreaterEqual(report["runtime_seconds"], trace[-1]["elapsed_wall_seconds"])
        self.assertGreaterEqual(report["cpu_seconds"], trace[-1]["elapsed_cpu_seconds"])
        self.assertEqual(sum(definition["evaluation_allocation"].values()), 81)
        self.assertEqual(matched_stability(signature, signature), (1.0, 1.0))

    def test_fold_local_preprocessing_and_fit_count(self):
        X = pd.DataFrame({"numeric": [float(i) for i in range(30)],
                          "category": ["a"] * 29 + ["b"]})
        y = np.arange(30., dtype=float)
        folds, columns = prepared_folds(X, y, 12)
        self.assertEqual(len(folds), 3)
        accounting = {"fits": 0, "failed_folds": 0}
        objective = candidate_fitness(folds, "construction", "quadratic", 1, "ols", None, 1, accounting)
        self.assertTrue(np.isfinite(objective(np.zeros(len(columns)))))
        self.assertEqual(accounting["fits"], 3)
        parameter, fits, scores = choose_learner("linear_panel_ridge", folds, 1)
        self.assertIsNone(parameter)
        self.assertEqual(fits, 3)
        baseline = candidate_fitness(folds, "construction", "quadratic", 0,
                                     "linear_panel_ridge", parameter, 1, {"fits": 0, "failed_folds": 0})([])
        self.assertAlmostEqual(baseline, min(scores))
        train = X.iloc[:20]
        prep = fit_preprocessor(train)
        original = transformed(prep, X.iloc[20:])
        changed = X.iloc[20:].copy()
        changed.iloc[0, 0] = 1e9
        np.testing.assert_allclose(transformed(prep, train), transformed(fit_preprocessor(train), train))
        self.assertFalse(np.allclose(original, transformed(prep, changed)))

    def test_representations_with_all_learners(self):
        rng = np.random.default_rng(41)
        X = pd.DataFrame(rng.normal(size=(36, 3)), columns=list("abc"))
        y = (X.a * X.b + X.c).to_numpy()
        folds, columns = prepared_folds(X, y, 4)
        for model_name in ("ols", "linear_panel_ols", "linear_panel_ridge", "linear_panel_lasso", "linear_panel_elastic", "linear_panel_huber"):
            for family, genes in (("quadratic", np.zeros(3)), ("rbf", np.zeros(4)),
                                  ("interactions", np.zeros(2)), ("hinge", np.zeros(4))):
                with self.subTest(model=model_name, family=family):
                    accounting = {"fits": 0, "failed_folds": 0, "failed_candidates": 0}
                    score = candidate_fitness(folds, "construction", family, 1, model_name,
                                              None, 4, accounting)(genes)
                    self.assertTrue(np.isfinite(score))
                    self.assertEqual(accounting["fits"], 3)

    def test_smoke_isolated_from_outer_test_and_old_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(5)
            X = rng.normal(size=(90, 3))
            frame = pd.DataFrame(X, columns=["a", "b", "c"])
            frame["target"] = X[:, 0] ** 2 + X[:, 1]
            frame.to_csv(root / "input.csv", index=False)
            dataset = [{"name": "synthetic", "stage": "exploration", "path": "input.csv", "target": "target"}]
            config = read_config(Path(__file__).resolve().parents[1] / "configs/phase_1/adaptive.yaml")
            first = run(config, dataset, root / "one", smoke=True, data_root=root)
            self.assertEqual({row["arm"] for row in first}, {"no_fe", "selection", "construction", "combined"})
            self.assertEqual(sum(row["candidate_evaluations"] for row in first), 120)
            for row in first[1:]:
                self.assertEqual(row["fe_model_fits"], 121)
                replayed = replay(root / "one", "synthetic", 0, "ols", row["arm"], 11, root)
                self.assertAlmostEqual(replayed["mae"], row["mae"])
            # The locked outer labels can change reporting, never inner choices.
            splits = read_config(root / "one/splits.json")
            frame.loc[splits[0]["test"], "target"] += 1000
            frame.to_csv(root / "input.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Source data changed"):
                replay(root / "one", "synthetic", 0, "ols", "construction", 11, root)
            second = run(config, dataset, root / "two", smoke=True, data_root=root)
            before = [item["inner_scores"] for item in map(__import__("json").loads,
                (root / "one/definitions.jsonl").read_text().splitlines())]
            after = [item["inner_scores"] for item in map(__import__("json").loads,
                (root / "two/definitions.jsonl").read_text().splitlines())]
            self.assertEqual(before, after)
            self.assertNotEqual(first[0]["mae"], second[0]["mae"])
            with self.assertRaises(FileExistsError):
                run(config, dataset, root / "one", smoke=True, data_root=root)
            final = {"study": "final", "locked": False}
            with self.assertRaisesRegex(ValueError, "Lock the final shortlist"):
                run(final, dataset, root / "never", smoke=True, data_root=root)

    def test_resume_after_completed_arm_and_partial_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(8)
            frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
            frame["target"] = frame.a ** 2 + frame.b
            frame.to_csv(root / "input.csv", index=False)
            datasets = [{"name": "synthetic", "stage": "exploration", "path": "input.csv", "target": "target"}]
            arm = {"name": "quadratic", "strategy": "construction", "family": "quadratic",
                   "optimizer": "GA", "k_grid": [0, 1, 2], "budget": 80,
                   "record_search_timing": True}
            config = {"study": "rq4", "outer_seeds": [101], "search_seeds": [11],
                      "inner_folds": 3, "learners": ["ols"],
                      "arms": [arm, {**arm, "name": "second"}]}
            real_optimize = optimize
            calls = 0
            def interrupt_on_fourth(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("simulated reboot")
                return real_optimize(*args, **kwargs)
            with patch("src.experiments.optimize", side_effect=interrupt_on_fourth):
                with self.assertRaisesRegex(RuntimeError, "simulated reboot"):
                    run(config, datasets, root / "resumed", data_root=root)
            checkpoints = root / "resumed/checkpoints"
            self.assertEqual(json.loads((checkpoints / "arm-0-0-0-0-0.json").read_text())["kind"], "complete")
            self.assertEqual(json.loads((checkpoints / "arm-0-0-0-0-1.json").read_text())["evaluations"], 40)
            self.assertEqual(len(pd.read_csv(root / "resumed/runs.csv")), 2)
            resumed = run(config, datasets, root / "resumed", data_root=root)
            timed = pd.read_csv(root / "resumed/convergence.csv")
            for _, group in timed.groupby("arm"):
                self.assertTrue(group.elapsed_wall_seconds.is_monotonic_increasing)
                self.assertTrue(group.elapsed_cpu_seconds.is_monotonic_increasing)
                self.assertEqual(group.elapsed_wall_seconds.notna().sum(), 80)
            fresh = run(config, datasets, root / "fresh", data_root=root)
            self.assertEqual(len(resumed), len(fresh))
            for left, right in zip(resumed, fresh):
                self.assertEqual((left["arm"], left["candidate_evaluations"], left["selected_k"]),
                                 (right["arm"], right["candidate_evaluations"], right["selected_k"]))
                self.assertAlmostEqual(left["mae"], right["mae"])
            self.assertEqual(json.loads((root / "resumed/progress.json").read_text())["status"], "complete")
            with self.assertRaisesRegex(RuntimeError, "Active compute limit reached"):
                run({**config, "max_active_hours": 1e-10}, datasets, root / "limited", data_root=root)
            self.assertEqual(json.loads((root / "limited/progress.json").read_text())["status"],
                             "compute_limit_reached")


if __name__ == "__main__":
    unittest.main()
