import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import fit_preprocessor, prepared_folds, transformed
from src.evaluation import candidate_fitness, choose_learner
from src.experiments import allocated_budgets, matched_stability, read_config, run, run_arm
from src.gp_features import feature_pair, node_count, optimize_gp
from src.optimizers import METHODS, optimize
from src.representations import construct
from src.strategies import transform


class RepresentationTests(unittest.TestCase):
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

    def test_gp_set_and_train_only_scaling(self):
        train, held = np.array([[0.], [1.], [2.]]), np.array([[100.]])
        first, second = feature_pair(train, held, [("x", 0), ("*", ("x", 0), ("x", 0))])
        self.assertEqual(first.shape, (3, 3))
        self.assertGreater(second[0, 1], 100.)
        self.assertEqual(node_count(("*", ("x", 0), ("x", 0))), 3)
        best, _, count, _ = optimize_gp(lambda trees: float(sum(node_count(t) for t in trees)), 2, 2, 7, 21)
        self.assertEqual(len(best), 2)
        self.assertEqual(count, 21)


class ProtocolTests(unittest.TestCase):
    def test_budget_allocation_and_optimizer_count(self):
        self.assertEqual(allocated_budgets([0, 1, 2, 4], 121, "construction"), {1: 41, 2: 40, 4: 40})
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
               "optimizer": "GA", "k_grid": [0, 1, 2], "budget": 81}
        report, definition, trace, signature = run_arm(
            arm, folds, X, X.iloc[:5], y, y[:5], columns, "ols", None, 4, 1.0, 1.0)
        self.assertEqual(report["candidate_evaluations"], 81)
        self.assertEqual(report["fe_model_fits"], 244)
        self.assertEqual(set(definition["inner_scores"]), {"0", "1", "2"})
        self.assertEqual(trace[-1]["global_evaluation"], 81)
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
        parameter, fits, _ = choose_learner("ridge", folds, 1)
        self.assertIn(parameter, [0.1, 1., 10., 100.])
        self.assertEqual(fits, 12)
        train = X.iloc[:20]
        prep = fit_preprocessor(train)
        original = transformed(prep, X.iloc[20:])
        changed = X.iloc[20:].copy()
        changed.iloc[0, 0] = 1e9
        np.testing.assert_allclose(transformed(prep, train), transformed(fit_preprocessor(train), train))
        self.assertFalse(np.allclose(original, transformed(prep, changed)))

    def test_smoke_isolated_from_outer_test_and_old_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(5)
            X = rng.normal(size=(90, 3))
            frame = pd.DataFrame(X, columns=["a", "b", "c"])
            frame["target"] = X[:, 0] ** 2 + X[:, 1]
            frame.to_csv(root / "input.csv", index=False)
            dataset = [{"name": "synthetic", "stage": "exploration", "path": "input.csv", "target": "target"}]
            config = read_config(Path(__file__).resolve().parents[1] / "configs/rq1.yaml")
            first = run(config, dataset, root / "one", smoke=True, data_root=root)
            self.assertEqual({row["arm"] for row in first}, {"no_fe", "selection", "construction", "combined"})
            self.assertEqual(sum(row["candidate_evaluations"] for row in first), 120)
            for row in first[1:]:
                self.assertEqual(row["fe_model_fits"], 121)
            # The locked outer labels can change reporting, never inner choices.
            splits = read_config(root / "one/splits.json")
            frame.loc[splits[0]["test"], "target"] += 1000
            frame.to_csv(root / "input.csv", index=False)
            second = run(config, dataset, root / "two", smoke=True, data_root=root)
            before = [item["inner_scores"] for item in map(__import__("json").loads,
                (root / "one/definitions.jsonl").read_text().splitlines())]
            after = [item["inner_scores"] for item in map(__import__("json").loads,
                (root / "two/definitions.jsonl").read_text().splitlines())]
            self.assertEqual(before, after)
            self.assertNotEqual(first[0]["mae"], second[0]["mae"])
            with self.assertRaises(FileExistsError):
                run(config, dataset, root / "one", smoke=True, data_root=root)
            final = read_config(Path(__file__).resolve().parents[1] / "configs/final.yaml")
            with self.assertRaisesRegex(ValueError, "Lock the final shortlist"):
                run(final, dataset, root / "never", smoke=True, data_root=root)


if __name__ == "__main__":
    unittest.main()
