import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_dataset, split_and_preprocess
from benchmark import es_normalization, stability, instrument, tuned_baseline
from evaluation import EvaluationCounter
from features import candidate_frame
from optimizers import EvolutionStrategy, GeneticAlgorithm, GeneticProgrammingFeatureEngineer, PSO
from replay_diagnostics import restore_tree
from compare_budgets import compare
from refresh_stability import selected_features
from probe_regularization import fit_probe


class MilestoneOneTests(unittest.TestCase):
    def test_missing_features_survive_load_and_train_only_imputation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame({"x": [1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10.],
                          "y": list(range(10))}).to_csv(path, index=False)
            X, y = load_dataset(path, "y")
            self.assertEqual(len(X), 10)
            train, val, test, *_ = split_and_preprocess(X, y, 3)
            self.assertTrue(np.isfinite(train.to_numpy()).all())
            self.assertTrue(np.isfinite(val.to_numpy()).all())
            self.assertTrue(np.isfinite(test.to_numpy()).all())
            self.assertAlmostEqual(train["x"].mean(), 0, places=8)

    def test_es_uses_training_statistics_for_validation_feature(self):
        train = pd.DataFrame({"x": [0., 1., 2., 3.]})
        val = pd.DataFrame({"x": [10., 11.]})
        individual = {"mask": np.array([True]), "coefs": np.array([[1.]]),
                      "type_mask": np.array([False])}
        transformed = candidate_frame(individual, val, "ES", reference=train)
        expected = (10 - 1.5) / (np.std([0., 1., 2., 3.]) + 1e-6)
        self.assertAlmostEqual(transformed["synth_0"].iloc[0], expected)
        self.assertAlmostEqual(es_normalization(individual, train)[0]["mean"], 1.5)

    def test_gp_mutation_does_not_change_parent_and_can_change_tree(self):
        X = pd.DataFrame({"x": np.arange(8.), "z": np.arange(8.) + 1})
        gp = GeneticProgrammingFeatureEngineer(X, X, np.arange(8.), np.arange(8.), seed=5)
        parent = gp._grow(3)
        before = str(parent)
        children = [str(gp._mutate(parent)) for _ in range(12)]
        self.assertEqual(str(parent), before)
        self.assertTrue(any(child != before for child in children))
        tree = parent
        for _ in range(200):
            tree = gp._crossover(tree, gp._grow(3))
            tree = gp._mutate(tree)
            self.assertLessEqual(gp._tree_depth(tree), gp.max_depth)

    def test_shared_budget_counts_every_stage(self):
        counter = EvaluationCounter(3)
        for _ in range(2):
            counter.evaluate(lambda _: 1., None)
        counter.evaluate(lambda _: 0.5, None)
        self.assertEqual(counter.count, 3)
        with self.assertRaises(RuntimeError):
            counter.evaluate(lambda _: 0., None)

    def test_stability_compares_seeds_within_split(self):
        signatures = [
            {"dataset": "D", "split": 0, "method": "GA", "seed": seed,
             "selected": selected, "constructed": [values]}
            for seed, selected, values in ((1, ["x", "y"], [1., 2., 3.]),
                                           (2, ["x"], [3., 2., 1.]))
        ]
        result = stability(signatures).iloc[0]
        self.assertAlmostEqual(result.selection_jaccard, 0.5)
        self.assertAlmostEqual(result.constructed_similarity, 1.0)

    def test_stability_matches_each_constructed_feature_once(self):
        signatures = [
            {"dataset": "D", "split": 0, "method": "GA", "seed": 1, "selected": ["x"],
             "constructed": [[1., 2., 3., 4.], [4., 3., 2., 1.]]},
            {"dataset": "D", "split": 0, "method": "GA", "seed": 2, "selected": ["x"],
             "constructed": [[1., 2., 3., 4.], [1., -1., -1., 1.]]},
        ]
        self.assertAlmostEqual(stability(signatures).iloc[0].constructed_similarity, 0.5)

    def test_exact_budget_including_partial_generation(self):
        X = pd.DataFrame({"x": np.linspace(-1, 1, 16), "z": np.linspace(1, 2, 16)})
        y = X.x * 2 + X.z
        optimizers = [
            (GeneticAlgorithm(X, X, y, y, pop_size=4, n_gens=5, r_synth=1,
                              seed=5, verbose=False), "fitness"),
            (PSO(X, X, y, y, n_particles=4, n_iters=5, r_synth=1,
                 seed=5, verbose=False), "fitness_particle"),
            (EvolutionStrategy(X, X, y, y, mu=3, lambda_=4, n_gens=5, r_synth=1,
                               seed=5, verbose=False), "fitness"),
            (GeneticProgrammingFeatureEngineer(X, X, y, y, pop_size=4, n_gens=5,
                                               max_depth=2, seed=5), "_evaluate_individual"),
        ]
        for optimizer, method in optimizers:
            with self.subTest(optimizer=type(optimizer).__name__):
                counter = EvaluationCounter(10)
                instrument(optimizer, counter, method)
                optimizer.run()
                self.assertEqual(counter.count, 10)
                self.assertEqual(len(counter.history), 10)

    def test_decoding_matches_optimizer_fitness_features(self):
        X = pd.DataFrame({"x": [0., 1., 2., 3.], "z": [1., 2., 3., 4.]})
        y = pd.Series([1., 2., 3., 4.])
        ga = GeneticAlgorithm(X, X, y, y, r_synth=1, seed=3, verbose=False)
        individual = ga.make_individual()
        pd.testing.assert_frame_equal(ga.individual_to_dataframe(individual, X),
                                      candidate_frame(individual, X, "GA"))
        pso = PSO(X, X, y, y, r_synth=1, seed=3, verbose=False)
        vector = np.array([1., -1., 0.2, 0.4, 1.])
        decoded = {"mask": np.array([1, 0]), "coefs": np.array([[0.2, 0.4]]),
                   "type_mask": np.array([1])}
        pd.testing.assert_frame_equal(pso.particle_df(vector, X),
                                      candidate_frame(decoded, X, "PSO"))

    def test_outer_test_changes_cannot_change_training_or_validation(self):
        X = pd.DataFrame({"x": np.arange(40, dtype=float), "z": np.arange(40, dtype=float) ** 2})
        y = pd.Series(np.arange(40, dtype=float))
        original = split_and_preprocess(X, y, 11, return_indices=True)
        changed = X.copy()
        changed.loc[original[-1]["test"], "x"] = 1e9
        changed_y = y.copy()
        changed_y.loc[original[-1]["test"]] = -1e9
        modified = split_and_preprocess(changed, changed_y, 11, return_indices=True)
        for index in (0, 1, 3, 4):
            assertion = pd.testing.assert_frame_equal if index < 3 else pd.testing.assert_series_equal
            assertion(original[index], modified[index])

    def test_saved_gp_tree_can_be_reconstructed(self):
        definition = {"operator": "+", "left": {"feature": 0, "name": "x"},
                      "right": {"operator": "sqrt", "child": {"constant": -4.0}}}
        tree = restore_tree(definition)
        np.testing.assert_allclose(tree.evaluate(np.array([[1.], [3.]])), [3., 5.])

    def test_budget_comparison_matches_runs_by_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            base = {"dataset": ["D"], "split": [0], "method": ["GA"],
                    "optimizer_seed": [101], "MAE": [2.], "RMSE": [3.], "R2": [0.1]}
            pd.DataFrame(base).to_csv(old / "raw_results.csv", index=False)
            pd.DataFrame({**base, "MAE": [1.5]}).to_csv(new / "raw_results.csv", index=False)
            result = compare(old, new, root / "comparison.csv")
            self.assertAlmostEqual(result.delta_MAE.iloc[0], -0.5)

    def test_baseline_tuning_ignores_test_values(self):
        train = pd.DataFrame({"x": np.arange(12, dtype=float), "z": np.arange(12, dtype=float) ** 2})
        val = pd.DataFrame({"x": [12., 13., 14.], "z": [144., 169., 196.]})
        test = pd.DataFrame({"x": [15., 16.], "z": [225., 256.]})
        y_train = pd.Series(np.arange(12, dtype=float))
        y_val = pd.Series([12., 13., 14.])
        for name in ("Ridge", "Lasso", "Elastic Net", "Polynomial + Ridge"):
            with self.subTest(name=name):
                first = tuned_baseline(name, train, val, test, y_train, y_val)[-1]
                second = tuned_baseline(name, train, val, test * 1e9, y_train, y_val)[-1]
                self.assertEqual(first, second)

    def test_construction_stability_uses_actual_original_features(self):
        record = {"method": "PSO construction", "input_columns": ["x", "z"],
                  "definition": {"individual": {"mask": [0, 1]}}}
        self.assertEqual(selected_features(record), {"x", "z"})
        record["method"] = "GP + PSO"
        record["input_columns"] = ["x", "GP_feature"]
        record["definition"]["individual"]["mask"] = [1, 1]
        self.assertEqual(selected_features(record), {"x"})

    def test_ridge_probe_tunes_without_test_data(self):
        train = pd.DataFrame({"x": np.arange(12, dtype=float)})
        val = pd.DataFrame({"x": [12., 13.]})
        test = pd.DataFrame({"x": [14., 15.]})
        y_train = pd.Series(np.arange(12, dtype=float))
        y_val = pd.Series([12., 13.])
        for clipped in (False, True):
            first = fit_probe(train, val, test, y_train, y_val, clipped=clipped)
            second = fit_probe(train, val, test * 1e9, y_train, y_val, clipped=clipped)
            self.assertEqual(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
