import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_dataset, split_and_preprocess
from benchmark import es_normalization, stability
from evaluation import EvaluationCounter
from features import candidate_frame
from optimizers import EvolutionStrategy, GeneticProgrammingFeatureEngineer


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


if __name__ == "__main__":
    unittest.main()
