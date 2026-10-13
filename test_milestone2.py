import unittest

import numpy as np
import pandas as pd

from continuous_features import construct, decode, make_fitness
from continuous_optimizers import METHODS, optimize
from data import split_and_preprocess


class ContinuousRepresentationTests(unittest.TestCase):
    def test_decoding_retains_inputs_and_squares_linear_combinations(self):
        X = np.array([[2.0, -3.0], [1.0, 4.0]])
        weights = np.array([[0.5, 0.25], [-1.0, 0.5]])
        expected = np.column_stack((X, np.square(X @ weights.T)))
        np.testing.assert_allclose(construct(X, weights.ravel()), expected)
        np.testing.assert_allclose(construct(X, decode(weights, 2)), expected)
        self.assertEqual(construct(X, weights).shape, (2, 4))
        with self.assertRaises(ValueError):
            decode(np.array([[1.1, 0.0], [0.0, 0.0]]), 2)

    def test_preprocessing_and_fitness_exclude_outer_test(self):
        # A test-only outlier must not alter training scaling or validation fitness.
        original = pd.DataFrame(np.arange(60.0).reshape(20, 3))
        target = pd.Series(np.arange(20.0))
        first = split_and_preprocess(original, target, 11, return_indices=True)
        # Locate a row in the held-out test partition before changing its values.
        test_row = first[-1]["test"][0]
        altered = original.copy()
        altered.loc[test_row] = 1e9
        second = split_and_preprocess(altered, target, 11, return_indices=True)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        candidate = np.zeros((2, 3))
        score_a = make_fitness(first[0], first[1], first[3], first[4])(candidate)
        score_b = make_fitness(second[0], second[1], second[3], second[4])(candidate)
        self.assertAlmostEqual(score_a, score_b)
        self.assertFalse(np.allclose(first[2], second[2]))


class OptimizerTests(unittest.TestCase):
    def test_exact_budget_and_reproducibility_including_partial_generation(self):
        sphere = lambda w: float(np.sum(w * w))
        for method in METHODS:
            with self.subTest(method=method):
                first = optimize(method, sphere, 2, 101, budget=103)
                second = optimize(method, sphere, 2, 101, budget=103)
                self.assertEqual(first.evaluations, 103)
                self.assertEqual(len(first.history), 103)
                self.assertEqual(first.history[-1][0], 103)
                np.testing.assert_allclose(first.weights, second.weights)
                self.assertAlmostEqual(first.score, second.score)

    def test_de_sphere_and_rosenbrock(self):
        sphere = lambda w: float(np.sum(w * w))
        shifted_rosenbrock = lambda w: float(
            100 * (w.ravel()[1] - w.ravel()[0] ** 2) ** 2 + (0.6 - w.ravel()[0]) ** 2)
        self.assertLess(optimize("DE", sphere, 2, 101).score, 0.01)
        self.assertLess(optimize("DE", shifted_rosenbrock, 1, 101).score, 0.01)

    def test_cma_sphere_and_rotated_quadratic(self):
        sphere = lambda w: float(np.sum(w * w))
        rotated = np.array([[20.0, -19.0], [-19.0, 20.0]])
        quadratic = lambda w: float(w.ravel() @ rotated @ w.ravel())
        self.assertLess(optimize("CMA-ES", sphere, 2, 101).score, 0.01)
        self.assertLess(optimize("CMA-ES", quadratic, 1, 101).score, 0.01)


if __name__ == "__main__":
    unittest.main()
