import unittest
import warnings

import numpy as np

from src.representations import components, construct, dimension
from src.experiments import construction_inputs
from sklearn.exceptions import ConvergenceWarning
from src.evaluation import fit_with_diagnostics, learner


class FollowupTests(unittest.TestCase):
    def test_linear_panel_uses_training_only_feature_and_target_scaling(self):
        train = np.array([[0., 5.], [1., 7.], [2., 9.], [3., 11.]])
        target = np.array([100., 110., 120., 130.])
        for name in ("ols", "ridge", "lasso", "elastic", "huber"):
            model = learner("linear_panel_" + name).fit(train, target)
            with self.subTest(name=name):
                np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, train.mean(axis=0))
                self.assertAlmostEqual(model.named_steps["transformedtargetregressor"].transformer_.mean_[0], target.mean())
                self.assertEqual(model.predict(np.array([[1000., -500.]])).shape, (1,))
                np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, train.mean(axis=0))

    def test_convergence_warning_is_recorded(self):
        class WarnModel:
            def fit(self, X, y):
                warnings.warn("not converged", ConvergenceWarning)
                return self
        accounting = {}
        fit_with_diagnostics(WarnModel(), [[1]], [1], accounting)
        self.assertEqual(accounting["convergence_warning_fits"], 1)

    def test_hinge_values_and_input_tracking(self):
        x = np.array([[1., 2.], [-2., 1.], [0., 0.]])
        genes = np.array([1., 0., 0.5, 0., -0.5, -0.5])
        self.assertEqual(dimension("hinge", 2, 2), 6)
        expected = np.column_stack((np.maximum(0, x[:, 0] - 1.5),
                                    np.maximum(0, -0.5 * x[:, 1] + 1.5)))
        np.testing.assert_allclose(construct(x, genes, "hinge", 2), expected)
        self.assertEqual(construction_inputs("hinge", genes, 2, 2), {0, 1})
        self.assertEqual(construct(x, [], "hinge", 0).shape, (3, 0))

    def test_mixed_construction_and_inputs(self):
        x = np.array([[1., 2.], [0., 1.]])
        genes = np.array([0.2, -0.3, 0.1, 0.5, -0.2])
        self.assertEqual(dimension("quadratic_rbf", 2, 2), len(genes))
        expected = np.column_stack((construct(x, genes[:2], "quadratic", 1),
                                    construct(x, genes[2:], "rbf", 1)))
        np.testing.assert_allclose(construct(x, genes, "quadratic_rbf", 2), expected)
        self.assertEqual(construction_inputs("quadratic_rbf", genes, 2, 2), {0, 1})
        self.assertEqual(construct(x, [], "quadratic_rbf", 0).shape, (2, 0))

    def test_new_balanced_mixtures(self):
        x = np.array([[1., 2.], [0., -1.]])
        q = np.array([0.2, -0.3])
        r = np.array([0.1, 0.5, -0.2])
        h = np.array([-0.4, 0.6, 0.3])
        for name, families, pieces in [
            ("quadratic_hinge", ("quadratic", "hinge"), (q, h)),
            ("quadratic_rbf_hinge", ("quadratic", "rbf", "hinge"), (q, r, h)),
        ]:
            with self.subTest(family=name):
                k = len(families)
                genes = np.concatenate(pieces)
                expected = np.column_stack([construct(x, part, family, 1)
                                            for family, part in zip(families, pieces)])
                self.assertEqual(dimension(name, 2, k), len(genes))
                np.testing.assert_allclose(construct(x, genes, name, k), expected)
                self.assertEqual(construction_inputs(name, genes, 2, k), {0, 1})
                self.assertEqual(construct(x, [], name, 0).shape, (2, 0))
                with self.assertRaisesRegex(ValueError, "divisible"):
                    dimension(name, 2, k + 1)

    def test_asymmetric_allocations_and_construction(self):
        x = np.array([[1., 2.], [-1., 0.]])
        cases = [
            ("quadratic_rbf_rbf_heavy", 4, (1, 3)),
            ("quadratic_rbf_rbf_heavy", 8, (2, 6)),
            ("quadratic_rbf_hinge_rbf_heavy", 4, (1, 2, 1)),
            ("quadratic_rbf_hinge_rbf_heavy", 6, (1, 4, 1)),
            ("quadratic_rbf_hinge_quadratic_heavy", 4, (2, 1, 1)),
            ("quadratic_rbf_hinge_quadratic_heavy", 6, (4, 1, 1)),
        ]
        for family, k, expected_counts in cases:
            with self.subTest(family=family, k=k):
                parts = components(family, k)
                self.assertEqual(tuple(count for _, count in parts), expected_counts)
                genes = np.linspace(-0.8, 0.8, dimension(family, 2, k))
                expected, offset = [], 0
                for name, count in parts:
                    end = offset + dimension(name, 2, count)
                    expected.append(construct(x, genes[offset:end], name, count))
                    offset = end
                np.testing.assert_allclose(construct(x, genes, family, k), np.column_stack(expected))
                self.assertEqual(construction_inputs(family, genes, 2, k), {0, 1})
                self.assertEqual(construct(x, [], family, 0).shape, (2, 0))
                with self.assertRaisesRegex(ValueError, "supports"):
                    dimension(family, 2, 3)


if __name__ == "__main__":
    unittest.main()
