"""NumPy-only exact masked-walk integration and finite-difference checks (NO FIT)."""
from __future__ import annotations
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from numpy_engine import ExactMaskedWalkNP
from numpy_objectives import (train_loss_and_grad, train_diagnostics,
                              validation_diagnostics, classification)
from runner import prepare_row, AdamNP
from report import check_record


class TinyExactWalkIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_x = np.array([
            [-2., -1.0, 0.0, -1.2],
            [-2.2, -0.5, 0.2, -0.9],
            [-1.7, -1.2, -0.1, -1.1],
            [0.0,  0.6, 0.2, 0.3],
            [0.3,  0.8, 0.1, 0.0],
            [-0.2, 0.4, -0.2, 0.1],
            [2.0, -0.2, 0.4, 1.2],
            [2.3, -0.4, 0.2, 0.8],
            [1.9, -0.1, 0.1, 1.1],
        ], dtype=np.float64)
        cls.train_y = np.repeat(np.arange(3), 3)
        cls.val_x = cls.train_x[[0, 3, 6]].copy()
        cls.val_y = np.arange(3)
        cls.gamma = np.array([.8, 1.2, .6, .6])
        # Mask one noise feature but keep enough reachable reference nodes.
        cls.missing_x = cls.train_x.copy()
        cls.missing_x[1, 2] = np.nan
        cls.missing_val_x = cls.val_x.copy()
        cls.missing_val_x[0, 2] = np.nan

    def graph(self, *, mask=False, horizon=3):
        x = self.missing_x if mask else self.train_x
        vx = self.missing_val_x if mask else self.val_x
        return ExactMaskedWalkNP(x, self.train_y, self.gamma,
                                 beta=.4, lam=.8, horizon=horizon, query_x=vx)

    def test_exact_walk_mass_with_and_without_missingness(self):
        for mask in (False, True):
            for family in ("shared", "class"):
                with self.subTest(mask=mask, family=family):
                    g = self.graph(mask=mask)
                    z = np.zeros(g.shape(family))
                    y, _ = g.forward_train(z, family)
                    self.assertTrue(np.allclose(y.sum(axis=1), 1., atol=1e-10))
                    v = g.forward_query(z, family)
                    self.assertTrue(np.allclose(v.sum(axis=1), 1., atol=1e-10))

    def test_loss_and_analytic_grads_against_central_difference(self):
        for family in ("shared", "class"):
            for objective in ("sigmoid_JS_margin_tau005", "softplus_JS_tau005"):
                for mask in (False, True):
                    with self.subTest(family=family, objective=objective, mask=mask):
                        graph = self.graph(mask=mask)
                        z = np.zeros(graph.shape(family))
                        value, grad, reps, centers, scores = train_loss_and_grad(
                            graph, z, family, objective)
                        self.assertTrue(np.isfinite(value))
                        self.assertTrue(np.isfinite(grad).all())
                        selected = [0, grad.size // 3, grad.size // 2, grad.size - 1]
                        for index in selected:
                            flat = z.ravel()
                            eps = 1e-5
                            flat[index] += eps
                            up = train_loss_and_grad(graph, z, family, objective)[0]
                            flat[index] -= 2 * eps
                            down = train_loss_and_grad(graph, z, family, objective)[0]
                            flat[index] += eps
                            approx = (up - down) / (2 * eps)
                            self.assertAlmostEqual(
                                float(grad.ravel()[index]), float(approx),
                                delta=2e-5 + 1e-3 * abs(approx))
                        train = train_diagnostics(graph, reps, centers, scores)
                        val = validation_diagnostics(graph, z, family, centers, self.val_y)
                        self.assertIn("density_r_js", train)
                        self.assertIn("separation_i", train)
                        self.assertEqual(set(val), {
                            "accuracy", "balanced_accuracy", "precision_macro",
                            "recall_macro", "precision_by_class", "recall_by_class"})

    def test_complete_record_and_adam_moments(self):
        for family in ("shared", "class"):
            g = self.graph()
            z = np.zeros(g.shape(family))
            adam = AdamNP(z.shape)
            run = {"run_id": "fixture_numpy_" + family,
                   "noise_count": 1, "feature_indices": [0, 1, 2, 3],
                   "family": family, "objective": "softplus_JS_tau005"}
            row, grad = prepare_row(g, z, adam, run, 0, self.val_y)
            update = adam.update(z, grad)
            row["optimizer"]["adam_update_to_next"] = update.ravel().tolist()
            check_record(row, run)
            self.assertEqual(adam.step_count, 1)
            self.assertTrue(np.linalg.norm(update) > 0)

    def test_undefined_precision_zero(self):
        result = classification(np.array([[1., 0., 0.]] * 3),
                                np.array([0, 1, 2]))
        self.assertEqual(result["precision_by_class"], [1 / 3, 0, 0])
        self.assertEqual(result["recall_by_class"], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
