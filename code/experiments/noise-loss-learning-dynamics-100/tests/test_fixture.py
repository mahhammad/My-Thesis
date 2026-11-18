"""Tiny exact-walk integration checks: gradients and metric scope, NO FIT."""
from __future__ import annotations
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code/experiments/benchmark_v1"))

import numpy as np
import torch
from engine import DT, ExactMaskedWalk
from runner import canonical_objective, prepare_row, classification
from report import check_record


class TinyExactWalkIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train_x = np.asarray([
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
        cls.val_x = np.asarray([cls.train_x[0], cls.train_x[3], cls.train_x[6]])
        cls.val_y = np.arange(3)
        cls.gamma = np.asarray([0.8, 1.2, 0.6, 0.6])

    def graph(self):
        graph = ExactMaskedWalk(self.train_x, self.train_y, self.gamma,
                                beta=.4, lam=.8, horizon=3, query_x=self.val_x)
        graph.validation_y = self.val_y
        return graph

    def test_both_losses_and_families_with_all_trace_fields(self):
        for family in ("shared", "class"):
            for objective in ("sigmoid_JS_margin_tau005", "softplus_JS_tau005"):
                with self.subTest(family=family, objective=objective):
                    graph = self.graph()
                    shape = ((5, 4) if family == "shared" else (3, 5, 4))
                    z = torch.nn.Parameter(torch.zeros(shape, dtype=DT))
                    optimizer = torch.optim.Adam([z], lr=.12)
                    run = {
                        "run_id": f"fixture__{objective}__{family}",
                        "dataset_id": "fixture", "noise_count": 1,
                        "feature_indices": [0, 1, 2, 3],
                        "family": family, "objective": objective,
                    }
                    row = prepare_row(graph, z, optimizer, run, 0, None)
                    self.assertTrue(np.isfinite(row["train"]["loss"]))
                    self.assertTrue(np.isfinite(row["train"]["separation_i"]))
                    self.assertTrue(np.isfinite(row["train"]["density_r_js"]))
                    self.assertEqual(set(row["validation"]), {
                        "accuracy", "balanced_accuracy", "precision_macro",
                        "recall_macro", "precision_by_class", "recall_by_class"})
                    self.assertEqual(len(row["optimizer"]["gradients"]), z.numel())
                    initial = z.detach().clone()
                    optimizer.step()
                    delta = z.detach() - initial
                    row["optimizer"]["adam_update_to_next"] = delta.flatten().tolist()
                    check_record(row, run)

    def test_precision_undefined_policy(self):
        scores = np.asarray([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        metrics = classification(scores, np.array([0, 1, 2]))
        self.assertEqual(metrics["precision_by_class"], [1/3, 0., 0.])
        self.assertEqual(metrics["recall_by_class"], [1., 0., 0.])


if __name__ == "__main__":
    unittest.main()
