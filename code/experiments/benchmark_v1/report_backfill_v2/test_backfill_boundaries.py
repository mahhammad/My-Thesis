"""[RESEARCHER-03] Tiny N00 boundary tests. No fitting, no full Benchmark replay, no Test."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/"code"/"experiments"/"benchmark_v1"))
from engine import ExactMaskedWalk, DT, masked_softmax
from metrics import kl_forward as legacy_kl

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

reporter = _module("r03_reporter", ROOT/"code"/"experiments"/"benchmark_v1"/"report_snapshot.py")
canonical = _module("r03_canonical", ROOT/"code"/"policy_evaluator"/"metrics.py")
rebuild = _module("r03_centers", HERE/"center_only_rebuild.py")


class BoundaryChecks(unittest.TestCase):
    def test_kl_unsupported_positive_mass_is_infinite_only_in_exact_contract(self):
        p = torch.tensor([[1., 0.]], dtype=DT)
        q = torch.tensor([[0., 1.]], dtype=DT)
        old = float(legacy_kl(p, q)[0])
        new = float(reporter.kl_forward(p.numpy(), q.numpy())[0])
        self.assertAlmostEqual(old, -np.log(1e-300), places=9)
        self.assertTrue(np.isposinf(new))
        self.assertTrue(np.isposinf(canonical.kl_divergence(p.numpy(), q.numpy())[0]))

    def test_masked_softmax_valid_and_all_invalid(self):
        logits = torch.tensor([[4., -2.]], dtype=DT)
        normal = masked_softmax(logits, torch.tensor([[True, False]]))
        np.testing.assert_allclose(normal.numpy(), [[1., 0.]], atol=1e-12)
        none = masked_softmax(logits, torch.tensor([[False, False]]))
        np.testing.assert_allclose(none.numpy(), [[.5, .5]], atol=1e-12)
        # A nominally invalid row contributes nothing ONLY if its state mass is zero.
        self.assertTrue(torch.equal(none * torch.zeros_like(none), torch.zeros_like(none)))
        self.assertGreater(float((none * torch.ones_like(none)).sum()), 0.0)

    def test_valid_graph_mass_and_all_missing_query_boundary(self):
        x = np.array([[0., 0.], [.2, 1.], [.4, 1.2], [.6, 1.6]], float)
        y = np.array([0, 0, 1, 1], int)
        g = ExactMaskedWalk(x, y, [1., 1.], horizon=3)
        logits = torch.zeros((3, 2), dtype=DT)
        train = g.represent_train(logits, "shared")
        np.testing.assert_allclose(train.sum(1).numpy(), np.ones(4), atol=1e-8)
        missing_query = ExactMaskedWalk(
            x, y, [1., 1.], horizon=3,
            query_x=np.full((1, 2), np.nan))
        with self.assertRaisesRegex(AssertionError, "mass lost"):
            missing_query.represent_query(logits, "shared")
        # Failure above is a mass invariant, NOT proof of a corrupt historical run.

    def test_true_ties_and_near_ties_are_distinct_from_balanced_accuracy(self):
        examples = [
            ([1., 1.], 0, 0, True, True, False),
            ([.7, .7 + 5e-13], 1, 1, True, False, False),
            ([.7, .7 + 5e-13], 0, 1, True, True, True)
        ]
        for scores, truth, prediction, near, hard, error in examples:
            s = np.asarray(scores)
            pred = int(np.argmax(s))
            runner = s[np.arange(len(s)) != truth].max()
            margin = s[truth] - runner
            is_near = np.isclose(s, s.max(), atol=1e-12, rtol=0).sum() > 1
            self.assertEqual(pred, prediction)
            self.assertEqual(bool(is_near), near)
            self.assertEqual(bool(margin <= 0), hard)
            self.assertEqual(pred != truth, error)

    def test_center_only_matches_referenced_chapter3_metric_function(self):
        c = np.asarray([[.75, .20, .05], [.15, .75, .10], [.08, .22, .70]])
        expected = canonical.center_metrics(c)
        got = rebuild.center_report(c, canonical.center_metrics, canonical.jsd)
        for field in ("T", "BT", "TNL", "BTNL", "I", "min_pairwise_sqrt_JS"):
            self.assertAlmostEqual(float(got[field]), float(
                expected["min_center_js_distance"] if field=="min_pairwise_sqrt_JS"
                else expected[field]), places=12)
        self.assertEqual(len(got["pairwise_JS"]), 3)


if __name__ == "__main__":
    unittest.main()
