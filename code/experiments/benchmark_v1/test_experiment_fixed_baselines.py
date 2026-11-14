"""Lightweight contract tests for the P1/P2 fixed-policy sweep; no dataset/Test."""
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment_fixed_baselines import experiment_plan, logits_for


class FixedBaselinePlanChecks(unittest.TestCase):
    def test_grid_counts_and_base_point_only_once(self):
        base = (.4, .8, 12)
        all_jobs = experiment_plan("both")
        u = [r for r in all_jobs if r["kind"] == "uniform"]
        g = [r for r in all_jobs if r["kind"] == "guided"]
        self.assertEqual(len(all_jobs), 13)
        self.assertEqual(len(u), 9)
        self.assertEqual(len(g), 4)
        self.assertEqual(len(set((r["beta"], r["lam"], r["H"]) for r in u)), 9)
        self.assertEqual(sum((r["beta"], r["lam"], r["H"]) == base for r in u), 1)
        self.assertEqual({r["alpha"] for r in g}, {.25, .5, .75, 1.})
        self.assertEqual(len({r["id"] for r in all_jobs}), 13)
        self.assertEqual(len(experiment_plan("guided")), 5)

    def test_alpha_zero_is_uniform_and_guided_scores_match_historical_matrix(self):
        jobs = experiment_plan("guided")
        zero = logits_for(jobs[0], 5).numpy()
        np.testing.assert_array_equal(zero, np.zeros((6, 5)))
        guided = logits_for(jobs[-1], 5).numpy()
        base = np.array([1.4, .2, -1.2, .9, -1.2])
        expected = np.tile(base, (6, 1))
        expected[:5, :] += .25 * np.eye(5)
        np.testing.assert_allclose(guided, expected, atol=1e-15, rtol=0)
        np.testing.assert_allclose(guided[5], base, atol=1e-15, rtol=0)


if __name__ == "__main__":
    unittest.main()
