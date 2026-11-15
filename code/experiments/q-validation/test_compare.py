"""Regression tests for the Q approximation experiment."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare
from generate import finite_one_step, population_one_step


class QValidationTests(unittest.TestCase):
    def setUp(self):
        self.spec = compare.load_config(compare.CONFIG)

    def test_noise_feature_population_equals_prior(self):
        queries = np.array([[0., 0., 1., 0., -1.], [1., 1., 2., -1., 1.]])
        true = population_one_step(self.spec, queries)
        for feature in (2, 4):
            np.testing.assert_allclose(
                true[:, feature, :],
                np.broadcast_to(self.spec["pi"], (2, 3)),
                atol=1e-14)

    def test_finite_reference_missingness(self):
        x, y = compare.sample(self.spec, 60, np.random.default_rng(711))
        q = finite_one_step(self.spec, x[:12], x, y)
        pop = population_one_step(self.spec, x[:12])
        m = compare.metrics(q, pop, x[:12])
        self.assertGreater(m["n_observed_feature_queries"], 0)
        self.assertGreaterEqual(m["mean_TV"], 0)
        self.assertLessEqual(m["mean_TV"], 1)

    def test_archived_aggregate(self):
        p = compare.OUT / "metrics.json"
        if not p.exists():
            self.skipTest("no archived data")
        result = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(result["reference_sizes"], list(compare.SIZES))
        self.assertEqual(len(result["replicate_data"]), 48)
        means = [r["avg_mean_TV"] for r in result["comparison"]]
        self.assertTrue(all(a > b for a, b in zip(means, means[1:])))
        self.assertAlmostEqual(means[0], .0431162320732477, places=9)
        self.assertAlmostEqual(result["exact_default_300_reference"]["mean_TV"],
                               .04731606358477095, places=9)


if __name__ == "__main__":
    unittest.main()
