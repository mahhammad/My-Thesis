"""Regression checks for the class-target Q projection solver."""
import unittest
import numpy as np
from projection import optimize_mixtures, jsd


class ProjectionTest(unittest.TestCase):
    def test_exactly_attainable_and_missing_feature(self):
        q = np.array([[[.9,.1],[.1,.9]], [[1.,0.],[np.nan,np.nan]]])
        targets = np.array([[.7,.3],[.3,.7]])
        pi, losses, gaps = optimize_mixtures(q, targets, steps=150)
        np.testing.assert_allclose(pi.sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(pi[0], [.75,.25], atol=1e-4)
        np.testing.assert_allclose(pi[1], [1.,0.])
        self.assertLess(losses[0], 1e-10)
        self.assertLess(np.max(gaps), 1e-5)

    def test_convex_hull_outside_target(self):
        q = np.array([[[.2,.8],[.3,.7]]])
        target = np.array([.95,.05])
        pi, value, gap = optimize_mixtures(q, target, steps=80)
        np.testing.assert_allclose(pi[0], [0.,1.], atol=1e-4)
        np.testing.assert_allclose(value[0], jsd(q[0,1], target), atol=1e-9)
        self.assertLess(gap[0], 1e-6)

    def test_no_admissible_features_raises(self):
        q = np.full((1,3,2),np.nan)
        with self.assertRaisesRegex(ValueError, 'no admissible feature'):
            optimize_mixtures(q, np.array([.5,.5]))

if __name__ == '__main__': unittest.main()
