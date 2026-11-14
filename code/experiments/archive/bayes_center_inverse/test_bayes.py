"""Small deterministic tests of Bayes targets and matrix-propagation experiments."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'code/data_generator'))
sys.path.insert(0, str(ROOT / 'code/experiments/multistep_policy_target_match'))
from generate import generate, load_config
from run import ExactGraph, DT, coefficients
from run_bayes import bayes_posterior, bayes_class_centers, center_geometry, summarize_training, jsdiv
from matrix_walk import deterministic_representations, approximation_diagnostic
from feasibility_bound import center_bounds


class BayesInverseTests(unittest.TestCase):
    def test_posterior_with_missing_features(self):
        spec = load_config(ROOT / 'code/data_generator/config.json')
        x = np.array([[np.nan] * len(spec['gamma']),
                      [-2., np.nan, 0., np.nan, np.nan],
                      [np.nan, 0., 1., np.nan, np.nan]])
        q = bayes_posterior(spec, x)
        np.testing.assert_allclose(q.sum(axis=1), 1., atol=1e-13)
        np.testing.assert_allclose(q[0], spec['pi'], atol=1e-13)
        self.assertTrue(np.all(q >= 0))

    def test_class_means_match_definition(self):
        spec = load_config(ROOT / 'code/data_generator/config.json')
        with tempfile.TemporaryDirectory() as tmp:
            d = generate(spec, tmp, finite_q=False)
        x, y = d['train'][0], d['train'][2]
        u, post = bayes_class_centers(spec, x, y)
        self.assertEqual(u.shape, (len(spec['pi']), len(spec['pi'])))
        for c in range(len(u)):
            np.testing.assert_allclose(u[c], post[y == c].mean(axis=0), atol=1e-14)
        self.assertTrue(np.allclose(u.sum(axis=1), 1))

    def test_truncated_exact_walk_error_bound(self):
        # A truncated, renormalized mixture of stochastic distributions is
        # at TV distance at most the omitted coefficient mass from the full sum.
        alpha = coefficients(.4, .8, 12)
        rng = np.random.default_rng(11)
        qs = rng.dirichlet([1., 2., 3.], size=12)
        exact = alpha @ qs
        for L in (1, 2, 3, 4, 8):
            approx = (alpha[:L] @ qs[:L]) / alpha[:L].sum()
            self.assertLessEqual(.5 * np.abs(exact - approx).sum(), alpha[L:].sum() + 1e-14)

    def test_matrix_propagation_probability_and_target(self):
        spec = load_config(ROOT / 'code/data_generator/config.json')
        with tempfile.TemporaryDirectory() as tmp:
            data = generate(spec, tmp, finite_q=False)
        x = data['train'][0][:24]
        y = data['train'][2][:24]
        if len(np.unique(y)) != len(spec['pi']):
            self.skipTest('tiny class subset happened to exclude a class')
        graph = ExactGraph(x, y, spec['gamma'], np.arange(len(x)), data['validation'][0][:5])
        z = torch.zeros((graph.d + 1, graph.d), dtype=DT)
        p = graph.center_train(z, np.arange(len(x)), return_all=True).numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-10)
        self.assertTrue(np.all(p >= -1e-12))
        U, fullpost = bayes_class_centers(spec, data['train'][0], data['train'][2])
        s = summarize_training(graph, y, [z] * graph.k, U, bayes_posterior(spec, x))
        np.testing.assert_allclose(np.array(s['centers']).sum(axis=1), 1., atol=1e-10)
        self.assertTrue(0 <= s['mean_target_JSD'] <= np.log(2))
        self.assertTrue(0 <= center_geometry(np.asarray(s['centers']))['I'] <= np.log(graph.k))


    def test_matrix_split_matches_existing_exact(self):
        spec = load_config(ROOT / 'code/data_generator/config.json')
        with tempfile.TemporaryDirectory() as tmp:
            data = generate(spec, tmp, finite_q=False)
        x = data['train'][0][:24]
        y = data['train'][2][:24]
        graph = ExactGraph(x, y, spec['gamma'], np.arange(len(x)), data['validation'][0][:5])
        z = torch.tensor(np.random.default_rng(2026).normal(size=(graph.d + 1, graph.d)), dtype=DT)
        ids = np.arange(len(x))
        with torch.no_grad():
            exact_ref = graph.center_train(z, ids, return_all=True)
            exact_ours = deterministic_representations(graph, z, ids, graph.H)
        np.testing.assert_allclose(exact_ref.numpy(), exact_ours.numpy(), atol=1e-13)
        approximation_diagnostic(graph, [z] * graph.k, y, cutoff=(1, 2, 3, 4))

    def test_feasibility_bounds_contain_any_fixed_policy(self):
        spec = load_config(ROOT / 'code/data_generator/config.json')
        with tempfile.TemporaryDirectory() as tmp:
            data = generate(spec, tmp, finite_q=False)
        x = data['train'][0][:18]
        y = data['train'][2][:18]
        graph = ExactGraph(x, y, spec['gamma'], np.arange(len(x)), data['validation'][0][:3])
        z = torch.tensor(np.random.default_rng(311).normal(size=(graph.d + 1, graph.d)), dtype=DT)
        bounds = center_bounds(graph, y)
        lower = np.asarray(bounds['lower_center_probabilities'])
        upper = np.asarray(bounds['upper_center_probabilities'])
        for c in range(graph.k):
            with torch.no_grad():
                v = graph.center_train(z, np.flatnonzero(y == c)).numpy()
            self.assertTrue(np.all(v >= lower[c] - 1e-11))
            self.assertTrue(np.all(v <= upper[c] + 1e-11))

    def test_identical_vectors_have_zero_jsd(self):
        a = np.array([[.2, .5, .3], [.01, .99, 0.]])
        np.testing.assert_allclose(jsdiv(a, a), 0., atol=1e-15)


if __name__ == '__main__':
    unittest.main()
