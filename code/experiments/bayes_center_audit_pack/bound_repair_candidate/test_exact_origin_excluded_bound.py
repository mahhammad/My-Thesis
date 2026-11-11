"""Locked tiny fixtures only. Never run full300 or the 37-config grid.

Uses the *original* ``bounds_for_all_origins`` from the archived module as
an oracle, with a tiny numpy-backed Graph stub (n <= 6). This test is meant
to execute from a full checkout of the repository, not a copied directory.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest

import numpy as np

from exact_origin_excluded_bound import (
    _excluded_denominator,
    exact_origin_excluded_center_bounds,
    shared_feature_kernel,
)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL_BOUND = ROOT / "code/experiments/archive/bayes_center_inverse/feasibility_bound.py"


def original_dp():
    if not ORIGINAL_BOUND.is_file():
        raise RuntimeError("Original frozen feasibility_bound.py not found: require full repository checkout")
    spec = importlib.util.spec_from_file_location("frozen_original_bound_for_tiny_fixture", ORIGINAL_BOUND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bounds_for_all_origins, mod.center_bounds


class _TorchLike:
    def __init__(self, x):
        self.x = np.asarray(x)

    def detach(self):
        return self

    def numpy(self):
        return self.x


class _TinyGraph:
    """Builds exact historical dense (origin,b,i,j) tensor only for n<=6."""

    def __init__(self, x, labels, gamma, alpha):
        self.n, self.d = x.shape
        if self.n > 6:
            raise AssertionError("DO NOT materialize full-scale origin kernels")
        self.k = int(labels.max()) + 1
        self.origins = np.arange(self.n)
        self.coeff = _TorchLike(alpha)
        base = shared_feature_kernel(x, gamma)
        self.start = _TorchLike(base[:, self.origins, :].transpose(1, 2, 0))
        banned = np.broadcast_to(base[None, ...], (self.n, self.d, self.n, self.n)).copy()
        for o in range(self.n):
            banned[o, :, :, o] = 0.0
        den = banned.sum(axis=-1, keepdims=True)
        np.divide(banned, den, out=banned, where=den > 0)
        self.transitions = _TorchLike(banned)
        self.valid = _TorchLike((den[..., 0] > 1e-12).transpose(0, 2, 1))
        self.valid_start = _TorchLike(np.isfinite(x))


def alpha_from_project(beta: float, lam: float, H: int):
    result = np.zeros(H)
    for length in range(1, H + 1):
        prob = beta * (1 - beta)**(length - 1) if length < H else (1 - beta)**(H - 1)
        total = sum(lam**(t - 1) for t in range(1, length + 1))
        for t in range(1, length + 1):
            result[t - 1] += prob * lam**(t - 1) / total
    return result


def masked_softmax(values, valid):
    valid = np.asarray(valid, bool)
    if not valid.any():
        raise ValueError("empty masked feature action")
    z = np.exp(np.asarray(values, float)[valid] - max(np.asarray(values, float)[valid]))
    result = np.zeros(len(valid))
    result[valid] = z / z.sum()
    return result


def exact_policy_expectation(g: _TinyGraph, labels, origin, logits):
    """Deterministic finite-length walk expectation under one FIXED Z matrix."""
    k, d, n = g.k, g.d, g.n
    start = g.start.numpy()[origin]
    valid = g.valid.numpy()[origin]
    startmask = g.valid_start.numpy()[origin]
    pi0 = masked_softmax(logits[d], startmask)
    state = start * pi0[None, :]  # [current_sample, arrival_feature]
    result = np.zeros(k)
    for t, weight in enumerate(g.coeff.numpy()):
        for c in range(k):
            result[c] += weight * state[labels == c].sum()
        if t + 1 == len(g.coeff.numpy()):
            break
        chosen = np.zeros((n, d))
        for i in range(n):
            for a in range(d):
                if state[i, a]:
                    chosen[i] += state[i, a] * masked_softmax(logits[a], valid[i])
        kernel = g.transitions.numpy()[origin]
        state = np.einsum("ib,bij->jb", chosen, kernel, optimize=True)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=0, atol=1e-12)
    return result


class BellmanRepairTinyTests(unittest.TestCase):
    def compare_original(self, x, y, gamma, alpha, *, test_realized=True):
        self.assertLessEqual(len(x), 6)
        g = _TinyGraph(x, y, gamma, alpha)
        canon_origin, canon_center = original_dp()
        old_lo, old_hi = canon_origin(g, y)
        canon = canon_center(g, y)
        new = exact_origin_excluded_center_bounds(x, y, gamma, alpha, return_per_origin=True)
        for key, old in [("lower_per_origin", old_lo), ("upper_per_origin", old_hi),
                         ("lower_center_probabilities", canon["lower_center_probabilities"]),
                         ("upper_center_probabilities", canon["upper_center_probabilities"])]:
            np.testing.assert_allclose(new[key], old, atol=5e-13, rtol=0, err_msg=key)
        if test_realized:
            d = x.shape[1]
            z = np.linspace(-0.3, 0.6, num=(d + 1) * d).reshape(d + 1, d)
            for origin in range(len(x)):
                realized = exact_policy_expectation(g, y, origin, z)
                lo, hi = new["lower_per_origin"][origin], new["upper_per_origin"][origin]
                self.assertTrue(np.all(realized >= np.array(lo) - 5e-13))
                self.assertTrue(np.all(realized <= np.array(hi) + 5e-13))
        return new

    def test_auditor_counterexample_h2(self):
        x = np.array([[0.0], [0.0], [math.sqrt(math.log(4))]])
        y = np.array([0, 0, 1]); gamma = np.array([1.0])
        alpha = alpha_from_project(0.4, 0.8, 2)
        np.testing.assert_allclose(alpha, [11 / 15, 4 / 15], atol=1e-15)
        new = self.compare_original(x, y, gamma, alpha)
        self.assertAlmostEqual(new["upper_per_origin"][0][1], 0.36, places=14)
        self.assertAlmostEqual(new["lower_per_origin"][0][1], 0.36, places=14)
        # Historical incorrect return-allowed B yields 71/375, NOT outer.
        invalid_old = (11 / 15) * (1 / 5) + (4 / 15) * (4 / 25)
        self.assertAlmostEqual(invalid_old, 71 / 375, places=14)
        self.assertGreater(new["lower_per_origin"][0][1], invalid_old)
        self.assertAlmostEqual(new["lower_center_probabilities"][0][1], 0.36, places=14)

    def test_h1_first_hop_only_and_beta_lambda_independence(self):
        x = np.array([[0.], [.2], [1.], [1.2]])
        y = np.array([0, 1, 0, 1]); gamma = np.array([1.2])
        a = alpha_from_project(0.4, 0.8, 1)
        b = alpha_from_project(0.7, 1.0, 1)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
        v = self.compare_original(x, y, gamma, a)
        np.testing.assert_allclose(v["lower_per_origin"], v["upper_per_origin"], atol=1e-14)

    def test_multifeature_masks_and_stochastic_fixed_policy(self):
        x = np.array([[0.0, 1.0, np.nan], [0.1, np.nan, 0.3],
                      [0.3, 0.5, 0.2], [0.4, 0.2, np.nan],
                      [np.nan, 0.8, 0.5], [0.8, 0.7, 0.2]])
        y = np.array([0, 1, 0, 1, 0, 1]); gamma = np.array([1.0, 0.5, 2.0])
        alpha = alpha_from_project(0.4, 0.8, 3)
        self.compare_original(x, y, gamma, alpha)
        B = shared_feature_kernel(x, gamma)
        for origin in range(6):
            den = _excluded_denominator(B, origin)
            for b in range(3):
                for i in range(6):
                    row = B[b, i, :].copy(); row[origin] = 0
                    if den[i, b] > 1e-12:
                        row /= den[i, b]
                        self.assertAlmostEqual(row.sum(), 1.0, places=13)
                    self.assertEqual(row[origin], 0.0)
                    self.assertEqual(row[i], 0.0)
                    self.assertTrue(np.all(row[~np.isfinite(x[:, b])] == 0))

    def test_seeded_multifeature_fixed_policies(self):
        # Two unrelated fixed class-conditioned logits, same tiny masked graph.
        seed = 20261111
        rng = np.random.default_rng(seed)
        x = np.array([[0.0, 1.0, np.nan], [0.1, np.nan, 0.3],
                      [0.3, 0.5, 0.2], [0.4, 0.2, np.nan],
                      [np.nan, 0.8, 0.5], [0.8, 0.7, 0.2]])
        y = np.array([0, 1, 0, 1, 0, 1]); gamma = np.array([1.0, 0.5, 2.0])
        alpha = alpha_from_project(0.4, 0.8, 3)
        bounds = exact_origin_excluded_center_bounds(x, y, gamma, alpha, return_per_origin=True)
        graph = _TinyGraph(x, y, gamma, alpha)
        for _ in range(2):
            logits = rng.normal(0, 1, size=(4, 3))
            for origin in range(len(x)):
                actual = exact_policy_expectation(graph, y, origin, logits)
                np.testing.assert_array_less(np.array(bounds["lower_per_origin"][origin])-5e-13, actual)
                np.testing.assert_array_less(actual, np.array(bounds["upper_per_origin"][origin])+5e-13)

    def test_fail_closed_on_missing_destinations(self):
        x = np.array([[0., 0.], [.2, np.nan], [np.nan, np.nan], [1.1, 1.]])
        y = np.array([0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "non-dead-end|at least one observed"):
            exact_origin_excluded_center_bounds(x, y, [1., 1.], [1.])

    def test_fail_closed_alpha(self):
        x = np.array([[0.], [.2], [1.]])
        y = np.array([0, 0, 1])
        with self.assertRaisesRegex(ValueError, "alpha"):
            exact_origin_excluded_center_bounds(x, y, [1.], [0.8, -0.2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
