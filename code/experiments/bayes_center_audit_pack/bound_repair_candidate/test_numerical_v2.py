"""V2 numerical regression: independent Gaussian path enumeration; n <= 6 only.

This oracle computes raw Gaussian weights, allowed destinations, and paths
separately; it NEVER calls candidate.shared_feature_kernel or candidate DP.
Original archived dense Bellman DP is compared via the other test module.
"""
from __future__ import annotations

import math
import unittest
from unittest import mock
import numpy as np

from exact_origin_excluded_bound import (
    _continuation, _excluded_denominator, exact_origin_excluded_center_bounds,
    shared_feature_kernel,
)
from test_exact_origin_excluded_bound import (
    _TinyGraph, alpha_from_project, exact_policy_expectation, original_dp,
)

SEED = 20261111
TOL = 1e-10


def independent_enumeration(x, y, gamma, alpha, origin, logits):
    """Independent Gaussian path enumerator (no normalized candidate B).

    Select feature with a finite-state masked softmax. For subsequent steps,
    exclude both current i and original s, and normalize *raw* Gaussian
    destination weights, not a subtraction of probabilities.
    """
    x = np.asarray(x, float); y = np.asarray(y, int)
    gamma = np.asarray(gamma, float); alpha = np.asarray(alpha, float)
    n, d = x.shape
    assert n <= 6
    K = int(y.max()) + 1
    present = np.isfinite(x)
    logs = np.asarray(logits, float)
    state = np.zeros((n, d), dtype=np.float64)
    actions0 = np.flatnonzero(present[origin])
    v = logs[d, actions0]
    prob0 = np.exp(v - v.max()); prob0 /= prob0.sum()
    for action_index, b in enumerate(actions0):
        dest = [j for j in range(n) if j != origin and present[j, b]]
        weight = np.array([math.exp(-gamma[b] * (x[origin, b]-x[j, b]) ** 2) for j in dest])
        assert weight.sum() > 0
        for j, w in zip(dest, weight / weight.sum()):
            state[j, b] += prob0[action_index] * w
    result = np.zeros(K, dtype=float)
    for t, a in enumerate(alpha):
        for j in range(n):
            result[y[j]] += a * state[j].sum()
        if t == len(alpha) - 1:
            break
        nxt = np.zeros_like(state)
        for i in range(n):
            for incoming in range(d):
                p_state = state[i, incoming]
                if not p_state:
                    continue
                legal = []
                for b in range(d):
                    if not present[i, b]:
                        continue
                    dest = [j for j in range(n) if j not in (origin, i) and present[j, b]]
                    weights = np.array([math.exp(-gamma[b] * (x[i,b]-x[j,b])**2) for j in dest])
                    if weights.sum() > 0:
                        legal.append((b,dest,weights))
                if not legal:
                    raise ValueError("independent oracle dead-end")
                z = np.array([logs[incoming, b] for b, _, _ in legal])
                action_prob = np.exp(z - z.max()); action_prob /= action_prob.sum()
                for (b, dest, weights), p_b in zip(legal, action_prob):
                    for j, q in zip(dest, weights / weights.sum()):
                        nxt[j,b] += p_state * p_b * q
        state = nxt
    np.testing.assert_allclose(result.sum(), 1.0, atol=2e-12, rtol=0)
    return result


class NumericalV2TinyTests(unittest.TestCase):
    def checked(self, x, y, gamma, alpha):
        self.assertLessEqual(len(x), 6)
        new = exact_origin_excluded_center_bounds(x, y, gamma, alpha, return_per_origin=True)
        graph = _TinyGraph(x, y, gamma, alpha)
        dense_lo, dense_hi = original_dp()[0](graph, y)
        np.testing.assert_allclose(new["lower_per_origin"], dense_lo, rtol=0, atol=TOL)
        np.testing.assert_allclose(new["upper_per_origin"], dense_hi, rtol=0, atol=TOL)
        for lo, hi in [(new["lower_per_origin"], new["upper_per_origin"]),
                       (new["lower_center_probabilities"], new["upper_center_probabilities"])]:
            lo = np.asarray(lo); hi=np.asarray(hi)
            self.assertTrue(np.isfinite(lo).all() and np.isfinite(hi).all())
            self.assertTrue((lo <= hi + TOL).all())
            self.assertTrue((lo >= -TOL).all() and (hi <= 1 + TOL).all())
        return new

    def test_original_auditor_counterexample_independent_paths(self):
        x = np.array([[0.],[0.],[math.sqrt(math.log(4))]])
        y = np.array([0,0,1]); gamma=np.array([1.])
        alpha=alpha_from_project(.4,.8,2)
        np.testing.assert_allclose(alpha,[11/15,4/15],rtol=0,atol=1e-15)
        logits = np.zeros((2,1))
        true = independent_enumeration(x,y,gamma,alpha,0,logits)
        self.assertAlmostEqual(true[1],0.36,places=14)
        new = self.checked(x,y,gamma,alpha)
        self.assertAlmostEqual(new["upper_per_origin"][0][1],true[1],places=14)
        self.assertAlmostEqual(new["lower_per_origin"][0][1],true[1],places=14)
        self.assertGreater(true[1],71/375)

    def test_new_cancellation_counterexample_independent_paths_and_center(self):
        x=np.array([[0.],[0.],[math.sqrt(math.log(1/(2e-12)))]])
        y=np.array([0,1,0]); gamma=np.array([1.])
        alpha=alpha_from_project(.4,.8,2)
        logits=np.zeros((2,1))
        new=self.checked(x,y,gamma,alpha)
        true_origins=np.stack([independent_enumeration(x,y,gamma,alpha,s,logits)
                               for s in range(3)])
        true_center0=true_origins[y==0].mean(axis=0)[0]
        self.assertAlmostEqual(true_center0,0.3833333333338,delta=1e-10)
        lo=np.asarray(new["lower_center_probabilities"])[0,0]
        hi=np.asarray(new["upper_center_probabilities"])[0,0]
        self.assertLessEqual(lo - true_center0,TOL)
        self.assertLessEqual(true_center0 - hi,TOL)
        np.testing.assert_allclose(new["lower_per_origin"],true_origins,atol=TOL,rtol=0)
        np.testing.assert_allclose(new["upper_per_origin"],true_origins,atol=TOL,rtol=0)
        base=shared_feature_kernel(x,gamma)
        den=_excluded_denominator(base,0)
        self.assertGreater(den[1,0],1e-12)
        # Old numerator cancellation, with the same stable denominator, fails.
        value=np.zeros((3,1));value[:,0]=[1.,0.,1.]
        bad=np.einsum('bij,jb->ib',base,value,optimize=True)
        bad-=base[:,:,0].T*value[0][None,:]
        bad/=den
        correct=_continuation(base,value,0,den,den>1e-12)
        self.assertGreater(abs(bad[1,0]-correct[1,0]),1e-6)
        self.assertAlmostEqual(correct[1,0],1.,delta=TOL)

    def test_h1_and_seeded_multifeature_masks_against_independent_oracle(self):
        x=np.array([[0.,1.,np.nan],[.1,np.nan,.3],[.3,.5,.2],
                    [.4,.2,np.nan],[np.nan,.8,.5],[.8,.7,.2]])
        y=np.array([0,1,0,1,0,1]);gamma=np.array([1.,.5,2.])
        rng=np.random.default_rng(SEED)
        for H in (1,3):
            alpha=alpha_from_project(.4,.8,H)
            new=self.checked(x,y,gamma,alpha)
            for _ in range(2):
                logits=rng.normal(size=(4,3))
                for s in range(6):
                    real=independent_enumeration(x,y,gamma,alpha,s,logits)
                    lo=np.asarray(new["lower_per_origin"])[s]
                    hi=np.asarray(new["upper_per_origin"])[s]
                    self.assertTrue((real >= lo-TOL).all())
                    self.assertTrue((real <= hi+TOL).all())
                    dense=exact_policy_expectation(_TinyGraph(x,y,gamma,alpha),y,s,logits)
                    np.testing.assert_allclose(real,dense,atol=TOL,rtol=0)

    def test_threshold_below_guard_and_underflow_are_rejected(self):
        y=np.array([0,1,0]);gamma=np.array([1.]);alpha=alpha_from_project(.4,.8,2)
        below=np.array([[0.],[0.],[math.sqrt(math.log(1/(4e-13)))]])
        with self.assertRaisesRegex(ValueError,"without a legal|degenerate"):
            exact_origin_excluded_center_bounds(below,y,gamma,alpha)
        underflow=np.array([[0.],[0.],[1000.]])
        with self.assertRaisesRegex(ValueError,"degenerate|without a legal"):
            exact_origin_excluded_center_bounds(underflow,y,gamma,alpha)

    def test_nonfinite_and_out_of_range_fail_closed(self):
        x=np.array([[0.],[.1],[.2]]);y=np.array([0,1,0]);gamma=np.array([1.])
        alpha=alpha_from_project(.4,.8,2)
        with mock.patch('exact_origin_excluded_bound._continuation',return_value=np.full((3,1),float('nan'))):
            with self.assertRaises(ArithmeticError):
                exact_origin_excluded_center_bounds(x,y,gamma,alpha)
        with mock.patch('exact_origin_excluded_bound._continuation',return_value=np.full((3,1),10.)):
            with self.assertRaisesRegex(ArithmeticError,"out-of-range"):
                exact_origin_excluded_center_bounds(x,y,gamma,alpha)

if __name__=='__main__':
    unittest.main(verbosity=2)
