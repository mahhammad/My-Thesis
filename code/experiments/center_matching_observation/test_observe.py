"""Tests for origin-aware Q, bounded JSD projection, and full-walk semantics."""
from __future__ import annotations
import unittest
import numpy as np
from projection import optimize_mixtures, jsd
from observe import CenterPolicyWalker, normalized_q, hypothesis_metrics
from walks import ReferenceGraph


class ObservationTest(unittest.TestCase):
    @staticmethod
    def graph():
        x=np.array([[0.,0.],[1.,1.],[2.,0.],[3.,1.],[4.,0.],[5.,1.]])
        y=np.array([0,0,1,1,2,2])
        return ReferenceGraph(x,y,np.ones(2))

    def test_optimal_mixture_exact_when_target_in_convex_hull(self):
        q=np.array([[[1.,0.],[0.,1.]],[[1.,0.],[np.nan,np.nan]]])
        v=np.array([[.35,.65],[.8,.2]])
        p,err,gap=optimize_mixtures(q,v,steps=90,tol=1e-8)
        self.assertAlmostEqual(float(p[0,0]),.35,places=4)
        self.assertAlmostEqual(float(p[0,1]),.65,places=4)
        self.assertEqual(float(p[1,1]),0.)
        self.assertAlmostEqual(float(err[0]),0.,places=10)
        self.assertAlmostEqual(float(err[1]),float(jsd(q[1,0],v[1])))
        self.assertLessEqual(float(gap[0]),1e-6)

    def test_exact_origin_exclusion(self):
        g=self.graph();r=CenterPolicyWalker(g,nwalks=100,beta=1.,horizon=1)
        for origin in range(len(g.y)):
            q=r.origin_q(origin)
            for curr in range(len(g.y)):
                for a in range(2):
                    w=g.weights[curr,a].copy();w[origin]=0.
                    expected=np.bincount(g.y,weights=w,minlength=3)/w.sum()
                    np.testing.assert_allclose(q[curr,a],expected,atol=1e-13)

    def test_beta_one_exact_one_step_distribution(self):
        g=self.graph();r=CenterPolicyWalker(g,nwalks=15000,beta=1.,lam=.3,horizon=4,
                                           optimizer_steps=120)
        origin=0;target=np.array([.1,.1,.8])
        p,_,_=r.origin_policies(origin,target)
        exact=np.nan_to_num(r.origin_q(origin)[origin],nan=0.).T@p[origin]
        observed=r._sample(g.weights[origin],p[origin],p,origin,20261104)
        np.testing.assert_allclose(observed,exact,atol=.017)
        self.assertAlmostEqual(float(observed.sum()),1.,places=12)

    def test_hypothesis_metrics_only_uses_labels_in_scoring(self):
        c=np.array([[.85,.1,.05],[.1,.8,.1],[.05,.1,.85]])
        r=np.array([[c[0],c[1],c[2]], [c[1],c[0],c[2]], [c[2],c[0],c[1]]])
        m,p=hypothesis_metrics(r,c,np.array([0,1,2]))
        m2,p2=hypothesis_metrics(r,c,np.array([1,0,2]))
        np.testing.assert_array_equal(p,p2)
        self.assertNotEqual(m['balanced_accuracy'],m2['balanced_accuracy'])

    def test_denominator_and_missing_data(self):
        numer=np.array([[[2.,3.],[0.,0.]]]);denom=np.array([[5.,0.]])
        q=normalized_q(numer,denom)
        np.testing.assert_allclose(q[0,0],[.4,.6])
        self.assertTrue(np.isnan(q[0,1]).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
