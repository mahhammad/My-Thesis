"""Tests for fixed policy, per-walk normalization, center separation and no leakage."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from walks import ReferenceGraph, load_policy, policy_fingerprint
from centers import build_centers
from evaluate import run
from metrics import classify, jsd, report

HERE = Path(__file__).parent


def simple_policy(nfeatures=1, **kwargs):
    return dict(logits=np.zeros((nfeatures + 1, nfeatures)),
                kernel_gamma=np.ones(nfeatures), beta=1., **{'lambda':1., 'max_steps':1,
                'walks_per_sample':10000, 'seed':5}, name='uniform', **kwargs)


class EvaluatorTest(unittest.TestCase):
    def test_uniform_and_guided_configuration(self):
        p = load_policy(HERE/'policy_uniform.json', 5)
        g = load_policy(HERE/'policy_guided.json', 5)
        self.assertEqual(p['logits'].shape, (6, 5))
        self.assertTrue(np.all(p['logits'] == 0))
        self.assertGreater(g['logits'][5, 0], g['logits'][5, 2])
        self.assertNotEqual(policy_fingerprint(p), policy_fingerprint(g))

    def test_one_step_matches_exact_reference_weights(self):
        x=np.array([[0.], [1.], [2.], [3.]])
        y=np.array([0,1,0,1])
        p=simple_policy()
        graph=ReferenceGraph(x,y,p['kernel_gamma'])
        q,_=graph.representation(x[:1],p,train_indices=np.array([0]))
        w=np.exp(-(x[1:,0]-x[0,0])**2)
        expected=np.bincount(y[1:],weights=w/w.sum(),minlength=2)
        np.testing.assert_allclose(q[0],expected,atol=0.015)

    def test_longer_walk_normalizes_each_walk_and_excludes_origin(self):
        x=np.array([[0.],[1.],[2.]],float)
        y=np.array([0,1,2])
        p=simple_policy()
        p.update(beta=0.6, **{'lambda':0.5}, max_steps=2,walks_per_sample=22000)
        graph=ReferenceGraph(x,y,p['kernel_gamma'])
        q,_=graph.representation(x[:1],p,train_indices=np.array([0]))
        w=np.exp(-np.array([1.,4.])); w/=w.sum()
        # length 1: weights [1]; length 2: per-walk weights [1, 0.5] / 1.5
        p1=0.6*w[0]+0.4*(w[0]+0.5*w[1])/1.5
        p2=1-p1
        np.testing.assert_allclose(q[0],[0.,p1,p2],atol=0.02)
        self.assertAlmostEqual(q[0].sum(),1.0,places=11)
        self.assertAlmostEqual(q[0,0],0.,places=11)

    def test_missing_features_are_not_selected(self):
        x=np.array([[0., np.nan],[1.,5.],[2.,np.nan],[3.,6.]])
        y=np.array([0,1,0,1])
        p=simple_policy(2)
        graph=ReferenceGraph(x,y,p['kernel_gamma'])
        q,_=graph.representation(x[:1],p,train_indices=np.array([0]))
        self.assertAlmostEqual(q.sum(),1.0)
        # first query feature 1 missing; sampled destinations via feature 0
        self.assertGreater(q[0,1],0)

    def test_missing_everything_raises_instead_of_silent_prior(self):
        x=np.array([[0.],[1.],[2.]])
        y=np.array([0,1,0])
        p=simple_policy()
        with self.assertRaisesRegex(ValueError, 'dead-end'):
            ReferenceGraph(x,y,p['kernel_gamma']).representation(np.array([[np.nan]]),p)

    def test_centers_are_class_means_and_evaluate_does_not_refit(self):
        x=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,1.],[4.,0.],[5.,1.]])
        y=np.array([0,0,1,1,0,1]); p=simple_policy(2)
        p.update(walks_per_sample=600)
        centers, train_px = build_centers(x,y,p)
        for c in range(2):
            np.testing.assert_allclose(centers[c],train_px[y==c].mean(axis=0))
        previous=centers.copy()
        query=x[:3]+0.1
        result1,pred1=run(x,y,query,centers,p,query_y=np.array([0,1,0]))
        result2,pred2=run(x,y,query,centers,p,query_y=np.array([1,0,1]))
        np.testing.assert_array_equal(pred1['neighborhoods'],pred2['neighborhoods'])
        np.testing.assert_array_equal(pred1['prediction'],pred2['prediction'])
        np.testing.assert_array_equal(centers,previous)
        self.assertEqual(result1['centers_updated'] if 'centers_updated' in result1 else False,False)
        self.assertNotEqual(result1['confusion'],result2['confusion'])

    def test_metric_identity(self):
        x=np.array([[1.,0.],[0.5,0.5],[0.,1.]])
        D=jsd(x[:,None,:],x[None,:,:])
        S=1-D/np.log(2)
        np.testing.assert_allclose(np.diag(D),0)
        self.assertAlmostEqual(S[0,2],0.0,places=12)
        self.assertGreater(S[0,1],S[0,2])
        np.testing.assert_array_equal(classify(x,x)['prediction'],np.arange(3))

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError): load_policy(HERE/'policy_uniform.json', 2)


if __name__=='__main__':
    unittest.main(verbosity=2)
