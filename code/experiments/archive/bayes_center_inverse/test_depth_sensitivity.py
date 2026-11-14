"""Tests for effective-hop coefficients and outer reachability bounds."""
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'code/data_generator'))
sys.path.insert(0,str(ROOT/'code/experiments/multistep_policy_target_match'))
from run_bayes import generate, load_config, bayes_class_centers
from run import ExactGraph, DT, coefficients, jsd
from depth_sensitivity import calculate, summarize_bounds
from feasibility_bound import center_bounds, origin_return_relaxed_bounds


class DepthSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec=load_config(ROOT/'code/data_generator/config.json')
        with tempfile.TemporaryDirectory() as tmp:
            d=generate(cls.spec,tmp,finite_q=False)
        cls.x,cls.y=d['train'][0],d['train'][2]
        cls.target,_=bayes_class_centers(cls.spec,cls.x,cls.y)

    def test_length_one_independent_of_beta_lambda(self):
        for beta in (.1,.4,.9):
            for lam in (.2,.8,1.):
                np.testing.assert_allclose(coefficients(beta,lam,1),[1.])

    def test_effective_depth_bounded_and_tail(self):
        a=coefficients(.4,.8,12)
        self.assertAlmostEqual(a.sum(),1.,places=13)
        self.assertAlmostEqual(sum(np.arange(1,13)*a),1.5883600890562777,places=12)
        self.assertAlmostEqual(a[3:].sum(),.05911321879836594,places=12)
        self.assertTrue(np.all(a>=0))

    def test_coordinate_bound_gives_nonzero_JS_lower_bound(self):
        ref=np.array([[.2,.8],[.3,.7]])
        # Only first coord infeasible: target binary JSD is a provable lower bound.
        target=np.array([[.8,.2],[.3,.7]])
        lo=np.zeros((2,2));hi=np.ones((2,2));hi[0,0]=.7
        r=summarize_bounds({'lower':lo.tolist(),'upper':hi.tolist()},target)
        self.assertEqual(r['proved_unattainable_class_rows'],[0])
        self.assertGreater(r['JSD_lower_bound_per_class'][0],0.)
        self.assertAlmostEqual(r['JSD_lower_bound_per_class'][1],0.,places=14)

    def test_relaxed_bound_contains_exact_origin_excluded_policy(self):
        x,y=self.x[:18],self.y[:18]
        g=ExactGraph(x,y,self.spec['gamma'],np.arange(len(x)),x[:1],beta=.2,lam=1.,horizon=4)
        z=torch.tensor(np.random.default_rng(20261107).normal(size=(g.d+1,g.d)),dtype=DT)
        tight=center_bounds(g,y)
        relaxed=origin_return_relaxed_bounds(x,y,self.spec['gamma'],coefficients(.2,1.,4))
        tl,tu=np.asarray(tight['lower_center_probabilities']),np.asarray(tight['upper_center_probabilities'])
        rl,ru=np.asarray(relaxed['lower']),np.asarray(relaxed['upper'])
        for c in range(g.k):
            v=g.center_train(z,np.flatnonzero(y==c)).detach().numpy()
            self.assertTrue(np.all(v>=tl[c]-1e-11))
            self.assertTrue(np.all(v<=tu[c]+1e-11))
            self.assertTrue(np.all(v>=rl[c]-1e-11))
            self.assertTrue(np.all(v<=ru[c]+1e-11))

    def test_full_data_Bayes_targets_unreachable_for_short_depths(self):
        # Frozen training Bayes target. Even the return-allowed relaxation on
        # all 300 reference samples has a lower diagonal than Bayes target.
        target=self.target
        for H in (1,2,3,4):
            alpha=coefficients(.4,.8,H)
            bounds=origin_return_relaxed_bounds(self.x,self.y,self.spec['gamma'],alpha)
            diag=np.diag(np.asarray(bounds['upper']))
            self.assertTrue(np.all(np.diag(target)>diag), (H,diag))


if __name__=='__main__':unittest.main()
