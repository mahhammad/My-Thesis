"""Cheap, deterministic E08 controls; no Benchmark replay or historical Test."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/"code"/"experiments"/"benchmark_v1"))
from engine import ExactMaskedWalk, DT, coefficients
from walk_analysis import (
    BETA,LAM,H,DEPTHS,forward_steps,expectation,tv,analyze_depths,
    sample_paths
)
from run import alpha_diagnostics,graph_size

class TestE08Numerical(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # Two missing features per distinct node; every node retains feature 0.
        x=np.array([[0,np.nan],[.5,0],[1,1],[2,np.nan],
                    [2.5,0],[3,1]],dtype=float)
        y=np.array([0,1,2,0,1,2])
        q=np.array([[.1,np.nan],[np.nan,.2],[2.8,.9]])
        self.g=ExactMaskedWalk(x,y,np.array([.7,.9]),beta=BETA,
                               lam=LAM,horizon=H,query_x=q)
        self.y=y
        self.vy=np.array([0,1,2])
        self.z=torch.tensor([[.2,-.5],[-.7,.4],[.1,-.2]],dtype=DT)

    def test_capped_geometric_and_effective_depth(self):
        alpha=coefficients(BETA,LAM,H).numpy()
        info=alpha_diagnostics(alpha)
        self.assertAlmostEqual(sum(alpha),1,12)
        self.assertAlmostEqual(info["expected_realized_walk_length"],
                               (1-(1-BETA)**H)/BETA,12)
        self.assertAlmostEqual(info["expected_realized_walk_length"],2.49455804416,10)
        self.assertAlmostEqual(info["expected_weighted_step_index"],1.5883600890562775,10)
        self.assertAlmostEqual(info["truncation_epsilon"]["3"],.05911321879836594,12)
        self.assertEqual(info["truncation_epsilon"]["12"],0.)
        eps=np.asarray([info["truncation_epsilon"][str(r)] for r in DEPTHS])
        self.assertTrue(np.all(np.diff(eps)<=0))

    def test_independent_recursion_matches_benchmark_train_validation(self):
        a=self.g.alpha.numpy()
        with torch.no_grad():
            tr=forward_steps(self.g,self.z,"train")
            va=forward_steps(self.g,self.z,"validation")
            expected_train=expectation(tr,a,H)
            expected_val=expectation(va,a,H)
            original_train=self.g.represent_train(self.z,"shared").numpy()
            original_val=self.g.represent_query(self.z,"shared").numpy()
        np.testing.assert_allclose(expected_train,original_train,rtol=0,atol=2e-12)
        np.testing.assert_allclose(expected_val,original_val,rtol=0,atol=2e-12)
        np.testing.assert_allclose(tr.sum(-1),1.,rtol=0,atol=2e-12)
        np.testing.assert_allclose(va.sum(-1),1.,rtol=0,atol=2e-12)

    def test_origin_exclusion_self_exclusion_and_missing_mask(self):
        g=self.g
        np.testing.assert_array_equal(g.base[1,0].numpy(),np.zeros(g.n))
        np.testing.assert_array_equal(g.base[1,3].numpy(),np.zeros(g.n))
        np.testing.assert_array_equal(g.qweights[0,1].numpy(),np.zeros(g.n))
        for b in range(g.d):
            self.assertTrue(np.allclose(np.diag(g.base[b].numpy()),0))
        for origin in range(g.n):
            p=np.asarray(g.base[:,origin,:])
            self.assertEqual(float(np.max(np.abs(p[:,origin]))),0.)
        # Values with all features missing must fail rather than be treated as uniform.
        a=ExactMaskedWalk(g.x,g.labels.numpy(),g.gamma,
                          query_x=np.full((1,g.d),np.nan))
        with self.assertRaises(ValueError):
            forward_steps(a,self.z,"validation")
        with self.assertRaises(ValueError):
            forward_steps(g,torch.zeros((g.d,g.d),dtype=DT),"train")

    def test_remainder_and_center_bounds_for_every_depth(self):
        a=self.g.alpha.numpy()
        tr=forward_steps(self.g,self.z,"train")
        va=forward_steps(self.g,self.z,"validation")
        r=analyze_depths(tr,va,a,self.y,self.vy)
        self.assertEqual([i["r"] for i in r],list(DEPTHS))
        self.assertEqual(r[-1]["epsilon_tail"],0.)
        for i in r:
            self.assertTrue(i["bound_verified"])
            for key in ("train_max_tv","validation_max_tv",
                        "class_centers_max_tv"):
                self.assertLessEqual(i[key],i["epsilon_tail"]+1e-9)

    def test_independent_path_sampling_and_seed_reproducibility(self):
        a=self.g.alpha.numpy()
        exact=expectation(forward_steps(self.g,self.z,"train"),a,H)
        paths,lengths=sample_paths(self.g,self.z,"train",0,20261121,2048)
        paths2,lengths2=sample_paths(self.g,self.z,"train",0,20261121,2048)
        np.testing.assert_array_equal(paths,paths2)
        np.testing.assert_array_equal(lengths,lengths2)
        self.assertTrue(np.all((lengths>=1)&(lengths<=H)))
        np.testing.assert_allclose(paths.sum(-1),1,atol=1e-12,rtol=0)
        self.assertLess(float(tv(paths.mean(0),exact[0])),.10)

    def test_graph_counts_and_depth_boundary(self):
        size=graph_size(self.g)
        self.assertEqual(size["m_sample_feature_nodes"],10)
        self.assertEqual(size["directed_vertical_possible"],6*5+4*3)
        self.assertEqual(size["horizontal_possible"],18)
        self.assertTrue(size["horizontal_feature_self_choices_included"])
        with self.assertRaises(ValueError):
            forward_steps(self.g,self.z,"train",0)
        with self.assertRaises(ValueError):
            forward_steps(self.g,self.z,"train",H+1)
        with self.assertRaises(ValueError):
            forward_steps(self.g,self.z,"test")

if __name__=="__main__":
    unittest.main()
