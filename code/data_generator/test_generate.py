"""Regression tests for generated data and analytic oracles (stdlib unittest)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate import (bayes_posterior, finite_one_step, generate, load_config,
                      population_one_step)


BASE = Path(__file__).with_name("config.json")


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_config(BASE)

    def test_marginal_rank_and_noise(self):
        self.assertEqual(self.spec["noise"], [2, 4])
        self.assertEqual(np.argsort(-self.spec["importance"]).tolist()[:3], [0, 3, 1])
        self.assertTrue(np.all(self.spec["importance"][[2, 4]] == 0))

    def test_correlations_and_covariances(self):
        self.assertAlmostEqual(self.spec["corr"][0,1], .35)
        self.assertTrue(all(np.linalg.eigvalsh(c).min()>0 for c in self.spec["cov"]))
        rng=np.random.default_rng(3)
        x=rng.multivariate_normal(self.spec["mu"][1],self.spec["cov"][1],size=20000)
        self.assertAlmostEqual(float(np.corrcoef(x[:,0],x[:,1])[0,1]), .35,delta=.03)

    def test_posterior_normalized_and_all_missing(self):
        x=np.array([[np.nan]*5,[0.,1.,np.nan,0.,np.nan]])
        p=bayes_posterior(self.spec,x)
        np.testing.assert_allclose(p[0],self.spec["pi"],rtol=0,atol=1e-14)
        np.testing.assert_allclose(p.sum(axis=1),1.,rtol=0,atol=1e-14)

    def test_joint_correlation_changes_posterior(self):
        x=np.array([[1.,.5,np.nan,np.nan,np.nan]])
        p1=bayes_posterior(self.spec,x)
        independent=copy.deepcopy(self.spec)
        independent["corr"]=np.eye(5)
        independent["cov"]=np.array([np.diag(s*s) for s in self.spec["sd"]])
        p2=bayes_posterior(independent,x)
        self.assertGreater(float(np.max(np.abs(p1-p2))),1e-3)

    def test_one_step_noise_matches_prior_and_missing_nan(self):
        x=np.array([[.25,-1.,0.,.4,.6],[np.nan,0.,1.,0.,0.]])
        q=population_one_step(self.spec,x)
        for a in self.spec["noise"]:
            np.testing.assert_allclose(q[:,a,:],np.broadcast_to(self.spec["pi"],(2,3)),atol=1e-14)
        self.assertTrue(np.isnan(q[1,0]).all())
        np.testing.assert_allclose(np.nansum(q[0],axis=-1),np.ones(5),atol=1e-14)

    def test_finite_q_excludes_self_and_respects_missing(self):
        ref=np.array([[0.,np.nan,1.,0.,0.],[0.5,1.,1.,0.,0.],[1.,2.,1.,0.,0.]])
        y=np.array([0,1,2])
        q=finite_one_step(self.spec,ref,ref,y,train_queries=True)
        self.assertEqual(q.shape,(3,5,3))
        self.assertAlmostEqual(q[0,0,0],0.)
        self.assertTrue(np.isnan(q[0,1]).all())
        np.testing.assert_allclose(q[0,0].sum(),1.)
        single=finite_one_step(self.spec,ref[:1],ref[:1],y[:1],train_queries=True)
        self.assertTrue(np.isnan(single).all())

    def test_reproducibility_and_mask(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            da=generate(self.spec,a)
            db=generate(self.spec,b)
            for split in da:
                xa,ma,ya=da[split]
                xb,mb,yb=db[split]
                np.testing.assert_array_equal(ya,yb)
                np.testing.assert_array_equal(ma,mb)
                np.testing.assert_array_equal(xa,xb)
                self.assertTrue(np.array_equal(np.isnan(xa),~ma))
                for prefix in ("oracle_","finite_q_"):
                    with np.load(Path(a)/f"{prefix}{split}.npz") as aa, np.load(Path(b)/f"{prefix}{split}.npz") as bb:
                        for key in aa.files:
                            np.testing.assert_array_equal(aa[key],bb[key])
            with np.load(Path(a)/"test.npz") as p:
                self.assertEqual(sorted(p.files),["mask","x","y"])
            meta=json.loads((Path(a)/"spec.json").read_text())
            self.assertEqual(meta["marginal_noise_features_zero_based"],[2,4])

    def test_invalid_correlation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=json.loads(BASE.read_text())
            cfg["correlation"][0][1]=0.99
            path=Path(d)/"bad.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,"symmetric"):
                load_config(path)
            cfg["correlation"]=np.ones((5,5)).tolist()
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,"positive definite"):
                load_config(path)

    def test_missing_percent_is_per_feature(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=json.loads(BASE.read_text())
            cfg["missing_percent"]=[0,100,0,100,0]
            path=Path(d)/"all.json"
            path.write_text(json.dumps(cfg))
            spec=load_config(path)
            x,mask,_=generate(spec,Path(d)/"data",finite_q=False)["train"]
            self.assertTrue(mask[:,[0,2,4]].all())
            self.assertFalse(mask[:,[1,3]].any())
            self.assertTrue(np.isnan(x[:,[1,3]]).all())


if __name__=="__main__":
    unittest.main(verbosity=2)
