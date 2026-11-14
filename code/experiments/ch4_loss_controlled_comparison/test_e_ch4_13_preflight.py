"""E-CH4-13 independent lightweight preflight fixtures. No optimizer step, no Test."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code/experiments/benchmark_v1"))
sys.path.insert(0, str(ROOT / "code/policy_evaluator"))
sys.path.insert(0, str(ROOT / "code/data_generator"))
from engine import DT, ExactMaskedWalk, jsd as torch_jsd, balanced
from policy_interface import evaluate_policy
from report_snapshot import standard_metrics
from snapshot_pipeline import save_snapshot, load_snapshot, array_fingerprint
from dataset_adapter import dataset_train_val
from generate import load_config
from metrics import center_metrics, jsd, sqrt_jsd, kl_divergence

def sample_graph():
    # Four nodes with two distinct classes; enough feasible destinations after
    # excluding current node and walk origin at every hop.
    x = np.array([[0., 0.], [1., 1.], [2., 2.],
                  [3., 3.], [4., 4.], [5., 5.]], float)
    y = np.array([0, 0, 0, 1, 1, 1])
    q = np.array([[.25, .75], [4.4, 4.2]])
    return ExactMaskedWalk(x, y, np.array([.3,.7]), horizon=3, query_x=q)

def sigmoid_loss(margin, tau):
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("positive finite tau required")
    return torch.sigmoid(-margin / tau)

def softplus_loss(scores, truth, tau):
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("positive finite tau required")
    target = scores.gather(1, truth[:,None])
    return tau * torch.logsumexp((scores-target)/tau, dim=1)

class ECH413Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_engine_simplex_mask_origin_and_manual_horizon_one(self):
        g = sample_graph()
        z = torch.zeros((g.d+1,g.d), dtype=DT)
        for h in (1,2,3):
            gh = ExactMaskedWalk(g.x,g.labels.numpy(),g.gamma,horizon=h,query_x=g.qx)
            actual = gh.represent_train(z,"shared").detach().numpy()
            self.assertEqual(actual.shape,(6,2))
            np.testing.assert_allclose(actual.sum(1),1.,atol=1e-12)
            self.assertTrue((actual>=-1e-14).all())
            if h==1:
                # one-hop reference: start feature probability 1/2; self excluded
                hand=np.zeros((g.n,g.k))
                for i in range(g.n):
                    for b in range(g.d):
                        hand[i] += .5*(g.base[b,i].numpy() @ g.yhot.numpy())
                np.testing.assert_allclose(actual,hand,atol=1e-12)
        # Forbidden origin (and current position) is zero in vertical kernel.
        for i in range(g.n):
            self.assertTrue(np.allclose(g.base[:,i,i].numpy(),0))
        # Missing query feature must be ignored by its start distribution.
        missing=np.array([[np.nan, .5]],float)
        qg=ExactMaskedWalk(g.x,g.labels.numpy(),g.gamma,horizon=1,query_x=missing)
        q1=qg.represent_query(z,"shared").detach().numpy()
        z2=z.clone();z2[g.d,0]=200.
        q2=qg.represent_query(z2,"shared").detach().numpy()
        np.testing.assert_allclose(q1,q2,atol=1e-12)
        np.testing.assert_allclose(q1.sum(1),1.,atol=1e-12)

    def test_two_step_manual_expected_walk_and_feature_self_loop(self):
        # Constant logits favor staying on the same feature. Exact walk must
        # agree with a manual origin-excluded Markov recurrence.
        g=sample_graph()
        z=torch.zeros((g.d+1,g.d),dtype=DT)
        z[0,0]=1.2; z[1,1]=1.4
        actual=g.represent_train(z,"shared").detach().numpy()
        alpha=g.alpha.numpy()
        base=g.base.numpy()
        for origin in (0,3):
            # At first step select feature uniformly.
            state=np.zeros((g.n,g.d))
            for f in range(g.d):state[:,f]=.5*base[f,origin,:]
            hand=np.zeros(g.k)
            for t in range(g.horizon):
                hand += alpha[t] * (state.sum(1) @ g.yhot.numpy())
                if t==g.horizon-1: break
                nxt=np.zeros_like(state)
                for node in range(g.n):
                    for prev in range(g.d):
                        if state[node,prev]==0:continue
                        weights=np.array([np.exp(float(z[prev,b])) for b in range(g.d)])
                        weights/=weights.sum()
                        for b in range(g.d):
                            destinations=base[b,node].copy()
                            destinations[origin]=0.
                            s=destinations.sum()
                            if s: nxt[:,b]+=state[node,prev]*weights[b]*destinations/s
                state=nxt
            np.testing.assert_allclose(actual[origin],hand,atol=1e-11)

    def test_sigmoid_and_softplus_gradient_finite_difference(self):
        tau=.005
        m=torch.tensor([-.023,.007,.055],dtype=DT,requires_grad=True)
        val=sigmoid_loss(m,tau).mean()
        val.backward()
        expected=-(torch.sigmoid(-m/tau)*(1-torch.sigmoid(-m/tau)))/tau/3
        torch.testing.assert_close(m.grad,expected,atol=1e-12,rtol=1e-12)
        eps=1e-7
        for i in range(3):
            mm=m.detach().clone();mm[i]+=eps
            mp=m.detach().clone();mp[i]-=eps
            fd=(sigmoid_loss(mm,tau).mean()-sigmoid_loss(mp,tau).mean())/(2*eps)
            self.assertAlmostEqual(float(fd),float(m.grad[i]),delta=1e-5)
        scores=torch.tensor([[.80,.78,.35],[.15,.64,.48]],dtype=DT,requires_grad=True)
        y=torch.tensor([0,1])
        value=softplus_loss(scores,y,tau).mean()
        value.backward()
        for i,j in ((0,0),(0,1),(1,2)):
            plus=scores.detach().clone();minus=plus.clone()
            plus[i,j]+=eps;minus[i,j]-=eps
            fd=(softplus_loss(plus,y,tau).mean()-softplus_loss(minus,y,tau).mean())/(2*eps)
            self.assertAlmostEqual(float(fd),float(scores.grad[i,j]),delta=1e-6)
        for invalid in (0.,-1.,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                sigmoid_loss(m,invalid)
            with self.assertRaises(ValueError):
                softplus_loss(scores,y,invalid)
        self.assertTrue(torch.isfinite(sigmoid_loss(torch.tensor([-1e3,1e3]),tau)).all())

    def test_loss_gradient_to_policy_and_tie_non_differentiability_boundary(self):
        g=sample_graph()
        z=torch.nn.Parameter(torch.zeros((g.d+1,g.d),dtype=DT))
        out=g.train_outputs(z,"shared")
        # Inspect tau sensitivity and gradients in the actual graph.
        for which in ("sigmoid","softplus"):
            z.grad=None
            out=g.train_outputs(z,"shared")
            v=(balanced(sigmoid_loss(out.margin,.005),g.labels,g.k) if which=="sigmoid"
               else balanced(softplus_loss(out.scores,g.labels,.005),g.labels,g.k))
            self.assertTrue(bool(torch.isfinite(v)))
            v.backward()
            self.assertTrue(bool(torch.isfinite(z.grad).all()))
            self.assertGreater(float(z.grad.abs().sum()),0.)
        # An exact tie has a deterministic hard decision but nonunique max
        # derivative: do not pretend derivative at the switching surface exists.
        reps=np.full((2,2,2),.5)
        centers=np.full((2,2),.5)
        scored=standard_metrics(reps,centers,np.array([0,1]),.005,split="train")
        self.assertEqual(scored["tie_count"],2)
        self.assertAlmostEqual(scored["balanced_accuracy"],.5)
        self.assertAlmostEqual(scored["hard_margin_error_conservative"],1.)

    def test_support_aware_kl_js_averaging_and_center_geometry(self):
        p=np.array([1.,0.]);q=np.array([0.,1.])
        self.assertTrue(np.isposinf(kl_divergence(p,q)))
        self.assertTrue(np.isfinite(np.sum(p*(np.log(np.maximum(p,1e-12))-np.log(np.maximum(q,1e-12))))))
        self.assertAlmostEqual(float(jsd(p,q)),np.log(2.),places=12)
        self.assertAlmostEqual(float(sqrt_jsd(p,q)),np.sqrt(np.log(2.)),places=12)
        reps=np.array([[.95,.05],[.5,.5],[.05,.95],[.5,.5]])
        y=np.array([0,0,1,1])
        centers=np.array([[.725,.275],[.275,.725]])
        by=np.broadcast_to(reps[:,None,:],(4,2,2)).copy()
        rep=standard_metrics(by,centers,y,.005,split="train")
        ownjs=np.array([jsd(reps[i],centers[y[i]]) for i in range(4)])
        self.assertAlmostEqual(rep["R_JS"],float(ownjs.mean()),places=12)
        self.assertAlmostEqual(rep["R_sqrt_JS"],float(np.sqrt(ownjs).mean()),places=12)
        # With unequal distances, mean sqrt differs from sqrt mean.
        self.assertGreater(abs(float(np.sqrt(ownjs).mean()-np.sqrt(ownjs.mean()))),1e-5)
        g=center_metrics(centers)
        for key in ("T","BT","TNL","BTNL","I"):
            self.assertAlmostEqual(rep[key],g[key],places=12)
        self.assertEqual(rep["confusion_true_rows_pred_cols"],[[2,0],[0,2]])
        self.assertAlmostEqual(rep["balanced_accuracy"],1.)

    def test_prediction_independent_of_validation_labels_and_free_node(self):
        g=sample_graph()
        data={"ref_x":g.x,"ref_y":g.labels.numpy(),"query_x":g.qx}
        for family,shape in (("shared",(g.d+1,g.d)),("class",(g.k,g.d+1,g.d)),
                             ("node",(g.n,g.d+1,g.d)),("node_class",(g.k,g.n,g.d+1,g.d))):
            output=evaluate_policy(data,family,torch.zeros(shape,dtype=DT),
                                   gamma=g.gamma,horizon=3)
            expected=(family in ("shared","class"))
            self.assertEqual(output["validation_status"]=="available",expected)
            if expected:
                centers=output["centers"]
                preds=output["validation_by_hypothesis"]
                self.assertEqual(preds.shape,(len(g.qx),g.k,g.k))
                self.assertTrue(np.allclose(preds.sum(-1),1))
                np.testing.assert_allclose(centers.sum(1),1)
                # No validation labels accepted by the evaluator.
                self.assertNotIn("query_y",data)
                np.testing.assert_array_equal(preds, evaluate_policy(
                    data,family,torch.zeros(shape,dtype=DT),
                    gamma=g.gamma,horizon=3)["validation_by_hypothesis"])
            else:
                self.assertEqual(output["validation_by_hypothesis"].shape[0],0)

    def test_snapshot_roundtrip_integrity_and_seed_split(self):
        cfg=ROOT/"code/data_generator/config.json"
        spec=load_config(cfg)
        spec["seed"]=20261102
        data=dataset_train_val(spec)
        tx,ty=data["train"];vx,vy=data["validation"]
        ids=np.concatenate([np.flatnonzero(ty==c)[:4] for c in range(len(spec["pi"]))])
        self.assertEqual(len(np.unique(ids)),len(ids))
        self.assertNotEqual(array_fingerprint(tx[ids]),array_fingerprint(vx))
        graph=ExactMaskedWalk(tx[ids],ty[ids],spec["gamma"],horizon=2,query_x=vx)
        z=torch.zeros((graph.d+1,graph.d),dtype=DT)
        with tempfile.TemporaryDirectory() as td:
            save_snapshot(td,graph,z,"shared",reference_ids=ids,
                          config_path=cfg,
                          config_reference="code/data_generator/config.json",
                          data_seed=20261102,reference_seed=20261107,
                          objective="preflight_uniform",tau=.005,beta=.4,lam=.8,horizon=2,
                          validation_x=vx)
            model,arrays=load_snapshot(td)
            self.assertEqual(model["data"]["data_seed"],20261102)
            np.testing.assert_array_equal(arrays["reference_ids"],ids)
            self.assertEqual(array_fingerprint(tx[ids]),
                             model["data"]["train_reference_x_sha256"])
            path=Path(td)/"distributions.npz"
            path.write_bytes(path.read_bytes()+b"altered")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                load_snapshot(td)

    def test_metric_contract_coverage_and_no_training(self):
        contract=json.loads((ROOT/"code/experiments/COMMON_METRICS_CONTRACT.json").read_text("utf8"))
        self.assertEqual(contract["schema"],"chapter4-common-evaluation-contract-v2")
        ids={g["group"] for g in contract["metric_groups"]}
        self.assertEqual(ids,{"density","center_geometry","classification",
                              "classification_surrogates","experiment_specific"})
        self.assertIn("undefined_or_infinite",contract["calculation"]["unavailable_states"])

if __name__=="__main__":
    unittest.main()
