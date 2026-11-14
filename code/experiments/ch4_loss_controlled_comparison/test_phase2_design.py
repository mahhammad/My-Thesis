"""Phase2 no-training technical/contract fixtures (NEVER call fit_once/execute)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from engine import DT,ExactMaskedWalk
from policy_interface import evaluate_policy
from report_snapshot import standard_metrics
from phase2_protocol import (SEEDS,FAMILIES,LOSSES,N_UPDATES,grid,validate_grid,
    reference_ids,plan_manifest,charge_attempt,new_ledger,atomic_new_json,
    expected_data_gate,evidence_gate)
from phase2_runner import canonical_loss,gradients_at_policy,numeric_nested,make_coverage
from phase2_data_preflight import inspect

def graph():
    x=np.array([[-2.,-1.],[-1.,-.5],[0.,0.],[.3,.2],[2.,1.],[3.,1.5]],float)
    y=np.array([0,0,1,1,2,2])
    q=np.array([[-1.5,-1.],[.4,.5],[2.6,1.4]],float)
    return ExactMaskedWalk(x,y,np.array([.8,1.2]),horizon=3,query_x=q)

class Phase2DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_grid_exact_budget_and_dry_run(self):
        q=grid()
        self.assertEqual(len(q),12)
        self.assertEqual(len(set(q)),12)
        self.assertEqual(validate_grid(q),q)
        for x in (q[:-1],q+(q[-1],),q[::-1]):
            with self.assertRaises(ValueError):validate_grid(x)
        for seed in SEEDS:
            for f in FAMILIES:
                self.assertEqual({x.loss for x in q if (x.seed,x.family)==(seed,f)},set(LOSSES))
        m=plan_manifest()
        self.assertEqual((m["fit_cap"],m["updates_per_fit"],m["n_train"]),(12,25,300))
        self.assertFalse(m["test_accessed"])
        self.assertEqual(m["checkpoint_selection"],"fixed_step_25")
        self.assertEqual(evidence_gate(execute=False),"DRY_RUN_NO_TRAINING")

    def test_fail_closed_missing_technical_qa_manager_approval_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"gate.json"
            with self.assertRaises(PermissionError):
                evidence_gate(execute=True,approval_file=f)
            f.write_text(json.dumps({"status":"PHASE2_EXECUTION_AUTHORIZED"}))
            with self.assertRaises(PermissionError):
                evidence_gate(execute=True,approval_file=f,cli_token="secret")
            fake={"status":"PHASE2_EXECUTION_AUTHORIZED",
                "technical_audit_status":"PASS","phase2_tests":"PASS",
                "technical_audit_url":"https://github.com/mahhammad/My-Thesis/issues/80#issuecomment-1",
                "manager_authorization_url":"https://github.com/mahhammad/My-Thesis/issues/81#issuecomment-2",
                "ci_url":"https://github.com/mahhammad/My-Thesis/actions/runs/42",
                "source_sha":"a"*40,"one_time_gate_token":"secret",
                "used":True,"training_authorized":True}
            f.write_text(json.dumps(fake))
            with self.assertRaises(PermissionError):
                evidence_gate(execute=True,approval_file=f,cli_token="secret")
            fake["used"]=False
            fake["training_authorized"]=False
            f.write_text(json.dumps(fake))
            with self.assertRaises(PermissionError):
                evidence_gate(execute=True,approval_file=f,cli_token="secret")
        committed=json.loads((HERE/"phase2_execution_gate.json").read_text())
        self.assertFalse(committed["training_authorized"])
        self.assertEqual(committed["technical_audit_status"],"NOT_RECORDED")
        self.assertEqual(committed["final_scientific_audit"],"DEFERRED_UNTIL_AFTER_PHASE2")

    def test_immutable_outputs_and_attempt_charges(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"new"/"x.json"
            self.assertEqual(len(atomic_new_json(p,{"a":1})),64)
            with self.assertRaises(FileExistsError):atomic_new_json(p,{"a":2})
        ledger=new_ledger()
        for job in grid():charge_attempt(ledger,job)
        self.assertEqual(len(ledger["attempts"]),12)
        with self.assertRaises(RuntimeError):charge_attempt(ledger,grid()[0])

    def test_all_ordered_ids_and_data_contract_reject_test(self):
        y=np.array([0]*75+[1]*135+[2]*90)
        np.testing.assert_array_equal(reference_ids(y),np.arange(300))
        with self.assertRaises(ValueError):reference_ids(y[:299])
        with self.assertRaises(ValueError):reference_ids(np.zeros(300,dtype=int))
        d={}
        for s in SEEDS:
            d[str(s)]={"n_train":300,"n_validation":90,"n_classes":3,"n_features":5,
                "reference_ids":list(range(300)),"config_sha256":"a"*64,
                "train_x_sha256":"b"*64,"train_y_sha256":"c"*64,
                "validation_x_sha256":"d"*64,"test_accessed":False}
        self.assertTrue(expected_data_gate(d))
        d["20261103"]["test_accessed"]=True
        with self.assertRaises(ValueError):expected_data_gate(d)

    def test_three_seeds_fingerprints_no_test_and_masks(self):
        with mock.patch.dict(os.environ,{"OPENBLAS_CORETYPE":"generic","OPENBLAS_NUM_THREADS":"1"}):
            data=inspect()
        self.assertEqual(set(data["datasets"]),set(map(str,SEEDS)))
        self.assertFalse(data["test_accessed"])
        for rec in data["datasets"].values():
            self.assertEqual(sum(rec["train_class_counts"]),300)
            self.assertEqual(sum(rec["validation_class_counts"]),90)
            self.assertEqual(rec["reference_ids"],list(range(300)))
            for k in ("train_x_sha256","train_y_sha256","validation_x_sha256",
                      "mask_train_sha256","mask_validation_sha256"):
                self.assertEqual(len(rec[k]),64)


    def test_numpy_only_frozen_fingerprint_matches_canonical(self):
        from phase2_frozen_data import array_fingerprint as frozen_hash,load as frozen_load,INPUTS
        from snapshot_pipeline import array_fingerprint as canonical_hash
        arr=np.array([[0.,np.nan],[-1.5,2.25]],dtype=np.float64)
        self.assertEqual(frozen_hash(arr),canonical_hash(arr))
        if (INPUTS/"manifest.json").exists():
            for seed in SEEDS:
                (tx,ty),(vx,vy)=frozen_load(seed)
                self.assertEqual((tx.shape,vx.shape),((300,5),(90,5)))
                self.assertEqual(frozen_hash(tx),canonical_hash(tx))
                self.assertEqual(frozen_hash(vx),canonical_hash(vx))

    def test_gradient_fd_tau_and_direction_both_families(self):
        g=graph()
        for family in FAMILIES:
            shape=(g.d+1,g.d) if family=="shared" else (g.k,g.d+1,g.d)
            for obj in LOSSES:
                with self.subTest(family=family,loss=obj):
                    z=torch.zeros(shape,dtype=DT,requires_grad=True)
                    v=canonical_loss(g,z,family,obj)
                    grad=torch.autograd.grad(v,z)[0]
                    self.assertTrue(bool(torch.isfinite(grad).all()))
                    eps=1e-6
                    for index in (tuple(0 for _ in shape),tuple(s-1 for s in shape)):
                        zz=z.detach().clone()
                        zz[index]+=eps
                        pos=float(canonical_loss(g,zz,family,obj))
                        zz[index]-=2*eps
                        neg=float(canonical_loss(g,zz,family,obj))
                        self.assertAlmostEqual(float(grad[index]),(pos-neg)/(2*eps),delta=2e-5)
            z=torch.zeros(shape,dtype=DT,requires_grad=True)
            alignment=gradients_at_policy(g,z,family)
            self.assertTrue(np.isfinite(alignment["sigmoid_gradient_l2"]))
            self.assertTrue(np.isfinite(alignment["softplus_gradient_l2"]))
            if alignment["cosine"] is not None:
                self.assertGreaterEqual(alignment["cosine"],-1.0000001)
                self.assertLessEqual(alignment["cosine"],1.0000001)

    def test_class_candidate_inference_without_validation_labels(self):
        g=graph()
        z=torch.zeros((g.k,g.d+1,g.d),dtype=DT)
        base={"ref_x":g.x,"ref_y":g.labels.numpy(),"query_x":g.qx}
        p=evaluate_policy(base,"class",z,gamma=g.gamma,horizon=3)
        self.assertEqual(p["validation_status"],"available")
        self.assertEqual(p["validation_by_hypothesis"].shape,(3,3,3))
        np.testing.assert_allclose(p["validation_by_hypothesis"].sum(-1),1,atol=1e-12)
        self.assertNotIn("query_y",base)
        np.testing.assert_allclose(p["centers"].sum(-1),1,atol=1e-12)
        for y in (np.array([0,1,2]),np.array([2,0,1])):
            out=standard_metrics(p["validation_by_hypothesis"],p["centers"],y,.005,split="validation")
            self.assertTrue(0<=out["balanced_accuracy"]<=1)
            np.testing.assert_array_equal(p["validation_by_hypothesis"],
                evaluate_policy(base,"class",z,gamma=g.gamma,horizon=3)["validation_by_hypothesis"])

    def test_support_aware_kl_and_metric_coverage(self):
        self.assertTrue(numeric_nested({"0.1":0.,"0.5":1.}))
        self.assertFalse(numeric_nested({"0.1":np.inf}))
        g=graph()
        centers=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
        reps=np.broadcast_to(np.eye(3)[np.array([0,0,1,1,2,2])][:,None,:],(6,3,3)).copy()
        rep=standard_metrics(reps,centers,g.labels.numpy(),.005,split="train")
        contract=json.loads((ROOT/"code/experiments/COMMON_METRICS_CONTRACT.json").read_text())
        r={"train":rep,"validation":rep}
        cov=make_coverage(r,contract)
        self.assertEqual(cov["test"]["status"],"not_applicable")
        self.assertEqual(cov["train.R_KL"]["status"],"measured")
        self.assertEqual(cov["validation.margin_quantiles"]["status"],"measured")
        # Forward KL with unsupported q=0 is infinite, never clipped.
        zeros=np.array([[1.,0.,0.]])
        bad=np.array([[0.,1.,0.]])
        from report_snapshot import kl_forward
        self.assertTrue(np.isposinf(kl_forward(zeros,bad)[0]))

if __name__=="__main__":unittest.main()
