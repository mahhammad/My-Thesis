"""P0 GitHub authorization, crash evidence and full synthetic reducer fixtures.
ZERO optimizer steps. Do not open Test or modify historical raw.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

import numpy as np

import phase2_github_gate as gate
import phase2_runner as runner
import phase2_aggregate as aggregate
from phase2_protocol import grid,LOSSES,SEEDS,FAMILIES

SHA="a"*40
BRANCH="experiment/ch4-loss-controlled-comparison-01"
def comment(id,body,created="2026-11-14T12:00:00Z",author="mahhammad"):
    return {"id":id,"html_url":f"https://github.com/mahhammad/My-Thesis/issues/81#issuecomment-{id}",
        "body":body,"created_at":created,
        "user":{"login":author},"author_association":"OWNER"}
def audit():
    return comment(1,"TECHNICAL_PREFLIGHT_FIXES_ACCEPTED candidate_sha="+SHA+" max_fits=12")
def approve():
    return comment(2,"PHASE2_EXECUTION_AUTHORIZED candidate_sha="+SHA+" max_fits=12",
                   created="2026-11-14T13:00:00Z")
def ci(sha=SHA,conclusion="success",workflow="repo/.github/workflows/e-ch4-13-phase2-design.yml"):
    return {"id":22,"html_url":"https://github.com/mahhammad/My-Thesis/actions/runs/22",
            "head_sha":sha,"status":"completed","conclusion":conclusion,
            "path":workflow,"event":"push","run_number":2}
def fake_api(comments=None,run=None,head=SHA):
    def api(method,path,token,body=None):
        if path.endswith("/branches/"+BRANCH):
            return {"commit":{"sha":head}}
        if "/issues/81/comments" in path:
            return [audit(),approve()] if comments is None else comments
        if "/actions/workflows/" in path:
            return {"total_count":1,"workflow_runs":[ci() if run is None else run]}
        raise AssertionError("Unexpected online endpoint "+path)
    return api
class GitHubGateTests(unittest.TestCase):
    def test_approved_same_sha_online_gate(self):
        result=gate.validate_online(SHA,BRANCH,"stub",api_fn=fake_api())
        self.assertEqual(result["source_sha"],SHA)
        self.assertEqual(result["fit_cap"],12)

    def test_missing_forged_or_wrong_order_comments_deny(self):
        variants=([],[approve()],
            [audit(),comment(2,approve()["body"],author="stranger")],
            [comment(1,audit()["body"],created="2026-11-14T14:00:00Z"),approve()],
            [comment(1,audit()["body"]+" candidate_sha="+"b"*40),approve()])
        for comments in variants:
            with self.subTest(comments=comments):
                with self.assertRaises(PermissionError):
                    gate.validate_online(SHA,BRANCH,"stub",api_fn=fake_api(comments=comments))

    def test_wrong_sha_ci_failure_and_wrong_workflow_all_deny(self):
        for record in (ci(sha="b"*40),ci(conclusion="failure"),
                       ci(workflow="repo/.github/workflows/other.yml")):
            with self.subTest(run=record):
                with self.assertRaises(PermissionError):
                    gate.validate_online(SHA,BRANCH,"stub",api_fn=fake_api(run=record))
        with self.assertRaises(PermissionError):
            gate.validate_online(SHA,"main","stub",api_fn=fake_api())
        with self.assertRaises(PermissionError):
            gate.validate_online(SHA,BRANCH,"stub",api_fn=fake_api(head="b"*40))

    def test_reservation_reused_ref_does_not_issue_second_token(self):
        calls=[]
        def api_fn(method,path,token,body=None):
            calls.append((method,path,body))
            if len(calls)>1:
                raise urllib.error.HTTPError("https://api.github.com",422,
                    "Reference already exists",{},None)
            return {"ref":body["ref"]}
        self.assertIn(SHA,gate.reserve_once(SHA,"stub",api_fn=api_fn))
        with self.assertRaises(PermissionError):
            gate.reserve_once(SHA,"stub",api_fn=api_fn)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0][2]["ref"],calls[1][2]["ref"])

    def test_token_not_written_on_failed_external_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"gate.json"
            with self.assertRaises(PermissionError):
                gate.issue_gate(SHA,BRANCH,"fake",f,
                    validate=lambda *a:(_ for _ in ()).throw(PermissionError("no comment")),
                    reserve=lambda *a: self.fail("reservation must never happen"))
            self.assertFalse(f.exists())

    def test_online_success_then_once_reservation_generates_bound_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"ephemeral.json"
            with mock.patch.dict(os.environ,{"GITHUB_RUN_ID":"321"}):
                g=gate.issue_gate(SHA,BRANCH,"stub",p,
                    validate=lambda *a:{"auditor_comment":audit()["html_url"],
                        "manager_comment":approve()["html_url"],
                        "ci_url":ci()["html_url"]},
                    reserve=lambda *a:"refs/tags/e-ch4-13-phase2-budget-spent-"+SHA)
            self.assertTrue(p.exists())
            self.assertEqual(g["github_run_id"],"321")
            self.assertEqual(len(g["one_time_gate_token"]),64)
            with self.assertRaises(FileExistsError):
                gate.issue_gate(SHA,BRANCH,"stub",p,
                    validate=lambda *a:{},reserve=lambda *a:"tag")

class CrashSafetyTests(unittest.TestCase):
    def authorize(self,tmp):
        d={"status":"PHASE2_EXECUTION_AUTHORIZED",
            "technical_audit_status":"PASS","phase2_tests":"PASS",
            "technical_audit_url":audit()["html_url"],
            "manager_authorization_url":approve()["html_url"],
            "ci_url":ci()["html_url"],"source_sha":SHA,
            "one_time_gate_token":"synthetic_token","used":False,
            "training_authorized":True,
            "reservation_ref":"refs/tags/e-ch4-13-phase2-budget-spent-"+SHA,
            "github_run_id":"321"}
        p=Path(tmp)/"synthetic_ephemeral.json"
        p.write_text(json.dumps(d))
        return p
    def mocks(self):
        # No real optimizer; test uses REAL frozen input fingerprint loading.
        env={"GITHUB_ACTIONS":"true","GITHUB_RUN_ID":"321",
            "GITHUB_REF_NAME":BRANCH,
            "GITHUB_WORKFLOW":"E-CH4-13 phase2 authorized 12-fit batch",
            "E_CH4_13_PHASE2_OPT_IN":"I_ACCEPT_12_FIT_CAP",
            "OPENBLAS_CORETYPE":"generic","OPENBLAS_NUM_THREADS":"1",
            "GH_TOKEN":"synthetic_never_network"}
        remote={"manager_comment":approve()["html_url"],
                "auditor_comment":audit()["html_url"],"ci_url":ci()["html_url"]}
        return env,remote

    def test_synthetic_optimizer_exception_preserves_attempt_and_no_retry(self):
        env,remote=self.mocks()
        with tempfile.TemporaryDirectory() as tmp:
            path=self.authorize(tmp)
            dest=Path(tmp)/"raw"
            with (mock.patch.dict(os.environ,env),
                  mock.patch.object(np,"__version__","2.4.6"),
                  mock.patch.object(runner.torch,"__version__","2.15.0+cpu"),
                  mock.patch.object(runner,"verify_github_gate",return_value=remote),
                  mock.patch.object(runner,"github_api",return_value={
                    "ref":"refs/tags/e-ch4-13-phase2-budget-spent-"+SHA,
                    "object":{"sha":SHA}}),
                  mock.patch.object(runner,"ExactMaskedWalk",return_value=object()),
                  mock.patch.object(runner,"fit_once",side_effect=RuntimeError("synthetic failure")) as fit):
                with self.assertRaisesRegex(RuntimeError,"synthetic failure"):
                    runner.execute(dest,SHA,"synthetic_token",path)
                self.assertEqual(fit.call_count,1)
            self.assertTrue((dest/"attempt_charges"/(grid()[0].key()+".json")).exists())
            spent=json.loads((dest/"budget_checkpoints"/"attempt_01.json").read_text())
            self.assertEqual(spent["attempt_count"],1)
            failure=json.loads((dest/"failures"/(grid()[0].key()+".json")).read_text())
            self.assertTrue(failure["attempt_charged"])
            self.assertFalse(failure["test_accessed"])
            self.assertFalse((dest/"manifest.json").exists())
            self.assertNotIn("one_time_gate_token",
                json.loads((dest/"authorization_record.json").read_text()))
            with mock.patch.dict(os.environ,env):
                with self.assertRaises((FileExistsError,PermissionError)):
                    runner.execute(dest,SHA,"synthetic_token",path)

    def test_frozen_data_corruption_refuses_before_optimizer_or_attempt(self):
        env,remote=self.mocks()
        original=runner.load_frozen
        def corrupt(seed):
            (x,y),(vx,vy)=original(seed)
            x=x.copy()
            x[0,0]+=1.0
            return (x,y),(vx,vy)
        with tempfile.TemporaryDirectory() as tmp:
            path=self.authorize(tmp);dest=Path(tmp)/"raw"
            with (mock.patch.dict(os.environ,env),
                  mock.patch.object(np,"__version__","2.4.6"),
                  mock.patch.object(runner.torch,"__version__","2.15.0+cpu"),
                  mock.patch.object(runner,"verify_github_gate",return_value=remote),
                  mock.patch.object(runner,"github_api",return_value={
                    "ref":"refs/tags/e-ch4-13-phase2-budget-spent-"+SHA,
                    "object":{"sha":SHA}}),
                  mock.patch.object(runner,"load_frozen",side_effect=corrupt),
                  mock.patch.object(runner,"fit_once") as fit):
                with self.assertRaisesRegex(ValueError,"Frozen data fingerprint drift"):
                    runner.execute(dest,SHA,"synthetic_token",path)
                fit.assert_not_called()
            self.assertFalse(dest.exists())

    def test_full_synthetic_12_fit_runner_wiring_without_optimizer(self):
        """Full online gate -> frozen hashes -> ledger -> mock snapshot/report.
        No optimizer is constructed. This is an interface fixture only.
        """
        env,remote=self.mocks()
        with tempfile.TemporaryDirectory() as tmp:
            auth=self.authorize(tmp)
            dest=Path(tmp)/"raw"
            def synthetic_fit(graph,family,objective,checkpoint_dir=None):
                d=Path(checkpoint_dir)
                d.mkdir(parents=True,exist_ok=False)
                (d/"step_00.json").write_text(json.dumps({"mock":True}))
                trace=[{"update_completed":i,"gradient_l2":float(i+1),
                    "objective_loss":1.0/(i+1)} for i in range(26)]
                return np.zeros((6,5)).tolist(),trace,0.01
            def synthetic_snapshot(folder,*a,**kw):
                folder=Path(folder)
                folder.mkdir(parents=True,exist_ok=False)
                (folder/"model.json").write_text('{"synthetic":true}')
                (folder/"distributions.npz").write_bytes(b"mock fixture")
            def synthetic_report(folder,seed):
                return {"train":{"balanced_accuracy":0.8,"I":0.02,
                    "R_KL":0.01,"R_JS":0.005,"R_sqrt_JS":0.02},
                    "validation":{"status":"ok","balanced_accuracy":0.7,
                    "margin_quantiles":{"0.5":0.02}},
                    "source":{"historical_test_accessed":False}}
            with (mock.patch.dict(os.environ,env),
                  mock.patch.object(np,"__version__","2.4.6"),
                  mock.patch.object(runner.torch,"__version__","2.15.0+cpu"),
                  mock.patch.object(runner,"verify_github_gate",return_value=remote),
                  mock.patch.object(runner,"github_api",return_value={
                      "ref":"refs/tags/e-ch4-13-phase2-budget-spent-"+SHA,
                      "object":{"sha":SHA}}),
                  mock.patch.object(runner,"ExactMaskedWalk",return_value=object()),
                  mock.patch.object(runner,"fit_once",side_effect=synthetic_fit) as fit,
                  mock.patch.object(runner,"save_snapshot",side_effect=synthetic_snapshot),
                  mock.patch.object(runner,"load_snapshot",return_value=({"data":{}},{})),
                  mock.patch.object(runner,"fingerprint_match"),
                  mock.patch.object(runner,"report_from_frozen",side_effect=synthetic_report),
                  mock.patch.object(runner,"make_coverage",
                                    return_value={"test":{"status":"not_applicable"}})):
                out=runner.execute(dest,SHA,"synthetic_token",auth)
            self.assertEqual(fit.call_count,12)
            self.assertEqual(out["fit_count"],12)
            self.assertTrue((dest/"manifest.json").exists())
            self.assertTrue((dest/"metrics_summary.csv").exists())
            self.assertEqual(len(list((dest/"attempt_charges").glob("*.json"))),12)
            self.assertEqual(len(list((dest/"budget_checkpoints").glob("*.json"))),12)
            self.assertEqual(len(list((dest/"success_checkpoints").glob("*.json"))),12)
            for job in grid():
                folder=dest/job.key()
                for file in ("trace.json","metrics_full.json","coverage.json","run_evidence.json"):
                    self.assertTrue((folder/file).exists())
            self.assertFalse(out["test_accessed"])

class SyntheticReducerTests(unittest.TestCase):
    def test_all_12_complete_synthetic_pairs_reduce_without_optimizer(self):
        from phase2_frozen_data import load
        from snapshot_pipeline import array_fingerprint
        fixture={}
        rows=[]
        for j in grid():
            (tx,ty),(vx,yv)=load(j.seed)
            pred=yv.copy()
            if j.loss==LOSSES[1]:
                pred[:3]=(pred[:3]+1)%3
            cm=np.zeros((3,3),int)
            for a,b in zip(yv,pred):cm[int(a),int(b)]+=1
            ba=float(np.mean(np.diag(cm)/cm.sum(1)))
            fixture[j.key()]={
                "model":{"family":j.family,"objective":{"name":j.loss},
                   "data":{"data_seed":j.seed,
                     "train_reference_x_sha256":array_fingerprint(tx),
                     "validation_x_sha256":array_fingerprint(vx)}},
                "array":{"seed":j.seed,"loss":j.loss}}
            rows.append({"run_key":j.key(),"validation_ba":ba,
                **{z:"synthetic_hash" for z in ("model_sha256","distributions_sha256",
                    "trace_sha256","metrics_full_sha256","coverage_sha256")},
                "I":0.1,"R_KL":0.2,"R_JS":0.03,"R_sqrt_JS":0.15,
                "margin_validation_median":0.02})
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            (dest/"manifest.json").write_text(json.dumps({
                "fit_count":12,"test_accessed":False,"runs":rows}))
            def fake_snapshot(folder):
                e=fixture[Path(folder).name]
                return e["model"],e["array"]
            def fake_pred(arr):
                (tx,ty),(vx,yval)=load(arr["seed"])
                pred=yval.copy()
                if arr["loss"]==LOSSES[1]:pred[:3]=(pred[:3]+1)%3
                return pred,np.zeros((90,3))
            with (mock.patch.object(aggregate,"file_hash",return_value="synthetic_hash"),
                  mock.patch.object(aggregate,"load_snapshot",side_effect=fake_snapshot),
                  mock.patch.object(aggregate,"predict",side_effect=fake_pred)):
                x=aggregate.aggregate(dest)
            self.assertEqual(len(x["pairs"]),6)
            self.assertEqual(len(x["per_family_descriptive"]),2)
            self.assertTrue(all(z["validation_flip_count"]==3 for z in x["pairs"]))
            self.assertTrue(all(z["paired_delta_validation_BA"]>0 for z in x["pairs"]))
            self.assertTrue(all(s["n_independent_generators"]==3 for s in
                                x["per_family_descriptive"]))
            self.assertFalse(x["test_accessed"])

if __name__=="__main__":
    unittest.main()
