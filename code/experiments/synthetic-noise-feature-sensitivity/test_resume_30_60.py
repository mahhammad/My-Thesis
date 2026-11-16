"""E14 resume protocol regression: no 300-sample training, no new budget, no Test."""
import copy, hashlib, io, json, pathlib, tempfile, unittest
from unittest.mock import patch
import numpy as np
import torch
from resume_30_60 import (nfi_table,read_run,source_evidence,compare_step30,
                          atomic_new,append_durable,TRACE_KEYS,restore_state)
from resume_gate import (TAG,FROZEN,validate_push,compare_approvals,OWNER_MARKER,
                         QA_MARKER,MESSAGE,REQUEST_PATH,parse_comment)

class ContinuationSafetyTests(unittest.TestCase):
    def test_origin_real_step30_boundary_shared_and_class(self):
        rows,prov=source_evidence();nfi=nfi_table()
        for job in ("N0-01__shared","N2-01__class"):
            sub=job.split("__")[0]
            graph,data,z,opt,snapshot,logits,checksum,_=read_run(job,rows[sub],prov,nfi)
            self.assertEqual(int(opt.state[z]["step"]),30)
            self.assertEqual(snapshot["step"],30)
            self.assertEqual(logits.dtype,np.float64)
            self.assertTrue(len(checksum)==64)
            self.assertEqual(snapshot["nfi3"]["shared"]["NFI3"],None) if sub=="N0-01" else None

    def test_malformed_source_mapping_rejected(self):
        rows,prov=source_evidence()
        self.assertEqual(len(rows),18)
        self.assertEqual(prov["n_fits"],36)
        target=copy.deepcopy(rows["N2-01"])
        target["hashes"]["train_x_sha256"]="f"*64
        from resume_30_60 import read_frozen_dataset
        with self.assertRaises(PermissionError):read_frozen_dataset(target)

    def test_full_adam_state_resume_same_as_uninterrupted_fixture(self):
        # Deterministic scalar fixture, NOT the actual IMC research graph.
        def loss(z):return (z-2.0).square().mean()+.07*(z+.8).square().mean()
        base=torch.nn.Parameter(torch.tensor([[-.3,1.2]],dtype=torch.float64))
        opt=torch.optim.Adam([base],lr=.12)
        def step(z,o):
            o.zero_grad(set_to_none=True);loss(z).backward();o.step()
        for _ in range(30):step(base,opt)
        original=base.detach().clone()
        buf=io.BytesIO()
        torch.save({"logits":base.detach().clone(),"optimizer":opt.state_dict()},buf)
        buf.seek(0)
        payload=torch.load(buf,weights_only=True,map_location="cpu")
        continued=torch.nn.Parameter(payload["logits"].clone())
        resumed=torch.optim.Adam([continued],lr=.12)
        resumed.load_state_dict(payload["optimizer"])
        self.assertTrue(torch.equal(continued,original))
        self.assertEqual(int(resumed.state[continued]["step"]),30)
        self.assertTrue(torch.equal(resumed.state[continued]["exp_avg"],opt.state[base]["exp_avg"]))
        self.assertTrue(torch.equal(resumed.state[continued]["exp_avg_sq"],opt.state[base]["exp_avg_sq"]))
        for t in range(31,61):
            step(base,opt);step(continued,resumed)
            self.assertTrue(torch.equal(base,continued),f"Adam state changed at {t}")
            self.assertEqual(int(resumed.state[continued]["step"]),t)
        self.assertEqual(int(opt.state[base]["step"]),60)
        # Resetting optimizer with same logits is NOT a legitimate continuation.
        resetparam=torch.nn.Parameter(original.clone())
        resetopt=torch.optim.Adam([resetparam],lr=.12)
        step(resetparam,resetopt)
        self.assertFalse(torch.equal(resetparam,continued))

    def test_new_origin_rows_not_implicitly_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            file=pathlib.Path(temp)/"trace.jsonl"
            first="frozen_immutable"
            digest=atomic_new(file,first)
            self.assertEqual(digest,hashlib.sha256(first.encode()).hexdigest())
            with self.assertRaises(FileExistsError):atomic_new(file,"RESTART")
            self.assertEqual(file.read_text(),first)

    def test_append_budget_ledger_exact_updates_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            file=pathlib.Path(temp)/"ledger.jsonl"
            with file.open("x",encoding="utf8") as stream:
                for t in range(31,61):
                    append_durable(stream,dict(event="UPDATE",step=t,completed=t-30))
            rows=[json.loads(x) for x in file.read_text().splitlines()]
            self.assertEqual(len(rows),30)
            self.assertEqual([x["step"] for x in rows],list(range(31,61)))
            self.assertEqual([x["completed"] for x in rows],list(range(1,31)))

class AuthorizationTests(unittest.TestCase):
    def request(self):
        return {"schema":"e-ch4-14-resume-request-v1","stage":"resume_only",
                "approved_code_sha":"a"*40,"owner_comment_id":101,
                "qa_comment_id":102,"test_accessed":False}
    def event(self):
        return {"after":"b"*40,"before":"a"*40,
                "ref":"refs/heads/experiment/ch4-14-noise-sensitivity-01",
                "repository":{"full_name":"mahhammad/My-Thesis"},
                "created":False,"deleted":False,"forced":False,
                "head_commit":{"id":"b"*40,"message":MESSAGE},
                "commits":[{"id":"b"*40}]}
    def run(self,**kwargs):
        return validate_push(kwargs.pop("event",self.event()),
                    kwargs.pop("request",self.request()),head="b"*40,
                    parent="a"*40,entries=kwargs.pop("entries",["A\t"+REQUEST_PATH]),
                    message=kwargs.pop("message",MESSAGE))
    def test_realistic_push_no_head_commit_file_arrays(self):
        self.assertTrue(self.run())
    def test_changed_or_extra_files_rejected(self):
        for entries in (["M\t"+REQUEST_PATH],["D\t"+REQUEST_PATH],
                        ["A\t"+REQUEST_PATH,"A\tREADME.md"]):
            with self.assertRaises(PermissionError):self.run(entries=entries)
    def test_force_push_wrong_before_multiple_commits_rejected(self):
        for key,value in (("forced",True),("before","c"*40),
                          ("commits",[{"id":"b"*40},{"id":"c"*40}])):
            ev=self.event();ev[key]=value
            with self.assertRaises(PermissionError):self.run(event=ev)
    def test_replay_and_wrong_approved_parent_rejected(self):
        req=self.request();req["approved_code_sha"]="c"*40
        with self.assertRaises(PermissionError):self.run(request=req)
        req=self.request();req["owner_comment_id"]=req["qa_comment_id"]
        with self.assertRaises(PermissionError):self.run(request=req)
    def test_approval_contract_bound_to_unique_budget_and_ci(self):
        base=dict(issue=90,stage="resume_only",approved_code_sha="a"*40,
                  frozen_results_sha=FROZEN,data_spec_sha256="82917272b0ac3850cecfd7bf4698dce55cbe955f8e3c308cf2c44d30e7f1ab8a",
                  checkpoint_set_sha256="b"*64,fits=36,from_step=30,to_step=60,
                  additional_updates=1080,budget_ref=TAG,ci_run_id=99,test_accessed=False)
        owner={**base,"schema":"e-ch4-14-resume-owner-v1","vote":"AUTHORIZE_RESUME_ONLY",
                "secret_sha256":"c"*64}
        qa={**base,"schema":"e-ch4-14-resume-qa-v1",
            "vote":"ACCEPT_RESUME_PROTOCOL_AND_BUDGET","owner_comment_id":101,
            "audit":"INDEPENDENT_CODE_AND_ORIGIN_REVIEW"}
        self.assertTrue(compare_approvals(owner,qa,code_sha="a"*40,ci_run_id=99,
                                         checkpoint_sha="b"*64,token_digest="c"*64))
        for item,field,value in ((owner,"budget_ref","refs/tags/e-ch4-14-fit-budget-spent"),
                                 (qa,"additional_updates",1081),(qa,"vote","PENDING"),
                                 (owner,"secret_sha256","f"*64)):
            changed=copy.deepcopy(item);changed[field]=value
            with self.assertRaises(PermissionError):
                compare_approvals(changed if item is owner else owner,
                                  changed if item is qa else qa,
                                  code_sha="a"*40,ci_run_id=99,
                                  checkpoint_sha="b"*64,token_digest="c"*64)
    def test_approval_marker_exact_json(self):
        d=dict(schema="e-ch4-14-resume-owner-v1")
        self.assertEqual(parse_comment({"body":OWNER_MARKER+json.dumps(d)},OWNER_MARKER),d)
        with self.assertRaises(PermissionError):
            parse_comment({"body":QA_MARKER+json.dumps(d)},OWNER_MARKER)
    def test_no_production_auth_without_context(self):
        import resume_30_60
        with self.assertRaises((PermissionError,FileNotFoundError,KeyError)):
            resume_30_60.authorized_execute(pathlib.Path("/tmp/no-e14-authorization.json"))

if __name__=="__main__":
    unittest.main()
