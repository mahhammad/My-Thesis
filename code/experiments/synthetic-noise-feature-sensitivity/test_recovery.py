"""Fail-closed E14 raw-noise recovery. No token, RNG, data or fit executed."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from urllib.error import HTTPError

sys.path.insert(0,str(Path(__file__).resolve().parent))
import execution_gate as g
import stage_launch as launch
import bounded_run
from test_execution_gate import evidence, CODE_SHA, TOKEN, OWNER_ID, QA_ID
from test_stage_launch import HEAD, PARENT, fixture

OLD_TAG = "refs/tags/" + g.TAG_NAMES["raw_noise"]
NEW_TAG = "refs/tags/" + g.RECOVERY["budget_tag"]


def signed_evidence():
    rows=evidence()
    for id,marker in ((OWNER_ID,g.OWNER_MARKER),(QA_ID,g.QA_MARKER)):
        key=f"/repos/{g.REPO}/issues/comments/{id}"
        row=rows[key]
        fields=json.loads(row["body"][len(marker):])
        fields["raw_noise_recovery"]=dict(g.RECOVERY)
        row["body"]=marker+json.dumps(fields)
    rows[f"/repos/{g.REPO}/git/ref/tags/{g.TAG_NAMES['raw_noise']}"]={
        "ref":OLD_TAG,"object":{"sha":g.RECOVERY["failed_request_sha"]}}
    rows[f"/repos/{g.REPO}/actions/runs/{g.RECOVERY['failed_run_id']}"]={
        "head_sha":g.RECOVERY["failed_request_sha"],
        "name":g.EXPECTED_WORKFLOWS["raw_noise"],
        "conclusion":"failure"}
    return rows


def verify_recovery(rows):
    environ={
        "GITHUB_EVENT_NAME":"push",
        "E14_REQUEST_RECOVERY":"raw_noise_recovery_01",
        "E14_PUSH_REQUEST_VERIFIED":"true",
        "E14_STAGE":"raw_noise",
        "E14_REQUEST_APPROVED_CODE_SHA":CODE_SHA,
    }
    with patch.dict(os.environ,environ),patch.object(g,"load_mother",
            return_value=SimpleNamespace(source_sha="f"*64)):
        return g.verify("raw_noise",OWNER_ID,QA_ID,fetch=lambda key:rows[key],
            token=TOKEN,current_sha=CODE_SHA,run_id="88",
            workflow=g.EXPECTED_WORKFLOWS["raw_noise"],
            branch=g.BRANCH,event="push",git_check=False)


class RecoveryTests(unittest.TestCase):
    def test_signed_old_failure_and_separate_budget(self):
        ctx=verify_recovery(signed_evidence())
        self.assertEqual(ctx["recovery_request"],g.RECOVERY["request"])
        self.assertEqual(ctx["budget_reservation"],NEW_TAG)
        self.assertNotEqual(ctx["budget_reservation"],OLD_TAG)
        with self.assertRaises(FileExistsError):
            g.verify_unspent("raw_noise",fetch=lambda _:{"ref":NEW_TAG},
                             budget_ref=NEW_TAG)

    def test_old_budget_is_never_reset_or_overwritten(self):
        calls=[]
        def fetch(path):
            if path.endswith(g.RECOVERY["budget_tag"]):
                raise HTTPError("https://api.github.com"+path,404,"missing",{},None)
            raise AssertionError("Old tag must not be changed")
        def post(path,payload):
            calls.append((path,payload))
            return {"ref":NEW_TAG,"object":{"sha":CODE_SHA}}
        self.assertEqual(g.reserve("raw_noise",CODE_SHA,fetch=fetch,post=post,
                                  budget_ref=NEW_TAG),NEW_TAG)
        self.assertEqual(calls,[("/repos/"+g.REPO+"/git/refs",
                                 {"ref":NEW_TAG,"sha":CODE_SHA})])

    def test_require_signed_matching_new_qa_decision(self):
        rows=signed_evidence()
        key=f"/repos/{g.REPO}/issues/comments/{QA_ID}"
        rows[key]["body"]=g.QA_MARKER+json.dumps({"schema":g.QA_SCHEMA})
        with self.assertRaises(PermissionError):
            verify_recovery(rows)

    def test_old_tag_and_failed_run_must_match_real_github_records(self):
        cases=("wrong_tag_sha","old_run_success","old_run_wrong_head")
        for case in cases:
            rows=signed_evidence()
            if case=="wrong_tag_sha":
                rows[f"/repos/{g.REPO}/git/ref/tags/{g.TAG_NAMES['raw_noise']}"]["object"]["sha"]="e"*40
            if case=="old_run_success":
                rows[f"/repos/{g.REPO}/actions/runs/{g.RECOVERY['failed_run_id']}"]["conclusion"]="success"
            if case=="old_run_wrong_head":
                rows[f"/repos/{g.REPO}/actions/runs/{g.RECOVERY['failed_run_id']}"]["head_sha"]="e"*40
            with self.subTest(case=case),self.assertRaises(PermissionError):
                verify_recovery(rows)

    def test_missing_recovery_request_env_must_fail(self):
        rows=signed_evidence()
        with patch.dict(os.environ,{"GITHUB_EVENT_NAME":"push",
                                    "E14_REQUEST_RECOVERY":""}):
            with self.assertRaises(PermissionError):
                g._budget_tag("raw_noise",
                    json.loads(rows[f"/repos/{g.REPO}/issues/comments/{OWNER_ID}"]["body"][len(g.OWNER_MARKER):]),
                    json.loads(rows[f"/repos/{g.REPO}/issues/comments/{QA_ID}"]["body"][len(g.QA_MARKER):]),
                    fetch=lambda key:rows[key])

    def test_push_request_must_add_one_new_recovery_file(self):
        req,event,_,parents,_=fixture("raw_noise")
        req["recovery_request"]="raw_noise_recovery_01"
        name="run(e-ch4-14): request raw_noise_recovery_01"
        event["head_commit"]["message"]=name
        event["commits"][0]["message"]=name
        added=["A\t"+launch.REQUEST_PREFIX+"raw_noise_recovery_01.json"]
        self.assertEqual(
            launch.validate_request("raw_noise",req,kind="data",
                changed_entries=added,event=event,checkout_sha=HEAD,
                parents_line=parents,commit_message=name),req)
        for changed in (["M\t"+launch.REQUEST_PREFIX+"raw_noise_recovery_01.json"],
                        added+["A\tother.txt"]):
            with self.assertRaises(PermissionError):
                launch.validate_request("raw_noise",req,kind="data",
                    changed_entries=changed,event=event,checkout_sha=HEAD,
                    parents_line=parents,commit_message=name)
        req["recovery_request"]="raw_noise_recovery_02"
        with self.assertRaises(PermissionError):
            launch.validate_request("raw_noise",req,kind="data",
                changed_entries=added,event=event,checkout_sha=HEAD,
                parents_line=parents,commit_message=name)

    def test_bounded_runner_uses_signed_recovery_ref(self):
        ctx={"source_sha":CODE_SHA,"recovery_request":"raw_noise_recovery_01",
             "budget_reservation":NEW_TAG}
        observed=[]
        def fetch(path):
            observed.append(path)
            return {"ref":NEW_TAG,"object":{"sha":CODE_SHA}}
        self.assertTrue(bounded_run.validate_reservation("raw_noise",ctx,fetch=fetch))
        self.assertEqual(observed,[
            f"/repos/{g.REPO}/git/ref/tags/{g.RECOVERY['budget_tag']}"])
        ctx["budget_reservation"]="refs/tags/FAKE"
        with self.assertRaises(PermissionError):
            bounded_run.validate_reservation("raw_noise",ctx,fetch=fetch)

    def test_other_stages_reject_recovery_budget(self):
        for stage in ("noise_masks","derived","fit"):
            with self.subTest(stage=stage),self.assertRaises(PermissionError):
                g.verify_unspent(stage,fetch=lambda _:None,budget_ref=NEW_TAG)


if __name__=="__main__":
    unittest.main()
