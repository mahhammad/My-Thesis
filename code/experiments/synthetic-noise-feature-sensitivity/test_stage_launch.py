"""Live-shaped GitHub Actions push event regression, no research execution."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import stage_launch as launch

PARENT="a"*40
HEAD="b"*40
APPROVED="c"*40


def fixture(stage="raw_noise"):
    path=launch.REQUEST_PREFIX+stage+".json"
    request={"schema":launch.SCHEMA,"stage":stage,
             "owner_comment_id":111,"qa_comment_id":222,
             "approved_code_sha":APPROVED,"test_accessed":False}
    # GitHub Actions push event has NO commit.added/modified/removed arrays.
    event={"ref":"refs/heads/"+launch.BRANCH,"before":PARENT,
           "after":HEAD,"created":False,"deleted":False,"forced":False,
           "repository":{"full_name":launch.REPO},
           "commits":[{"id":HEAD,"message":f"run(e-ch4-14): request {stage}"}],
           "head_commit":{"id":HEAD,"message":f"run(e-ch4-14): request {stage}"}}
    return request,event,[f"A\t{path}"],HEAD+" "+PARENT,f"run(e-ch4-14): request {stage}"


def verify(stage="raw_noise",kind="data",data=None):
    request,event,changes,parents,message = data or fixture(stage)
    return launch.validate_request(stage,request,kind=kind,changed_entries=changes,
                                   event=event,checkout_sha=HEAD,
                                   parents_line=parents,commit_message=message)


class PushGateTests(unittest.TestCase):
    def test_exact_github_actions_payload_without_file_lists(self):
        for stage in launch.STAGES:
            with self.subTest(stage=stage):
                request,event,change,parents,message=fixture(stage)
                self.assertNotIn("added",event["head_commit"])
                self.assertEqual(verify(stage,"fit" if stage=="fit" else "data"),
                                 request)

    def test_wrong_workflow(self):
        with self.assertRaises(PermissionError):
            verify("raw_noise","fit")

    def test_multiple_changed_files_and_modified_or_deleted_request(self):
        for changes in (["A\t"+launch.REQUEST_PREFIX+"raw_noise.json",
                         "M\t"+launch.REQUEST_PREFIX+"data.py"],
                        ["M\t"+launch.REQUEST_PREFIX+"raw_noise.json"],
                        ["D\t"+launch.REQUEST_PREFIX+"raw_noise.json"]):
            p=list(fixture());p[2]=changes
            with self.subTest(changes=changes),self.assertRaises(PermissionError):
                verify(data=p)

    def test_multi_parent_and_parent_mismatch(self):
        for parents in (HEAD+" "+PARENT+" "+APPROVED,HEAD+" "+APPROVED,
                        HEAD, "x"*40+" "+PARENT):
            p=list(fixture());p[3]=parents
            with self.subTest(parents=parents),self.assertRaises(PermissionError):
                verify(data=p)

    def test_event_before_after_and_head_mismatch(self):
        for target,key,value in (
            ("event","after","d"*40),("event","before","d"*40),
            ("event","ref","refs/heads/main"),("head_commit","id","d"*40),
            ("commits","id","d"*40),
        ):
            p=list(fixture())
            obj=p[1] if target=="event" else p[1]["head_commit"] if target=="head_commit" else p[1]["commits"][0]
            obj[key]=value
            with self.subTest(target=target,key=key),self.assertRaises(PermissionError):
                verify(data=p)

    def test_forced_deleted_created_and_multiple_commits(self):
        for field in ("forced","deleted","created"):
            p=list(fixture());p[1][field]=True
            with self.subTest(field=field),self.assertRaises(PermissionError):
                verify(data=p)
        p=list(fixture());p[1]["commits"].append({"id":PARENT})
        with self.assertRaises(PermissionError):
            verify(data=p)

    def test_commit_message_and_payload_sha(self):
        p=list(fixture());p[4]="run(e-ch4-14): request fit"
        with self.assertRaises(PermissionError):
            verify(data=p)
        p=list(fixture());p[0]["approved_code_sha"]="invalid"
        with self.assertRaises(PermissionError):
            verify(data=p)
        p=list(fixture());p[0]["test_accessed"]=True
        with self.assertRaises(PermissionError):
            verify(data=p)

    def test_duplicate_request_and_unapproved_field(self):
        p=list(fixture());p[0]["unknown"]="attack"
        with self.assertRaises(PermissionError):verify(data=p)
        p=list(fixture());p[0]["owner_comment_id"]="111"
        with self.assertRaises(PermissionError):verify(data=p)

    def test_dispatch_no_secret_output(self):
        r=launch.validate_dispatch("fit",111,222,kind="fit")
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/"output.txt"
            with patch.dict(os.environ,{"GITHUB_OUTPUT":str(f)}):
                launch._outputs(r,verified=False)
            saved=f.read_text("utf8")
            self.assertIn("stage=fit",saved)
            self.assertNotIn("TOKEN",saved)
            self.assertNotIn("GITHUB_ENV",saved)
        with self.assertRaises(PermissionError):
            launch.validate_dispatch("fit",111,222,kind="data")


if __name__=="__main__":
    unittest.main()
