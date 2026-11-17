"""No research runs; mock verified/reserved/execute stage and inspect safe evidence."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import secure_stage as st

TOKEN="synthetic-fixture-token-"+"x"*45

class SecretScopeTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{"GITHUB_RUN_ID":"987",
            "GITHUB_SHA":"a"*40, "GITHUB_EVENT_NAME":"push",
            "RUNNER_TEMP":"/tmp","E14_STAGE_TOKEN":TOKEN})
        self.env.start()
    def tearDown(self):
        self.env.stop()

    def test_fail_closed_missing_or_wrong_secret_before_any_operation(self):
        calls=[]
        with self.assertRaises(PermissionError):
            st.run("raw_noise",111,222,token="",verify=lambda _:calls.append("verify"))
        with self.assertRaises(PermissionError):
            st.run("raw_noise",111,222,token="z"*45,verify=lambda _:calls.append("verify"))
        self.assertEqual(calls,[])

    def test_verify_preflight_reserve_execute_order(self):
        events=[]
        def event(name):
            def step(args):
                events.append(name)
                return 0
            return step
        with tempfile.TemporaryDirectory() as path:
            def saved(stage,outcome,phase,**kwargs):
                events.append("receipt")
                return st.safe_receipt(stage,outcome,phase,root=path,**kwargs)
            st.run("raw_noise",111,222,token=TOKEN,
                verify=event("verify"),preflight=event("preflight"),
                reserve=event("reserve"),execute=event("execute"),receipt=saved)
            self.assertEqual(events,["verify","preflight","reserve","execute","receipt"])
            files=list(Path(path).glob("*.json"))
            self.assertEqual(len(files),1)
            raw=files[0].read_text()
            self.assertNotIn(TOKEN,raw)
            self.assertEqual(json.loads(raw)["outcome"],"EXECUTED")

    def test_preflight_failure_does_not_spend(self):
        events=[]
        def fail(_):
            events.append("preflight")
            raise ValueError("Bad frozen parent")
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ValueError):
                st.run("derived",111,222,token=TOKEN,
                    verify=lambda _:events.append("verify"),
                    preflight=fail,
                    reserve=lambda _:events.append("reserve"),
                    execute=lambda _:events.append("execute"),
                    receipt=lambda stage,outcome,phase,**kw:
                        st.safe_receipt(stage,outcome,phase,root=path,**kw))
            self.assertEqual(events,["verify","preflight"])
            rec=json.loads(next(Path(path).glob("*.json")).read_text())
            self.assertEqual(rec["outcome"],"FAIL_STOP")
            self.assertEqual(rec["last_phase"],"PREFLIGHT")
            self.assertNotIn(TOKEN,json.dumps(rec))

    def test_replay_of_same_receipt_filename_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            st.safe_receipt("fit","FAIL_STOP","VERIFY",root=t,token=TOKEN)
            with self.assertRaises(FileExistsError):
                st.safe_receipt("fit","FAIL_STOP","VERIFY",root=t,token=TOKEN)

if __name__=="__main__":
    unittest.main()
