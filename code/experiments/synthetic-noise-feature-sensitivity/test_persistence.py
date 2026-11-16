"""No real research outputs; synthetic files only check persistence invariants."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import persistence_check as persist
import recovery_status


class DurabilityTests(unittest.TestCase):
    def test_recovery_receipt_has_no_secret_field(self):
        env={"GITHUB_RUN_ID":"58","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ATTEMPT":"1","E14_STATUS_STAGE":"raw_noise",
             "E14_STATUS_RAW":"failure", "E14_FIT_TOKEN":"SHOULD_NOT_APPEAR"}
        result=recovery_status.safe_status(env)
        self.assertFalse(result["persistence_confirmed"])
        self.assertNotIn("SHOULD_NOT_APPEAR",json.dumps(result))
        self.assertFalse(result["secret_values_included"])

    def test_failure_in_missing_file_never_reports_persisted(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                persist.check_evidence("raw_noise",root=t)
            with self.assertRaises(FileNotFoundError):
                persist.check_evidence("fit",root=t)

    def test_modified_raw_stage_sha_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            folder=root/"inputs"/"noise_raw"
            folder.mkdir(parents=True)
            (folder/"new_noise_seed20261102.npz").write_bytes(b"fixture")
            (folder/"manifest.json").write_text(json.dumps({"file_sha256":"0"*64}))
            with self.assertRaisesRegex(ValueError,"checksum"):
                persist.check_evidence("raw_noise",root=root)

    def test_no_artifact_id_means_no_success(self):
        with self.assertRaises(PermissionError):
            persist.check_complete("raw_noise","",root=".")
        with self.assertRaises(PermissionError):
            persist.check_complete("raw_noise","invalid",root=".")

    def test_remote_sha_drift_fails(self):
        with self.assertRaises(RuntimeError):
            persist.remote_exact("a"*40,runner=lambda *a,**k:"b"*40+
                                 "  refs/heads/"+persist.BRANCH+"\n")

    def test_remote_sha_exact_is_accepted(self):
        self.assertTrue(persist.remote_exact("a"*40,
            runner=lambda *a,**k:"a"*40+"\trefs/heads/"+persist.BRANCH+"\n"))


if __name__=="__main__":
    unittest.main()
