"""Inspect protected workflow layout, no secret access during pip installation."""
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[3]
WORKFLOWS=ROOT/".github/workflows"
DATA="e-ch4-14-authorized-data.yml"
FIT="e-ch4-14-authorized-36fit.yml"
PREFIX="code/experiments/synthetic-noise-feature-sensitivity/requests/"


def step_blocks(text):
    chunks=re.split(r"(?=^      - name: )",text,flags=re.MULTILINE)
    return [c for c in chunks if c.startswith("      - name: ")]


class WorkflowContract(unittest.TestCase):
    def test_only_branch_request_or_manual_dispatch(self):
        for filename in (DATA,FIT):
            with self.subTest(filename=filename):
                text=(WORKFLOWS/filename).read_text("utf8")
                self.assertRegex(text,r"(?m)^  workflow_dispatch:")
                self.assertRegex(text,r"(?m)^  push:")
                self.assertNotRegex(text,r"(?m)^  (pull_request|schedule|repository_dispatch):")
                self.assertIn("cancel-in-progress: false",text)
                self.assertIn("github.ref == 'refs/heads/experiment/ch4-14-noise-sensitivity-01'",text)
                self.assertIn("stage_launch.py",text)
                self.assertIn("secure_stage.py",text)
                self.assertIn("persistence_check.py",text)
                self.assertIn("actions/upload-artifact@v4",text)
                self.assertIn("retention-days: 90",text)
                self.assertIn("if: always()",text)
                self.assertIn("if: success()",text)
                self.assertIn("steps.recovery.outputs.artifact-id",text)
                self.assertNotIn("GITHUB_ENV",text)
                self.assertLess(text.index("Install fixed versions"),text.index("secrets.E14_"))
                job_env=text.split("    env:\n",1)[1].split("    steps:\n",1)[0]
                self.assertNotIn("E14_",job_env)
                self.assertNotIn("GITHUB_TOKEN",job_env)
                self.assertLess(text.index("actions/upload-artifact@v4"),text.index("git push origin"))
                self.assertIn("'numpy==2.4.6'",text)
                self.assertIn("'psutil==7.0.0'",text)

    def test_stage_specific_token_is_only_visible_in_its_step(self):
        data=(WORKFLOWS/DATA).read_text("utf8")
        fit=(WORKFLOWS/FIT).read_text("utf8")
        for stage,secret in (("raw_noise","E14_RAW_NOISE_TOKEN"),
                             ("noise_masks","E14_NOISE_MASKS_TOKEN"),
                             ("derived","E14_DERIVED_TOKEN")):
            self.assertIn(PREFIX+stage+".json",data)
            block=next(c for c in step_blocks(data) if
                       "Authorized "+stage+" (" in c)
            visible=set(re.findall(r"secrets\.(E14_[A-Z_]+_TOKEN)",block))
            self.assertEqual(visible,{secret})
            self.assertNotIn("secrets."+secret,fit)
        self.assertNotIn(PREFIX+"fit.json",data)
        self.assertIn(PREFIX+"fit.json",fit)
        fb=next(c for c in step_blocks(fit) if
                "Authorized 36 sequential fits" in c)
        self.assertEqual(set(re.findall(r"secrets\.(E14_[A-Z_]+_TOKEN)",fb)),
                         {"E14_FIT_TOKEN"})
        self.assertIn("'torch==2.9.1'",fit)

    def test_normal_ci_can_never_run_real_stage(self):
        text=(WORKFLOWS/"e-ch4-14-branch-ci.yml").read_text("utf8")
        self.assertIn("E_CH4_14_FIT_AUTHORIZED: 'false'",text)
        self.assertNotIn("secure_stage.py",text)
        self.assertNotIn("requests/fit.json'",text)


if __name__=="__main__":
    unittest.main()
