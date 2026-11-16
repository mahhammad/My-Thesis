"""Static branch-only workflow safeguards (NO-FIT CI never launches research)."""
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[3]
WF=ROOT/'.github/workflows'
FILES=('e-ch4-14-authorized-data.yml','e-ch4-14-authorized-36fit.yml')


class WorkflowContract(unittest.TestCase):
    def test_manual_only_branch_only_no_retry(self):
        for name in FILES:
            with self.subTest(file=name):
                text=(WF/name).read_text('utf8')
                self.assertRegex(text,r'(?m)^  workflow_dispatch:')
                self.assertRegex(text,r'(?m)^  push:')
                self.assertNotRegex(text,r'(?m)^  (pull_request|schedule|repository_dispatch):')
                self.assertIn('stage_launch.py',text)
                self.assertIn("code/experiments/synthetic-noise-feature-sensitivity/requests/",text)
                self.assertIn('cancel-in-progress: false',text)
                self.assertIn("github.ref == 'refs/heads/experiment/ch4-14-noise-sensitivity-01'",text)
                self.assertIn('E14_STAGE_TOKEN:',text)
                self.assertIn('execution_gate.py',text)
                self.assertIn('--mode verify',text)
                self.assertIn('--mode reserve',text)
                self.assertIn('bounded_run.py',text)
                self.assertIn('if: always()',text)

    def test_workflow_stage_separation(self):
        data=(WF/FILES[0]).read_text('utf8')
        fit=(WF/FILES[1]).read_text('utf8')
        self.assertIn('noise_masks',data)
        self.assertIn('raw_noise',data)
        self.assertIn('derived',data)
        self.assertNotIn('36-fit evidence or irreversible failure',data)
        self.assertIn('--stage fit',fit)
        self.assertIn('requests/fit.json',fit)
        self.assertIn('stage_launch.py --kind fit',fit)
        for stage in ('raw_noise','noise_masks','derived'):
            self.assertIn('requests/'+stage+'.json',data)
            self.assertNotIn('requests/'+stage+'.json',fit)


if __name__=='__main__':
    unittest.main()
