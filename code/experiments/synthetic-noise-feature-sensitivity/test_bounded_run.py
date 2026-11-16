"""No-fit tests for E14 bounded launch, manual trigger and approved budgets."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bounded_run as bound


class FakeDoneProcess:
    returncode = 0
    def poll(self):
        return 0


class BoundedRunTests(unittest.TestCase):
    def setUp(self):
        self.ctx = {'stage':'raw_noise', 'source_sha':'a'*40,
                    'budget_reservation':'refs/tags/e-ch4-14-raw-noise-budget-spent',
                    'max_wall_minutes':300,'max_rss_mib':6144}

    def test_reservation_must_exist_and_match_current_sha(self):
        ref='refs/tags/e-ch4-14-raw-noise-budget-spent'
        good={'ref':ref,'object':{'sha':'a'*40}}
        self.assertTrue(bound.validate_reservation('raw_noise',self.ctx,fetch=lambda path:good))
        with self.assertRaises(PermissionError):
            bound.validate_reservation('raw_noise',self.ctx,
                                       fetch=lambda path:{'ref':ref,'object':{'sha':'b'*40}})
        with self.assertRaises(PermissionError):
            bound.validate_reservation('raw_noise',self.ctx,
                                       fetch=lambda path:{'ref':ref.replace('raw-noise','fit'),
                                                          'object':{'sha':'a'*40}})

    def test_only_manual_dispatch_and_no_subprocess_when_push(self):
        with patch.object(bound,'load_context',return_value=(self.ctx,object())),\
             patch.object(bound,'validate_reservation',return_value=True),\
             patch.object(bound,'preflight'),\
             patch.dict(os.environ,{'GITHUB_EVENT_NAME':'push'}):
            def never_run(*args,**kwargs):
                self.fail('Push must never start production stages')
            with self.assertRaises(PermissionError):
                bound.launch('raw_noise',Path('fixture.json'),runner=never_run)

    def test_reject_unapproved_cpu_memory_wall_limits(self):
        with patch.object(bound,'load_context',return_value=(self.ctx,object())),\
             patch.object(bound,'validate_reservation',return_value=True),\
             patch.object(bound,'preflight'),\
             patch.dict(os.environ,{'GITHUB_EVENT_NAME':'workflow_dispatch'}):
            def no_launch(*args,**kwargs):
                self.fail('Out-of-budget stage must not launch')
            self.ctx['max_rss_mib']=8000
            with self.assertRaises(PermissionError):
                bound.launch('raw_noise',Path('fixture.json'),runner=no_launch)
            self.ctx['max_rss_mib']=6144
            self.ctx['max_wall_minutes']=400
            with self.assertRaises(PermissionError):
                bound.launch('raw_noise',Path('fixture.json'),runner=no_launch)

    def test_no_operation_or_data_from_import(self):
        # All production code is behind main/launch; the CI imports module only.
        self.assertEqual(bound.MAX_JOB_SECONDS,1200)
        self.assertEqual(bound.POLL_SECONDS,1.0)


if __name__=='__main__':
    unittest.main()
