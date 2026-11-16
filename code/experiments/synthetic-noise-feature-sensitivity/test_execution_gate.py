"""Offline negative security tests for G4. No live API, no real data or fits."""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import execution_gate as g
from data import NoiseSpec, spec_sha

CODE_SHA = 'a'*40
TOKEN = 'random-one-time-run-token-' + 'X'*40
OWNER_ID, QA_ID = 111, 222


def _sha(s):
    return sha256(s.encode()).hexdigest()


def evidence():
    spec = NoiseSpec(2026111601, 2026111602, (1.1,)*3, (0.6,)*3,
                     (0.2,)*3, 'independent_zero_mean_gaussian')
    owner = {
        'schema':g.OWNER_SCHEMA, 'issue':90, 'repo':g.REPO,
        'branch':g.BRANCH, 'approved_code_sha':CODE_SHA,
        'approved_stages':list(g.STAGES), 'independent_auditor_login':'qa-auditor',
        'stop_on_failure':True, 'no_auto_retry':True, 'sequential_fits':True,
        'test_accessed':False, 'n_train':300, 'n_validation':90,
        'fit_count':36, 'updates_per_fit':30, 'total_update_budget':1080,
        'noise_spec':asdict(spec), 'noise_spec_sha256':spec_sha(spec),
        'wall_minutes':300, 'max_rss_mib':6144,
        'stage_token_sha256':{name:_sha(name+':'+TOKEN) for name in g.STAGES}
    }
    qa = {'schema':g.QA_SCHEMA, 'issue':90, 'owner_comment_id':OWNER_ID,
          'approved_code_sha':CODE_SHA, 'noise_spec_sha256':spec_sha(spec),
          'vote':'ACCEPT_G4_AND_PROTOCOL', 'approved_stages':list(g.STAGES),
          'max_rss_mib':6144, 'wall_minutes':300, 'ci_run_id':100}
    def comment(n,user,payload,marker):
        return {'id':n, 'issue_url':f'https://api.github.com/repos/{g.REPO}/issues/90',
                'html_url':f'https://github.com/{g.REPO}/issues/90#issuecomment-{n}',
                'created_at':'2026-11-16T00:00:00Z',
                'updated_at':'2026-11-16T00:00:00Z',
                'user':{'login':user},'body':marker+json.dumps(payload)}
    rows = {
        f'/repos/{g.REPO}':{'full_name':g.REPO,'owner':{'login':'mahhammad'}},
        f'/repos/{g.REPO}/issues/comments/{OWNER_ID}':
            comment(OWNER_ID,'mahhammad',owner,g.OWNER_MARKER),
        f'/repos/{g.REPO}/issues/comments/{QA_ID}':
            comment(QA_ID,'qa-auditor',qa,g.QA_MARKER),
        f'/repos/{g.REPO}/actions/runs/100':{
            'head_sha':CODE_SHA,'conclusion':'success',
            'name':'E-CH4-14 branch checks (NO FIT)','head_branch':g.BRANCH}
    }
    return rows


def verify(rows, *, stage='raw_noise', token=TOKEN, event='workflow_dispatch',
           current_sha=CODE_SHA):
    with patch.object(g,'load_mother',return_value=SimpleNamespace(source_sha='f'*64)):
        return g.verify(stage,OWNER_ID,QA_ID,fetch=lambda path:rows[path],token=token,
                        current_sha=current_sha,run_id='88',
                        workflow=g.EXPECTED_WORKFLOWS[stage],branch=g.BRANCH,
                        event=event,git_check=False)


class VerificationTests(unittest.TestCase):
    def test_exact_owner_qa_spec_and_ci_verified(self):
        ctx=verify(evidence())
        self.assertEqual(ctx['owner_comment_id'],OWNER_ID)
        self.assertEqual(ctx['approved_spec_sha256'],ctx['data_spec_sha256'])
        self.assertFalse(ctx['test_accessed'])
        self.assertEqual(ctx['max_wall_minutes'],300)

    def test_fake_owner_identity_rejected(self):
        rows=evidence()
        rows[f'/repos/{g.REPO}/issues/comments/{OWNER_ID}']['user']['login']='attacker'
        with self.assertRaises(PermissionError):verify(rows)

    def test_fake_qa_identity_rejected(self):
        rows=evidence()
        rows[f'/repos/{g.REPO}/issues/comments/{QA_ID}']['user']['login']='mahhammad'
        with self.assertRaises(PermissionError):verify(rows)

    def test_edited_owner_comment_rejected(self):
        rows=evidence()
        rows[f'/repos/{g.REPO}/issues/comments/{OWNER_ID}']['updated_at']='2026-11-17T00:00:00Z'
        with self.assertRaises(PermissionError):verify(rows)

    def test_wrong_issue_and_unstructured_url_rejected(self):
        rows=evidence()
        rows[f'/repos/{g.REPO}/issues/comments/{QA_ID}']['issue_url']='https://api.github.com/repos/other/repo/issues/90'
        with self.assertRaises(PermissionError):verify(rows)

    def test_missing_qa_vote_rejected(self):
        rows=evidence()
        key=f'/repos/{g.REPO}/issues/comments/{QA_ID}'
        qa=json.loads(rows[key]['body'][len(g.QA_MARKER):])
        qa['vote']='PENDING_REVIEW'
        rows[key]['body']=g.QA_MARKER+json.dumps(qa)
        with self.assertRaises(PermissionError):verify(rows)

    def test_mismatched_code_or_ci_rejected(self):
        rows=evidence()
        rows[f'/repos/{g.REPO}/actions/runs/100']['head_sha']='b'*40
        with self.assertRaises(PermissionError):verify(rows)
        rows=evidence()
        with self.assertRaises(PermissionError):
            verify(rows,current_sha='b'*40)

    def test_wrong_token_stage_and_event_rejected(self):
        with self.assertRaises(PermissionError):verify(evidence(),token='fake-token'*4)
        with self.assertRaises(PermissionError):verify(evidence(),event='push')
        with self.assertRaises(PermissionError):
            verify(evidence(),stage='fit',token=TOKEN,event='push')

    def test_overspent_budget_and_unapproved_RSS_rejected(self):
        rows=evidence()
        key=f'/repos/{g.REPO}/issues/comments/{OWNER_ID}'
        x=json.loads(rows[key]['body'][len(g.OWNER_MARKER):])
        x['total_update_budget']=1081
        rows[key]['body']=g.OWNER_MARKER+json.dumps(x)
        with self.assertRaises(PermissionError):verify(rows)
        rows=evidence()
        x=json.loads(rows[key]['body'][len(g.OWNER_MARKER):])
        x['max_rss_mib']=7168
        rows[key]['body']=g.OWNER_MARKER+json.dumps(x)
        with self.assertRaises(PermissionError):verify(rows)

    def test_one_time_reservation_prevents_replay(self):
        ref={'ref':'refs/tags/'+g.TAG_NAMES['fit'],'object':{'sha':CODE_SHA}}
        fetch=lambda path:ref
        with self.assertRaises(FileExistsError):
            g.verify_unspent('fit',fetch=fetch)
        calls=[]
        def post(path,payload):
            calls.append((path,payload))
            return ref
        with self.assertRaises(FileExistsError):
            g.reserve('fit',CODE_SHA,fetch=fetch,post=post)
        self.assertEqual(calls,[])

    def test_unspent_404_can_proceed_then_single_tag_reserved(self):
        def fetch(path):
            raise HTTPError('https://api.github.com'+path,404,'missing',{},None)
        seen=[]
        def post(path,payload):
            seen.append(payload)
            return {'ref':'refs/tags/'+g.TAG_NAMES['raw_noise'],
                    'object':{'sha':CODE_SHA}}
        result=g.reserve('raw_noise',CODE_SHA,fetch=fetch,post=post)
        self.assertIn('raw-noise-budget-spent',result)
        self.assertEqual(len(seen),1)


if __name__=='__main__':
    unittest.main()
