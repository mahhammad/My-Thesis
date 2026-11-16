"""E14 branch-only launch is manual-by-commit, never by ordinary push."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage_launch import SCHEMA, validate_request, validate_dispatch


SHA='a'*40
PREFIX='code/experiments/synthetic-noise-feature-sensitivity/requests/'


def payload(stage='raw_noise'):
    path=PREFIX+stage+'.json'
    req={'schema':SCHEMA,'stage':stage,'owner_comment_id':111,
         'qa_comment_id':222,'approved_code_sha':SHA,'test_accessed':False}
    event={'commits':[{}],'head_commit':{'added':[path], 'modified':[],
           'removed':[], 'message':f'run(e-ch4-14): request {stage}'},
           'forced':False,'deleted':False}
    return req,event,[path]


class BranchLaunchTests(unittest.TestCase):
    def test_one_added_request_file_only(self):
        for stage in ('raw_noise','noise_masks','derived','fit'):
            r,e,p=payload(stage)
            result=validate_request(stage,r,kind='fit' if stage=='fit' else 'data',
                                    changed_paths=p,event=e)
            self.assertEqual(result['stage'],stage)

    def test_cannot_trigger_wrong_workflow_or_stage(self):
        r,e,p=payload()
        with self.assertRaises(PermissionError):
            validate_request('raw_noise',r,kind='fit',changed_paths=p,event=e)
        with self.assertRaises(PermissionError):
            validate_dispatch('fit',111,222,kind='data')

    def test_reject_multiple_commits_any_other_modified_file(self):
        for case in ('another_commit','modified','different_path','force','removed'):
            r,e,p=payload()
            if case=='another_commit':e['commits'].append({})
            if case=='modified':e['head_commit']['modified']=['README_FA.md']
            if case=='different_path':p.append('code/experiments/synthetic-noise-feature-sensitivity/data.py')
            if case=='force':e['forced']=True
            if case=='removed':e['head_commit']['removed']=['README_FA.md']
            with self.subTest(case=case),self.assertRaises(PermissionError):
                validate_request('raw_noise',r,kind='data',changed_paths=p,event=e)

    def test_request_must_be_fresh_and_cannot_overwrite(self):
        r,e,p=payload()
        e['head_commit']['added']=[]
        e['head_commit']['modified']=[p[0]]
        with self.assertRaises(PermissionError):
            validate_request('raw_noise',r,kind='data',changed_paths=p,event=e)

    def test_reject_forged_approval_reordered_or_test(self):
        for field,value in (('schema','fake'),('test_accessed',True),
                            ('approved_code_sha','deadbeef'),
                            ('owner_comment_id','111'),('qa_comment_id',-1)):
            r,e,p=payload()
            r[field]=value
            with self.subTest(field=field),self.assertRaises(PermissionError):
                validate_request('raw_noise',r,kind='data',changed_paths=p,event=e)

    def test_request_does_not_hide_extra_keys_or_changed_message(self):
        r,e,p=payload()
        r['authorization']='pretend'
        with self.assertRaises(PermissionError):
            validate_request('raw_noise',r,kind='data',changed_paths=p,event=e)
        r,e,p=payload()
        e['head_commit']['message']='regular push'
        with self.assertRaises(PermissionError):
            validate_request('raw_noise',r,kind='data',changed_paths=p,event=e)


if __name__=='__main__':
    unittest.main()
