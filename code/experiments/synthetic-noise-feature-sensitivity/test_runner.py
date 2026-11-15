"""Pure fixture regressions: these are NOT the 36 research fits."""
from pathlib import Path
import json
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import SUBSETS
from runner import (FAMILIES, OBJECTIVE, UPDATES, JOB_GRID, BETA, LAMBDA, H,
                    grid, plan, shape_for, initial_logits, fit_trace, main,
                    auth_gate, validation_scores, trace_as_csv)
from engine import ExactMaskedWalk, DT
from train_missing_objectives import train_loss


def small_graph(d=3):
    rng=np.random.default_rng(777+d)
    train=rng.normal(size=(6,d)).astype('float64')
    val=rng.normal(size=(3,d)).astype('float64')
    # Every class is represented, all train/query features observed.
    return ExactMaskedWalk(train,np.array([0,1,2,0,1,2],dtype='int64'),
                           np.ones(d,dtype='float64')*.7,
                           beta=BETA,lam=LAMBDA,horizon=2,query_x=val)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_grid(self):
        self.assertEqual(len(grid()),36)
        self.assertEqual(len(set(grid())),36)
        self.assertEqual([sum(subset.startswith(f'N{k}-') for subset,_ in grid())
                          for k in range(6)], [2,8,8,8,8,2])
        self.assertEqual(plan()['total_update_budget'],1080)
        self.assertEqual(OBJECTIVE,'softplus_JS_tau005')
        self.assertFalse(plan()['test_accessed'])
        self.assertEqual(shape_for(5,'shared'),(6,5))
        self.assertEqual(shape_for(5,'class'),(3,6,5))
        for d in range(3,9):
            for family in FAMILIES:
                z=initial_logits(d,family)
                self.assertEqual(z.dtype,torch.float64)
                self.assertTrue(torch.equal(z,torch.zeros_like(z)))
                self.assertEqual(tuple(z.shape),shape_for(d,family))

    def test_no_fit_default_or_without_independent_authorization(self):
        self.assertEqual(main([]),0)
        with self.assertRaises(PermissionError):
            main(['--execute'])
        with self.assertRaises(PermissionError):
            auth_gate({},source_sha='1'*40,run_token='bad',spec_digest='1'*64)
        g=small_graph()
        with self.assertRaises(PermissionError):
            fit_trace(g,[0,1,2],'shared')
        with self.assertRaises(ValueError):
            shape_for(9,'shared')

    def test_trace_records_are_at_completed_update_count(self):
        g=small_graph(3)
        y=np.array([0,1,2],dtype='int64')
        for family in FAMILIES:
            z,optim,trace=fit_trace(g,y,family,steps=30,fixture=True)
            self.assertEqual(len(trace),31)
            self.assertEqual([row['step'] for row in trace],list(range(31)))
            self.assertEqual([row['update_completed'] for row in trace],list(range(31)))
            self.assertIsNone(trace[0]['adam_update_l2'])
            self.assertIsNone(trace[0]['loss_delta'])
            self.assertTrue(all(row['adam_update_l2']>=0 for row in trace[1:]))
            self.assertTrue(all(row['numeric_status']=='finite' for row in trace))
            self.assertTrue(all(0<=row['validation_BA']<=1 for row in trace))
            self.assertTrue(all(0<=row['validation_Accuracy']<=1 for row in trace))
            self.assertTrue(all(trace[i]['elapsed_s']<=trace[i+1]['elapsed_s']
                                for i in range(30)))
            with torch.no_grad():
                terminal=float(train_loss(g,z,family,OBJECTIVE))
            self.assertAlmostEqual(terminal,trace[-1]['train_softplus_loss'],places=11)
            # There is no 31st optimizer step after terminal measurement.
            pid=optim.param_groups[0]['params'][0]
            self.assertEqual(int(optim.state[pid]['step']),30)
            self.assertIn('step,update_completed,train_softplus_loss',trace_as_csv(trace))

    def test_validation_labels_do_not_enter_candidate_scores(self):
        g=small_graph()
        z=initial_logits(g.d,'class')
        with torch.no_grad():
            s1=validation_scores(g,z,'class').clone()
            s2=validation_scores(g,z,'class').clone()
        self.assertTrue(torch.equal(s1,s2))
        self.assertEqual(tuple(s1.shape),(3,3))
        # Label vectors appear only in external scoring: not an input to prediction.
        self.assertTrue(torch.isfinite(s1).all())

    def test_each_dimension_and_no_validation_training_leak(self):
        with torch.no_grad():
            for d in range(3,9):
                g=small_graph(d)
                for family in FAMILIES:
                    z=initial_logits(d,family)
                    g.check_logits(z,family)
                    self.assertEqual(tuple(validation_scores(g,z,family).shape),(3,3))
        g=small_graph(3)
        z1,_,rows1=fit_trace(g,[0,1,2],'shared',steps=2,fixture=True)
        z2,_,rows2=fit_trace(g,[1,2,0],'shared',steps=2,fixture=True)
        self.assertTrue(torch.equal(z1,z2))
        self.assertEqual([r['train_softplus_loss'] for r in rows1],
                         [r['train_softplus_loss'] for r in rows2])

    def test_fixture_rejects_research_size(self):
        class Fake:
            n=300;d=5;k=3;qx=np.zeros((90,5))
        with self.assertRaises(PermissionError):
            fit_trace(Fake(),np.zeros(90,dtype='int64'),'shared',fixture=True)

if __name__=='__main__':unittest.main()
