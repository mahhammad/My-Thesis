"""Regression tests for center inverse solver, including cost and safeguards."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from run import DT, jsd
from bootstrap_inverse import benchmark, damped_gauss_newton, make_problem


class TestBootstrapInverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.graph, cls.labels, cls.targets, cls.teachers, _ = make_problem(20261107, 12)

    def test_target_is_reachable_by_witness(self):
        g=self.graph
        for c in range(g.k):
            i=np.flatnonzero(self.labels==c)
            z=torch.tensor(self.teachers[c],dtype=DT)
            actual=g.center_train(z,i)
            np.testing.assert_allclose(actual.detach().numpy(),self.targets[c],atol=1e-13)
            self.assertLess(float(jsd(actual,torch.tensor(self.targets[c],dtype=DT))),1e-12)

    def test_center_jacobian_matches_finite_difference(self):
        g=self.graph;c=0
        idx=np.flatnonzero(self.labels==c)
        z=torch.randn((g.d+1,g.d),dtype=DT)*.12
        z.requires_grad_(True)
        v=g.center_train(z,idx)[:-1]
        grad=torch.autograd.grad(v[0],z)[0]
        h=1e-5
        for a,b in [(0,0),(1,2),(g.d,0)]:
            p=z.detach().clone();m=z.detach().clone()
            p[a,b]+=h;m[a,b]-=h
            fp=g.center_train(p,idx)[0]
            fm=g.center_train(m,idx)[0]
            self.assertAlmostEqual(grad[a,b].item(),((fp-fm)/(2*h)).item(),places=6)

    def test_gn_never_accepts_worse_actual_jsd(self):
        g=self.graph;c=0
        idx=np.flatnonzero(self.labels==c)
        u=torch.tensor(self.targets[c],dtype=DT)
        z,t,counters,trace=damped_gauss_newton(g,idx,u,
                        torch.zeros((g.d+1,g.d),dtype=DT),10,1e-9)
        errs=[r['error'] for r in trace]
        self.assertTrue(all(a>=b-1e-13 for a,b in zip(errs,errs[1:])))
        self.assertAlmostEqual(errs[-1],jsd(g.center_train(z,idx),u).item(),places=12)
        self.assertEqual(counters['jacobian'],len(trace)-1)
        self.assertGreaterEqual(counters['exact_forward'],counters['jacobian']+1)
        self.assertGreater(t,0)

    def test_methods_report_cost_and_feasible_target(self):
        result=benchmark(seed=20261107,nref=12,q_steps=3,
                         exact_steps=4,bootstrap_steps=3,tol=1e-9)
        for method in ('q_only','exact_adam','q_bootstrap_gn'):
            info=result['summary'][method]
            self.assertTrue(np.isfinite(info['mean_JSD']))
            self.assertGreater(info['wall_s_total'],0)
        self.assertEqual(result['summary']['q_only']['cost_total']['q_grad'],9)
        self.assertEqual(result['summary']['q_bootstrap_gn']['cost_total']['q_grad'],9)
        self.assertEqual(len(result['per_class']),3)


if __name__=='__main__':
    unittest.main(verbosity=2)
