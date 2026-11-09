"""Independent checks for the exact expectation and its gradient."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from run import DT, ExactGraph, coefficients, jsd


def small_graph(seed=29,n=18):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,3));x[rng.random(x.shape)<0.17]=np.nan
    y=np.arange(n)%3
    q=rng.normal(size=(5,3));q[rng.random(q.shape)<.15]=np.nan
    return ExactGraph(x,y,np.array([.9,.6,.75]),np.arange(n),q,horizon=9)


def sample_walk(g,z,s,nwalks=30000,seed=42):
    rng=np.random.default_rng(seed)
    out=np.zeros(g.k)
    z=z.detach().numpy()
    x=g.x
    for _ in range(nwalks):
        length=min(rng.geometric(.4),g.H)
        denom=sum(.8**t for t in range(length))
        i=s;prev=g.d
        for t in range(length):
            allowed=[]
            for b in range(g.d):
                if not np.isfinite(x[i,b]):
                    continue
                row=g.base[b,i].numpy().copy()
                if t>=1:
                    row[s]=0.
                if row.sum()>1e-12:allowed.append(b)
            logits=z[prev,allowed]
            p=np.exp(logits-logits.max());p/=p.sum()
            b=int(rng.choice(allowed,p=p))
            row=g.base[b,i].numpy().copy()
            if t>=1:
                row[s]=0
            row/=row.sum()
            j=int(rng.choice(len(row),p=row))
            if j==i or j==s:
                raise AssertionError('forbidden state reached')
            out[g.labels[j]]+=.8**t/denom
            prev=b;i=j
    return out/nwalks


class TestExactMatch(unittest.TestCase):
    def test_coefficients_are_probability_distribution(self):
        a=coefficients();self.assertAlmostEqual(sum(a),1,places=12)
        self.assertAlmostEqual(a[0],0.656647587924167,places=12)

    def test_stochastic_and_origin_exclusion(self):
        g=small_graph()
        for s in range(g.n):
            v=g.transitions[s].numpy()
            self.assertTrue(np.all(v[:,:,s]==0))
            self.assertTrue(np.allclose(np.diagonal(v,axis1=1,axis2=2),0))
        z=torch.zeros((g.d+1,g.d),dtype=DT)
        center=g.center_train(z,torch.arange(g.n),return_all=True)
        self.assertTrue(torch.allclose(center.sum(-1),torch.ones(g.n,dtype=DT)))

    def test_exact_expectation_matches_independent_sampling(self):
        g=small_graph()
        z=torch.tensor([[0.0,1.,-.5],[1.,-.4,.6],[-.7,.9,.1],[.5,-.5,.2]],dtype=DT)
        for s in [0,7]:
            exact=g.center_train(z,torch.tensor([s])).detach().numpy()
            mc=sample_walk(g,z,s,nwalks=12000,seed=20+s)
            self.assertLess(float(np.max(abs(exact-mc))),0.018)

    def test_gradient_vs_centered_difference(self):
        g=small_graph(n=12)
        z=torch.nn.Parameter(torch.randn((g.d+1,g.d),dtype=DT)*.1)
        target=torch.tensor([.9,.05,.05],dtype=DT)
        loss=jsd(g.center_train(z,torch.tensor([0,3,6])),target)
        grad=torch.autograd.grad(loss,z)[0]
        eps=1e-5
        for i,j in [(0,0),(1,1),(g.d,2)]:
            xp=z.detach().clone(); xm=z.detach().clone()
            xp[i,j]+=eps;xm[i,j]-=eps
            fd=(jsd(g.center_train(xp,torch.tensor([0,3,6])),target)-
                jsd(g.center_train(xm,torch.tensor([0,3,6])),target))/(2*eps)
            self.assertLess(abs(float(fd-grad[i,j])),1e-6)


if __name__=='__main__':unittest.main(verbosity=2)
