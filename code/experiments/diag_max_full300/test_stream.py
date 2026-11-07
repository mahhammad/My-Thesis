import sys, unittest, tempfile
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'code/data_generator'))
sys.path.insert(0,str(ROOT/'code/experiments/multistep_policy_target_match'))
from generate import generate,load_config
from run import ExactGraph,DT
from stream_graph import StreamGraph

class TestExactEquivalence(unittest.TestCase):
 def test_stream_matches_legacy_on_subgraph(self):
  spec=load_config(ROOT/'code/data_generator/config.json')
  with tempfile.TemporaryDirectory() as p:data=generate(spec,p,finite_q=False)
  x,y=data['train'][0][:36],data['train'][2][:36]
  vals=data['validation'][0]
  n=len(x)
  legacy=ExactGraph(x,y,spec['gamma'],np.arange(n),vals,beta=.4,lam=.8,horizon=12)
  newer=StreamGraph(x,y,spec['gamma'],vals,beta=.4,lam=.8,horizon=12)
  idx=np.arange(n)
  for seed in (0,10):
   z=torch.tensor(np.random.default_rng(seed).normal(size=(6,5)),dtype=DT,requires_grad=True)
   for cls in np.unique(y):
    ids=idx[y==cls]
    aa=legacy.center_train(z,ids,return_all=True)
    bb=newer.center_train(z,ids,return_all=True)
    self.assertLess(float(torch.max(torch.abs(aa-bb)).detach()),1e-12)
    target=torch.tensor([.6,.2,.2],dtype=DT)
    loss1=legacy.local_q_loss(z,ids,target)
    loss2=newer.local_q_loss(z,ids,target)
    self.assertLess(abs(float((loss1-loss2).detach())),1e-12)
    g1=torch.autograd.grad(aa.mean(dim=0)[0],z,retain_graph=True)[0]
    g2=torch.autograd.grad(bb.mean(dim=0)[0],z)[0]
    self.assertLess(float((g1-g2).abs().max()),1e-12)
   with torch.no_grad():
    a=legacy.evaluate_val(z);b=newer.evaluate_val(z)
    self.assertLess(float((a-b).abs().max()),1e-12)

if __name__=='__main__':unittest.main()
