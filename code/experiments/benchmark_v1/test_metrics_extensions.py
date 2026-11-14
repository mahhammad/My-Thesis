"""Checks for posthoc JS NLL / multiclass Softplus score diagnostics.

Run: python -m unittest -v code/experiments/benchmark_v1/test_metrics_extensions.py
No generation or training, and no Test split.
"""
import pathlib, sys, unittest
import numpy as np
import torch

ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from engine import EvalOutputs,DT,jsd
from metrics import summarize


class ScoreDiagnosticsChecks(unittest.TestCase):
    def test_balanced_js_softmax_nll_and_multiclass_softplus(self):
        C=torch.tensor([[.8,.2],[.1,.9]],dtype=DT)
        labels=torch.tensor([0,0,1],dtype=torch.long)
        scores=torch.tensor([[.9,.2],[.7,.6],[.35,.8]],dtype=DT)
        true=scores.gather(1,labels[:,None]).squeeze(1)
        other=scores.masked_fill(torch.nn.functional.one_hot(labels,2).bool(),-1e20).max(1).values
        own=torch.tensor([[.7,.3],[.75,.25],[.2,.8]],dtype=DT)
        out=EvalOutputs(C,scores,true-other,own)
        tau=.05
        got=summarize(out,labels,tau=tau,center_is_class_mean=False)
        nll=-torch.log_softmax(scores,dim=1).gather(1,labels[:,None]).squeeze(1)
        plus=tau*torch.logsumexp((scores-true[:,None])/tau,dim=1)
        expected_nll=((nll[:2].mean()+nll[2])/2).item()
        expected_plus=((plus[:2].mean()+plus[2])/2).item()
        self.assertAlmostEqual(got["balanced_JS_similarity_NLL"],expected_nll,places=12)
        self.assertAlmostEqual(got["balanced_softplus_all_rivals"],expected_plus,places=12)
        self.assertFalse(got["R_KL_entropy_identity_applicable"])
        self.assertIsNone(got["R_KL_entropy_identity"])

    def test_zero_margin_tie(self):
        C=torch.tensor([[.6,.4],[.4,.6]],dtype=DT)
        label=torch.tensor([0,1])
        own=torch.tensor([[.6,.4],[.4,.6]],dtype=DT)
        s=torch.full((2,2),.5,dtype=DT)
        out=EvalOutputs(C,s,torch.zeros(2,dtype=DT),own)
        m=summarize(out,label,tau=.01)
        self.assertAlmostEqual(m["balanced_sigmoid_margin_loss"],.5,places=12)
        self.assertAlmostEqual(m["balanced_JS_similarity_NLL"],np.log(2),places=12)
        self.assertAlmostEqual(m["balanced_softplus_all_rivals"],.01*np.log(2),places=12)

    def test_tau_must_be_positive(self):
        C=torch.tensor([[.6,.4],[.4,.6]],dtype=DT)
        label=torch.tensor([0,1])
        out=EvalOutputs(C,torch.eye(2,dtype=DT),torch.ones(2,dtype=DT),C)
        with self.assertRaisesRegex(ValueError,"tau"):
            summarize(out,label,tau=0)


if __name__=="__main__": unittest.main()
