"""True generative Bayes benchmark and a SEPARATE posterior-geometry diagnostic.
The Bayes classifier uses P(Y|X_observed), not random-walk label-neighborhood vectors.
Thus posterior R_sqrt_JS/R_JS/R_KL must never be mixed with graph-walk R.
By default, no historical Test data are read; --historical-test is reporting only.
"""
from pathlib import Path
import argparse, json, sys, tempfile
import numpy as np
import torch
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(HERE))
from generate import load_config,generate
from engine import DT, EvalOutputs, jsd, LN2
from metrics import summarize

def observed_log_likelihood(x,mean,cov):
    mask=np.isfinite(x);d=int(mask.sum())
    if not d:
        return 0.
    delta=x[mask]-mean[mask]
    S=cov[np.ix_(mask,mask)]
    sign,logdet=np.linalg.slogdet(S)
    if sign<=0:
        raise ValueError("non-positive Gaussian covariance")
    return -.5*(d*np.log(2*np.pi)+logdet+delta@np.linalg.solve(S,delta))

def oracle(spec,xx):
    ll=np.array([[observed_log_likelihood(x,spec["mu"][c],spec["cov"][c])
                  for c in range(len(spec["pi"]))] for x in xx])
    lp=ll+np.log(spec["pi"])[None,:]
    return np.exp(lp-logsumexp(lp,axis=1,keepdims=True)),ll

def diagnose(seed=20261102,historical_test=False):
    spec=load_config(ROOT/"code/data_generator/config.json")
    spec["seed"]=seed
    with tempfile.TemporaryDirectory() as directory:
        data=generate(spec,directory)
    x,_,y=data["train"]
    post,_=oracle(spec,x)
    centers=np.stack([post[y==c].mean(0) for c in range(len(spec["pi"]))])
    C=torch.as_tensor(centers,dtype=DT)
    result={"status":"diagnostic_true_generating_distribution",
            "data_seed":seed,"class_priors":spec["pi"].tolist(),
            "centers_of_true_posteriors":centers.tolist(),
            "CAUTION":"posterior geometry, not random-walk neighborhood geometry",
            "sets":{},"historical_test_reopened":bool(historical_test)}
    for split in ("train","validation")+ (("test",) if historical_test else ()):
        xx,_,yy=data[split]
        p,ll=oracle(spec,xx)
        y=torch.as_tensor(yy,dtype=torch.long)
        P=torch.as_tensor(p,dtype=DT)
        scores=1-jsd(P[:,None,:],C[None,:,:])/LN2
        correct=scores.gather(1,y[:,None]).squeeze(1)
        rivals=scores.masked_fill(torch.nn.functional.one_hot(y,len(spec["pi"])).bool(),-1e20).max(1).values
        metric=summarize(EvalOutputs(C,scores,correct-rivals,P),y,center_is_class_mean=(split=="train"))
        def accuracy(pred):
            return {"balanced":float(np.mean([(pred[yy==c]==c).mean() for c in range(len(spec["pi"]))])),
                    "ordinary":float(np.mean(pred==yy))}
        result["sets"][split]={"n":len(yy),
            "class_counts":np.bincount(yy,minlength=len(spec["pi"])).tolist(),
            "bayes_MAP":accuracy(p.argmax(-1)),
            "bayes_balanced_likelihood":accuracy(ll.argmax(-1)),
            "nearest_posterior_center":accuracy(scores.argmax(1).numpy()),
            "R_sqrt_JS_posterior":metric["R_sqrt_JS"],
            "R_JS_posterior":metric["R_JS"],
            "R_KL_posterior":metric["R_KL"],
            "R_KL_entropy_identity":metric["R_KL_entropy_identity"],
            "I_posterior_centers":metric["I"],
            "average_true_posterior_prob":float(p[np.arange(len(yy)),yy].mean()),
            "oracle_log_loss":float(-np.log(p[np.arange(len(yy)),yy]).mean())}
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=20261102)
    parser.add_argument("--historical-test",action="store_true")
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result=diagnose(args.seed,args.historical_test)
    destination=Path(args.out);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("saved",destination,"splits",list(result["sets"]))
