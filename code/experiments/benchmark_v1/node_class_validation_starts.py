"""Validation diagnostic for SAVED node_class tau=.005 policy; NO training and NO Test.

One candidate start distribution per class hypothesis, for every Validation
example. Donors are always labeled TRAIN nodes of the candidate class, not
ground-truth Validation labels. All intermediate transitions use frozen logits.
The local rule is a *new hybrid policy*, not automatic generalization of nK.

python code/experiments/benchmark_v1/node_class_validation_starts.py \
 --archive code/experiments/benchmark_v1/results/core_50_16runs.zip \
 --out code/experiments/benchmark_v1/results/node_class_starts_tau005.json
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys, zipfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from engine import ExactMaskedWalk, DT, LN2, jsd

ZIP_SHA="9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040"
CHECKPOINT="ds20261102_n300_node_class_margin_JSD_tau0.005_s50.json"
METHODS=("uniform","class_logit_mean","class_probability_mean",
         "class_local_5","class_local_15")

def train_val_only(cfg,seed):
    """Equivalent to the first two generate.py splits; never generate Test."""
    pi=np.asarray(cfg["class_priors"],float)
    mu=np.asarray(cfg["means"],float)
    sd=np.asarray(cfg["stds"],float)
    corr=np.asarray(cfg["correlation"],float)
    cov=sd[:,:,None]*corr[None,:,:]*sd[:,None,:]
    missing=np.asarray(cfg["missing_percent"],float)/100
    children=np.random.SeedSequence(seed).spawn(3)
    splits={}
    for name,rng_seed in zip(("train","validation"),children[:2]):
        rng=np.random.default_rng(rng_seed)
        n=cfg["split_sizes"][name]
        y=rng.choice(len(pi),size=n,p=pi)
        x=np.empty((n,mu.shape[1]))
        for c in range(len(pi)):
            ii=np.flatnonzero(y==c)
            if len(ii):x[ii]=rng.multivariate_normal(mu[c],cov[c],size=len(ii))
        x[rng.random(x.shape)<missing[None,:]]=np.nan
        splits[name]=(x,y)
    return splits

def softmax_valid(a,mask):
    a=np.where(mask,a,-1e25)
    a=a-a.max(axis=-1,keepdims=True)
    p=np.exp(a)*mask
    return p/np.maximum(p.sum(axis=-1,keepdims=True),1e-300)

def start_probabilities(z,x,y,q,method):
    """K x n_validation x d. Class labels are TRAIN only; never query truth."""
    k,n,dp1,d=z.shape
    assert dp1==d+1 and x.shape==(n,d) and q.shape[1]==d
    refmask=np.isfinite(x); qmask=np.isfinite(q)
    assert qmask.any(1).all()
    scale=np.nanstd(x,axis=0)
    scale=np.maximum(np.where(np.isfinite(scale),scale,0),1e-3)
    result=np.zeros((k,len(q),d),float)
    for c in range(k):
        donors=np.flatnonzero(y==c)
        assert len(donors)>0
        logits=z[c,donors,d,:]
        ps=softmax_valid(logits,refmask[donors])
        global_p=ps.mean(0)
        for j,query in enumerate(q):
            avail=qmask[j]
            if method=="uniform":
                p=np.ones(d)
            elif method=="class_logit_mean":
                p=softmax_valid(logits.mean(0),avail)
            elif method=="class_probability_mean":
                p=global_p.copy()
            elif method.startswith("class_local_"):
                knn=int(method.removeprefix("class_local_"))
                overlap=refmask[donors]&avail[None,:]
                n_common=overlap.sum(1)
                delta=(np.where(overlap,x[donors],0)-
                       np.where(avail,query,0)[None,:])/scale[None,:]
                distance=((delta**2)*overlap).sum(1)/np.maximum(n_common,1)
                distance+=.25*(1-n_common/max(1,int(avail.sum())))
                distance=np.where(n_common>0,distance,np.inf)
                order=np.argsort(distance,kind="stable")[:min(knn,len(donors))]
                finite=np.isfinite(distance[order])
                if not finite.any():
                    p=global_p.copy()
                else:
                    order=order[finite]
                    selected=distance[order]
                    weights=np.exp(-(selected-selected.min()))
                    weights/=weights.sum()
                    p=weights@ps[order]
            else:
                raise ValueError(method)
            p=np.where(avail,p,0)
            if p.sum()<=1e-15:p=avail.astype(float)
            result[c,j]=p/p.sum()
    assert np.isfinite(result).all()
    assert np.allclose(result.sum(2),1,atol=1e-12)
    assert np.all(result[~np.broadcast_to(qmask[None,:,:],result.shape)]==0)
    return result

def evaluate(g,z,centers,x,y,q,qy):
    yy=np.asarray(qy,int);k=g.k
    assert np.all(np.bincount(yy,minlength=k)>0)
    zt=torch.as_tensor(z,dtype=DT)
    C=torch.as_tensor(centers,dtype=DT)
    output={}
    for method in METHODS:
        p=start_probabilities(z,x,y,q,method)
        starts=torch.as_tensor(np.log(np.maximum(p,1e-300)),dtype=DT)
        with torch.no_grad():
            scores=torch.stack([
                1-jsd(g.represent_query(zt,"node_class",c,
                         query_start_logits=starts),C[c])/LN2
                for c in range(k)],dim=1).cpu().numpy()
        prediction=scores.argmax(1)
        recall=[float((prediction[yy==c]==c).mean()) for c in range(k)]
        confusion=np.zeros((k,k),int)
        for a,b in zip(yy,prediction):confusion[a,b]+=1
        r={
          "method":method,
          "validation_balanced_accuracy":float(np.mean(recall)),
          "validation_ordinary_accuracy":float(np.mean(prediction==yy)),
          "per_class_recall":recall,
          "confusion_true_rows_pred_cols":confusion.tolist(),
          "n_correct":int(np.sum(prediction==yy)),
          "n_validation":len(yy),
          "validation_predicted_labels":prediction.tolist(),
          "validation_candidate_scores_JS":scores.tolist(),
          "validation_start_probabilities":p.tolist(),
        }
        output[method]=r
        print(method,"BA",round(r["validation_balanced_accuracy"],8),
              "accuracy",round(r["validation_ordinary_accuracy"],8),flush=True)
    return output

def main(archive,config,out):
    torch.set_num_threads(1)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==ZIP_SHA
    with zipfile.ZipFile(archive) as ar:
        manifest=json.loads(ar.read("manifest.json"))
        entry=next(v for v in manifest["runs"] if v["file"]==CHECKPOINT)
        raw=ar.read(CHECKPOINT)
        assert hashlib.sha256(raw).hexdigest()==entry["sha256"]
        ck=json.loads(raw)
    assert ck["family"]=="node_class" and ck["objective"]=="margin_JSD"
    assert ck["tau"]==.005 and ck["steps"]==50
    cfg=json.loads(config.read_text(encoding="utf8"))
    assert cfg["seed"]==ck["dataset"]["data_seed"]==20261102
    splits=train_val_only(cfg,cfg["seed"])
    allx,ally=splits["train"];q,qy=splits["validation"]
    idx=np.asarray(ck["dataset"]["reference_ids"],int)
    assert len(idx)==ck["dataset"]["ref_n"]==300 and len(set(idx))==300
    x,y=allx[idx],ally[idx]
    assert np.bincount(y,minlength=3).tolist()==ck["dataset"]["counts"]
    g=ExactMaskedWalk(x,y,cfg["kernel_gamma"],query_x=q)
    z=np.asarray(ck["raw_logits"],float)
    assert z.shape==(g.k,g.n,g.d+1,g.d)
    centers=np.asarray(ck["final"]["train"]["centers"],float)
    assert ck["final"]["train"]["balanced_accuracy"]==1
    scores=evaluate(g,z,centers,x,y,q,qy)
    report={
      "schema":"benchmark-node-class-start-v1",
      "status":"frozen-checkpoint-new-hybrid-start-rules-no-training-no-Test",
      "checkpoint":CHECKPOINT,"checkpoint_sha256":entry["sha256"],
      "archive_sha256":ZIP_SHA,"data_seed":cfg["seed"],
      "n_reference":len(x),"n_validation":len(q),
      "n_validation_by_class":np.bincount(qy,minlength=g.k).tolist(),
      "training_tau":ck["tau"],"training_steps":ck["steps"],
      "train_balanced_accuracy_existing":1.0,
      "local_distance":"mean standardized squared distance across jointly observed features + .25 missing overlap penalty; weights exp(-distance); k=5/15",
      "labels_for_start":"only labeled Train donors conditioned on candidate hypothesis c; never validation labels",
      "test_used":False,"results":scores,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    with out.with_suffix(".csv").open("w",newline="",encoding="utf8") as f:
        rows=[{key:item[key] for key in ("method","validation_balanced_accuracy",
                  "validation_ordinary_accuracy","per_class_recall","n_correct","n_validation")}
              for item in scores.values()]
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,default=ROOT/"code/experiments/benchmark_v1/results/core_50_16runs.zip")
    p.add_argument("--config",type=Path,default=ROOT/"code/data_generator/config.json")
    p.add_argument("--out",type=Path,default=ROOT/"code/experiments/benchmark_v1/results/node_class_starts_tau005.json")
    a=p.parse_args()
    main(a.archive,a.config,a.out)
