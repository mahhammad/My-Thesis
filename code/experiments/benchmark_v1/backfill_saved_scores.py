"""Re-evaluate SAVED benchmark logits without any optimizer step or historical Test access.

Provides train/Validation scores for each class hypothesis and backfills:
- multiclass cross-entropy of JS similarity scores (not a Bayes posterior);
- cross-entropy of -KL(sample^hypothesis||center_hypothesis);
- Chapter-3 standard-deviation-normalized -KL score cross-entropy;
- multiclass Softplus and balanced sigmoid risk on JS decision scores.

All metrics are diagnostic; they do NOT represent separately trained NLL or
Softplus policies. Manifest + raw-file SHA256, class centers, I, R_JS and BA
are checked against saved JSON. The generator below reproduces only the two
permitted splits, so the historical Test is not even instantiated.

Usage, from repo root:
 python code/experiments/benchmark_v1/backfill_saved_scores.py \
   --archive-dir code/experiments/standard-policy-comparison/results \
   --out code/experiments/benchmark_v1/results/posthoc_score_backfill_76.csv

No training is ever performed. Optional --per-sample writes predictions,
margins and every candidate's JS score for later directional-error analysis.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys,zipfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from generate import load_config
from engine import ExactMaskedWalk,DT,LN2,jsd,balanced
from metrics import kl_forward

ARCHIVE_SHA={
 "core_50_16runs.zip":"9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040",
 "extra_25_12runs.zip":"061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3",
 "size_25_24runs_seed20261102.zip":"d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f",
 "size_25_24runs_seeds20261103_20261104.zip":"fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd",
}

def dataset_train_val(spec):
    """Identical first two SeedSequence children of generate(); never generates Test."""
    children=np.random.SeedSequence(spec["seed"]).spawn(3)
    out={}
    for name,child in zip(("train","validation"),children[:2]):
        rng=np.random.default_rng(child)
        n=spec["sizes"][name]
        y=rng.choice(len(spec["pi"]),size=n,p=spec["pi"])
        x=np.empty((n,len(spec["mu"][0])))
        for c in range(len(spec["pi"])):
            ids=np.flatnonzero(y==c)
            if len(ids):
                x[ids]=rng.multivariate_normal(spec["mu"][c],spec["cov"][c],size=len(ids))
        mask=rng.random(x.shape)>=spec["missing"][None,:]
        x[~mask]=np.nan
        out[name]=(x,y)
    return out

def candidate_scores(g,z,family,query=False,centers=None):
    if query and family in ("node","node_class"):
        return None
    if family in ("shared","node"):
        reps=g.represent_query(z,family) if query else g.represent_train(z,family)
        by=[reps]*g.k
    else:
        by=[(g.represent_query(z,family,c) if query else
             g.represent_train(z,family,c)) for c in range(g.k)]
    if centers is None:
        C=torch.stack([by[c][g.labels==c].mean(0) for c in range(g.k)])
    else:
        C=centers
    similarities=torch.stack([1-jsd(by[c],C[c])/LN2 for c in range(g.k)],dim=1)
    divergences=torch.stack([kl_forward(by[c],C[c]) for c in range(g.k)],dim=1)
    return C,similarities,divergences,by

def diagnostics(S,D,y,tau=.005):
    k=S.shape[1]
    truth=S.gather(1,y[:,None]).squeeze(1)
    rival=S.masked_fill(torch.nn.functional.one_hot(y,k).bool(),-1e20).max(1).values
    margin=truth-rival
    kl_scale=D.std(1,correction=0,keepdim=True).clamp_min(1e-14)
    # If all KL candidate distances are identical, normalized logits are 0.
    norm_kl=-(D-D.mean(1,keepdim=True))/kl_scale
    loss=lambda logits:-torch.log_softmax(logits,dim=1).gather(1,y[:,None]).squeeze(1)
    out={
      "balanced_accuracy":float(balanced((S.argmax(1)==y).to(DT),y,k)),
      "NLL_JS_similarity":float(balanced(loss(S),y,k)),
      "NLL_KL_simple":float(balanced(loss(-D),y,k)),
      "NLL_KL_normalized":float(balanced(loss(norm_kl),y,k)),
      "softplus_JS_tau005":float(balanced(
          tau*torch.logsumexp((S-truth[:,None])/tau,dim=1),y,k)),
      "sigmoid_JS_tau005":float(balanced(torch.sigmoid(-margin/tau),y,k)),
      "mean_JS_margin_balanced":float(balanced(margin,y,k)),
    }
    return out,{"score_JS":S.tolist(),"margin_JS":margin.tolist(),
                "prediction_JS":S.argmax(1).tolist(),"actual":y.tolist()}

def run(archive_dir,out,per_sample=False):
    torch.set_num_threads(1)
    specs={};data={};graphs={};records=[];details={}
    for basename,expected in ARCHIVE_SHA.items():
        path=archive_dir/basename
        assert path.is_file(),f"missing original archive: {path}"
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==expected, f"outer SHA256 mismatch: {basename}"
        with zipfile.ZipFile(path) as z:
            manifests=[name for name in z.namelist() if name=="manifest.json"
                       or name.endswith("/manifest.json")]
            assert manifests,basename
            for mf in manifests:
                prefix=mf.removesuffix("manifest.json")
                manifest=json.loads(z.read(mf))
                for item in manifest["runs"]:
                    name=prefix+item["file"]
                    raw=z.read(name)
                    assert hashlib.sha256(raw).hexdigest()==item["sha256"],name
                    run=json.loads(raw);info=run["dataset"]
                    seed=info["data_seed"]
                    if seed not in data:
                        spec=load_config(ROOT/"code/data_generator/config.json")
                        spec["seed"]=seed;specs[seed]=spec
                        data[seed]=dataset_train_val(spec)
                    x,y=data[seed]["train"];vx,vy=data[seed]["validation"]
                    ids=np.asarray(info["reference_ids"],dtype=int)
                    assert len(ids)==info["ref_n"] and len(set(ids))==len(ids)
                    key=(seed,tuple(ids))
                    if key not in graphs:
                        graphs[key]=ExactMaskedWalk(x[ids],y[ids],
                           specs[seed]["gamma"],query_x=vx)
                    g=graphs[key];fam=run["family"]
                    zt=torch.as_tensor(run["raw_logits"],dtype=DT)
                    with torch.no_grad():
                        C,S,D,by=candidate_scores(g,zt,fam)
                        train,train_detail=diagnostics(S,D,g.labels)
                        ref=run["final"]["train"]
                        assert np.max(np.abs(C.numpy()-np.asarray(ref["centers"])))<2e-8,name
                        assert abs(train["balanced_accuracy"]-ref["balanced_accuracy"])<5e-8,name
                        own=torch.stack([by[int(c)][i] for i,c in enumerate(g.labels)])
                        rjs=float(balanced(jsd(own,C[g.labels]),g.labels,g.k))
                        assert abs(rjs-ref["R_JS"])<2e-8,name
                        H=lambda p:-(p*torch.log(p.clamp_min(1e-300))).sum(-1)
                        I=float(H(C.mean(0))-H(C).mean())
                        assert abs(I-ref["I"])<2e-8,name
                        if fam in ("shared","class"):
                            _,Sv,Dv,_=candidate_scores(g,zt,fam,query=True,centers=C)
                            val,val_detail=diagnostics(Sv,Dv,torch.as_tensor(vy,dtype=torch.long))
                            assert abs(val["balanced_accuracy"]-run["final"]["validation"]["balanced_accuracy"])<5e-8,name
                        else:
                            val=None;val_detail=None
                    row={
                      "archive":basename,"run":name,"data_seed":seed,
                      "n_reference":len(ids),"family":fam,
                      "objective":run["objective"],"tau_training":run["tau"],
                      "steps":run["steps"],"train_I":ref["I"],"train_R_JS":ref["R_JS"],
                    }
                    for stem,section in (("train",train),("validation",val)):
                        for key in ("balanced_accuracy","NLL_JS_similarity","NLL_KL_simple",
                                    "NLL_KL_normalized","softplus_JS_tau005",
                                    "sigmoid_JS_tau005","mean_JS_margin_balanced"):
                            row[stem+"_"+key]=section[key] if section is not None else None
                    records.append(row)
                    if per_sample:details[name]={"train":train_detail,"validation":val_detail}
        print("AUDITED",basename,len([r for r in records if r["archive"]==basename]),flush=True)
    assert len(records)==76, len(records)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)
    if per_sample:
        out.with_suffix(".samples.json").write_text(
            json.dumps(details,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("backfilled",len(records),"saved checkpoints; no optimizer or Test",out)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive-dir",type=Path,
             default=ROOT/"code/experiments/standard-policy-comparison/results")
    parser.add_argument("--out",type=Path,
             default=ROOT/"code/experiments/benchmark_v1/results/posthoc_score_backfill_76.csv")
    parser.add_argument("--per-sample",action="store_true")
    args=parser.parse_args()
    run(args.archive_dir,args.out,args.per_sample)
