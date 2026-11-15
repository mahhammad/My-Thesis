"""Controlled Benchmark v1: class-tied initial logits, free node_class mid logits.

The original unconstrained node_class has Z[c,i,d,b] for each training
origin i; no start row exists for unseen queries. Here we tie that row
across all training origins with the SAME candidate class c:
  A[c,b]              -- learned start logits, K x d
  B[c,j,a,b]          -- learned intermediate logits, K x n x d x d
  Z[c,j,d,b] = A[c,b] -- tied starts
  Z[c,j,a,b] = B[c,j,a,b] for a<d

Train and validation data, hyperparameters, mask, exclusion of train origin,
reference nodes, and all K class hypotheses use the SAME Benchmark v1 engine.
No historical Test data is generated or accessed. --execute is mandatory.

Run from repository root:
 python code/experiments/benchmark_v1/train_class_tied_node_intermediate.py
 python code/experiments/benchmark_v1/train_class_tied_node_intermediate.py --execute

The saved logits support replay without retraining. This is a DIFFERENT
family from free node_class; its attainable Train BA may be lower.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time, zipfile
import numpy as np
import torch
from engine import ExactMaskedWalk, DT, LN2, jsd, balanced
from metrics import summarize
from node_class_validation_starts import train_val_only

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EXPECTED_ZIP_SHA="9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040"
BASELINE_FILE="ds20261102_n300_node_class_margin_JSD_tau0.005_s50.json"

def make_model_logits(start,mid,g):
    """Keep the constraint exact in every forward pass; autograd sums gradients."""
    k,n,d=g.k,g.n,g.d
    assert start.shape==(k,d) and mid.shape==(k,n,d,d)
    return torch.cat((mid,start[:,None,None,:].expand(k,n,1,d)),dim=2)

def load_graph(zip_path):
    if hashlib.sha256(zip_path.read_bytes()).hexdigest()!=EXPECTED_ZIP_SHA:
        raise ValueError("Archive SHA256 mismatch")
    with zipfile.ZipFile(zip_path) as ar:
        manifest=json.loads(ar.read("manifest.json"))
        entry=next(x for x in manifest["runs"] if x["file"]==BASELINE_FILE)
        raw=ar.read(BASELINE_FILE)
        assert hashlib.sha256(raw).hexdigest()==entry["sha256"]
        old=json.loads(raw)
    assert (old["family"],old["objective"],old["tau"],old["steps"])==(
        "node_class","margin_JSD",.005,50)
    config=json.loads((ROOT/"code/data_generator/config.json").read_text(encoding="utf8"))
    assert config["seed"]==old["dataset"]["data_seed"]==20261102
    data=train_val_only(config,20261102) # only Train+Validation, NOT Test
    allx,ally=data["train"]
    queryx,queryy=data["validation"]
    ids=np.asarray(old["dataset"]["reference_ids"],dtype=int)
    assert len(ids)==old["dataset"]["ref_n"]==300 and len(np.unique(ids))==300
    x,y=allx[ids],ally[ids]
    assert np.bincount(y,minlength=3).tolist()==old["dataset"]["counts"]
    g=ExactMaskedWalk(x,y,config["kernel_gamma"],query_x=queryx)
    return g,queryy,old,entry

def validate_class_tied(g,z,start,centers,vy,tau):
    """All K class hypotheses; true query labels used ONLY to score results."""
    starts=start[:,None,:].expand(g.k,len(g.qx),g.d)
    reps=[g.represent_query(z,"node_class",c,query_start_logits=starts)
          for c in range(g.k)]
    scores=torch.stack([1-jsd(reps[c],centers[c])/LN2
                        for c in range(g.k)],dim=1)
    labels=torch.as_tensor(vy,dtype=torch.long)
    target=scores.gather(1,labels[:,None]).squeeze(1)
    other=scores.masked_fill(torch.nn.functional.one_hot(labels,g.k).bool(),-1e20).max(1).values
    own=torch.stack([reps[int(c)][i] for i,c in enumerate(labels)])
    from engine import EvalOutputs
    result=summarize(EvalOutputs(centers,scores,target-other,own),
                     labels,tau,center_is_class_mean=False)
    result["status"]="out_of_graph_hybrid_class_tied_start"
    return result

def run(steps,lr,tau,zip_path,out,trace_every):
    if steps<1 or tau<=0 or lr<=0:raise ValueError("positive steps/lr/tau required")
    torch.set_num_threads(1)
    g,vy,old,source=load_graph(zip_path)
    start=torch.nn.Parameter(torch.zeros((g.k,g.d),dtype=DT))
    middle=torch.nn.Parameter(torch.zeros((g.k,g.n,g.d,g.d),dtype=DT))
    opt=torch.optim.Adam([start,middle],lr=lr)
    history=[];began=time.perf_counter()
    for i in range(steps+1):
        z=make_model_logits(start,middle,g)
        o=g.train_outputs(z,"node_class")
        loss=balanced(torch.sigmoid(-o.margin/tau),g.labels,g.k)
        if i in (0,steps) or i%trace_every==0:
            with torch.no_grad():
                train=summarize(o,g.labels,tau)
                val=validate_class_tied(g,z,start,o.centers.detach(),vy,tau)
            summary={"step":i,"train_BA":train["balanced_accuracy"],
                     "validation_BA":val["balanced_accuracy"],
                     "objective_loss":float(loss.detach()),
                     "elapsed_s":round(time.perf_counter()-began,4)}
            history.append(summary)
            print(i,summary,flush=True)
        if i==steps:break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if (start.grad is None or middle.grad is None or
            not bool(torch.isfinite(start.grad).all()) or
            not bool(torch.isfinite(middle.grad).all())):
            raise RuntimeError("Nonfinite gradient")
        opt.step()
    with torch.no_grad():
        z=make_model_logits(start,middle,g)
        o=g.train_outputs(z,"node_class")
        train=summarize(o,g.labels,tau)
        validation=validate_class_tied(g,z,start,o.centers,vy,tau)
    result={
        "schema":"benchmark-v1-class-tied-start-nodeclass-mid-v1",
        "family":"class_tied_start_node_class_middle",
        "description":"For each candidate class, a learned common start row; per-node per-class intermediate policies remain free",
        "dataset":{"seed":20261102,"reference_ids":old["dataset"]["reference_ids"],
                   "train_n":g.n,"validation_n":len(vy),"test_used":False},
        "source":{"archive":zip_path.name,"archive_sha256":EXPECTED_ZIP_SHA,
                  "file":BASELINE_FILE,"sha256":source["sha256"],
                  "baseline_train_BA":old["final"]["train"]["balanced_accuracy"],
                  "baseline_validation_BA":None},
        "optimizer":{"name":"Adam","steps":steps,"lr":lr,"tau":tau,"init":"zeros"},
        "walk":{"beta":.4,"lambda":.8,"horizon":12,"dtype":"float64"},
        "history":history,"train":train,"validation":validation,
        "logits":{"start_by_class":start.detach().tolist(),
                  "intermediate_by_node_class":middle.detach().tolist()},
        "runtime_s":time.perf_counter()-began,
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
        "caveats":["single seed pilot; not a held-out final Test evaluation",
                   "new constrained policy family, not same as full free node_class",
                   "the model knows the candidate hypothesis, never the true validation class",
                   "validation reference graph still contains only Train nodes"]
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("SAVED",out,flush=True)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--steps",type=int,default=50)
    ap.add_argument("--lr",type=float,default=.12)
    ap.add_argument("--tau",type=float,default=.005)
    ap.add_argument("--trace-every",type=int,default=10)
    ap.add_argument("--archive",type=Path,
        default=ROOT/"code/experiments/standard_policy_comparison/results/core_50_16runs.zip")
    ap.add_argument("--out",type=Path,
        default=ROOT/"code/experiments/benchmark_v1/results/class_tied_start_node_mid_50.json")
    args=ap.parse_args()
    if not args.execute:
        print(json.dumps({"status":"dry-run","model":"class-tied start plus free node_class middle",
                          "steps":args.steps,"tau":args.tau,"test_used":False},indent=2))
    else:run(args.steps,args.lr,args.tau,args.archive,args.out,args.trace_every)
