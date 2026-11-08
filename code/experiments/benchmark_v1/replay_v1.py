"""Unified Benchmark v1 objective grid. Default is dry-run; --execute performs training.
All runs use the same exact-walk engine and score/reporting contract. Train-only
node/node_class tables cannot produce heldout query starts until F is trained.
Do not use the previously opened historical Test split for model selection.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, sys, time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from generate import load_config
from engine import ExactMaskedWalk, EvalOutputs, DT, LN2, jsd, balanced
from metrics import summarize, kl_forward
from train_margin import prepare, validate

FAMILIES=["shared","class","node","node_class"]
TAUS=[.002,.005,.020]

def entropy(x):
    return -(x*torch.log(x.clamp_min(1e-300))).sum(-1)

def objective_outputs(g,z,family,metric):
    if metric=="JSD":
        return g.train_outputs(z,family)
    centers=[]
    if family in ("shared","node"):
        reps=g.represent_train(z,family)
        centers=[reps[g.labels==c].mean(0) for c in range(g.k)]
        by_class=[reps]*g.k
        own=reps
    else:
        by_class=[]
        for c in range(g.k):
            reps=g.represent_train(z,family,c)
            by_class.append(reps)
            centers.append(reps[g.labels==c].mean(0))
        own=torch.stack([by_class[int(c)][i] for i,c in enumerate(g.labels)])
    C=torch.stack(centers)
    if metric=="sqrtJSD":
        S=torch.stack([1-torch.sqrt(jsd(by_class[c],C[c]).clamp_min(1e-14)/LN2)
                       for c in range(g.k)],dim=1)
    elif metric=="KL":
        S=torch.stack([-kl_forward(by_class[c],C[c]) for c in range(g.k)],dim=1)
    else:
        raise ValueError(metric)
    target=S.gather(1,g.labels[:,None]).squeeze(-1)
    rival=S.masked_fill(torch.nn.functional.one_hot(g.labels,g.k).bool(),-1e20).max(1).values
    return EvalOutputs(C,S,target-rival,own)

def objective_loss(g,out,objective,tau):
    C=out.centers
    I=entropy(C.mean(0))-entropy(C).mean()
    if objective=="max_I":
        return -I
    if objective.startswith("margin_"):
        return balanced(torch.sigmoid(-out.margin/tau),g.labels,g.k)
    R=balanced(jsd(out.correct_representations,C[g.labels]),g.labels,g.k)
    if objective=="combined_I_R_margin":
        return balanced(torch.sigmoid(-out.margin/tau),g.labels,g.k)+R-.5*I
    if objective=="min_R_JS":
        return R
    if objective=="min_R_sqrt_JS":
        return balanced(torch.sqrt(jsd(out.correct_representations,C[g.labels]).clamp_min(1e-14)),g.labels,g.k)
    if objective=="min_R_KL":
        return balanced(kl_forward(out.correct_representations,C[g.labels]),g.labels,g.k)
    raise ValueError(objective)

def forward(g,z,family,objective,tau):
    metric={"margin_sqrtJSD":"sqrtJSD","margin_KL":"KL"}.get(objective,"JSD")
    raw=objective_outputs(g,z,family,metric)
    loss=objective_loss(g,raw,objective,tau)
    canonical=raw if metric=="JSD" else g.train_outputs(z,family)
    return loss,canonical,raw

def fit(g,vy,family,objective,steps=50,lr=.12,tau=.005,trace_every=10):
    torch.set_num_threads(1)
    shape={"shared":(g.d+1,g.d),"class":(g.k,g.d+1,g.d),
           "node":(g.n,g.d+1,g.d),"node_class":(g.k,g.n,g.d+1,g.d)}[family]
    z=torch.nn.Parameter(torch.zeros(shape,dtype=DT))
    adam=torch.optim.Adam([z],lr=lr)
    start=time.perf_counter();hist=[]
    for step in range(steps+1):
        objective_value,canon,raw=forward(g,z,family,objective,tau)
        if step in (0,steps) or step%trace_every==0:
            with torch.no_grad():
                train=summarize(canon,g.labels,tau)
                val=validate(g,z,family,canon.centers.detach(),vy,tau)
                hist.append({"step":step,
                    "objective_loss":float(objective_value.detach()),
                    "objective_margin_mean":float(raw.margin.mean().detach()),
                    "train":{k:v for k,v in train.items() if k not in ("predictions","margins")},
                    "validation":{k:v for k,v in val.items() if k not in ("predictions","margins")},
                    "walltime_s":round(time.perf_counter()-start,4)})
        if step==steps:
            break
        adam.zero_grad(set_to_none=True)
        objective_value.backward()
        if z.grad is None or not bool(torch.isfinite(z.grad).all()):
            raise ValueError("nonfinite gradient")
        adam.step()
    return {"family":family,"objective":objective,"tau":tau,"steps":steps,
            "optimizer":{"name":"Adam","lr":lr,"init":"zero"},"history":hist,
            "final":hist[-1],"raw_logits":z.detach().tolist(),
            "elapsed_s":time.perf_counter()-start,
            "validation_status":"not_defined_without_F" if family in ("node","node_class") else "out_of_graph"}

def jobs(mode,steps,size_steps):
    core=[(300,f,"max_I",.005,steps) for f in FAMILIES]
    core.extend((300,f,"margin_JSD",tau,steps) for tau in TAUS for f in FAMILIES)
    extra=[(300,f,obj,.005,steps) for obj in ("margin_sqrtJSD","margin_KL","combined_I_R_margin") for f in FAMILIES]
    size=[(n,f,"margin_JSD",.005,size_steps) for n in (50,100,150,200,250,300) for f in FAMILIES]
    return {"core":core,"extra":extra,"size":size,"all":core+extra+size}[mode]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true",help="opt in to model fitting")
    p.add_argument("--mode",choices=["core","extra","size","all"],default="core")
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--size-steps",type=int,default=25)
    p.add_argument("--data-seed",type=int,default=20261102)
    p.add_argument("--reference-seed",type=int,default=20261107)
    p.add_argument("--out-dir",default="code/experiments/benchmark_v1/results/replays")
    a=p.parse_args()
    plan=jobs(a.mode,a.steps,a.size_steps)
    if not a.execute:
        print(json.dumps({"status":"dry-run","jobs":plan},ensure_ascii=False,indent=2))
        return
    spec=load_config(ROOT/"code/data_generator/config.json")
    spec["seed"]=a.data_seed
    dest=Path(a.out_dir);dest.mkdir(parents=True,exist_ok=True)
    cache={}
    manifest={"schema":"benchmark-v1-replay",
              "status":"single-seed-comparative-no-test",
              "dataset_seed":a.data_seed,"reference_seed":a.reference_seed,
              "beta":.4,"lambda":.8,"H":12,"dtype":"float64",
              "score":"1-JSD/ln2","metrics":["R_sqrt_JS","R_JS","R_KL","I"],
              "train_eval":"leave-one-origin","validation_eval":"outside-train-reference",
              "test_used":False,"py":platform.python_version(),"torch":torch.__version__,
              "runs":[]}
    for i,(n,f,obj,tau,steps) in enumerate(plan,1):
        if n not in cache:
            x,y,vx,vy,ids=prepare(spec,n,a.reference_seed)
            cache[n]=(ExactMaskedWalk(x,y,spec["gamma"],query_x=vx),vy,ids)
        g,vy,ids=cache[n]
        print(f"{i}/{len(plan)} n={n} {f} {obj} tau={tau}",flush=True)
        result=fit(g,vy,f,obj,steps,tau=tau)
        result["dataset"]={"seed":a.data_seed,"reference_ids":ids.tolist(),
                           "class_counts":torch.bincount(g.labels).tolist(),
                           "n_reference":n,"n_validation":len(vy)}
        filename=f"seed{a.data_seed}_n{n}_{f}_{obj}_tau{tau:.3f}_steps{steps}.json"
        payload=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
        (dest/filename).write_text(payload,encoding="utf8")
        tr=result["final"]["train"];v=result["final"]["validation"]
        manifest["runs"].append({"file":filename,"sha256":hashlib.sha256(payload.encode()).hexdigest(),
            "n":n,"family":f,"objective":obj,"tau":tau,"steps":steps,
            "train_BA":tr["balanced_accuracy"],"validation_BA":v.get("balanced_accuracy"),
            "objective_loss":result["final"]["objective_loss"],
            "I":tr["I"],"R_sqrt_JS":tr["R_sqrt_JS"],"R_JS":tr["R_JS"],"R_KL":tr["R_KL"],
            "elapsed_s":result["elapsed_s"]})
        (dest/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    fields=["n","family","objective","tau","steps","train_BA","validation_BA",
            "objective_loss","I","R_sqrt_JS","R_JS","R_KL","elapsed_s","file","sha256"]
    with (dest/"summary.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows([{name:item.get(name) for name in fields} for item in manifest["runs"]])
    print("Done:",len(plan),"runs",flush=True)

if __name__=="__main__":main()
