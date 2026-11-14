"""Train-size sensitivity of margin-only policy, fixed synthetic train split.

Reproduce: python code/experiments/margin_training/train_size_study.py
Uses original generator; no test data used. 5 overlapping stratified subsets
per n<300, unique full n=300. Balanced accuracies are IN-SAMPLE.
"""
from pathlib import Path
import argparse, json, sys, tempfile, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(ROOT/"code/experiments/diag_max_full300"))
sys.path.insert(0,str(ROOT/"code/experiments/multistep_policy_target_match"))
from generate import load_config, generate, bayes_posterior
from stream_graph import StreamGraph
from run import DT
from train import outputs, balanced, training_metrics, validation_metrics

SIZES=(50,100,150,200,250,300)
SEEDS=(20261107,20261108,20261109,20261110,20261111)

def quota(counts,n):
    wanted=n*np.asarray(counts)/sum(counts)
    out=np.floor(wanted).astype(int)
    remain=n-int(out.sum())
    if remain:
        order=np.argsort(-(wanted-out),kind="stable")
        out[order[:remain]]+=1
    if out.sum()!=n or np.any(out>counts):raise AssertionError((n,counts,out))
    return out

def bal(pred,y,k):
    return float(np.mean([(pred[y==c]==c).mean() for c in range(k)]))

def fit_one(spec,x,y,vx,vy,ids,seed,n,tau,steps,lr):
    torch.set_num_threads(1)
    g=StreamGraph(x,y,spec["gamma"],vx,beta=.4,lam=.8,horizon=12)
    z=torch.nn.Parameter(torch.zeros((g.d+1,g.d),dtype=DT))
    opt=torch.optim.Adam([z],lr=lr)
    start=time.perf_counter()
    history=[]
    checks=set([0,10,20,30,40,steps])
    for t in range(steps+1):
        reps,centers,margin=outputs(g,z,y)
        loss=balanced(torch.sigmoid(-margin/tau),y,g.k)
        if t in checks:
            mt=training_metrics(reps,centers,margin,y,tau)
            record={"step":t,
                "train":{"loss":float(loss.detach()),
                         "train_BA":mt["balanced_accuracy"],
                         "I":mt["I"],"R_JS":mt["R_JS"],
                         "margin_q10":mt["margin_q10"]},
                "validation_BA":validation_metrics(g,z,centers.detach(),vy,tau)["balanced_accuracy"],
                "wall_time_sec":time.perf_counter()-start}
            history.append(record)
        if t==steps:break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(z.grad).all():raise ValueError("Nonfinite gradient")
        opt.step()
    post=bayes_posterior(spec,x)
    # MAP minimizes ordinary expected error; likelihood rule targets balanced accuracy.
    return {"nref":n,"reference_seed":seed,"reference_ids":ids.tolist(),
        "train_counts":np.bincount(y,minlength=g.k).tolist(),
        "oracle_MAP_BA":bal(np.argmax(post,axis=1),y,g.k),
        "oracle_balanced_BA":bal(np.argmax(post/spec["pi"][None,:],axis=1),y,g.k),
        "initial":history[0],"final":history[-1],
        "checkpoints":history,"Z_final":z.detach().tolist(),
        "elapsed_seconds":time.perf_counter()-start}

def aggregate(runs,sizes):
    fields={
       "Bayes_MAP_BA":lambda r:r["oracle_MAP_BA"],
       "Bayes_balanced_BA":lambda r:r["oracle_balanced_BA"],
       "uniform_train_BA":lambda r:r["initial"]["train"]["train_BA"],
       "trained_train_BA":lambda r:r["final"]["train"]["train_BA"],
       "gain_train_BA":lambda r:r["final"]["train"]["train_BA"]-r["initial"]["train"]["train_BA"],
       "uniform_val_BA":lambda r:r["initial"]["validation_BA"],
       "trained_val_BA":lambda r:r["final"]["validation_BA"],
       "initial_loss":lambda r:r["initial"]["train"]["loss"],
       "final_loss":lambda r:r["final"]["train"]["loss"],
       "initial_I":lambda r:r["initial"]["train"]["I"],
       "final_I":lambda r:r["final"]["train"]["I"],
       "initial_R_JS":lambda r:r["initial"]["train"]["R_JS"],
       "final_R_JS":lambda r:r["final"]["train"]["R_JS"]}
    out=[]
    for n in sizes:
        sub=[r for r in runs if r["nref"]==n]
        if not sub:continue
        stats={}
        for key,fun in fields.items():
            vals=np.asarray([fun(r) for r in sub])
            stats[key]={"mean":float(vals.mean()),
                "variance_sample":float(vals.var(ddof=1)) if len(vals)>1 else None,
                "std_sample":float(vals.std(ddof=1)) if len(vals)>1 else None,
                "observations":vals.tolist()}
        out.append({"n_train":n,"n_subsets":len(sub),
                    "counts":sub[0]["train_counts"],"metrics":stats})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=ROOT/"code/experiments/margin_training/results/size_study")
    p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--sizes",nargs="+",type=int,default=list(SIZES))
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--lr",type=float,default=.12)
    p.add_argument("--tau",type=float,default=.005)
    a=p.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=True)
    spec=load_config(ROOT/"code/data_generator/config.json")
    with tempfile.TemporaryDirectory() as tmp:
        splits=generate(spec,tmp,finite_q=False)
    full_x,_,full_y=splits["train"]
    vx,_,vy=splits["validation"]
    counts=np.bincount(full_y,minlength=len(spec["pi"]))
    seeds=SEEDS[:a.seeds]
    runs=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        class_order=[rng.permutation(np.flatnonzero(full_y==c)) for c in range(len(counts))]
        for n in a.sizes:
            if n==len(full_y):
                if seed!=seeds[0]:continue
                ids=np.arange(len(full_y))
            else:
                q=quota(counts,n)
                ids=np.concatenate([class_order[c][:q[c]] for c in range(len(q))])
            path=a.out/f"n{n}_seed{seed}.json"
            if path.exists():
                result=json.loads(path.read_text(encoding="utf-8"))
            else:
                result=fit_one(spec,full_x[ids],full_y[ids],vx,vy,ids,seed,n,
                               a.tau,a.steps,a.lr)
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            runs.append(result)
            print("DONE",seed,n,result["initial"]["train"]["train_BA"],
                  result["final"]["train"]["train_BA"],flush=True)
            summary={"meta":{"base_data_seed":spec["seed"],"reference_seeds":list(seeds),
                "sizes":a.sizes,"train_class_counts":counts.tolist(),
                "sampling":"nested proportional-stratified subsets of one 300-sample dataset",
                "variance":"sample ddof=1 over overlapping reference subsets; not independent datasets; n300 has no variance estimate",
                "training":"one shared Z, zero init, margin sigmoid only, Adam 50 x lr=.12, tau=.005, beta=.4, lambda=.8, H12",
                "test":"not evaluated"},"aggregate":aggregate(runs,a.sizes)}
            (a.out/"summary.json").write_text(
                json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
