"""Targeted missing-objective study, NOT a rerun of existing benchmark training.

Four objectives x shared/class, 25 Adam steps, LR .12, zero logits.
References exactly the saved 300-node size-study ordering. Data seed 20261102,
beta=.4, lambda=.8, H=12 and all other graph rules inherit engine.py.
No Test split is generated or read. --execute is REQUIRED to fit.
Default is a dry-run plan.

Run:
 python code/experiments/benchmark_v1/train_missing_objectives.py
 python code/experiments/benchmark_v1/train_missing_objectives.py --execute

Old 25-step sigmoid baseline must be read from size_25_24runs_seed20261102.zip;
this script deliberately does not retrain that baseline. These pilot results
do not establish global optimum or independent multi-seed generalization.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,zipfile
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from generate import load_config
from engine import ExactMaskedWalk,DT,balanced
from backfill_saved_scores import dataset_train_val,candidate_scores,diagnostics,ARCHIVE_SHA

OBJECTIVES=("NLL_JS_similarity","NLL_KL_simple","NLL_KL_normalized",
            "softplus_JS_tau005")
FAMILIES=("shared","class")
REFERENCE_ZIP=ROOT/"code/experiments/standard-policy-comparison/results/size_25_24runs_seed20261102.zip"
REFERENCE_FILE="ds20261102_n300_shared_margin_JSD_tau0.005_s25.json"

def train_loss(g,z,family,objective):
    C,S,D,_=candidate_scores(g,z,family)
    labels=g.labels
    true_score=S.gather(1,labels[:,None]).squeeze(1)
    if objective=="NLL_JS_similarity":
        v=-torch.log_softmax(S,dim=1).gather(1,labels[:,None]).squeeze(1)
    elif objective=="NLL_KL_simple":
        v=-torch.log_softmax(-D,dim=1).gather(1,labels[:,None]).squeeze(1)
    elif objective=="NLL_KL_normalized":
        sig=D.std(1,correction=0,keepdim=True).clamp_min(1e-14)
        scores=-(D-D.mean(1,keepdim=True))/sig
        v=-torch.log_softmax(scores,dim=1).gather(1,labels[:,None]).squeeze(1)
    elif objective=="softplus_JS_tau005":
        v=.005*torch.logsumexp((S-true_score[:,None])/.005,dim=1)
    else:
        raise ValueError(objective)
    return balanced(v,labels,g.k)

def load_graph():
    assert hashlib.sha256(REFERENCE_ZIP.read_bytes()).hexdigest()==ARCHIVE_SHA[REFERENCE_ZIP.name]
    with zipfile.ZipFile(REFERENCE_ZIP) as z:
        m=json.loads(z.read("manifest.json"))
        entry=next(r for r in m["runs"] if r["file"]==REFERENCE_FILE)
        raw=z.read(REFERENCE_FILE)
        assert hashlib.sha256(raw).hexdigest()==entry["sha256"]
        source=json.loads(raw)
    info=source["dataset"]
    assert info["data_seed"]==20261102 and info["ref_n"]==300
    spec=load_config(ROOT/"code/data_generator/config.json")
    spec["seed"]=20261102
    data=dataset_train_val(spec)
    x,y=data["train"];vx,vy=data["validation"]
    ids=np.asarray(info["reference_ids"],dtype=int)
    g=ExactMaskedWalk(x[ids],y[ids],spec["gamma"],query_x=vx)
    ref={"archive":REFERENCE_ZIP.name,"checkpoint":REFERENCE_FILE,
         "checkpoint_sha256":hashlib.sha256(raw).hexdigest(),
         "reference_ids":ids.tolist(),"data_seed":20261102}
    return g,vy,ref

def one_run(g,vy,reference,family,objective,steps=25):
    shape=((g.d+1,g.d) if family=="shared" else (g.k,g.d+1,g.d))
    z=torch.nn.Parameter(torch.zeros(shape,dtype=DT))
    optimizer=torch.optim.Adam([z],lr=.12)
    start=time.perf_counter()
    init=float(train_loss(g,z,family,objective).detach())
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss=train_loss(g,z,family,objective)
        loss.backward()
        if z.grad is None or not bool(torch.isfinite(z.grad).all()):
            raise ValueError(f"non-finite grad: {family} {objective} at {step}")
        optimizer.step()
    final_loss=float(train_loss(g,z,family,objective).detach())
    with torch.no_grad():
        C,S,D,_=candidate_scores(g,z,family)
        train,_=diagnostics(S,D,g.labels)
        _,Sv,Dv,_=candidate_scores(g,z,family,query=True,centers=C)
        val,_=diagnostics(Sv,Dv,torch.as_tensor(vy,dtype=torch.long))
    return {"schema":"benchmark-v1-new-objectives-v1",
            "objective":objective,"family":family,"steps":steps,"tau":.005,
            "init":"zero","lr":.12,"initial_objective":init,
            "final_objective":final_loss,
            "elapsed_s":time.perf_counter()-start,
            "train":train,"validation":val,
            "centers":C.detach().tolist(),
            "raw_logits":z.detach().tolist(),
            "source_reference":reference,
            "test_accessed":False,
            "note":"new objective trained at 25 steps; compare same-budget stored sigmoid baseline"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true")
    p.add_argument("--family",action="append",choices=FAMILIES)
    p.add_argument("--steps",type=int,default=25)
    p.add_argument("--out-dir",type=Path,
          default=ROOT/"code/experiments/benchmark_v1/results/new_objectives_25")
    a=p.parse_args()
    if a.steps<1:raise ValueError("steps must be positive")
    choices=a.family or FAMILIES
    plan=[{"family":f,"objective":obj,"steps":a.steps}
          for f in choices for obj in OBJECTIVES]
    if not a.execute:
        print(json.dumps({"status":"DRY_RUN","jobs":plan,"test_accessed":False},
                         indent=2))
        return
    torch.set_num_threads(1)
    graph,vy,reference=load_graph()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={"schema":"benchmark-v1-new-objectives-manifest",
              "status":"single-seed-pilot-no-test",
              "reference":reference,"steps":a.steps,"lr":.12,
              "objective_set":list(OBJECTIVES),"runs":[]}
    for i,item in enumerate(plan,1):
        f=item["family"];obj=item["objective"]
        run=one_run(graph,vy,reference,f,obj,a.steps)
        filename=f"{f}_{obj}_s{a.steps}.json"
        payload=json.dumps(run,ensure_ascii=False,indent=2)+"\n"
        (a.out_dir/filename).write_text(payload,encoding="utf8")
        manifest["runs"].append({
            "file":filename,"sha256":hashlib.sha256(payload.encode()).hexdigest(),
            "family":f,"objective":obj,"steps":a.steps,
            "train_BA":run["train"]["balanced_accuracy"],
            "validation_BA":run["validation"]["balanced_accuracy"],
            "train_NLL_KL":run["train"]["NLL_KL_simple"],
            "validation_NLL_KL":run["validation"]["NLL_KL_simple"],
            "elapsed_s":run["elapsed_s"]})
        (a.out_dir/"manifest.json").write_text(
           json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        print(i,"/",len(plan),f,obj,"validation_BA",
              run["validation"]["balanced_accuracy"],flush=True)

if __name__=="__main__":main()
