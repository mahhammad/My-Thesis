"""Review-first benchmark stage-1 runner. Default: dry run; --execute required."""
import argparse,datetime as dt,hashlib,json,pathlib
from train_margin import run
FAMILIES=["shared","class","node","node_class"]

def plan(steps=50,nref=300,ref_seed=20261107,tau=.005,lr=.12):
    return [dict(family=f,steps=steps,nref=nref,ref_seed=ref_seed,tau=tau,lr=lr,
                 objective="balanced sigmoid JSD margin",walk="exact capped beta=.4 lam=.8 H=12",
                 validation="not_defined_train_only" if f in ("node","node_class") else "out_of_graph")
            for f in FAMILIES]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true",help="fit models; otherwise only print the plan")
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--nref",type=int,default=300)
    p.add_argument("--ref-seed",type=int,default=20261107)
    p.add_argument("--tau",type=float,default=.005)
    p.add_argument("--lr",type=float,default=.12)
    p.add_argument("--family",action="append",choices=FAMILIES)
    p.add_argument("--out-dir",default="code/experiments/benchmark_v1/results/stage1")
    a=p.parse_args()
    jobs=[j for j in plan(a.steps,a.nref,a.ref_seed,a.tau,a.lr)
          if not a.family or j["family"] in a.family]
    if not a.execute:
        print(json.dumps({"status":"dry-run","jobs":jobs},ensure_ascii=False,indent=2))
        return
    d=pathlib.Path(a.out_dir);d.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for job in jobs:
        result=run(job["family"],job["steps"],job["lr"],job["tau"],job["nref"],job["ref_seed"])
        path=d/(job["family"]+".json")
        payload=json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n"
        path.write_text(payload,encoding="utf-8")
        manifest.append({**job,"file":path.name,"sha256":hashlib.sha256(payload.encode()).hexdigest(),
                         "final":result["history"][-1],
                         "note":"Validation undefined for free node/node_class tables"})
        print("saved",path)
    (d/"manifest.json").write_text(json.dumps({
        "schema":"benchmark-v1-stage1",
        "generated_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "status":"pilot-not-final-independent-test","jobs":manifest
    },ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")

if __name__=="__main__":main()
