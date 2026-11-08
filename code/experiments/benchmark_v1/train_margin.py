"""Train one of four fixed-logit policy families under exact JS margin.

node/node_class are TRAIN-ONLY capacity references; unseen query start logits
do not exist until a functional generator is fitted.
"""
import argparse,json,pathlib,sys,tempfile,time,platform
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from generate import generate,load_config
from engine import ExactMaskedWalk,DT,balanced,jsd,LN2

def prepare(spec,ref_count,ref_seed):
    with tempfile.TemporaryDirectory() as temp:splits=generate(spec,temp,finite_q=False)
    x,_,y=splits["train"];vx,_,vy=splits["validation"]
    if ref_count==len(y):idx=np.arange(len(y))
    else:
        rng=np.random.default_rng(ref_seed)
        k=len(np.unique(y));base=ref_count//k;rest=ref_count%k
        per=[base+(i<rest) for i in range(k)]
        idx=np.concatenate([rng.choice(np.flatnonzero(y==c),per[c],replace=False) for c in range(k)])
    return x[idx],y[idx],vx,vy,idx

def validate(graph,z,family,centers,vy):
    with torch.no_grad():
        if family in ("node","node_class"):return None
        if family=="shared":
            reps=graph.represent_query(z,family)
            score=1-jsd(reps[:,None,:],centers[None,:,:])/LN2
        else:
            score=torch.stack([1-jsd(graph.represent_query(z,family,c),centers[c])/LN2
                               for c in range(graph.k)],dim=1)
        pred=score.argmax(1).numpy()
        return {"balanced_accuracy":float(np.mean([np.mean(pred[vy==c]==c) for c in range(graph.k)])),
                "ordinary_accuracy":float(np.mean(pred==vy)),"predictions":pred.tolist()}

def run(family="shared",steps=50,lr=.12,tau=.005,nref=300,ref_seed=20261107,log_every=10):
    torch.set_num_threads(1)
    if family not in ("shared","class","node","node_class"):raise ValueError(family)
    spec=load_config(ROOT/"code/data_generator/config.json")
    x,y,vx,vy,ids=prepare(spec,nref,ref_seed)
    g=ExactMaskedWalk(x,y,spec["gamma"],query_x=vx)
    shapes={"shared":(g.d+1,g.d),"class":(g.k,g.d+1,g.d),
            "node":(g.n,g.d+1,g.d),"node_class":(g.k,g.n,g.d+1,g.d)}
    z=torch.nn.Parameter(torch.zeros(shapes[family],dtype=DT))
    optimizer=torch.optim.Adam([z],lr=lr)
    start=time.perf_counter();history=[]
    for step in range(steps+1):
        out=g.train_outputs(z,family)
        loss=balanced(torch.sigmoid(-out.margin/tau),g.labels,g.k)
        if step==0 or step==steps or step%log_every==0:
            history.append({"step":step,"loss":float(loss.detach()),"train":g.metrics(out,tau),
                            "validation":validate(g,z,family,out.centers.detach(),vy),
                            "elapsed_s":time.perf_counter()-start})
        if step==steps:break
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if not bool(torch.isfinite(z.grad).all()):raise ValueError("nonfinite gradient")
        optimizer.step()
    return {"schema":"benchmark-v1-pilot",
            "optimizer":{"type":"Adam","lr":lr,"steps":steps,"init":"zero","tau":tau},
            "dataset":{"data_seed":spec["seed"],"ref_seed":ref_seed,"reference_ids":ids.tolist(),
                       "train_count":len(y),"class_counts":np.bincount(y).tolist(),
                       "validation_count":len(vy),"test_used":False},
            "walk":{"beta":.4,"lambda":.8,"H":12,"dtype":"float64","expectation":"exact"},
            "policy":{"family":family,"shape":list(z.shape),"logit_count":z.numel(),
                      "out_of_graph_validation_supported":family in ("shared","class")},
            "history":history,"logits":z.detach().tolist(),
            "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--family",choices=["shared","class","node","node_class"],default="shared")
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--lr",type=float,default=.12)
    p.add_argument("--tau",type=float,default=.005)
    p.add_argument("--nref",type=int,default=300)
    p.add_argument("--ref-seed",type=int,default=20261107)
    p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    res=run(a.family,a.steps,a.lr,a.tau,a.nref,a.ref_seed,a.log_every)
    pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    final=res["history"][-1]
    print(a.family,"loss",round(final["loss"],6),"trainBA",round(final["train"]["balanced_accuracy"],6),
          "valBA",None if final["validation"] is None else round(final["validation"]["balanced_accuracy"],6))
if __name__=="__main__":main()
