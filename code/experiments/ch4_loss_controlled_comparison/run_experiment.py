"""E-CH4-13 strictly bounded two-fit pilot, not run unless --execute.
No Test split is generated. This runner is not a multi-seed/grid program.
"""
from __future__ import annotations
import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from generate import load_config
from dataset_adapter import dataset_train_val
from engine import DT, ExactMaskedWalk
from replay_v1 import objective_loss
from train_missing_objectives import train_loss
from snapshot_pipeline import save_snapshot, load_snapshot, array_fingerprint, config_fingerprint
from report_snapshot import report_snapshot

DATA_SEED=20261102
REF_SEED=20261107
N=50
STEPS=10
TAU=.005
BETA=.4
LAM=.8
H=12
LR=.12
OBJECTIVES=("sigmoid_JS_margin_tau005","softplus_JS_tau005")
CFG=ROOT/"code/data_generator/config.json"

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def finite_metric_value(value):
    """Nested margin quantile dicts are measured metrics, not numeric scalars."""
    if value is None:return False
    if isinstance(value,dict):
        return all(finite_metric_value(v) for v in value.values())
    try:
        return bool(np.isfinite(np.asarray(value,dtype=float)).all())
    except (ValueError,TypeError):
        return False

def proportional_ids(y,n,seed):
    y=np.asarray(y,int)
    count=np.bincount(y)
    amounts=count*(n/len(y))
    per=np.floor(amounts).astype(int)
    per[np.argsort(-(amounts-per),kind="stable")[:n-int(per.sum())]]+=1
    if per.sum()!=n or (per>count).any() or (per<1).any():
        raise ValueError("Unusable class proportional split")
    rng=np.random.default_rng(seed)
    result=np.concatenate([rng.permutation(np.flatnonzero(y==c))[:per[c]] for c in range(len(per))])
    if len(set(result.tolist()))!=n:raise ValueError("Reference ID duplication")
    return result

def loss_for(g,z,name):
    if name==OBJECTIVES[0]:
        return objective_loss(g,g.train_outputs(z,"shared"),"margin_JSD",TAU)
    if name==OBJECTIVES[1]:
        return train_loss(g,z,"shared","softplus_JS_tau005")
    raise ValueError("Objective not approved")

def run_pair(output, *, source_sha):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Pilot output must be new and immutable")
    if os.environ.get("E_CH4_13_PREFLIGHT_PASS")!="true":
        raise RuntimeError("Fail closed: explicit current-commit preflight PASS required")
    torch.set_num_threads(1)
    torch.manual_seed(301413)
    spec=load_config(CFG);spec["seed"]=DATA_SEED
    splits=dataset_train_val(spec)  # Train and Validation only; never generate Test
    x,y=splits["train"];vx,vy=splits["validation"]
    ids=proportional_ids(y,N,REF_SEED)
    if len(spec["pi"])!=3 or x.shape[1]!=5 or len(vx)!=90:
        raise ValueError("Unexpected benchmark dataset protocol")
    if not np.isfinite(np.asarray(spec["gamma"])).all():
        raise ValueError("Invalid gamma")
    graph=ExactMaskedWalk(x[ids],y[ids],spec["gamma"],beta=BETA,lam=LAM,horizon=H,query_x=vx)
    if graph.k!=3:raise ValueError("Missing training class")
    output.mkdir(parents=True,exist_ok=False) if not output.exists() else None
    run_rows=[]
    manifest={
        "schema":"e-ch4-13-two-fit-pilot-v1",
        "status":"PILOT_UNAUDITED",
        "base_source_sha":source_sha,
        "data_seed":DATA_SEED,"reference_seed":REF_SEED,
        "reference_ids":ids.tolist(),
        "reference_x_sha256":array_fingerprint(x[ids]),
        "reference_y_sha256":array_fingerprint(y[ids].astype(np.int64)),
        "validation_x_sha256":array_fingerprint(vx),
        "config_sha256":config_fingerprint(CFG),
        "n_train":N,"n_validation":len(vx),"n_classes":graph.k,"n_features":graph.d,
        "family":"shared","dtype":"float64","exact":True,"beta":BETA,"lambda":LAM,"H":H,
        "optimizer":"Adam","learning_rate":LR,"updates_per_fit":STEPS,"zero_init":True,
        "tau_width":TAU,"objectives":list(OBJECTIVES),"test_accessed":False,"runs":[]
    }
    for name in OBJECTIVES:
        z=torch.nn.Parameter(torch.zeros((graph.d+1,graph.d),dtype=DT))
        opt=torch.optim.Adam([z],lr=LR)
        trace=[]
        started=time.perf_counter()
        for step in range(STEPS+1):
            opt.zero_grad(set_to_none=True)
            v=loss_for(graph,z,name)
            if not bool(torch.isfinite(v)):raise FloatingPointError("nonfinite training loss")
            v.backward()
            if z.grad is None or not bool(torch.isfinite(z.grad).all()):
                raise FloatingPointError("nonfinite gradient")
            with torch.no_grad():
                out=graph.train_outputs(z,"shared")
                if not bool(torch.isfinite(out.centers).all()):
                    raise FloatingPointError("nonfinite centers")
                pred=out.scores.argmax(1)
                class_ba=float(torch.stack([
                    (pred[graph.labels==c]==c).to(DT).mean() for c in range(graph.k)
                ]).mean())
                trace.append({
                    "update_completed":step,"train_objective":float(v.detach()),
                    "gradient_l2":float(torch.linalg.vector_norm(z.grad)),
                    "gradient_linf":float(z.grad.abs().max()),
                    "train_BA_at_checkpoint":class_ba,
                    "train_mean_true_margin":float(out.margin.mean())
                })
            if step<STEPS: opt.step()
        elapsed=time.perf_counter()-started
        run_dir=output/name
        save_snapshot(run_dir,graph,z.detach(),"shared",
                      reference_ids=ids,config_path=CFG,
                      config_reference="code/data_generator/config.json",
                      data_seed=DATA_SEED,reference_seed=REF_SEED,objective=name,
                      tau=TAU,beta=BETA,lam=LAM,horizon=H,validation_x=vx,
                      code_commit=source_sha,
                      experiment_metadata={"mission":"E-CH4-13","updates":STEPS,
                           "optimizer":"Adam","lr":LR,"init":"zero",
                           "data_seed":DATA_SEED,"reference_seed":REF_SEED})
        model,arrays=load_snapshot(run_dir)
        report=report_snapshot(run_dir)
        if report["validation"]["status"]!="ok":
            raise AssertionError("Required validation not available")
        conf=np.asarray(report["validation"]["confusion_true_rows_pred_cols"],int)
        counts=conf.sum(axis=1)
        if (counts==0).any():raise AssertionError("Class missing from Validation")
        report["validation"]["per_class_recall"]=(np.diag(conf)/counts).tolist()
        tr_conf=np.asarray(report["train"]["confusion_true_rows_pred_cols"],int)
        report["train"]["per_class_recall"]=(np.diag(tr_conf)/tr_conf.sum(axis=1)).tolist()
        if report["source"]["historical_test_accessed"]:
            raise AssertionError("Unexpected Test access")
        p=run_dir/"metrics_full.json"
        p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
        t=run_dir/"trace.json"
        t.write_text(json.dumps(trace,indent=2,allow_nan=False)+"\n",encoding="utf8")
        run={
            "objective":name,"updates":STEPS,"elapsed_s":elapsed,
            "model_json_sha256":sha(run_dir/"model.json"),
            "distributions_npz_sha256":sha(run_dir/"distributions.npz"),
            "metrics_full_sha256":sha(p),"trace_sha256":sha(t),
            "train_BA":report["train"]["balanced_accuracy"],
            "validation_BA":report["validation"]["balanced_accuracy"],
            "I":report["train"]["I"],"R_JS":report["train"]["R_JS"],
            "R_KL":report["train"]["R_KL"],
            "R_sqrt_JS":report["train"]["R_sqrt_JS"],
            "initial_objective":trace[0]["train_objective"],
            "final_objective":trace[-1]["train_objective"],
            "initial_gradient_l2":trace[0]["gradient_l2"],
            "final_gradient_l2":trace[-1]["gradient_l2"]
        }
        manifest["runs"].append(run)
        run_rows.append(run)
    if len(manifest["runs"])!=2:raise AssertionError("Exactly two fits required")
    contract=json.loads((ROOT/"code/experiments/COMMON_METRICS_CONTRACT.json").read_text("utf8"))
    coverage={"schema":contract["schema"],
              "test":{"status":"not_applicable","reason":"Test access forbidden for E-CH4-13"},
              "optimization":{"status":"measured","reason":"trace.json: gradient L2/Linf per update"},
              "runtime_memory_peak_bytes":{"status":"not_measured","reason":"not instrumented"},
              "objectives":{}}
    for name in OBJECTIVES:
        report=json.loads((output/name/"metrics_full.json").read_text("utf8"))
        per={}
        for group in contract["metric_groups"]:
            gname=group["group"]
            if gname=="experiment_specific":continue
            locations=("train",) if group.get("center_only") else ("train","validation")
            for split in locations:
                for mid in group["ids"]:
                    section=report[split]
                    value=section.get(mid)
                    if gname=="center_geometry":
                        value=section.get("center_geometry",{}).get(mid,section.get(mid))
                    status=("measured" if finite_metric_value(value)
                            else ("undefined_or_infinite" if mid in
                                  ("R_KL","NLL_KL_simple","NLL_KL_normalized_historical")
                                  else "not_measured"))
                    per[f"{split}.{mid}"]={"status":status,
                        "reason":("value in metrics_full.json" if status=="measured"
                                  else "missing/undefined in canonical reporter; no invented zero")}
        coverage["objectives"][name]=per
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    (output/"coverage.json").write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    with (output/"metrics_summary.csv").open("w",encoding="utf8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(run_rows[0]))
        writer.writeheader();writer.writerows(run_rows)
    # SVG generated solely from the same fixed raw summary (no aesthetic inference).
    ba=[row["validation_BA"] for row in run_rows]
    coords=[(140,250-200*ba[0]),(420,250-200*ba[1])]
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 300">
<rect width="560" height="300" fill="white"/>
<text x="30" y="25" font-size="16">E-CH4-13: Validation BA (one seed, two 10-step fits)</text>
<path d="M70 50V250H510" fill="none" stroke="black"/>
<path d="M70 150H510" stroke="gray" stroke-dasharray="4,4"/>
<text x="15" y="155" font-size="13">50%</text><text x="15" y="55" font-size="13">100%</text>
<polyline points="{coords[0][0]},{coords[0][1]:.3f} {coords[1][0]},{coords[1][1]:.3f}" fill="none" stroke="black"/>
<circle cx="{coords[0][0]}" cy="{coords[0][1]:.3f}" r="5"/>
<circle cx="{coords[1][0]}" cy="{coords[1][1]:.3f}" r="5"/>
<text x="95" y="275" font-size="12">Sigmoid-JS</text>
<text x="370" y="275" font-size="12">Softplus-JS</text>
</svg>'''
    (output/"validation_ba.svg").write_text(svg,encoding="utf8")
    # Verify CSV values by directly rereading the immutable metrics.
    for name,row in zip(OBJECTIVES,run_rows):
        d=json.loads((output/name/"metrics_full.json").read_text("utf8"))
        if abs(d["validation"]["balanced_accuracy"]-row["validation_BA"])>1e-13:
            raise AssertionError("Summary not regenerated from raw metrics")
    print(json.dumps({"status":"PILOT_UNAUDITED","fits":2,"updates_per_fit":10,
                      "runs":run_rows,"test_accessed":False},ensure_ascii=False))
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true")
    p.add_argument("--out",type=Path,default=Path("/tmp/e-ch4-13-pilot"))
    p.add_argument("--source-sha",default="not_recorded")
    args=p.parse_args()
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","fits":2,"updates_per_fit":10,
            "train_n":50,"objectives":OBJECTIVES,"test_accessed":False}))
        return
    if args.source_sha in ("not_recorded",""):
        raise ValueError("Exact source commit required")
    run_pair(args.out,source_sha=args.source_sha)
if __name__=="__main__":main()
