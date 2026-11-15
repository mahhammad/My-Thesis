"""Complete Chapter-4 metric bundles from already registered 9+5 exact experiments.

Default dry run. --execute performs a LIGHT deterministic exact re-evaluation,
with NO optimization, NO Monte Carlo and NO Test. Existing CSV outcomes are
required to agree to numerical precision; any disagreement aborts the report.

Usage from repository root:
  python code/experiments/common_metrics_backfill.py
  python code/experiments/common_metrics_backfill.py --execute --mode both --out /tmp/ch4-common-metrics
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = HERE / "benchmark_v1"
MIXTURE = HERE / "fixed-policy-evaluation"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(MIXTURE))
sys.path.insert(0, str(ROOT / "code" / "data_generator"))

from dataset_adapter import dataset_train_val
from engine import DT, ExactMaskedWalk
from experiment_fixed_baselines import experiment_plan
from generate import load_config
from probability_walk import ProbabilityMixtureWalk
from report_snapshot import standard_metrics

BASELINE = HERE / "benchmark_v1/results/pipeline_v1/fixed_baselines_summary_20261109.csv"
MIXTURE_SAVED = MIXTURE / "results/summary.json"
CONTRACT = HERE / "COMMON_METRICS_CONTRACT.json"
ALPHAS = (0., .25, .5, .75, 1.)


def check(actual, expected, label, atol=3e-10):
    if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def scores_for(train_out, val_out, y, vy, tau=.005):
    centers = np.stack([train_out[y == c].mean(0) for c in range(3)])
    n, m = len(y), len(vy)
    train_h = np.broadcast_to(train_out[:, None, :], (n,3,3))
    val_h = np.broadcast_to(val_out[:, None, :], (m,3,3))
    tr = standard_metrics(train_h, centers, y, tau, split="train", per_sample=False)
    va = standard_metrics(val_h, centers, vy, tau, split="validation", per_sample=False)
    assert np.allclose(centers.sum(axis=1),1,atol=1e-12)
    return centers.tolist(), tr, va


def uniform_runs(x,y,vx,vy,config):
    with BASELINE.open(encoding="utf8", newline="") as fp:
        stored={r["id"]:r for r in csv.DictReader(fp) if r["kind"]=="uniform"}
    out=[]
    for job in experiment_plan("uniform"):
        g=ExactMaskedWalk(x,y,config["gamma"],beta=job["beta"],
                          lam=job["lam"],horizon=job["H"],query_x=vx)
        z=torch.zeros((6,5),dtype=DT)
        with torch.no_grad():
            tr_out=g.represent_train(z,"shared").numpy()
            va_out=g.represent_query(z,"shared").numpy()
        centers,tr,va=scores_for(tr_out,va_out,y,vy)
        old=stored[job["id"]]
        for new,prev in [("I","I"),("R_JS","R_JS"),
                         ("R_sqrt_JS","R_sqrt_JS"),("R_KL","R_KL"),
                         ("balanced_accuracy","train_BA")]:
            check(tr[new],old[prev],f"uniform {job['id']} {new}")
        check(va["balanced_accuracy"],old["validation_BA"],job["id"]+" valBA")
        out.append({"run_id":job["id"],"setting":{k:job[k] for k in ("beta","lam","H")},
                    "centers":centers,"center_geometry":tr["center_geometry"],
                    "train":tr,"validation":va})
    if len(out)!=9:
        raise AssertionError("Nine unique Uniform runs required")
    return out


def guided_mixture_runs(x,y,vx,vy,config):
    saved=json.loads(MIXTURE_SAVED.read_text(encoding="utf8"))
    g=ProbabilityMixtureWalk(x,y,config["gamma"],beta=.4,lam=.8,
                             horizon=12,query_x=vx)
    out=[]
    for alpha,old in zip(ALPHAS,saved["results"]):
        check(alpha,old["alpha"],"mixture alpha")
        with torch.no_grad():
            tr_out=g.represent_train_mixture(alpha).numpy()
            va_out=g.represent_validation_mixture(alpha).numpy()
        centers,tr,va=scores_for(tr_out,va_out,y,vy)
        for new,prev in [("I","I_train"),("R_JS","R_JS_train"),
                         ("R_sqrt_JS","R_sqrt_JS_train"),
                         ("R_KL","R_KL_train"),("balanced_accuracy","train_BA")]:
            check(tr[new],old[prev],f"mixture alpha={alpha} {new}")
        check(va["balanced_accuracy"],old["validation_BA"],
              f"mixture alpha={alpha} validation BA")
        out.append({"run_id":f"mixture_alpha_{alpha:.2f}",
                    "setting":{"alpha":alpha,"beta":.4,"lam":.8,"H":12},
                    "centers":centers,"center_geometry":tr["center_geometry"],
                    "train":tr,"validation":va})
    if len(out)!=5:
        raise AssertionError("Five Guided mixture runs required")
    return out


def run(mode,output):
    config=load_config(ROOT/"code/data_generator/config.json")
    data=dataset_train_val(config)
    x,y=data["train"]
    vx,vy=data["validation"]
    assert x.shape==(300,5) and vx.shape==(90,5)
    assert np.array_equal(np.bincount(y,minlength=3),[72,147,81])
    assert np.array_equal(np.bincount(vy,minlength=3),[13,49,28])
    torch.set_num_threads(1)
    contract=json.loads(CONTRACT.read_text(encoding="utf8"))
    results={}
    if mode in ("uniform","both"):
        results["uniform"]=uniform_runs(x,y,vx,vy,config)
    if mode in ("mixture","both"):
        results["mixture"]=guided_mixture_runs(x,y,vx,vy,config)
    output.mkdir(parents=True,exist_ok=True)
    for label,runs in results.items():
        bundle={
            "schema":"ch4-full-common-metrics-backfill-v2",
            "contract_schema":contract["schema"],
            "source":"deterministic exact re-evaluation checked against original saved summaries",
            "mode":label,"has_test_data":False,"fit_policy":False,
            "generated_new_optimization":False,
            "data_seed":int(config["seed"]),"train_n":len(y),"validation_n":len(vy),
            "center_source":"train class means for same policy, fixed on validation",
            "class_weights":"equal 1/K",
            "tau_for_reporting":.005,
            "inference_protocol":"same class-independent walk for all candidate classes",
            "runs":runs
        }
        filename=output/f"{label}_metrics_full.json"
        filename.write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                            encoding="utf8")
        # Emit full numerical data for archival commits without ZIP parsing:
        print("COMMON_METRIC_BUNDLE_"+label.upper()+"_JSON "+json.dumps(bundle,ensure_ascii=False,separators=(",", ":"),allow_nan=False),flush=True)
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--mode",choices=("uniform","mixture","both"),default="both")
    p.add_argument("--out",type=Path,default=HERE/"_temporary_common_metrics")
    a=p.parse_args()
    if a.execute:
        run(a.mode,a.out)
    else:
        print(json.dumps({"status":"dry_run","mode":a.mode,
                          "full_report_runs":{"uniform":9,"mixture":5},
                          "test_accessed":False,"policy_training":False},indent=2))
