"""E-CH4-13 recovery from already-computed 2-fit artifact. NO TRAINING.
Reads only saved snapshots and traces; reconstructs reports, checks fingerprints,
writes manifest, coverage, CSV and SVG. Never imports an optimizer or reads Test.
"""
from __future__ import annotations
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from generate import load_config
from dataset_adapter import dataset_train_val
from snapshot_pipeline import load_snapshot, array_fingerprint, config_fingerprint
from report_snapshot import report_snapshot

NAMES=("sigmoid_JS_margin_tau005","softplus_JS_tau005")
SOURCE_SHA="66a007928e129ad6dd5acffeaa93fc2485ee9775"
CFG=ROOT/"code/data_generator/config.json"

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def finite_status(v):
    if v is None:
        return False
    if isinstance(v,dict):
        return all(finite_status(q) for q in v.values())
    try:
        return bool(np.isfinite(np.asarray(v,dtype=float)).all())
    except (TypeError,ValueError):
        return False

def finalize(folder):
    folder=Path(folder)
    if not folder.is_dir():raise FileNotFoundError(folder)
    config=load_config(CFG);config["seed"]=20261102
    # Does not create or read historical Test split.
    splits=dataset_train_val(config)
    tx,ty=splits["train"];vx,vy=splits["validation"]
    contract=json.loads((ROOT/"code/experiments/COMMON_METRICS_CONTRACT.json").read_text("utf8"))
    runs=[]
    coverage={"schema":contract["schema"],
              "source":"raw E-CH4-13 pilot artifact; zero optimizer steps in recovery",
              "test":{"status":"not_applicable","reason":"Test access explicitly prohibited"},
              "runtime_memory_peak_bytes":{"status":"not_measured","reason":"not instrumented"},
              "objectives":{}}
    common_ids=None
    for name in NAMES:
        path=folder/name
        files={n:path/n for n in ("model.json","distributions.npz","trace.json","metrics_full.json")}
        if not all(v.is_file() for v in files.values()):
            raise ValueError("Incomplete raw run: "+name)
        model,arrays=load_snapshot(path)
        info=model["data"]
        if model["schema"]!="imc-snapshot-v2" or model["family"]!="shared":
            raise ValueError("Unexpected snapshot schema/family")
        if model["provenance"]["code_commit"]!=SOURCE_SHA:
            raise ValueError("Different source commit, forbidden replay")
        if info["test_accessed"] or info["data_seed"]!=20261102 or info["reference_seed"]!=20261107:
            raise ValueError("Data seed or Test-access violation")
        if model["objective"]["name"]!=name or model["objective"]["tau"]!=.005:
            raise ValueError("Objective/tau mismatch")
        pars=model["parameters"]
        if (pars["beta"],pars["lambda"],pars["H"],pars["dtype"])!=( .4, .8, 12, "float64"):
            raise ValueError("Walk parameter mismatch")
        ids=arrays["reference_ids"]
        if len(ids)!=50 or len(np.unique(ids))!=50 or np.max(ids)>=len(tx):
            raise ValueError("Reference IDs invalid")
        if (array_fingerprint(tx[ids])!=info["train_reference_x_sha256"]
            or array_fingerprint(ty[ids].astype(np.int64))!=info["train_reference_y_sha256"]
            or array_fingerprint(vx)!=info["validation_x_sha256"]
            or config_fingerprint(CFG)!=info["config_sha256"]):
            raise ValueError("Train/Val/config integrity failure")
        if common_ids is None:common_ids=ids.copy()
        else:np.testing.assert_array_equal(common_ids,ids)
        trace=json.loads(files["trace.json"].read_text("utf8"))
        if len(trace)!=11 or [t["update_completed"] for t in trace]!=list(range(11)):
            raise ValueError("Not exactly 10 optimizer updates per run")
        if not all(np.isfinite(t["gradient_l2"]) and np.isfinite(t["train_objective"])
                   for t in trace):
            raise ValueError("Gradient/loss contains nonfinite value")
        saved=json.loads(files["metrics_full.json"].read_text("utf8"))
        independent=report_snapshot(path)
        for split in ("train","validation"):
            for key in ("balanced_accuracy","ordinary_accuracy","R_JS","R_sqrt_JS","I"):
                if not np.isclose(saved[split][key],independent[split][key],atol=1e-12,rtol=0):
                    raise ValueError(f"Raw/report mismatch {name} {split} {key}")
            if saved[split]["confusion_true_rows_pred_cols"]!=independent[split]["confusion_true_rows_pred_cols"]:
                raise ValueError("Confusion mismatch")
        per={}
        for group in contract["metric_groups"]:
            gname=group["group"]
            if gname=="experiment_specific":continue
            locations=("train",) if group.get("center_only") else ("train","validation")
            for split in locations:
                for mid in group["ids"]:
                    section=saved[split]
                    value=(section.get("center_geometry",{}).get(mid,section.get(mid))
                           if gname=="center_geometry" else section.get(mid))
                    measured=finite_status(value)
                    status=("measured" if measured else
                            ("undefined_or_infinite" if mid in
                             ("R_KL","NLL_KL_simple","NLL_KL_normalized_historical")
                             else "not_measured"))
                    per[f"{split}.{mid}"]={
                        "status":status,
                        "reason":("raw canonical metrics_full.json" if measured
                                  else "canonical reporter field missing/nonfinite; no zero substituted")}
        coverage["objectives"][name]=per
        runs.append({
            "objective":name,"updates":10,"family":"shared","tau":.005,
            "train_BA":saved["train"]["balanced_accuracy"],
            "validation_BA":saved["validation"]["balanced_accuracy"],
            "I":saved["train"]["I"],"R_KL":saved["train"]["R_KL"],
            "R_JS":saved["train"]["R_JS"],"R_sqrt_JS":saved["train"]["R_sqrt_JS"],
            "initial_objective":trace[0]["train_objective"],
            "final_objective":trace[-1]["train_objective"],
            "initial_gradient_l2":trace[0]["gradient_l2"],
            "final_gradient_l2":trace[-1]["gradient_l2"],
            **{n+"_sha256":sha(p) for n,p in files.items()}
        })
    manifest={
       "schema":"e-ch4-13-two-fit-recovered-v1","status":"PILOT_UNAUDITED",
       "source_commit":SOURCE_SHA,"recovery_training_updates":0,
       "n_fits":2,"steps_per_fit":10,"data_seed":20261102,
       "reference_seed":20261107,"reference_ids":common_ids.tolist(),
       "reference_x_sha256":array_fingerprint(tx[common_ids]),
       "validation_x_sha256":array_fingerprint(vx),
       "config_sha256":config_fingerprint(CFG),
       "family":"shared","K":3,"d":5,"n_train":50,"n_validation":len(vx),
       "beta":.4,"lambda":.8,"H":12,"dtype":"float64","optimizer":"Adam",
       "lr":.12,"initialization":"zero","test_accessed":False,
       "runs":runs
    }
    (folder/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    (folder/"coverage.json").write_text(json.dumps(coverage,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    with (folder/"metrics_summary.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
    vals=[float(q["validation_BA"]) for q in runs]
    ys=[250-200*v for v in vals]
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 300">
<rect width="560" height="300" fill="white"/>
<text x="25" y="25" font-size="15">E-CH4-13: Validation BA (1 seed; 10 updates / fit)</text>
<path d="M70 50V250H510" fill="none" stroke="black"/>
<path d="M70 150H510" stroke="gray" stroke-dasharray="4 4"/>
<text x="15" y="155" font-size="13">50%</text><text x="15" y="55" font-size="13">100%</text>
<polyline points="145,{ys[0]:.3f} 420,{ys[1]:.3f}" fill="none" stroke="black"/>
<circle cx="145" cy="{ys[0]:.3f}" r="5"/><circle cx="420" cy="{ys[1]:.3f}" r="5"/>
<text x="99" y="275" font-size="12">Sigmoid-JS</text>
<text x="371" y="275" font-size="12">Softplus-JS</text>
</svg>'''
    (folder/"validation_ba.svg").write_text(svg,encoding="utf8")
    # Check outputs were derived from the same source bytes, with no resampling.
    with (folder/"metrics_summary.csv").open(encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    for row,run in zip(rows,runs):
        if abs(float(row["validation_BA"])-run["validation_BA"])>1e-13:
            raise ValueError("CSV not identical to raw")
    print(json.dumps({"status":"PILOT_UNAUDITED",
                      "recovery_optimizer_updates":0,
                      "runs":runs,
                      "coverage_entries":sum(len(v) for v in coverage["objectives"].values())},
                     ensure_ascii=False))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--raw-dir",type=Path,required=True)
    a=p.parse_args()
    finalize(a.raw_dir)
