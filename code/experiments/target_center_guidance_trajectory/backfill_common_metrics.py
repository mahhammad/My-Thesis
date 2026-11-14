"""Recover FULL Chapter-3 metrics for E-CH4-06 from SAVED representations.

Run from repository root, opt-in:
    python code/experiments/target_center_guidance_trajectory/backfill_common_metrics.py
    python code/experiments/target_center_guidance_trajectory/backfill_common_metrics.py \
        --execute --out-dir /tmp/e-ch4-06-metrics

This never replays Monte Carlo walks, learns policies, fits Validation centers,
or reads historical Test. Existing raw result files are left untouched.
For ALL 49 rounds, center-only metrics are available. For 3 saved checkpoints
only, within-class representation and prediction metrics are recoverable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "benchmark_v1"))
from report_snapshot import standard_metrics, _COMMON_MODULE

EXPERIMENT_ID = "E-CH4-06"
CHECKPOINTS = {
    "uniform": "representations_uniform.json",
    "guided_round0": "representations_guided_round0.json",
    "guided_round4": "representations_guided_round4.json",
}


def check_close(actual, expected, key, tol=4e-10):
    if actual is None or not np.isclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f"{key}: {actual!r} != saved {expected!r}")


def center_summary(centers):
    v = np.asarray(centers, dtype=float)
    g = _COMMON_MODULE.center_metrics(v)
    pair = np.asarray(g["center_jsd"], dtype=float)
    sq = np.sqrt(pair)
    k = len(v)
    off = np.triu_indices(k, 1)
    near = np.where(np.eye(k, dtype=bool), np.inf, sq).min(axis=1)
    return {
        "centers": v.tolist(),
        "T": float(g["T"]), "BT": float(g["BT"]),
        "TNL": None if not np.isfinite(g["TNL"]) else float(g["TNL"]),
        "BTNL": None if not np.isfinite(g["BTNL"]) else float(g["BTNL"]),
        "I": float(g["I"]),
        "pairwise_JS": pair.tolist(),
        "pairwise_sqrt_JS": sq.tolist(),
        "pairwise_JS_similarity": np.asarray(g["center_similarity"]).tolist(),
        "nearest_center_sqrt_JS_per_class": near.tolist(),
        "min_pairwise_sqrt_JS": float(np.min(sq[off])),
        "mean_pairwise_sqrt_JS": float(np.mean(sq[off])),
        "mean_pairwise_JS": float(np.mean(pair[off])),
    }


def load_representation_checkpoint(name, filename, historical_detail):
    data = json.loads((HERE/"results"/filename).read_text(encoding="utf8"))
    centers = np.asarray(data["centers"], dtype=float)
    train, val = data["training"], data["validation"]
    y, vy = np.asarray(train["y"],int), np.asarray(val["y"],int)
    p=np.asarray(train["P"],float)
    if p.shape!=(len(y),3) or centers.shape!=(3,3):
        raise ValueError("invalid saved training distributions or centers")
    recomputed=np.stack([p[y==c].mean(axis=0) for c in range(3)])
    np.testing.assert_allclose(recomputed,centers,atol=1e-12,rtol=0)
    if name=="uniform":
        if "P" not in val:
            raise ValueError("Uniform validation P missing")
        v=np.asarray(val["P"],float)
        vby=np.broadcast_to(v[:,None,:],(len(v),3,3))
        train_by=np.broadcast_to(p[:,None,:],(len(p),3,3))
    else:
        # Training P was generated using the KNOWN origin class. This is
        # ORACLE conditioned and its classification BA is NOT leakage-free.
        # Broadcasting training P across candidate hypotheses gives a
        # DESCRIPTIVE score only, not a valid deployment classifier.
        train_by=np.broadcast_to(p[:,None,:],(len(p),3,3))
        vby=np.asarray(val["candidate_P"],float)
        if vby.shape!=(len(vy),3,3):
            raise ValueError("candidate validation tensor shape mismatch")
    tr=standard_metrics(train_by,centers,y,.005,split="train",per_sample=False)
    va=standard_metrics(vby,centers,vy,.005,split="validation",per_sample=False)
    hist=historical_detail[name]
    for key, old in [
        ("I",hist["geometry"]["I"]),
        ("R_KL",hist["training_within"]["R_KL"]),
        ("R_JS",hist["training_within"]["JSD_within"]),
        ("R_sqrt_JS",hist["training_within"]["sqrt_JSD_within"]),
    ]:
        check_close(tr[key],old,name+" train "+key)
    for key, old in [
        ("R_KL",hist["validation_within_true_class_hypothesis"]["R_KL"]),
        ("R_JS",hist["validation_within_true_class_hypothesis"]["JSD_within"]),
        ("R_sqrt_JS",hist["validation_within_true_class_hypothesis"]["sqrt_JSD_within"]),
        ("balanced_accuracy",hist["validation_prediction"]["balanced_accuracy"]),
        ("ordinary_accuracy",hist["validation_prediction"]["accuracy"]),
    ]:
        check_close(va[key],old,name+" validation "+key)
    geom=center_summary(centers)
    return {
        "case":name,
        "train_origin_class_oracle_conditioned":name!="uniform",
        "training_prediction_comparability":(
            "descriptive_only_origin_class_oracle" if name!="uniform"
            else "proper_class_independent"
        ),
        "validation_protocol":(
            "candidate_class_conditional" if name!="uniform"
            else "class_independent"
        ),
        "center_geometry":geom, "train":tr, "validation":va
    }


def run(out_dir):
    source=HERE/"results"
    history=json.loads((source/"history_00_48.json").read_text("utf8"))
    geo=json.loads((source/"center_geometry_49.json").read_text("utf8"))
    detail=json.loads((source/"detailed_three_cases.json").read_text("utf8"))
    original={x["case"]:x for x in detail["cases"]}
    rounds=[]
    for i,(row,g) in enumerate(zip(history["history"],geo)):
        if row["round"]!=i or g["round"]!=i:
            raise ValueError("round index misalignment")
        stats=center_summary(row["realized_centers"])
        for key,old in [
            ("I",g["geometry"]["I"]),("T",g["geometry"]["T"]),
            ("BT",g["geometry"]["BT"]),("TNL",g["geometry"]["TNL"]),
            ("BTNL",g["geometry"]["BTNL"]),
            ("min_pairwise_sqrt_JS",g["geometry"]["min_center_sqrt_JSD"]),
            ("mean_pairwise_sqrt_JS",g["geometry"]["mean_center_sqrt_JSD"]),
            ("mean_pairwise_JS",g["geometry"]["mean_center_JSD"])
        ]:
            check_close(stats[key],old,f"round {i} {key}")
        rounds.append({
            "round":i,
            "center_geometry":stats,
            "target_realized_JSD":g["mean_target_realized_JSD"],
            "validation_BA_recorded":g["validation_balanced_accuracy"],
            "representation_metrics_status":(
                "measured_three_checkpoints_only" if i in (0,4)
                else "not_measured_no_per_sample_history"
            )
        })
    if len(rounds)!=49:
        raise AssertionError("49 center-only states required")
    checkpoints=[
        load_representation_checkpoint(name, file, original)
        for name,file in CHECKPOINTS.items()
    ]
    # Only 25 BA values were actually stored and observed.
    observed=[x["round"] for x in rounds
              if x["validation_BA_recorded"] is not None]
    if len(observed)!=25:
        raise AssertionError("Expected 25 observed BA among 49 rounds")
    out={
        "schema":"chapter4-complete-available-metrics-v2",
        "experiment_id":EXPERIMENT_ID,
        "source":"byte-preserved historical Monte Carlo reports and saved per-sample representations",
        "no_new_walk":True,
        "no_Test_accessed":True,
        "tau_for_reporting":.005,
        "center_only_rounds":rounds,
        "full_representation_checkpoints":checkpoints,
        "observed_validation_rounds":observed,
        "coverage":{
            "center_only_rounds":"49/49 measured",
            "validation_BA_rounds":"25/49 measured",
            "full_representation_checkpoints":"3/3 saved",
            "other_rounds_R_KL_R_JS_R_sqrt_JS_and_margin":"not_measured; no per-sample representations archived",
            "guided_Train_prediction":"descriptive oracle-origin score only",
            "between_seed_SD":"not_applicable; one correlated Monte Carlo trajectory"
        }
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"metrics_full.json").write_text(
        json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
    with (out_dir/"metrics_summary.csv").open("w",newline="",encoding="utf8") as f:
        keys=["case","I","T","BT","TNL","BTNL","min_pairwise_sqrt_JS",
              "mean_pairwise_sqrt_JS","R_KL_train","R_JS_train",
              "R_sqrt_JS_train","R_KL_validation","R_JS_validation",
              "R_sqrt_JS_validation","BA_validation","Accuracy_validation"]
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in checkpoints:
            c=r["center_geometry"];t=r["train"];v=r["validation"]
            w.writerow({
                "case":r["case"],**{k:c[k] for k in keys[1:8]},
                "R_KL_train":t["R_KL"],"R_JS_train":t["R_JS"],
                "R_sqrt_JS_train":t["R_sqrt_JS"],
                "R_KL_validation":v["R_KL"],"R_JS_validation":v["R_JS"],
                "R_sqrt_JS_validation":v["R_sqrt_JS"],
                "BA_validation":v["balanced_accuracy"],
                "Accuracy_validation":v["ordinary_accuracy"],
            })
    # Provide evidence through the same workflow even if artifact cannot be
    # materialized as a local filesystem path for the connector client.
    print("COMMON_46_SUMMARY_JSON "+json.dumps({
        "schema":out["schema"], "experiment_id":EXPERIMENT_ID,
        "n_center_rounds":len(rounds),"n_observed_BA":len(observed),
        "checkpoints":checkpoints, "coverage":out["coverage"]
    },separators=(",",":"),ensure_ascii=False,allow_nan=False),flush=True)
    print("COMMON_46_ROUNDS_JSON "+json.dumps(rounds,separators=(",",":"),ensure_ascii=False,allow_nan=False),flush=True)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--out-dir",type=Path,default=HERE/"_temporary_metrics")
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({"status":"dry_run",
                          "source":"existing 49 center rounds/3 checkpoints",
                          "replay_monte_carlo":False,"test_accessed":False}))
    else:
        run(a.out_dir)
