"""FIX-CH4-09: one lightweight walk-count convergence check (NO training/Test).

The 300-walk historical reproduction is ALREADY stored; compare only NEW
budgets 1200 / 4800 / 9600 using unchanged legacy ReferenceGraph. These are
independent deterministic replays at a fixed RNG seed, NOT nested budgets.
Exact H=12 distributions come from the preserved E08 output, not recomputed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import numpy as np

from analyze import (
    CONFIG, E08, HERE, POLICY, OUT, ROOT,
    class_centers, confusion_result, dataset_train_val,
    fingerprint, load_config, load_policy, locate_changes, ReferenceGraph
)

DEST = HERE / "results" / "mc_convergence.json"
BUDGETS = (1200, 4800, 9600)


def run():
    if DEST.exists():
        raise FileExistsError("Previously saved convergence evidence; do not repeat")
    if not OUT.is_file():
        raise FileNotFoundError("Run/verify 300-walk baseline before convergence")
    base = json.loads(OUT.read_text(encoding="utf8"))
    spec = load_config(CONFIG)
    generated = dataset_train_val(spec)
    x,y = generated["train"]
    q,qy = generated["validation"]
    if fingerprint(x) != base["data"]["train_canonical_fingerprint"]:
        raise AssertionError("Generator Train data drift since 300-walk baseline")
    if fingerprint(q) != base["data"]["validation_fingerprint"]:
        raise AssertionError("Generator Validation data drift since 300-walk baseline")
    policy = load_policy(POLICY,5)
    if policy["walks_per_sample"]!=300 or policy["seed"]!=20261103:
        raise AssertionError("Original fixed policy contract changed")
    with np.load(E08/"raw_step_distributions.npz",allow_pickle=False) as d:
        exact_reference = json.loads((E08/"provenance.json").read_text(encoding="utf8"))
        ids = np.asarray(exact_reference["source_reference_ids"],dtype=int)
        exact_train = np.empty((300,3),dtype=float)
        exact_train[ids] = d["uniform_train_exact"]
        exact_val = d["uniform_validation_exact"].copy()
    exact_centers=class_centers(exact_train,y)
    baseline=base["cases"]["mc_val_mc_centers"]
    exact_results=base["cases"]["exact_val_exact_centers"]
    ref_pred=exact_results["predictions"]
    graph=ReferenceGraph(x,y,policy["kernel_gamma"])
    histories=[{
       "walks":300,
       "copied_from_saved_original_replay":True,
       "validation_BA":baseline["balanced_accuracy"],
       "validation_confusion":baseline["confusion_true_rows_pred_cols"],
       "validation_disagreement_count_vs_exact":base["disagreement_counts"]["MC_vs_exact"],
       "validation_TV_mean":base["reference"]["validation_TV_MC_vs_exact"]["mean"],
       "train_TV_mean":base["reference"]["train_TV_MC_vs_exact"]["mean"],
       "center_TV_max":base["reference"]["max_center_tv_replay_vs_exact"],
       "mc_val_exact_centers_BA":base["cases"]["mc_val_exact_centers"]["balanced_accuracy"],
       "exact_val_mc_centers_BA":base["cases"]["exact_val_mc_centers"]["balanced_accuracy"],
       "origin":str(OUT.relative_to(ROOT))
    }]
    for n in BUDGETS:
        started=time.perf_counter()
        p=copy.copy(policy)
        p["walks_per_sample"]=n
        tr,st=graph.representation(x,p,train_indices=np.arange(300))
        val,sv=graph.representation(q,p)
        if st.any() or sv.any():
            raise AssertionError("Unexpected dead-end at high budget")
        centers=class_centers(tr,y)
        a=confusion_result(val,centers,qy)
        b=confusion_result(val,exact_centers,qy)
        c=confusion_result(exact_val,centers,qy)
        diff=locate_changes(a["predictions"],ref_pred,qy)
        hist={
            "walks":n,
            "copied_from_saved_original_replay":False,
            "rng_seed":int(p["seed"]),
            "validation_BA":a["balanced_accuracy"],
            "validation_confusion":a["confusion_true_rows_pred_cols"],
            "validation_predictions":a["predictions"],
            "validation_disagreement_count_vs_exact":len(diff),
            "disagreement_rows":diff,
            "validation_TV_mean":float((.5*np.abs(val-exact_val).sum(axis=1)).mean()),
            "validation_TV_max":float((.5*np.abs(val-exact_val).sum(axis=1)).max()),
            "train_TV_mean":float((.5*np.abs(tr-exact_train).sum(axis=1)).mean()),
            "center_TV_max":float((.5*np.abs(centers-exact_centers).sum(axis=1)).max()),
            "mc_val_exact_centers_BA":b["balanced_accuracy"],
            "exact_val_mc_centers_BA":c["balanced_accuracy"],
            "reference_MC_train_centers":centers.tolist(),
            "MC_train_representations":tr.tolist(),
            "MC_validation_representations":val.tolist(),
            "seconds":float(time.perf_counter()-started),
        }
        histories.append(hist)
        print("FIX09_CONVERGENCE",json.dumps({k:v for k,v in hist.items()
            if k in ("walks","validation_BA","validation_TV_mean","train_TV_mean",
                     "validation_disagreement_count_vs_exact","mc_val_exact_centers_BA",
                     "exact_val_mc_centers_BA","seconds")},ensure_ascii=False),flush=True)
    report={
        "schema":"FIX-CH4-09-MC-convergence-v1",
        "research_new_training":False,"historical_test_accessed":False,
        "provenance_300_walk":str(OUT.relative_to(ROOT)),
        "policy_file":str(POLICY.relative_to(ROOT)),
        "data_seed":20261102,"MC_seed":20261103,
        "beta":.4,"lambda":.8,"H":12,
        "per_walk_normalization":True,
        "budgets_are_nested":False,
        "all_runs_use_identical_newly_generated_train_validation_arrays":True,
        "exact_reference_is_saved_not_recomputed":True,
        "exact_reference_BA":exact_results["balanced_accuracy"],
        "rows":histories,
        "limits":["Historical generated_data NPZ files not stored, so their byte identity cannot be checked",
                  "Larger budgets use the same seed but independent RNG streams, not nested walks",
                  "No independent data seeds or confidence intervals; do not claim theorem of monotonic BA"],
    }
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                    encoding="utf8")
    print("FIX09_CONVERGENCE_SAVED",str(DEST),flush=True)


if __name__=="__main__":
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument("--execute",action="store_true")
    args=cli.parse_args()
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","historical_training":False,"historical_test_accessed":False,
                         "budgets":BUDGETS,"will_reuse_saved_300_walk_and_exact":True}))
    else:
        run()
