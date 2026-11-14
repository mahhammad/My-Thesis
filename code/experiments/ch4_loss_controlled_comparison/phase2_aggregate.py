"""Read-only 12-fit paired report from immutable phase2 snapshots.

No fitting; refuses incomplete grid or corrupted hashes. Test never read.
Run only AFTER protected training and all 12 immutable raw snapshots exist.
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
from snapshot_pipeline import load_snapshot
from report_snapshot import jsd,LN2
from snapshot_pipeline import array_fingerprint
from phase2_frozen_data import load as load_frozen
from phase2_protocol import grid,SEEDS,FAMILIES,LOSSES,MAX_FITS,atomic_new_json

def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def predict(arrays):
    """All hypotheses, no labels used until *after* predictions are frozen."""
    q=arrays["validation_by_hypothesis"]
    c=arrays["centers"]
    scores=1-jsd(q,c[None,:,:])/LN2
    if not np.isfinite(scores).all():raise ValueError("nonfinite classification scores")
    return scores.argmax(1),scores

def validate_run(folder):
    model,arrays=load_snapshot(folder)
    if model["family"] not in FAMILIES:raise ValueError("Disallowed family")
    return model,arrays

def aggregate(base):
    base=Path(base)
    if not base.is_dir():raise FileNotFoundError("No phase2 immutable raw")
    manifest=json.loads((base/"manifest.json").read_text("utf8"))
    if manifest.get("fit_count")!=MAX_FITS or manifest.get("test_accessed") is not False:
        raise ValueError("Incomplete or unauthorized phase2 manifest")
    if {r["run_key"] for r in manifest["runs"]}!={r.key() for r in grid()}:
        raise ValueError("Grid does not match preregistered 12 cells")
    by={}
    for job in grid():
        meta=next(x for x in manifest["runs"] if x["run_key"]==job.key())
        folder=base/job.key()
        for file,label in (("model.json","model_sha256"),
                           ("distributions.npz","distributions_sha256"),
                           ("trace.json","trace_sha256"),
                           ("metrics_full.json","metrics_full_sha256"),
                           ("coverage.json","coverage_sha256")):
            if file_hash(folder/file)!=meta[label]:raise ValueError("File byte corruption: "+job.key()+"/"+file)
        model,arrays=validate_run(folder)
        if model["objective"]["name"]!=job.loss or model["data"]["data_seed"]!=job.seed:
            raise ValueError("Model protocol mismatch")
        (tx,ytrain),(vx,yval)=load_frozen(job.seed)
        if (array_fingerprint(tx)!=model['data']['train_reference_x_sha256']
            or array_fingerprint(vx)!=model['data']['validation_x_sha256']):
            raise ValueError('Frozen Train/Validation and snapshot differ')
        if len(ytrain)!=300 or len(yval)!=90:raise ValueError("Split mismatch")
        pred,scores=predict(arrays)  # *no* Validation labels passed
        if len(pred)!=len(yval):raise ValueError("Prediction count mismatch")
        conf=np.zeros((3,3),dtype=int)
        for a,p in zip(yval,pred):conf[int(a),int(p)]+=1
        ba=float(np.mean(np.diag(conf)/conf.sum(1)))
        if not np.isclose(ba,meta["validation_ba"],atol=1e-12,rtol=0):
            raise ValueError("BA disagreement with raw stored report")
        by[(job.seed,job.family,job.loss)]={
           "pred":pred,"yval":yval,"scores":scores,"meta":meta,
           "confusion":conf,"recall":np.diag(conf)/conf.sum(1),"ba":ba}
    pairs=[]
    for seed in SEEDS:
        for family in FAMILIES:
            a=by[(seed,family,LOSSES[0])]
            b=by[(seed,family,LOSSES[1])]
            if not np.array_equal(a["yval"],b["yval"]):
                raise ValueError("Paired Validation labels differ")
            pairs.append({"data_seed":seed,"family":family,
               "BA_sigmoid":a["ba"],"BA_softplus":b["ba"],
               "paired_delta_validation_BA":a["ba"]-b["ba"],
               "validation_flip_count":int(np.sum(a["pred"]!=b["pred"])),
               "per_class_recall_sigmoid":a["recall"].tolist(),
               "per_class_recall_softplus":b["recall"].tolist(),
               "per_class_recall_delta":(a["recall"]-b["recall"]).tolist(),
               "confusion_sigmoid":a["confusion"].tolist(),
               "confusion_softplus":b["confusion"].tolist(),
               "I_sigmoid":a["meta"]["I"],"I_softplus":b["meta"]["I"],
               "R_KL_sigmoid":a["meta"]["R_KL"],"R_KL_softplus":b["meta"]["R_KL"],
               "R_JS_sigmoid":a["meta"]["R_JS"],"R_JS_softplus":b["meta"]["R_JS"],
               "R_sqrt_JS_sigmoid":a["meta"]["R_sqrt_JS"],
               "R_sqrt_JS_softplus":b["meta"]["R_sqrt_JS"],
               "validation_margin_median_sigmoid":a["meta"]["margin_validation_median"],
               "validation_margin_median_softplus":b["meta"]["margin_validation_median"]})
    stats=[]
    for family in FAMILIES:
        values=[r["paired_delta_validation_BA"] for r in pairs if r["family"]==family]
        if len(values)!=3:raise ValueError("Exactly three independent data seeds required")
        stats.append({"family":family,"n_independent_generators":3,
            "mean_paired_delta_BA":float(np.mean(values)),
            "sd_paired_delta_BA_ddof1":float(np.std(values,ddof=1)),
            "signs":[int(np.sign(v)) for v in values],
            "note":"Descriptive only; no general superiority or strong CI"})
    report={"schema":"e-ch4-13-phase2-paired-summary-v1","status":"PROVISIONAL_AWAITING_FINAL_SCIENTIFIC_QA",
       "n_fits":12,"n_independent_data_seeds":3,"checkpoint":"fixed_after_25",
       "test_accessed":False,"pairs":pairs,"per_family_descriptive":stats}
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw-dir",type=Path,default=HERE/"results/phase2_300_25")
    p.add_argument("--out",type=Path,default=HERE/"results/phase2_300_25"/"paired_summary.json")
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("Paired summary immutable")
    report=aggregate(a.raw_dir)
    atomic_new_json(a.out,report)
    print(json.dumps({"status":report["status"],"n_fits":12,"n_pairs":len(report["pairs"]),
        "n_generators":3,"test_accessed":False}))
if __name__=="__main__":main()
