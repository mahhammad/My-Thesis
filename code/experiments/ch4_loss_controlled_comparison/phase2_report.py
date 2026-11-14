"""Canonical COMMON metrics from immutable phase2 Train/Validation-only inputs.

Avoids unrelated NumPy/OpenBLAS host regenerated X mismatch: verifies raw frozen
X,Y SHA256 against snapshot metadata, canonical Train centers, then uses
benchmark_v1.report_snapshot.standard_metrics unchanged. Never Test.
"""
from __future__ import annotations
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
from snapshot_pipeline import load_snapshot,array_fingerprint,config_fingerprint
from report_snapshot import standard_metrics,REPORT_SCHEMA
from phase2_frozen_data import load as load_frozen,CFG

def report_from_frozen(run_dir,seed):
    run_dir=Path(run_dir)
    model,arrays=load_snapshot(run_dir)
    (tx,ty),(vx,vy)=load_frozen(seed)
    info=model["data"]
    ids=np.asarray(arrays["reference_ids"],dtype=np.int64)
    if len(ids)!=300 or not np.array_equal(ids,np.arange(300)):
        raise ValueError("Phase2 must use ordered full300 Train reference IDs")
    if (info["data_seed"]!=seed or info["train_reference_x_sha256"]!=array_fingerprint(tx[ids])
        or info["train_reference_y_sha256"]!=array_fingerprint(ty[ids])
        or info["validation_x_sha256"]!=array_fingerprint(vx)
        or info["config_sha256"]!=config_fingerprint(CFG)):
        raise ValueError("Frozen dataset / snapshot provenance mismatch")
    if len(vy)!=90 or model["prediction"]["validation_status"]!="available":
        raise ValueError("Only labeled held-out Validation of 90 allowed")
    c=arrays["centers"]
    by=arrays["train_by_hypothesis"]
    expected=np.stack([by[ty==j,j,:].mean(0) for j in range(len(c))])
    if not np.allclose(expected,c,atol=1e-10,rtol=0):
        raise ValueError("Centers were not Train-only class means")
    tau=model["objective"]["tau"]
    train=standard_metrics(by,c,ty,tau,split="train")
    val=standard_metrics(arrays["validation_by_hypothesis"],c,vy,tau,split="validation")
    return {"schema":REPORT_SCHEMA,"generated_utc":datetime.now(timezone.utc).isoformat(),
       "source":{"schema":model["schema"],"family":model["family"],
         "objective":model["objective"],"metric_tau":tau,"metric_tau_source":"snapshot",
         "data_seed":seed,"reference_seed":info["reference_seed"],
         "model_json_sha256":sha256((run_dir/"model.json").read_bytes()).hexdigest(),
         "distributions_sha256":model["distributions_sha256"],
         "data_origin":"immutable_phase2_train_validation_frozen_npz",
         "historical_test_accessed":False},
       "centers":c.tolist(),"train":train,"validation":val}
