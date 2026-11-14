"""Reproducible 3-seed Train/Validation fingerprint preflight. NEVER Test/train."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from generate import load_config
from dataset_adapter import dataset_train_val
from snapshot_pipeline import array_fingerprint,config_fingerprint
from phase2_protocol import SEEDS,N_TRAIN,N_VALIDATION,K,D,reference_ids,expected_data_gate
from phase2_frozen_data import load as load_frozen,INPUTS

def inspect():
    if os.environ.get("OPENBLAS_CORETYPE")!="generic" or os.environ.get("OPENBLAS_NUM_THREADS")!="1":
        raise RuntimeError("Exact reproducibility gate requires generic single-thread BLAS")
    cfg=ROOT/"code/data_generator/config.json"
    spec=load_config(cfg)
    if spec["sizes"]["train"]!=N_TRAIN or spec["sizes"]["validation"]!=N_VALIDATION:
        raise ValueError("Generator split sizes differ from phase2 protocol")
    data={}
    for seed in SEEDS:
        local=dict(spec);local["seed"]=seed
        if (INPUTS/"manifest.json").is_file():
            (x,y),(vx,vy)=load_frozen(seed)
            frozen=True
            # Regeneration comparability is diagnostic, not a substitute for
            # the immutable frozen NPZ on a different BLAS/CPU backend.
            try:
                regenerated=dataset_train_val(local)
                regen_matches=(array_fingerprint(regenerated["train"][0])==array_fingerprint(x)
                               and array_fingerprint(regenerated["validation"][0])==array_fingerprint(vx))
            except (ValueError,RuntimeError):
                regen_matches=False
        else:
            split=dataset_train_val(local)
            x,y=split["train"];vx,vy=split["validation"]
            frozen=False
            regen_matches=None
        if x.shape!=(N_TRAIN,D) or vx.shape!=(N_VALIDATION,D):
            raise ValueError("feature dimension mismatch")
        ids=reference_ids(y)
        if set(np.unique(vy).tolist())!=set(range(K)):
            raise ValueError("Validation lost class")
        data[str(seed)]={
            "n_train":len(x),"n_validation":len(vx),
            "n_classes":K,"n_features":D,
            "train_class_counts":np.bincount(y,minlength=K).tolist(),
            "validation_class_counts":np.bincount(vy,minlength=K).tolist(),
            "reference_ids":ids.tolist(),
            "config_sha256":config_fingerprint(cfg),
            "train_x_sha256":array_fingerprint(x[ids]),
            "train_y_sha256":array_fingerprint(y[ids].astype(np.int64)),
            "validation_x_sha256":array_fingerprint(vx),
            "mask_train_sha256":array_fingerprint(np.isfinite(x[ids])),
            "mask_validation_sha256":array_fingerprint(np.isfinite(vx)),
            "test_accessed":False,
            "frozen_immutable_source":frozen,
            "local_regeneration_matches_frozen":regen_matches}
    expected_data_gate(data)
    return {"schema":"e-ch4-13-phase2-data-preflight-v2","numpy":np.__version__,
            "python":platform.python_version(),"blas_coretype":"generic",
            "blas_threads":1,"test_accessed":False,"datasets":data,
            "frozen_inputs_exist":(INPUTS/"manifest.json").is_file()}

if __name__=="__main__":
    print(json.dumps(inspect(),sort_keys=True,indent=2))
