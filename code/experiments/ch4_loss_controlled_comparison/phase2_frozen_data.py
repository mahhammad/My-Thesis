"""Freeze/reuse exact Train and Validation arrays for all three phase2 seeds.
No Test generation/read; no fit. Frozen input SHA is required on every run.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import io
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
# Byte-identical implementation of benchmark_v1.snapshot_pipeline fingerprints,
# kept NumPy-only so the data freeze does not require an ML runtime.
def array_fingerprint(array):
    a=np.ascontiguousarray(np.asarray(array))
    header=json.dumps({"dtype":a.dtype.str,"shape":list(a.shape)},sort_keys=True).encode("utf8")
    return sha256(header+bytes((0,))+a.tobytes()).hexdigest()

def config_fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()
from phase2_protocol import SEEDS,N_TRAIN,N_VALIDATION,K,D,reference_ids,atomic_new_json

INPUTS=HERE/"inputs/phase2_train_validation"
CFG=ROOT/"code/data_generator/config.json"
SCHEMA="e-ch4-13-phase2-frozen-train-validation-v1"

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def validate_array_pair(train_x,train_y,val_x,val_y):
    if train_x.shape!=(N_TRAIN,D) or val_x.shape!=(N_VALIDATION,D):
        raise ValueError("Freeze input shape not full 300/90 x 5")
    if train_y.shape!=(N_TRAIN,) or val_y.shape!=(N_VALIDATION,):
        raise ValueError("Freeze labels shape mismatch")
    if not (train_x.dtype==np.float64 and val_x.dtype==np.float64
            and train_y.dtype==np.int64 and val_y.dtype==np.int64):
        raise ValueError("Frozen arrays require float64 and int64")
    reference_ids(train_y)
    if set(val_y.tolist())!=set(range(K)):raise ValueError("Validation missing class")
    if np.any(np.isinf(train_x)) or np.any(np.isinf(val_x)):
        raise ValueError("Only NaN allowed for feature masking")
    return True

def freeze(out=INPUTS,*,source_sha):
    if os.environ.get("E_CH4_13_FREEZE_INPUTS_ONCE")!="true":
        raise PermissionError("No explicit data-only freeze opt-in")
    if os.environ.get("OPENBLAS_CORETYPE")!="generic" or os.environ.get("OPENBLAS_NUM_THREADS")!="1":
        raise RuntimeError("Use documented single-thread generic BLAS")
    if len(source_sha)!=40:raise ValueError("Exact source commit SHA required")
    out=Path(out)
    if out.exists():raise FileExistsError("Frozen input folder already exists")
    config=load_config(CFG)
    config_sha=config_fingerprint(CFG)
    records={}
    for seed in SEEDS:
        sp=dict(config);sp["seed"]=seed
        split=dataset_train_val(sp)  # never Test
        tx,ty=split["train"];vx,vy=split["validation"]
        tx=np.ascontiguousarray(tx,dtype=np.float64)
        vx=np.ascontiguousarray(vx,dtype=np.float64)
        ty=np.ascontiguousarray(ty,dtype=np.int64)
        vy=np.ascontiguousarray(vy,dtype=np.int64)
        validate_array_pair(tx,ty,vx,vy)
        buffer=io.BytesIO()
        np.savez_compressed(buffer,train_x=tx,train_y=ty,
                            validation_x=vx,validation_y=vy)
        blob=buffer.getvalue()
        records[str(seed)]={"file":f"ds{seed}_train_val.npz",
          "file_sha256":sha256(blob).hexdigest(),
          "train_x_sha256":array_fingerprint(tx),
          "train_y_sha256":array_fingerprint(ty),
          "validation_x_sha256":array_fingerprint(vx),
          "validation_y_sha256":array_fingerprint(vy),
          "mask_train_sha256":array_fingerprint(np.isfinite(tx)),
          "mask_validation_sha256":array_fingerprint(np.isfinite(vx)),
          "train_counts":np.bincount(ty,minlength=K).tolist(),
          "validation_counts":np.bincount(vy,minlength=K).tolist(),
          "reference_ids":"0..299","n_train":300,"n_validation":90,
          "test_accessed":False}
    out.mkdir(parents=True,exist_ok=False)
    for seed in SEEDS:
        sp=dict(config);sp["seed"]=seed
        split=dataset_train_val(sp)
        tx,ty=split["train"];vx,vy=split["validation"]
        # Verify new regeneration on the *same* runner and backend before freeze.
        rec=records[str(seed)]
        if (array_fingerprint(tx)!=rec["train_x_sha256"] or
            array_fingerprint(vx)!=rec["validation_x_sha256"]):
            raise ValueError("Generator nondeterministic on same machine; refuse freeze")
        with (out/rec["file"]).open("xb") as f:
            np.savez_compressed(f,train_x=np.asarray(tx,dtype=np.float64),
                train_y=np.asarray(ty,dtype=np.int64),validation_x=np.asarray(vx,dtype=np.float64),
                validation_y=np.asarray(vy,dtype=np.int64))
        if sha(out/rec["file"])!=rec["file_sha256"]:
            raise ValueError("NPZ archive bytes not reproducible; reject freeze")
    manifest={"schema":SCHEMA,"status":"FROZEN_INPUTS_UNAUDITED","source_commit":source_sha,
        "config_sha256":config_sha,"seed_set":list(SEEDS),
        "numpy":np.__version__,"python":platform.python_version(),
        "openblas_coretype":"generic","openblas_threads":1,
        "train_validation_only":True,"test_accessed":False,"records":records}
    atomic_new_json(out/"manifest.json",manifest)
    for seed in SEEDS:load(seed,out=out)
    return manifest

def load(seed,*,out=INPUTS):
    if seed not in SEEDS:raise ValueError("Not preregistered generator seed")
    out=Path(out)
    man=json.loads((out/"manifest.json").read_text("utf8"))
    if man["schema"]!=SCHEMA or man["test_accessed"] is not False:
        raise ValueError("Invalid frozen data provenance")
    if man["config_sha256"]!=config_fingerprint(CFG):
        raise ValueError("Generator config changed since freeze")
    rec=man["records"][str(seed)]
    file=out/rec["file"]
    if sha(file)!=rec["file_sha256"]:raise ValueError("Frozen NPZ byte corruption")
    with np.load(file,allow_pickle=False) as payload:
        if set(payload.files)!={"train_x","train_y","validation_x","validation_y"}:
            raise ValueError("Frozen data keys invalid")
        tx,ty,vx,vy=(payload[key].copy() for key in
                      ("train_x","train_y","validation_x","validation_y"))
    validate_array_pair(tx,ty,vx,vy)
    for key,value in (("train_x_sha256",array_fingerprint(tx)),
                      ("train_y_sha256",array_fingerprint(ty)),
                      ("validation_x_sha256",array_fingerprint(vx)),
                      ("validation_y_sha256",array_fingerprint(vy)),
                      ("mask_train_sha256",array_fingerprint(np.isfinite(tx))),
                      ("mask_validation_sha256",array_fingerprint(np.isfinite(vx)))):
        if rec[key]!=value:raise ValueError("Frozen raw array mismatch: "+key)
    if rec["test_accessed"] is not False:raise ValueError("Frozen data Test violation")
    return (tx,ty),(vx,vy)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--source-sha",default="")
    p.add_argument("--out",type=Path,default=INPUTS)
    args=p.parse_args()
    if args.freeze:
        print(json.dumps({"status":"FROZEN_DATA_UNAUDITED",
                          "records":freeze(args.out,source_sha=args.source_sha)["records"]}))
    else:
        print(json.dumps({"status":"NO_TRAINING_DATA_CONTRACT_ONLY",
            "seeds":list(SEEDS),"expected":str(args.out),
            "existing":args.out.is_dir()}))
if __name__=="__main__":main()
