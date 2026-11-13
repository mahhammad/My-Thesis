#!/usr/bin/env python3
"""R02 §4.11 Gate-A tiny-frozen-source recheck ONLY (n<=6).

No data generator, no full300, no optimizer, no parameter sweep; only optional new audit output.
Usage: python sec411_gate_a_smoke.py --out /tmp/sec411_gate_a_smoke.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent / "bound_repair_candidate"
sys.path.insert(0, str(CANDIDATE))
from exact_origin_excluded_bound import exact_origin_excluded_center_bounds
from test_numerical_v2 import independent_enumeration
from test_exact_origin_excluded_bound import alpha_from_project

PIN = {
    "exact_origin_excluded_bound.py": "7fe3ae82f10b3120054854f82ee3916c55605f30",
    "test_numerical_v2.py": "af68374c8876a434950a674c4435c426bdb0f509",
    "test_exact_origin_excluded_bound.py": "abd4668bd1e5986473b1f5c579f3f84bf08afe36",
}

def verify_hashes():
    out = {}
    for file, expected in PIN.items():
        path = CANDIDATE / file
        data = path.read_bytes()
        calculated = subprocess.check_output(["git", "hash-object", "--", str(path)], text=True).strip()
        if calculated != expected:
            raise AssertionError(f"frozen git blob changed: {file}: {calculated} != {expected}")
        out[file] = {"git_blob_sha": calculated, "sha256": hashlib.sha256(data).hexdigest()}
    return out

def check_case(name, x, y, gamma, H, n_logits, seed):
    if len(y)>6: raise RuntimeError("Safety gate: maximum n=6")
    alpha=alpha_from_project(.4,.8,H)
    b=exact_origin_excluded_center_bounds(x,y,gamma,alpha,return_per_origin=True)
    low=np.asarray(b["lower_per_origin"])
    high=np.asarray(b["upper_per_origin"])
    lowc=np.asarray(b["lower_center_probabilities"])
    highc=np.asarray(b["upper_center_probabilities"])
    rng=np.random.default_rng(seed)
    max_violation=0.0
    originals=[]
    for _ in range(n_logits):
        z=rng.normal(size=(x.shape[1]+1,x.shape[1])) if x.shape[1]>1 else np.zeros((2,1))
        truth=np.vstack([independent_enumeration(x,y,gamma,alpha,s,z) for s in range(len(y))])
        originals.append(truth)
        max_violation=max(max_violation,float(np.maximum(low-truth,0).max()),float(np.maximum(truth-high,0).max()))
        for c in range(lowc.shape[0]):
            m=truth[y==c].mean(axis=0)
            max_violation=max(max_violation,float(np.maximum(lowc[c]-m,0).max()),float(np.maximum(m-highc[c],0).max()))
    if max_violation>1e-10: raise AssertionError(f"{name}: policy outside V2 interval by {max_violation}")
    return {"case":name,"n":int(len(y)),"d":int(x.shape[1]),"H":H,"independent_fixed_Z_checks":len(y)*n_logits,
            "max_interval_violation":max_violation,"bound_kernel_version":b["kernel_version"],
            "true_reference_origin0":originals[0][0].tolist(),
            "true_class0_center":originals[0][y==0].mean(axis=0).tolist()}

def main():
    t0=time.perf_counter()
    hashes=verify_hashes()
    gamma1=np.array([1.])
    original_x=np.array([[0.],[0.],[math.sqrt(math.log(4))]])
    cancel_x=np.array([[0.],[0.],[math.sqrt(math.log(1/(2e-12)))]])
    results=[]
    results.append(check_case("auditor_origin_return_counterexample",original_x,np.array([0,0,1]),gamma1,2,1,1))
    assert abs(results[-1]["true_reference_origin0"][1]-0.36)<1e-12
    results.append(check_case("auditor_numerator_cancellation_counterexample",cancel_x,np.array([0,1,0]),gamma1,2,1,1))
    assert abs(results[-1]["true_class0_center"][0]-0.3833333333338)<1e-10
    x6=np.array([[0.,1.,np.nan],[.1,np.nan,.3],[.3,.5,.2],[.4,.2,np.nan],[np.nan,.8,.5],[.8,.7,.2]])
    y6=np.array([0,1,0,1,0,1]);gamma6=np.array([1.,.5,2.])
    for H in (1,3):
        results.append(check_case(f"masked_n6_H{H}",x6,y6,gamma6,H,3,20261111))
    out={"mission":"R02-SEC411-SELF-MANAGED-GATE-A-01","scope":"toy_snapshot_recheck_only",
         "scientific_status":"NO_NEW_FULL_DATA_CERTIFICATE", "status":"PASS_TOY_SMOKE_ONLY",
         "frozen_sha":hashes,"source_snapshot":"numeric_v2_candidate_zip_and_Git_blob_SHA",
         "python":platform.python_version(),"numpy":np.__version__,"case_results":results,
         "total_independent_policy_oracle_calls":sum(c["independent_fixed_Z_checks"] for c in results),
         "wall_seconds":time.perf_counter()-t0,"peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "notes":"Reuses the frozen test_numerical_v2 independent path oracle; not a second independently developed numerical implementation or an interval certificate."}
    print(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True))
    return out

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("--out",default=None);args=cli.parse_args()
    res=main()
    if args.out:Path(args.out).write_text(json.dumps(res,indent=2,sort_keys=True)+"\n",encoding="utf-8")
