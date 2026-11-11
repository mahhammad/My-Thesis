#!/usr/bin/env python3
"""n<=6 toy-only V2 numeric QA runner. No data generator, optimizer or grid.

Run from full checkout with OPENBLAS_NUM_THREADS=1:
  python code/experiments/bayes_center_audit_pack/bound_repair_candidate/numeric_v2_validation.py \
    --out code/experiments/bayes_center_audit_pack/bound_repair_candidate/numeric_v2_test_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
import unittest
from pathlib import Path

import numpy as np

from exact_origin_excluded_bound import exact_origin_excluded_center_bounds, shared_feature_kernel, _excluded_denominator
from test_exact_origin_excluded_bound import _TinyGraph, alpha_from_project, original_dp
from test_numerical_v2 import SEED, TOL, independent_enumeration


def run():
    # Set HARD limits on this toy runner. Fail if cannot apply them (Linux).
    cap = 900 * (1024 ** 2)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (min(cap, hard) if hard >= 0 else cap, hard))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    started = time.perf_counter()
    alpha = alpha_from_project(.4, .8, 2)
    logits = np.zeros((2, 1))
    cases = [
        ("original_return_kernel_counterexample", [[0.], [0.], [math.sqrt(math.log(4))]], [0, 0, 1], 0, 1),
        ("auditor_catastrophic_cancellation_counterexample", [[0.], [0.], [math.sqrt(math.log(1/(2e-12)))]], [0, 1, 0], 0, 0),
    ]
    reports = []
    for name, xx, yy, s, c in cases:
        x = np.asarray(xx, float)
        y = np.asarray(yy, int)
        gamma = np.array([1.])
        assert len(x) <= 6
        bounds = exact_origin_excluded_center_bounds(x, y, gamma, alpha, return_per_origin=True)
        independent = np.stack([independent_enumeration(x,y,gamma,alpha,origin,logits)
                                for origin in range(3)])
        ref_lo,ref_hi = original_dp()[0](_TinyGraph(x,y,gamma,alpha),y)
        true_center = np.stack([independent[y==cls].mean(axis=0)
                                for cls in range(int(y.max())+1)])
        low=np.asarray(bounds["lower_center_probabilities"])
        high=np.asarray(bounds["upper_center_probabilities"])
        per_low=np.asarray(bounds["lower_per_origin"])
        per_high=np.asarray(bounds["upper_per_origin"])
        discrepancies = {
            "max_abs_vs_independent_enumeration": float(max(np.max(np.abs(per_low-independent)),np.max(np.abs(per_high-independent)))),
            "max_abs_vs_reference_dense_DP": float(max(np.max(np.abs(per_low-ref_lo)),np.max(np.abs(per_high-ref_hi)))),
            "max_realized_lower_excess_center":float(np.max(np.maximum(low-true_center,0))),
            "max_realized_upper_shortfall_center":float(np.max(np.maximum(true_center-high,0))),
        }
        if any(v > TOL for v in discrepancies.values()):
            raise AssertionError(f"{name} failed independent gate: {discrepancies}")
        item={"name":name,"n":3,"d":1,"K":2,"H":2,"beta":0.4,"lambda":0.8,
              "alpha":alpha.tolist(),"seed":SEED,"origin":s,"component":c,
              "independent_origin_value":float(independent[s,c]),
              "new_origin_lower":float(per_low[s,c]),
              "new_origin_upper":float(per_high[s,c]),
              "independent_center_value":float(true_center[y[s],c]),
              "new_center_lower":float(low[y[s],c]),
              "new_center_upper":float(high[y[s],c]),
              "difference_metrics":discrepancies}
        if name.startswith("original"):
            item["expected_origin_exact"]=0.36
            item["invalid_historical_return_bound"]=71/375
        else:
            item["expected_center_approx"]=0.3833333333338
            item["auditor_v1_lower_failure"]=0.38333778525821427
            base=shared_feature_kernel(x,gamma)
            item["near_threshold_legal_denominator"]=float(_excluded_denominator(base,0)[1,0])
        reports.append(item)
    suite=unittest.TestLoader().discover(str(Path(__file__).resolve().parent),pattern="test_*.py")
    result=unittest.TestResult();suite.run(result)
    if not result.wasSuccessful():
        raise AssertionError("toy tests failed: "+repr(result.errors+result.failures))
    wall=time.perf_counter()-started
    maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data={"mission":"CH4-D10D11-NUMERICAL-REPAIR-V2-01","agent":"RESEARCHER-02",
          "status":"PASS_TOY_ONLY_PENDING_INDEPENDENT_QA",
          "kernel_version":"exact_origin_excluded_low_memory_candidate_v2_direct_numerator",
          "n_max_permitted":6,"tolerance":TOL,
          "numeric_wall_seconds":wall,"numeric_budget_seconds":120,
          "rlimit_AS_bytes":cap,"rlimit_CPU_soft_seconds":20,
          "peak_rss_kib_linux":maxrss,"peak_rss_gib":maxrss/1048576,
          "python_version":platform.python_version(),"numpy_version":np.__version__,
          "os":platform.platform(),"policy_seed":SEED,
          "reference_dp_blob":"79251967ed0d799185d502c8da44548eddd6e3ed",
          "original_v1_blob":"c0b26a8d42fe6985a1b7fcd1c4e3d036f072c68e",
          "unit_test_count":result.testsRun,"unit_failures":len(result.failures),
          "unit_errors":len(result.errors),"cases":reports,
          "full300_executed":False,"grid37_executed":False,"subset72_executed":False,
          "optimizer_or_training_executed":False,"validation_or_test_queries":False,
          "qualification":"Only tiny float64 checks; no rigorous universal roundoff envelope; D10/D11 still SCIENTIFIC_HOLD."}
    if wall>120 or maxrss>1048576:
        raise RuntimeError("toy numerical resource budget exceeded")
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    data=run()
    args.out.write_text(json.dumps(data,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({key:data[key] for key in ['status','unit_test_count','unit_failures','unit_errors',
          'numeric_wall_seconds','peak_rss_kib_linux','rlimit_AS_bytes','n_max_permitted']},indent=2))

if __name__=='__main__':main()
