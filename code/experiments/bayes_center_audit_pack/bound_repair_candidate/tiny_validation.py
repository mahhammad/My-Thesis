#!/usr/bin/env python3
"""Audit report of original dense-vs-new low-memory DP on ONLY n<=6 fixtures.

No CLI pathway for loading thesis datasets, full300 or a grid exists here.
Run inside a full repo checkout:
  python code/experiments/bayes_center_audit_pack/bound_repair_candidate/tiny_validation.py \
      --out code/experiments/bayes_center_audit_pack/bound_repair_candidate/tiny_test_results.json
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

from exact_origin_excluded_bound import exact_origin_excluded_center_bounds
from test_exact_origin_excluded_bound import (
    _TinyGraph, alpha_from_project, exact_policy_expectation, original_dp,
)

SEED = 20261111


def cases():
    yield ("auditor_counterexample", np.array([[0.], [0.], [math.sqrt(math.log(4))]]),
           np.array([0, 0, 1]), np.array([1.0]), alpha_from_project(.4, .8, 2))
    yield ("H1", np.array([[0.], [.2], [1.], [1.2]]),
           np.array([0, 1, 0, 1]), np.array([1.2]), alpha_from_project(.4, .8, 1))
    yield ("masked_multifeature", np.array([
        [0., 1., np.nan], [.1, np.nan, .3], [.3, .5, .2],
        [.4, .2, np.nan], [np.nan, .8, .5], [.8, .7, .2]]),
        np.array([0, 1, 0, 1, 0, 1]), np.array([1., .5, 2.]),
        alpha_from_project(.4, .8, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    original_per_origin, original_center = original_dp()
    rng = np.random.default_rng(SEED)
    case_reports = []
    for name, x, y, gamma, alpha in cases():
        assert len(x) <= 6
        g = _TinyGraph(x, y, gamma, alpha)
        lower, upper = original_per_origin(g, y)
        orig_cent = original_center(g, y)
        candidate = exact_origin_excluded_center_bounds(x, y, gamma, alpha, return_per_origin=True)
        matrices = {
            "origin_lower": (candidate["lower_per_origin"], lower),
            "origin_upper": (candidate["upper_per_origin"], upper),
            "center_lower": (candidate["lower_center_probabilities"], orig_cent["lower_center_probabilities"]),
            "center_upper": (candidate["upper_center_probabilities"], orig_cent["upper_center_probabilities"]),
        }
        maxdiff = {key: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                   for key, (a, b) in matrices.items()}
        violation = 0.0
        for cls in range(g.k):
            logits = rng.normal(size=(g.d + 1, g.d))
            for s in range(g.n):
                actual = exact_policy_expectation(g, y, s, logits)
                violation = max(violation,
                                float(np.max(np.maximum(np.asarray(candidate["lower_per_origin"][s]) - actual, 0))),
                                float(np.max(np.maximum(actual - np.asarray(candidate["upper_per_origin"][s]), 0))))
        assert max(maxdiff.values()) <= 5e-13, (name, maxdiff)
        assert violation <= 5e-13, (name, violation)
        item = {"name": name, "n": g.n, "d": g.d, "K": g.k, "H": len(alpha),
                "alpha": alpha.tolist(), "max_abs_error_vs_original_DP": maxdiff,
                "max_realized_outer_bound_violation": violation,
                "fixed_policy_logits_seed": SEED, "fixed_class_policies_tested": g.k}
        if name == "auditor_counterexample":
            item["counterexample"] = {
                "origin": 0, "label_component": 1,
                "original_walk_value": float(candidate["upper_per_origin"][0][1]),
                "original_walk_expected_exact": 0.36,
                "original_error_abs": abs(candidate["upper_per_origin"][0][1] - .36),
                "invalid_old_return_upper": 71 / 375,
                "invalid_upper_excess": float(candidate["upper_per_origin"][0][1] - 71 / 375),
            }
        case_reports.append(item)

    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TestResult()
    suite.run(result)
    if not result.wasSuccessful():
        raise AssertionError("unit tests failed " + repr(result.errors + result.failures))
    elapsed = time.perf_counter() - started
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data = {
        "mission": "CH4-D10D11-BOUND-REPAIR-01", "agent": "RESEARCHER-02",
        "status": "PASS_TINY_FIXTURES_ONLY", "numeric_budget_seconds": 120,
        "numeric_elapsed_wall_seconds": elapsed,
        "rss_peak_kib_linux": rss_kb,
        "peak_rss_gib": rss_kb / (1024 * 1024),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "policy_seed": SEED,
        "source_reference": "code/experiments/archive/bayes_center_inverse/feasibility_bound.py::bounds_for_all_origins",
        "reference_code_blob_sha": "79251967ed0d799185d502c8da44548eddd6e3ed",
        "unit_test_count": result.testsRun,
        "unit_test_failures": len(result.failures),
        "unit_test_errors": len(result.errors),
        "fixture_reports": case_reports,
        "forbidden_executions": {"full300": False, "grid_37": False, "optimizer": False,
                                 "train": False, "MC": False, "validation_or_test_queries": False},
        "limitation": "Only checks n<=6 and tiny fixed policies; no proof of general numerical implementation or full300 results.",
    }
    if elapsed > 120 or data["peak_rss_gib"] > 1:
        raise RuntimeError("numeric resource budget exceeded")
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in data.items() if k != "fixture_reports"}, indent=2))


if __name__ == "__main__":
    main()
