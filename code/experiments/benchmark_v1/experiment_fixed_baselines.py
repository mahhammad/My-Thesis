"""Controlled no-training P1/P2 baselines: uniform walk sensitivity and fixed guided policies.

Default: dry run; --execute writes imc-snapshot-v2 model/distributions and
independent reports for Train and Validation ONLY. Historical Test is not
materialized, and historical runner_extended.py is never imported.

From repository root:
 python code/experiments/benchmark_v1/experiment_fixed_baselines.py
 python code/experiments/benchmark_v1/experiment_fixed_baselines.py --execute \
   --out-dir code/experiments/benchmark_v1/results/pipeline_v1/fixed_baselines
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "data_generator"))

from generate import load_config
from dataset_adapter import dataset_train_val
from engine import DT, ExactMaskedWalk
from snapshot_pipeline import save_snapshot, config_fingerprint
from report_snapshot import report_snapshot, save_report


BASE_BETA = .4
BASE_LAMBDA = .8
BASE_H = 12
SCALE = (0.0, .25, .5, .75, 1.0)


def tag(value):
    return f"{value:.3f}".replace(".", "p")


def experiment_plan(mode="both"):
    """9 unique uniform settings plus guided alpha>0, reusing uniform at alpha=0."""
    jobs = {}
    for beta in (.2, .4, .7):
        jobs[(beta, BASE_LAMBDA, BASE_H)] = {
            "kind": "uniform", "beta": beta, "lam": BASE_LAMBDA, "H": BASE_H,
            "alpha": 0.0}
    for lam in (.4, .8, 1.):
        jobs[(BASE_BETA, lam, BASE_H)] = {
            "kind": "uniform", "beta": BASE_BETA, "lam": lam, "H": BASE_H,
            "alpha": 0.0}
    for horizon in (1, 2, 3, 4, 12):
        jobs[(BASE_BETA, BASE_LAMBDA, horizon)] = {
            "kind": "uniform", "beta": BASE_BETA, "lam": BASE_LAMBDA, "H": horizon,
            "alpha": 0.0}
    base = (BASE_BETA, BASE_LAMBDA, BASE_H)
    baseline = jobs[base]
    uniform = [baseline] + [job for key, job in jobs.items() if key != base]
    guided = [
        {"kind": "guided", "alpha": alpha, "beta": BASE_BETA,
         "lam": BASE_LAMBDA, "H": BASE_H}
        for alpha in SCALE[1:]
    ]
    plan = (uniform if mode == "uniform" else
            [baseline] + guided if mode == "guided" else
            uniform + guided)
    for item in plan:
        item["id"] = (
            f"{item['kind']}_a{tag(item['alpha'])}_b{tag(item['beta'])}"
            f"_l{tag(item['lam'])}_h{item['H']}")
    return plan


def logits_for(job, d):
    if d != 5:
        raise ValueError("Guided baseline is defined on the five-feature base scenario")
    base = torch.tensor([1.4, .2, -1.2, .9, -1.2], dtype=DT)
    guided = base[None, :].repeat(d + 1, 1)
    guided[:d, :] += .25 * torch.eye(d, dtype=DT)
    return job["alpha"] * guided


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="Opt in: execute exact walks, no fitting")
    p.add_argument("--mode", choices=("uniform", "guided", "both"), default="both")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "code/experiments/benchmark_v1/results/pipeline_v1/fixed_baselines")
    p.add_argument("--config", type=Path, default=ROOT / "code/data_generator/config.json")
    p.add_argument("--code-commit", default=os.environ.get("GITHUB_SHA"))
    a = p.parse_args()
    plan = experiment_plan(a.mode)
    if not a.execute:
        print(json.dumps({"status": "dry-run", "training": False,
                          "historical_test_accessed": False,
                          "unique_jobs": len(plan), "jobs": plan}, indent=2))
        return

    config_path = a.config.resolve()
    if config_path != (ROOT / "code/data_generator/config.json").resolve():
        raise ValueError("This registered baseline uses the fixed base generator config")
    spec = load_config(config_path)
    if spec["seed"] != 20261102 or len(spec["pi"]) != 3 or len(spec["gamma"]) != 5:
        raise ValueError("Unexpected base config: update protocol before changing source")
    data = dataset_train_val(spec)
    x, y = data["train"]
    vx, _ = data["validation"]
    ids = np.arange(len(y), dtype=np.int64)
    torch.set_num_threads(1)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    start_all = time.perf_counter()
    for j, job in enumerate(plan, 1):
        path = a.out_dir / job["id"]
        graph = ExactMaskedWalk(x, y, spec["gamma"],
                                beta=job["beta"], lam=job["lam"],
                                horizon=job["H"], query_x=vx)
        z = logits_for(job, graph.d)
        start = time.perf_counter()
        save_snapshot(
            path, graph, z, "shared", reference_ids=ids, config_path=config_path,
            config_reference="code/data_generator/config.json",
            data_seed=spec["seed"], reference_seed=20261107,
            objective="fixed_guided_no_training" if job["kind"] == "guided" else
                      "uniform_no_training",
            tau=.005, beta=job["beta"], lam=job["lam"], horizon=job["H"],
            validation_x=vx, code_commit=a.code_commit,
            policy_variant="fixed_guided" if job["kind"] == "guided" else "uniform",
            experiment_metadata={"research_phase": "P2" if job["kind"] == "guided" else "P1",
                                 "alpha_guided": job["alpha"], "training_steps": 0,
                                 "reference_selection": "entire_generator_train_order",
                                 "historical_test_accessed": False,
                                 "generator_config_sha256": config_fingerprint(config_path)})
        report = report_snapshot(path, per_sample=True)
        save_report(path, report)
        alpha = graph.alpha.detach().cpu().numpy()
        expected_L = (1.0 - (1.0 - job["beta"]) ** job["H"]) / job["beta"]
        depth_eff = float(np.arange(1, job["H"] + 1) @ alpha)
        tr, va = report["train"], report["validation"]
        row = {
            **job, "alpha_mass": float(alpha.sum()), "expected_length": expected_L,
            "effective_depth": depth_eff,
            "train_BA": tr["balanced_accuracy"], "validation_BA": va["balanced_accuracy"],
            "train_accuracy": tr["ordinary_accuracy"],
            "validation_accuracy": va["ordinary_accuracy"],
            "I": tr["I"], "R_JS": tr["R_JS"],
            "R_sqrt_JS": tr["R_sqrt_JS"], "R_KL": tr["R_KL"],
            "train_ties": tr["tie_count"], "validation_ties": va["tie_count"],
            "validation_median_margin": va["margin_quantiles"]["0.5"],
            "min_pairwise_sqrt_JS": tr["min_pairwise_sqrt_JS"],
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "run_dir": job["id"]
        }
        summary.append(row)
        print(json.dumps({"run": j, "total": len(plan), **row},
                         ensure_ascii=False, allow_nan=False), flush=True)

    manifest = {
        "schema": "ch4-fixed-baselines-p1p2-v1",
        "status": "completed_no_training_no_Test",
        "engine": "ExactMaskedWalk/float64",
        "data_seed": spec["seed"], "reference_n": len(y),
        "validation_n": len(vx), "config_sha256": config_fingerprint(config_path),
        "code_commit": a.code_commit, "baseline_alpha0_reused": True,
        "run_count": len(summary), "run_ids": [r["id"] for r in summary],
        "walltime_seconds": round(time.perf_counter() - start_all, 4),
        "runs": summary
    }
    (a.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    with (a.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("FINISHED", json.dumps({"count": len(summary),
                                 "out": str(a.out_dir),
                                 "training": False, "test_accessed": False},
                                ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
