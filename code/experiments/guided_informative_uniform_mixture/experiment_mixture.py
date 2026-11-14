#!/usr/bin/env python3
"""E-CH4-05: five *probability-space* mixtures of Uniform and noise-free Guided.

Default is dry-run. --execute evaluates only the registered 20261102 Train
and Validation sets, no fitting, no Monte Carlo sampling, NO historical Test.
The original logits/Softmax experiment and its saved results are not changed.

From repository root:
  python code/experiments/guided_informative_uniform_mixture/experiment_mixture.py
  python code/experiments/guided_informative_uniform_mixture/experiment_mixture.py --execute
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = HERE.parent / "benchmark_v1"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "code/data_generator"))
from generate import load_config
from dataset_adapter import dataset_train_val
from report_snapshot import standard_metrics
from probability_walk import ProbabilityMixtureWalk
from policy import load_policy

ALPHAS = (0.0, .25, .5, .75, 1.0)
BETA, LAMBDA, H = .4, .8, 12
CONFIG = ROOT / "code/data_generator/config.json"
PRIOR_CSV = BENCH / "results/pipeline_v1/fixed_baselines_summary_20261109.csv"


def _base_check(row):
    """The alpha=0 result must reproduce the previously VERIFIED Uniform."""
    with PRIOR_CSV.open(encoding="utf8", newline="") as f:
        old = next(r for r in csv.DictReader(f)
                   if r["kind"] == "uniform" and float(r["beta"]) == BETA
                   and float(r["lam"]) == LAMBDA and int(r["H"]) == H)
    correspondences = {
        "train_BA": "train_BA",
        "validation_BA": "validation_BA",
        "train_accuracy": "train_accuracy",
        "validation_accuracy": "validation_accuracy",
        "I": "I",
        "R_JS": "R_JS",
        "R_KL": "R_KL",
        "R_sqrt_JS": "R_sqrt_JS",
        "validation_median_margin": "validation_median_margin",
    }
    for k, old_key in correspondences.items():
        if not np.isclose(row[k], float(old[old_key]), atol=2e-10, rtol=0):
            raise AssertionError(
                f"Alpha=0 must recover exact historic Uniform {k}: "
                f"{row[k]} != {old[old_key]}"
            )
    return True


def compact_metrics(alpha, tr, va, elapsed):
    return {
        "alpha": float(alpha),
        "beta": BETA, "lambda": LAMBDA, "H": H,
        "train_BA": tr["balanced_accuracy"],
        "validation_BA": va["balanced_accuracy"],
        "train_accuracy": tr["ordinary_accuracy"],
        "validation_accuracy": va["ordinary_accuracy"],
        "I_train": tr["I"],
        "R_JS_train": tr["R_JS"],
        "R_sqrt_JS_train": tr["R_sqrt_JS"],
        "R_KL_train": tr["R_KL"],
        "validation_median_margin": va["margin_quantiles"]["0.5"],
        "validation_sigmoid_margin_loss": va["balanced_sigmoid_margin_loss"],
        "validation_balanced_JS_NLL": va["balanced_JS_similarity_NLL"],
        "train_ties": tr["tie_count"],
        "validation_ties": va["tie_count"],
        "validation_predictions": va["per_sample"]["predicted"],
        "validation_confusion": va["confusion_true_rows_pred_cols"],
        "elapsed_seconds": round(elapsed, 4),
    }


def evaluate(out_dir: Path, *, store_distributions=True):
    config = load_config(CONFIG)
    policy = load_policy()
    assert config["seed"] == 20261102
    assert policy["alpha_grid"] == list(ALPHAS)
    assert policy["informative_features"] == [0, 1, 3]
    assert policy["noise_features"] == [2, 4]
    data = dataset_train_val(config)  # Explicitly excludes Test.
    x, y = data["train"]
    q, qy = data["validation"]
    assert x.shape == (300, 5) and q.shape == (90, 5)
    assert np.array_equal(np.bincount(y, minlength=3), [72, 147, 81])
    assert np.array_equal(np.bincount(qy, minlength=3), [13, 49, 28])
    torch.set_num_threads(1)
    graph = ProbabilityMixtureWalk(
        x, y, config["gamma"], beta=BETA, lam=LAMBDA,
        horizon=H, query_x=q, informative=tuple(policy["informative_features"])
    )
    metrics = []
    all_reports = {}
    distributions = {}
    for alpha in ALPHAS:
        started = time.perf_counter()
        with torch.no_grad():
            tr = graph.represent_train_mixture(alpha).numpy()
            va = graph.represent_validation_mixture(alpha).numpy()
        centers = np.stack([tr[y == c].mean(axis=0) for c in range(3)])
        # The policy is shared, not class-hypothesis-specific: repeat the
        # *same* representation for each candidate center, as in exact engine.
        tr_by = np.broadcast_to(tr[:, None, :], (300, 3, 3))
        va_by = np.broadcast_to(va[:, None, :], (90, 3, 3))
        train_stats = standard_metrics(
            tr_by, centers, y, .005, split="train", per_sample=True
        )
        valid_stats = standard_metrics(
            va_by, centers, qy, .005, split="validation", per_sample=True
        )
        row = compact_metrics(alpha, train_stats, valid_stats,
                              time.perf_counter()-started)
        if alpha == 0:
            _base_check({
                "train_BA": row["train_BA"],
                "validation_BA": row["validation_BA"],
                "train_accuracy": row["train_accuracy"],
                "validation_accuracy": row["validation_accuracy"],
                "I": row["I_train"],
                "R_JS": row["R_JS_train"],
                "R_KL": row["R_KL_train"],
                "R_sqrt_JS": row["R_sqrt_JS_train"],
                "validation_median_margin": row["validation_median_margin"],
            })
        # Keep per-sample predictions inside per-run report; summary is compact.
        metrics.append({k: v for k, v in row.items()
                        if k not in ("validation_predictions", "validation_confusion")})
        all_reports[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha, "train": train_stats, "validation": valid_stats,
            "centers": centers.tolist(),
        }
        distributions[f"train_a{alpha:.2f}"] = tr
        distributions[f"validation_a{alpha:.2f}"] = va
        distributions[f"centers_a{alpha:.2f}"] = centers
        print("RESULT_ROW_JSON "+json.dumps(row, separators=(",", ":")),
              flush=True)

    # End-point sanity: noise features have exactly zero direct selection
    # probability when alpha=1; intermediate alphas preserve the Uniform share.
    start = graph._mixture(graph.qavailable, 1.0)
    assert torch.all(start[:, [2, 4]] == 0)
    assert torch.allclose(start.sum(1), torch.ones(len(q), dtype=torch.float64))
    result = {
        "schema": "E-CH4-05-mixture-exact-v1",
        "policy": policy["policy_id"],
        "probability_space_mixture": True,
        "training_steps": 0, "historical_test_accessed": False,
        "n_train": len(y), "n_validation": len(q),
        "data_seed": config["seed"],
        "beta": BETA, "lambda": LAMBDA, "H": H,
        "all_alpha0_metrics_recovered_prior_uniform": True,
        "results": metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(metrics[0])
    with (out_dir/"summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(metrics)
    (out_dir/"summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
        encoding="utf8")
    (out_dir/"reports.json").write_text(
        json.dumps(all_reports, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
        encoding="utf8")
    if store_distributions:
        np.savez_compressed(out_dir/"distributions.npz", **distributions)
    print("FINAL_SUMMARY_JSON "+json.dumps(result, separators=(",", ":")), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE/"results")
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "dry_run",
            "policy": "probability-space mixture, not alpha-scaled logits",
            "alpha_grid": ALPHAS,
            "train_n": 300, "validation_n": 90, "historical_test_accessed": False
        }, indent=2))
    else:
        evaluate(args.out_dir)
