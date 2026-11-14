"""E-CH4-07: recover full common metrics from three SAVED 300-row policies.

No training, no Monte Carlo, and no Test generation. Exactly reproduces the
archived shared-Z expectation engine, then uses the Chapter-4 common reporter.
Refuse to publish results if the historical BA/I/R_JS checks disagree.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARCH = ROOT / "code/experiments/archive"
for source in (
    ROOT / "code/data_generator",
    ROOT / "code/experiments/benchmark_v1",
    ARCH / "multistep_policy_target_match",
    ARCH / "diag_max_full300",
):
    sys.path.insert(0, str(source))
from generate import load_config
from dataset_adapter import dataset_train_val
from stream_graph import StreamGraph
from report_snapshot import standard_metrics


def evaluate(g, z, y, vy, tau):
    with torch.no_grad():
        p = g.center_train(z, np.arange(len(y)), return_all=True).numpy()
        centers = np.stack([p[y == c].mean(0) for c in range(g.k)])
        vp = g.evaluate_val(z).numpy()
    tr = standard_metrics(np.broadcast_to(p[:, None, :], (len(y), g.k, g.k)),
                          centers, y, tau, split="train", per_sample=True)
    va = standard_metrics(np.broadcast_to(vp[:, None, :], (len(vy), g.k, g.k)),
                          centers, vy, tau, split="validation", per_sample=True)
    return {"centers": centers.tolist(), "train": tr, "validation": va}


def main():
    torch.set_num_threads(1)
    spec = load_config(ROOT / "code/data_generator/config.json")
    splits = dataset_train_val(spec)
    x, y = splits["train"]
    val, vy = splits["validation"]
    if len(y) != 300 or len(vy) != 90:
        raise AssertionError("Unexpected archived experiment data shape")
    saved = json.loads((ARCH / "margin_training/results/full300_policy_snapshots.json")
                       .read_text(encoding="utf-8"))
    if saved["meta"]["data_seed"] != spec["seed"]:
        raise AssertionError("Generator seed mismatch")
    with (HERE / "results/margin_training_summary.csv").open(newline="", encoding="utf-8") as f:
        historical = list(csv.DictReader(f))
    g = StreamGraph(x, y, spec["gamma"], val, beta=.4, lam=.8, horizon=12)
    baseline = evaluate(g, torch.zeros((g.d + 1, g.d), dtype=torch.float64), y, vy, .005)
    results = []
    for tau, key in ((.002, "tau0.002"), (.005, "tau0.005"), (.02, "tau0.02")):
        row = next(r for r in historical if int(r["nref"]) == 300
                   and abs(float(r["tau"]) - tau) < 1e-12)
        z = torch.tensor(saved["trained_policies"][key], dtype=torch.float64)
        metrics = evaluate(g, z, y, vy, tau)
        checks = {
            "train_BA_50": (metrics["train"]["balanced_accuracy"], float(row["train_BA_50"])),
            "val_BA_50": (metrics["validation"]["balanced_accuracy"], float(row["val_BA_50"])),
            "I_50": (metrics["train"]["I"], float(row["I_50"])),
            "R_JS_50": (metrics["train"]["R_JS"], float(row["R_JS_50"])),
            "train_loss_50": (metrics["train"]["balanced_sigmoid_margin_loss"], float(row["train_loss_50"])),
        }
        for name, (actual, previous) in checks.items():
            if abs(actual - previous) > 0.00003:
                raise AssertionError(f"tau={tau}: {name} differs: {actual} vs {previous}")
        results.append({"nref": 300, "reference_seed": int(row["seed"]), "tau": tau,
                        "source": "saved final shared-Z matrix, no re-training",
                        "saved_weights_rounded": True,
                        "roundtrip_checks": {k: abs(a-b) for k, (a,b) in checks.items()},
                        **metrics})
        print(f"REPLAY_47_300 tau={tau} val_BA={metrics['validation']['balanced_accuracy']:.9f}"
              f" I={metrics['train']['I']:.9f}", flush=True)
    report_path = HERE / "results/metrics_full.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["full300_replay"] = {
        "status": "replayed_verified_saved_policy_no_training",
        "data_source": "archive/margin_training/results/full300_policy_snapshots.json",
        "no_historical_Test": True,
        "protocol": "original shared-Z exact expectation; class-independent inference",
        "initial_uniform": baseline,
        "trained": results
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    coverage_path = HERE / "results/coverage.json"
    cov = json.loads(coverage_path.read_text(encoding="utf-8"))
    cov["measurement_counts"]["full300_policy_checkpoints_verified_v2"] = 3
    cov["measured"].append({
        "scope": "three full300 trained policies, final 50-step saved-Z",
        "ids": ["R_KL","R_JS","R_sqrt_JS","T","BT","TNL","BTNL","I",
                "pairwise_center_JS","BA","ordinary_accuracy","confusion",
                "per_class_recall","margins","NLL","sigmoid_margin"],
        "source": "no-training replay of persisted rounded final matrices; numeric checks against archive summary"
    })
    for kind in ("not_measured", "not_verified"):
        for r in cov[kind]:
            if r.get("scope", "").startswith("twelve training runs"):
                r["scope"] = "nine 72-member subset training runs (full300 final policies verified separately)"
    cov["full300_replay_verified"] = True
    coverage_path.write_text(json.dumps(cov, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                             encoding="utf-8")
    print("E-CH4-07 saved-Z audit verified, Test not accessed", flush=True)


if __name__ == "__main__":
    main()
