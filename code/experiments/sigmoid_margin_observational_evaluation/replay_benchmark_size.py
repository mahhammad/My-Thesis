"""E-CH4-07: recover full common metrics of 36 standard Benchmark size runs.

Input: immutable ZIP checkpoints and master_runs_76.csv with SHA256.
No optimizer, no Monte Carlo, no Test split or historical Test access.
One exact forward evaluation of each saved logits tensor.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BENCHMARK = HERE.parent / "benchmark_v1"
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(ROOT / "code/data_generator"))
from generate import load_config
from dataset_adapter import dataset_train_val
from engine import ExactMaskedWalk, DT
from report_snapshot import standard_metrics


def load_archived_by_sha():
    bundles = [
        BENCHMARK / "results/size_25_24runs_seed20261102.zip",
        BENCHMARK / "results/size_25_24runs_seeds20261103_20261104.zip",
    ]
    found = {}
    for path in bundles:
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                if not item.filename.endswith(".json") or item.is_dir():
                    continue
                data = z.read(item)
                digest = hashlib.sha256(data).hexdigest()
                found[digest] = (data, path.name, item.filename)
    return found


def eval_one(entry, checkpoint, splits, spec):
    n = int(entry["n"])
    family = entry["family"]
    model = json.loads(checkpoint)
    ds = model["dataset"]
    ids = np.asarray(ds["reference_ids"], dtype=int)
    x, y = splits["train"]
    vx, vy = splits["validation"]
    if (len(ids) != n or ds["seed"] != int(entry["data_seed"]) or
            ds["n_reference"] != n or model["family"] != family or
            model["objective"] != "margin_JSD" or model["steps"] != 25):
        raise ValueError("Run metadata disagrees with official Benchmark CSV")
    if len(np.unique(ids)) != n or not (0 <= ids).all() or not (ids < len(y)).all():
        raise ValueError("Invalid saved reference ids")
    counts = np.bincount(y[ids], minlength=3)
    if counts.tolist() != ds["class_counts"]:
        raise ValueError("Reconstructed classes differ from saved checkpoint")
    g = ExactMaskedWalk(x[ids], y[ids], spec["gamma"], beta=.4, lam=.8,
                        horizon=12, query_x=vx)
    z = torch.as_tensor(model["raw_logits"], dtype=DT)
    with torch.no_grad():
        if family == "shared":
            p = g.represent_train(z, family).numpy()
            train_by = np.broadcast_to(p[:, None, :], (n, g.k, g.k))
            q = g.represent_query(z, family).numpy()
            val_by = np.broadcast_to(q[:, None, :], (len(vy), g.k, g.k))
        elif family == "class":
            train_by = np.stack([
                g.represent_train(z, family, c).numpy() for c in range(g.k)
            ], axis=1)
            val_by = np.stack([
                g.represent_query(z, family, c).numpy() for c in range(g.k)
            ], axis=1)
        else:
            raise ValueError(f"Only shared/class can be used here: {family}")
    centers = np.stack([train_by[y[ids] == c, c].mean(0) for c in range(g.k)])
    tr = standard_metrics(train_by, centers, y[ids], .005,
                          split="train", per_sample=False)
    val = standard_metrics(val_by, centers, vy, .005,
                           split="validation", per_sample=False)
    checks = {
        "train_BA": tr["balanced_accuracy"],
        "val_BA": val["balanced_accuracy"],
        "I": tr["I"],
        "R_KL": tr["R_KL"],
        "R_JS": tr["R_JS"],
        "R_sqrt_JS": tr["R_sqrt_JS"],
        "objective_loss": tr["balanced_sigmoid_margin_loss"],
    }
    for key, actual in checks.items():
        expected = float(entry[key])
        if actual is None or abs(actual - expected) > 2e-8:
            raise AssertionError(
                f"Evidence mismatch n={n} family={family} data_seed={entry['data_seed']} "
                f"{key}: {actual} != {expected}"
            )
    return {
        "n": n, "family": family, "data_seed": int(entry["data_seed"]),
        "source_sha256": entry["source_sha256"],
        "centers": centers.tolist(), "train": tr, "validation": val,
        "max_diff_vs_benchmark": max(
            abs(actual - float(entry[key])) for key, actual in checks.items()
        ),
    }


def main():
    torch.set_num_threads(1)
    with (HERE / "results/benchmark_size_3seeds.csv").open(newline="",encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    if len(runs) != 36:
        raise ValueError("Expected 36 selected Benchmark runs")
    zips = load_archived_by_sha()
    reports = []
    cache = {}
    for i, row in enumerate(runs):
        sha = row["source_sha256"]
        if sha not in zips:
            raise FileNotFoundError(f"Original Benchmark checkpoint with SHA256 {sha} not found")
        payload, bundle, original_name = zips[sha]
        seed = int(row["data_seed"])
        if seed not in cache:
            spec = load_config(ROOT / "code/data_generator/config.json")
            spec["seed"] = seed
            cache[seed] = (dataset_train_val(spec), spec)
        splits, spec = cache[seed]
        record = eval_one(row, payload, splits, spec)
        record["raw_bundle"] = bundle
        record["raw_name"] = original_name
        reports.append(record)
        print(f"VERIFY_SIZE {i+1}/36 n={row['n']} {row['family']} seed={seed} "
              f"max_err={record['max_diff_vs_benchmark']:.3e}",flush=True)
    if (len(reports) != 36 or
        len({(r["n"],r["family"],r["data_seed"]) for r in reports}) != 36):
        raise AssertionError("Incorrect 36-run coverage")
    out = {
        "schema": "E-CH4-07-size-benchmark-common-metrics-v2",
        "status": "verified_saved_checkpoints_exact_forward",
        "experiment_id": "E-CH4-07",
        "source": "benchmark_v1/results/master_runs_76.csv and immutable size-study ZIPs",
        "retraining": False, "new_walk_sampling": False, "test_accessed": False,
        "policy_families": ["shared", "class"],
        "generator_seeds": [20261102,20261103,20261104],
        "n_values": [50,100,150,200,250,300],
        "fixed_budget_steps": 25,
        "tau": .005,
        "runs": reports,
    }
    target = HERE / "results/benchmark_size_metrics_full.json"
    target.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                      encoding="utf8")
    coverage = HERE / "results/coverage.json"
    cov = json.loads(coverage.read_text(encoding="utf8"))
    cov["benchmark_full_v2_replay"] = {
        "status": "36/36 exact forward passes verified against original Benchmark",
        "source": "results/benchmark_size_metrics_full.json",
        "complete_train_validation_metrics": True,
        "new_training": False, "independent_data_seeds": 3, "test": False
    }
    cov["measurement_counts"]["benchmark_full_v2_verified_runs"] = 36
    cov["measured"].append({
        "scope": "36 Benchmark size runs, full Train/Validation common contract",
        "ids": ["R_KL","R_JS","R_sqrt_JS","T","BT","TNL","BTNL","I",
                "pairwise_JS","pairwise_sqrt_JS","confusion",
                "per_class_recall","margin_quantiles","NLL","BA","Acc"],
        "source": "exact replay of stored logits with SHA256 cross-check"
    })
    for item in cov.get("not_measured", []):
        if item.get("scope","").startswith("standardized Benchmark size study"):
            item["scope"] += " (superseded by verified full forward replay)"
            item["reason"] += " Recovered via benchmark_size_metrics_full.json."
    coverage.write_text(json.dumps(cov,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                        encoding="utf8")
    print("E-CH4-07 36/36 full contract verified; no Test or retraining", flush=True)


if __name__ == "__main__":
    main()
