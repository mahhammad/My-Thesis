"""E-CH4-08: preregistered, no-training exact matrix-walk validation.

The default is a dry run. --execute computes E08 once, from SHA-verified
Benchmark v1 checkpoint data. It NEVER generates or loads Test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import platform
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = ROOT / "code" / "experiments" / "benchmark_v1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "code" / "data_generator"))
from generate import load_config
from dataset_adapter import dataset_train_val
from engine import ExactMaskedWalk, DT, coefficients
from snapshot_pipeline import save_snapshot, array_fingerprint
from report_snapshot import report_snapshot, save_report, standard_metrics
from walk_analysis import (
    BETA, LAM, H, DEPTHS, MC_BUDGETS, MC_SEEDS, MC_ORIGINS,
    forward_steps, expectation, analyze_depths, monte_carlo
)
from presentation import make_assets, make_report

ARCHIVE = ROOT / "code/experiments/standard-policy-comparison/results/core_50_16runs.zip"
ARCHIVE_SHA256 = "9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040"
CONFIG = ROOT / "code" / "data_generator" / "config.json"
DATA_SEED = 20261102
TAU = 0.005
RESULTS = HERE / "results"
POLICY_NAMES = ("uniform", "trained_shared")


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf8")


def hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def original_checkpoint():
    """Use only the registered archive and member covered by its own manifest."""
    if hash_file(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Benchmark core archive SHA256 mismatch")
    matches = []
    with zipfile.ZipFile(ARCHIVE) as zf:
        manifests = [s for s in zf.namelist() if s == "manifest.json"
                     or s.endswith("/manifest.json")]
        if not manifests:
            raise ValueError("Missing original archive manifest")
        for manifest_name in manifests:
            prefix = manifest_name[:-len("manifest.json")]
            manifest = json.loads(zf.read(manifest_name))
            for item in manifest["runs"]:
                member = prefix + item["file"]
                raw = zf.read(member)
                digest = sha256(raw).hexdigest()
                if digest != item["sha256"]:
                    raise ValueError("Checkpoint member digest mismatch: " + member)
                candidate = json.loads(raw)
                data = candidate.get("dataset", {})
                dseed = int(data.get("data_seed", data.get("seed", -1)))
                if (candidate.get("family") == "shared"
                        and candidate.get("objective") == "margin_JSD"
                        and abs(float(candidate.get("tau", -1)) - TAU) < 1e-13
                        and int(candidate.get("steps", -1)) == 50
                        and dseed == DATA_SEED
                        and len(data.get("reference_ids", [])) == 300):
                    matches.append((member, digest, candidate))
    if len(matches) != 1:
        raise ValueError("Expected exactly ONE preregistered shared checkpoint; found "
                         + str(len(matches)))
    return matches[0]


def model_sources():
    member, member_sha, checkpoint = original_checkpoint()
    spec = load_config(CONFIG)
    spec["seed"] = DATA_SEED
    splits = dataset_train_val(spec)   # Only Train and Validation: NEVER Test.
    x, y = splits["train"]
    vx, vy = splits["validation"]
    ids = np.asarray(checkpoint["dataset"]["reference_ids"], dtype=np.int64)
    if not np.array_equal(np.sort(ids), np.arange(300)):
        raise ValueError("Checkpoint must contain EVERY Train sample exactly once")
    if len(vy) != 90 or len(x) != 300:
        raise ValueError("Pre-registered sample counts mismatch")
    if not np.array_equal(np.unique(y), np.arange(len(spec["pi"]))):
        raise ValueError("Class absent in Train")
    graph = ExactMaskedWalk(x[ids], y[ids], spec["gamma"],
                            beta=BETA, lam=LAM, horizon=H, query_x=vx)
    trained = torch.as_tensor(checkpoint["raw_logits"], dtype=DT)
    graph.check_logits(trained, "shared")
    uniform = torch.zeros_like(trained)
    provenance = {
        "original_archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_sha256": ARCHIVE_SHA256,
        "original_member": member,
        "member_sha256": member_sha,
        "checkpoint_family": "shared",
        "checkpoint_objective": "margin_JSD",
        "checkpoint_tau": TAU,
        "checkpoint_steps": 50,
        "source_reference_ids": ids.tolist(),
        "reference_seed": int(checkpoint["dataset"].get("reference_seed", 20261107)),
        "reference_order_sha256": array_fingerprint(ids),
        "generator_config": CONFIG.relative_to(ROOT).as_posix(),
        "generator_config_sha256": hash_file(CONFIG),
        "train_reference_x_sha256": array_fingerprint(x[ids]),
        "train_reference_y_sha256": array_fingerprint(np.asarray(y[ids], dtype=np.int64)),
        "validation_x_sha256": array_fingerprint(vx),
        "data_seed": DATA_SEED,
        "train_count": int(len(ids)), "validation_count": int(len(vy)),
        "historical_test_accessed": False,
        "retrained": False,
        "preregistration": "code/experiments/exact-matrix-evaluation/PROTOCOL_FA.md",
    }
    return graph, np.asarray(y[ids], int), np.asarray(vy, int), ids, \
           {"uniform": uniform, "trained_shared": trained}, provenance, spec


def graph_size(g):
    avail = g.available.numpy()
    m = int(avail.sum())
    vertical = int(sum(int(n) * (int(n) - 1) for n in avail.sum(0)))
    horizontal = int(np.sum(avail.sum(1) ** 2))  # Include legal a->a feature self-choices.
    return {
        "n": g.n, "d": g.d, "k": g.k,
        "m_sample_feature_nodes": m,
        "directed_vertical_possible": vertical,
        "horizontal_possible": horizontal,
        "horizontal_feature_self_choices_included": True,
        "old_dense_n_d_n_n_bytes": int(g.n*g.d*g.n*g.n*8),
        "actual_implementation": "factored dense Torch kernels; no n*d*n*n allocation",
    }


def alpha_diagnostics(a):
    a = np.asarray(a, dtype=float)
    lengths = np.arange(1, H+1)
    probs = np.array([BETA*(1-BETA)**(t-1) if t<H
                      else (1-BETA)**(H-1) for t in lengths], dtype=float)
    if not np.isclose(probs.sum(), 1., atol=1e-13, rtol=0):
        raise AssertionError("Invalid capped geometric distribution")
    if not np.isclose(a.sum(), 1., atol=1e-13, rtol=0):
        raise AssertionError("Invalid within-walk normalized coefficients")
    direct = np.array([
        sum(probs[L-1] * LAM**(t-1) /
            sum(LAM**j for j in range(L)) for L in range(t, H+1))
        for t in lengths
    ])
    if not np.allclose(a, direct, atol=1e-13, rtol=0):
        raise AssertionError("Alpha mismatch against independent length sum")
    return {
        "beta": BETA, "lambda": LAM, "H": H,
        "length_pmf": probs.tolist(),
        "alpha": a.tolist(),
        "expected_realized_walk_length": float(np.dot(probs, lengths)),
        "expected_weighted_step_index": float(np.dot(a, lengths)),
        "survival_at_t": [(1-BETA)**(t-1) for t in lengths],
        "truncation_epsilon": {str(r): float(sum(a[r:])) for r in DEPTHS},
        "termination_at_H_probability": float(probs[-1]),
        "coefficient_mass_error": float(abs(a.sum()-1.)),
        "independent_coefficient_max_abs_error": float(max(abs(a-direct))),
    }


def run(*, code_commit=None):
    torch.set_num_threads(1)
    wall_start = time.perf_counter()
    graph, yt, yv, ids, policies, source, spec = model_sources()
    alpha = graph.alpha.detach().cpu().numpy()
    length = alpha_diagnostics(alpha)
    raw_steps = {}
    exact_train, exact_validation = {}, {}
    results = {}
    metrics = {}
    max_error = {"train": 0.0, "validation": 0.0}
    for policy_name in POLICY_NAMES:
        z = policies[policy_name]
        with torch.no_grad():
            step_t = forward_steps(graph, z, "train")
            step_v = forward_steps(graph, z, "validation")
            train = expectation(step_t, alpha, H)
            validation = expectation(step_v, alpha, H)
            official_train = graph.represent_train(z, "shared").numpy()
            official_validation = graph.represent_query(z, "shared").numpy()
        err_t = float(np.max(np.abs(train - official_train)))
        err_v = float(np.max(np.abs(validation - official_validation)))
        if max(err_t, err_v) > 1e-9:
            raise AssertionError("Independent forward recursion mismatches official engine")
        max_error["train"] = max(max_error["train"], err_t)
        max_error["validation"] = max(max_error["validation"], err_v)
        exact_train[policy_name] = train
        exact_validation[policy_name] = validation
        raw_steps[policy_name + "_train_steps"] = step_t
        raw_steps[policy_name + "_validation_steps"] = step_v
        raw_steps[policy_name + "_train_exact"] = train
        raw_steps[policy_name + "_validation_exact"] = validation
        trunc = analyze_depths(step_t, step_v, alpha, yt, yv)
        results[policy_name] = {
            "max_abs_error_independent_vs_official_train": err_t,
            "max_abs_error_independent_vs_official_validation": err_v,
            "truncation": trunc,
        }
        rows = {}
        for depth in DEPTHS:
            t = expectation(step_t, alpha, depth)
            v = expectation(step_v, alpha, depth)
            centers = np.stack([t[yt==c].mean(0) for c in range(graph.k)])
            arr_t = np.broadcast_to(t[:, None, :], (len(t), graph.k, graph.k)).copy()
            arr_v = np.broadcast_to(v[:, None, :], (len(v), graph.k, graph.k)).copy()
            tr = standard_metrics(arr_t, centers, yt, TAU, split="train", per_sample=False)
            va = standard_metrics(arr_v, centers, yv, TAU, split="validation", per_sample=False)
            rows[str(depth)] = {"train": tr, "validation": va,
                                "centers": centers.tolist()}
        metrics[policy_name] = rows

        # Immutable raw v2 snapshot: one for each original FULL H=12 model only.
        snap = RESULTS / "snapshots" / policy_name
        model = save_snapshot(
            snap, graph, z, "shared", reference_ids=ids,
            config_path=CONFIG, config_reference=source["generator_config"],
            data_seed=DATA_SEED,
            reference_seed=int(source.get("reference_seed", 20261107)),
            objective=("uniform" if policy_name == "uniform" else "margin_JSD"),
            tau=TAU, beta=BETA, lam=LAM, horizon=H, validation_x=graph.qx,
            code_commit=code_commit, notes=("E08: original archived checkpoint "
                                            if policy_name != "uniform" else "E08: Z=0")
                                            + source["original_member"],
            experiment_metadata={"training_repeated": False,
                                 "e08_policy_name": policy_name},
        )
        public = report_snapshot(snap, per_sample=False)
        save_report(snap, public)
        for split in ("train", "validation"):
            if abs(public[split]["balanced_accuracy"] -
                   rows["12"][split]["balanced_accuracy"]) > 1e-10:
                raise AssertionError("Independent common reporter metrics differ")

    mc = monte_carlo(graph, policies, exact_train, exact_validation)
    mc_h1 = {}
    for policy_name in POLICY_NAMES:
        e128 = next(x["mean_tv"] for x in mc["summary"]
                    if x["policy"]==policy_name and x["nwalks"]==128)
        e2048 = next(x["mean_tv"] for x in mc["summary"]
                     if x["policy"]==policy_name and x["nwalks"]==2048)
        mc_h1[policy_name] = bool(e2048 < e128)  # Falsifiable; preserve failures.

    # Median of 3 AFTER one warm-up per case, CPU one thread.
    positions = np.linspace(0, graph.n-1, 24, dtype=int)
    timing = []
    for policy_name in POLICY_NAMES:
        for depth in (2,4,12):
            z = policies[policy_name]
            with torch.no_grad():
                forward_steps(graph, z, "train", max_depth=depth,
                              origin_positions=positions)
                durations = []
                for _ in range(3):
                    start = time.perf_counter()
                    forward_steps(graph, z, "train", max_depth=depth,
                                  origin_positions=positions)
                    durations.append(time.perf_counter() - start)
            timing.append({"policy": policy_name, "r": depth,
                           "origin_positions": positions.tolist(),
                           "repeats_seconds": durations,
                           "median_seconds": float(np.median(durations)),
                           "threads": torch.get_num_threads(),
                           "warmup_repeats": 1})
    # ru_maxrss in KiB on GitHub Actions Linux, not a per-model allocation.
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "schema": "e-ch4-08-experiment-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": source,
        "graph_size": graph_size(graph),
        "length_and_coefficients": length,
        "policy_results": results,
        "monte_carlo": mc,
        "timing": timing,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_used": False, "rss_peak_kib_linux": peak_rss_kib,
            "rss_measures_entire_process_not_policy": True,
            "wall_seconds": float(time.perf_counter()-wall_start)
        },
        "hypothesis_outcomes": {
            "independent_forward_matches_engine": True,
            "preregistered_mc_mean_tv_decreases_from_128_to_2048": mc_h1,
            "all_individual_and_center_truncation_bounds_hold": True,
        },
        "historical_test_accessed": False, "training_performed": False
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "raw_step_distributions.npz", **raw_steps)
    dump(RESULTS / "experiment_report.json", report)
    dump(RESULTS / "metrics_full.json", {
        "schema": "e08-common-metrics-by-depth-v1",
        "tau_report": TAU, "policies": metrics,
        "depths": list(DEPTHS), "historical_test_accessed": False})
    dump(RESULTS / "coverage.json", {
        "schema": "e08-coverage-v1",
        "policies": list(POLICY_NAMES), "depths": list(DEPTHS),
        "train_samples_per_run": len(yt), "validation_samples_per_run": len(yv),
        "metrics_complete": True,
        "common_metrics": ["R_KL", "R_JS", "R_sqrt_JS", "T", "BT",
                           "TNL", "BTNL", "I", "balanced_accuracy",
                           "ordinary_accuracy", "confusion_true_rows_pred_cols",
                           "balanced_sigmoid_margin_loss",
                           "balanced_JS_similarity_NLL",
                           "balanced_softplus_all_rivals",
                           "center_geometry", "margin_quantiles"],
        "train_validation_policy_depth_rows": 2*len(DEPTHS)*2,
        "monte_carlo_records": len(mc["raw"]),
        "train_origins": list(MC_ORIGINS["train"]),
        "validation_origins": list(MC_ORIGINS["validation"]),
        "MC_budgets": list(MC_BUDGETS), "MC_seeds": list(MC_SEEDS),
        "historical_test_accessed": False})
    source["engine_comparison_max_abs_error"] = max_error
    source["raw_step_distributions"] = str(
        (RESULTS / "raw_step_distributions.npz").relative_to(ROOT))
    source["original_archive_verified"] = True
    source["snapshot_schema"] = "imc-snapshot-v2"
    source["results_paths"] = [
        str(p.relative_to(ROOT)) for p in [
            RESULTS / "experiment_report.json", RESULTS / "metrics_full.json",
            RESULTS / "coverage.json", RESULTS / "raw_step_distributions.npz",
            RESULTS / "snapshots" / "uniform" / "model.json",
            RESULTS / "snapshots" / "trained_shared" / "model.json"
        ]]
    dump(RESULTS / "provenance.json", source)
    make_assets(report)
    make_report(report)
    print(json.dumps({
        "result": "PASS", "run_seconds": report["environment"]["wall_seconds"],
        "independent_engine_error": max_error,
        "mc_summary": mc["summary"],
        "effective_length": length["expected_realized_walk_length"],
        "depth_3_tail_bound": length["truncation_epsilon"]["3"],
        "raw_dir": str(RESULTS)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN", "will_train": False,
                          "will_access_test": False,
                          "archive": str(ARCHIVE.relative_to(ROOT)),
                          "raw_dir": str(RESULTS.relative_to(ROOT)),
                          "reference": 300, "validation": 90}, indent=2))
        return
    if (RESULTS / "experiment_report.json").exists():
        raise FileExistsError("Existing E08 results: inspect instead of rerunning")
    run(code_commit=args.code_commit)


if __name__ == "__main__":
    main()
