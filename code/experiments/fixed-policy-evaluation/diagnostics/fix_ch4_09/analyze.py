"""FIX-CH4-09: lightweight, reproducible historical Uniform MC/exact comparison.

Only reproduces the 300-walk *fixed* Uniform policy once, without Test or
training, against the ALREADY STORED exact per-sample H=12 E08 baseline.
Historical generated_data/*.npz are NOT tracked, so their bytes cannot be
asserted equal: independent regenerated Train/Validation hashes are checked
against the existing E08 provenance and the exact E05 centers/predictions.
Default DRY_RUN. Never silently overwrite evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
LEGACY = ROOT / "code" / "policy_evaluator"
BENCH = ROOT / "code" / "experiments" / "benchmark_v1"
E05 = HERE.parents[1]
E08 = ROOT / "code" / "experiments" / "exact-matrix-evaluation" / "results"
CONFIG = ROOT / "code" / "data_generator" / "config.json"
OUT = HERE / "results" / "controlled_uniform_replay.json"

sys.path.insert(0, str(ROOT / "code" / "data_generator"))
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(LEGACY))
from generate import load_config
from dataset_adapter import dataset_train_val
from walks import ReferenceGraph, load_policy, reference_fingerprint, policy_fingerprint
from metrics import report, class_centers, jsd

POLICY = LEGACY / "policy_uniform.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(a):
    """Match benchmark_v1.snapshot_pipeline.array_fingerprint, NumPy only."""
    a = np.ascontiguousarray(np.asarray(a))
    hdr = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                     sort_keys=True).encode("utf8")
    return sha256(hdr + b"\0" + a.tobytes()).hexdigest()


def confusion_result(v, centers, y):
    stats, detail = report(v, centers, y, metric="js_similarity")
    pred = detail["prediction"]
    score = detail["decision_scores"]
    sorted_scores = np.sort(score, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    return {
        "balanced_accuracy": stats["balanced_accuracy"],
        "ordinary_accuracy": stats["accuracy"],
        "confusion_true_rows_pred_cols": stats["confusion"],
        "per_class_recall": stats["per_class_recall"],
        "predictions": pred.tolist(),
        "winning_margin_median": float(np.median(margin)),
        "nll_kl_balanced": stats["NLL"],
        "ties": int(np.sum(np.sum(np.isclose(score, score.max(axis=1, keepdims=True),
                                             rtol=0, atol=1e-12), axis=1)>1)),
    }


def locate_changes(a, b, labels):
    ia = np.flatnonzero(np.asarray(a) != np.asarray(b))
    return [{
        "validation_position": int(i),
        "true_label": int(labels[i]),
        "first_pred": int(a[i]), "second_pred": int(b[i])
    } for i in ia]


def analyze():
    start = time.perf_counter()
    spec = load_config(CONFIG)
    if int(spec["seed"]) != 20261102:
        raise AssertionError("Original generator seed must remain 20261102")
    x, y = dataset_train_val(spec)["train"]
    q, qy = dataset_train_val(spec)["validation"]
    if (x.shape, q.shape) != ((300, 5), (90, 5)):
        raise AssertionError("Split dimensions changed")
    if (np.bincount(y, minlength=3).tolist(),
        np.bincount(qy, minlength=3).tolist()) != ([72,147,81], [13,49,28]):
        raise AssertionError("Unexpected class counts")
    provenance = json.loads((E08/"provenance.json").read_text(encoding="utf8"))
    if digest(CONFIG) != provenance["generator_config_sha256"]:
        raise AssertionError("Generator bytes mismatch E08")
    ids = np.asarray(provenance["source_reference_ids"], dtype=np.int64)
    if not np.array_equal(np.sort(ids), np.arange(300)):
        raise AssertionError("E08 reference is not a permutation of all 300 Train")
    observed_fingerprints = {
        "train_features": fingerprint(x[ids]),
        "train_labels": fingerprint(np.asarray(y[ids], np.int64)),
        "validation_features": fingerprint(q),
    }
    e08_fingerprints = {
        "train_features": provenance["train_reference_x_sha256"],
        "train_labels": provenance["train_reference_y_sha256"],
        "validation_features": provenance["validation_x_sha256"],
    }
    fingerprint_match = {
        k: observed_fingerprints[k] == e08_fingerprints[k]
        for k in observed_fingerprints
    }
    if not fingerprint_match["train_labels"]:
        raise AssertionError("Train label generator/ordering differs from E08")
    # A floating-array byte hash can be sensitive to NumPy BLAS/NaN bits.
    # Preserve any mismatch; it is an empirical finding, not a reason to
    # silently claim that generated_data and the saved E08 reference coincide.
    print("FIX09_ARRAY_HASHES", json.dumps({
        "local":observed_fingerprints,
        "E08":e08_fingerprints,
        "matches":fingerprint_match
    }))

    # E08 exact distributions are ALREADY calculated and persisted.
    with np.load(E08/"raw_step_distributions.npz", allow_pickle=False) as raw:
        exact_train = np.empty((300,3),dtype=float)
        exact_train[ids] = raw["uniform_train_exact"]
        exact_val = raw["uniform_validation_exact"].copy()
    exact_centers = class_centers(exact_train,y)
    e05 = json.loads((E05/"results"/"centers_comparison.json").read_text(encoding="utf8"))
    e05_uniform = next(r for r in e05["results"] if r["alpha"] == 0)
    center_error = float(np.max(np.abs(exact_centers -
                        np.asarray(e05_uniform["centers"]))))
    if center_error>1e-10:
        raise AssertionError("E08 exact Uniform centers disagree with E05")
    saved = json.loads((E05/"results"/"validation_predictions.json").read_text(encoding="utf8"))
    saved_uniform = next(r for r in saved["by_alpha"] if r["alpha"] == 0)
    exact_stats = confusion_result(exact_val, exact_centers, qy)
    if exact_stats["predictions"] != saved_uniform["predictions"]:
        raise AssertionError("E08/E05 exact per-sample predictions disagree")
    if exact_stats["confusion_true_rows_pred_cols"] != saved_uniform["confusion"]:
        raise AssertionError("E08/E05 exact confusion disagree")
    if abs(exact_stats["balanced_accuracy"] - .9676870748299321)>1e-12:
        raise AssertionError("Expected 96.77% exact BA baseline")

    policy = load_policy(POLICY,5)
    if not (policy["walks_per_sample"]==300 and policy["seed"]==20261103
            and policy["max_steps"]==12 and
            abs(policy["beta"]-.4)<1e-14 and
            abs(policy["lambda"]-.8)<1e-14 and np.all(policy["logits"]==0)
            and np.array_equal(policy["kernel_gamma"],spec["gamma"])):
        raise AssertionError("Historical fixed Uniform policy config changed")
    # Actual NumPy historical MC path, not a newly invented sampler:
    graph = ReferenceGraph(x,y,policy["kernel_gamma"])
    mc_train, no_walk_train = graph.representation(
        x,policy,train_indices=np.arange(len(y)))
    mc_val, no_walk_val = graph.representation(q,policy)
    if no_walk_train.any() or no_walk_val.any():
        raise AssertionError("Dead end; no walk")
    mc_centers = class_centers(mc_train,y)
    cases = {
        "mc_val_mc_centers": confusion_result(mc_val,mc_centers,qy),
        "mc_val_exact_centers": confusion_result(mc_val,exact_centers,qy),
        "exact_val_mc_centers": confusion_result(exact_val,mc_centers,qy),
        "exact_val_exact_centers": exact_stats,
    }
    # Keep the ORIGINAL historic published value and flag, do not force match.
    hist_BA = .8825
    replicate_BA = cases["mc_val_mc_centers"]["balanced_accuracy"]
    same_rounded_two_decimals = round(replicate_BA*100,2) == 88.25
    differences = {
        "MC_vs_exact": locate_changes(cases["mc_val_mc_centers"]["predictions"],
                                      exact_stats["predictions"],qy),
        "MC_query_only_vs_exact": locate_changes(
            cases["mc_val_exact_centers"]["predictions"],
            exact_stats["predictions"],qy),
        "MC_centers_only_vs_exact": locate_changes(
            cases["exact_val_mc_centers"]["predictions"],
            exact_stats["predictions"],qy),
    }
    dtr = .5 * np.abs(mc_train - exact_train).sum(axis=1)
    dva = .5 * np.abs(mc_val - exact_val).sum(axis=1)
    cm = .5 * np.abs(mc_centers - exact_centers).sum(axis=1)
    result = {
        "schema":"FIX-CH4-09-controlled-uniform-replay-v1",
        "scope":"one deterministic replay of the original fixed 300-walk MC; exact distributions loaded from saved E08",
        "retrain":False, "historical_test_loaded":False,
        "historical_generated_data_npz_available_in_git":False,
        "data": {
            "generator_config":str(CONFIG.relative_to(ROOT)),
            "generator_config_sha256":digest(CONFIG),
            "train_n":300,"validation_n":90,
            "train_class_counts":[72,147,81],
            "validation_class_counts":[13,49,28],
            "data_seed":int(spec["seed"]),
            "train_canonical_fingerprint":fingerprint(x),
            "validation_fingerprint":fingerprint(q),
            "train_reference_fingerprint_legacy":reference_fingerprint(x,y),
            "matches_existing_e08_train_validation_fingerprints":all(fingerprint_match.values()),
            "E08_fingerprint_match_by_split":fingerprint_match,
            "local_fingerprints":observed_fingerprints,
            "E08_fingerprints":e08_fingerprints,
            "original_historical_generated_data_byte_identity_verifiable":False
        },
        "policy": {
            "path":str(POLICY.relative_to(ROOT)),
            "sha256":digest(POLICY),
            "fingerprint":policy_fingerprint(policy),
            "beta":policy["beta"],"lambda":policy["lambda"],"H":policy["max_steps"],
            "MC_walks_per_origin":policy["walks_per_sample"],
            "MC_seed":policy["seed"],
            "gamma":policy["kernel_gamma"].tolist(),
            "policy_family":"shared, logits all zeros"
        },
        "reference": {
            "saved_exact_step_distributions":str((E08/"raw_step_distributions.npz").relative_to(ROOT)),
            "saved_exact_npz_sha256":digest(E08/"raw_step_distributions.npz"),
            "saved_exact_provenance":str((E08/"provenance.json").relative_to(ROOT)),
            "E05_exact_center_max_abs_error":center_error,
            "E05_exact_predictions_match_per_sample":True,
            "max_center_tv_replay_vs_exact":float(cm.max()),
            "per_class_center_tv_replay_vs_exact":cm.tolist(),
            "train_TV_MC_vs_exact":{"mean":float(dtr.mean()),"max":float(dtr.max())},
            "validation_TV_MC_vs_exact":{"mean":float(dva.mean()),"max":float(dva.max())}
        },
        "published_historical_mc_BA_rounded_percent":88.25,
        "published_exact_BA_percent":96.77,
        "replayed_MC_BA_equals_historical_to_two_decimals":same_rounded_two_decimals,
        "replayed_MC_BA_minus_historical_unrounded_reference":replicate_BA-hist_BA,
        "cases":cases,
        "disagreement_counts":{name:len(v) for name,v in differences.items()},
        "disagreement_rows":differences,
        "replayed_center_rows":mc_centers.tolist(),
        "exact_center_rows":exact_centers.tolist(),
        "execution_seconds":float(time.perf_counter()-start),
        "limitation":"Saved historical generated_data NPZ and historical per-sample MC predictions are not tracked; reproduction uses source code/config and hashes against stored E08 baseline."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():
        raise FileExistsError("Controlled replay already persisted; do not repeat")
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,
                              allow_nan=False)+"\n",encoding="utf8")
    print("FIX09_MCID",json.dumps({
        "replayed_mc_BA":replicate_BA,
        "historical_two_decimal_match":same_rounded_two_decimals,
        "exact_BA":exact_stats["balanced_accuracy"],
        "crossed_BAs":{key:val["balanced_accuracy"] for key,val in cases.items()},
        "confusions":{key:val["confusion_true_rows_pred_cols"] for key,val in cases.items()},
        "different_predictions":result["disagreement_counts"],
        "train_mc_tv_mean":result["reference"]["train_TV_MC_vs_exact"]["mean"],
        "validation_mc_tv_mean":result["reference"]["validation_TV_MC_vs_exact"]["mean"],
        "center_tv_max":result["reference"]["max_center_tv_replay_vs_exact"],
        "elapsed_s":result["execution_seconds"]
    },ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({"status":"DRY_RUN","retraining":False,
            "test_access":False,"replayed_MC_origin_walks":300,
            "will_load_saved_E08_exact":True,"result":str(OUT.relative_to(ROOT))}))
        return
    analyze()

if __name__=="__main__":
    main()
