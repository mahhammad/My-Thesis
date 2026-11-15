"""E-CH4-09: true-horizon sensitivity, entirely offline from saved E08 steps.

No data generator, training, graph walk engine, Monte Carlo, Test or model fit.
Default dry-run. Fails closed on provenance/numerical inconsistency, and
refuses to overwrite any already persisted scientific evidence.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E08 = HERE.parent / "exact_matrix_walk_validation"
E04 = HERE.parent / "uniform-hyperparameter-sensitivity"
LABELS = HERE.parent / "target_center_guidance_trajectory" / "results" / "representations_uniform.json"
ARCHIVE = HERE.parent / "standard_policy_comparison" / "results" / "core_50_16runs.zip"
OUT = HERE / "results"
sys.path.insert(0, str(HERE.parent / "benchmark_v1"))
from snapshot_pipeline import array_fingerprint
from report_snapshot import standard_metrics
from engine import coefficients as benchmark_coefficients

HORIZONS = (1, 2, 3, 4, 8, 12)
POLICIES = ("uniform", "trained_shared")
BETA, LAM, TAU = .4, .8, .005
EPS = 2e-10
FIELDS = (
    "balanced_accuracy", "ordinary_accuracy", "R_JS", "R_sqrt_JS",
    "R_KL", "T", "BT", "TNL", "BTNL", "I", "tie_count",
    "NLL_KL_simple", "NLL_KL_normalized_historical",
    "balanced_JS_similarity_NLL", "balanced_sigmoid_margin_loss",
    "balanced_softplus_all_rivals", "min_pairwise_sqrt_JS",
    "mean_pairwise_sqrt_JS", "mean_pairwise_JS",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf8")


def coefficients(h, beta=BETA, lam=LAM):
    """Independent explicit capped-geometric length sum, float64."""
    if not isinstance(h, int) or not 1 <= h <= 12:
        raise ValueError("Invalid true horizon")
    lengths = np.arange(1, h + 1)
    pmf = np.array([beta * (1 - beta)**(t - 1) if t < h
                    else (1-beta)**(h-1) for t in lengths], dtype=np.float64)
    alpha = np.zeros(h, dtype=np.float64)
    for length, prob in zip(lengths, pmf):
        within = lam**np.arange(length, dtype=np.float64)
        alpha[:length] += prob * within / within.sum()
    if not (np.all(alpha >= -1e-15) and
            abs(alpha.sum() - 1.) < 1e-13 and abs(pmf.sum() - 1.) < 1e-13):
        raise AssertionError("True-H probability coefficients invalid")
    return pmf, alpha


def independent_small_h(h):
    """Alternative direct formula by t, not the vectorized coefficient loop."""
    p = lambda l: BETA*(1-BETA)**(l-1) if l < h else (1-BETA)**(h-1)
    return np.array([
        sum(p(l) * LAM**(t-1) / sum(LAM**j for j in range(l))
            for l in range(t, h+1))
        for t in range(1, h+1)
    ], dtype=np.float64)


def full_array_check(name, arr, shape):
    if arr.shape != shape or arr.dtype != np.float64:
        raise AssertionError(f"{name}: expected float64 {shape}, got {arr.dtype} {arr.shape}")
    if not np.isfinite(arr).all():
        raise AssertionError(f"{name}: non-finite distribution")
    if np.min(arr) < -1e-12:
        raise AssertionError(f"{name}: negative probability")
    if np.max(np.abs(arr.sum(axis=-1)-1.)) > 1e-9:
        raise AssertionError(f"{name}: probability normalization broken")
    return float(np.max(np.abs(arr.sum(axis=-1)-1.)))


def max_abs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch {a.shape}, {b.shape}")
    return float(np.max(np.abs(a-b)))


def by_hypothesis(arr):
    n,k = arr.shape
    return np.broadcast_to(arr[:, None, :], (n,k,k))


def class_centers(train, labels):
    return np.stack([train[labels == c].mean(axis=0)
                     for c in range(train.shape[1])])


def collapse(steps, alpha):
    return np.tensordot(alpha, steps[:len(alpha)], axes=(0,0))


def compare_scalar_report(observed, expected, location, *, tolerance=EPS):
    maxima = {}
    for key in FIELDS:
        x, y = observed.get(key), expected.get(key)
        if x is None or y is None:
            if x is not y:
                raise AssertionError(f"{location}: mismatched unavailable metric {key}")
        else:
            delta = abs(float(x)-float(y))
            maxima[key] = delta
            if delta > tolerance:
                raise AssertionError(f"{location}: {key} mismatch {x} vs {y}")
    for key in ("confusion_true_rows_pred_cols", "counts"):
        if observed[key] != expected[key]:
            raise AssertionError(f"{location}: {key} mismatch")
    for key in ("pairwise_center_sqrt_JS",):
        if max_abs(observed[key], expected[key]) > tolerance:
            raise AssertionError(f"{location}: {key} mismatch")
    if observed["tie_count"] != expected["tie_count"]:
        raise AssertionError(f"{location}: tie mismatch")
    return max(maxima.values(), default=0.)


def source_inventory():
    provenance = read_json(E08 / "results" / "provenance.json")
    metric_e08 = read_json(E08 / "results" / "metrics_full.json")
    e04 = read_json(E04 / "results" / "metrics_full.json")
    labels = read_json(LABELS)
    id_order = np.asarray(provenance["source_reference_ids"], dtype=np.int64)
    if id_order.shape != (300,) or not np.array_equal(np.sort(id_order), np.arange(300)):
        raise AssertionError("E08 IDs not a complete permutation of Train")
    y_train = np.asarray(labels["training"]["y"], dtype=np.int64)[id_order]
    y_val = np.asarray(labels["validation"]["y"], dtype=np.int64)
    if (y_train.shape, y_val.shape) != ((300,), (90,)):
        raise AssertionError("Labels or sample counts wrong")
    if not np.array_equal(np.bincount(y_train, minlength=3), [72,147,81]):
        raise AssertionError("Train label classes unexpected")
    if not np.array_equal(np.bincount(y_val, minlength=3), [13,49,28]):
        raise AssertionError("Validation label classes unexpected")
    if array_fingerprint(y_train) != provenance["train_reference_y_sha256"]:
        raise AssertionError("E06 saved Train label order/fingerprint differs from E08")
    if provenance["data_seed"] != 20261102 or provenance["retrained"] or provenance["historical_test_accessed"]:
        raise AssertionError("E08 provenance protocol mismatch")
    if provenance["train_count"] != 300 or provenance["validation_count"] != 90:
        raise AssertionError("Sample count mismatch")
    if metric_e08["historical_test_accessed"]:
        raise AssertionError("E08 metrics unexpectedly accessed Test")

    npz = E08 / "results" / "raw_step_distributions.npz"
    with ZipFile(npz) as archive:
        if archive.testzip() is not None:
            raise AssertionError("E08 raw step NPZ corrupt")
    step_arrays = {}
    required = {
        f"{p}_{split}_{field}" for p in POLICIES
        for split in ("train", "validation") for field in ("steps", "exact")
    }
    with np.load(npz, allow_pickle=False) as raw:
        if set(raw.files) != required:
            raise AssertionError(f"Unknown/missing E08 step keys: {set(raw.files)^required}")
        for p in POLICIES:
            for split, n in (("train",300),("validation",90)):
                for field, shape in (("steps",(12,n,3)),("exact",(n,3))):
                    key = f"{p}_{split}_{field}"
                    arr = raw[key].copy()
                    full_array_check(key, arr, shape)
                    step_arrays[key] = arr

    models={}
    stored={}
    for p in POLICIES:
        folder = E08 / "results" / "snapshots" / p
        model = read_json(folder / "model.json")
        if model["family"] != "shared" or model["data"]["test_accessed"]:
            raise AssertionError("Unexpected saved snapshot family/Test")
        if model["parameters"]["H"] != 12 or model["parameters"]["beta"] != BETA or model["parameters"]["lambda"] != LAM:
            raise AssertionError("Unexpected snapshot walk configuration")
        if model["parameters"]["dtype"] != "float64":
            raise AssertionError("Snapshot must use float64")
        if model["data"]["reference_ids"] != id_order.tolist() or model["data"]["data_seed"] != 20261102:
            raise AssertionError("Snapshot reference order/data seed mismatched")
        if model["data"]["train_reference_y_sha256"] != provenance["train_reference_y_sha256"]:
            raise AssertionError("Snapshot label provenance mismatch")
        if (model["data"]["train_reference_x_sha256"] != provenance["train_reference_x_sha256"] or
            model["data"]["validation_x_sha256"] != provenance["validation_x_sha256"]):
            raise AssertionError("Snapshot feature provenance mismatch")
        snapfile=folder/"distributions.npz"
        if digest(snapfile) != model["distributions_sha256"]:
            raise AssertionError("Stored E08 snapshot digest mismatch")
        with np.load(snapfile,allow_pickle=False) as z:
            data={k:z[k].copy() for k in z.files}
        for k in ("train_by_hypothesis","validation_by_hypothesis","centers","reference_ids"):
            if k not in data:
                raise AssertionError("Missing snapshot field " + k)
        if not np.array_equal(data["reference_ids"],id_order):
            raise AssertionError("Snapshot ordered IDs mismatch")
        for split,n in (("train",300),("validation",90)):
            expected = step_arrays[f"{p}_{split}_exact"]
            snapshot = data[f"{split}_by_hypothesis"]
            if snapshot.shape != (n,3,3) or max_abs(snapshot,by_hypothesis(expected)) > 2e-12:
                raise AssertionError("E08 exact per-sample snapshot mismatch")
        if max_abs(data["centers"],class_centers(step_arrays[f"{p}_train_exact"],y_train))>2e-12:
            raise AssertionError("E08 snapshot centers not Train class means")
        logits=np.asarray(model["parameters"]["logits"],dtype=np.float64)
        if logits.shape != (6,5):
            raise AssertionError("Unexpected shared logits shape")
        if p=="uniform" and not np.array_equal(logits,np.zeros((6,5))):
            raise AssertionError("Uniform logits are not zero")
        if p=="trained_shared" and (model["objective"]["name"] != "margin_JSD" or
                                    model["objective"]["tau"] != TAU):
            raise AssertionError("Trained policy objective mismatch")
        models[p] = model
        stored[p] = data

    if digest(ARCHIVE) != provenance["archive_sha256"]:
        raise AssertionError("Archived trained checkpoint digest mismatch")
    with ZipFile(ARCHIVE) as zip:
        member = next((k for k in zip.namelist()
                       if k.endswith(provenance["original_member"])),None)
        if member is None:
            raise AssertionError("Trained checkpoint member not found")
        raw=zip.read(member)
        if hashlib.sha256(raw).hexdigest() != provenance["member_sha256"]:
            raise AssertionError("Trained checkpoint raw bytes differ")
        checkpoint=json.loads(raw)
        if max_abs(checkpoint["raw_logits"],models["trained_shared"]["parameters"]["logits"])>1e-14:
            raise AssertionError("Trained checkpoint logits differ from E08")
        if (checkpoint["steps"]!=50 or checkpoint["objective"]!="margin_JSD"
            or checkpoint["tau"]!=TAU):
            raise AssertionError("Trained historical protocol differs")
    if not np.array_equal(np.asarray(models["uniform"]["parameters"]["gamma"]),
                          np.asarray(models["trained_shared"]["parameters"]["gamma"])):
        raise AssertionError("Kernel gamma differs between two policies")

    return {
        "source_checks": {
            "npz_crc32_pass": True, "step_array_key_count":len(required),
            "step_array_shapes_correct":True,"float64":True,
            "label_source":"code/experiments/target_center_guidance_trajectory/results/representations_uniform.json",
            "labels_fingerprint_matched_E08":True,
            "data_generator_called":False,
            "historical_Test_accessed":False,
            "source_reference_ids_sha256":array_fingerprint(id_order),
            "source_raw_steps_sha256":digest(npz),
            "source_provenance_sha256":digest(E08/"results"/"provenance.json"),
            "source_E04_metrics_sha256":digest(E04/"results"/"metrics_full.json"),
            "source_archived_checkpoint_sha256":digest(ARCHIVE),
            "source_checkpoint_member_sha256":provenance["member_sha256"],
            "source_labels_sha256":digest(LABELS),
            "source_reporter_sha256":digest(HERE.parent/"benchmark_v1"/"report_snapshot.py"),
            "source_engine_sha256":digest(HERE.parent/"benchmark_v1"/"engine.py"),
            "source_e08_recursion_sha256":digest(E08/"walk_analysis.py"),
            "policy_logits_sha256":{p:array_fingerprint(np.asarray(models[p]["parameters"]["logits"],
                                                   dtype=np.float64)) for p in POLICIES},
            "transitions_independent_of_true_H":(
                "Verified by E08 forward_steps and Benchmark engine source: "
                "H only controls geometric coefficients/loop extent; masked transition "
                "kernels and fixed logits do not depend on H."
            ),
            "train_leave_origin":True, "validation_out_of_graph":True,
        },
        "e08":metric_e08,"e04":e04,"labels_train":y_train,"labels_val":y_val,
        "steps":step_arrays,"snapshots":stored, "models":models,
    }


def validate_reports_and_originals(case, inventory, comparison):
    name=case["policy"]
    h=case["H"]
    if h==12:
        old=inventory["e08"]["policies"][name]["12"]
        for split in ("train","validation"):
            comparison["E08_H12_metric_max_abs_diff"][f"{name}_{split}"] = (
                compare_scalar_report(case[split],old[split],
                    f"E08 H12 {name} {split}"))
        comparison["E08_H12_center_max_abs_diff"][name] = max_abs(
            case["centers"],inventory["snapshots"][name]["centers"])
        if comparison["E08_H12_center_max_abs_diff"][name] > EPS:
            raise AssertionError("E08 H12 centers do not match")
    if name=="uniform" and h in (1,2,3,4,12):
        candidates=[r for r in inventory["e04"]["runs"]
                    if r["setting"]["H"]==h and
                       r["setting"]["beta"]==BETA and
                       r["setting"]["lam"]==LAM]
        if len(candidates)!=1:
            raise AssertionError(f"Expected one E04 Uniform true H={h}, found {len(candidates)}")
        old=candidates[0]
        comparison["E04_uniform_true_h_max_abs_diff"][str(h)]={
            split:compare_scalar_report(case[split],old[split],
                                        f"E04 uniform h={h} split={split}")
            for split in ("train","validation")
        }
        if max_abs(case["centers"],old["centers"])>EPS:
            raise AssertionError(f"E04 uniform center mismatch at h={h}")


def run():
    start=time.perf_counter()
    if (OUT / "metrics_full.json").exists():
        raise FileExistsError("E09 full metrics already stored: verify, do not rerun")
    inv=source_inventory()  # Fail closed BEFORE computing/publishing results.
    comparison={
        "E08_H12_metric_max_abs_diff":{},
        "E08_H12_center_max_abs_diff":{},
        "E04_uniform_true_h_max_abs_diff":{},
        "true_H_vs_E08_fixed_H12_truncation":{},
    }
    alphas={}
    for h in HORIZONS:
        p,a=coefficients(h)
        b=benchmark_coefficients(BETA,LAM,h).detach().numpy()
        if max_abs(a,b)>1e-13:
            raise AssertionError(f"Benchmark coefficient mismatch h={h}")
        if h in (2,3) and max_abs(a,independent_small_h(h))>1e-13:
            raise AssertionError(f"Independent length-sum coefficient mismatch h={h}")
        alphas[str(h)]={
            "H":h, "length_pmf":p.tolist(), "alpha_true_H":a.tolist(),
            "expected_realized_walk_length":float(p@np.arange(1,h+1)),
            "expected_weighted_step_index":float(a@np.arange(1,h+1)),
            "normalization_abs_error":float(abs(a.sum()-1)),
            "max_abs_error_vs_benchmark_coefficient":max_abs(a,b),
        }
    original_alpha=coefficients(12)[1]
    derived_cases=[]
    computed={}
    max_simplex_error=0.
    for policy in POLICIES:
        train_steps=inv["steps"][f"{policy}_train_steps"]
        val_steps=inv["steps"][f"{policy}_validation_steps"]
        per_h={}
        for h in HORIZONS:
            tic=time.perf_counter()
            a=np.asarray(alphas[str(h)]["alpha_true_H"])
            train=collapse(train_steps,a)
            validation=collapse(val_steps,a)
            max_simplex_error=max(
                max_simplex_error, full_array_check(f"{policy}/{h}/train",train,(300,3)),
                full_array_check(f"{policy}/{h}/validation",validation,(90,3))
            )
            if h==1 and (max_abs(train,train_steps[0])>1e-13 or
                         max_abs(validation,val_steps[0])>1e-13):
                raise AssertionError("H=1 is not the first step")
            centers=class_centers(train,inv["labels_train"])
            full_array_check(f"{policy}/{h}/centers",centers,(3,3))
            stats_tr=standard_metrics(by_hypothesis(train),centers,inv["labels_train"],
                                      TAU,split="train",per_sample=True)
            stats_val=standard_metrics(by_hypothesis(validation),centers,inv["labels_val"],
                                       TAU,split="validation",per_sample=True)
            for split, stats in (("train",stats_tr),("validation",stats_val)):
                counts=np.asarray(stats["counts"])
                conf=np.asarray(stats["confusion_true_rows_pred_cols"])
                if not np.array_equal(conf.sum(axis=1),counts):
                    raise AssertionError("Confusion counts mismatch")
                expected_BA=float(np.mean(np.diag(conf)/counts))
                if abs(expected_BA-stats["balanced_accuracy"])>1e-13:
                    raise AssertionError("BA does not match confusion")
                stats["per_class_recall"]=(np.diag(conf)/counts).tolist()
            case={
                "policy":policy,"H":h,"beta":BETA,"lambda":LAM,
                "coefficients":alphas[str(h)]["alpha_true_H"],
                "centers":centers.tolist(),
                "train":stats_tr,"validation":stats_val,
                "offline_recombination_seconds":float(time.perf_counter()-tic),
                "walk_engine_execution_seconds":None,
                "walk_engine_not_invoked":True
            }
            if h==12:
                for split, values in (("train",train),("validation",validation)):
                    original=inv["steps"][f"{policy}_{split}_exact"]
                    if max_abs(values,original)>2e-12:
                        raise AssertionError(f"E08 original H12 mismatch {policy}/{split}")
            validate_reports_and_originals(case,inv,comparison)
            per_h[h]=(train,validation,centers,case)
            derived_cases.append(case)
        baseline=per_h[12][3]
        for h in HORIZONS:
            train,validation,centers,case=per_h[h]
            for split in ("train","validation"):
                current=case[split]["per_sample"]["predicted"]
                orig=baseline[split]["per_sample"]["predicted"]
                changes=np.flatnonzero(np.asarray(current)!=np.asarray(orig))
                case[split]["prediction_disagreements_vs_true_H12"]=int(len(changes))
                case[split]["prediction_disagreement_indices_vs_true_H12"]=changes.tolist()
            # A DIFFERENT model: retaining h steps and reweighting original H=12 coefficients.
            weight=original_alpha[:h]/original_alpha[:h].sum()
            t12=collapse(train_steps,weight)
            v12=collapse(val_steps,weight)
            ct12=class_centers(t12,inv["labels_train"])
            rt12=standard_metrics(by_hypothesis(t12),ct12,inv["labels_train"],TAU,
                                   split="train",per_sample=False)
            rv12=standard_metrics(by_hypothesis(v12),ct12,inv["labels_val"],TAU,
                                   split="validation",per_sample=False)
            diff={
                "H_true":h,"E08_fixed_H12_r":h,
                "coefficient_l1":float(np.abs(weight-np.asarray(alphas[str(h)]["alpha_true_H"])).sum()),
                "train_mean_TV":float((.5*np.abs(train-t12).sum(-1)).mean()),
                "validation_mean_TV":float((.5*np.abs(validation-v12).sum(-1)).mean()),
                "centers_max_TV":float((.5*np.abs(centers-ct12).sum(-1)).max()),
                "train_BA_true_H":case["train"]["balanced_accuracy"],
                "train_BA_fixed_H12_truncation":rt12["balanced_accuracy"],
                "validation_BA_true_H":case["validation"]["balanced_accuracy"],
                "validation_BA_fixed_H12_truncation":rv12["balanced_accuracy"],
                "train_I_true_H":case["train"]["I"],
                "train_I_fixed_H12_truncation":rt12["I"],
                "method_distinction":"E09 recalculate capped-geometric pmf at H=h; E08 reweight the truncated alpha(H=12)"
            }
            if h in (2,3,4,8,12):
                original=inv["e08"]["policies"][policy][str(h)]
                for split, stats in (("train",rt12),("validation",rv12)):
                    compare_scalar_report(stats,
                                          inv["e08"]["policies"][policy][str(h)][split],
                                          f"E08 original r={h} {policy} {split}")
            comparison["true_H_vs_E08_fixed_H12_truncation"][f"{policy}_H{h}"]=diff
        computed[policy]=per_h

    # Persist results only AFTER every source, coefficient, E04/E08, metric and BA check passes.
    OUT.mkdir(parents=True,exist_ok=True)
    coverage={
        "schema":"e-ch4-09-coverage-v1","policies":list(POLICIES),
        "real_horizons":list(HORIZONS),"splits":["train","validation"],
        "common_metric_blocks":len(derived_cases)*2,
        "source_n_train":300,"source_n_validation":90,
        "all_required_metric_groups_present":True,
        "historical_test_accessed":False,
        "data_generated":False,"trained_policy_updated":False,
        "monte_carlo_run":False,"original_E08_engine_rerun":False,
    }
    source=dict(inv["source_checks"])
    source.update({
        "schema":"e-ch4-09-provenance-v1",
        "source_base_branch_commit":"a49ea148cbe68d12d21b40c0c914f2e258928c8d",
        "E08_checkpoint_family":"shared",
        "E08_checkpoint_objective":"margin_JSD",
        "E08_checkpoint_tau":TAU,
        "E08_historical_training_steps":50,
        "E08_data_seed":20261102,
        "reference_sample_order":read_json(E08/"results"/"provenance.json")["source_reference_ids"],
        "mask_and_leave_origin_contract":"fixed by E08 saved per-hop distributions; no new transitions built",
        "validation_labels_used_only_for_metrics":True,
        "all_control_checks_passed":True,
        "max_probability_simplex_error":max_simplex_error,
        "validation_label_class_counts":[13,49,28],
        "train_label_class_counts":[72,147,81],
        "offline_total_seconds":float(time.perf_counter()-start),
        "walk_engine_runtime_not_measured":True,
        "created_utc":datetime.now(timezone.utc).isoformat(),
    })
    contents={
        "metrics_full.json":{
            "schema":"e-ch4-09-true-horizon-standard-metrics-v1",
            "common_reporter":"code/experiments/benchmark_v1/report_snapshot.py::standard_metrics",
            "case_count":len(derived_cases),
            "metric_tau_report":TAU,
            "depth_interpretation":"actual capped geometric H per case, not E08 fixed-H truncation",
            "cases":derived_cases,
            "historical_test_accessed":False
        },
        "coefficients.json":{"schema":"e-ch4-09-true-horizon-coefficients-v1",
                             "beta":BETA,"lambda":LAM,"by_H":alphas},
        "comparisons.json":{"schema":"e-ch4-09-independent-controls-v1",**comparison},
        "coverage.json":coverage,
        "provenance.json":source,
    }
    for fname, value in contents.items():
        dump(OUT/fname,value)
    fields=["policy","H","split","BA","accuracy","R_KL","R_JS","R_sqrt_JS",
            "T","BT","TNL","BTNL","I","min_center_sqrt_JS",
            "margin_median","JS_similarity_NLL","NLL_KL_simple",
            "NLL_KL_normalized_historical","ties","differences_vs_H12",
            "offline_recombination_seconds"]
    with (OUT/"metrics_summary.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for case in derived_cases:
            for split in ("train","validation"):
                s=case[split]
                writer.writerow({
                    "policy":case["policy"],"H":case["H"],"split":split,
                    "BA":s["balanced_accuracy"],"accuracy":s["ordinary_accuracy"],
                    "R_KL":s["R_KL"],"R_JS":s["R_JS"],"R_sqrt_JS":s["R_sqrt_JS"],
                    "T":s["T"],"BT":s["BT"],"TNL":s["TNL"],"BTNL":s["BTNL"],
                    "I":s["I"],"min_center_sqrt_JS":s["min_pairwise_sqrt_JS"],
                    "margin_median":s["margin_quantiles"]["0.5"],
                    "JS_similarity_NLL":s["balanced_JS_similarity_NLL"],
                    "NLL_KL_simple":s["NLL_KL_simple"],
                    "NLL_KL_normalized_historical":s["NLL_KL_normalized_historical"],
                    "ties":s["tie_count"],
                    "differences_vs_H12":s["prediction_disagreements_vs_true_H12"],
                    "offline_recombination_seconds":case["offline_recombination_seconds"],
                })
    # Separate derived reports/print assets; raw E08 files stay untouched.
    from presentation import render_all
    render_all(OUT/"metrics_full.json",OUT/"comparisons.json",OUT/"coefficients.json")
    print("E09_SUCCESS",json.dumps({
        "cases":len(derived_cases),
        "all_controls":True,
        "source":source["source_raw_steps_sha256"],
        "max_simplex_error":max_simplex_error,
        "E04_max_metric_abs_error":max(
            (max(v.values()) for v in comparison["E04_uniform_true_h_max_abs_diff"].values()),default=0),
        "summary":[{"policy":c["policy"],"H":c["H"],
                    "train_BA":c["train"]["balanced_accuracy"],
                    "val_BA":c["validation"]["balanced_accuracy"],
                    "I":c["train"]["I"],"R_JS":c["train"]["R_JS"],
                    "validation_diff_preds":c["validation"]["prediction_disagreements_vs_true_H12"]}
                   for c in derived_cases],
        "offline_s":source["offline_total_seconds"]},ensure_ascii=False))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--execute",action="store_true")
    args=a.parse_args()
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","no_model_training":True,
                          "no_data_regeneration":True,"no_Test":True,
                          "no_E08_rerun":True,"policies":POLICIES,
                          "real_H":HORIZONS},ensure_ascii=False))
        return
    run()


if __name__=="__main__":
    main()
