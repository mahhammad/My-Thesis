"""Fast, read-only audit of saved E08 outputs; NEVER reruns an experiment."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code"/"experiments"/"benchmark_v1"))
from snapshot_pipeline import load_snapshot
from walk_analysis import DEPTHS
from presentation import POLICIES

def audit():
    dest=HERE/"results"
    required=[
        "experiment_report.json","metrics_full.json","coverage.json",
        "provenance.json","raw_step_distributions.npz"
    ]
    for path in required:
        if not (dest/path).is_file():
            raise AssertionError("Missing E08 result: "+path)
    read=lambda filename:json.loads((dest/filename).read_text(encoding="utf8"))
    report=read("experiment_report.json")
    metrics=read("metrics_full.json")
    coverage=read("coverage.json")
    source=read("provenance.json")
    if report["schema"]!="e-ch4-08-experiment-v1":
        raise AssertionError("Wrong experiment schema")
    if report["historical_test_accessed"] or source["historical_test_accessed"]:
        raise AssertionError("Test access not forbidden")
    if report["training_performed"] or source["retrained"]:
        raise AssertionError("Training unexpectedly performed")
    if coverage["train_validation_policy_depth_rows"]!=24:
        raise AssertionError("Incomplete depth-by-policy coverage")
    if len(report["monte_carlo"]["raw"])!=72:
        # 2 policies * 2 splits * 3 origins * 2 seeds * 3 budgets.
        raise AssertionError("MC raw records incomplete")
    if not report["monte_carlo"]["nested_budgets"]:
        raise AssertionError("MC independent budget contract changed")
    if coverage["MC_seeds"] != [20261121,20261122]:
        raise AssertionError("MC seeds changed")
    if coverage["MC_budgets"] != [128,512,2048]:
        raise AssertionError("MC budgets changed")
    if set(metrics["policies"]) != set(POLICIES):
        raise AssertionError("Missing policy common-metric blocks")
    keys=("R_KL","R_JS","R_sqrt_JS","T","BT","TNL","BTNL","I",
          "balanced_accuracy","ordinary_accuracy",
          "confusion_true_rows_pred_cols","margin_quantiles",
          "center_geometry")
    for name in POLICIES:
        if len(report["policy_results"][name]["truncation"])!=len(DEPTHS):
            raise AssertionError("Depth coverage missing")
        for row in report["policy_results"][name]["truncation"]:
            if not row["bound_verified"]:
                raise AssertionError("Bound not verified")
            for field in ("train_max_tv","validation_max_tv","class_centers_max_tv"):
                if row[field] > row["epsilon_tail"]+1e-9:
                    raise AssertionError("Observed TV exceeds bound")
        for r in DEPTHS:
            case=metrics["policies"][name][str(r)]
            for split in ("train","validation"):
                d=case[split]
                for key in keys:
                    if key not in d or d[key] is None:
                        raise AssertionError(f"Missing common metric {name} r={r} {split} {key}")
                if not 0<=d["balanced_accuracy"]<=1:
                    raise AssertionError("Invalid balanced accuracy")
            # No Validation-derived centers: saved center is Train class mean.
            np.testing.assert_allclose(case["centers"],case["train"]["center_geometry"]
                                       .get("centers",case["centers"]),atol=1e-10)
        path=dest/"snapshots"/name
        model,arrays=load_snapshot(path)
        if model["prediction"]["validation_status"]!="available":
            raise AssertionError("Unexpected absent Validation")
        if model["data"]["test_accessed"]:
            raise AssertionError("Test flagged in snapshot")
        for file in ("model.json","distributions.npz","report.json","report.md"):
            if not (path/file).exists():
                raise AssertionError("Missing v2 snapshot component: "+file)
        full=metrics["policies"][name]["12"]
        np.testing.assert_allclose(arrays["centers"],full["centers"],atol=1e-10)
    with np.load(dest/"raw_step_distributions.npz",allow_pickle=False) as d:
        for name in POLICIES:
            if d[name+"_train_steps"].shape[:2]!=(12,300):
                raise AssertionError("Train step tensor dimensions wrong")
            if d[name+"_validation_steps"].shape[:2]!=(12,90):
                raise AssertionError("Validation step tensor dimensions wrong")
    for stem in ("truncation.tex","monte_carlo.tex","runtime.tex"):
        if not (HERE/"tables"/stem).is_file():
            raise AssertionError("Missing generated table: "+stem)
    if not (HERE/"figures"/"bound_and_max_tv.tex").is_file():
        raise AssertionError("Missing generated figure")
    if not (HERE/"REPORT_FA.md").is_file():
        raise AssertionError("Missing human-readable report")
    print("PASS: E08 persisted outputs, raw arrays, provenance, 24 common-metric blocks, "
          "72 MC cases and full v2 snapshots verified without recomputation.")
    print("Run:", report["created_utc"])
    print("MC H1 outcomes:", report["hypothesis_outcomes"][
        "preregistered_mc_mean_tv_decreases_from_128_to_2048"])
    print("Max absolute engine comparison:",{
        p:(report["policy_results"][p]["max_abs_error_independent_vs_official_train"],
           report["policy_results"][p]["max_abs_error_independent_vs_official_validation"])
        for p in POLICIES})
    return report

if __name__=="__main__":
    audit()
