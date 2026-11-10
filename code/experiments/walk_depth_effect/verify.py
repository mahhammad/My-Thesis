"""Read-only verification of persisted E09 true-H scientific evidence.

Loads only archived Train/Validation labels and E08 per-step data;
recomputes deterministic coefficients/reports without running walks.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from analyze import (
    HERE, HORIZONS, POLICIES, OUT, TAU, coefficients,
    source_inventory, class_centers, collapse, by_hypothesis,
    full_array_check, max_abs, compare_scalar_report,
    validate_reports_and_originals
)
from report_snapshot import standard_metrics
from presentation import load

REQUIRED_RESULTS = (
    "metrics_full.json", "coefficients.json", "coverage.json",
    "comparisons.json", "provenance.json", "metrics_summary.csv"
)
REQUIRED_PRINT = (
    "tables/main_depth_effect.tex",
    "tables/true_H_vs_E08_truncation.tex",
    "figures/validation_ba_vs_H.tex",
    "figures/center_information_vs_H.tex",
    "figures/dispersion_vs_H.tex",
    "REPORT_FA.md",
)


def audit():
    for name in REQUIRED_RESULTS:
        if not (OUT/name).is_file():
            raise AssertionError("Missing persisted E09 output: "+name)
    for name in REQUIRED_PRINT:
        if not (HERE/name).is_file():
            raise AssertionError("Missing generated E09 print artifact: "+name)
    metrics=load(OUT/"metrics_full.json")
    c=load(OUT/"coefficients.json")
    cov=load(OUT/"coverage.json")
    comp=load(OUT/"comparisons.json")
    provenance=load(OUT/"provenance.json")
    inv=source_inventory()
    if len(metrics["cases"])!=12 or cov["common_metric_blocks"]!=24:
        raise AssertionError("Missing policy/horizon/split combinations")
    if not cov["all_required_metric_groups_present"]:
        raise AssertionError("Incomplete common metric evidence")
    for key in ("historical_test_accessed","data_generated","trained_policy_updated",
                "monte_carlo_run","original_E08_engine_rerun"):
        if cov[key] or key=="historical_test_accessed" and metrics[key]:
            raise AssertionError("Forbidden action marked in evidence: "+key)
    if not provenance["labels_fingerprint_matched_E08"] or not provenance["all_control_checks_passed"]:
        raise AssertionError("Source validation did not pass")
    for key in ("source_raw_steps_sha256","source_labels_sha256",
                "source_provenance_sha256","source_archived_checkpoint_sha256"):
        if provenance[key]!=inv["source_checks"][key]:
            raise AssertionError("Source digest changed since E09 run: "+key)
    if len(comp["true_H_vs_E08_fixed_H12_truncation"])!=12:
        raise AssertionError("Missing true-H versus truncation controls")
    if set(c["by_H"])!={str(h) for h in HORIZONS}:
        raise AssertionError("Wrong coefficient coverage")
    pairs=set()
    for case in metrics["cases"]:
        name,h=case["policy"],case["H"]
        if name not in POLICIES or h not in HORIZONS or (name,h) in pairs:
            raise AssertionError("Duplicate/unknown E09 policy+horizon")
        pairs.add((name,h))
        pmf,alpha=coefficients(h)
        if max_abs(alpha,c["by_H"][str(h)]["alpha_true_H"])>1e-14:
            raise AssertionError("Saved coefficient not true-H")
        if max_abs(pmf,c["by_H"][str(h)]["length_pmf"])>1e-14:
            raise AssertionError("Saved capped geometric pmf not true-H")
        if max_abs(alpha,case["coefficients"])>1e-14:
            raise AssertionError("Case coefficients changed")
        t=collapse(inv["steps"][f"{name}_train_steps"],alpha)
        v=collapse(inv["steps"][f"{name}_validation_steps"],alpha)
        full_array_check(name+str(h)+"train",t,(300,3))
        full_array_check(name+str(h)+"validation",v,(90,3))
        centers=class_centers(t,inv["labels_train"])
        if max_abs(centers,case["centers"])>2e-12:
            raise AssertionError("Center not built on Train")
        for split,data,y in (("train",t,inv["labels_train"]),
                             ("validation",v,inv["labels_val"])):
            metric=standard_metrics(by_hypothesis(data),centers,y,TAU,
                                    split=split,per_sample=True)
            compare_scalar_report(metric,case[split],f"verify {name} h{h} {split}")
            if metric["per_sample"]["predicted"]!=case[split]["per_sample"]["predicted"]:
                raise AssertionError("Saved per-sample decisions inconsistent")
            if max_abs(metric["per_sample"]["margins"],
                       case[split]["per_sample"]["margins"])>2e-12:
                raise AssertionError("Saved margins inconsistent")
            if case[split]["per_class_recall"] != [
                    r["accuracy"] for r in metric["per_class"]]:
                raise AssertionError("Per-class recall inconsistent")
        validate_reports_and_originals(case,inv,{
            "E08_H12_metric_max_abs_diff":{},
            "E08_H12_center_max_abs_diff":{},
            "E04_uniform_true_h_max_abs_diff":{},
            "true_H_vs_E08_fixed_H12_truncation":{}
        })
    if pairs!={(p,h) for p in POLICIES for h in HORIZONS}:
        raise AssertionError("Incomplete 12 true-H cases")
    for path,label in (
        ("tables/main_depth_effect.tex","tab:ch4-e09-depth-main"),
        ("tables/true_H_vs_E08_truncation.tex","tab:ch4-e09-true-vs-truncated"),
        ("figures/validation_ba_vs_H.tex","fig:ch4-e09-validation-ba"),
        ("figures/center_information_vs_H.tex","fig:ch4-e09-I"),
        ("figures/dispersion_vs_H.tex","fig:ch4-e09-RJS")
    ):
        content=(HERE/path).read_text(encoding="utf8")
        if content.count("\\label{"+label+"}")!=1:
            raise AssertionError("Missing/multiple TeX label: "+label)
    print("E09_PASS: archived source CRC/digests, labels, coefficients, 12 true-H cases, "
          "24 common metric blocks, per-sample decisions, H12 E08 and E04 comparisons.")
    print("E09_SUMMARY",[(d["policy"],d["H"],
                          round(d["validation"]["balanced_accuracy"],7),
                          round(d["train"]["I"],7),
                          d["validation"]["prediction_disagreements_vs_true_H12"])
                         for d in metrics["cases"]])


if __name__=="__main__":
    audit()
