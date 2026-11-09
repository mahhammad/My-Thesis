#!/usr/bin/env python3
"""Reproduce retrospective sigmoid-margin diagnostics from versioned representations.

Run from anywhere:
    python code/experiments/sigmoid_margin_diagnostics/analyze.py

No random walk, optimizer, gradient training, refitting of centers, or oracle data.
Only standard-library Python is required.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "code/experiments/center_matching_comparison/results"
OUTPUT = Path(__file__).resolve().parent / "results"

SOURCES = (
    ("uniform", "یکنواخت", "class_independent", "representations_uniform.json"),
    ("guided_round0", "هدایت‌شده، دور صفر", "candidate_class_conditional", "representations_guided_round0.json"),
    ("guided_round4", "هدایت‌شده، دور چهار", "candidate_class_conditional", "representations_guided_round4.json"),
)
TAUS = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05)
GIT_SOURCE = "d49e031738c5f7eed1ecb6b3c7e99139e96fdf4a"


def jsd(p, q):
    if len(p) != len(q):
        raise ValueError("distribution dimension mismatch")
    out = 0.0
    for a, b in zip(p, q):
        m = (a + b) / 2
        if a > 0:
            out += 0.5 * a * math.log(a / m)
        if b > 0:
            out += 0.5 * b * math.log(b / m)
    return out


def sigmoid_loss(m, tau):
    if not tau > 0:
        raise ValueError("tau must be positive")
    if m >= 0:
        v = math.exp(-m / tau)
        return v / (1 + v)
    v = math.exp(m / tau)
    return 1 / (1 + v)


def quantile(values, q):
    a = sorted(values)
    j = (len(a) - 1) * q
    lo = int(math.floor(j))
    hi = int(math.ceil(j))
    return a[lo] + (j - lo) * (a[hi] - a[lo])


def analyze_case(case, display_name, protocol, filename):
    source = INPUT / filename
    d = json.loads(source.read_text(encoding="utf-8"))
    val, centers = d["validation"], d["centers"]
    labels = val["y"]
    k = len(centers)
    counts = [labels.count(c) for c in range(k)]
    if not labels or any(n <= 0 for n in counts):
        raise ValueError("empty class in validation set")
    if "P" in val:
        if protocol != "class_independent":
            raise ValueError("unexpected inference protocol")
    elif "candidate_P" in val:
        if protocol != "candidate_class_conditional":
            raise ValueError("unexpected inference protocol")
    else:
        raise ValueError("no validation representations found")

    per_sample = []
    for i, true in enumerate(labels):
        scores = []
        for candidate in range(k):
            p = val["P"][i] if protocol == "class_independent" else val["candidate_P"][i][candidate]
            scores.append(1 - jsd(p, centers[candidate]) / math.log(2))
        rival = max(score for c, score in enumerate(scores) if c != true)
        per_sample.append({
            "index": i,
            "true_class": true,
            "scores_by_candidate_class": scores,
            "margin": scores[true] - rival,
            "balanced_weight": 1 / (k * counts[true]),
        })

    def weighted(f):
        return sum(row["balanced_weight"] * f(row) for row in per_sample)

    accuracy = 1 - weighted(lambda row: float(row["margin"] <= 0))
    metrics = []
    for tau in TAUS:
        groups = {name: {"mass": 0.0, "sensitivity": 0.0} for name in
                  ("near", "far_wrong", "far_correct", "mid_wrong", "mid_correct")}
        total_loss = total_relative = wrong_relative = 0.0
        for row in per_sample:
            margin, w = row["margin"], row["balanced_weight"]
            loss = sigmoid_loss(margin, tau)
            relative = 4 * loss * (1 - loss)
            total_loss += w * loss
            total_relative += w * relative
            wrong_relative += (w * relative if margin < 0 else 0.0)
            if abs(margin) <= tau:
                group = "near"
            elif margin < -2 * tau:
                group = "far_wrong"
            elif margin > 2 * tau:
                group = "far_correct"
            elif margin < 0:
                group = "mid_wrong"
            else:
                group = "mid_correct"
            groups[group]["mass"] += w
            groups[group]["sensitivity"] += w * relative
        for group in groups.values():
            group["sensitivity_share"] = group["sensitivity"] / total_relative
        assert abs(sum(g["mass"] for g in groups.values()) - 1) < 1e-10
        assert abs(sum(g["sensitivity_share"] for g in groups.values()) - 1) < 1e-10
        metrics.append({
            "tau": tau,
            "balanced_sigmoid_loss": total_loss,
            "mean_relative_sensitivity": total_relative,
            "wrong_sensitivity_share": wrong_relative / total_relative,
            "mass_abs_margin_le_tau": groups["near"]["mass"],
            "mass_margin_lt_minus_2tau": groups["far_wrong"]["mass"],
            "mass_margin_gt_2tau": groups["far_correct"]["mass"],
            "groups": groups,
        })
    margins = [row["margin"] for row in per_sample]
    return {
        "case": case,
        "display_name_fa": display_name,
        "inference_protocol": protocol,
        "source_file": "code/experiments/center_matching_comparison/results/" + filename,
        "n_validation": len(labels),
        "class_counts": counts,
        "balanced_accuracy": accuracy,
        "balanced_hard_error": 1 - accuracy,
        "min_margin": min(margins),
        "median_margin": quantile(margins, 0.5),
        "max_margin": max(margins),
        "balanced_mean_margin": weighted(lambda row: row["margin"]),
        "metrics_by_tau": metrics,
        "per_sample": per_sample,
    }


def analyze():
    return {
        "title": "Retrospective sigmoid decision-margin diagnostics",
        "date": "2026-11-07",
        "source_branch": "chapter4",
        "source_commit": GIT_SOURCE,
        "method": ("No new walks or policy fitting. Reuse exact saved 90-validation-sample "
                   "representations and realized training centers; class-balanced weights. "
                   "JS similarity and per-candidate-class inference as in stored experiment."),
        "formulae": {
            "js_similarity": "1 - JSD(P,V)/ln(2)",
            "margin": "s_true - max_{r != true}s_r",
            "sigmoid": "1/(1+exp(m/tau))",
            "relative_sensitivity": "4*sigmoid(m,tau)*(1-sigmoid(m,tau)) = 4*tau*abs(dloss/dm)",
            "weight": "1/(K*n_class(validation))",
        },
        "limitations": [
            "Selection and retrospective tau comparison on validation do not establish generalization.",
            "Uniform and guided runs have different inference protocols and are not controlled policy-only comparisons.",
            "This is score-margin sensitivity, not gradient with respect to feature-selection policy parameters.",
            "Large negative margins can occur for hard but potentially correctable samples; no oracle ambiguity assessment yet.",
            "The sigmoid is different from the preexisting softplus diagnostic; no policy training was performed.",
        ],
        "cases": [analyze_case(*s) for s in SOURCES],
    }


def save(result, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    (output / "sigmoid_validation_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "case", "tau", "balanced_accuracy", "balanced_hard_error",
        "balanced_sigmoid_loss", "mean_relative_sensitivity", "wrong_sensitivity_share",
        "mass_abs_margin_le_tau", "mass_margin_lt_minus_2tau",
        "mass_margin_gt_2tau", "near_sensitivity_share",
        "far_wrong_sensitivity_share", "far_correct_sensitivity_share",
    ]
    with (output / "sigmoid_validation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            for point in case["metrics_by_tau"]:
                writer.writerow({
                    "case": case["case"],
                    "tau": point["tau"],
                    "balanced_accuracy": case["balanced_accuracy"],
                    "balanced_hard_error": case["balanced_hard_error"],
                    **{name: point[name] for name in (
                        "balanced_sigmoid_loss", "mean_relative_sensitivity",
                        "wrong_sensitivity_share", "mass_abs_margin_le_tau",
                        "mass_margin_lt_minus_2tau", "mass_margin_gt_2tau"
                    )},
                    **{
                        key + "_sensitivity_share": point["groups"][key]["sensitivity_share"]
                        for key in ("near", "far_wrong", "far_correct")
                    },
                })
    return output


if __name__ == "__main__":
    report = analyze()
    destination = save(report)
    print("Saved 3 cases x 6 tau values (validation only) to:", destination)
    for c in report["cases"]:
        print(c["case"], "balanced accuracy =", round(c["balanced_accuracy"], 6))
