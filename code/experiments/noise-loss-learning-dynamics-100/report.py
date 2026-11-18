"""Read-only, complete-trace verification, comparison tables and figures.

Only A_N sizes 1..6 enter comparisons; each point is an average of six paired,
dependent subsets. Both §4.13 objectives are drawn on each panel. No retraining.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from protocol import HERE, UPDATES, make_design, sha

TRAIN_KEYS = (
    "loss", "gradient_l2", "accuracy", "balanced_accuracy",
    "precision_macro", "recall_macro",
    "density_r_kl", "density_r_js", "density_r_sqrt_js",
    "separation_i", "separation_t", "separation_bt",
    "separation_tnl", "separation_btnl", "margin_mean"
)
VALIDATION_KEYS = ("accuracy", "balanced_accuracy",
                   "precision_macro", "recall_macro")
TRAIN_CLASS = ("precision_by_class", "recall_by_class")
VALUES = (["train." + x for x in TRAIN_KEYS]
          + ["validation." + x for x in VALIDATION_KEYS]
          + ["train." + k + f".{i}" for k in TRAIN_CLASS for i in range(3)]
          + ["validation." + k + f".{i}" for k in TRAIN_CLASS for i in range(3)]
          + ["optimizer.previous_update_l2"])


def fetch(row, field):
    value = row
    for item in field.split("."):
        value = value[int(item)] if isinstance(value, list) else value[item]
    return value


def check_record(row, run):
    if any(key in row for key in ("test", "test_results", "test_accessed")):
        raise ValueError("Test fields must not appear in trace")
    if row.get("run_id") != run["run_id"]:
        raise ValueError("Wrong run ID")
    if set(row["validation"]) != set(VALIDATION_KEYS + TRAIN_CLASS):
        raise ValueError("Validation can contain only accuracy, BA, precision and recall")
    n = 3 + run["noise_count"]
    expected = n * (n + 1) * (3 if run["family"] == "class" else 1)
    opt = row["optimizer"]
    for key in ("logits", "gradients", "adam_m", "adam_v"):
        arr = opt[key]
        if len(arr) != expected or any(not math.isfinite(float(x)) for x in arr):
            raise ValueError(f"{key} has wrong size/nonfinite entries")
    update = opt["adam_update_to_next"]
    if row["step"] == UPDATES:
        if update is not None:
            raise ValueError("Final step cannot have next update")
    elif (update is None or len(update) != expected or
          any(not math.isfinite(float(x)) for x in update)):
        raise ValueError("Missing/invalid next Adam update")
    if row["step"] == 0 and (any(x != 0 for x in opt["logits"])
                            or opt["previous_update_l2"] is not None):
        raise ValueError("Step 0 must be exact zero initialization")
    for key in ("loss", "gradient_l2", "accuracy", "balanced_accuracy",
                "precision_macro", "recall_macro", "density_r_js",
                "density_r_sqrt_js", "separation_i", "separation_t",
                "separation_bt", "margin_mean"):
        value = row["train"].get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Invalid Train metric {key}: {value}")
    for key in VALIDATION_KEYS:
        x = row["validation"][key]
        if not isinstance(x, (float, int)) or not 0 <= x <= 1:
            raise ValueError(f"Invalid Validation classification metric: {key}")
    for split in ("train", "validation"):
        for metric in TRAIN_CLASS:
            values = row[split][metric]
            if len(values) != 3 or any(not 0 <= v <= 1 for v in values):
                raise ValueError("Invalid three-class precision/recall")
    if len(row["policy"]["centers"]) != 3:
        raise ValueError("Missing three Train-derived centers")
    if len(row["policy"]["unmasked"]) != (3 if run["family"] == "class" else 1):
        raise ValueError("Unmasked policy hypotheses do not match family")


def read_trace(run, root):
    folder = Path(root) / run["run_id"]
    complete = folder / "complete.json"
    trace = folder / "trace.jsonl"
    if not complete.is_file() or not trace.is_file():
        raise FileNotFoundError(f"Missing completed trace: {run['run_id']}")
    proof = json.loads(complete.read_text("utf8"))
    if proof["run_id"] != run["run_id"] or proof["trace_sha256"] != sha(trace):
        raise ValueError(f"Failed immutable trace fingerprint: {run['run_id']}")
    with trace.open(encoding="utf8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if [r["step"] for r in rows] != list(range(UPDATES + 1)):
        raise ValueError(f"Trace must contain 0..{UPDATES}: {run['run_id']}")
    for row in rows:
        check_record(row, run)
    return rows


def gather(require_endpoints=True):
    design = make_design()
    root = HERE / "results"
    complete = defaultdict(dict)
    endpoints = []
    for run in design["runs"]:
        if run["noise_count"] in (0, 7):
            if require_endpoints:
                endpoints.append(run["run_id"])
            continue
        complete[(run["family"], run["noise_count"])][
            (run["objective"], run["dataset_id"])] = read_trace(run, root)
    for family in design["families"]:
        for count in range(1, 7):
            group = complete[(family, count)]
            if len(group) != 12:
                raise ValueError("Incomplete group: six subsets for each of two objectives")
    return design, complete, endpoints


def aggregation(group, objective, metric):
    """Mean and descriptive subset SD at each true Adam update."""
    members = [group[(objective, ds)] for obj, ds in group if obj == objective]
    if len(members) != 6:
        raise ValueError("Cannot average fewer than six complete subsets")
    sequences = []
    for rows in members:
        seq = [fetch(row, metric) for row in rows]
        if any(v is None or not math.isfinite(float(v)) for v in seq):
            return None
        sequences.append([float(v) for v in seq])
    return [[statistics.mean(x), statistics.stdev(x), min(x), max(x)]
            for x in zip(*sequences)]


def write_summary(design, complete):
    path = HERE / "results" / "summary.csv"
    fields = ("family", "noise_count", "objective", "step", "metric",
              "mean", "subset_sd", "minimum", "maximum", "subset_count")
    rows = []
    for (family, count), group in sorted(complete.items()):
        for objective in design["objectives"]:
            for metric in VALUES:
                aggregates = aggregation(group, objective, metric)
                if aggregates is None:
                    continue  # Missing/undefined metrics remain visibly absent.
                for step, item in enumerate(aggregates):
                    rows.append(dict(zip(fields, (family, count, objective,
                        step, metric, *item, 6))))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite previous derived summary")
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def chart_metric(design, complete, family, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True,
                               layout="constrained")
    colors = {"sigmoid_JS_margin_tau005": "#245b8c",
              "softplus_JS_tau005": "#c56c32"}
    for count, ax in enumerate(axes.flat, 1):
        group = complete[(family, count)]
        for objective in design["objectives"]:
            series = [rows for (name, ds), rows in group.items()
                      if name == objective]
            if len(series) != 6:
                raise AssertionError("Missing six traces in plot")
            factor = 100 if ("accuracy" in metric or "precision" in metric
                              or "recall" in metric) else 1
            for rows in series:
                y = [fetch(row, metric) for row in rows]
                if any(x is None for x in y):
                    raise ValueError(f"Undefined metric in {family}/{count}: {metric}")
                ax.plot(range(UPDATES + 1), [factor * x for x in y],
                        color=colors[objective], alpha=0.15, linewidth=0.8)
            aggregate = aggregation(group, objective, metric)
            if aggregate is None:
                raise ValueError("Metric is undefined; omit only after audited explanation")
            label = "Sigmoid-JS" if objective.startswith("sigmoid") else "Softplus-JS"
            ax.plot(range(UPDATES + 1),
                    [factor * x[0] for x in aggregate],
                    color=colors[objective], linewidth=2.0, label=label)
        ax.set(title=f"|A_N| = {count}", xlim=(0, UPDATES))
        ax.grid(alpha=0.18)
        if metric.endswith("balanced_accuracy"):
            ax.set_ylim(70, 100)
        elif "accuracy" in metric or "precision" in metric or "recall" in metric:
            ax.set_ylim(0, 100)
        ax.set_xlabel("Adam updates")
        ax.set_ylabel(metric.replace("_", " "))
    h, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(h, labels, loc="upper center", ncol=2)
    figure.suptitle(f"{family}: {metric} | 6 dependent subsets per noise level", y=1.02)
    dest = HERE / "figures" / (family + "__" + metric.replace(".", "_") + ".png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return dest


def write_report(design, grouped, summary_rows, figures):
    """Document actual outputs only; no causal-noise claims or inferred CIs."""
    path = HERE / "results" / "REPORT_FA.md"
    if path.exists():
        raise FileExistsError("Refusing to replace existing report")
    text = [
        "# نتیجهٔ محاسباتی تکمیلی §۴.۱۴ — ۱۰۰ آپدیت",
        "",
        "**دامنه:** دو زیان Sigmoid-JS و Softplus-JS، دو خانوادهٔ shared/class، "
        "شش زیرمجموعهٔ وابسته در هر سطح |A_N|=1,...,6. حالت‌های ۰ و ۷ "
        "از مقایسهٔ میانگین‌ها کنار گذاشته شده‌اند.",
        "",
        "این گزارش از ۱۴۴ مسیر دارای ۱۰۱ وضعیت واقعی تولید شده است. "
        "هیچ نتیجهٔ علّی دربارهٔ نقش نویز در هموارشدن مسیر ادعا نمی‌شود.",
        "Validation فقط شامل Accuracy، BA، Precision و Recall است. "
        "R و I فقط در Train گزارش می‌شوند.",
        "",
        "## شواهد و محل فایل‌ها",
        "",
        "- فایل summary.csv شامل میانگین، SD توصیفی، حداقل و حداکثر "
        "تمام معیارها در تک‌تک وضعیت‌های ۰ تا ۱۰۰ است.",
        "- هر trace.jsonl شامل logits، گرادیان، ممان‌های Adam، "
        "آپدیت واقعی، مراکز Train و سنجه‌های گام به گام است.",
        "- فایل complete.json شامل SHA-256 برای حفاظت از هر trace است.",
        "",
        "## شکل‌های استخراج‌شده",
        ""
    ]
    text.extend("- ![{}](../figures/{})".format(f.name, f.name) for f in figures)
    text.extend([
        "", "## محدودیت‌های استنباط", "",
        "شش زیرمجموعه از یک دادهٔ مادر مشترک به دست آمده‌اند و تکرارهای "
        "مستقل آماری نیستند؛ SD میان آن‌ها فاصلهٔ اطمینان نیست. "
        "این طراحی اثر تعداد نویز را از تغییر بُعد و ظرفیت سیاست "
        "به‌طور علّی تفکیک نمی‌کند. هیچ دادهٔ Test استفاده نشده است.",
        "",
    ])
    path.write_text("\n".join(text), encoding="utf8")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Verify complete data without writing figures")
    a = p.parse_args()
    design, grouped, endpoints = gather()
    print(f"Verified 144 comparison traces across {len(grouped)} family/noise groups")
    if a.check:
        return
    summary, rows = write_summary(design, grouped)
    figures = []
    for family in design["families"]:
        for metric in VALUES:
            if metric == "optimizer.previous_update_l2":
                # Step 0 is intrinsically undefined; cannot be averaged as a normal value.
                continue
            try:
                figures.append(chart_metric(design, grouped, family, metric))
            except ValueError as exc:
                print("UNDEFINED, omitted with explicit reason:", metric, str(exc))
    report = write_report(design, grouped, rows, figures)
    print("Wrote", summary, report, len(figures), "figures")


if __name__ == "__main__":
    main()
