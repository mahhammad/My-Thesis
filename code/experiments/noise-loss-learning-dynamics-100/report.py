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
    """Six-panel SVG chart using only standard-library rendering, no matplotlib."""
    import html
    width, height = 1320, 910
    colors = {"sigmoid_JS_margin_tau005": "#265ea7",
              "softplus_JS_tau005": "#db7837"}
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937} .grid{stroke:#e5e7eb;stroke-width:1}</style>',
        f'<text x="54" y="30" font-size="20" font-weight="bold">{html.escape(family)} | {html.escape(metric)}</text>',
        '<line x1="58" y1="49" x2="94" y2="49" stroke="#265ea7" stroke-width="3"/>',
        '<text x="101" y="54" font-size="13">Sigmoid-JS</text>',
        '<line x1="250" y1="49" x2="286" y2="49" stroke="#db7837" stroke-width="3"/>',
        '<text x="293" y="54" font-size="13">Softplus-JS</text>',
        '<text x="510" y="54" font-size="12">Six dependent subsets: faint individual curves, bold arithmetic mean</text>'
    ]
    factor = (100 if any(x in metric for x in
              ("accuracy", "precision", "recall")) else 1)
    for count in range(1, 7):
        row, col = divmod(count - 1, 3)
        px, py = 65 + col * 425, 85 + row * 405
        w, h = 365, 295
        group = complete[(family, count)]
        curves = []
        for objective in design["objectives"]:
            members = [rows for (obj, _), rows in group.items()
                       if obj == objective]
            if len(members) != 6:
                raise AssertionError("Missing six subset traces")
            for rows in members:
                v = [fetch(record, metric) for record in rows]
                if any(x is None for x in v):
                    raise ValueError(f"Undefined metric: {metric}")
                curves.append((objective, False, [factor * float(x) for x in v]))
            agg = aggregation(group, objective, metric)
            if agg is None:
                raise ValueError("Missing aggregate for chart")
            curves.append((objective, True, [factor * x[0] for x in agg]))
        all_y = [value for _, _, vals in curves for value in vals]
        low, high = min(all_y), max(all_y)
        if metric.endswith("balanced_accuracy"):
            low, high = 70., 100.
        elif any(x in metric for x in ("accuracy", "precision", "recall")):
            low, high = 0., 100.
        else:
            margin = max((high - low) * .08, max(abs(high), abs(low), 1e-8) * .025)
            low -= margin
            high += margin
        if high <= low:
            high = low + 1.
        items.append(f'<text x="{px+8}" y="{py+9}" font-size="17" font-weight="bold">|A_N| = {count}</text>')
        clip_id = f"panel-{family}-{count}-{metric.replace('.', '-')}"
        x0, y0 = px + 52, py + 28
        pw, ph = w - 64, h - 55
        items.append(f'<clipPath id="{html.escape(clip_id)}"><rect x="{x0}" y="{y0}" width="{pw}" height="{ph}"/></clipPath>')
        for tick in range(5):
            val = low + (high - low) * tick / 4
            yy = y0 + ph * (1 - tick/4)
            items.append(f'<line class="grid" x1="{x0}" y1="{yy:.2f}" x2="{x0+pw}" y2="{yy:.2f}"/>')
            items.append(f'<text x="{x0-8}" y="{yy+4:.2f}" font-size="10" text-anchor="end">{val:.3g}</text>')
        for step in (0, 50, 100):
            xx = x0 + pw * step / UPDATES
            items.append(f'<text x="{xx:.2f}" y="{y0+ph+17}" font-size="11" text-anchor="middle">{step}</text>')
        items.append(f'<g clip-path="url(#{html.escape(clip_id)})">')
        for objective, mean, values in sorted(curves, key=lambda v: v[1]):
            points = " ".join(f'{x0+pw*t/UPDATES:.2f},{y0+ph*(1-(v-low)/(high-low)):.2f}'
                              for t, v in enumerate(values))
            items.append(f'<polyline points="{points}" fill="none" stroke="{colors[objective]}" '
                         f'stroke-width="{2.5 if mean else .9}" stroke-opacity="{1 if mean else .22}"/>')
        items.append('</g>')
        items.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#4b5563" stroke-width="1"/>')
        items.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#4b5563" stroke-width="1"/>')
        items.append(f'<text x="{x0+pw/2}" y="{y0+ph+32}" font-size="11" text-anchor="middle">Adam updates</text>')
        if metric.endswith("balanced_accuracy") and (min(all_y) < low or max(all_y) > high):
            items.append(f'<text x="{x0}" y="{py+h+13}" fill="#9a3412" font-size="11">Note: off-scale values are clipped by fixed BA axis</text>')
    items.append("</svg>")
    dest = HERE / "figures" / (family + "__" + metric.replace(".", "_") + ".svg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf8") as f:
        f.write("\n".join(items) + "\n")
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
