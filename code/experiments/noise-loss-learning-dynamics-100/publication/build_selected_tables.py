#!/usr/bin/env python3
"""Generate/check the three E14 Chapter4 LaTeX tables directly from frozen t100 CSV.

Read-only by default. No IMC fitting and no access to Test data.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_ROOT = ROOT / "feature-rank"
DEST = ROOT / "latex" / "tables"
FAMILIES = ("shared", "class")
LOSSES = ("sigmoid_JS_margin_tau005", "softplus_JS_tau005")
ROW_END = r"\\"

def read_csv(name):
    with (CSV_ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def heading(family, loss):
    return r"\lr{" + family + " / " + (
        "Sigmoid--JS" if loss == LOSSES[0] else "Softplus--JS") + "}"

def table(label, caption, shape, headings, rows):
    return "\n".join([
        "% AUTO-GENERATED; numerical entries verified against t100 CSV files.",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + shape + r"}\hline",
        headings + ROW_END + r"\hline",
        *[row + ROW_END for row in rows],
        r"\hline", r"\end{tabular}", r"\end{table}", ""
    ])

def read_aggregate_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def extended_tables():
    """Additional scientific tables: group means, not independent replicates."""
    metrics = read_aggregate_csv(ROOT / "all_metrics_101_steps.csv")
    policy = read_aggregate_csv(ROOT / "noise-policy" / "noise_policy_101_steps.csv")
    assert len(metrics) == 46056 and len(policy) == 29088
    m = {}
    for row in metrics:
        key = (row["family"], row["objective"], int(row["noise_count"]),
               int(row["step"]), row["split"], row["metric"])
        assert key not in m and int(row["subset_count"]) == 6
        m[key] = float(row["mean"])
    p = {}
    for row in policy:
        key = (row["family"], row["objective"], int(row["noise_count"]),
               int(row["step"]), row["metric"])
        assert key not in p and int(row["subset_count"]) == 6
        p[key] = float(row["mean"])

    def get(fam, loss, k, split, metric, step=100):
        return m[fam, loss, k, step, split, metric]

    def getp(fam, loss, k, metric, step=100):
        return p[fam, loss, k, step, metric]

    def decimal(value, digits=3):
        return "$" + format(value, "." + str(digits) + "f") + "$"

    perf, geom, exposure, normal = [], [], [], []
    for fam in FAMILIES:
        for loss in LOSSES:
            for k in (1, 3, 6):
                title = heading(fam, loss) + " & $" + str(k) + "$"
                ba_t = get(fam, loss, k, "train", "balanced_accuracy")
                ba_v = get(fam, loss, k, "validation", "balanced_accuracy")
                assert abs(ba_t - get(fam, loss, k, "train", "recall_macro")) < 1e-10
                assert abs(ba_v - get(fam, loss, k, "validation", "recall_macro")) < 1e-10
                perf.append(title + " & " + " & ".join(
                    decimal(100*get(fam, loss, k, split, metric), 2)
                    for split in ("train", "validation")
                    for metric in ("accuracy", "balanced_accuracy", "precision_macro")))
                geom.append(title + " & " + " & ".join(
                    decimal(get(fam, loss, k, "train", metric), 4)
                    for metric in ("separation_i", "density_r_kl", "density_r_js",
                                   "density_r_sqrt_js", "margin_mean")))
                exposure.append(title + " & " + " & ".join(
                    decimal(getp(fam, loss, k, metric), 4)
                    for metric in ("q1", "q2", "q3", "hit3", "nbar3")))
                assert abs(getp(fam, loss, k, "s3") +
                           getp(fam, loss, k, "sp3") - 1) < 1e-9
                normal.append(title + " & " + " & ".join(
                    decimal(getp(fam, loss, k, metric), 4)
                    for metric in ("s3", "sp3", "NFI3", "NSR3")))
    return {
      "performance_t100.tex": table(
          "tab:ch4-e14-perf-complete",
          "Accuracy و دقت متوازن (BA، معادل Recall کلان) و Precision کلان در آموزش و اعتبارسنجی؛ میانگین شش زیرمجموعهٔ وابسته در گام $100$، برحسب درصد.",
          "rccccccc",
          r"خانواده و زیان & $|A_N|$ & \lr{Tr Acc} & \lr{Tr BA/R} & \lr{Tr P} & \lr{Val Acc} & \lr{Val BA/R} & \lr{Val P}",
          perf),
      "geometry_t100.tex": table(
          "tab:ch4-e14-geom-complete",
          "هندسهٔ مراکز آموزش در گام $100$؛ جدایی $I$، تراکم‌های سه‌گانه و حاشیهٔ تصمیم؛ میانگین شش زیرمجموعهٔ وابسته.",
          "rcccccc",
          r"خانواده و زیان & $|A_N|$ & $I$ & $R_{\mathrm{KL}}$ & $R_{\mathrm{JS}}$ & $R_{\sqrt{\mathrm{JS}}}$ & \lr{Margin}",
          geom),
      "noise_exposure_t100.tex": table(
          "tab:ch4-e14-noise-exposure",
          "مواجههٔ سیاست کمکی بدون ماسک با نویز؛ احتمال انتخاب در سه نوبت، احتمال برخورد و شمار موردانتظار تا افق سه در گام $100$.",
          "rcccccc",
          r"خانواده و زیان & $|A_N|$ & $q_1$ & $q_2$ & $q_3$ & $p_{\mathrm{hit},N}^{(3)}$ & $\bar n_N^{(3)}$",
          exposure),
      "noise_normalized_t100.tex": table(
          "tab:ch4-e14-noise-normalized",
          "سهم نویز و غیرنویز و نسبت احتمال برخورد و سهم به سیاست یکنواخت هم‌بُعد، در افق سه و گام $100$.",
          "rccccc",
          r"خانواده و زیان & $|A_N|$ & $s_N^{(3)}$ & $s_P^{(3)}$ & $\mathrm{NFI}_N^{(3)}$ & $\mathrm{NSR}_N^{(3)}$",
          normal),
    }


def render_tables():
    runs = read_csv("final_run_overview_t100.csv")
    assert len(runs) == 152
    assert len({r["run_id"] for r in runs}) == 152
    groups = defaultdict(list)
    for r in runs:
        assert r["family"] in FAMILIES and r["objective"] in LOSSES
        k = int(r["noise_count"])
        assert 0 <= k <= 7
        groups[(r["family"], r["objective"], k)].append(r)
    for family in FAMILIES:
        for loss in LOSSES:
            assert len(groups[family, loss, 0]) == 1
            assert len(groups[family, loss, 7]) == 1
            for k in range(1, 7):
                g = groups[family, loss, k]
                assert len(g) == 6
                assert len({r["dataset_id"] for r in g}) == 6

    shares = defaultdict(list)
    records = read_csv("final_per_feature_macro_t100.csv")
    for record in records:
        shares[record["run_id"]].append(record)
    assert len(shares) == 152
    assert len(records) == sum(3 + int(r["noise_count"]) for r in runs)
    for run in runs:
        rr = shares[run["run_id"]]
        assert len(rr) == 3 + int(run["noise_count"])
        assert sum(r["feature_type"] == "noise" for r in rr) == int(run["noise_count"])
        assert abs(sum(float(r["share_s_a_3"]) for r in rr) - 1) < 1e-8

    zero, ba, noise = [], [], []
    for family in FAMILIES:
        for loss in LOSSES:
            h = heading(family, loss)
            z = groups[family, loss, 0][0]
            zero.append(h + " & $" + format(100*float(z["validation_BA"]), ".2f")
                        + "$ & $" + format(100*float(z["validation_accuracy"]), ".2f") + "$")
            values = []
            for k in range(1, 7):
                vals = [float(r["validation_BA"]) for r in groups[family, loss, k]]
                values.append("$" + format(100*sum(vals)/6, ".2f") + "$")
            ba.append(h + " & " + " & ".join(values))

            s = {}
            for k in (1, 3, 6):
                forrun = []
                for r in groups[family, loss, k]:
                    forrun.append(sum(float(q["share_s_a_3"])
                                      for q in shares[r["run_id"]]
                                      if q["feature_type"] == "noise"))
                s[k] = sum(forrun)/6
            values = ["$" + format(100*s[k], ".2f") + "$" for k in (1, 3, 6)]
            values += ["$" + format(s[k]/(k/(k+3)), ".3f") + "$" for k in (1, 3, 6)]
            noise.append(h + " & " + " & ".join(values))

    result = {
      "zero_noise_t100.tex": table(
        "tab:ch4-e14-zero",
        "مرجع تجربی بدون نویز در گام $100$؛ هر سطر تنها یک پیکربندی است و میانگین زیرمجموعه‌ها نیست. مقادیر بر حسب درصد.",
        "rcc",
        r"خانواده و زیان & \lr{Validation BA} & \lr{Validation Accuracy}",
        zero),
      "validation_ba_t100.tex": table(
        "tab:ch4-e14-ba",
        "میانگین دقت متوازن اعتبارسنجی در گام $100$ (درصد)، بر شش زیرمجموعهٔ وابسته در هر سطح نویز؛ تنها $|A_N|=1,\ldots,6$.",
        "rcccccc",
        r"خانواده و زیان & $|A_N|=1$ & $2$ & $3$ & $4$ & $5$ & $6$",
        ba),
      "noise_share_t100.tex": table(
        "tab:ch4-e14-noise-share",
        "میانگین سهم سه‌انتخابی نویز $s_N^{(3)}$ (درصد) و نسبت $\mathrm{NSR}_N^{(3)}$ در گام $100$؛ سیاست کمکی بدون ماسک.",
        "rcccccc",
        r"خانواده و زیان & $s_N^{(3)}(1)$ & $s_N^{(3)}(3)$ & $s_N^{(3)}(6)$ & $\mathrm{NSR}(1)$ & $\mathrm{NSR}(3)$ & $\mathrm{NSR}(6)$",
        noise)
    }
    result.update(extended_tables())
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Generate derived tables; no fitting")
    args = parser.parse_args()
    generated = render_tables()
    if args.write:
        DEST.mkdir(parents=True, exist_ok=True)
        for name, contents in generated.items():
            (DEST/name).write_text(contents, encoding="utf-8")
            print("WROTE", DEST/name)
    else:
        for name, contents in generated.items():
            p = DEST/name
            assert p.is_file() and p.read_text(encoding="utf-8") == contents, (
                "Incorrect or stale published table: " + str(p))
        print("E14_SELECTED_TABLES_PASS: 7 CSV-derived tables, NO FIT, NO TEST")

if __name__ == "__main__":
    main()
