"""E09 conditional signed margins from stored common-reporter scores only.

No walks, training, Monte Carlo, new data or historical Test.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"results"/"metrics_full.json"
OUTPUT=ROOT/"results"/"margin_analysis.json"
CSV=ROOT/"results"/"margin_analysis.csv"
TABLE=ROOT/"tables"/"error_margins_by_depth.tex"
REPORT=ROOT/"README_FA.md"
TITLE="## حاشیه مشروط به خطای طبقه‌بندی"
POLICIES=("uniform","trained_shared")
DEPTHS=(1,2,3,4,8,12)


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def avg(a):
    return float(a.mean()) if len(a) else None


def quantiles(a):
    return None if not len(a) else {
        str(q):float(np.quantile(a,q)) for q in (0.,.25,.5,.75,1.)
    }


def measure(case,split):
    v=case[split]
    p=v["per_sample"]
    y=np.asarray(p["labels"],dtype=np.int64)
    predictions=np.asarray(p["predicted"],dtype=np.int64)
    margins=np.asarray(p["margins"],dtype=float)
    scores=np.asarray(p["scores_all_candidates"],dtype=float)
    n=300 if split=="train" else 90
    if (y.shape!=(n,) or predictions.shape!=(n,) or
        margins.shape!=(n,) or scores.shape!=(n,3)):
        raise AssertionError("Invalid saved per-sample arrays")
    if not np.all(np.isfinite(margins)) or not np.all(np.isfinite(scores)):
        raise AssertionError("Invalid nonfinite margin")
    rivals=scores.copy()
    rivals[np.arange(n),y]=-np.inf
    if np.max(np.abs(margins-(scores[np.arange(n),y]-rivals.max(1))))>1e-12:
        raise AssertionError("Signed margin inconsistent with shared reporter scores")
    if not np.array_equal(scores.argmax(axis=1),predictions):
        raise AssertionError("Saved predictions inconsistent with scores")
    wrong=predictions!=y
    confusion=np.asarray(v["confusion_true_rows_pred_cols"],dtype=np.int64)
    if int(wrong.sum())!=n-int(np.trace(confusion)):
        raise AssertionError("Wrong count does not match confusion")
    if v["tie_count"] or np.asarray(p["ties"]).any():
        raise AssertionError("Unexpected tie: needs special signed-margin handling")
    if np.any(margins[wrong]>=0) or np.any(margins[~wrong]<=0):
        raise AssertionError("Signed margin does not separate right/wrong")
    class_rows=[]
    nonempty=[]
    for c in range(3):
        cm=y==c
        bad=cm&wrong
        good=cm&~wrong
        value=avg(margins[bad])
        if value is not None:
            nonempty.append(value)
        class_rows.append({
            "class":c,"n":int(cm.sum()),"errors":int(bad.sum()),
            "mean_error_margin":value,
            "mean_correct_margin":avg(margins[good]),
            "error_margin_quantiles":quantiles(margins[bad])
        })
    return {
        "policy":case["policy"],"H":case["H"],"split":split,
        "n":n,"error_count":int(wrong.sum()),"correct_count":int((~wrong).sum()),
        "mean_error_margin":avg(margins[wrong]),
        "mean_correct_margin":avg(margins[~wrong]),
        "mean_all_margin":float(margins.mean()),
        "mean_error_severity":avg(-margins[wrong]),
        "class_balanced_conditional_error_margin":avg(np.asarray(nonempty)),
        "classes_with_errors":len(nonempty),
        "error_margin_quantiles":quantiles(margins[wrong]),
        "correct_margin_quantiles":quantiles(margins[~wrong]),
        "per_class":class_rows,
        "sigmoid_loss_balanced":v["balanced_sigmoid_margin_loss"],
        "hard_margin_error_balanced":v["hard_margin_error_conservative"],
        "tie_count":v["tie_count"]
    }


def calculate():
    d=read(SOURCE)
    if d["case_count"]!=12 or d["historical_test_accessed"]:
        raise AssertionError("Unexpected common-reporter source coverage")
    cases={(c["policy"],c["H"]):c for c in d["cases"]}
    if set(cases)!={(p,h) for p in POLICIES for h in DEPTHS}:
        raise AssertionError("Missing or repeated policy/depth combination")
    rows=[measure(cases[(p,h)],split)
          for p in POLICIES for h in DEPTHS for split in ("train","validation")]
    return {
        "schema":"e09-conditional-error-margin-v1",
        "source":"results/metrics_full.json",
        "margin_definition":"true-class JS similarity - strongest rival JS similarity",
        "misclassified_condition":"predicted != true label",
        "mean_empty_error_subset":"null, not zero",
        "balanced_conditional_mean":"mean over only classes with >=1 error",
        "rows":rows,
        "historical_test_accessed":False,
        "walk_engine_rerun":False,
        "training_performed":False
    }


def fmt(x):
    return "---" if x is None else f"{x:.5f}"


def table(data):
    v={(x["policy"],x["H"],x["split"]):x for x in data["rows"]}
    lines=[
        "% E09 error margins from archived per-sample common metrics",
        r"\begin{table}[htbp]",r"\centering\scriptsize",
        r"\caption{میانگین حاشیه مشروط به خطا و درستی؛ حاشیه منفی‌تر برای خطا، شدت بیشتر اشتباه را نشان می‌دهد.}",
        r"\label{tab:ch4-e09-conditional-margin}",
        r"\begin{tabular}{rcrrrrrr}\hline",
        r"سیاست & $H$ & خطای آموزش & میانگین حاشیه خطا آموزش & میانگین حاشیه درست آموزش & خطای اعتبارسنجی & میانگین حاشیه خطا اعتبارسنجی & میانگین حاشیه درست اعتبارسنجی\\\hline"
    ]
    for p in POLICIES:
        for h in DEPTHS:
            t=v[(p,h,"train")]
            a=v[(p,h,"validation")]
            name="یکنواخت" if p=="uniform" else "آموزش‌دیده"
            lines.append(
                f"{name} & {h} & {t['error_count']} & {fmt(t['mean_error_margin'])} & "
                f"{fmt(t['mean_correct_margin'])} & {a['error_count']} & "
                f"{fmt(a['mean_error_margin'])} & {fmt(a['mean_correct_margin'])}"+r"\\"
            )
        lines.append(r"\hline")
    return "\n".join(lines+[r"\end{tabular}",r"\end{table}",""])


def markdown(data):
    out=[TITLE,"",
         "حاشیه امتیاز برچسب واقعی منهای قوی‌ترین رقیب است. "
         "بنابراین برای نمونه اشتباه‌طبقه‌بندی‌شده حاشیه منفی است و "
         "منفی‌ترشدن میانگین آن به معنی شدت بیشتر خطاست. "
         "حاشیه‌ها از گزارشگر مشترک خوانده شدند؛ گشت جدیدی اجرا نشده است.","",
         "| سیاست | H | خطای Train | میانگین حاشیه خطا Train | "
         "خطای Validation | میانگین حاشیه خطا Validation | "
         "میانگین حاشیه صحیح Validation | زیان سیگموئیدی Train / Validation |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    d={(x["policy"],x["H"],x["split"]):x for x in data["rows"]}
    for p in POLICIES:
        for h in DEPTHS:
            t=d[(p,h,"train")]
            v=d[(p,h,"validation")]
            out.append(
                f"| {p} | {h} | {t['error_count']} | {fmt(t['mean_error_margin'])} | "
                f"{v['error_count']} | {fmt(v['mean_error_margin'])} | "
                f"{fmt(v['mean_correct_margin'])} | "
                f"{t['sigmoid_loss_balanced']:.6f} / {v['sigmoid_loss_balanced']:.6f} |"
            )
    out+=["",
          "منبع کامل شامل میانگین و کوانتیل‌های حاشیه خطا و نمونه‌های درست، "
          "آمار هر کلاس، میانگین مشروط کلاس‌متوازن فقط بین کلاس‌های دارای خطا، "
          "نرخ خطا و زیان سیگموئیدی است. نبود نمونه خطا با null ثبت می‌شود، نه صفر.",
          "",
          "میانگین خطا در Validation با چند نمونه محدود محاسبه می‌شود و "
          "نباید بدون توجه به شمار خطا یا زیان حاشیه کل داده تفسیر شود.",
          "",
          "فایل‌ها: results/margin_analysis.json، results/margin_analysis.csv، "
          "tables/error_margins_by_depth.tex.",""]
    return "\n".join(out)


def csv_export(data):
    keys=("policy","H","split","n","error_count","correct_count",
          "mean_error_margin","mean_correct_margin","mean_all_margin",
          "mean_error_severity","class_balanced_conditional_error_margin",
          "classes_with_errors","sigmoid_loss_balanced","hard_margin_error_balanced",
          "tie_count")
    output=io.StringIO()
    w=csv.DictWriter(output,fieldnames=keys)
    w.writeheader()
    for r in data["rows"]:
        w.writerow({k:r[k] for k in keys})
    return output.getvalue()


def verify():
    if read(OUTPUT)!=calculate():
        raise AssertionError("Saved error margins are inconsistent with source")
    if not CSV.is_file() or not TABLE.is_file():
        raise AssertionError("Missing derived table or CSV")
    if TABLE.read_text(encoding="utf8").count(r"\label{tab:ch4-e09-conditional-margin}")!=1:
        raise AssertionError("TeX table label absent/duplicated")
    if TITLE not in REPORT.read_text(encoding="utf8"):
        raise AssertionError("Main report missing conditional margins")
    print("E09_MARGIN_PASS: 24 Train/Validation conditional error margins validated.")
    print("E09_MARGIN_ROWS",json.dumps([
        {"p":r["policy"],"H":r["H"],"split":r["split"],
         "n_error":r["error_count"],"m_error":r["mean_error_margin"],
         "m_correct":r["mean_correct_margin"]}
        for r in read(OUTPUT)["rows"]],ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write",action="store_true")
    group.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.write:
        if OUTPUT.exists():
            raise FileExistsError("Margin evidence exists: verify; do not recompute")
        out=calculate()
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        TABLE.parent.mkdir(parents=True,exist_ok=True)
        report=REPORT.read_text(encoding="utf8")
        if TITLE in report:
            raise AssertionError("Refusing to duplicate report section")
        OUTPUT.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
        CSV.write_text(csv_export(out),encoding="utf8")
        TABLE.write_text(table(out),encoding="utf8")
        REPORT.write_text(report.rstrip()+"\n\n"+markdown(out),encoding="utf8")
    verify()


if __name__=="__main__":
    main()
