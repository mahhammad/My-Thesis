"""E09 conditional margin: first reported percentile above observed error fraction.

Reads ONLY existing margin_analysis.json and margin_summary.json, which
are derived from the common-reporter's archived per-sample signed margins.
Never runs a walker, data generator, policy training or Test.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
MARGIN=HERE/"results"/"margin_analysis.json"
QUANT=HERE/"results"/"margin_summary.json"
OUTPUT=HERE/"results"/"error_quantile_boundary.json"
CSV=HERE/"results"/"error_quantile_boundary.csv"
TABLE=HERE/"tables"/"error_quantile_boundary.tex"
REPORT=HERE/"REPORT_FA.md"
HEADING="## اولین چندک بالاتر از نرخ خطا"
GRID=((0.,"min"),(.1,"p10"),(.25,"p25"),(.5,"median"),
      (.75,"p75"),(.9,"p90"),(1.,"max"))
POLICIES=("uniform","trained_shared")
DEPTHS=(1,2,3,4,8,12)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def convert():
    margins=read(MARGIN)
    quant=read(QUANT)
    if margins["schema"]!="e09-conditional-error-margin-v1":
        raise AssertionError("Missing common-reporter conditional margin source")
    if quant["schema"]!="e09-margin-from-common-metrics-v1":
        raise AssertionError("Missing common-reporter percentile source")
    if margins["historical_test_accessed"] or quant["Test_accessed"]:
        raise AssertionError("Historical Test access")
    mm={(r["policy"],r["H"],r["split"]):r for r in margins["rows"]}
    qq={(r["policy"],r["H"],r["split"]):r for r in quant["rows"]}
    expected={(p,h,split) for p in POLICIES for h in DEPTHS
              for split in ("train","validation")}
    if set(mm)!=expected or set(qq)!=expected:
        raise AssertionError("Missing policy-depth-split block")
    rows=[]
    for p in POLICIES:
        for h in DEPTHS:
            for split in ("train","validation"):
                r=mm[(p,h,split)]
                s=qq[(p,h,split)]
                n=r["n"]; k=r["error_count"]
                if n!=s["n"] or k<0 or k>n or r["tie_count"] or s["ties"]:
                    raise AssertionError("Bad sample count/tie contract")
                if s["nonpositive_count"]!=k:
                    raise AssertionError("Nonpositive margins do not match classifier errors")
                error_fraction=k/n
                after=next(((level,name) for level,name in GRID
                            if level>error_fraction),None)
                positive=next(((level,name) for level,name in GRID
                               if s["quantiles"][name]>0),None)
                first_correct=(r["correct_margin_quantiles"] or {}).get("0.0")
                last_wrong=(r["error_margin_quantiles"] or {}).get("1.0")
                if k and (last_wrong is None or last_wrong>=0):
                    raise AssertionError("Last error must have negative signed margin")
                if k<n and (first_correct is None or first_correct<=0):
                    raise AssertionError("First correct sample must have positive margin")
                if k==n and (first_correct is not None or after is not None):
                    raise AssertionError("No percentile beyond all-error subset")
                if k==0 and last_wrong is not None:
                    raise AssertionError("Unexpected wrong subset for zero-error case")
                gap=(first_correct-last_wrong
                     if first_correct is not None and last_wrong is not None else None)
                rows.append({
                    "policy":p,"H":h,"split":split,"n":n,
                    "errors":k,"observed_unweighted_error_fraction":error_fraction,
                    "balanced_margin_error_fraction":s["hard_margin_error_balanced"],
                    "first_reported_percentile_above_error_fraction":(
                        after[0] if after else None),
                    "margin_at_first_reported_percentile_above_error_fraction":(
                        float(s["quantiles"][after[1]]) if after else None),
                    "first_reported_positive_margin_percentile":(
                        positive[0] if positive else None),
                    "margin_at_first_reported_positive_percentile":(
                        float(s["quantiles"][positive[1]]) if positive else None),
                    "last_misclassified_margin":last_wrong,
                    "first_correct_margin":first_correct,
                    "margin_gap_around_zero":gap,
                    "grid_percentiles":{str(level):float(s["quantiles"][name])
                                        for level,name in GRID},
                    "all_reported_quantiles_above_error":(
                        [level for level,name in GRID if level>error_fraction]),
                })
    return {
        "schema":"e09-error-boundary-quantile-v1",
        "source":["results/margin_analysis.json","results/margin_summary.json"],
        "grid":[p for p,_ in GRID],
        "definition":"min reported percentile p strictly above raw error count / n",
        "not_balanced_error":"Use raw error fraction, NOT 1-balanced accuracy, for empirical sample quantiles",
        "zero_crossing_definition":"last negative margin and first positive margin in sorted all-sample scores",
        "interpolation_caveat":"A quantile p strictly above error fraction is not universally guaranteed positive under linear quantile interpolation",
        "rows":rows,"historical_test_accessed":False,
        "training_performed":False,"walker_invoked":False
    }


def fmt(v):
    return "---" if v is None else f"{v:.5f}"


def level_label(q):
    return "---" if q is None else str(int(q*100)) + r"\%"


def tex(data):
    by={(r["policy"],r["H"],r["split"]):r for r in data["rows"]}
    out=[
        "% E09 first available quantile above raw misclassification rate",
        r"\begin{table}[htbp]",r"\centering\scriptsize",
        r"\caption{اولین چندک گزارش‌شده بالاتر از نرخ خطای مشاهده‌شده در دو مجموعه؛ "
        "سطح چندک و مقدار حاشیه جداگانه آمده‌اند.}",
        r"\label{tab:ch4-e09-first-quantile-after-error}",
        r"\begin{tabular}{rcrrrrrr}\hline",
        r"سیاست & $H$ & خطای آموزش & چندک آموزش & حاشیه چندک آموزش & خطای اعتبارسنجی & چندک اعتبارسنجی & حاشیه چندک اعتبارسنجی\\\hline"
    ]
    for p in POLICIES:
        for h in DEPTHS:
            t=by[(p,h,"train")];v=by[(p,h,"validation")]
            name="یکنواخت" if p=="uniform" else "آموزش‌دیده"
            qt=t["first_reported_percentile_above_error_fraction"]
            qv=v["first_reported_percentile_above_error_fraction"]
            out.append(
                f"{name} & {h} & {t['errors']} & "
                f"{'---' if qt is None else str(int(qt*100))+'\\%'} & "
                f"{fmt(t['margin_at_first_reported_percentile_above_error_fraction'])} & "
                f"{v['errors']} & "
                f"{'---' if qv is None else str(int(qv*100))+'\\%'} & "
                f"{fmt(v['margin_at_first_reported_percentile_above_error_fraction'])}"+r"\\"
            )
        out.append(r"\hline")
    return "\n".join(out+[r"\end{tabular}",r"\end{table}",""])


def csv_table(data):
    columns=("policy","H","split","n","errors","observed_unweighted_error_fraction",
             "balanced_margin_error_fraction",
             "first_reported_percentile_above_error_fraction",
             "margin_at_first_reported_percentile_above_error_fraction",
             "first_reported_positive_margin_percentile",
             "margin_at_first_reported_positive_percentile",
             "last_misclassified_margin","first_correct_margin",
             "margin_gap_around_zero")
    f=io.StringIO()
    writer=csv.DictWriter(f,fieldnames=columns)
    writer.writeheader()
    for row in data["rows"]:
        writer.writerow({name:row[name] for name in columns})
    return f.getvalue()


def markdown(data):
    rows=data["rows"]
    text=[HEADING,"",
          "برای تشخیص فاصله از ناحیهٔ اشتباه، نسبت خطای **غیرمتوازن نمونه‌ای** "
          "را برابر تعداد خطا تقسیم بر تعداد نمونه‌ها در همان Train یا Validation "
          "می‌گیریم. از میان چندک‌های گزارش‌شدهٔ "
          "۰، ۱۰، ۲۵، ۵۰، ۷۵، ۹۰ و ۱۰۰ درصد، **کوچک‌ترین سطحی که "
          "اکیداً بزرگ‌تر از این نسبت است** انتخاب و مقدار حاشیه در آن سطح "
          "گزارش می‌شود. این نسبت با خطای متوازن ۱−BA متفاوت است.","",
          "| سیاست | H | مجموعه | خطا / تعداد | اولین چندک بعد از خطا | مقدار حاشیه چندک | "
          "بزرگ‌ترین حاشیه منفی | کوچک‌ترین حاشیه مثبت |",
          "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        q=r["first_reported_percentile_above_error_fraction"]
        text.append(
            f"| {r['policy']} | {r['H']} | {r['split']} | "
            f"{r['errors']}/{r['n']} | "
            f"{'—' if q is None else f'{q*100:.0f}%'} | "
            f"{fmt(r['margin_at_first_reported_percentile_above_error_fraction'])} | "
            f"{fmt(r['last_misclassified_margin'])} | "
            f"{fmt(r['first_correct_margin'])} |"
        )
    text+=["",
           "آخرین حاشیهٔ منفی و نخستین حاشیهٔ مثبت، مرز **واقعیِ داده‌های مرتب‌شده** "
           "را نشان می‌دهند؛ مقدار چندک از درون‌یابی گزارشگر مشترک به دست آمده "
           "و این دو نباید با یکدیگر اشتباه شوند. با درون‌یابی خطی، "
           "صرفِ سطح چندک بزرگ‌تر از نرخ خطا همیشه علامت مثبت آن چندک را تضمین "
           "نمی‌کند؛ نخستین چندکِ **مثبت** نیز مستقل از شرط نرخ خطا در JSON ثبت شده است.",
           "",
           "منبع: results/error_quantile_boundary.json و "
           "results/error_quantile_boundary.csv؛ جدول چاپی "
           "tables/error_quantile_boundary.tex. بدون اجرای مجدد گشت.",""]
    return "\n".join(text)


def check():
    current=read(OUTPUT)
    if current!=convert():
        raise AssertionError("Persisted error-boundary quantiles differ from saved source")
    if len(current["rows"])!=24:
        raise AssertionError("Expected 24 policy/H/split rows")
    if not CSV.is_file() or not TABLE.is_file():
        raise AssertionError("Missing outputs")
    if TABLE.read_text(encoding="utf8").count(
            r"\label{tab:ch4-e09-first-quantile-after-error}")!=1:
        raise AssertionError("TeX label mismatch")
    if HEADING not in REPORT.read_text(encoding="utf8"):
        raise AssertionError("Scientific report missing quantile section")
    print("E09_QUANTILE_PASS: 24 strict-after-error percentiles and signed-margin zero boundaries verified.")
    print("E09_QUANTILE_ROWS",json.dumps([
        {"p":r["policy"],"h":r["H"],"split":r["split"],
         "error":r["errors"],"q":r["first_reported_percentile_above_error_fraction"],
         "margin":r["margin_at_first_reported_percentile_above_error_fraction"]}
         for r in current["rows"]],ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write",action="store_true")
    action.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.write:
        if OUTPUT.exists():
            raise FileExistsError("Output already exists; use --verify")
        data=convert()
        report=REPORT.read_text(encoding="utf8")
        if HEADING in report:
            raise AssertionError("Would duplicate quantile section")
        OUTPUT.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
        CSV.write_text(csv_table(data),encoding="utf8")
        TABLE.parent.mkdir(parents=True,exist_ok=True)
        TABLE.write_text(tex(data),encoding="utf8")
        REPORT.write_text(report.rstrip()+"\n\n"+markdown(data),encoding="utf8")
    check()


if __name__=="__main__":
    main()
