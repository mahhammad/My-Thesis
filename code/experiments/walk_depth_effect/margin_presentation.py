"""Export existing E09 signed-margin measurements; no training or walk execution."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from presentation import HERE, POLICIES, HS, load, save, texplot

MARKER="## حاشیه تصمیم: Train و Validation"
SOURCE=HERE/"results"/"metrics_full.json"
DEST=HERE/"results"/"margin_summary.json"

def main():
    data=load(SOURCE)
    tau=float(data["metric_tau_report"])
    if abs(tau-0.005)>1e-14:
        raise AssertionError("Unregistered margin temperature")
    cases={(c["policy"],int(c["H"])):c for c in data["cases"]}
    if set(cases)!={(p,h) for p in POLICIES for h in HS}:
        raise AssertionError("Missing true-H policy/depth cases")
    rows=[]
    for policy in POLICIES:
        for h in HS:
            for split in ("train","validation"):
                d=cases[policy,h][split]
                per=d["per_sample"]
                m=np.asarray(per["margins"],dtype=np.float64)
                y=np.asarray(per["labels"],dtype=np.int64)
                if m.shape!=(d["rows"],) or y.shape!=m.shape or not np.isfinite(m).all():
                    raise AssertionError("Malformed saved margin array")
                quant={}
                for name,q in (("min",0.),("p10",.1),("p25",.25),
                               ("median",.5),("p75",.75),("p90",.9),("max",1.)):
                    value=float(np.quantile(m,q))
                    if abs(value-d["margin_quantiles"][str(q)])>1e-13:
                        raise AssertionError("Quantile differs from common reporter")
                    quant[name]=value
                if not np.array_equal(np.unique(y),[0,1,2]):
                    raise AssertionError("Missing label class")
                per_class_means=[float(m[y==c].mean()) for c in range(3)]
                if max(abs(per_class_means[c]-d["per_class"][c]["mean_margin"])
                       for c in range(3))>1e-13:
                    raise AssertionError("Class mean margin mismatch")
                hard=float(np.mean([(m[y==c]<=0).mean() for c in range(3)]))
                if abs(hard-d["hard_margin_error_conservative"])>1e-13:
                    raise AssertionError("Hard margin error mismatch")
                sigmoid=np.exp(-np.logaddexp(0.,m/tau))
                loss=float(np.mean([sigmoid[y==c].mean() for c in range(3)]))
                if abs(loss-d["balanced_sigmoid_margin_loss"])>1e-13:
                    raise AssertionError("Sigmoid loss mismatch")
                ties=int(np.count_nonzero(per["ties"]))
                if ties!=d["tie_count"]:
                    raise AssertionError("Tie mismatch")
                n_nonpositive=int(np.count_nonzero(m<=0))
                rows.append({
                    "policy":policy,"H":h,"split":split,"n":len(m),
                    "quantiles":quant,"class_balanced_mean_margin":float(np.mean(per_class_means)),
                    "per_class_mean_margin":per_class_means,
                    "nonpositive_count":n_nonpositive,
                    "nonpositive_fraction_unweighted":n_nonpositive/len(m),
                    "hard_margin_error_balanced":hard,
                    "sigmoid_margin_loss_balanced":loss,
                    "BA":float(d["balanced_accuracy"]),"ties":ties
                })
    if len(rows)!=24:
        raise AssertionError("Expected 24 margin blocks")
    output={
        "schema":"e09-margin-from-common-metrics-v1",
        "definition":"true-class score minus largest rival score, score=1-JSD(P,V)/log(2)",
        "source":"code/experiments/walk_depth_effect/results/metrics_full.json",
        "source_reporter":"code/experiments/benchmark_v1/report_snapshot.py::standard_metrics",
        "tau":tau,"true_H":list(HS),"policies":list(POLICIES),
        "centers_Train_only":True,"Test_accessed":False,
        "trained_policy_updated":False,"walker_executed":False,
        "rows":rows
    }
    save(DEST,json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    cols=("policy","H","split","n","min","p10","p25","median","p75","p90","max",
          "class_balanced_mean_margin","nonpositive_count",
          "nonpositive_fraction_unweighted","hard_margin_error_balanced",
          "sigmoid_margin_loss_balanced","BA","ties")
    with (HERE/"results"/"margin_summary.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k:(r["quantiles"][k] if k in r["quantiles"] else r[k]) for k in cols})
    by={(r["policy"],r["H"],r["split"]):r for r in rows}
    table=[
      "% E09 signed margin: results/margin_summary.json via common reporter",
      "\\begin{table}[htbp]","\\centering\\scriptsize",
      "\\caption{چندک‌ها و زیان حاشیه در آموزش و اعتبارسنجی؛ چندک‌ها نمونه‌ای "
      "و زیان سیگموئیدی میانگین متوازن کلاس‌هاست.}",
      "\\label{tab:ch4-e09-margin-by-h}",
      "\\begin{tabular}{rcrrrrrr}\\hline",
      "سیاست & $H$ & $q_{.1}^{\\mathrm{Train}}$ & $q_{.5}^{\\mathrm{Train}}$ "
      "& $q_{.1}^{\\mathrm{Val}}$ & $q_{.5}^{\\mathrm{Val}}$ "
      "& $L_{\\mathrm{sig}}^{\\mathrm{Train}}$ & $L_{\\mathrm{sig}}^{\\mathrm{Val}}$"
      "\\\\\\hline"
    ]
    for p in POLICIES:
        for h in HS:
            t,v=by[p,h,"train"],by[p,h,"validation"]
            label="یکنواخت" if p=="uniform" else "آموزش‌دیده"
            table.append(label+" & $"+str(h)+"$ & "+
                         f"{t['quantiles']['p10']:.5f} & {t['quantiles']['median']:.5f} & "+
                         f"{v['quantiles']['p10']:.5f} & {v['quantiles']['median']:.5f} & "+
                         f"{t['sigmoid_margin_loss_balanced']:.5f} & "+
                         f"{v['sigmoid_margin_loss_balanced']:.5f}"+r"\\")
        table.append("\\hline")
    table+=["\\end{tabular}","\\end{table}",""]
    save(HERE/"tables"/"margin_by_H.tex","\n".join(table))
    for split in ("train","validation"):
        fig=texplot(cases,
                    lambda c,split=split:c[split]["margin_quantiles"]["0.5"],
                    "میانه حاشیه علامت‌دار "+("آموزش" if split=="train" else "اعتبارسنجی")+
                    " با تغییر سقف واقعی گشت.",
                    "fig:ch4-e09-"+split+"-margin")
        save(HERE/"figures"/f"{split}_margin_median_vs_H.tex",fig)
    path=HERE/"REPORT_FA.md"
    pre=path.read_text(encoding="utf8").split(MARKER)[0].rstrip()
    report=[
        MARKER,"",
        "حاشیه علامت‌دار از امتیاز کلاس درست منهای بزرگ‌ترین امتیاز رقیب "
        "به‌دست آمده و در گزارشگر مشترک محاسبه شده است. چندک‌ها بر اساس "
        "همه نمونه‌ها، و زیان سیگموئیدی با دمای ۰٫۰۰۵ به‌صورت متوازن "
        "بین سه کلاس محاسبه می‌شوند. حاشیه غیرمثبت نشان‌دهنده خطا یا تساوی است.",
        "",
        "| سیاست | H | مجموعه | q10 | q25 | میانه | q75 | q90 | "
        "میانگین متوازن حاشیه | زیان سیگموئیدی متوازن | خطای حاشیه متوازن | شمار غیرمثبت |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in rows:
        q=r["quantiles"]
        report.append(f"| {r['policy']} | {r['H']} | {r['split']} | "+
                      f"{q['p10']:.6f} | {q['p25']:.6f} | {q['median']:.6f} | "+
                      f"{q['p75']:.6f} | {q['p90']:.6f} | "+
                      f"{r['class_balanced_mean_margin']:.6f} | "+
                      f"{r['sigmoid_margin_loss_balanced']:.6f} | "+
                      f"{r['hard_margin_error_balanced']:.6f} | {r['nonpositive_count']} |")
    report+=["",
             "کاهش حاشیه با افزایش H حتی در صورت ثابت‌ماندن BA دیده می‌شود. "
             "بهبود تراکم لزوماً به معنای افزایش اطمینان تصمیم نیست.",
             "",
             "منبع: results/metrics_full.json؛ خلاصه کامل در results/margin_summary.json "
             "و results/margin_summary.csv. جدول چاپی tables/margin_by_H.tex و "
             "نمودارهای figures/train_margin_median_vs_H.tex و "
             "figures/validation_margin_median_vs_H.tex. هیچ گشت یا آموزش جدید انجام نشده است.",
             ""]
    path.write_text(pre+"\n\n"+"\n".join(report),encoding="utf8")
    print("E09_MARGIN_PASS",json.dumps({
        "rows":len(rows),"tau":tau,
        "endpoints":{p:{split:{
            "median_H1":by[p,1,split]["quantiles"]["median"],
            "median_H12":by[p,12,split]["quantiles"]["median"],
            "loss_H1":by[p,1,split]["sigmoid_margin_loss_balanced"],
            "loss_H12":by[p,12,split]["sigmoid_margin_loss_balanced"],
            "hard_H1":by[p,1,split]["hard_margin_error_balanced"],
            "hard_H12":by[p,12,split]["hard_margin_error_balanced"]
        } for split in ("train","validation")} for p in POLICIES}
    },ensure_ascii=False))

if __name__=="__main__":
    main()
