"""Generate E09 TeX figures, tables and report from saved JSON (no experiments)."""
from __future__ import annotations

import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
POLICIES=("uniform","trained_shared")
HS=(1,2,3,4,8,12)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value,encoding="utf8")


def texplot(cases, metric, title, label):
    """Plain TikZ plot with two fixed-policy lines."""
    vals={p:[metric(cases[(p,h)]) for h in HS] for p in POLICIES}
    flat=vals["uniform"]+vals["trained_shared"]
    ymin=0. if min(flat)>=0 else min(flat)*1.1
    ymax=max(flat)*1.15 if max(flat)>0 else 1.
    if ymax-ymin<1e-14:
        ymax=ymin+1e-5
    x=lambda h:1.4+5.8*(h-1)/11.
    y=lambda v:1.0+3.1*(v-ymin)/(ymax-ymin)
    lines=["% Derived from E09 verified results/metrics_full.json",
           "\\begin{figure}[htbp]","\\centering",
           "\\begin{tikzpicture}[x=1cm,y=1cm]",
           "\\draw[->] (1.3,1)--(7.55,1);",
           "\\draw[->] (1.3,1)--(1.3,4.4);",
           "\\node[font=\\scriptsize] at (4.3,.45) {$H$};"]
    for h in HS:
        lines.append("\\node[font=\\scriptsize,anchor=north] at ("+
                     f"{x(h):.3f},.94"+") {"+str(h)+"};")
    for i in range(5):
        yy=1+3.1*i/4.
        vv=ymin+(ymax-ymin)*i/4.
        lines.append(f"\\draw[gray!20] (1.35,{yy:.3f})--(7.3,{yy:.3f});")
        lines.append("\\node[font=\\tiny,anchor=east] at ("+
                     f"1.21,{yy:.3f}"+") {"+f"{vv:.3f}"+"};")
    for p,color in (("uniform","blue"),("trained_shared","orange!80!black")):
        dots=" ".join(f"({x(h):.3f},{y(v):.3f})" for h,v in zip(HS,vals[p]))
        lines.append("\\draw[very thick,"+color+"] plot coordinates {"+dots+"};")
        for h,v in zip(HS,vals[p]):
            lines.append("\\fill["+color+"] ("+
                         f"{x(h):.3f},{y(v):.3f}"+") circle (2pt);")
    lines+=["\\node[font=\\scriptsize,blue] at (2.6,4.65) {یکنواخت};",
            "\\node[font=\\scriptsize,orange!80!black] at (5.35,4.65) {آموزش‌دیده};",
            "\\end{tikzpicture}",
            "\\caption{"+title+"}",
            "\\label{"+label+"}",
            "\\end{figure}",""]
    return "\n".join(lines)


def render_all(metrics_path,comp_path,alpha_path):
    doc=load(metrics_path)
    cmp=load(comp_path)
    alpha=load(alpha_path)
    cases={(r["policy"],r["H"]):r for r in doc["cases"]}
    if len(cases)!=12 or set(cases)!={(p,h) for p in POLICIES for h in HS}:
        raise AssertionError("Incomplete 12-case presentation")
    main=[
        "% E09 derived exclusively from metrics_full.json",
        "\\begin{table}[htbp]","\\centering\\scriptsize",
        "\\caption{اثر سقف واقعی گشت بر هندسه و دقت دو سیاست ثابت؛ داده آزمون استفاده نشده است.}",
        "\\label{tab:ch4-e09-depth-main}",
        "\\begin{tabular}{rcrrrrrr}\\hline",
        "سیاست & $H$ & $D_{\\mathrm{eff}}$ & $I$ آموزش & $R_{\\mathrm{JS}}$ آموزش & "
        "BA آموزش (\\%) & BA اعتبارسنجی (\\%) & اختلاف پیش‌بینی با $H=12$\\\\\\hline"
    ]
    for p in POLICIES:
        for h in HS:
            d=cases[(p,h)]
            t,v=d["train"],d["validation"]
            n="یکنواخت" if p=="uniform" else "آموزش‌دیده"
            main.append(n+" & $"+str(h)+"$ & "+
                        f"{alpha['by_H'][str(h)]['expected_weighted_step_index']:.4f} & "+
                        f"{t['I']:.5f} & {t['R_JS']:.5f} & "+
                        f"{100*t['balanced_accuracy']:.2f} & "+
                        f"{100*v['balanced_accuracy']:.2f} & "+
                        str(v["prediction_disagreements_vs_true_H12"])+r"\\")
        main.append("\\hline")
    main+=["\\end{tabular}","\\end{table}",""]
    save(HERE/"tables"/"main_depth_effect.tex","\n".join(main))
    other=[
        "% E09 computed from true-H alpha versus fixed-H12 truncated alpha",
        "\\begin{table}[htbp]","\\centering\\scriptsize",
        "\\caption{تفاوت سقف واقعی $H=h$ و تقریب $r=h$ از گشت ثابت $H=12$.}",
        "\\label{tab:ch4-e09-true-vs-truncated}",
        "\\begin{tabular}{rcrrrr}\\hline",
        "سیاست & $h=r$ & اختلاف ضرایب $L_1$ & میانگین TV اعتبارسنجی & "
        "BA واقعی (\\%) & BA برش (\\%)\\\\\\hline"
    ]
    for p in POLICIES:
        for h in HS:
            d=cmp["true_H_vs_E08_fixed_H12_truncation"][f"{p}_H{h}"]
            name="یکنواخت" if p=="uniform" else "آموزش‌دیده"
            other.append(name+" & $"+str(h)+"$ & "+
                         f"{d['coefficient_l1']:.5f} & "+
                         f"{d['validation_mean_TV']:.5f} & "+
                         f"{100*d['validation_BA_true_H']:.2f} & "+
                         f"{100*d['validation_BA_fixed_H12_truncation']:.2f}"+
                         r"\\")
        other.append("\\hline")
    other+=["\\end{tabular}","\\end{table}",""]
    save(HERE/"tables"/"true_H_vs_E08_truncation.tex","\n".join(other))
    for file,fn,title,label in (
        ("validation_ba_vs_H.tex",lambda x:x["validation"]["balanced_accuracy"],
         "دقت متوازن اعتبارسنجی در برابر سقف واقعی طول گشت.",
         "fig:ch4-e09-validation-ba"),
        ("center_information_vs_H.tex",lambda x:x["train"]["I"],
         "جدایی مراکز $I$ در برابر سقف واقعی گشت.",
         "fig:ch4-e09-I"),
        ("dispersion_vs_H.tex",lambda x:x["train"]["R_JS"],
         "پراکندگی درون‌کلاسی $R_{\\mathrm{JS}}$ در برابر سقف واقعی گشت.",
         "fig:ch4-e09-RJS")
    ):
        save(HERE/"figures"/file,texplot(cases,fn,title,label))

    lines=[
        "# گزارش E-CH4-09 — تغییر واقعی سقف گشت","",
        "پروتکل: دو سیاست مشترک ثابت (Uniform و trained_shared)، "
        "β=.4، λ=.8، H=1,2,3,4,8,12، داده 20261102، 300 Train و 90 Validation؛ "
        "تمام نتایج از توزیع‌های گام‌به‌گام ذخیره‌شده E08 بازترکیب شده‌اند. "
        "هیچ آموزش، تولید داده، Test، مونت‌کارلو یا اجرای مجدد موتور گشت انجام نشد.","",
        "## همه ۱۲ ترکیب", "",
        "| سیاست | H | BA Train (%) | BA Validation (%) | I Train | R_JS Train | "
        "R_sqrtJS Train | میانه حاشیه Validation | تغییر برچسب نسبت به H12 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for p in POLICIES:
        for h in HS:
            d=cases[(p,h)]
            t,v=d["train"],d["validation"]
            lines.append(f"| {p} | {h} | {100*t['balanced_accuracy']:.4f} | "
                         f"{100*v['balanced_accuracy']:.4f} | {t['I']:.7f} | "
                         f"{t['R_JS']:.7f} | {t['R_sqrt_JS']:.7f} | "
                         f"{v['margin_quantiles']['0.5']:.6f} | "
                         f"{v['prediction_disagreements_vs_true_H12']} |")
    lines+=["","## نتیجه‌های قابل استناد",""]
    for p in POLICIES:
        x=cases[(p,1)]; y=cases[(p,12)]
        bas=[cases[(p,h)]["validation"]["balanced_accuracy"] for h in HS]
        maxba=max(bas)
        best=[h for h,ba in zip(HS,bas) if abs(ba-maxba)<1e-12]
        change=[cases[(p,h)]["validation"]["prediction_disagreements_vs_true_H12"] for h in HS]
        lines.append(
            f"**{p}:** BA اعتبارسنجی از {100*x['validation']['balanced_accuracy']:.4f}% "
            f"(H1) به {100*y['validation']['balanced_accuracy']:.4f}% (H12) رسید؛ "
            f"I از {x['train']['I']:.7f} به {y['train']['I']:.7f} و "
            f"R_JS از {x['train']['R_JS']:.7f} به {y['train']['R_JS']:.7f} رسید. "
            f"بالاترین BA مشاهده‌شده {100*maxba:.4f}% در Hهای {best} بود؛ "
            f"اختلاف برچسب نسبت به H12 به‌ترتیب {change} است."
        )
    short=[(p,h) for p in POLICIES for h in HS if h<12 and
           cases[(p,h)]["validation"]["balanced_accuracy"] >=
           cases[(p,12)]["validation"]["balanced_accuracy"]-1e-12]
    stable=[(p,h) for p in POLICIES for h in HS if h<12 and
            cases[(p,h)]["validation"]["prediction_disagreements_vs_true_H12"]==0 and
            abs(cases[(p,h)]["train"]["I"]-cases[(p,12)]["train"]["I"])>1e-9]
    lines+=["",
        "**آیا کوتاه‌تر برابر یا بهتر است؟** "+
        (", ".join(f"{p}, H={h}" for p,h in short) if short else "در این داده مشاهده نشد.")+
        ". این نتیجه به معنای بهترین عمق در جامعه نیست.",
        "",
        "**تغییر هندسه بدون تغییر هیچ پیش‌بینی Validation:** "+
        (", ".join(f"{p}, H={h}" for p,h in stable) if stable else "در مقایسه با H12 تأیید نشد.")+".",
        "",
        "## تفاوت با E08","",
        "در E09 برای هر H=h توزیع هندسی سقف‌دار از نو تعریف می‌شود؛ "
        "در E08 ضرایب H=12 فقط در r=h بریده و بازنرمال می‌شوند. "
        "این دو را هم‌ارز فرض نکرده‌ایم. مقدار اختلاف در جدول زیر و در فایل مقایسه ثبت شده است.",
        "",
        "| سیاست | h=r | اختلاف L1 ضرایب | میانگین TV Validation | BA واقعی (%) | BA برش (%) |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for p in POLICIES:
        for h in HS:
            d=cmp["true_H_vs_E08_fixed_H12_truncation"][f"{p}_H{h}"]
            lines.append(f"| {p} | {h} | {d['coefficient_l1']:.7f} | "
                         f"{d['validation_mean_TV']:.7f} | "
                         f"{100*d['validation_BA_true_H']:.4f} | "
                         f"{100*d['validation_BA_fixed_H12_truncation']:.4f} |")
    max4=max((max(v.values()) for v in cmp["E04_uniform_true_h_max_abs_diff"].values()),default=0.)
    max8=max(cmp["E08_H12_center_max_abs_diff"].values())
    lines+=["","## کنترل‌ها، قرارداد کامل سنجه‌ها و محدودیت‌ها","",
        f"- کنترل E04 Uniform: بیشینه اختلاف معیارهای ذخیره‌شده {max4:.3e}.",
        f"- کنترل E08 H12: بیشینه اختلاف مرکز {max8:.3e}.",
        "- از گزارشگر مشترک benchmark_v1/report_snapshot.py::standard_metrics استفاده شد؛ "
        "تعریف دوباره معیارهای R_KL/R_JS/R_sqrtJS، T/BT/TNL/BTNL/I، "
        "BA/Accuracy/Recall/Confusion، حاشیه و NLL صورت نگرفت.",
        "- ۲۴ گزارش کامل Train و Validation با هندسه مراکز، فاصله/شباهت زوجی، "
        "NLLهای KL ساده، KL نرمال تاریخی و JS مبتنی بر شباهت، "
        "تعداد tie، توزیع حاشیه و پیش‌بینی تک‌نمونه‌ای در metrics_full.json حفظ شدند.",
        "- Validation در برازش مراکز یا سیاست شرکت نکرده و فقط برای گزارش معیارها استفاده شده است.",
        "- زمان‌های ثبت‌شده مربوط به بازترکیب و محاسبه سنجه‌ها هستند، "
        "نه زمان اجرای موتور گشت در عمق‌های متفاوت.",
        "- یک بذر مولد و یک تقسیم داریم؛ ادعای معناداری آماری، بهترین H عمومی "
        "یا برتری سیاست در جامعه مطرح نیست.",
        "- D10 درباره کران دست‌یابی به مراکز هدف است، نه کیفیت طبقه‌بندی این دو سیاست.",
        "",
        "## فایل‌ها","",
        "- results/metrics_full.json: تمام ۱۲ ترکیب، دو تقسیم و خروجی‌های نمونه‌ای.",
        "- results/metrics_summary.csv: ۲۴ ردیف خلاصه برای دو تقسیم.",
        "- results/coefficients.json: pmf طول و ضرایب هر H، طول متوسط و عمق مؤثر.",
        "- results/comparisons.json: کنترل ۴.۴، E08 و تفاوت روش برش.",
        "- results/provenance.json و coverage.json: منشأ، آزمون سلامت و وضعیت پوشش.",
        "- tables/ و figures/: خروجی‌های چاپی تولیدشده از JSONهای ثابت.",
        ""
    ]
    save(HERE/"results"/"presentation_generated_report.txt","\n".join(lines))


if __name__=="__main__":
    render_all(HERE/"results"/"metrics_full.json",
               HERE/"results"/"comparisons.json",
               HERE/"results"/"coefficients.json")
