#!/usr/bin/env python3
"""R05 §4.10 raw-derived XeLaTeX table/figure include wrappers (NO solver)."""
from __future__ import annotations
import collections,itertools,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"results/metrics_full.json"
COVER=ROOT/"results/coverage.json"
TABLE=ROOT/"tables"
FIG=ROOT/"figures"

def fail(cond,msg):
    if not cond:raise RuntimeError(msg)
def write_new(path,data):
    fail(not path.exists(),"refuse overwrite existing LaTeX: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(data,encoding="utf-8")
def fmt(v):
    if v is None:return "---"
    if v==0:return "$0$"
    e=int(math.floor(math.log10(abs(v))))
    if abs(e)>=3:
        return "$%.2f\\times10^{%d}$"%(v/10**e,e)
    return "$%.5f$"%v

def main():
    m=json.loads(DATA.read_text())
    c=json.loads(COVER.read_text())
    fail(c.get("status_counts")=={"COMPLETED":108},"NOT_108_COMPLETED")
    fail(len(m.get("reconstructed_center_density_and_geometry",[]))==6,
         "NOT_SIX_EXACT_REPLAYS")
    groups=m["group_metrics"]
    methods=("Q-only","Exact-Adam","Q+Bootstrap-GN")
    names={"Q-only":"\\lr{Q-only}","Exact-Adam":"\\lr{Exact-Adam}",
           "Q+Bootstrap-GN":"\\lr{Q+Bootstrap-GN}"}
    lines=[
      "\\begin{table}[htbp]",
      "\\centering\\small",
      "\\caption{موفقیت در بازسازی دقیق مراکز دست‌یافتنی در سه بذر مولد برای هر اندازه مرجع. برای هر روش در هر اندازه، ۱۸ اجرای برنامه‌ریزی‌شده (سه بذر، سه کلاس و دو آغاز) در مخرج باقی مانده‌اند؛ این تعداد، ۱۸ تکرار مستقل مولد نیست.}",
      "\\label{tab:ch4-inverse-full-success}",
      "\\begin{tabular}{rcccc}\\hline",
      "روش & موفقیت $n=150$ & موفقیت $n=300$ & میانه \\lr{JSD}، $n=150$ & میانه \\lr{JSD}، $n=300$\\\\\\hline"
    ]
    for method in methods:
        out=[]
        for n in (150,300):
            a=[g for g in groups if g["solver"]==method and g["nref"]==n]
            fail(len(a)==3 and all(g["planned"]==g["completed"]==6 for g in a),
                 "MISSING_BLOCK_FOR_"+method+str(n))
            success=sum(g["success"] for g in a)
            err=[r["best_exact_center_JSD"] for r in c["records"]
                 if r["solver"]==method and r["nref"]==n]
            fail(len(err)==18 and all(math.isfinite(x) for x in err),"MISSING_ERROR")
            out.extend([f"${success}/18$",fmt(statistics.median(err))])
        lines.append(f'{names[method]} & {out[0]} & {out[2]} & {out[1]} & {out[3]}\\\\')
    lines.extend(["\\hline","\\end{tabular}","\\end{table}",""])
    write_new(TABLE/"sec410_success.tex","\n".join(lines))
    lines=[
      "\\begin{table}[htbp]","\\centering\\small",
      "\\caption{پوشش بلوک‌های استاندارد مسئله معکوس. در هر بلوک ۳ کلاس، ۲ آغاز و ۳ روش بررسی شده است؛ طرح از سه بذر مستقل مولد تشکیل می‌شود.}",
      "\\label{tab:ch4-inverse-blocks}",
      "\\begin{tabular}{rcccc}\\hline",
      "بلوک & اندازه مرجع & بذر داده & تعداد اجرا & موفقیت کل\\\\\\hline"
    ]
    for a in sorted(m["group_metrics"],key=lambda z:z["block_id"]):
        pass
    keys=list(dict.fromkeys(g["block_id"] for g in groups))
    for bid in keys:
        a=[g for g in groups if g["block_id"]==bid]
        fail(len(a)==3,"BLOCK_ROWS")
        lines.append(f"\\lr{{{bid.split('-')[1]}}} & {a[0]['nref']} & "+
            f"\\lr{{{a[0]['data_seed']}}} & 18 & {sum(g['success'] for g in a)}/18\\\\")
    lines.extend(["\\hline","\\end{tabular}","\\end{table}",""])
    write_new(TABLE/"sec410_blocks.tex","\n".join(lines))
    rec=m["reconstructed_center_density_and_geometry"]
    lines=["\\begin{table}[htbp]","\\centering\\small",
      "\\caption{معیارهای تراکم $R$ و جدایی $I$ محاسبه‌شده از بازپخش سیاست‌های منتخب روی مراجع آموزش؛ اعداد میانگین توصیفی بلوک‌ها و دو آغاز هستند و دقت طبقه‌بندی را بیان نمی‌کنند.}",
      "\\label{tab:ch4-inverse-R-I}",
      "\\begin{tabular}{rcccc}\\hline",
      "روش & $R_{\\mathrm{KL}}$، $n=150$ & $R_{\\mathrm{KL}}$، $n=300$ & $I$، $n=150$ & $I$، $n=300$\\\\\\hline"]
    for method in methods:
        cols=[]
        for n in (150,300):
            arr=[v for b in rec if b["nref"]==n for v in b["combinations"]
                 if v["method"]==method]
            fail(len(arr)==6,"MISSING_R_I")
            cols.append((statistics.mean(v["R_KL_uniform_class"] for v in arr),
                         statistics.mean(v["I"] for v in arr)))
        lines.append(f"{names[method]} & {fmt(cols[0][0])} & {fmt(cols[1][0])} & "+
                     f"{fmt(cols[0][1])} & {fmt(cols[1][1])}\\\\")
    lines.extend(["\\hline","\\end{tabular}","\\end{table}",""])
    write_new(TABLE/"sec410_density.tex","\n".join(lines))
    rootname="code/experiments/inverse_attainable_standard/figures"
    figfiles=[
      ("sec410_error.tex","best_jsd_by_n.pdf","fig:ch4-inverse-best-jsd",
       "خطای بهترین مرکز دقیق در ۱۰۸ اجرای بازسازی معکوس برحسب اندازه مرجع؛ خط چین آستانه $10^{-8}$ است."),
      ("sec410_density_vs_I.tex","density_vs_separation.pdf","fig:ch4-inverse-density-vs-I",
       "پراکندگی میان تراکم درون‌طبقه‌ای $R_{\\mathrm{KL}}$ و جدایی اطلاعاتی مراکز $I$ برای سیاست‌های منتخب روی داده‌های آموزش. نزدیکی مراکز به هدف، یکسانی $R$ را تضمین نمی‌کند.")
    ]
    for fname,figure,label,caption in figfiles:
        fail((FIG/figure).is_file(),"FIGURE_MISSING_"+figure)
        tex="\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.83\\linewidth]{./"+rootname+"/"+figure+"}\n"+"\\caption{"+caption+"}\n"+"\\label{"+label+"}\n"+"\\end{figure}\n"
        write_new(FIG/fname,tex)
    print("R05_SECTION410_LATEX_EXPORT_PASS",len(lines),"6 blocks / 108 runs",flush=True)
if __name__=="__main__":main()
