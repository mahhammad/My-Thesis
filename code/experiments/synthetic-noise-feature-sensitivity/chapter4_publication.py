"""Persian vector figures and TeX tables for E-CH4-14; read-only, no fit/Test."""
from __future__ import annotations
import argparse,csv,io,json,pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import cairosvg
from continuation_analysis import audit,require
from gradient_empirical import empirical_envelope,gradient_series

ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT/"latex"
FDIR=BASE/"figures"
TDIR=BASE/"tables"
CHAPTER=ROOT.parents[2]/"tex/chapter4.tex"
STEPS=np.arange(61)
COLORS=plt.get_cmap("tab10").colors
FAMILIES=("shared","class")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
 "svg.fonttype":"path","svg.hashsalt":"e14-fa-latex-v1","figure.dpi":115})

def fa(s):return get_display(arabic_reshaper.reshape(s))
def t(v,dec=2):return "$"+(f"{v:.{dec}f}" if isinstance(v,float) else str(v))+"$"
def txt(s):return (s.strip()+"\n").encode("utf8")
def by(rows,f,k):
    out=[x for x in rows if x["family"]==f and int(x["k"])==k]
    require(len(out)==1,"Missing level "+f+" "+str(k))
    return out[0]
def tab(label,caption,cols,head,rows):
    return txt("\n".join([
      "% E-CH4-14: Persian publication results, frozen 60-step sources.",
      r"\begin{table}[htbp]",r"\centering\small",
      r"\caption{"+caption+"}",r"\label{"+label+"}",
      r"\begin{tabular}{"+cols+r"}\hline",*head,*rows,
      r"\hline",r"\end{tabular}",r"\end{table}"]))
def render(fig):
    mem=io.BytesIO()
    fig.savefig(mem,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    svg=mem.getvalue()
    pdf=cairosvg.svg2pdf(bytestring=svg)
    require(svg.startswith(b"<?xml") and pdf.startswith(b"%PDF-"),
            "PDF or SVG format broken")
    require(len(svg)>4000 and len(pdf)>5000,"PDF or SVG image empty")
    # Render once for QA: proves fonts and vector shapes are consumable.
    png=cairosvg.svg2png(bytestring=svg,output_width=1100)
    require(png.startswith(b"\x89PNG") and len(png)>10000,"PNG QA failed")
    return svg,pdf,png
def wrap(name,label,caption,tall=False):
    return txt("\n".join([
       "% E-CH4-14: Persian vector publication figure",
       r"\begin{figure}[" + ("p" if tall else "htbp")+"]",
       r"\centering",
       r"\includegraphics[width=\linewidth,height="+("0.82" if tall else "0.52")+
          r"\textheight,keepaspectratio]{"+
          "code/experiments/synthetic-noise-feature-sensitivity/latex/figures/"+name+".pdf}",
       r"\caption{"+caption+"}",r"\label{"+label+"}",r"\end{figure}"]))
def read():
    s=json.loads((ROOT/"tables/final_60_summary.json").read_text())
    e=json.loads((ROOT/"tables/gradient_empirical_envelopes_60.json").read_text())
    b=json.loads((ROOT/"tables/final_60_bayes_reference.json").read_text())
    with (ROOT/"tables/continuation_30_60_by_noise_level.csv").open(newline="") as file:
        n=list(csv.DictReader(file))
    with (ROOT/"tables/gradient_peak_decay_by_run_60.csv").open(newline="") as file:
        rates=list(csv.DictReader(file))
    require(s["fits"]==36 and s["total_updates"]==2160 and not s["test_accessed"],
            "Bad frozen summary")
    require(b["data_seed"]==20261102 and not b["test_accessed"],"Bayes mismatch")
    require(len(s["grouped"])==len(e["groups"])==len(n)==12 and len(rates)==36,
            "Data coverage incomplete")
    return s,e,b,n,rates
def fig_ba(traces,bayes):
    fig,axs=plt.subplots(2,2,figsize=(11.6,7.7),sharex=True,sharey=True)
    for i,(split,key) in enumerate((("train","train_BA"),("validation","validation_BA"))):
      for j,f in enumerate(FAMILIES):
        ax=axs[i,j]
        for k in range(6):
            runs=[trace for run,trace in sorted(traces.items())
                  if run.endswith("__"+f) and int(run[1])==k]
            require(len(runs)==(1 if k in (0,5) else 4),"Invalid group")
            avg=np.mean([[100*x[key] for x in trace] for trace in runs],axis=0)
            ax.plot(STEPS,avg,lw=1.55,color=COLORS[k],label=rf"$|A_N|={k}$")
        ax.axhline(100*bayes[split]["BA"],color="#373737",ls="--",
                   lw=1.4,label=fa("بیز: داده مادر ۵ ویژگی"))
        ax.set_title(fa("آموزش" if i==0 else "اعتبارسنجی")+" — "+
                     fa("مشترک" if f=="shared" else "شرطی کلاس"))
        ax.set_xlim(0,60);ax.set_ylim(70,100.5)
        ax.set_xlabel(fa("گام آموزش"))
        if j==0:ax.set_ylabel(fa("دقت متوازن (درصد)"))
        ax.grid(alpha=.15)
        if i==0 and j==0:ax.legend(fontsize=7,ncol=4,loc="lower right")
    fig.suptitle(fa("دقت متوازن و مرجع زمینه‌ای بیزی داده مادر پنج‌ویژگی"),fontsize=13)
    fig.tight_layout()
    return render(fig)
def fig_gradient(traces):
    fig,axs=plt.subplots(6,2,figsize=(11.7,17.3),sharex=True,sharey="row")
    for k in range(6):
      for j,f in enumerate(FAMILIES):
        ax=axs[k,j]
        members=gradient_series(traces,f,k)
        avg=np.mean([a for _,a in members],axis=0)
        env=empirical_envelope(avg)
        for _,a in members:ax.plot(STEPS,a,color=COLORS[k],alpha=.20,lw=.68)
        ax.plot(STEPS,avg,color=COLORS[k],lw=1.55,label=fa("میانگین"))
        ax.plot(STEPS,env["upper"],color="#aa6800",ls="--",lw=1.3,
                label=fa("پوش بالا"))
        ax.plot(STEPS,env["lower"],color="#078890",ls="--",lw=1.3,
                label=fa("پوش پایین"))
        ax.fill_between(STEPS,env["lower"],env["upper"],color="gray",alpha=.11)
        ax.scatter(env["peaks"],avg[env["peaks"]],s=10,color="#aa6800",zorder=4)
        ax.scatter(env["valleys"],avg[env["valleys"]],s=10,color="#078890",zorder=4)
        ax.set_title(fa("مشترک" if f=="shared" else "شرطی کلاس")+
                     rf"    $|A_N|={k}$",fontsize=10)
        ax.set_xlim(0,60);ax.set_xticks([0,10,20,30,40,50,60])
        ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
        if j==0:ax.set_ylabel(r"$\|\nabla_Z\mathcal{L}\|_2$")
        if k==5:ax.set_xlabel(fa("گام آموزش"))
        if k==0 and j==0:ax.legend(fontsize=6.5,ncol=2)
        ax.grid(alpha=.14)
        require(env["late_gap_smaller"],"Observed damping absent "+f+str(k))
    fig.suptitle(fa("پوش تجربی قله‌ها و دره‌های هنجار گرادیان"),
                 fontsize=13,y=.997)
    fig.subplots_adjust(hspace=.42,wspace=.20,top=.975,bottom=.045)
    return render(fig)
def fig_noise(rows):
    fig,axs=plt.subplots(1,2,figsize=(11.1,4.7))
    kk=np.arange(6)
    for ax,key,title,uni in (
      (axs[0],"E3","شمار مورد انتظار انتخاب نویز",
       [3*k/(3+k) for k in kk]),
      (axs[1],"F3","احتمال برخورد با نویز",
       [1-(3/(3+k))**3 for k in kk])):
        for i,f in enumerate(FAMILIES):
            y=[float(by(rows,f,k)[key+"_60_mean"]) for k in kk]
            ax.plot(kk,y,marker="o",lw=1.7,color=COLORS[i],
                    label=fa("مشترک" if f=="shared" else "شرطی کلاس"))
        ax.plot(kk,uni,ls="--",lw=1.4,color=".25",
                label=fa("سیاست یکنواخت"))
        ax.set(title=fa(title),xticks=kk,xlabel=r"$|A_N|$")
        ax.set_ylabel("$"+key[0]+"_3$")
        ax.grid(alpha=.17)
    axs[0].legend(fontsize=8)
    fig.suptitle(fa("تمایل سیاست به انتخاب ویژگی نویزی در گام شصت"),fontsize=13)
    fig.tight_layout()
    return render(fig)
def produce():
    summary,gradient,bayes,noise,rates=read()
    _,traces,_,_,ledger=audit()
    require(len(traces)==36 and ledger.get("UPDATE")==1080,
            "Audit not complete")
    require(sum(r["full_better_log_fit"]=="exponential" for r in rates)==29,
            "Decay model comparison changed")
    data={}
    rows=[]
    for k in range(6):
        a=by(summary["grouped"],"shared",k);c=by(summary["grouped"],"class",k)
        rows.append(" & ".join([t(k),t(a["n_subsets"]),t(100*a["train_BA_mean"]),
             t(100*a["validation_BA_mean"]),t(100*c["train_BA_mean"]),
             t(100*c["validation_BA_mean"])])+r"\\")
    data[TDIR/"e14_ba_fa.tex"]=tab("tab:ch4-e14-ba",
      "دقت متوازن نهایی بر حسب تعداد ویژگی‌های نویزی؛ مقادیر میانگین زیرمجموعه‌های هر گروه و بر حسب درصد هستند.",
      "rrrrrr",[r"$|A_N|$ & تعداد & \multicolumn{2}{c}{\lr{shared}} & \multicolumn{2}{c}{\lr{class}}\\",
       r" & زیرمجموعه & آموزش & اعتبارسنجی & آموزش & اعتبارسنجی\\\hline"],rows)
    rows=[]
    for k in range(6):
        a=by(summary["grouped"],"shared",k);c=by(summary["grouped"],"class",k)
        u=by(gradient["groups"],"shared",k);v=by(gradient["groups"],"class",k)
        rows.append(" & ".join([t(k),t(a["ba_tv_0_60_pp_mean"],3),
          t(c["ba_tv_0_60_pp_mean"],3),t(u["late_over_early"],3),
          t(v["late_over_early"],3)])+r"\\")
    data[TDIR/"e14_stability_fa.tex"]=tab("tab:ch4-e14-stability",
      "مجموع نوسان دقت متوازن در کل شصت گام (واحد درصد) و نسبت عرض پوش گرادیان در گام‌های ۴۵ تا ۵۵ به گام‌های ۵ تا ۱۵.",
      "rrrrr",[r"$|A_N|$ & \multicolumn{2}{c}{نوسان دقت متوازن} & \multicolumn{2}{c}{نسبت عرض پوش گرادیان}\\",
       r" & \lr{shared} & \lr{class} & \lr{shared} & \lr{class}\\\hline"],rows)
    rows=[]
    for k in range(6):
        a=by(summary["grouped"],"shared",k);c=by(summary["grouped"],"class",k)
        rows.append(" & ".join([t(k),t(1000*a["train_I_mean"],3),
          t(1000*c["train_I_mean"],3),t(1000*a["validation_R_JS_mean"],3),
          t(1000*c["validation_R_JS_mean"],3)])+r"\\")
    data[TDIR/"e14_geometry_fa.tex"]=tab("tab:ch4-e14-geometry",
      "جدایی مراکز آموزش و تراکم جنسن--شنون اعتبارسنجی در گام شصت؛ همهٔ اعداد در هزار ضرب شده‌اند.",
      "rrrrr",[r"$|A_N|$ & \multicolumn{2}{c}{$10^3 I$ آموزش} & \multicolumn{2}{c}{$10^3 R_{\mathrm{JS}}$ اعتبارسنجی}\\",
       r" & \lr{shared} & \lr{class} & \lr{shared} & \lr{class}\\\hline"],rows)
    rows=[]
    for k in range(6):
        a=by(noise,"shared",k);c=by(noise,"class",k)
        rows.append(" & ".join([t(k),t(float(a["E3_60_mean"]),3),
           t(float(c["E3_60_mean"]),3),t(3*k/(3+k),3),
           t(float(a["F3_60_mean"]),3),t(float(c["F3_60_mean"]),3)])+r"\\")
    data[TDIR/"e14_noise_fa.tex"]=tab("tab:ch4-e14-noise",
      "شاخص‌های نویز سیاست در گام شصت؛ $E_3$ تعداد مورد انتظار و $F_3$ احتمال دست‌کم یک انتخاب نویز در سه انتخاب بدون ماسک هستند.",
      "rrrrrr",[r"$|A_N|$ & \multicolumn{3}{c}{$E_3$} & \multicolumn{2}{c}{$F_3$}\\",
       r" & \lr{shared} & \lr{class} & یکنواخت & \lr{shared} & \lr{class}\\\hline"],rows)
    figures=[
      ("e14_ba_bayes_fa","fig:ch4-e14-ba-bayes",
       "دقت متوازن آموزش و اعتبارسنجی در طول شصت گام برای دو خانواده سیاست و شش سطح نویز؛ خطوط افقی مرجع زمینه‌ای \lr{Bayes MAP} دادهٔ مادر پنج‌ویژگی با بذر $20261102$ هستند، نه مرجع بیزی محاسبه‌شده برای هر زیرمجموعهٔ نویزی.",False,
       fig_ba(traces,bayes)),
      ("e14_gradient_6x2_fa","fig:ch4-e14-gradient-envelopes",
       "هنجار گرادیان و پوش تجربی قله‌ها و دره‌ها؛ شش ردیف برای $|A_N|=0,\ldots,5$ و دو ستون برای سیاست مشترک و شرطی کلاس. پوش‌ها از میانگین هر گروه ساخته شده‌اند.",True,
       fig_gradient(traces)),
      ("e14_noise_fa","fig:ch4-e14-noise-selection",
       "شمار مورد انتظار و احتمال برخورد با ویژگی نویزی در گام شصت، در مقایسه با سیاست یکنواخت و برحسب $|A_N|$.",False,
       fig_noise(noise)),
    ]
    for name,label,caption,tall,(svg,pdf,png) in figures:
        data[FDIR/(name+".svg")]=svg
        data[FDIR/(name+".pdf")]=pdf
        data[FDIR/(name+".png")]=png
        data[FDIR/(name+".tex")]=wrap(name,label,caption,tall)
    require(len(data)==16,"Wrong figure/table file count")
    return data
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=("write","check"))
    args=p.parse_args()
    outputs=produce()
    for path,content in outputs.items():
        if args.mode=="write":
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
        else:
            require(path.is_file(),"Missing Persian artifact "+str(path))
            require(path.read_bytes()==content if path.suffix!=".pdf"
                    else path.read_bytes().startswith(b"%PDF-") and
                    path.stat().st_size>5000,
                    "Derived Persian artifact mismatches "+str(path))
    if args.mode=="check":
        chapter=CHAPTER.read_text(encoding="utf8")
        for path in outputs:
            if path.parent==TDIR or (path.parent==FDIR and path.suffix==".tex"):
                ref=("code/experiments/synthetic-noise-feature-sensitivity/latex/"
                    + ("tables/" if path.parent==TDIR else "figures/")
                    + path.name)
                require(ref in chapter,"LaTeX chapter missing asset "+str(path))
    print("E14_PERSIAN_PUBLICATION_PASS",json.dumps({
        "mode":args.mode,"files":len(outputs),"figures":3,"tables":4,
        "test_accessed":False,"optimizer_updates":0}))
if __name__=="__main__":main()
