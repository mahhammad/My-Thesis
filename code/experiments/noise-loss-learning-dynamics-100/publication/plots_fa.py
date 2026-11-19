"""Persian thesis-ready graphics from verified E14 trace artifacts. NO MODEL FITS.

Separate Loss and gradient scale per objective and family; split the policy
diagnostics into objective/family charts; show six paired subsets and their mean.
Uses matplotlib only for REPORTING, not for training (which remains NumPy-only).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from source import load, group_runs, get_series, policy_at, FAMILIES, OBJECTIVES, COMPARABLE

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def fa(text):
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError as exc:
    raise RuntimeError("Persian plotting requires arabic-reshaper and python-bidi") from exc

fonts = font_manager.findSystemFonts()
arabic_font = next((f for f in fonts if "NotoSansArabic-Regular" in f),
                   next((f for f in fonts if "NotoNaskhArabic-Regular" in f), None))
if not arabic_font:
    raise RuntimeError("Noto Arabic font required; install fonts-noto-core")
font_manager.fontManager.addfont(arabic_font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=arabic_font).get_name(),
                     "font.size": 9, "svg.fonttype": "none"})

OBJ = {"sigmoid_JS_margin_tau005": ("sigmoid", "سیگمویدی", "#265f9d"),
       "softplus_JS_tau005": ("softplus", "سافت‌پلاس", "#d77a38")}
FAM = {"shared": "سیاست مشترک", "class": "سیاست وابسته به کلاس"}

CORE = {
    "separation_i": ("train", "شاخص جدایی مراکز I", "I"),
    "density_r_js": ("train", "تراکم درونی R بر پایه JS", "R_JS"),
    "density_r_kl": ("train", "تراکم درونی R بر پایه KL", "R_KL"),
    "density_r_sqrt_js": ("train", "تراکم درونی R بر پایه ریشه JS", "R_sqrt_JS"),
    "separation_t": ("train", "معیار جدایی T", "T"),
    "separation_bt": ("train", "معیار جدایی BT", "BT"),
    "separation_tnl": ("train", "معیار جدایی TNL", "TNL"),
    "separation_btnl": ("train", "معیار جدایی BTNL", "BTNL"),
    "margin_mean": ("train", "میانگین حاشیه تصمیم", "margin"),
    "balanced_accuracy_train": ("train", "دقت متوازن آموزش", "BA (%)"),
    "balanced_accuracy_validation": ("validation", "دقت متوازن اعتبارسنجی", "BA (%)"),
    "accuracy_validation": ("validation", "دقت طبقه‌بندی اعتبارسنجی", "Accuracy (%)"),
    "precision_macro_validation": ("validation", "میانگین پرسیژن اعتبارسنجی", "Precision (%)"),
    "recall_macro_validation": ("validation", "میانگین ریکال اعتبارسنجی", "Recall (%)"),
}
POLICY = {
    "q1": ("انتخاب نویز در نوبت اول", "q1", "share"),
    "q2": ("انتخاب نویز در نوبت دوم", "q2", "share"),
    "q3": ("انتخاب نویز در نوبت سوم", "q3", "share"),
    "hit2": ("احتمال حداقل یک برخورد نویزی در دو نوبت", "p_hit,N^(2)", "share"),
    "hit3": ("احتمال حداقل یک برخورد نویزی در سه نوبت", "p_hit,N^(3)", "share"),
    "nbar3": ("تعداد مورد انتظار انتخاب نویز", "nbar_N^(3)", "count"),
    "s2": ("سهم نویز در دو انتخاب", "s_N^(2)", "share"),
    "s3": ("سهم نویز در سه انتخاب", "s_N^(3)", "share"),
    "sp3": ("سهم غیرنویز در سه انتخاب", "s_P^(3)", "share"),
    "NFI3": ("نسبت احتمال برخورد نویز به یکنواخت", "NFI_N^(3)", "ratio"),
    "NSR2": ("نسبت سهم نویز به یکنواخت در دو نوبت", "NSR_N^(2)", "ratio"),
    "NSR3": ("نسبت سهم نویز به یکنواخت در سه نوبت", "NSR_N^(3)", "ratio"),
}


def draw_six(data, out, family, specification, objective=None, scale="zoom", policy=False):
    kind, title, ylabel = specification[:3]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), sharex=True, layout="constrained")
    objectives = (objective,) if objective else OBJECTIVES
    for noise, ax in zip(COMPARABLE, axes.flat):
        all_values = []
        for obj in objectives:
            group = group_runs(data, family, obj, noise)
            if policy:
                key = kind
                series = np.array([[policy_at(run,row)[0][key] for row in run["rows"]]
                                   for run in group], dtype=float)
            else:
                split, key = kind
                series = np.stack([get_series(run, split, key) for run in group])
                if key in ("accuracy","balanced_accuracy","precision_macro","recall_macro"):
                    series *= 100
            assert series.shape == (6,101)
            assert np.isfinite(series).all(), ("Nonfinite values",family,obj,noise,kind)
            color = OBJ[obj][2]
            for trace in series:
                y = np.where(trace > 0,trace,np.nan) if scale == "log" else trace
                ax.plot(np.arange(101),y,color=color,alpha=.19,lw=.8)
            mean = series.mean(axis=0)
            ax.plot(np.arange(101),np.where(mean>0,mean,np.nan) if scale=="log" else mean,
                    color=color,lw=2.4,label=OBJ[obj][1])
            all_values.extend(series.flatten().tolist())
        vals = np.array(all_values,dtype=float)
        if scale == "log":
            pos = vals[vals>0]
            assert len(pos), ("Cannot log nonpositive-only series",kind)
            ax.set_yscale("log")
            low,high=float(pos.min()),float(pos.max())
            ax.set_ylim(low/1.15,high*1.15)
        elif kind in (("train","balanced_accuracy"),("validation","balanced_accuracy")):
            ax.set_ylim(70,100)
            if np.min(vals)<70 or np.max(vals)>100:
                ax.annotate(fa("بخشی از مقادیر خارج از بازه"),(.02,.04),
                            xycoords="axes fraction",fontsize=8,color="#a04a22")
        else:
            lo,hi = float(vals.min()),float(vals.max())
            pad=max((hi-lo)*.1,abs(hi)*.015,1e-7)
            ax.set_ylim(lo-pad,hi+pad)
        ax.set_xlim(0,100)
        ax.set_xticks([0,25,50,75,100])
        ax.set_title("|A_N| = "+str(noise))
        ax.set_xlabel(fa("گام آموزش"))
        ax.grid(alpha=.16)
        if noise in (1,4):
            ax.set_ylabel(ylabel)
    familyfa = FAM[family]
    objectivefa = ("؛ زیان "+OBJ[objective][1]) if objective else "؛ دو تابع هدف"
    fig.suptitle(fa(title+"؛ "+familyfa+objectivefa),fontsize=16)
    fig.text(.50,.010,fa("خط کم‌رنگ: شش زیرمجموعه وابسته؛ خط پررنگ: میانگین شش زیرمجموعه؛ گام‌های صفر تا صد"),
             ha="center",fontsize=10)
    if not objective:
        h,l=axes[0,0].get_legend_handles_labels()
        fig.legend(h,[fa(s) for s in l],loc="upper right")
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix(".png"),dpi=150,facecolor="white",bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"),facecolor="white",bbox_inches="tight")
    plt.close(fig)


def core_figures(data, output):
    count=0
    for family in FAMILIES:
        for metric,(split,title,ylabel) in CORE.items():
            key = metric
            if metric.endswith("_train"):
                key = "balanced_accuracy"
            elif metric.endswith("_validation"):
                key = metric.removesuffix("_validation")
            spec=((split,key),title,ylabel)
            draw_six(data,output/family/(metric),family,spec,scale="zoom")
            count+=1
    return count


def loss_gradient_figures(data, output):
    count=0
    for family in FAMILIES:
        for obj in OBJECTIVES:
            short=OBJ[obj][0]
            for metric,title,ylabel in (
                ("loss","تابع زیان آموزش","Loss"),
                ("gradient_l2","هنجار گرادیان پارامترهای سیاست","||grad L||2")):
                for scale in ("zoom","log"):
                    stem=f"{family}__{metric}__{short}__{scale}"
                    draw_six(data,output/stem,family,
                             (("train",metric),title,ylabel),
                             objective=obj,scale=scale)
                    count+=1
    return count


def policy_figures(data,output):
    count=0
    for family in FAMILIES:
        for obj in OBJECTIVES:
            short=OBJ[obj][0]
            for key,(title,ylabel,kind) in POLICY.items():
                stem=f"{family}__{short}__{key}"
                draw_six(data,output/stem,family,
                         (key,title,ylabel),objective=obj,scale="zoom",policy=True)
                count+=1
    return count


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--part",choices=("core","loss","policy"),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    data=load(a.artifacts,a.dataset)
    if a.part=="core":
        n=core_figures(data,a.output)
    elif a.part=="loss":
        n=loss_gradient_figures(data,a.output)
    else:
        n=policy_figures(data,a.output)
    print("E14_PERSIAN_FIGURES_COMPLETE",a.part,n,"figure pairs")
if __name__=="__main__":
    main()
