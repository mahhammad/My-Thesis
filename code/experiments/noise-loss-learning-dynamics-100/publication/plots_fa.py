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

# A single dual-script font avoids missing glyph boxes (Persian, ASCII F0, BA).
# SVG text is converted to vector paths so browser font fallback cannot break it.
from math import floor, ceil, log10
from matplotlib.ft2font import FT2Font
import unicodedata
font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
font_chars = FT2Font(font_path).get_charmap()
def check_glyphs(text):
    missing = {f"U+{ord(ch):04X}" for ch in str(text)
               if ord(ch) >= 32 and unicodedata.category(ch) != "Cf"
               and ord(ch) not in font_chars}
    if missing:
        raise ValueError("Persian figure font misses glyphs: " + ", ".join(sorted(missing)))
    return text

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "path", "axes.unicode_minus": False})

def nice_bounds(low, high, percent=False):
    """One rounded scale per comparable figure group, including Bayes baseline.

    For example values 79..98 percent lead to limits 70..100, not 0..100.
    """
    low, high = float(low), float(high)
    if not np.isfinite([low, high]).all() or high < low:
        raise ValueError("Invalid chart limits")
    if percent:
        step = 10.
        lower = max(0., step * floor(low / step))
        upper = min(100., step * ceil(high / step))
        if upper <= lower:
            lower = max(0., lower-step)
            upper = min(100., upper+step)
        return (lower, upper)
    magnitude = max(high-low, max(abs(high), abs(low), 1e-9)*.025)
    target = magnitude/5.
    unit = 10**floor(log10(target))
    increment = next(c*unit for c in (1., 2., 2.5, 5., 10.)
                     if c*unit >= target)
    lower = floor((low-.04*magnitude)/increment)*increment
    upper = ceil((high+.04*magnitude)/increment)*increment
    if upper <= lower:
        upper += increment
    return lower, upper

# These are *observed MAP performance on the fixed base seed*, NOT hard
# upper bounds for any finite split. See chapter4.tex tab:ch4-base-bayes-split.
BAYES_MAP_PERCENT = {
    ("train", "accuracy"): 82.67,
    ("train", "balanced_accuracy"): 84.32,
    ("validation", "accuracy"): 96.67,
    ("validation", "balanced_accuracy"): 97.45,
}

OBJ = {"sigmoid_JS_margin_tau005": ("sigmoid", "سیگمویدی", "#265f9d"),
       "softplus_JS_tau005": ("softplus", "سافت‌پلاس", "#d77a38")}
FAM = {"shared": "سیاست مشترک", "class": "سیاست وابسته به کلاس"}

CORE = {
    "separation_i": ("train", "جدایی مراکز کلاس‌ها", "I"),
    "density_r_js": ("train", "تراکم درونی بر پایه واگرایی جنسن شنون", "R_JS"),
    "density_r_kl": ("train", "تراکم درونی بر پایه واگرایی کولبک لایبلر", "R_KL"),
    "density_r_sqrt_js": ("train", "تراکم درونی بر پایه ریشهٔ واگرایی جنسن شنون", "R_sqrt_JS"),
    "separation_t": ("train", "معیار جدایی مراکز", "T"),
    "separation_bt": ("train", "معیار جدایی مراکز", "BT"),
    "separation_tnl": ("train", "معیار جدایی مراکز", "TNL"),
    "separation_btnl": ("train", "معیار جدایی مراکز", "BTNL"),
    "margin_mean": ("train", "میانگین حاشیهٔ تصمیم", "Margin"),
    "accuracy_train": ("train", "دقت معمول آموزش", "Accuracy (%)"),
    "balanced_accuracy_train": ("train", "دقت متوازن آموزش", "BA (%)"),
    "balanced_accuracy_validation": ("validation", "دقت متوازن اعتبارسنجی", "BA (%)"),
    "accuracy_validation": ("validation", "دقت معمول اعتبارسنجی", "Accuracy (%)"),
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


def draw_six(data, out, family, specification, objective=None,
             scale="zoom", policy=False):
    kind, title, ylabel = specification[:3]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7),
                             sharex=True, layout="constrained")
    objectives = (objective,) if objective else OBJECTIVES
    comparable = {}
    all_group_values = []
    ref = (None if policy or not isinstance(kind, tuple)
           else BAYES_MAP_PERCENT.get(kind))
    percent = (not policy and isinstance(kind, tuple)
               and kind[1] in ("accuracy", "balanced_accuracy",
                               "precision_macro", "recall_macro"))
    for noise in COMPARABLE:
        sets = {}
        for obj in objectives:
            group = group_runs(data, family, obj, noise)
            if policy:
                series = np.array(
                    [[policy_at(run, row)[0][kind] for row in run["rows"]]
                     for run in group], dtype=float)
            else:
                split, key = kind
                series = np.stack([get_series(run, split, key)
                                   for run in group])
                if percent:
                    series *= 100.
            assert series.shape == (6, 101)
            assert np.isfinite(series).all(), (
                "Nonfinite metric trace", family, obj, noise, kind)
            sets[obj] = series
            all_group_values.append(series.ravel())
        comparable[noise] = sets

    # Each group (e.g. validation Accuracy, shared) has one shared y range
    # over all 6 noise panels, both objectives and their six subsets.
    group_values = np.concatenate(all_group_values)
    if ref is not None:
        group_values = np.append(group_values, ref)
    independent_scale = kind in (("train", "loss"),
                                  ("train", "gradient_l2"))
    common_bounds = (None if independent_scale else
                     nice_bounds(group_values.min(), group_values.max(),
                                 percent=percent))
    for noise, ax in zip(COMPARABLE, axes.flat):
        all_panel_values = []
        for obj in objectives:
            series = comparable[noise][obj]
            color = OBJ[obj][2]
            for trace in series:
                y = (np.where(trace > 0, trace, np.nan)
                     if scale == "log" else trace)
                ax.plot(np.arange(101), y, color=color, alpha=.19, lw=.8)
            mean = series.mean(axis=0)
            plotted_mean = (np.where(mean > 0, mean, np.nan)
                            if scale == "log" else mean)
            ax.plot(np.arange(101), plotted_mean, color=color,
                    lw=2.4, label=fa(OBJ[obj][1]))
            all_panel_values.extend(series.ravel())
        if ref is not None:
            ax.axhline(ref, color="#334155", ls="--", lw=1.6,
                       label=fa("مرجع بیزی"))
        values = np.array(all_panel_values, dtype=float)
        if scale == "log":
            positive = values[values > 0]
            assert positive.size, ("No positive values for logarithmic plot", kind)
            ax.set_yscale("log")
            ax.set_ylim(positive.min() / 1.15, positive.max() * 1.15)
        elif common_bounds is not None:
            ax.set_ylim(*common_bounds)
        else:
            lo, hi = float(values.min()), float(values.max())
            pad = max((hi-lo)*.1, abs(hi)*.015, 1e-7)
            ax.set_ylim(lo-pad, hi+pad)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(fa("نویز " + str(noise)))
        ax.set_xlabel(fa("گام آموزش"))
        ax.grid(alpha=.16)
        if noise in (1, 4):
            ax.set_ylabel(ylabel)
    family_name = FAM[family]
    objective_name = ("؛ زیان " + OBJ[objective][1]
                      if objective else "؛ دو تابع هدف")
    title_text = fa(title+"؛ "+family_name+objective_name)
    check_glyphs(title_text)
    fig.suptitle(title_text, fontsize=15)
    # Keep caption outside plot: Persian footers previously overlapped axes.
    if len(objectives) > 1 or ref is not None:
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150,
                facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"),
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def core_figures(data, output):
    count=0
    for family in FAMILIES:
        for metric,(split,title,ylabel) in CORE.items():
            key = metric
            if metric.endswith("_train"):
                key = metric.removesuffix("_train")
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
