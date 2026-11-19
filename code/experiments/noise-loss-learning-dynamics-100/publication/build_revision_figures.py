#!/usr/bin/env python3
"""E-CH4-14 thesis trend figures from frozen 101-step Train/Validation aggregates.

No model fitting, no Test access and no checkpoint selection.
Each curve is the descriptive mean of six *dependent* subsets. Only 1..6
noisy features qualify for six-subset comparisons. Zero-noise is a separate
reference in the main table; |A_N|=7 has a single setting.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "revision-trends"
AGG = ROOT / "all_metrics_101_steps.csv"
FAMILIES = ("shared", "class")
OBJECTIVES = ("sigmoid_JS_margin_tau005", "softplus_JS_tau005")
NOISE = range(1, 7)
STEPS = np.arange(101)
PALETTE = ("#285f96", "#ce7d29", "#27856d", "#8d4e93", "#b55b62", "#657a31")
SERIES = (
    ("shared", OBJECTIVES[0], "#285f96", "-", "سیاست مشترک، زیان سیگمویدی"),
    ("shared", OBJECTIVES[1], "#ce7d29", "--", "سیاست مشترک، زیان سافت‌پلاس"),
    ("class", OBJECTIVES[0], "#27856d", "-.", "سیاست وابسته به کلاس، زیان سیگمویدی"),
    ("class", OBJECTIVES[1], "#8d4e93", ":", "سیاست وابسته به کلاس، زیان سافت‌پلاس"),
)
GEOMETRY = (
    ("separation_i", r"$I$", "جدایی مراکز"),
    ("density_r_kl", r"$R_{\mathrm{KL}}$", "تراکم درونی مبتنی بر واگرایی کولبک–لایبلر"),
    ("density_r_js", r"$R_{\mathrm{JS}}$", "تراکم درونی مبتنی بر واگرایی جنسن–شنون"),
    ("density_r_sqrt_js", r"$R_{\sqrt{\mathrm{JS}}}$", "تراکم درونی مبتنی بر ریشهٔ واگرایی جنسن–شنون"),
    ("margin_mean", r"$\mathrm{Margin}$", "حاشیهٔ تصمیم"),
)
BAYES = {"train": 84.32, "validation": 97.45}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans", "font.size": 11,
})


def fa(text):
    return get_display(arabic_reshaper.reshape(str(text)))


def read_data():
    data = {}
    with AGG.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            family, obj = r["family"], r["objective"]
            noise, t = int(r["noise_count"]), int(r["step"])
            assert family in FAMILIES and obj in OBJECTIVES
            assert noise in NOISE and t in STEPS and int(r["subset_count"]) == 6
            key = (family, obj, noise, t, r["split"], r["metric"])
            assert key not in data
            vals = tuple(float(r[k]) for k in
                         ("mean", "minimum", "maximum", "descriptive_sd"))
            assert all(math.isfinite(v) for v in vals)
            assert vals[1] <= vals[0]+1e-10 <= vals[2]+1e-10
            data[key] = vals
    assert len(data) == 46056
    return data


def trace(data, family, obj, noise, split, metric):
    vals = np.array([data[family, obj, noise, int(t), split, metric]
                     for t in STEPS], dtype=float)
    assert vals.shape == (101, 4) and np.isfinite(vals).all()
    return vals


def format_axes(ax, ylabel, log=False):
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(fa("گام آموزش"))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.17)
    if log:
        ax.set_yscale("log")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    assert path.stat().st_size > 8000, path


def make_ba(data, split):
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.8), layout="constrained",
                            sharex=True, sharey=True)
    labels = []
    handles = []
    for noise, ax in zip(NOISE, axs.flat):
        for family, obj, color, ls, name in SERIES:
            vals = trace(data, family, obj, noise, split, "balanced_accuracy")
            artist, = ax.plot(STEPS, 100*vals[:, 0], lw=2.0,
                              ls=ls, color=color, label=fa(name))
            if noise == 1:
                handles.append(artist)
                labels.append(fa(name))
        bayes_artist = ax.axhline(BAYES[split], color="#3b4653",
                                 ls=(0, (5, 3)), lw=1.7,
                                 label=fa("مرجع بیزی"))
        if noise == 1:
            handles.append(bayes_artist)
            labels.append(fa("مرجع بیزی"))
        ax.set_title(fa("تعداد ویژگی نویزی: "+str(noise)))
        format_axes(ax, fa("دقت متوازن (درصد)"))
        ax.set_ylim(70, 100)
    fig.suptitle(fa("مسیر دقت متوازن "+("آموزش" if split=="train" else "اعتبارسنجی")),
                 fontsize=16)
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(.5, -.02))
    save(fig, "ba_"+split+"_combined.png")


def make_geometry(data, family, metric, symbol, description):
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.2), layout="constrained")
    labels = []
    handles = []
    for obj, ax in zip(OBJECTIVES, axs):
        for i, noise in enumerate(NOISE):
            vals = trace(data, family, obj, noise, "train", metric)
            artist, = ax.plot(STEPS, vals[:, 0], color=PALETTE[i],
                              lw=2.0, label=r"$|A_N|="+str(noise)+"$")
            if obj == OBJECTIVES[0]:
                handles.append(artist)
                labels.append(r"$|A_N|="+str(noise)+"$")
        ax.set_title(fa("زیان سیگمویدی" if obj == OBJECTIVES[0]
                        else "زیان سافت‌پلاس"), fontsize=12)
        format_axes(ax, symbol)
    fig.suptitle(fa(description+"؛ "+("سیاست مشترک" if family=="shared"
                                     else "سیاست وابسته به کلاس")), fontsize=15)
    fig.legend(handles, labels, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(.5, -.045), fontsize=11)
    save(fig, family+"__"+metric+".png")


def make_loss(data, family, obj):
    fig, ax = plt.subplots(figsize=(10.5, 5.4), layout="constrained")
    for i, noise in enumerate(NOISE):
        vals = trace(data, family, obj, noise, "train", "loss")
        assert np.all(vals[:, 0] > 0)
        ax.plot(STEPS, vals[:, 0], color=PALETTE[i], lw=2.1,
                label=r"$|A_N|="+str(noise)+"$")
    format_axes(ax, r"$\mathcal{L}$", log=True)
    fig.suptitle(fa("تابع زیان آموزش؛ " +
                    ("سیاست مشترک" if family=="shared" else "سیاست وابسته به کلاس")+
                    "؛ " + ("زیان سیگمویدی" if obj==OBJECTIVES[0]
                             else "زیان سافت‌پلاس")), fontsize=15)
    ax.legend(ncol=3, loc="best", frameon=False)
    save(fig, family+"__"+("sigmoid" if obj==OBJECTIVES[0] else "softplus")+
         "__loss_log.png")


def make_gradient(data, family, obj):
    fig, axs = plt.subplots(2, 3, figsize=(15.4, 8.8),
                            layout="constrained", sharex=True)
    for noise, ax in zip(NOISE, axs.flat):
        vals = trace(data, family, obj, noise, "train", "gradient_l2")
        assert np.all(vals[:, 0] > 0)
        ax.plot(STEPS, vals[:, 0], lw=2.2, color=PALETTE[noise-1],
                label=fa("میانگین شش زیرمجموعه"))
        low = np.maximum(vals[:, 1], 1e-12)
        high = np.maximum(vals[:, 2], low)
        ax.fill_between(STEPS, low, high, color=PALETTE[noise-1],
                        alpha=.13, lw=0)
        ax.set_title(fa("تعداد ویژگی نویزی: "+str(noise)))
        format_axes(ax, r"$\|\nabla_Z \mathcal{L}\|_2$", log=True)
    fig.suptitle(fa("هنجار گرادیان آموزش؛ " +
                    ("سیاست مشترک" if family=="shared" else "سیاست وابسته به کلاس")+
                    "؛ " + ("زیان سیگمویدی" if obj==OBJECTIVES[0]
                             else "زیان سافت‌پلاس")), fontsize=15)
    save(fig, family+"__"+("sigmoid" if obj==OBJECTIVES[0] else "softplus")+
         "__gradient_log.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = read_data()
    for split in ("train","validation"):
        for family in FAMILIES:
            for obj in OBJECTIVES:
                for noise in NOISE:
                    ba=trace(data,family,obj,noise,split,"balanced_accuracy")
                    recall=trace(data,family,obj,noise,split,"recall_macro")
                    assert np.allclose(ba[:, 0], recall[:, 0], atol=1e-12)
    if not args.write:
        print("E14_REVISION_FIGURES_DATA_PASS: 20 figures / 101 states")
        return
    for split in ("train", "validation"):
        make_ba(data,split)
    for metric,symbol,description in GEOMETRY:
        for family in FAMILIES:
            make_geometry(data,family,metric,symbol,description)
    for family in FAMILIES:
        for obj in OBJECTIVES:
            make_loss(data,family,obj)
            make_gradient(data,family,obj)
    assert len(list(OUT.glob("*.png"))) == 20
    print("E14_REVISION_FIGURES_COMPLETE: 20 figures; NO FIT, NO TEST")


if __name__ == "__main__":
    main()
