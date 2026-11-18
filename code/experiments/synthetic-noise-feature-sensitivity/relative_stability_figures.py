"""Descriptive all-k relative postfit E14 figures; no model evaluation or fit."""
from __future__ import annotations
import argparse
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
TABLE=HERE/"tables"
FIG=HERE/"figures"
plt.rcParams.update({"figure.dpi":120, "font.size":9,
                     "svg.fonttype":"path","svg.hashsalt":"e14-relative-v1"})

def rows(filename):
    with (TABLE/filename).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

def render(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    return buf.getvalue()

def build():
    g=rows("relative_geometry_by_noise_level_60.csv")
    a=rows("feature_allocation_percent_by_level_60.csv")
    assert len(g)==len(a)==12
    families=("shared","class")
    out={}

    fig,axs=plt.subplots(3,2,figsize=(11,10),sharex=True)
    items=[("mean_pairwise_pct_change","Mean pairwise sqrt-JSD distance"),
           ("I_pct_change","Center separation I"),
           ("validation_R_JS_pct_change","Validation within-class R_JS")]
    for j,fam in enumerate(families):
        gg=sorted((r for r in g if r["family"]==fam),key=lambda x:int(x["noise_count"]))
        xx=np.arange(6)
        for i,(metric,name) in enumerate(items):
            ax=axs[i,j]
            y=np.array([float(r[metric+"_mean"]) for r in gg])
            low=np.array([float(r[metric+"_min"]) for r in gg])
            hi=np.array([float(r[metric+"_max"]) for r in gg])
            ax.plot(xx,y,marker="o",lw=1.9)
            ax.fill_between(xx,low,hi,alpha=.15)
            ax.axhline(0,ls="--",color=".4",lw=.9)
            ax.set(title=f"{fam} / {name}",ylabel="Change vs k=0 (%)",
                   xticks=xx)
            if i==2:ax.set_xlabel("Number of noise features")
            ax.grid(alpha=.2)
    fig.suptitle("Geometry and density: relative changes in the same-family step-60 reference")
    fig.tight_layout()
    out["47_relative_center_geometry_percent_60.svg"]=render(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharex=True)
    items=[("center_movement_pct_ref_spacing",
            "Center movement / reference spacing"),
           ("mean_abs_pairwise_shift_pct_ref",
            "Absolute pairwise change / reference spacing")]
    for j,fam in enumerate(families):
        gg=sorted((r for r in g if r["family"]==fam),key=lambda x:int(x["noise_count"]))
        xx=np.arange(6)
        for metric,name in items:
            axs[j].plot(xx,[float(r[metric+"_mean"]) for r in gg],marker="o",
                        label=name)
        axs[j].set(title=fam,xticks=xx,xlabel="Number of noise features",
                   ylabel="Percent of k=0 mean center-pair spacing",ylim=(0,None))
        axs[j].grid(alpha=.2)
        axs[j].legend(fontsize=7)
    fig.suptitle("Normalized geometric displacement: NOT a percentage change of center coordinates")
    fig.tight_layout()
    out["48_center_movement_percent_of_spacing_60.svg"]=render(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharex=True,sharey=True)
    for j,fam in enumerate(families):
        pp=sorted((r for r in a if r["family"]==fam),key=lambda x:int(x["noise_count"]))
        xx=np.arange(6)
        for h in (1,2,3):
            axs[j].plot(xx,[float(r[f"informative_share_h{h}_pct"]) for r in pp],
                        marker="o",lw=1.7,label=f"h={h}")
        axs[j].plot(xx,[100*3/(3+k) for k in range(6)],ls="--",
                    color=".35",label="Uniform informative")
        axs[j].set(title=fam,xlabel="Number of noise features",
                   ylabel="All informative features' selection share (%)",
                   xticks=xx,ylim=(0,105))
        axs[j].grid(alpha=.2)
        axs[j].legend()
    fig.suptitle("Policy mass on all 3 informative features (unmasked chain, horizons 1-3)")
    fig.tight_layout()
    out["49_informative_mass_h1_h3_all_k_60.svg"]=render(fig)

    fig,axs=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for j,fam in enumerate(families):
        aa=sorted((r for r in a if r["family"]==fam),key=lambda x:int(x["noise_count"]))
        gg=sorted((r for r in g if r["family"]==fam),key=lambda x:int(x["noise_count"]))
        xx=np.arange(6)
        for i,(metric,name) in enumerate(
            (("gradient_RMS_30_60_pct_change","Gradient RMS (steps 30-60)"),
             ("BA_TV_0_60_pct_change","Validation BA total variation (steps 0-60)"))):
            y=[float(r[metric]) for r in aa] if i==0 else [
                float(r[metric+"_mean"]) for r in gg]
            axs[i,j].plot(xx,y,marker="o")
            axs[i,j].axhline(0,color=".4",ls="--",lw=.9)
            axs[i,j].set(title=f"{fam} / {name}",ylabel="Change from k=0 (%)",
                          xlabel="Number of noise features",xticks=xx)
            axs[i,j].grid(alpha=.2)
    fig.suptitle("Relative learning-variability diagnostics; not evidence of Softmax causality")
    fig.tight_layout()
    out["50_gradient_BA_variability_relative_60.svg"]=render(fig)
    return {FIG/k:v for k,v in out.items()}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("write","check"),default="check")
    mode=p.parse_args().mode
    for path,data in build().items():
        if mode=="write":path.write_bytes(data)
        else:assert path.is_file() and path.read_bytes()==data,path
    print("E14_RELATIVE_FIGURES_PASS")
