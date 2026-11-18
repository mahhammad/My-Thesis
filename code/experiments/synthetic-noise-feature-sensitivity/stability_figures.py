"""Render four postfit E14 figures from derived tables only (NO FIT)."""
from __future__ import annotations
import argparse,csv,json,io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
TABLE=HERE/"tables"
FIG=HERE/"figures"
plt.rcParams.update({"figure.dpi":120,"svg.fonttype":"path",
                     "svg.hashsalt":"e14-geometry-preference-v1"})


def rows(name):
    with (TABLE/name).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def chart(fig):
    file=io.BytesIO()
    fig.savefig(file,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    return file.getvalue()


def build():
    geo=rows("geometry_stability_by_noise_level_60.csv")
    pref=rows("feature_preference_by_noise_level_60.csv")
    runs=rows("geometry_stability_by_run_60.csv")
    colors=plt.get_cmap("tab10").colors
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for col,fam in enumerate(("shared","class")):
        group=sorted((r for r in geo if r["family"]==fam),key=lambda r:int(r["noise_count"]))
        xx=np.arange(6)
        for row,metric in enumerate(("mean_center_drift","mean_abs_pairwise_shift")):
            ax=axes[row,col]
            for label,suffix,style in (("mean","mean","-"),("min","min","--"),("max","max","--")):
                ax.plot(xx,[float(r[metric+"_"+suffix]) for r in group],
                        style,marker="o" if label=="mean" else None,
                        label=label)
            ax.set(title=fam+" / "+metric.replace("_"," "),xlabel="Number of noise columns",
                   ylabel="sqrt-JSD distance")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Center movement and pairwise geometry change vs frozen no-noise policy")
    fig.tight_layout()
    f1=chart(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for col,fam in enumerate(("shared","class")):
        group=sorted((r for r in geo if r["family"]==fam),key=lambda r:int(r["noise_count"]))
        xx=np.arange(6)
        for row,metric in enumerate(("delta_I","validation_R_JS_delta")):
            ax=axs[row,col]
            ax.axhline(0,ls="--",color=".4",lw=1)
            ax.plot(xx,[float(r[metric+"_mean"]) for r in group],marker="o",color=colors[row])
            ax.set(title=f"{fam}: {metric}",xlabel="Number of noise columns")
            ax.grid(alpha=.2)
    fig.suptitle("Separation and Validation density shift relative to k=0")
    fig.tight_layout()
    f2=chart(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for row,fam in enumerate(("shared","class")):
        for col,h in enumerate((1,2,3)):
            ax=axs[row,col]
            group=sorted((r for r in pref if r["family"]==fam and int(r["h"])==h),
                         key=lambda r:int(r["noise_count"]))
            xx=np.arange(6)
            for j,feature in enumerate((0,1,3)):
                ax.plot(xx,[float(r[f"feature_{feature}_conditional_share_mean"]) for r in group],
                        color=colors[j],marker="o",label=f"feature {feature}")
            ax.set(title=f"{fam}, horizon {h}",xlabel="Number of noise columns",
                   ylabel="Conditional main-feature share")
            ax.set_ylim(0,1)
            ax.grid(alpha=.2)
            if col==0:ax.legend(fontsize=8)
    fig.suptitle("Informative-feature preference profiles conditional on staying informative")
    fig.tight_layout()
    f3=chart(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for row,fam in enumerate(("shared","class")):
        for col,h in enumerate((1,2,3)):
            ax=axs[row,col]
            group=sorted((r for r in pref if r["family"]==fam and int(r["h"])==h),
                         key=lambda r:int(r["noise_count"]))
            xx=np.arange(6)
            yy=[float(r["main_conditional_TV_mean"]) for r in group]
            ax.plot(xx,yy,marker="o",color=colors[0],label="conditional TV")
            ax.set(title=f"{fam}, horizon {h}",xlabel="Number of noise columns",
                   ylabel="Conditional main-feature TV",ylim=(0,.6))
            ax.grid(alpha=.2)
            twin=ax.twinx()
            twin.plot(xx,[float(r["rank_reversal_count_mean"]) for r in group],
                      color=colors[1],marker="x",ls="--",label="pair reversals")
            twin.set_ylim(-.15,3.15)
            if col==2:twin.set_ylabel("Pairwise rank reversals (of 3)")
    fig.suptitle("Relative preference drift and rank order under added noise")
    fig.tight_layout()
    f4=chart(fig)
    rep=rows("representation_drift_by_noise_level_60.csv")
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharex=True,sharey=True)
    for col,split in enumerate(("train","validation")):
        ax=axs[col]
        for j,fam in enumerate(("shared","class")):
            group=sorted((r for r in rep if r["split"]==split and r["family"]==fam),
                         key=lambda r:int(r["noise_count"]))
            xx=np.arange(6)
            yy=np.array([float(r["mean"]) for r in group])
            low=np.array([float(r["min"]) for r in group])
            high=np.array([float(r["max"]) for r in group])
            ax.plot(xx,yy,marker="o",label=fam,color=colors[j])
            ax.fill_between(xx,low,high,alpha=.12,color=colors[j])
        ax.set(title=split.capitalize(),xlabel="Number of noise columns",
               ylabel="Mean per-sample sqrt-JSD to k=0",ylim=(0,.035))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Exact representation movement from frozen step-60 checkpoints")
    fig.tight_layout()
    f5=chart(fig)
    return {"39_center_geometry_stability_60.svg":f1,
            "40_center_separation_density_deltas_60.svg":f2,
            "41_main_feature_preferences_h1_h3_60.svg":f3,
            "42_feature_rank_vs_relative_drift_60.svg":f4,
            "43_exact_representation_drift_60.svg":f5}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("check","write"),default="check")
    a=p.parse_args()
    for file,content in build().items():
        target=FIG/file
        if a.mode=="write":
            target.write_bytes(content)
        else:
            assert target.is_file() and target.read_bytes()==content,target
    print("E14_STABILITY_FIGURES_PASS")
