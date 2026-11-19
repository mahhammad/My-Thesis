"""Final t=100 selection shares/rankings, original feature IDs from frozen manifest.

No training; ranks represent unmasked feature-selection frequency, not causal
predictive feature importance. BA equality does not imply identical predictions.
"""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from source import load, policy_at, policy_metrics, write_table, PRIMARY, FAMILIES, OBJECTIVES, COMPARABLE
from plots_fa import fa, OBJ, FAM

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"path"})


def rankings(data):
    detail=[]
    macro=[]
    meta=[]
    for run_id,run in sorted(data.items()):
        row=run["rows"][100]
        old=run["feature_indices"]
        fam,obj = run["family"],run["objective"]
        hypotheses=row["policy"]["unmasked"]
        shares_list=[]
        for i,h in enumerate(hypotheses):
            metrics,s=policy_metrics(h["p0"],h["P"],old)
            shares_list.append(s)
            order=sorted(range(len(s)),key=lambda j:(-s[j],old[j]))
            ranks={j:n+1 for n,j in enumerate(order)}
            for j,feature in enumerate(old):
                detail.append(dict(run_id=run_id,dataset_id=run["dataset_id"],
                    objective=obj,family=fam,noise_count=run["noise_count"],
                    hypothesis=("shared" if fam=="shared" else "class_"+str(i)),
                    original_feature_id=feature,feature_type=("informative" if feature in PRIMARY else "noise"),
                    share_s_a_3=float(s[j]),expected_visits_3=float(3*s[j]),
                    rank_3=ranks[j],uniform_share=1/len(old),
                    validation_BA=float(row["validation"]["balanced_accuracy"])))
        mean=np.mean(shares_list,axis=0)
        order=sorted(range(len(mean)),key=lambda j:(-mean[j],old[j]))
        ranks={j:n+1 for n,j in enumerate(order)}
        for j,feature in enumerate(old):
            macro.append(dict(run_id=run_id,dataset_id=run["dataset_id"],
                objective=obj,family=fam,noise_count=run["noise_count"],
                original_feature_id=feature,
                feature_type=("informative" if feature in PRIMARY else "noise"),
                share_s_a_3=float(mean[j]),expected_visits_3=float(3*mean[j]),
                rank_3=ranks[j],uniform_share=1/len(old),
                validation_BA=float(row["validation"]["balanced_accuracy"])))
        meta.append(dict(run_id=run_id,dataset_id=run["dataset_id"],family=fam,
                         objective=obj,noise_count=run["noise_count"],
                         validation_BA=float(row["validation"]["balanced_accuracy"]),
                         validation_accuracy=float(row["validation"]["accuracy"])))
    return detail,macro,meta


def pairs_table(macro,meta):
    m={}
    for row in macro:
        m.setdefault(row["run_id"],{})[row["original_feature_id"]]=row
    out=[]
    for family in FAMILIES:
        for obj in OBJECTIVES:
            for noise in COMPARABLE:
                runs=[r for r in meta if (r["family"],r["objective"],r["noise_count"])==(family,obj,noise)]
                assert len(runs)==6
                for r1,r2 in combinations(runs,2):
                    a,b=m[r1["run_id"]],m[r2["run_id"]]
                    common=sorted(set(a)&set(b))
                    inv=0
                    for f,g in combinations(common,2):
                        before=(a[f]["share_s_a_3"]-a[g]["share_s_a_3"])
                        after=(b[f]["share_s_a_3"]-b[g]["share_s_a_3"])
                        if before*after < 0:
                            inv+=1
                    same=abs(r1["validation_BA"]-r2["validation_BA"])<1e-12
                    out.append(dict(family=family,objective=obj,noise_count=noise,
                        run_a=r1["run_id"],run_b=r2["run_id"],
                        validation_BA_a=r1["validation_BA"],validation_BA_b=r2["validation_BA"],
                        common_features=len(common),inversions_among_common=inv,
                        same_BA=same,same_BA_different_rank=(same and inv>0)))
    assert len(out)==360
    return out


def draw_rank(macro,root,family,obj):
    rows=[r for r in macro if (r["family"],r["objective"])==(family,obj)
          and r["noise_count"] in COMPARABLE]
    ds=sorted(set(r["dataset_id"] for r in rows),key=lambda k:(int(k[1:k.index("-")]),int(k.split("-")[-1])))
    assert len(ds)==36
    ar=np.full((len(ds),10),np.nan)
    ba={}
    for i,dataset in enumerate(ds):
        for r in rows:
            if r["dataset_id"]==dataset:
                ar[i,r["original_feature_id"]]=r["rank_3"]
                ba[dataset]=r["validation_BA"]
    fig,ax=plt.subplots(figsize=(13,12.5),layout="constrained")
    import matplotlib.colors as colors
    cmap=plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#eceff2")
    heat=ax.imshow(np.ma.masked_invalid(ar),cmap=cmap,vmin=1,vmax=10,aspect="auto")
    ax.set_xticks(range(10),["F"+str(x) for x in range(10)])
    ax.set_yticks(range(len(ds)),[d+" | "+f"{100*ba[d]:.2f}%" for d in ds],fontsize=8)
    ax.set_xlabel(fa("شناسهٔ ویژگی در دادهٔ مادر"))
    ax.set_ylabel(fa("زیرمجموعه و دقت متوازن اعتبارسنجی"))
    for i in range(len(ds)):
        for j in range(10):
            if np.isfinite(ar[i,j]):
                ax.text(j,i,str(int(ar[i,j])),ha="center",va="center",fontsize=7,
                        color="white" if ar[i,j]>6 else "#101827")
    fig.colorbar(heat,ax=ax,shrink=.73,label=fa("رتبه؛ یک یعنی بیشترین سهم"))
    ax.set_title(fa("رتبهٔ انتخاب هر ویژگی در گام ۱۰۰؛ "+FAM[family]+"؛ زیان "+OBJ[obj][1]))
    fig.savefig(root.with_suffix(".png"),dpi=155,bbox_inches="tight")
    fig.savefig(root.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)


def draw_shares(macro,root,family,obj):
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,layout="constrained")
    rows=[r for r in macro if r["family"]==family and r["objective"]==obj]
    for noise,ax in zip(COMPARABLE,axes.flat):
        subset=[r for r in rows if r["noise_count"]==noise]
        vals=[];numbers=[]
        for feature in range(10):
            have=[r["share_s_a_3"] for r in subset if r["original_feature_id"]==feature]
            vals.append(np.mean(have) if have else np.nan)
            numbers.append(len(have))
        for feature,v in enumerate(vals):
            if np.isfinite(v):
                ax.bar(feature,v,width=.75,
                       color=("#265f9d" if feature in PRIMARY else "#d77a38"),alpha=.83)
                ax.text(feature,v+max(.001,abs(v)*.035),str(numbers[feature]),
                        ha="center",fontsize=8)
        ax.axhline(1/(3+noise),ls="--",lw=1,color="#444b59")
        ax.set_ylim(0,max([x for x in vals if np.isfinite(x)]+[1/(3+noise)])*1.23)
        ax.set_xticks(range(10),["F"+str(x) for x in range(10)],fontsize=8)
        ax.grid(axis="y",alpha=.14)
        ax.set_title(fa("تعداد نویز: "+str(noise)))
        if noise in (1,4):ax.set_ylabel("s_a^(3)")
        ax.set_xlabel(fa("شناسهٔ ویژگی"))
    fig.suptitle(fa("میانگین سهم انتخاب ویژگی‌های حاضر؛ "+FAM[family]+"؛ زیان "+OBJ[obj][1]))
    # The Persian explanatory caption is in README_FA.md, not on top of axes.
    fig.savefig(root.with_suffix(".png"),dpi=155,bbox_inches="tight")
    fig.savefig(root.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    data=load(a.artifacts,a.dataset)
    detail,macro,meta=rankings(data)
    pairs=pairs_table(macro,meta)
    write_table(a.output/"final_per_feature_by_hypothesis_t100.csv",detail,tuple(detail[0]))
    write_table(a.output/"final_per_feature_macro_t100.csv",macro,tuple(macro[0]))
    write_table(a.output/"final_run_overview_t100.csv",meta,tuple(meta[0]))
    write_table(a.output/"pairwise_common_feature_rank_t100.csv",pairs,tuple(pairs[0]))
    for family in FAMILIES:
        for obj in OBJECTIVES:
            stem=f"{family}__{OBJ[obj][0]}"
            draw_rank(macro,a.output/(stem+"__final_ranks"),family,obj)
            draw_shares(macro,a.output/(stem+"__final_shares"),family,obj)
    count_equal=sum(p["same_BA"] for p in pairs)
    count_changed=sum(p["same_BA_different_rank"] for p in pairs)
    assert count_equal>=0 and count_changed>=0
    print("E14_FEATURE_RANK_COMPLETE",len(macro),count_equal,count_changed,8)
if __name__=="__main__":
    main()
