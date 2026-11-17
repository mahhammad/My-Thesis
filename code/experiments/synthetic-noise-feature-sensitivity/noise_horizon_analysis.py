"""E-CH4-14: replay exact §3.8 noise-exposure metrics at h=1,2,3,4.

Uses frozen float64 policy logits at cumulative steps 30..60. No model
training, raw-file rewrites, Test access, or extrapolation to 0..29.
Legacy F3/E3/NFI3 are verified against independent §3.8 calculations.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import pathlib
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from continuation_analysis import audit, read_json, read_jsonl, require

HERE=pathlib.Path(__file__).resolve().parent
RAW=HERE/"results/continuation_30_60"
TABLES=HERE/"tables"
FIGURES=HERE/"figures"
USABLE={0,1,3}
FAMILIES=("shared","class")
HORIZONS=(1,2,3,4)
COLORS=plt.get_cmap("tab10").colors
plt.rcParams.update({"font.size":9,"figure.dpi":110,
                     "svg.fonttype":"path","svg.hashsalt":"e14-ch3-horizons-v1"})

def softmax(logits):
    z=np.asarray(logits,dtype=np.float64)
    require(z.ndim==2 and np.isfinite(z).all(),"Bad float64 logits")
    e=np.exp(z-z.max(axis=1,keepdims=True))
    return e/e.sum(axis=1,keepdims=True)

def derived(logits,mapping,h):
    """Exact unmasked-chain formulas §3.8 for a single fixed policy."""
    m=[int(v) for v in mapping]
    d=len(m);k=d-3
    require(h in HORIZONS and d>=3 and len(set(m))==d,"Bad horizon/mapping")
    J=np.asarray([i for i,v in enumerate(m) if v in USABLE],dtype=int)
    N=np.asarray([i for i,v in enumerate(m) if v not in USABLE],dtype=int)
    require(len(J)==3 and len(N)==k,"Feature sets do not partition mapping")
    require(np.asarray(logits).shape==(d+1,d),"Wrong shared policy shape")
    probs=softmax(logits)
    p0=probs[d,:];P=probs[:d,:]
    visits=[p0]
    for _ in range(1,h):visits.append(visits[-1]@P)
    marginals=[float(v[N].sum()) for v in visits]
    nbar=float(sum(marginals))
    hit=(1.0-float(p0[J]@np.linalg.matrix_power(P[np.ix_(J,J)],h-1)
                   @np.ones(3))) if k else 0.0
    hit=float(np.clip(hit,0,1))
    s=nbar/h
    r=k/d
    uniform_hit=1-(3/d)**h
    result={"p_hit":hit,"nbar":nbar,"s":s,
            "NSR":(s/r if k else None),
            "NFI":(hit/uniform_hit if k else None),
            "uniform_p_hit":uniform_hit,"uniform_nbar":h*r,
            "uniform_s":r,"q_last":marginals[-1]}
    features=[{"feature":v,"is_noise":v not in USABLE,
              "nbar":float(sum(p[i] for p in visits)),
              "selection_share":float(sum(p[i] for p in visits))/h}
              for i,v in enumerate(m)]
    require(abs(sum(v["nbar"] for v in features)-h)<1e-11,
            "Per-feature expected count does not sum to horizon")
    require(abs(sum(v["nbar"] for v in features if v["is_noise"])-nbar)<1e-11,
            "Noise set not equal to sum of its per-feature shares")
    require(0<=hit<=1 and 0<=s<=1+1e-12 and
            hit<=nbar+1e-12 and nbar<=h*hit+1e-12,
            "Probability/count bound violated")
    if not k:
        require(hit==nbar==s==uniform_hit==0 and
                result["NSR"] is None and result["NFI"] is None,
                "No-noise undefined ratios must not be set to zero")
    return result,features,marginals

def class_summaries(logits,mapping,h,family):
    raw={"shared":derived(logits,mapping,h)} if family=="shared" else {
        str(c):derived(logits[c],mapping,h) for c in range(3)}
    labels=list(raw)
    scalars={key:(float(np.mean([raw[c][0][key] for c in labels]))
                  if all(raw[c][0][key] is not None for c in labels) else None)
             for key in raw[labels[0]][0]}
    feats=[]
    for idx in range(len(mapping)):
        f=raw[labels[0]][1][idx]
        feats.append({"feature":f["feature"],"is_noise":f["is_noise"],
             "nbar":float(np.mean([raw[c][1][idx]["nbar"] for c in labels])),
             "selection_share":float(np.mean(
                 [raw[c][1][idx]["selection_share"] for c in labels]))})
    return scalars,feats,{c:r[0] for c,r in raw.items()}

def validate_legacy(vals,original,h,run):
    if h!=3:return
    for key,old_key in (("p_hit","F3"),("nbar","E3"),("NFI","NFI3"),
                        ("q_last","q3")):
        new=vals[key];old=original[old_key]
        if new is None:
            require(old is None,"Undefined old NFI3 should remain null "+run)
        else:
            require(old is not None and np.isclose(new,float(old),
                    atol=2e-11,rtol=2e-9),
                    "Three-step legacy mismatch "+run+" "+key)

def csv_data(rows):
    require(len(rows)>0,"Empty report")
    s=io.StringIO(newline="")
    w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows)
    return s.getvalue().encode("utf8")

def jbytes(obj):return (json.dumps(obj,ensure_ascii=False,indent=2,
                                  sort_keys=True)+"\n").encode("utf8")

def group(rows):
    groups=[]
    for family in FAMILIES:
      for k in range(6):
       for h in HORIZONS:
        selected=[r for r in rows if r["family"]==family and
                  r["noise_count"]==k and r["h"]==h]
        require(len(selected)==(1 if k in (0,5) else 4),
                "Group does not contain prescribed dependent subsets")
        rec={"family":family,"noise_count":k,"h":h,
             "n_subsets":len(selected),"uniform_s":k/(k+3),
             "uniform_p_hit":1-(3/(k+3))**h}
        for key in ("p_hit","nbar","s","NSR","NFI"):
            for step in (30,60):
                arr=[r[f"{key}_{step}"] for r in selected]
                rec[f"{key}_{step}_mean"]=(float(np.mean(arr)) if
                     all(x is not None for x in arr) else None)
                rec[f"{key}_{step}_min"]=(float(min(arr)) if
                     all(x is not None for x in arr) else None)
                rec[f"{key}_{step}_max"]=(float(max(arr)) if
                     all(x is not None for x in arr) else None)
        groups.append(rec)
    require(len(groups)==48,"Expected 12 policy/noise groups and four horizons")
    return groups

def svg(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    return buf.getvalue()

def figures(groups,timesteps):
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex="col")
    for j,family in enumerate(FAMILIES):
        for i,metric in enumerate(("s","NSR")):
            ax=axes[i,j]
            for h in HORIZONS:
                vals=[next(r for r in groups if r["family"]==family and
                           r["noise_count"]==k and r["h"]==h)
                      for k in range(6)]
                y=[r[f"{metric}_60_mean"] if r[f"{metric}_60_mean"] is not None
                   else np.nan for r in vals]
                ax.plot(range(6),y,color=COLORS[h-1],lw=1.9,
                        marker="o",label=f"h={h}")
            if metric=="s":
                ax.plot(range(6),[k/(k+3) for k in range(6)],
                        color=".35",ls="--",lw=1.35,label="Uniform")
                ax.set_ylim(0,.85)
            else:
                ax.axhline(1,color=".35",ls="--",lw=1.35,label="Uniform = 1")
                ax.set_ylim(0,2.2)
            ax.set(title=f"{family}: "+("noise selection share" if i==0 else
                       "normalized share / Uniform"),xticks=list(range(6)),
                   xlabel="|A_N|",ylabel=("s_N^(h)" if i==0 else "NSR_N^(h)"))
            ax.grid(alpha=.18)
            if j==0:ax.legend(fontsize=7,ncol=3)
    fig.suptitle("Exact unmasked noise exposure at step 60: horizons 1, 2, 3, 4")
    fig.tight_layout()
    first=svg(fig)
    fig,axs=plt.subplots(6,2,figsize=(13,18),sharex=True,sharey="row")
    for k in range(6):
      for j,family in enumerate(FAMILIES):
        ax=axs[k,j]
        rows=[r for r in timesteps if r["family"]==family and r["noise_count"]==k]
        for h in HORIZONS:
            values=[]
            for step in range(30,61):
                rr=[r["s"] for r in rows if r["h"]==h and r["step"]==step]
                require(len(rr)==(1 if k in (0,5) else 4),"Timeline incomplete")
                values.append(float(np.mean(rr)))
            ax.plot(range(30,61),values,color=COLORS[h-1],
                    lw=1.6,label=f"h={h}")
        ax.axhline(k/(k+3),color=".30",ls="--",lw=1.15,label="Uniform")
        ax.set(title=f"{family}, |A_N|={k}",xlim=(30,60))
        ax.grid(alpha=.17)
        if j==0:ax.set_ylabel("s_N^(h)")
        if k==5:ax.set_xlabel("Adam update 30..60")
        if k==0 and j==0:ax.legend(fontsize=7,ncol=5)
    fig.suptitle("Unmasked noise selection share: observed steps 30 to 60")
    fig.tight_layout()
    return {"37_noise_horizon_normalized_reference.svg":first,
            "38_noise_horizon_paths_30_60.svg":svg(fig)}

def build():
    old,traces,_,_,count=audit()
    require(len(traces)==36 and count.get("UPDATE")==1080,
            "Full frozen 60-step ledger failed")
    meta=read_json(HERE/"inputs/derived/manifest.json")
    datasets={x["subset_id"]:x for x in meta["datasets"]}
    require(len(datasets)==18,"Expected eighteen dependent feature subsets")
    timeline=[];last_run=[];byfeature=[];rawclass=[]
    for run in sorted(traces):
        subset,family=run.split("__")
        info=datasets[subset]
        k=int(info["k"]);mapping=info["feature_mapping"]
        matrices=read_jsonl(RAW/run/"logits_30_60.jsonl")
        metrics=read_jsonl(RAW/run/"trace_30_60.jsonl")
        require(len(matrices)==len(metrics)==31,"Missing checkpoint steps")
        local={}
        for log,ob in zip(matrices,metrics):
            step=int(log["step"])
            require(step==ob["step"] and 30<=step<=60,"Misaligned logits")
            for h in HORIZONS:
                vals,features,perclass=class_summaries(
                    np.asarray(log["logits"],dtype=np.float64),
                    mapping,h,family)
                if h==3:
                    legacy=ob["nfi3"]["macro" if family=="class" else "shared"]
                    validate_legacy(vals,legacy,h,run)
                    require(abs(vals["q_last"]-float(legacy["q3"]))<1e-12,
                            "q3 mismatch")
                row={"run_id":run,"family":family,"noise_count":k,
                     "step":step,"h":h,**vals}
                timeline.append(row)
                if step in (30,60):
                    local[(step,h)]=vals
                if step==60:
                    for v in features:
                        byfeature.append({"run_id":run,"family":family,
                                          "noise_count":k,"h":h,**v})
                    for label,v in perclass.items():
                        rawclass.append({"run_id":run,"family":family,
                                         "policy":label,"noise_count":k,"h":h,
                                         **v})
        for h in HORIZONS:
            a,b=local[(30,h)],local[(60,h)]
            last_run.append({"run_id":run,"family":family,"noise_count":k,
                "h":h,**{f"{key}_{step}":v[key] for step,v in ((30,a),(60,b))
                        for key in ("p_hit","nbar","s","NSR","NFI")}})
    require(len(timeline)==36*31*4 and len(last_run)==36*4,
            "Incorrect timeline/group sample coverage")
    groups=group(last_run)
    require(len(rawclass)==(18+18*3)*4,
            "Policy-conditional summary coverage wrong")
    # Signed relative comparisons are meaningful only for nonzero noise.
    requires=[r for r in last_run if r["noise_count"]==0]
    require(all(r["NSR_60"] is None and r["NFI_60"] is None
                for r in requires),"Undefined no-noise ratios were filled")
    def equal(a,b):return abs(a-b)<=2e-11
    for r in timeline:
        if r["h"]==1:
            require(equal(r["p_hit"],r["nbar"]) and
                    equal(r["nbar"],r["s"]) and
                    equal(r["s"],r["q_last"]),
                    "Horizon-one identity invalid")
        if r["noise_count"]>0:
            require(equal(r["NSR"]*r["uniform_s"],r["s"]),
                    "NSR normalization mismatch")
    outputs={
      TABLES/"noise_horizon_by_step_30_60.csv":csv_data(timeline),
      TABLES/"noise_horizon_by_run_30_60.csv":csv_data(last_run),
      TABLES/"noise_horizon_by_group_30_60.csv":csv_data(groups),
      TABLES/"noise_horizon_by_feature_60.csv":csv_data(byfeature),
      TABLES/"noise_horizon_by_class_60.csv":csv_data(rawclass),
      TABLES/"noise_horizon_summary.json":jbytes({
        "schema":"e-ch4-14-ch3-noise-horizon-v1",
        "theory":"§3.8; eq:ch3-noise-general-horizon; eq:ch3-noise-normalized-expectation; eq:ch3-noise-normalized-uniform-ratio",
        "horizons":list(HORIZONS),"observed_steps":[30,60],
        "step_count":31,"n_runs":36,"n_subsets":18,
        "n_timeline_rows":len(timeline),"n_group_rows":len(groups),
        "ratio_undefined_when_noise_count_zero":True,
        "conditional_policy_aggregation":"unweighted macro mean across three candidate classes",
        "no_mask":True,"test_accessed":False,
        "no_training":True,"groups":groups,
      })}
    outputs.update({FIGURES/name:content for name,content in
                    figures(groups,timeline).items()})
    return outputs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("write","check"),required=True)
    args=p.parse_args()
    outputs=build()
    for path,data in outputs.items():
        if args.mode=="write":
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
        else:
            require(path.is_file() and path.read_bytes()==data,
                    "Missing or different derived file "+str(path))
    print("E14_CH3_HORIZONS_AUDIT_PASS",json.dumps({
        "mode":args.mode,"horizons":list(HORIZONS),"runs":36,
        "timeline_rows":4464,"raw_training_updates":0,
        "test_accessed":False,"files":len(outputs)}))
if __name__=="__main__":main()
