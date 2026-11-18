"""E-CH4-14 postfit Softmax sensitivity and noise-allocation diagnostics.

Exactly replays saved logits/trace at steps 30..60. It neither trains nor
evaluates Test. Probabilities are the auxiliary unmasked §3.8 feature chain,
not real masked sample-feature visits. Local Jacobians measure partial
derivatives at fixed logits, NOT causal impact of adding noise during training.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TABLE = HERE / "tables"
FIG = HERE / "figures"
RAW = HERE / "results/continuation_30_60"
INFO = (0, 1, 3)
FAMILIES = ("shared", "class")
HORIZONS = (1, 2, 3)


def softmax(z):
    a=np.asarray(z,dtype=float)
    e=np.exp(a-a.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def row_jacobian(p, idx):
    """Jacobian of informative output probabilities wrt all d logits."""
    pp=np.asarray(p,dtype=float)
    ids=np.asarray(idx,dtype=int)
    j=-np.outer(pp[ids],pp)
    j[np.arange(len(ids)),ids]+=pp[ids]
    return j


def row_stats(z, info):
    p=softmax(z)
    pp=p[list(info)]
    mass=float(pp.sum())
    w=pp/mass
    jf=row_jacobian(p,info)
    jp=jf[:,list(info)]
    jc=np.diag(w)-np.outer(w,w)
    return {
        "row_main_mass":mass,
        "row_noise_mass":1-mass,
        "jac_main_to_main_frob":float(np.linalg.norm(jp)),
        "jac_main_to_all_frob":float(np.linalg.norm(jf)),
        "jac_main_to_noise_frob":float(np.linalg.norm(
            np.delete(jf,list(info),axis=1))),
        "jac_conditional_main_frob":float(np.linalg.norm(jc)),
        "informative_logit_spread":float(np.max(np.asarray(z)[list(info)])
                                         -np.min(np.asarray(z)[list(info)]))
    }


def metrics(z,mapping):
    """Equally average class policies, each with start and 3 informative rows."""
    d=len(mapping)
    ii=[j for j,v in enumerate(mapping) if v in INFO]
    assert len(ii)==3 and len(set(mapping))==d
    policies=np.asarray(z,dtype=float)
    if policies.ndim==2:policies=policies[None,:,:]
    assert policies.shape[1:]==(d+1,d) and policies.shape[0] in (1,3)
    rs=[]
    feature_by_class=[]
    for policy in policies:
        p=softmax(policy)
        assert np.allclose(p.sum(-1),1,atol=1e-12)
        for row in [d,*ii]:
            rs.append(row_stats(policy[row],ii))
        v=p[d]
        series=[v]
        for _ in (2,3):
            v=v @ p[:d]
            series.append(v)
        by_h=[]
        for h in HORIZONS:
            shares=np.mean(series[:h],axis=0)
            assert abs(shares.sum()-1)<1e-12
            mass=float(shares[ii].sum())
            by_h.append((mass,shares[ii]/mass,shares))
        feature_by_class.append(by_h)
    result={key:float(np.mean([r[key] for r in rs])) for key in rs[0]}
    result["sensitivity_ratio_to_renormalized_main"]=(
        result["jac_main_to_main_frob"]/result["jac_conditional_main_frob"])
    for idx,h in enumerate(HORIZONS):
        mass=float(np.mean([c[idx][0] for c in feature_by_class]))
        # Macro probability average, then conditional normalization. Do not
        # confuse with the mean of class-conditional normalized preferences.
        shares=np.mean([c[idx][2] for c in feature_by_class],axis=0)
        conditional=shares[ii]/float(shares[ii].sum())
        result[f"main_share_h{h}"]=mass
        result[f"noise_share_h{h}"]=1-mass
        for j,feature in enumerate(INFO):
            result[f"feature_{feature}_share_h{h}"]=float(shares[ii[j]])
            result[f"feature_{feature}_conditional_h{h}"]=float(conditional[j])
    return result


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf8").splitlines()
            if line.strip()]


def build():
    source=json.loads((HERE/"inputs/derived/manifest.json").read_text("utf8"))
    assert source.get("test_accessed") is False
    datasets={r["subset_id"]:r for r in source["datasets"]}
    assert len(datasets)==18
    timeline=[]
    for subset,row in sorted(datasets.items()):
        mapping=[int(v) for v in row["feature_mapping"]]
        for fam in FAMILIES:
            run=f"{subset}__{fam}"
            logs=read_jsonl(RAW/run/"logits_30_60.jsonl")
            trace=read_jsonl(RAW/run/"trace_30_60.jsonl")
            assert len(logs)==len(trace)==31,run
            for a,b in zip(logs,trace):
                step=int(a["step"])
                assert int(b["step"])==step and 30<=step<=60,run
                z=np.asarray(a["logits"],dtype=float)
                assert np.all(np.isfinite(z))
                m=metrics(z,mapping)
                g=float(b["gradient_l2"])
                n=z.size
                timeline.append({"run_id":run,"family":fam,
                    "noise_count":len(mapping)-3,"step":step,
                    "n_logits":n,"gradient_l2":g,
                    "gradient_rms":g/np.sqrt(n),
                    "validation_BA":float(b["validation_BA"]),
                    **m})
    assert len(timeline)==36*31
    timeline.sort(key=lambda r:(r["family"],r["noise_count"],
                                r["run_id"],r["step"]))
    # Temporal continuity without treating different runs as consecutive.
    previous={}
    for r in timeline:
        old=previous.get(r["run_id"])
        if old:
            assert r["step"]==old["step"]+1
            for metric in ("main_share_h3","sensitivity_ratio_to_renormalized_main",
                           "jac_main_to_main_frob","gradient_rms"):
                r[f"abs_change_{metric}"]=abs(r[metric]-old[metric])
            r["conditional_main_TV_step_h3"]=.5*sum(
                abs(r[f"feature_{a}_conditional_h3"]-
                    old[f"feature_{a}_conditional_h3"]) for a in INFO)
        else:
            for metric in ("main_share_h3","sensitivity_ratio_to_renormalized_main",
                           "jac_main_to_main_frob","gradient_rms"):
                r[f"abs_change_{metric}"]=None
            r["conditional_main_TV_step_h3"]=None
        previous[r["run_id"]]=r
    last=[r for r in timeline if r["step"]==60]
    assert len(last)==36
    groups=[]
    for fam in FAMILIES:
        for k in range(6):
            runs=[r for r in last if r["family"]==fam and r["noise_count"]==k]
            paths=[r for r in timeline if r["family"]==fam and r["noise_count"]==k]
            assert len(runs)==(1 if k in (0,5) else 4)
            rec={"family":fam,"noise_count":k,"n_dependent_subsets":len(runs)}
            keys=("main_share_h1","main_share_h2","main_share_h3",
                  "noise_share_h3","row_main_mass","jac_main_to_main_frob",
                  "jac_main_to_all_frob","jac_main_to_noise_frob",
                  "jac_conditional_main_frob",
                  "sensitivity_ratio_to_renormalized_main",
                  "informative_logit_spread","gradient_l2","gradient_rms",
                  "validation_BA")
            for key in keys:
                x=np.asarray([r[key] for r in runs])
                rec[key+"_60_mean"]=float(x.mean())
                rec[key+"_60_min"]=float(x.min())
                rec[key+"_60_max"]=float(x.max())
            for key in ("gradient_rms","jac_main_to_main_frob",
                        "main_share_h3","conditional_main_TV_step_h3"):
                x=np.asarray([r[key] for r in paths if r[key] is not None])
                rec[key+"_30_60_mean"]=float(x.mean())
            groups.append(rec)
    assert len(groups)==12
    # Cross-check against independently generated step-60 exact §3.8 shares.
    with (TABLE/"feature_preference_by_run_60.csv").open(newline="",encoding="utf8") as file:
        frows=list(csv.DictReader(file))
    reference={(r["run_id"],int(r["h"])):r for r in frows}
    assert len(reference)==108
    for r in last:
        for h in HORIZONS:
            f=reference[(r["run_id"],h)]
            assert abs(r[f"main_share_h{h}"]-float(f["main_share"]))<2e-10
            for feature in INFO:
                assert abs(r[f"feature_{feature}_conditional_h{h}"]-
                           float(f[f"feature_{feature}_conditional_share"]))<2e-10
    return timeline,last,groups


def csvbytes(rows):
    s=io.StringIO(newline="")
    w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows)
    return s.getvalue().encode("utf8")


def svg(fig):
    import matplotlib.pyplot as plt
    b=io.BytesIO()
    fig.savefig(b,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    return b.getvalue()


def plots(timeline,last,groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=plt.get_cmap("tab10").colors
    plt.rcParams.update({"font.size":9,"figure.dpi":110,
                         "svg.fonttype":"path","svg.hashsalt":"e14-softmax-buffer-01"})
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for j,fam in enumerate(FAMILIES):
        g=sorted((r for r in groups if r["family"]==fam),
                 key=lambda r:r["noise_count"])
        xx=range(6)
        for i,(metric,title) in enumerate((
            ("main_share_h3_60_mean","Informative selection share h=3"),
            ("jac_main_to_main_frob_60_mean","Main/Main Softmax Jacobian norm"))):
            ax=axes[i,j]
            ax.plot(xx,[r[metric] for r in g],marker="o",lw=2)
            if i==0:ax.plot(xx,[3/(3+k) for k in xx],ls="--",color=".4",
                            label="Uniform main share")
            ax.set(xlabel="Number of noise features",ylabel=title,
                   title=f"{fam} / {title}")
            ax.grid(alpha=.2)
            if i==0:ax.legend()
    fig.suptitle("Unmasked policy: noise allocation and local informative-logit response")
    fig.tight_layout()
    out={"44_noise_allocation_and_softmax_jacobian_60.svg":svg(fig)}
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for j,fam in enumerate(FAMILIES):
        for i,(metric,label) in enumerate((
            ("jac_main_to_main_frob","Main/Main Jacobian Frobenius"),
            ("gradient_rms","Training gradient L2 / sqrt(n_logits)"))):
            ax=axes[i,j]
            for k in range(6):
                ys=[]
                for step in range(30,61):
                    rr=[r[metric] for r in timeline if r["family"]==fam and
                        r["noise_count"]==k and r["step"]==step]
                    assert len(rr)==(1 if k in (0,5) else 4)
                    ys.append(float(np.mean(rr)))
                ax.plot(range(30,61),ys,label=f"k={k}",lw=1.7,
                        color=colors[k])
            ax.set(xlabel="Adam step 30..60",ylabel=label,
                   title=f"{fam}: {label}")
            ax.grid(alpha=.15)
            if i==0:ax.legend(ncol=3,fontsize=8)
    fig.suptitle("Postfit Softmax sensitivity vs per-parameter gradient scale")
    fig.tight_layout()
    out["45_softmax_gradient_paths_30_60.svg"]=svg(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
    for j,fam in enumerate(FAMILIES):
        g=sorted((r for r in groups if r["family"]==fam),
                 key=lambda r:r["noise_count"])
        xx=range(6)
        axes[j].plot(xx,[r["sensitivity_ratio_to_renormalized_main_60_mean"] for r in g],
                     marker="o",label="Jac_PP / Jac_conditional")
        axes[j].axhline(1,color=".35",ls="--",label="Reference ratio = 1")
        axes[j].set(title=fam,xlabel="Number of noise features",
                    ylabel="Relative local Jacobian response")
        axes[j].grid(alpha=.2);axes[j].legend()
    fig.suptitle("Does allocating probability to noise attenuate the informative-only response?")
    fig.tight_layout()
    out["46_softmax_dilution_ratio_60.svg"]=svg(fig)
    return out


def artifacts():
    timeline,last,groups=build()
    outs={TABLE/"softmax_buffer_by_step_30_60.csv":csvbytes(timeline),
          TABLE/"softmax_buffer_by_run_60.csv":csvbytes(last),
          TABLE/"softmax_buffer_by_noise_level_60.csv":csvbytes(groups)}
    outs.update({FIG/name:data for name,data in plots(timeline,last,groups).items()})
    return outs


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("write","check"),default="check")
    args=p.parse_args()
    for path,data in artifacts().items():
        if args.mode=="write":
            path.write_bytes(data)
        else:
            assert path.is_file() and path.read_bytes()==data, f"Missing/stale: {path}"
    print("E14_SOFTMAX_BUFFER_POSTFIT_PASS")
