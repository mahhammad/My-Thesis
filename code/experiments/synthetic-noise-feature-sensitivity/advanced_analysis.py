"""Read-only E-CH4-14 diagnostics: 36 traces and SHA-pinned step-30 policies.

No graph inference, no training, no Test and no modifications to inputs/results.
--write writes only derived tables/ and figures/; --check verifies those files.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, itertools, json, math, pathlib
import numpy as np
import torch
from checkpoints import load_checkpoint

HERE=pathlib.Path(__file__).resolve().parent
INFORMATIVE=frozenset((0,1,3))
RAW_SHA="7bc66e4d2535fb15dfa5b569933b2d6c5a0962ac"
FAMILIES=("shared","class")
NFI_FIELDS=("run_id","subset_id","k","family","candidate_class","F3","E3","q1","q2","q3","F3_uniform","E3_uniform","NFI3","delta_F3","F3_step0","E3_step0","NFI3_step0","delta_E3","p0_informative","p0_noise","mean_abs_delta_prob_informative","mean_abs_delta_prob_noise","mean_abs_final_logit_informative","mean_abs_final_logit_noise","logits_l2")
STABILITY_FIELDS=("run_id","subset_id","k","family","ba_start","ba_end","ba_peak","max_drawdown_pp","total_variation_pp","ba_down_steps","loss_up_steps","loss_down_ba_down_steps","gradient_peak_count","gradient_trough_count","gradient_extrema_amplitude_mean","gradient_range_early","gradient_range_late","gradient_late_early_range_ratio","gradient_tv_early","gradient_tv_late","gradient_late_early_tv_ratio","initial_gradient","final_gradient","initial_loss","final_loss")

def softmax(z):
    z=np.asarray(z,dtype=np.float64)
    if z.ndim!=2 or z.shape[0]<4 or not np.isfinite(z).all():
        raise ValueError("Finite 2D policy with start row required")
    v=z-np.max(z,axis=1,keepdims=True)
    e=np.exp(v)
    w=e/np.sum(e,axis=1,keepdims=True)
    if not np.allclose(w.sum(axis=1),1,atol=2e-14):
        raise ValueError("Invalid policy normalization")
    return w

def policy3(z,feature_mapping):
    """Counterfactual three consecutive feature selections, ignoring all masks."""
    mapping=tuple(int(x) for x in feature_mapping)
    d=len(mapping)
    if len(set(mapping))!=d or not set(INFORMATIVE).issubset(mapping):
        raise ValueError("Invalid local/parent feature mapping")
    z=np.asarray(z,dtype=np.float64)
    if z.shape!=(d+1,d):raise ValueError("Policy logits must be (d+1,d)")
    Pfull=softmax(z)
    p0=Pfull[d]
    P=Pfull[:d]
    j=np.array([i for i,x in enumerate(mapping) if x in INFORMATIVE],dtype=int)
    n=np.array([i for i,x in enumerate(mapping) if x not in INFORMATIVE],dtype=int)
    if len(j)!=3 or len(n)!=d-3:raise ValueError("Informative/noisy partition mismatch")
    p1,p2,p3=p0,p0@P,p0@P@P
    q=[float(p[n].sum()) for p in (p1,p2,p3)]
    no_noise=float(p0[j]@P[np.ix_(j,j)]@P[np.ix_(j,j)]@np.ones(3))
    F3=float(1-no_noise)
    E3=float(sum(q))
    uni=1-(3/d)**3
    uniE=3*len(n)/d
    if len(n)==0:
        F3=0.0;E3=0.0;uni=0.0;uniE=0.0
    if not (-1e-12<=F3<=1+1e-12 and -1e-12<=E3<=3+1e-12 and all(-1e-12<=v<=1+1e-12 for v in q)):
        raise AssertionError("Noise diagnostic probability out of bounds")
    if F3+1e-12<max(q):raise AssertionError("Union probability < marginal event")
    delta=np.abs(Pfull-1/d)
    return {
       "F3":F3,"E3":E3,"q1":q[0],"q2":q[1],"q3":q[2],
       "F3_uniform":uni,"E3_uniform":uniE,
       "NFI3":(F3/uni if len(n) else None),"delta_F3":F3-uni,
       "p0_informative":float(p0[j].sum()),"p0_noise":float(p0[n].sum()),
       "mean_abs_delta_prob_informative":float(delta[:,j].mean()),
       "mean_abs_delta_prob_noise":float(delta[:,n].mean()) if len(n) else None,
       "mean_abs_final_logit_informative":float(np.abs(z[:,j]).mean()),
       "mean_abs_final_logit_noise":float(np.abs(z[:,n]).mean()) if len(n) else None,
       "logits_l2":float(np.linalg.norm(z)),
    }

def enumerate3(z, mapping):
    """Independent literal d^3-path calculation for tests, not production."""
    d=len(mapping);w=softmax(z);noise={i for i,x in enumerate(mapping) if x not in INFORMATIVE}
    F=E=0.0
    for a,b,c in itertools.product(range(d),repeat=3):
        mass=w[d,a]*w[a,b]*w[b,c]
        hits=sum(x in noise for x in (a,b,c))
        F+=mass*bool(hits)
        E+=mass*hits
    return F,E

def stability(trace,run_id,k,family):
    if len(trace)!=31 or [x["step"] for x in trace]!=list(range(31)):
        raise ValueError("Trace must include exact steps 0..30")
    ba=np.array([x["validation_BA"] for x in trace],dtype=float)
    loss=np.array([x["train_softplus_loss"] for x in trace],dtype=float)
    grad=np.array([x["gradient_l2"] for x in trace],dtype=float)
    if not np.isfinite(ba).all() or not np.isfinite(loss).all() or not np.isfinite(grad).all():raise ValueError("Nonfinite trace")
    if not (np.all((ba>=0)&(ba<=1)) and np.all(grad>=0)):raise ValueError("Invalid BA/gradient")
    dba=np.diff(ba);dl=np.diff(loss)
    # Early and late intervals each contain 15 successive deltas; shared step 15.
    extrema=[i for i in range(1,30) if (grad[i]>grad[i-1] and grad[i]>grad[i+1]) or (grad[i]<grad[i-1] and grad[i]<grad[i+1])]
    peak=[i for i in extrema if grad[i]>grad[i-1] and grad[i]>grad[i+1]]
    trough=[i for i in extrema if grad[i]<grad[i-1] and grad[i]<grad[i+1]]
    amps=[abs(grad[a]-grad[b]) for a,b in zip(extrema,extrema[1:])]
    rng1=float(np.ptp(grad[:16]));rng2=float(np.ptp(grad[15:]))
    tv1=float(np.abs(np.diff(grad[:16])).sum());tv2=float(np.abs(np.diff(grad[15:])).sum())
    return dict(run_id=run_id,subset_id=run_id.split("__")[0],k=k,family=family,
      ba_start=float(ba[0]),ba_end=float(ba[-1]),ba_peak=float(ba.max()),
      max_drawdown_pp=float(np.max(np.maximum.accumulate(ba)-ba)*100),
      total_variation_pp=float(np.abs(dba).sum()*100),
      ba_down_steps=int(np.count_nonzero(dba < -1e-12)),
      loss_up_steps=int(np.count_nonzero(dl > 1e-12)),
      loss_down_ba_down_steps=int(np.count_nonzero((dl < -1e-12)&(dba < -1e-12))),
      gradient_peak_count=len(peak),gradient_trough_count=len(trough),
      gradient_extrema_amplitude_mean=float(np.mean(amps)) if amps else None,
      gradient_range_early=rng1,gradient_range_late=rng2,
      gradient_late_early_range_ratio=(rng2/rng1 if rng1>0 else None),
      gradient_tv_early=tv1,gradient_tv_late=tv2,
      gradient_late_early_tv_ratio=(tv2/tv1 if tv1>0 else None),
      initial_gradient=float(grad[0]),final_gradient=float(grad[-1]),
      initial_loss=float(loss[0]),final_loss=float(loss[-1]))

def numeric_group(rows,keys,group_cols):
    out=[]
    for values in sorted(set(tuple(r[c] for c in group_cols) for r in rows)):
        subset=[r for r in rows if tuple(r[c] for c in group_cols)==values]
        item=dict(zip(group_cols,values));item["n_subsets"]=len(subset)
        for key in keys:
            v=np.array([r[key] for r in subset if r[key] is not None],dtype=float)
            item[key+"_mean"]=float(np.mean(v)) if len(v) else None
            item[key+"_min"]=float(np.min(v)) if len(v) else None
            item[key+"_max"]=float(np.max(v)) if len(v) else None
            item[key+"_subset_sd"]=float(np.std(v,ddof=1)) if len(v)>1 else None
        out.append(item)
    return out

def csv_text(rows,columns):
    out=io.StringIO(newline="");w=csv.DictWriter(out,fieldnames=columns,lineterminator="\n")
    w.writeheader();w.writerows([{k:v for k,v in row.items() if k in columns} for row in rows])
    return out.getvalue()

def verify_metadata(payload,row,meta,job,checkpoint,trace_path,source):
    if payload.get("step")!=30 or meta.get("test_accessed") is not False:raise ValueError("Checkpoint provenance")
    if meta.get("subset_id")!=row["subset_id"] or meta.get("family")!=job.split("__")[1]:
        raise ValueError("Checkpoint row / family mismatch")
    if meta.get("feature_mapping")!=row["feature_mapping"] or meta.get("gamma")!=row["gamma"]:
        raise ValueError("Checkpoint feature mapping/gamma mismatch")
    if meta.get("source_commit")!=source:raise ValueError("Checkpoint source commit changed")
    if meta.get("trace_sha256")!=hashlib.sha256(trace_path.read_bytes()).hexdigest():
        raise ValueError("Checkpoint trace binding mismatch")
    if meta.get("data_sha256")!=row["hashes"]["train_x_sha256"] or meta.get("validation_sha256")!=row["hashes"]["validation_x_sha256"]:
        raise ValueError("Checkpoint dataset hashes mismatch")
    if meta.get("objective")!="softplus_JS_tau005" or meta.get("H")!=12 or meta.get("lr")!=0.12:
        raise ValueError("Wrong objective/horizon/Adam learning rate")
    family=job.split("__")[1];d=row["d"]
    expected=(d+1,d) if family=="shared" else (3,d+1,d)
    if tuple(payload["logits"].shape)!=expected:raise ValueError("Checkpoint shape mismatch")
    if not np.isfinite(payload["logits"].numpy()).all():raise ValueError("Nonfinite checkpoint logits")

def collect():
    raw=json.loads((HERE/"inputs/derived/manifest.json").read_text())
    prov=json.loads((HERE/"results/metrics/provenance.json").read_text())
    if prov["n_fits"]!=36 or prov["updates_per_fit"]!=30 or prov["test_accessed"] is not False or raw["test_accessed"] is not False:
        raise ValueError("Frozen provenance mismatch")
    subsets={x["subset_id"]:x for x in raw["datasets"]}
    if len(subsets)!=18:raise ValueError("Expected 18 derived subsets")
    nfi=[];stabil=[];trace_map={};policy_shifts=[]
    for subset,row in sorted(subsets.items()):
        for family in FAMILIES:
            job=subset+"__"+family
            trace_path=HERE/"results/traces"/job/"trace.json"
            record=json.loads(trace_path.read_text())
            if record["run_id"]!=job or record.get("test_accessed") is not False:raise ValueError("Trace identity")
            trace=record["trace"]
            stabil.append(stability(trace,job,row["k"],family))
            trace_map[job]=trace
            cp=HERE/"results/checkpoints"/job/"step_30.pt"
            checksum=cp.with_suffix(".pt.sha256").read_text().strip().split()[0]
            payload=load_checkpoint(cp,expected_sha256=checksum)
            verify_metadata(payload,row,payload["metadata"],job,cp,trace_path,prov["source_sha"])
            logits=payload["logits"].detach().cpu().numpy()
            matrices=[("shared",logits)] if family=="shared" else [(str(c),logits[c]) for c in range(3)]
            for c,z in matrices:
                vals=policy3(z,row["feature_mapping"])
                init=policy3(np.zeros_like(z),row["feature_mapping"])
                if abs(init["F3"]-vals["F3_uniform"])>1e-12 or abs(init["E3"]-vals["E3_uniform"])>1e-12:
                    raise AssertionError("Actual zero-logit policy is not Uniform")
                vals.update(F3_step0=init["F3"],E3_step0=init["E3"],
                            NFI3_step0=init["NFI3"],delta_E3=vals["E3"]-init["E3"])
                nfi.append(dict(run_id=job,subset_id=subset,k=row["k"],family=family,candidate_class=c,**vals))
            if family=="class":
                crows=nfi[-3:]
                macro={key:(sum(v[key] for v in crows)/3 if all(v[key] is not None for v in crows) else None)
                       for key in NFI_FIELDS if key not in ("run_id","subset_id","k","family","candidate_class")}
                nfi.append(dict(run_id=job,subset_id=subset,k=row["k"],family=family,candidate_class="macro",**macro))
    if len(nfi)!=90 or len(stabil)!=36:raise ValueError("Missing 36 policies")
    nfi_group=numeric_group([x for x in nfi if x["candidate_class"] in ("shared","macro")],
      ["F3","E3","q1","q2","q3","F3_uniform","E3_uniform","NFI3","delta_F3","F3_step0","E3_step0","NFI3_step0","delta_E3","p0_noise","mean_abs_delta_prob_informative","mean_abs_delta_prob_noise","mean_abs_final_logit_informative","mean_abs_final_logit_noise"],["k","family"])
    st_group=numeric_group(stabil,["max_drawdown_pp","total_variation_pp","ba_down_steps","loss_up_steps",
      "loss_down_ba_down_steps","gradient_peak_count","gradient_trough_count","gradient_extrema_amplitude_mean",
      "gradient_late_early_range_ratio","gradient_late_early_tv_ratio"],["k","family"])
    result={"schema":"e-ch4-14-noise-dynamics-postfit-v1","frozen_results_sha":RAW_SHA,
            "method":"31 observed trace steps, SHA-pinned logits at steps 0 and 30; no checkpoint interpolation",
            "stability_by_run":stabil,"stability_by_level":st_group,
            "nfi3_by_policy":nfi,"nfi3_by_level":nfi_group}
    return result,trace_map

def plot_all(result,traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none","svg.hashsalt":"e14-advanced-analysis-v1"})
    palette=["#1d4ed8","#0891b2","#0f766e","#ca8a04","#ea580c","#7c3aed"]
    subsets={k:[j for j in traces if j.startswith(f"N{k}-") and j.endswith("__shared")]
             for k in range(6)}
    def arr(k,f,field):
        rows=[np.array([v[field] for v in traces[s.replace("__shared","__"+f)]],dtype=float)
              for s in subsets[k]]
        return rows,np.mean(rows,axis=0)
    def save(fig,name):
        fig.savefig(HERE/"figures"/name,format="svg",metadata={"Date":None})
        plt.close(fig)
    allba=np.concatenate([arr(k,f,"validation_BA")[1]*100 for k in range(6) for f in FAMILIES])
    ymin=float(np.floor(min(allba.min(),min(traces[j][t]["validation_BA"]*100 for j in traces for t in range(31)))-1))
    ymax=float(np.ceil(max(allba.max(),max(traces[j][t]["validation_BA"]*100 for j in traces for t in range(31)))+1))
    for f in FAMILIES:
        fig,ax=plt.subplots(figsize=(10.8,5.6))
        for k in range(6):
            _,means=arr(k,f,"validation_BA")
            ax.plot(range(31),means*100,color=palette[k],linewidth=2.2,label=f"k={k} (n={len(subsets[k])})")
        ax.set(xlim=(0,30),ylim=(ymin,ymax),xlabel="Adam update step",ylabel="Validation balanced accuracy (%)",
               title=f"All noise levels | {f} | mean of dependent subsets")
        ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=9);fig.tight_layout()
        save(fig,f"08_validation_BA_all_k_{f}.svg")
        # Six facets preserve each unaveraged subset trajectory.
        fig,axes=plt.subplots(2,3,figsize=(14.5,7.7),sharex=True,sharey=True)
        for k,ax in enumerate(axes.flat):
            curves,mean=arr(k,f,"validation_BA")
            if k in (1,2,3,4):
                for s in curves:ax.plot(range(31),s*100,color=palette[k],alpha=.22,lw=1)
                ax.plot(range(31),mean*100,color=palette[k],lw=2.4,label=f"mean ({len(curves)})")
            else:ax.plot(range(31),mean*100,color=palette[k],lw=2.4,label="single subset")
            ax.set_title(f"k={k}",fontsize=11);ax.set_xlim(0,30);ax.grid(alpha=.2)
        for ax in axes[1]:ax.set_xlabel("Adam update")
        for ax in axes[:,0]:ax.set_ylabel("Validation BA (%)")
        axes[0,0].set_ylim(ymin,ymax)
        fig.suptitle(f"Individual dependent subsets | {f} | common BA axis",fontsize=14)
        fig.tight_layout();save(fig,f"09_validation_BA_individual_{f}.svg")
        # Shared step axis makes objective/decision mismatch visible.
        fig,axes=plt.subplots(2,2,figsize=(13.6,8.0),sharex=True)
        for k in range(6):
            for ax,field,scale in zip(axes.flat,["validation_BA","train_softplus_loss","gradient_l2","adam_update_l2"],[100,1,1,1]):
                _,mean=arr(k,f,field)
                ax.plot(range(31),mean*scale,color=palette[k],linewidth=1.8,label=f"k={k}")
        for ax,title,ylabel in zip(axes.flat,["Validation BA","Train Softplus-JS loss","Gradient L2","Adam update L2"],
             ["BA (%)","Loss","Gradient norm (log y)","Update norm"]):
            ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(alpha=.22);ax.set_xlim(0,30)
        axes.flat[2].set_yscale("log")
        for ax in axes[1]:ax.set_xlabel("Adam update step")
        axes[0,0].legend(ncol=3,fontsize=8)
        fig.suptitle(f"Aligned BA / loss / gradient / update | {f} | all k=0..5")
        fig.tight_layout();save(fig,f"10_full_learning_{f}.svg")
        # Individual gradient traces detect smoothing by subset averaging.
        fig,axes=plt.subplots(2,3,figsize=(14.5,7.7),sharex=True,sharey=True)
        for k,ax in enumerate(axes.flat):
            curves,mean=arr(k,f,"gradient_l2")
            if k in (1,2,3,4):
                for s in curves:ax.plot(range(31),s,color=palette[k],alpha=.18,lw=1)
            ax.plot(range(31),mean,color=palette[k],lw=2.1);ax.set_title(f"k={k}");ax.grid(alpha=.22)
        for ax in axes[1]:ax.set_xlabel("Adam update step")
        for ax in axes[:,0]:ax.set_ylabel("Gradient L2")
        fig.suptitle(f"Per-subset gradient waveform | {f} | common linear axis")
        fig.tight_layout();save(fig,f"11_gradient_individual_{f}.svg")
    # F3, E3 and NFI3 summary: learned vs uniform.
    rows=result["nfi3_by_level"];fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for f in FAMILIES:
        vals=[next(x for x in rows if x["k"]==k and x["family"]==f) for k in range(6)]
        color="#2563eb" if f=="shared" else "#ea580c"
        for ax,key in zip(axes,["F3","E3","NFI3"]):
            y=[x[key+"_mean"] if x[key+"_mean"] is not None else np.nan for x in vals]
            ax.plot(range(6),y,marker="o",color=color,label=f)
    uni=[1-(3/(3+k))**3 for k in range(6)]
    axes[0].plot(range(6),uni,color="#64748b",ls="--",label="Uniform")
    axes[1].plot(range(6),[3*k/(3+k) for k in range(6)],color="#64748b",ls="--",label="Uniform")
    axes[2].axhline(1,color="#64748b",ls="--",label="Uniform ratio")
    for ax,name in zip(axes,["F3: P(any noise in 3 choices)","E3: expected noise selections","NFI3 = F3 / uniform F3"]):
        ax.set_title(name);ax.set_xlabel("Noise features k");ax.grid(alpha=.25)
    axes[0].set_ylim(0,1.05);axes[1].set_ylim(0,3.05)
    axes[0].legend(fontsize=9);fig.tight_layout();save(fig,"12_nfi3_f3_e3.svg")
    # Hypothesis diagnostic scatter, not causal or p-value.
    st={x["run_id"]:x for x in result["stability_by_run"]}
    fig,ax=plt.subplots(figsize=(8.6,5.7))
    for f in FAMILIES:
        group=[x for x in result["nfi3_by_policy"] if x["family"]==f and x["candidate_class"] in ("shared","macro") and x["k"]>0]
        ax.scatter([x["NFI3"] for x in group],[st[x["run_id"]]["max_drawdown_pp"] for x in group],
                   c=[palette[x["k"]] for x in group],marker="o" if f=="shared" else "^",
                   edgecolors="#374151",linewidths=.4,label=f,alpha=.85,s=50)
    ax.axvline(1,color="#64748b",ls="--");ax.set(xlabel="NFI3 (counterfactual, step 30)",
        ylabel="Validation BA max drawdown (pp; updates 0..30)",
        title="Descriptive association; 18 dependent subsets, 2 families")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();save(fig,"13_nfi3_vs_BA_drawdown.svg")

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("write","check"),default="check")
    args=p.parse_args(argv)
    result,traces=collect()
    outputs={
      "stability_by_run.csv":csv_text(result["stability_by_run"],STABILITY_FIELDS),
      "stability_by_noise_level.csv":csv_text(result["stability_by_level"],list(result["stability_by_level"][0])),
      "nfi3_by_policy.csv":csv_text(result["nfi3_by_policy"],NFI_FIELDS),
      "nfi3_by_noise_level.csv":csv_text(result["nfi3_by_level"],list(result["nfi3_by_level"][0])),
      "noise_dynamics_summary.json":json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
    }
    for name,content in outputs.items():
        file=HERE/"tables"/name
        if args.mode=="check":
            if file.read_text()!=content:raise ValueError("Postfit analysis changed: "+name)
        else:file.write_text(content)
    if args.mode=="write":plot_all(result,traces)
    print("E14_ADVANCED_ANALYSIS_PASS",json.dumps(dict(mode=args.mode,runs=36,policies=len(result["nfi3_by_policy"]),
          fixed_training_updates=1080,k_levels=6,test_accessed=False)))
if __name__=="__main__":main()
