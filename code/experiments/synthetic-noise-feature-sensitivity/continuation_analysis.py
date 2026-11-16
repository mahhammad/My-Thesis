"""Read-only E-CH4-14 0..60 continuation audit and reproducible group plots.

Never runs an optimizer, reads Test, or modifies frozen training evidence.
Only writes tables/continuation_* and figures/14_* ... 21_* on --mode write.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, pathlib
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=pathlib.Path(__file__).resolve().parent
DEST=HERE/"results/continuation_30_60"
FIG=HERE/"figures"
TAB=HERE/"tables"
FAMILIES=("shared","class")
METRICS=("train_BA","validation_BA","train_softplus_loss","gradient_l2","adam_update_l2",
         "train_Accuracy","validation_Accuracy")
NFI=("F3","E3","NFI3","q1","q2","q3","p0_noise","p_N_to_N_given_start_N")
EXPECTED_TAG="refs/tags/e-ch4-14-resume-30-60-budget-spent"
plt.rcParams.update({"font.size":8,"svg.hashsalt":"e-ch4-14-continuation-v1",
 "figure.dpi":110,"savefig.dpi":110})
COLORS=plt.get_cmap("tab10").colors

def require(condition,message):
    if not condition: raise AssertionError(message)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]

def candidates(row,family):
    return [row["nfi3"]["macro" if family=="class" else "shared"]]

def get_nfi(row,family,key):
    return candidates(row,family)[0][key]

def tv(rows,key):
    return float(np.abs(np.diff([r[key] for r in rows])).sum())

def signature(rows):
    return [(x["step"],x["validation_BA"],x["train_softplus_loss"]) for x in rows]

def audit():
    metadata=read_json(HERE/"inputs/derived/manifest.json")
    data={r["subset_id"]:r for r in metadata["datasets"]}
    require(len(data)==18,"Exactly 18 source subsets are required")
    ledger=read_jsonl(DEST/"resume_budget.jsonl")
    count=Counter(x["event"] for x in ledger)
    require(dict(count)=={"BUDGET_RESERVED":1,"START":36,"UPDATE":1080,"SUCCESS":36},
            "Resume ledger must be exact and without failures: "+str(count))
    byrun=defaultdict(list)
    for r in ledger:
        if "run_id" in r: byrun[r["run_id"]].append(r)
    for id,recs in byrun.items():
        updates=[x for x in recs if x["event"]=="UPDATE"]
        require(len(updates)==30 and
            [x["cumulative_step"] for x in updates]==list(range(31,61)),
            "Thirty uninterrupted authorized updates required: "+id)
    traces={}; old={}; run_summary=[]; class_curves={}
    for subset,info in sorted(data.items()):
        k=int(info["k"])
        for family in FAMILIES:
            id=subset+"__"+family
            source=read_json(HERE/"results/traces"/id/"trace.json")
            before=source["trace"]
            folder=DEST/id
            manifest=read_json(folder/"manifest.json")
            after=read_jsonl(folder/"trace_30_60.jsonl")
            policy=read_jsonl(folder/"logits_30_60.jsonl")
            require(source["run_id"]==id and source.get("test_accessed") is False
                    and manifest["run_id"]==id and manifest["test_accessed"] is False,
                    "Identity/provenance/test mismatch "+id)
            require(len(before)==31 and len(after)==31 and len(policy)==31
                    and [r["step"] for r in before]==list(range(31))
                    and [r["step"] for r in after]==list(range(30,61))
                    and [r["step"] for r in policy]==list(range(30,61)),
                    "Step sequence invalid "+id)
            require(manifest["step_start"]==30 and manifest["step_end"]==60
                    and manifest["updates_added"]==30 and
                    manifest["budget_ref"]==EXPECTED_TAG,"Manifest mismatch "+id)
            for r in after:
                require(r["run_id"]==id and r["test_accessed"] is False
                        and r["numeric_status"]=="finite"
                        and r["update_completed"]==r["step"],
                        "Trace provenance/status wrong "+id)
            for metric in METRICS:
                if metric=="adam_update_l2" and before[-1][metric] is None:continue
                require(np.isclose(after[0][metric],before[-1][metric],atol=5e-9,rtol=2e-8),
                        "Step30 metric boundary differs "+id+" "+metric)
            require(manifest["trace_sha256"]==sha(folder/"trace_30_60.jsonl")
                    and manifest["logits_journal_sha256"]==sha(folder/"logits_30_60.jsonl")
                    and manifest["logits_npz_sha256"]==sha(folder/"logits_steps30_60.npz")
                    and manifest["new_checkpoint_sha256"]==sha(folder/"step_60.pt")
                    and (folder/"step_60.pt.sha256").read_text().strip()==
                       manifest["new_checkpoint_sha256"],"SHA mismatch "+id)
            cp30=HERE/"results/checkpoints"/id/"step_30.pt"
            require(sha(cp30)==manifest["old_checkpoint_sha256"]
                    and (cp30.with_suffix(".pt.sha256")).read_text().strip()==
                       manifest["old_checkpoint_sha256"],"Origin SHA mismatch "+id)
            with np.load(folder/"logits_steps30_60.npz",allow_pickle=False) as npz:
                require(set(npz.files)=={"steps","logits"} and
                        np.array_equal(npz["steps"],np.arange(30,61)),
                        "NPZ steps mismatch "+id)
                stacked=npz["logits"].copy()
            require(stacked.dtype==np.float64 and
                    list(stacked.shape)==manifest["logits_shape"],"NPZ logits shape/dtype mismatch "+id)
            for j,entry in enumerate(policy):
                matrix=np.asarray(entry["logits"],dtype=np.float64)
                require(np.array_equal(matrix,stacked[j]),"Journal/NPZ logits differ "+id)
                require(hashlib.sha256(np.asarray(matrix,dtype="<f8").tobytes()).hexdigest()
                        ==entry["sha256_float64_le"],"Logit float64 SHA mismatch "+id)
            cp=torch.load(folder/"step_60.pt",map_location="cpu",weights_only=True)
            require(cp["step"]==60 and cp["schema"]=="e-ch4-14-resume-checkpoint-v1"
                    and cp["metadata"]["test_accessed"] is False
                    and cp["metadata"]["source_checkpoint30_sha256"]==
                        manifest["old_checkpoint_sha256"],"Checkpoint provenance mismatch "+id)
            z=cp["logits"].detach().cpu().numpy()
            require(np.array_equal(z,stacked[-1]),"Step60 checkpoint logits mismatch "+id)
            states=cp["optimizer"]["state"]
            require(len(states)==1 and all(int(v["step"])==60 for v in states.values()),
                    "Adam optimizer step not 60 "+id)
            require(all(v["exp_avg"].dtype==torch.float64 and
                        v["exp_avg_sq"].dtype==torch.float64 for v in states.values()),
                    "Adam moments not float64 "+id)
            require(len([x for x in byrun[id] if x["event"]=="START"])==1
                    and len([x for x in byrun[id] if x["event"]=="SUCCESS"])==1,
                    "Start/success mismatch "+id)
            # Stitch the original 0..29 diagnostics to the richer recomputed\n            # step30..60 records; the common step30 scalars were checked above.\n            all_steps=before[:-1]+after
            old[id]=before;traces[id]=all_steps
            if family=="class":
                class_curves[id]=[{str(c):r["nfi3"][str(c)] for c in range(3)}
                                  for r in after]
            record={"run_id":id,"subset_id":subset,"family":family,"k":k}
            for key in ("validation_BA","train_BA","train_softplus_loss","gradient_l2",
                        "validation_Accuracy","train_Accuracy"):
                record[key+"_30"]=after[0][key]
                record[key+"_60"]=after[-1][key]
                record["delta_"+key]=after[-1][key]-after[0][key]
            for key in ("F3","E3","NFI3","q1","q2","q3","p0_noise",
                        "p_N_to_N_given_start_N"):
                v0,v1=get_nfi(after[0],family,key),get_nfi(after[-1],family,key)
                record[key+"_30"]=v0;record[key+"_60"]=v1
                record["delta_"+key]=v1-v0 if v0 is not None and v1 is not None else None
            record["ba_tv_0_30_pp"]=100*tv(before,"validation_BA")
            record["ba_tv_30_60_pp"]=100*tv(after,"validation_BA")
            record["ba_drawdown_30_60_pp"]=100*max(
                float(np.max(np.maximum.accumulate([r["validation_BA"] for r in after])-
                              [r["validation_BA"] for r in after])),0)
            record["loss_tv_0_30"]=tv(before,"train_softplus_loss")
            record["loss_tv_30_60"]=tv(after,"train_softplus_loss")
            record["gradient_tv_0_30"]=tv(before,"gradient_l2")
            record["gradient_tv_30_60"]=tv(after,"gradient_l2")
            record["gradient_mean_0_30"]=float(np.mean([r["gradient_l2"] for r in before]))
            record["gradient_mean_30_60"]=float(np.mean([r["gradient_l2"] for r in after]))
            record["ba_drops_30_60"]=int(np.sum(np.diff([r["validation_BA"] for r in after])<-1e-12))
            record["ba_changes_30_60"]=int(np.sum(np.abs(np.diff([r["validation_BA"] for r in after]))>1e-12))
            record["F3_BA_delta_sign_agreement_count"]=int(np.sum([
                (np.sign(get_nfi(after[j],family,"F3")-get_nfi(after[j-1],family,"F3"))==
                 np.sign(after[j]["validation_BA"]-after[j-1]["validation_BA"]))
                and abs(after[j]["validation_BA"]-after[j-1]["validation_BA"])>1e-12
                for j in range(1,31)]))
            run_summary.append(record)
    require(len(traces)==36 and len(run_summary)==36 and len(byrun)==36,
            "36 complete independent trajectory identities expected")
    return old,traces,run_summary,class_curves,dict(count)

def group_table(rows):
    result=[]
    for family in FAMILIES:
        for k in range(6):
            sample=[x for x in rows if x["family"]==family and x["k"]==k]
            require(len(sample)==(1 if k in (0,5) else 4),
                    "Unexpected subset count for k="+str(k))
            grouped={"family":family,"k":k,"n_subsets":len(sample)}
            for key in sample[0]:
                if key in ("family","k","run_id","subset_id"):continue
                vals=[r[key] for r in sample if isinstance(r[key],(int,float))
                      and r[key] is not None and math.isfinite(r[key])]
                grouped[key+"_mean"]=float(np.mean(vals)) if vals else None
                grouped[key+"_min"]=float(np.min(vals)) if vals else None
                grouped[key+"_max"]=float(np.max(vals)) if vals else None
            result.append(grouped)
    return result

def csv_bytes(records):
    buf=io.StringIO(newline="")
    keys=list(records[0].keys())
    writer=csv.DictWriter(buf,fieldnames=keys,lineterminator="\n")
    writer.writeheader();writer.writerows(records)
    return buf.getvalue().encode("utf8")

def svg(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    return buf.getvalue()

def family_curves(traces, family, key, *,start=0,nfi=False):
    out={}
    for k in range(6):
        individual=[]
        for id,t in sorted(traces.items()):
            if not id.endswith("__"+family) or int(id[1])!=k:continue
            z=t[start:]
            if nfi:
                series=[get_nfi(r,family,key) for r in z]
            else:
                series=[r[key] for r in z]
            individual.append(np.array([np.nan if x is None else x for x in series],dtype=float))
        out[k]=(np.arange(start,61),np.array(individual))
    return out

def plot_group(ax,traces,family,key,start,scale=1):
    curves=family_curves(traces,family,key,start=start,nfi=key in NFI)
    for k,(xs,arr) in curves.items():
        ys=arr*scale
        for v in ys:
            if len(ys)>1:ax.plot(xs,v,color=COLORS[k],alpha=.17,lw=.65)
        mean=np.nanmean(ys,axis=0) if np.any(np.isfinite(ys)) else np.full(len(xs),np.nan)
        if np.any(np.isfinite(mean)):
            ax.plot(xs,mean,color=COLORS[k],lw=2,label=f"k={k} (n={len(ys)})")
    if start==0:ax.axvline(30,color=".3",ls=":",lw=.8)
    ax.set_xlim(start,60);ax.grid(alpha=.16)

def charts(old,traces,run_rows,class_curves):
    figs={}
    for family in FAMILIES:
        fig,ax=plt.subplots(figsize=(10.5,5.3))
        plot_group(ax,traces,family,"validation_BA",0,100)
        ax.set(title=f"{family}: validation balanced accuracy, steps 0–60",
               xlabel="Cumulative Adam updates",ylabel="Validation BA (%)")
        ax.legend(ncol=3,fontsize=8,loc="best")
        figs[f"14_validation_BA_0_60_{family}.svg"]=svg(fig)
        fig,axs=plt.subplots(2,1,figsize=(10.5,7.4),sharex=True)
        for a,key,scale,label in zip(axs,("train_softplus_loss","gradient_l2"),(1,1),
                                    ("Train Softplus-JS loss","Gradient L2")):
            plot_group(a,traces,family,key,0,scale)
            a.set_ylabel(label)
        axs[0].set_title(f"{family}: loss and gradient, 0–60")
        axs[1].set_xlabel("Cumulative Adam updates")
        axs[0].legend(ncol=3,fontsize=7)
        figs[f"15_loss_gradient_0_60_{family}.svg"]=svg(fig)
        fig,axs=plt.subplots(3,1,figsize=(10.5,9.1),sharex=True)
        for a,key,label in zip(axs,("F3","E3","NFI3"),
                                ("F3: any noise in 3 selections",
                                 "E3: expected noise selections",
                                 "NFI3 relative to uniform")):
            plot_group(a,traces,family,key,30)
            a.set_ylabel(label)
            if key=="NFI3":a.axhline(1,ls=":",lw=.9,color=".4")
        axs[0].set_title(f"{family}: policy noise diagnostics, steps 30–60 (no mask)")
        axs[0].legend(ncol=3,fontsize=7)
        axs[-1].set_xlabel("Cumulative Adam updates")
        figs[f"16_F3_E3_NFI3_30_60_{family}.svg"]=svg(fig)
        fig,axs=plt.subplots(3,1,figsize=(10.5,8.8),sharex=True)
        for a,key,label in zip(axs,("q1","q2","q3"),
                               ("P(noise at selection 1)","P(noise at selection 2)",
                                "P(noise at selection 3)")):
            plot_group(a,traces,family,key,30)
            a.set_ylabel(label)
        axs[0].set_title(f"{family}: raw noise probability by selection 30–60")
        axs[0].legend(ncol=3,fontsize=7)
        axs[-1].set_xlabel("Cumulative Adam updates")
        figs[f"17_q1_q2_q3_30_60_{family}.svg"]=svg(fig)
        fig,axs=plt.subplots(2,1,figsize=(10.5,7.0),sharex=True)
        for a,key,label in zip(axs,("p0_noise","p_N_to_N_given_start_N"),
                               ("P(first selection is noise)",
                                "P(second noise | first is noise)")):
            plot_group(a,traces,family,key,30)
            a.set_ylabel(label)
        axs[0].set_title(f"{family}: selection and conditional noise repetition")
        axs[0].legend(ncol=3,fontsize=7)
        axs[-1].set_xlabel("Cumulative Adam updates")
        figs[f"18_noise_repeat_30_60_{family}.svg"]=svg(fig)
        fig,ax=plt.subplots(figsize=(10.5,4.9))
        ks=np.arange(6)
        for j,(key,label) in enumerate((("ba_tv_0_30_pp","0–30"),("ba_tv_30_60_pp","30–60"))):
            means=np.array([np.mean([r[key] for r in run_rows if r["family"]==family and
                                   r["k"]==k]) for k in range(6)])
            ax.bar(ks+(.36*(j-.5)),means,width=.36,label=label)
            for k in range(1,5):
                vals=[r[key] for r in run_rows if r["family"]==family and r["k"]==k]
                ax.scatter(np.full(len(vals),ks[k]+.36*(j-.5)),vals,s=9,alpha=.6,color=".2")
        ax.set(xticks=ks,xticklabels=[str(k) for k in ks],xlabel="Number of noisy features (k)",
               ylabel="Total variation in validation BA (percentage points)",
               title=f"{family}: within-window BA variability, 0–30 vs 30–60")
        ax.legend();ax.grid(axis="y",alpha=.2)
        figs[f"19_BA_TV_window_{family}.svg"]=svg(fig)
    for key,name in (("F3","F3"),("NFI3","NFI3")):
        fig,axs=plt.subplots(3,1,figsize=(10.5,9.1),sharex=True)
        for c,ax in enumerate(axs):
            for k in range(1,6):
                values=[]
                for id,curves in sorted(class_curves.items()):
                    if int(id[1])==k:
                        values.append([r[str(c)][key] for r in curves])
                arr=np.array(values,dtype=float)
                for row in arr:
                    if len(arr)>1:ax.plot(range(30,61),row,color=COLORS[k],
                                          alpha=.16,lw=.7)
                ax.plot(range(30,61),arr.mean(axis=0),label=f"k={k}",
                        color=COLORS[k],lw=1.8)
            if key=="NFI3":ax.axhline(1,color=".5",ls=":",lw=.8)
            ax.set(ylabel=f"Candidate class {c}",xlim=(30,60))
            ax.grid(alpha=.16)
        axs[0].legend(ncol=5,fontsize=7)
        axs[0].set_title(f"class: class-conditional {name}, steps 30–60")
        axs[-1].set_xlabel("Cumulative Adam updates")
        figs[f"{'20' if key=='F3' else '21'}_class_conditional_{name}.svg"]=svg(fig)
    return figs

def build():
    old,traces,run_rows,classes,count=audit()
    groups=group_table(run_rows)
    plots=charts(old,traces,run_rows,classes)
    summary={"schema":"e-ch4-14-continuation-analysis-v1","experiment":"E-CH4-14",
       "n_fits":36,"additional_updates":1080,"budget_counts":count,
       "fit_ids":sorted(traces),"group_summary":groups,"run_summary":run_rows,
       "limits":{"k_1_to_4_subsets_are_not_independent_seeds":True,
                 "NFI_k0_undefined":True,
                 "policy3_ignores_masks":True,
                 "no_causal_identification":True,
                 "no_retraining_or_test":True}}
    outputs={
       TAB/"continuation_30_60_by_run.csv":csv_bytes(run_rows),
       TAB/"continuation_30_60_by_noise_level.csv":csv_bytes(groups),
       TAB/"continuation_30_60_summary.json":
          (json.dumps(summary,indent=2,sort_keys=True,allow_nan=False,ensure_ascii=False)+"\n").encode("utf8")}
    for name,content in plots.items():outputs[FIG/name]=content
    return outputs,summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("write","check"),required=True)
    args=parser.parse_args()
    outputs,summary=build()
    for path,blob in outputs.items():
        if args.mode=="write":
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(blob)
        else:
            require(path.is_file() and path.read_bytes()==blob,
                    "Derived analysis differs from reproducible source "+str(path))
    top=sorted(summary["group_summary"],key=lambda r:(r["family"],r["k"]))
    print("E14_CONTINUATION_POSTFIT_AUDIT_PASS",
          json.dumps({"mode":args.mode,"fits":36,"updates":1080,
              "derived_files":len(outputs),"k2_tv":{fam:next(x["ba_tv_30_60_pp_mean"]
                      for x in top if x["family"]==fam and x["k"]==2)
                      for fam in FAMILIES},
              "no_test":True},sort_keys=True))
if __name__=="__main__":main()
