"""E-CH4-14: single sixty-update experiment, final-60 canonical metrics.

Recalculate class centers, geometry and classification from frozen step-60
checkpoints. All 36 full 0..60 training curves are reconstructed from recorded
immutable traces. Supplemental F3/E3/NFI3 begin at step30; no backfill.
No optimization step, no Test, no modification of source artifacts.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,pathlib
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from continuation_analysis import audit,require,read_json,sha
from resume_30_60 import read_frozen_dataset
from e14_metrics import report_graph
from runner import ExactMaskedWalk,BETA,LAMBDA,H

HERE=pathlib.Path(__file__).resolve().parent
TABLE=HERE/"tables"
FIG=HERE/"figures"
METRICS=("train_BA","validation_BA","train_Accuracy","validation_Accuracy",
         "validation_R_JS","train_I","validation_median_margin",
         "validation_hard_margin_error","train_R_JS","train_T","train_BT",
         "train_TNL","train_BTNL")
FAMILIES=("shared","class")
COLORS=plt.get_cmap("tab10").colors
plt.rcParams.update({"font.size":8,"svg.hashsalt":"e14-unified60-final-v1",
                     "figure.dpi":110,"savefig.dpi":110})
torch.set_num_threads(1)

def b(data):
    return (json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False,
                       allow_nan=False)+"\n").encode("utf8")

def csv_data(rows):
    buf=io.StringIO(newline="")
    writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    return buf.getvalue().encode()

def stats(values):
    arr=np.asarray(values,dtype=float)
    require(len(arr)>0 and np.isfinite(arr).all(),"Invalid grouped values")
    return {"mean":float(arr.mean()),"min":float(arr.min()),
            "max":float(arr.max()),
            "subset_sd":float(arr.std(ddof=1)) if len(arr)>1 else None}

def tv(arr):
    return float(np.abs(np.diff(arr)).sum())

def figure(fig):
    f=io.BytesIO()
    fig.savefig(f,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    return f.getvalue()

def load_step60_reports(traces):
    manifest=read_json(HERE/"inputs/derived/manifest.json")
    mapping={r["subset_id"]:r for r in manifest["datasets"]}
    reports={}
    for id in sorted(traces):
        subset,family=id.split("__")
        row=mapping[subset]
        raw=read_frozen_dataset(row)
        ck=HERE/"results/continuation_30_60"/id/"step_60.pt"
        zpack=torch.load(ck,map_location="cpu",weights_only=True)
        require(zpack["step"]==60,"Nonfinal checkpoint "+id)
        z=zpack["logits"]
        require(z.dtype==torch.float64 and torch.isfinite(z).all(),"Invalid logits "+id)
        graph=ExactMaskedWalk(raw["train_x"],raw["train_y"],raw["gamma"],
                 beta=BETA,lam=LAMBDA,horizon=H,query_x=raw["validation_x"])
        origin=read_json(HERE/"results/metrics"/(id+".json"))
        require(origin["test_accessed"] is False and origin["family"]==family,
                "Origin report mismatch "+id)
        calculated=report_graph(graph,z,family,raw["train_y"],raw["validation_y"],
                    data_hashes=origin["data_hashes"],subset_id=subset)
        last=traces[id][-1]
        for split,key_ba,key_accuracy in (("train","train_BA","train_Accuracy"),
                                           ("validation","validation_BA","validation_Accuracy")):
            s=calculated[split]
            require(np.isclose(s["balanced_accuracy"],last[key_ba],atol=2e-11,rtol=2e-8)
                    and np.isclose(s["ordinary_accuracy"],last[key_accuracy],
                                   atol=2e-11,rtol=2e-8),
                    "Step60 snapshot vs canonical mismatch "+id+" "+split)
        reports[id]=calculated
    require(len(reports)==36,"Exactly 36 terminal reports needed")
    return reports

def metrics_of(report):
    tr=report["train"];va=report["validation"]
    return dict(
       train_BA=tr["balanced_accuracy"],validation_BA=va["balanced_accuracy"],
       train_Accuracy=tr["ordinary_accuracy"],validation_Accuracy=va["ordinary_accuracy"],
       train_R_JS=tr["R_JS"],validation_R_JS=va["R_JS"],
       train_I=tr["I"],validation_median_margin=va["margin_quantiles"]["0.5"],
       validation_hard_margin_error=va["hard_margin_error_conservative"],
       train_T=tr["T"],train_BT=tr["BT"],train_TNL=tr["TNL"],train_BTNL=tr["BTNL"])

def assemble(traces,reports):
    per=[]; per_map={}
    for id,trace in sorted(traces.items()):
        subset,family=id.split("__");k=int(subset[1])
        rep=reports[id];m=metrics_of(rep)
        require(len(trace)==61 and [r["step"] for r in trace]==list(range(61)),
                "Incomplete 0..60 trace "+id)
        ba=np.array([r["validation_BA"] for r in trace],dtype=float)
        loss=np.array([r["train_softplus_loss"] for r in trace],dtype=float)
        grad=np.array([r["gradient_l2"] for r in trace],dtype=float)
        s=dict(run_id=id,subset_id=subset,family=family,k=k,
               **m,ba_tv_0_60_pp=100*tv(ba),
               ba_max_drawdown_0_60_pp=float(100*np.max(np.maximum.accumulate(ba)-ba)),
               loss_total_variation_0_60=tv(loss),gradient_total_variation_0_60=tv(grad),
               gradient_start=grad[0],gradient_final=grad[-1],
               loss_start=loss[0],loss_final=loss[-1],
               ba_start=ba[0],ba_final=ba[-1],
               ba_down_steps=int(np.count_nonzero(np.diff(ba)<-1e-12)),
               total_adam_updates=60,
               policy_parameter_count=int(np.prod(reports[id]["centers"][0].__len__() * [1])) if False else None)
        origin=read_json(HERE/"results/metrics"/(id+".json"))
        s["policy_parameter_count"]=origin["resource_usage"]["policy_parameter_count"]
        s["compute_time_reported_s"]=float(origin["resource_usage"]["job_compute_elapsed_s"]+
             trace[-1]["wall_elapsed_s"])
        s["source_step60_sha256"]=sha(HERE/"results/continuation_30_60"/id/"step_60.pt")
        s["validation_class0_recall"]=rep["validation"]["per_class"][0]["accuracy"]
        s["validation_class1_recall"]=rep["validation"]["per_class"][1]["accuracy"]
        s["validation_class2_recall"]=rep["validation"]["per_class"][2]["accuracy"]
        s["validation_pred_flips_vs_no_noise"]=None
        per.append(s);per_map[id]=s
    for id,rep in reports.items():
        subset,family=id.split("__")
        expected=rep["validation"]["per_sample"]
        base=reports["N0-01__"+family]["validation"]["per_sample"]
        require(expected["labels"]==base["labels"],
                "Validation row alignment mismatch "+id)
        per_map[id]["validation_pred_flips_vs_no_noise"]=sum(
            a!=b for a,b in zip(expected["predicted"],base["predicted"]))
    groups=[]
    for family in FAMILIES:
        for k in range(6):
            selected=[r for r in per if r["family"]==family and r["k"]==k]
            require(len(selected)==(1 if k in (0,5) else 4),"Group count incorrect")
            out=dict(family=family,k=k,n_subsets=len(selected))
            for key in (METRICS+("ba_tv_0_60_pp","ba_max_drawdown_0_60_pp",
                      "loss_total_variation_0_60","gradient_total_variation_0_60",
                      "gradient_final","compute_time_reported_s",
                      "policy_parameter_count","validation_pred_flips_vs_no_noise",
                      "validation_class0_recall","validation_class1_recall","validation_class2_recall")):
                summary=stats([v[key] for v in selected])
                for stat,number in summary.items():out[key+"_"+stat]=number
            groups.append(out)
    paired=[]
    for subset in sorted(set(r["subset_id"] for r in per)):
        a=per_map[subset+"__class"];s=per_map[subset+"__shared"]
        row=dict(subset_id=subset,k=a["k"])
        for key in METRICS:
            row[key+"_class_minus_shared"]=a[key]-s[key]
        paired.append(row)
    return per,groups,paired

def groups_for(group,family,key):
    return np.array([next(g for g in group if g["family"]==family and g["k"]==k)
                 [key+"_mean"] for k in range(6)],dtype=float)

def draw_traces(ax,traces,family,key,scale=1, *,legend=True):
    for k in range(6):
        data=[]
        for id,t in sorted(traces.items()):
            if id.endswith("__"+family) and int(id[1])==k:
                data.append([np.nan if r[key] is None else r[key]*scale for r in t])
        arr=np.asarray(data,dtype=float)
        for line in arr:
            if len(arr)>1:ax.plot(np.arange(61),line,alpha=.15,lw=.6,color=COLORS[k])
        ax.plot(np.arange(61),np.nanmean(arr,axis=0),lw=1.9,color=COLORS[k],
                label=f"k={k} (n={len(arr)})")
    ax.set_xlim(0,60);ax.grid(alpha=.18)
    if legend:ax.legend(ncol=3,fontsize=7)

def figures(traces,per,groups,paired):
    outs={}
    # Full 61-point traces are the PRIMARY graphs, one per family.
    for family in FAMILIES:
        fig,axs=plt.subplots(4,1,figsize=(11,11),sharex=True)
        for ax,key,scale,label in zip(axs,("validation_BA","train_softplus_loss",
                           "gradient_l2","adam_update_l2"),(100,1,1,1),
                         ("Validation BA (%)","Train Softplus-JS loss",
                          "Gradient L2","Adam update L2")):
            draw_traces(ax,traces,family,key,scale=scale,legend=(key=="validation_BA"))
            ax.set_ylabel(label)
        axs[0].set_title(f"{family}: single uninterrupted 60-update experiment")
        axs[-1].set_xlabel("Cumulative Adam update (0–60)")
        outs[f"22_full_60_learning_{family}.svg"]=figure(fig)
        fig,axs=plt.subplots(2,1,figsize=(11,6.9),sharex=True)
        draw_traces(axs[0],traces,family,"train_BA",100)
        draw_traces(axs[1],traces,family,"validation_BA",100,legend=False)
        axs[0].set_ylabel("Train BA (%)");axs[1].set_ylabel("Validation BA (%)")
        axs[0].set_title(f"{family}: balanced accuracy, full 0–60")
        axs[1].set_xlabel("Cumulative Adam update")
        outs[f"23_train_validation_BA_60_{family}.svg"]=figure(fig)
        fig,axs=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
        for k,ax in enumerate(axs.flat):
            members=[(id,trace) for id,trace in sorted(traces.items())
                     if id.endswith("__"+family) and int(id[1])==k]
            for idx,(id,t) in enumerate(members):
                ax.plot(np.arange(61),[100*r["validation_BA"] for r in t],
                        alpha=.55,lw=1,label=id.split("__")[0])
            if len(members)>1:
                avg=np.mean([[100*r["validation_BA"] for r in t] for _,t in members],axis=0)
                ax.plot(np.arange(61),avg,lw=2.5,color="black",label="subset mean")
            ax.set_title(f"k={k}, subsets={len(members)}")
            ax.grid(alpha=.16);ax.set_xlim(0,60)
            if k in (0,3):ax.set_ylabel("Validation BA (%)")
            if k>=3:ax.set_xlabel("Adam update (0–60)")
        axs[0,0].legend(fontsize=6)
        fig.suptitle(f"{family}: all individual subset trajectories, 0–60")
        outs[f"24_all_subsets_60_{family}.svg"]=figure(fig)
    k=np.arange(6);width=.34
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,fam in zip(axs,FAMILIES):
        ba=100*groups_for(groups,fam,"validation_BA")
        acc=100*groups_for(groups,fam,"validation_Accuracy")
        ax.bar(k-width/2,ba,width,label="BA")
        ax.bar(k+width/2,acc,width,label="Accuracy")
        ax.set(title=fam,xticks=k,xlabel="Noisy feature count k",ylim=(75,102))
        ax.grid(axis="y",alpha=.18)
    axs[0].set_ylabel("Validation score at step 60 (%)")
    axs[0].legend()
    fig.suptitle("Step-60 classification quality (all 36 fits)")
    outs["25_final_60_BA_accuracy.svg"]=figure(fig)
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.2))
    for ax,key,label,scale in zip(axs,("train_I","validation_R_JS",
                           "validation_median_margin"),
                                ("Train center separation I","Validation density R_JS",
                                 "Validation median margin"),(1000,1000,1000)):
        for fam in FAMILIES:
            vals=groups_for(groups,fam,key)*scale
            ax.plot(k,vals,marker="o",lw=1.7,label=fam)
        ax.set(xlabel="Noisy features (k)",ylabel=label+" ×1000",xticks=k)
        ax.grid(alpha=.2)
    axs[0].legend()
    fig.suptitle("Step-60 Train-only centers and Validation decision geometry")
    outs["26_final_60_geometry_margin.svg"]=figure(fig)
    fig,ax=plt.subplots(figsize=(11,4.7))
    x=np.arange(len(paired))
    differences=np.array([100*r["validation_BA_class_minus_shared"] for r in paired])
    ax.bar(x,differences)
    ax.axhline(0,ls=":",color=".4")
    ax.set(xticks=x,xticklabels=[r["subset_id"] for r in paired],xlabel="Paired subset",
           ylabel="Class - shared Validation BA (percentage points)",
           title="Step-60 paired decision differences")
    plt.setp(ax.get_xticklabels(),rotation=55,ha="right",fontsize=7)
    ax.grid(axis="y",alpha=.15)
    outs["27_final_60_paired_family.svg"]=figure(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,fam in zip(axs,FAMILIES):
        means=groups_for(groups,fam,"validation_pred_flips_vs_no_noise")
        ax.bar(k,means)
        ax.set(xticks=k,title=fam,xlabel="Noisy features (k)")
        ax.grid(axis="y",alpha=.14)
    axs[0].set_ylabel("Changed Validation predictions vs k=0 (of 90)")
    fig.suptitle("Step-60 prediction stability against noiseless subset")
    outs["28_final_60_prediction_flips.svg"]=figure(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,fam in zip(axs,FAMILIES):
        for cl in range(3):
            vals=groups_for(groups,fam,f"validation_class{cl}_recall")*100
            ax.plot(k,vals,marker="o",label=f"Class {cl}")
        ax.set(xticks=k,xlabel="Noisy features (k)",title=fam)
        ax.grid(alpha=.18)
    axs[0].set_ylabel("Validation class recall (%)")
    axs[0].legend()
    fig.suptitle("Step-60 per-class recall, no Test")
    outs["29_final_60_class_recall.svg"]=figure(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for ax,key,label in zip(axs,("policy_parameter_count","compute_time_reported_s"),
                            ("Trainable logits","Summed job compute seconds (two segments)")):
        for fam in FAMILIES:
            ax.plot(k,groups_for(groups,fam,key),marker="o",label=fam)
        ax.set(xlabel="Noisy features (k)",ylabel=label,xticks=k)
        ax.grid(alpha=.2)
    axs[0].legend()
    fig.suptitle("60-step parameter count and recorded compute work")
    outs["30_final_60_cost.svg"]=figure(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,fam in zip(axs,FAMILIES):
        means=groups_for(groups,fam,"ba_tv_0_60_pp")
        ax.bar(k,means)
        for level in range(1,5):
            arr=[r["ba_tv_0_60_pp"] for r in per if r["family"]==fam and r["k"]==level]
            ax.scatter(np.full(len(arr),level),arr,color=".2",s=13,zorder=3,alpha=.7)
        ax.set(xticks=k,title=fam,xlabel="Noisy feature count (k)")
        ax.grid(axis="y",alpha=.17)
    axs[0].set_ylabel("TV(Validation BA), complete 0–60 (percentage points)")
    fig.suptitle("One full 60-update experiment; not two 30-update experiments")
    outs["31_full_60_BA_variation.svg"]=figure(fig)
    return outs

def build():
    _,traces,_,_,count=audit()
    require(count=={"BUDGET_RESERVED":1,"START":36,"UPDATE":1080,"SUCCESS":36},
            "Ledger incorrect")
    reports=load_step60_reports(traces)
    individual,group,paired=assemble(traces,reports)
    fig=figures(traces,individual,group,paired)
    summary={"schema":"e-ch4-14-single-60-update-v1","total_updates":2160,
        "updates_per_fit":60,"fits":36,"split_used":["train","validation"],
        "test_accessed":False,"single_experiment":True,
        "new_policy_metrics_available_steps":[30,60],
        "grouped":group,"individual":individual,"paired":paired,
        "note":"k=1..4 are dependent feature subsets, not independent seeds"}
    outputs={
        TABLE/"final_60_by_run.csv":csv_data(individual),
        TABLE/"final_60_by_noise_level.csv":csv_data(group),
        TABLE/"final_60_paired.csv":csv_data(paired),
        TABLE/"final_60_summary.json":b(summary),
        TABLE/"final_60_canonical_reports.json":b(reports)}
    for name,blob in fig.items():outputs[FIG/name]=blob
    return outputs,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("check","write"),required=True)
    args=p.parse_args()
    output,summary=build()
    for path,blob in output.items():
        if args.mode=="write":
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(blob)
        else:
            require(path.is_file() and path.read_bytes()==blob,
                    "Derived final60 artifact changed: "+str(path))
    print("E14_UNIFIED_60_READ_ONLY_PASS",
          json.dumps({"mode":args.mode,"fits":summary["fits"],
                      "total_updates":summary["total_updates"],
                      "files":len(output),"no_test":True}))
if __name__=="__main__":main()
