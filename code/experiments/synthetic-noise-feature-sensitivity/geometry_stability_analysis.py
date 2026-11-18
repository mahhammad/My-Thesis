"""E-CH4-14 postfit geometry stability, using canonical 60-step reports only.

No training, no Test, no checkpoint alteration. Outputs are strictly derived.
The sample-wise *representations* are not retained in final-60 canonical metrics; only
sample-wise scores, margins, density distances, labels and predictions are.
Consequently this script never claims to measure sample representation drift.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "tables" / "final_60_canonical_reports.json"
TABLE = HERE / "tables"
IDS = ["N0-01"] + [f"N{k}-{i:02d}" for k in range(1,5)
                     for i in range(1,5)] + ["N5-01"]
FAMILIES = ("shared", "class")
PAIRS = ((0,1),(0,2),(1,2))
EPS = 1e-12


def jsd(p,q):
    p=np.asarray(p,dtype=np.float64)
    q=np.asarray(q,dtype=np.float64)
    assert p.shape == q.shape
    m=(p+q)/2
    def entropy(x):
        return -(np.where(x>0,x*np.log(np.maximum(x,1e-300)),0)).sum(axis=-1)
    return np.maximum(0,entropy(m)-(entropy(p)+entropy(q))/2)


def mean(a):
    return float(np.mean(np.asarray(a,dtype=float)))


def quant(a):
    return {f"q{int(q*100):02d}":float(np.quantile(a,q))
            for q in (0,.1,.25,.5,.75,.9,1)}


def record(run,base,split):
    a=run[split]; b=base[split]
    pa=a["per_sample"]; pb=b["per_sample"]
    labels=np.asarray(pa["labels"],dtype=int)
    assert np.array_equal(labels,np.asarray(pb["labels"],dtype=int))
    n=len(labels)
    assert n in (300,90) and set(labels)=={0,1,2}
    pred=np.asarray(pa["predicted"],dtype=int)
    pbase=np.asarray(pb["predicted"],dtype=int)
    margins=np.asarray(pa["margins"],dtype=float)
    basem=np.asarray(pb["margins"],dtype=float)
    scores=np.asarray(pa["scores_all_candidates"],dtype=float)
    basescores=np.asarray(pb["scores_all_candidates"],dtype=float)
    assert scores.shape==basescores.shape==(n,3)
    flips=pred!=pbase
    old_correct=pbase==labels
    new_correct=pred==labels
    cross={"wrong_to_right":int(np.sum(~old_correct & new_correct)),
           "right_to_wrong":int(np.sum(old_correct & ~new_correct)),
           "right_to_right":int(np.sum(old_correct & new_correct)),
           "wrong_to_wrong":int(np.sum(~old_correct & ~new_correct))}
    out={"n":n,"BA":a["balanced_accuracy"],
         "BA_reference":b["balanced_accuracy"],
         "delta_BA":a["balanced_accuracy"]-b["balanced_accuracy"],
         "prediction_flips":int(flips.sum()),
         "prediction_flip_rate":mean(flips),
         "outcome_transitions":cross,
         "mean_absolute_margin_shift":mean(abs(margins-basem)),
         "margin_shift_quantiles":quant(margins-basem),
         "mean_abs_candidate_score_shift":mean(abs(scores-basescores)),
         "mean_max_candidate_score_shift":mean(np.max(abs(scores-basescores),axis=1)),
         "per_class":[]}
    for metric in ("R_KL","R_JS","R_sqrt_JS"):
        current=np.asarray(pa[metric],dtype=float)
        previous=np.asarray(pb[metric],dtype=float)
        assert len(current)==len(previous)==n
        out[metric+"_delta"]=float(a[metric]-b[metric])
        out[metric+"_per_sample_delta_quantiles"]=quant(current-previous)
        out[metric+"_mean_abs_sample_shift"]=mean(abs(current-previous))
    for c in range(3):
        sel=labels==c
        out["per_class"].append({
            "class":c,
            "count":int(sel.sum()),
            "prediction_flip_rate":mean(flips[sel]),
            "mean_abs_margin_shift":mean(abs(margins[sel]-basem[sel])),
            **{metric+"_delta":mean(np.asarray(pa[metric])[sel])-mean(np.asarray(pb[metric])[sel])
               for metric in ("R_KL","R_JS","R_sqrt_JS")}
        })
    return out


def geometry(run,base):
    v=np.asarray(run["centers"],dtype=float)
    v0=np.asarray(base["centers"],dtype=float)
    assert v.shape==v0.shape==(3,3)
    assert np.allclose(v.sum(axis=1),1,atol=1e-9)
    drift=np.sqrt(jsd(v,v0))
    pairs=np.array([np.sqrt(jsd(v[i],v[j])) for i,j in PAIRS])
    basepairs=np.array([np.sqrt(jsd(v0[i],v0[j])) for i,j in PAIRS])
    geom=run["train"]["center_geometry"]
    basegeom=base["train"]["center_geometry"]
    res={"center_drift_sqrt_JSD":drift.tolist(),
         "mean_center_drift":mean(drift),
         "max_center_drift":float(max(drift)),
         "center_pairwise_sqrt_JSD":pairs.tolist(),
         "reference_pairwise_sqrt_JSD":basepairs.tolist(),
         "pairwise_distance_shift":(pairs-basepairs).tolist(),
         "mean_absolute_pairwise_distance_shift":mean(abs(pairs-basepairs)),
         "min_pairwise_distance_shift":float(np.min(pairs)-np.min(basepairs)),
         "center_coordinates":v.tolist()}
    for metric in ("T","BT","TNL","BTNL","I"):
        x=geom.get(metric,run["train"].get(metric))
        y=basegeom.get(metric,base["train"].get(metric))
        res["delta_"+metric]=(float(x)-float(y)
                                if x is not None and y is not None else None)
    return res


def build():
    reports=json.loads(REPORTS.read_text("utf-8"))
    assert len(reports)==36
    for subset in IDS:
        for family in FAMILIES:
            name=f"{subset}__{family}"
            report=reports[name]
            assert report["schema"]=="e-ch4-14-metrics-v1"
            assert report["test_accessed"] is False
            assert report["center_source"]=="train_only"
            assert report["subset_id"]==subset and report["family"]==family
            assert report["subset_id"]==subset
    rows=[]
    for subset in IDS:
        for family in FAMILIES:
            name=f"{subset}__{family}"
            report=reports[name]
            ref=reports[f"N0-01__{family}"]
            rows.append({"run_id":name,"family":family,
                         "noise_count":int(subset[1]),"reference_run_id":f"N0-01__{family}",
                         "center_geometry":geometry(report,ref),
                         "train":record(report,ref,"train"),
                         "validation":record(report,ref,"validation")})
    assert len(rows)==36
    for row in rows:
        if row["noise_count"]==0:
            assert abs(row["center_geometry"]["mean_center_drift"])<EPS
            assert row["train"]["prediction_flips"]==0
            assert row["validation"]["prediction_flips"]==0
    compact=[]
    for r in rows:
        g=r["center_geometry"]
        item={"run_id":r["run_id"],"family":r["family"],
              "noise_count":r["noise_count"],
              "mean_center_drift":g["mean_center_drift"],
              "max_center_drift":g["max_center_drift"],
              "mean_abs_pairwise_shift":g["mean_absolute_pairwise_distance_shift"],
              "delta_I":g["delta_I"]}
        for j,x in enumerate(g["center_drift_sqrt_JSD"]):
            item[f"center_{j}_drift"]=x
        for j,x in enumerate(g["pairwise_distance_shift"]):
            item[f"pair_{PAIRS[j][0]}{PAIRS[j][1]}_distance_shift"]=x
        for split in ("train","validation"):
            z=r[split]
            for metric in ("delta_BA","prediction_flip_rate",
                           "mean_absolute_margin_shift","mean_abs_candidate_score_shift",
                           "R_JS_delta","R_KL_delta","R_sqrt_JS_delta"):
                item[f"{split}_{metric}"]=z[metric]
        compact.append(item)
    grouped=[]
    for family in FAMILIES:
        for k in range(6):
            sel=[r for r in compact if r["family"]==family and r["noise_count"]==k]
            assert len(sel)==(1 if k in (0,5) else 4)
            d={"family":family,"noise_count":k,"n_dependent_subsets":len(sel)}
            for key in compact[0]:
                if key in ("run_id","family","noise_count"):
                    continue
                vals=[v[key] for v in sel]
                if all(isinstance(v,(int,float)) for v in vals):
                    d[key+"_mean"]=mean(vals)
                    d[key+"_min"]=float(min(vals))
                    d[key+"_max"]=float(max(vals))
            grouped.append(d)
    return rows,compact,grouped


def csvbytes(rows):
    output=io.StringIO(newline="")
    writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def artifacts():
    rows,compact,grouped=build()
    data={"schema":"e-ch4-14-postfit-geometry-stability-v1",
          "source":"tables/final_60_canonical_reports.json; independently replayed step-60 outputs",
          "reference":"N0-01 within same policy family",
          "note":"Sample representations not stored; score/density shifts are NOT JSD representation drift",
          "n_runs":len(rows),"runs":rows}
    return {
        TABLE/"geometry_stability_by_run_60.json":(
            json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode("utf-8"),
        TABLE/"geometry_stability_by_run_60.csv":csvbytes(compact),
        TABLE/"geometry_stability_by_noise_level_60.csv":csvbytes(grouped)
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("check","write"),default="check")
    args=parser.parse_args()
    products=artifacts()
    if args.mode=="write":
        TABLE.mkdir(parents=True,exist_ok=True)
        for p,b in products.items():
            p.write_bytes(b)
    else:
        for p,b in products.items():
            assert p.exists() and p.read_bytes()==b,f"Missing/stale derived output: {p}"
    print("E14_GEOMETRY_STABILITY_PASS",len(products))


if __name__=="__main__":
    main()
