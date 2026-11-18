"""Read-only 36-checkpoint replay of sample distribution movement at step 60.

Unlike the canonical metric bundle, this replay explicitly compares the full
K-dimensional distributions, not only scores or distances to class centers.
Never calls fit/optimizer; only frozen Train/Validation data and saved logits.
"""
from __future__ import annotations
import argparse,csv,io,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from runner import ExactMaskedWalk,BETA,LAMBDA,H
from e14_metrics import by_hypothesis
from resume_30_60 import read_frozen_dataset
from data import file_sha

HERE=Path(__file__).resolve().parent
TABLE=HERE/"tables"
RAW=HERE/"results/continuation_30_60"
IDS=["N0-01"]+[f"N{k}-{i:02d}" for k in range(1,5) for i in range(1,5)]+["N5-01"]
FAMILIES=("shared","class")

def jsd(p,q):
    m=(p+q)*.5
    def ent(v):return -np.where(v>0,v*np.log(np.maximum(v,1e-300)),0).sum(axis=-1)
    return np.maximum(0,ent(m)-.5*(ent(p)+ent(q)))

def csvbytes(rows):
    s=io.StringIO(newline="")
    w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows)
    return s.getvalue().encode("utf-8")

def build():
    meta=json.loads((HERE/"inputs/derived/manifest.json").read_text("utf8"))
    mapping={x["subset_id"]:x for x in meta["datasets"]}
    assert len(mapping)==18 and meta["test_accessed"] is False
    reports=json.loads((TABLE/"final_60_canonical_reports.json").read_text("utf8"))
    assert len(reports)==36
    data={}
    byrun=[]
    bysample=[]
    for subset in IDS:
        raw=read_frozen_dataset(mapping[subset])
        for family in FAMILIES:
            run=f"{subset}__{family}"
            cp=RAW/run/"step_60.pt"
            expected=(RAW/run/"step_60.pt.sha256").read_text("utf8").split()[0]
            assert file_sha(cp)==expected,f"Checkpoint SHA: {run}"
            pack=torch.load(cp,map_location="cpu",weights_only=True)
            assert int(pack["step"])==60
            z=pack["logits"]
            assert z.dtype==torch.float64 and torch.isfinite(z).all()
            graph=ExactMaskedWalk(raw["train_x"],raw["train_y"],raw["gamma"],
                                  beta=BETA,lam=LAMBDA,horizon=H,
                                  query_x=raw["validation_x"])
            train,val=by_hypothesis(graph,z,family)
            saved=reports[run]
            centers=np.stack([train[raw["train_y"]==c,c,:].mean(0) for c in range(3)])
            assert np.allclose(centers,saved["centers"],atol=3e-8,rtol=1e-8)
            for split,representations in (("train",train),("validation",val)):
                y=np.asarray(raw[split+"_y"],dtype=int)
                stored=saved[split]["per_sample"]["labels"]
                assert np.array_equal(y,np.asarray(stored))
                assert np.allclose(representations.sum(axis=-1),1,atol=2e-8)
                assert np.all(representations>=-1e-12)
                key=(family,split)
                if subset=="N0-01":
                    data[key]=representations.copy()
                reference=data[key]
                assert reference.shape==representations.shape
                per_h=np.sqrt(jsd(representations,reference))
                true=per_h[np.arange(len(y)),y]
                mean_all=per_h.mean(axis=1)
                quantiles=np.quantile(true,[0,.1,.25,.5,.75,.9,1])
                byrun.append({"run_id":run,"family":family,
                    "noise_count":int(subset[1]),"split":split,
                    "mean_true_hypothesis_sqrt_JSD":float(true.mean()),
                    "median_true_hypothesis_sqrt_JSD":float(np.median(true)),
                    "p90_true_hypothesis_sqrt_JSD":float(quantiles[5]),
                    "max_true_hypothesis_sqrt_JSD":float(max(true)),
                    "mean_all_hypotheses_sqrt_JSD":float(mean_all.mean()),
                    **{f"class_{c}_mean_true_sqrt_JSD":float(true[y==c].mean())
                       for c in range(3)}})
                for i in range(len(y)):
                    bysample.append({"run_id":run,"family":family,
                        "noise_count":int(subset[1]),"split":split,
                        "sample_index":i,"true_class":int(y[i]),
                        "true_hypothesis_sqrt_JSD":float(true[i]),
                        "mean_all_hypotheses_sqrt_JSD":float(mean_all[i]),
                        **{f"hypothesis_{c}_sqrt_JSD":float(per_h[i,c]) for c in range(3)}})
    assert len(byrun)==72 and len(bysample)==36*(300+90)
    assert all(r["mean_true_hypothesis_sqrt_JSD"]<1e-9
               for r in byrun if r["noise_count"]==0)
    groups=[]
    for fam in FAMILIES:
        for split in ("train","validation"):
            for k in range(6):
                sel=[r for r in byrun if r["family"]==fam and r["split"]==split
                     and r["noise_count"]==k]
                assert len(sel)==(1 if k in (0,5) else 4)
                x=np.array([r["mean_true_hypothesis_sqrt_JSD"] for r in sel])
                groups.append({"family":fam,"split":split,"noise_count":k,
                    "n_dependent_subsets":len(sel),"mean":float(x.mean()),
                    "min":float(x.min()),"max":float(x.max())})
    return byrun,bysample,groups

def artifacts():
    r,s,g=build()
    return {TABLE/"representation_drift_by_run_60.csv":csvbytes(r),
            TABLE/"representation_drift_by_sample_60.csv":csvbytes(s),
            TABLE/"representation_drift_by_noise_level_60.csv":csvbytes(g)}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("write","check"),default="check")
    mode=parser.parse_args().mode
    for p,data in artifacts().items():
        if mode=="write":p.write_bytes(data)
        else:assert p.is_file() and p.read_bytes()==data,p
    print("E14_EXACT_REPRESENTATION_DRIFT_PASS")
