"""Read-only E-CH4-13 Validation BA summary versus generative Bayes MAP.

No optimizer, no model fitting, no Test access, no regenerated synthetic samples.
Only frozen npz Validation arrays from original three-seed input snapshot and
SHA-verified per-run metrics from original twelve-fit manifest are consulted.

Usage: python code/experiments/ch4_loss_controlled_comparison/validation_bayes_summary.py
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SIZES=(20261102,20261103,20261104)
FAMILIES=("shared","class")
LOSSES=("sigmoid_JS_margin_tau005","softplus_JS_tau005")
SOURCE_SHA="cdb45644c245e04a276eb2a419e4d64f71c6c18a"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def balanced(y,pred):
    return float(np.mean([np.mean(pred[y==c]==c) for c in range(3)]))
def stats(v):
    v=np.asarray(v,dtype=np.float64)
    mean=float(v.mean());return {"mean_percent":mean,"sample_variance_pp_squared":float(((v-mean)**2).sum()/(len(v)-1)),
         "sample_standard_deviation_pp":float(v.std(ddof=1)),"n_independent_data_seeds":len(v)}

def bayes_map(x,config):
    means=np.asarray(config["means"],dtype=np.float64)
    std=np.asarray(config["stds"],dtype=np.float64)
    corr=np.asarray(config["correlation"],dtype=np.float64)
    if means.shape!=(3,5) or std.shape!=(3,5) or corr.shape!=(5,5):raise ValueError("Unexpected config dimensions")
    cov=np.array([np.outer(sd,sd)*corr for sd in std])
    logprior=np.log(np.asarray(config["class_priors"],dtype=np.float64))
    predictions=[]
    for sample in x:
        mask=np.isfinite(sample);k=int(mask.sum())
        scores=[]
        for c in range(3):
            d=sample[mask]-means[c,mask]
            sub=cov[c][np.ix_(mask,mask)]
            sign,logdet=np.linalg.slogdet(sub)
            if sign<=0:raise ValueError("Gaussian covariance is not positive definite")
            ll=-.5*(k*math.log(2*math.pi)+logdet+float(d@np.linalg.solve(sub,d)))
            scores.append(float(logprior[c]+ll))
        predictions.append(int(np.argmax(scores)))
    return np.asarray(predictions,dtype=np.int64)

def main():
    raw=HERE/"results/phase2_300_25"
    frozen=HERE/"inputs/phase2_train_validation"
    fm=json.loads((frozen/"manifest.json").read_text())
    rm=json.loads((raw/"manifest.json").read_text())
    config_path=HERE.parents[2]/"data_generator/config.json"
    config=json.loads(config_path.read_text())
    if sha(config_path)!=fm["config_sha256"]:raise AssertionError("Configuration fingerprint mismatch")
    if rm["source_commit"]!=SOURCE_SHA or rm["fit_count"]!=12:raise AssertionError("Unexpected training evidence")
    if fm["seed_set"]!=list(SIZES) or fm["test_accessed"] is not False:raise AssertionError("Bad frozen input manifest")
    map_oracle={}
    for seed in SIZES:
        record=fm["records"][str(seed)];file=frozen/record["file"]
        if sha(file)!=record["file_sha256"]:raise AssertionError("Frozen input checksum mismatch")
        with np.load(file,allow_pickle=False) as v:
            x=np.asarray(v["validation_x"],dtype=np.float64)
            y=np.asarray(v["validation_y"],dtype=np.int64)
        if x.shape!=(90,5) or y.shape!=(90,):raise AssertionError("Unexpected Validation dimensions")
        if np.bincount(y,minlength=3).tolist()!=record["validation_counts"]:raise AssertionError("Validation class counts changed")
        map_oracle[seed]=100*balanced(y,bayes_map(x,config))
    # Historic seed 20261102 Bayes MAP validation BA from prior original oracle audit
    if abs(map_oracle[20261102]-97.44897959183674)>1e-10:raise AssertionError("Historical Bayes reference mismatch")
    models={}
    for a in rm["runs"]:
        if a["data_seed"] not in SIZES or a["family"] not in FAMILIES or a["objective"] not in LOSSES:
            raise AssertionError("Unexpected model in archived grid")
        p=raw/a["run_key"]/"metrics_full.json"
        if sha(p)!=a["metrics_full_sha256"]:raise AssertionError("Original model metrics changed")
        v=json.loads(p.read_text())["validation"]["balanced_accuracy"]
        key=(a["family"],a["objective"],a["data_seed"])
        if key in models:raise AssertionError("Duplicate model")
        models[key]=100*float(v)
    if len(models)!=12:raise AssertionError("Incomplete grid")
    out={"schema":"e-ch4-13-validation-bayes-map-summary-v1",
         "training_source_sha":SOURCE_SHA,"frozen_inputs_sha256":sha(frozen/"manifest.json"),
         "raw_manifest_sha256":sha(raw/"manifest.json"),"no_optimizer":True,"test_accessed":False,
         "variance_convention":"unbiased sample variance over 3 independent data seeds, BA in percent, variance in pp^2",
         "bayes_reference":"MAP posterior from original Gaussian class-conditional generator with observed-feature mask and class priors; not a mathematical upper bound for empirical balanced accuracy",
         "bayes_by_seed_percent":{str(s):map_oracle[s] for s in SIZES},
         "rows":[]}
    oracle_stats=stats([map_oracle[s] for s in SIZES])
    for family in FAMILIES:
        for loss in LOSSES:
            v=[models[(family,loss,s)] for s in SIZES]
            result={"family":family,"objective":loss,"by_seed_percent":dict(zip(map(str,SIZES),v)),**stats(v)}
            result["delta_mean_vs_bayes_pp"]=result["mean_percent"]-oracle_stats["mean_percent"]
            out["rows"].append(result)
    out["rows"].append({"family":"generative_oracle","objective":"Bayes_MAP",**oracle_stats,"delta_mean_vs_bayes_pp":0.0})
    return out

if __name__=="__main__":
    print(json.dumps(main(),ensure_ascii=False,indent=2,sort_keys=True))
