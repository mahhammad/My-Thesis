#!/usr/bin/env python3
"""RESEARCHER-03: verify 76 center metrics, preserve prior 36 and emit candidate coverage.

No computation of per-sample representations, BA, NLL, R or margins.
Inputs: all-archive preflight, checksum-guarded center-only script output,
        previous immutable 36-member verified evidence and 76-run coverage.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json, math
from pathlib import Path
from collections import Counter
from center_only_rebuild import EXPECTED_ARCHIVES

HERE=Path(__file__).resolve().parent
OUT=HERE/"full_76_candidate"
OLD=HERE/"center_only_metrics.csv"
SOURCE=HERE/"verified_sources_36.json"
PREVIOUS=HERE/"coverage_manifest.json"
MAIN=OUT/"center_only_metrics.csv"
REQUIRED=("T","BT","TNL","BTNL","I","min_pairwise_sqrt_JS",
    "mean_pairwise_sqrt_JS","mean_pairwise_JS","nearest_center_sqrt_JS_per_class",
    "pairwise_JS","pairwise_sqrt_JS","pairwise_JS_similarity")
COMPARISON_FIELDS=REQUIRED+("legacy_I","legacy_min_pairwise_sqrt_JS","legacy_mean_pairwise_JS")

def read_csv(path:Path):
    with path.open(encoding="utf8",newline="") as stream:
        return list(csv.DictReader(stream))

def typed(cell):
    if cell=="" or cell is None:return None
    if cell[0]=="[":return json.loads(cell)
    return float(cell)

def delta(a,b):
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):raise ValueError("Array length mismatch")
        return max((delta(x,y) for x,y in zip(a,b)),default=0.0)
    if a is None or b is None:
        return 0.0 if a==b else math.inf
    x,y=float(a),float(b)
    if not math.isfinite(x) or not math.isfinite(y):
        return 0.0 if x==y else math.inf
    return abs(x-y)

def main():
    preflight=json.loads((OUT/"archive_preflight.json").read_text(encoding="utf8"))
    if preflight["run_count"]!=76 or preflight["archive_count"]!=4:
        raise ValueError("All four archive checks required before candidate verification")
    if {a["source_archive"]:a["n"] for a in preflight["archives"]} != {
        k:v[1] for k,v in EXPECTED_ARCHIVES.items()}:
        raise ValueError("Archive member counts disagreed")
    member={}
    for archive in preflight["archives"]:
        if not archive["sha256_verified"]:raise ValueError("Unverified archive")
        for m in archive["members"]:
            if m["run_id"] in member:raise ValueError("Duplicate original member")
            member[m["run_id"]]=(archive,m)
    fresh=read_csv(MAIN)
    old=read_csv(OLD)
    raw=json.loads(SOURCE.read_text(encoding="utf8"))
    original=json.loads(PREVIOUS.read_text(encoding="utf8"))
    if len(fresh)!=76 or len(old)!=36 or len(raw["items"])!=36 or len(original["runs"])!=76:
        raise ValueError("Unexpected historical/new row count")
    mnew={r["run_id"]:r for r in fresh}
    mold={r["run_id"]:r for r in old}
    if len(mnew)!=76 or len(mold)!=36 or set(mnew)!=set(member):
        raise ValueError("RUN_ID mismatch/duplicate")
    checks={k:0.0 for k in COMPARISON_FIELDS}
    for oldrun in raw["items"]:
        name=oldrun["run_id"]
        if name not in mnew or name not in mold:
            raise ValueError("Lost one of the previously verified 36")
        if oldrun["member_sha256"]!=mnew[name]["member_sha256"]:
            raise ValueError("Previously-verified member fingerprint changed "+name)
    for name,oldrow in mold.items():
        newrow=mnew[name]
        for key in ("family","objective","data_seed","n_ref","steps",
                    "source_archive","source_member","member_sha256"):
            if oldrow[key]!=newrow[key]:
                raise ValueError("Old 36 metadata mismatch "+name+" "+key)
        for key in COMPARISON_FIELDS:
            diff=delta(typed(oldrow[key]),typed(newrow[key]))
            checks[key]=max(checks[key],diff)
            if diff>1e-10:
                raise ValueError(f"PREVIOUS_36_REGRESSION run={name} metric={key} delta={diff}")
    fullcov=json.loads(PREVIOUS.read_text(encoding="utf8"))
    runindex={r["run_id"]:r for r in fullcov["runs"]}
    assert set(runindex)==set(mnew)
    for name,row in mnew.items():
        previous=runindex[name]
        archive,m=member[name]
        if archive["source_archive"]!=row["source_archive"] or m["sha256"]!=row["member_sha256"]:
            raise ValueError("Candidate/archive mismatch "+name)
        if previous["member_sha256_declared"]!=row["member_sha256"]:
            raise ValueError("Master/ZIP member digest mismatch "+name)
        if str(previous["training_steps"])!=row["steps"] or str(previous["data_seed"])!=row["data_seed"]:
            raise ValueError("Candidate / original metadata mismatch "+name)
        if previous["family"]!=row["family"] or previous["objective"]!=row["objective"]:
            raise ValueError("Family/objective mismatch "+name)
        previous["source_archive_sha256_verified"]=True
        previous["source_archive_sha256_declared"]=archive["sha256"]
        previous["source_member"]=m["member"]
        previous["member_sha256_verified"]=True
        previous["source_evidence"]="original_archive_bytes_sha_verified_center_only"
        previous["metric_contract_version"]="legacy-v1-plus-center-only-v2"
        for key in REQUIRED:
            value=typed(row[key])
            if value is None or delta(value,value)!=0:
                raise ValueError("Invalid center metric "+name+" "+key)
            slot="center."+key
            if slot in previous["metrics"]:
                entry=previous["metrics"][slot]
            elif key=="pairwise_JS_similarity":
                # Optional center-only diagnostic, not part of the 55 metric IDs
                # in chapter4-common-evaluation-contract-v2. Do not fabricate
                # a required contract field or silently change the schema.
                entry=previous.setdefault("auxiliary_center_metrics",{}).setdefault(key,{})
            else:
                raise ValueError("Unexpected missing contract metric "+slot)
            entry["status"]="measured"
            entry["reason"]="center_recomputed_from_sha_verified_original_checkpoint"
            entry["metric_version"]="chapter4-common-evaluation-contract-v2:center_only"
            entry["value"]=value
    fullcov["schema"]="researcher03-benchmark-v1-center-only-coverage-v2-76-candidate"
    fullcov["source"]="four SHA-verified original ZIP archives + historical master CSV + original 36 SHA-verified evidence; center-only Chapter-3 canonical NumPy metrics"
    fullcov["source_ref"]="chapter4@de02fdda91e23db177de4429b3e1a9693c046d83"
    fullcov["recovery"]="76/76 center-only candidate; unreviewed until independent QA"
    fullcov["recovered_center_count"]=76
    fullcov["previous_36_compared"]=36
    fullcov["test_accessed"]=False
    fullcov["optimizer_steps_executed"]=0
    fullcov["inference_replay_count"]=0
    fullcov["metric_coverage_counts"]={}
    counts=Counter()
    for run in fullcov["runs"]:
        for key,entry in run["metrics"].items():
            status=entry["status"]
            counts[(key,status)]+=1
    vocabulary=("measured","not_measured","not_applicable","undefined_or_infinite","not_verified")
    for key in fullcov["metric_ids"]:
        coverage={status:counts[(key,status)] for status in vocabulary}
        if sum(coverage.values())!=76:raise ValueError("Coverage count mismatch "+key)
        fullcov["metric_coverage_counts"][key]=coverage
    if any(fullcov["metric_coverage_counts"]["center."+k]["measured"]!=76 for k in REQUIRED):
        raise ValueError("Not all 76 centers measured")

    with (OUT/"coverage_manifest_76.json").open("w",encoding="utf8") as fh:
        json.dump(fullcov,fh,ensure_ascii=False,indent=2,allow_nan=False)
        fh.write("\n")
    with (OUT/"coverage_summary_76.csv").open("w",encoding="utf8",newline="") as fh:
        wr=csv.writer(fh)
        wr.writerow(["metric_id",*vocabulary])
        for key in fullcov["metric_ids"]:
            wr.writerow([key,*[counts[(key,status)] for status in vocabulary]])
    fields=["run_id","data_seed","reference_n","reference_seed","family",
        "objective","training_tau","training_steps","train_BA","validation_BA",
        "query_start","source_archive","source_member","member_sha256_declared",
        "member_sha256_verified","source_archive_sha256_declared",
        "source_archive_sha256_verified","source_evidence","metric_contract_version"]
    with (OUT/"coverage_manifest_76.csv").open("w",encoding="utf8",newline="") as fh:
        wr=csv.writer(fh)
        wr.writerow([*fields,*[col for k in fullcov["metric_ids"]
                          for col in (k+".status",k+".reason")]])
        for row in fullcov["runs"]:
            wr.writerow([*(row.get(key) for key in fields),
                *(val for k in fullcov["metric_ids"] for val in
                 (row["metrics"][k]["status"],row["metrics"][k]["reason"]))])
    audit={"schema":"researcher03-archive-recovery-76-candidate-v1",
        "source_ref":"chapter4@de02fdda91e23db177de4429b3e1a9693c046d83",
        "checked_utc":datetime.now(timezone.utc).isoformat(),
        "previous_36_preserved":True,"previous_36_compared":36,
        "max_abs_differences_previous_36":checks,
        "previous_36_numeric_tolerance":1e-10,
        "archive_sha_member_verified":76,
        "center_metric_complete":76,
        "center_only_metrics":list(REQUIRED),
        "validation_numeric":sum(r["validation_BA"] is not None for r in fullcov["runs"]),
        "validation_not_applicable":sum(r["validation_BA"] is None for r in fullcov["runs"]),
        "test_accessed":False,"optimizer_steps":0,"inference_replays":0,
        "sample_level_R_NLL_BA_recomputed":False,
        "status":"READY_FOR_INDEPENDENT_QA_76_CANDIDATE"}
    if audit["validation_numeric"]!=50 or audit["validation_not_applicable"]!=26:
        raise ValueError("Train/Validation status regression")
    (OUT/"recovery_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"result":"PASS",**audit},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
