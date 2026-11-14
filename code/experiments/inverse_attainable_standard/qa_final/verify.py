#!/usr/bin/env python3
"""R05 §4.10 — final internal read-only publication evidence gate.

No graph, optimizer, Train/Test data generation or solver import. Every numerical
claim is recomputed only from canonical archived JSON and checksummed raw files.
The final independent scientific audit is a separate future authority.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import re

BASE=Path(__file__).resolve().parents[1]
REPO=Path(__file__).resolve().parents[4]
METHODS=("Q-only","Exact-Adam","Q+Bootstrap-GN")
BLOCKS=("B01","B02","B03","B04","B05","B06")
FROZEN_B01=BASE/"standard_plan/results/B01_v2/validated_run_01/summary.json"

def check(ok,code):
    if not ok:raise AssertionError(code)
def data(path):
    return json.loads(path.read_text(encoding="utf-8"))
def sha256(b):
    return hashlib.sha256(b).hexdigest()
def finite(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def check_entry(record, row):
    check(record["run_id"]==row["run_id"],"RUN_ID_MISMATCH")
    for k in ("block_id","class_index","initialization","solver","nref","data_seed"):
        check(record[k]==row[k],"RUN_FIELD_MISMATCH_"+k)
    check(record["status"]=="COMPLETED","INCOMPLETE_RUN")
    check(finite(record["best_exact_center_JSD"]) and
          record["best_exact_center_JSD"]>=0,"JSD_INVALID")
    check(record["success"]==(record["best_exact_center_JSD"]<=1e-8),
          "SUCCESS_THRESHOLD_MISMATCH")

def verify():
    plan=data(BASE/"standard_plan/plan_108.json")
    coverage=data(BASE/"results/coverage.json")
    metric=data(BASE/"results/metrics_full.json")
    check(plan["status"]=="PROPOSED_AWAITING_MAIN_RUN_GATE","HISTORICAL_PLAN_STATUS")
    check(plan["count"]["solver_runs"]==108,"PLAN_108")
    check(len(plan["runs"])==108 and len(plan["blocks"])==6,"PLAN_MATRIX")
    check(len(set(r["run_id"] for r in plan["runs"]))==108,"DUPLICATE_RUN_ID")
    check(len(coverage["records"])==108 and coverage["complete_all"] is True,
          "COVERAGE_NOT_COMPLETE")
    check(coverage["status_counts"]=={"COMPLETED":108},"COVERAGE_STATUS")
    check(coverage["B01_v2_independently_accepted"]==18 and coverage["new_run_count"]==90,
          "PROVENANCE_HISTORIC_VS_NEW")
    check(coverage["plan_sha256"]==sha256((BASE/"standard_plan/plan_108.json").read_bytes()),
          "PLAN_DIGEST_MISMATCH")
    check(metric["status"]=="COMPLETE_PROVISIONAL","METRIC_NOT_COMPLETE_PROVISIONAL")
    check(len(metric["group_metrics"])==18,"METRIC_GROUP_18")
    expected={r["run_id"]:r for r in plan["runs"]}
    observed={r["run_id"]:r for r in coverage["records"]}
    check(set(expected)==set(observed),"NOT_EXACTLY_PLAN_108")
    b01=data(FROZEN_B01)
    check(b01["status"]=="completed" and len(b01["runs"])==18,"HISTORIC_B01_NOT_ACCEPTED")
    audit_dir=BASE/"results/audit_attachments"
    counts=collections.Counter()
    statuses=collections.Counter()
    for runid,row in expected.items():
        rec=observed[runid]
        check_entry(rec,row)
        bid=row["block_id"].split("-")[1]
        counts[bid]+=1
        statuses[(bid,row["solver"])]+=bool(rec["success"])
        if bid=="B01":
            check(rec["evidence_status"]=="ACCEPT_B01_V2","B01_PROVENANCE")
            continue
        check(rec["evidence_status"]=="PROVISIONAL_PENDING_FINAL_QA",
              "NEW_RESULT_PREMATURE_QA_PASS")
        path=BASE/rec["raw_file"]
        check(path.is_file() and sha256(path.read_bytes())==rec["raw_summary_sha256"],
              "RAW_SUMMARY_MISSING_SHA_"+runid)
        result=data(path)
        check(result["status"]=="COMPLETED" and result["run_id"]==runid,"RAW_RESULT_STATUS")
        check(result["success"]==rec["success"],"RAW_SUMMARY_SUCCESS")
        check(abs(result["best_exact_center_JSD"]-rec["best_exact_center_JSD"])<1e-16,
              "RAW_SUMMARY_ERROR")
        check(result.get("test_used") is False and
              result.get("witness_given_to_solver") is False,"DATA_LEAKAGE")
        check(result["owner_comment_id"] in (6852780179,6853301767),"UNPINNED_OWNER_ROOT")
        check(result["source_pin_sha256"]==rec.get("source_pin_sha256"),"SOURCE_PIN_DRIFT")
        check(result["peak_rss_mib"]<1536,"RSS_HARD_LIMIT")
        attempt=result["attempt_id"]
        check(len(attempt)==32 and path.parent.name==attempt,"ATTEMPT_ID")
        trace=path.parent/"trace.json"
        curve=path.parent/"curve.csv"
        check(trace.is_file() and sha256(trace.read_bytes())==rec["raw_trace_sha256"],
              "TRACE_SHA_"+runid)
        check(curve.is_file() and sha256(curve.read_bytes())==rec["raw_curve_sha256"],
              "CURVE_SHA_"+runid)
        audit_ref=data(path.parent/"audit_ref.json")
        check(audit_ref["attempt_id"]==attempt,"AUDIT_REF_ID")
        for stage in ("start","terminal"):
            af=audit_dir/(attempt+"."+stage+".json")
            check(af.is_file() and
                  sha256(af.read_bytes())==audit_ref[stage+"_sha256"],
                  "AUDIT_REF_INVALID_"+stage)
        terminal=data(audit_dir/(attempt+".terminal.json"))
        check(terminal["status"]=="COMPLETED","AUDIT_TERMINAL_STATUS")
        check(data(path.parent/"input_pins.json")["run_id"]==runid,"INPUT_PIN_RUNID")
    check(all(counts[b]==18 for b in BLOCKS),"INCOMPLETE_BLOCK")
    check([sum(statuses[(b,m)] for b in BLOCKS) for m in METHODS]==[2,33,36],
          "SUCCESS_COUNTS_DRIFT")
    for g in metric["group_metrics"]:
        check(g["completed"]==g["planned"]==6 and g["failed"]==0
              and g["not_run"]==0 and g["not_verified"]==0,"INCOMPLETE_GROUP")
        check(g["success"]==statuses[(g["block_id"].split("-")[1],g["solver"])],
              "GROUP_VS_RUN_SUCCESS")
    replay=metric["reconstructed_center_density_and_geometry"]
    check(len(replay)==6 and {r["block"] for r in replay}==set(BLOCKS),
          "EXACT_REPLAY_ALL_BLOCKS")
    for r in replay:
        check(r["status"]=="PASS_INTERNAL_EXACT_REPLAY","REPLAY_NOT_PASS")
        check(r["replayed_policy_count"]==18 and len(r["combinations"])==6
              and r["never_read_Test_or_Validation"] is True,"REPLAY_SCOPE_OR_COVERAGE")
        combos={(x["method"],x["initialization"]) for x in r["combinations"]}
        check(len(combos)==6,"REPLAY_DUPLICATE_METHOD_INIT")
        for combo in r["combinations"]:
            check(len(combo["per_class"])==3,"REPLAY_MISSING_CLASS")
            for c in combo["per_class"]:
                check(finite(c["exact_center_JSD"]) and
                      abs(c["exact_center_JSD"]-c["raw_center_JSD"])<=1e-12,
                      "REPLAY_SELECTED_CENTER_ERROR_DRIFT")
                for key in ("R_KL","R_JSD","R_sqrt_JSD"):
                    check(finite(c[key]) and c[key]>=0,"INVALID_SAMPLE_DENSITY")
            for key in ("I","T","BT","TNL","BTNL","R_KL_uniform_class"):
                check(finite(combo[key]) and combo[key]>=0,"INVALID_CENTER_GEOMETRY")
    check(metric["missing_reconstruction_blocks"]==[],"MISSING_REPLAY")
    incomplete=(audit_dir/"0f545f2ca40c4171853fa09043f24219.terminal.json")
    check(incomplete.is_file(),"LOST_FAILURE_PROVENANCE")
    bad=data(incomplete)
    check(bad["status"]=="INVALID_PRECHECK" and bad["error"]=="RUN_ID_NOT_PLANNED"
          and bad["graph_constructed"]==0,"FIRST_B02_FAILURE_NOT_RETAINED")
    expected_media=[
      "best_jsd_by_n.pdf","density_vs_separation.pdf",
      "success_n150.pdf","success_n300.pdf","reconstructed_I.pdf"]
    for n in expected_media:
        file=BASE/"figures"/n
        check(file.exists() and file.stat().st_size>2000 and
              file.read_bytes().startswith(b"%PDF"),"FIGURE_PDF_INVALID_"+n)
    tex=REPO/"tex/chapter4.tex"
    body=tex.read_text(encoding="utf-8")
    section=r"\section{بازیابی معکوس سیاست از مراکز دست‌یافتنی}"
    trailing=r"\section{مراکز برچسبی بیزی، فضای قابل‌دستیابی و معیارهای بازنمایی}"
    check(body.count(section)==1 and body.count(trailing)==1,"CHAPTER_BOUNDARIES")
    sec=body[body.index(section):body.index(trailing)]
    checks=[
      "108","33/36","36/36","2/36",
      r"\label{sec:ch4-inverse-attainable}",
      r"\label{eq:ch4-inverse-success}",
      r"\input{code/experiments/inverse_attainable_standard/tables/sec410_blocks.tex}",
      r"\input{code/experiments/inverse_attainable_standard/tables/sec410_success.tex}",
      r"\input{code/experiments/inverse_attainable_standard/tables/sec410_density.tex}",
      r"\input{code/experiments/inverse_attainable_standard/figures/sec410_error.tex}",
      r"\input{code/experiments/inverse_attainable_standard/figures/sec410_density_vs_I.tex}",
    ]
    check(all(x in sec for x in checks),"TEX_MISSING_EVIDENCE_OR_REFERENCES")
    wrappers=[BASE/"tables/sec410_blocks.tex",BASE/"tables/sec410_success.tex",
              BASE/"tables/sec410_density.tex",BASE/"figures/sec410_error.tex",
              BASE/"figures/sec410_density_vs_I.tex"]
    for file in wrappers: check(file.is_file(),"MISSING_LATEX_INPUT")
    labels=[]
    for t in [body]+[p.read_text(encoding="utf-8") for p in wrappers]:
        labels.extend(re.findall(r"\\label\{([^}]+)\}",t))
    check(len(labels)==len(set(labels)),"DUPLICATE_LABEL_IN_CHAPTER_OR_SEC410_INPUT")
    for x in (r"tab:ch4-inverse-blocks",r"tab:ch4-inverse-full-success",
              r"tab:ch4-inverse-R-I",r"fig:ch4-inverse-best-jsd",
              r"fig:ch4-inverse-density-vs-I"):
        check(x in labels,"UNDEFINED_SECTION410_LABEL")
    for method,n150,n300 in ((METHODS[0],0,2),(METHODS[1],17,16),(METHODS[2],18,18)):
        s=(BASE/"tables/sec410_success.tex").read_text(encoding="utf-8")
        a=r"\lr{"+method+"} & $"+str(n150)+r"/18$ & $"+str(n300)+r"/18$"
        check(a in s,"LATEX_SUCCESS_TABLE_DRIFT_"+method)
    return {
        "schema":"R05_SEC410_FINAL_INTERNAL_CHECK_V1",
        "verdict":"INTERNAL_DATA_AND_TEX_STRUCTURE_PASS",
        "planned":108,"completed":sum(counts.values()),
        "independent_historic_B01_v2":18,
        "new_provisional_runs":90,"exact_replayed_selected_policies":108,
        "independent_generator_seeds":3,
        "method_successes":{k:sum(statuses[(b,k)] for b in BLOCKS) for k in METHODS},
        "full_metrics_sha256":sha256((BASE/"results/metrics_full.json").read_bytes()),
        "coverage_sha256":sha256((BASE/"results/coverage.json").read_bytes()),
        "chapter4_tex_sha256":sha256(tex.read_bytes()),
        "figures_pdf_checked":expected_media,
        "historical_bad_B02_invocation_retained":True,
        "external_final_scientific_QA":"PENDING",
        "latex_compilation_visual_PDF_QA":"NOT_RUN",
        "other_chapter_sections_changed_by_this_phase":False,
    }

if __name__=="__main__":
    result=verify()
    target=BASE/"qa_final/internal_check.json"
    serial=(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    if target.exists():
        check(target.read_bytes()==serial,"QA_MANIFEST_HAS_CHANGED")
    else:
        target.write_bytes(serial)
    print("R05_SEC410_INTERNAL_FINAL_QA_PASS",json.dumps(result,ensure_ascii=False))
