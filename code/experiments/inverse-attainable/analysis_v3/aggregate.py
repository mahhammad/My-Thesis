#!/usr/bin/env python3
"""R05 §4.10 reproducible raw->coverage->metrics->tables->figures pipeline.

Uses only existing immutable solver evidence; DOES NOT execute graph or optimizer.
No Test/Validation. B01-v2 independently accepted; other runs are provisional.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,itertools,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"standard_plan/plan_108.json"
B01=ROOT/"standard_plan/results/B01_v2/validated_run_01"
RAW=ROOT/"results/raw"
TOL=1e-8
METHODS=("Q-only","Exact-Adam","Q+Bootstrap-GN")
COLORS={"Q-only":"#475569","Exact-Adam":"#0f766e","Q+Bootstrap-GN":"#b45309"}
AUDIT_ROOT=ROOT/"results/audit_attachments"

def sha(x):return hashlib.sha256(x).hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def emit(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,
                               allow_nan=False)+"\n",encoding="utf-8")
def table(path,rows,cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader()
        for row in rows:w.writerow({k:row.get(k) for k in cols})
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def median(a):return statistics.median(a) if a else None
def quantiles(a):
    if not a:return {"median":None,"q25":None,"q75":None,"max":None}
    a=sorted(a)
    def q(x):
        i=(len(a)-1)*x;k=int(i);return a[k]+(a[min(len(a)-1,k+1)]-a[k])*(i-k)
    return {"median":q(.5),"q25":q(.25),"q75":q(.75),"max":a[-1]}
def kl(p,q):
    value=0.0
    for a,b in zip(p,q):
        if a<=0:continue
        if b<=0:return None
        value+=a*math.log(a/b)
    return max(0.,value)
def jsd(p,q):
    mix=[(a+b)/2 for a,b in zip(p,q)]
    return (kl(p,mix)+kl(q,mix))/2
def check_distribution(v):
    return len(v)==3 and all(finite(x) and 0<=x<=1 for x in v) and abs(sum(v)-1)<1e-9
def geometry(centers,weights=None):
    assert len(centers)==3 and all(check_distribution(v) for v in centers)
    omega=weights or [1/3]*3
    avg=[sum(omega[c]*centers[c][k] for c in range(3)) for k in range(3)]
    I=sum(omega[c]*kl(centers[c],avg) for c in range(3))
    T=sum(omega[c]*centers[c][c] for c in range(3))
    TNL=-sum(omega[c]*math.log(centers[c][c]) for c in range(3)) if all(centers[c][c]>0 for c in range(3)) else None
    perms=list(itertools.permutations(range(3)))
    BT=max(sum(omega[c]*centers[c][p[c]] for c in range(3)) for p in perms)
    score=[]
    for p in perms:
        vs=[centers[c][p[c]] for c in range(3)]
        score.append(-sum(omega[c]*math.log(vs[c]) for c in range(3)) if min(vs)>0 else math.inf)
    BTNL=min(score)
    if not math.isfinite(BTNL):BTNL=None
    pairs=[{"class_a":a,"class_b":b,"JSD":jsd(centers[a],centers[b]),
            "sqrt_JSD":math.sqrt(jsd(centers[a],centers[b])),
            "KL_a_to_b":kl(centers[a],centers[b]),
            "KL_b_to_a":kl(centers[b],centers[a])}
           for a,b in itertools.combinations(range(3),2)]
    return {"I":I,"T":T,"BT":BT,"TNL":TNL,"BTNL":BTNL,"pairwise":pairs,
            "class_weights":omega,"global_center":avg}
def sources_and_targets(plan):
    m={b["block_id"]:b for b in plan["blocks"]}
    out={}
    for bid,b in m.items():
        if "-B01-" in bid:
            bf=read(B01/"manifest.json")
            centers=[r["target"] for r in bf["targets"]]
            hashes={"manifest_sha256":sha((B01/"manifest.json").read_bytes())}
        else:
            path=(ROOT/"b02_preflight/manifest.json" if "-B02-" in bid
                  else ROOT/"snapshot_v2/results"/bid/"manifest.json")
            bf=read(path)
            centers=[r.get("vector",r.get("values")) for r in bf["targets"]["per_class"]]
            hashes={"manifest_sha256":sha(path.read_bytes())}
        assert all(check_distribution(v) for v in centers),(bid,centers)
        out[bid]={"config":b,"centers":centers,**hashes}
    return out
def sample_raw(row,accepted,allow_missing=True):
    run=row["run_id"]
    if "-B01-" in row["block_id"]:
        c=row["class_index"];init=row["initialization"];method=row["solver"]
        candidates=[z for z in accepted if z["class"]==c and z["start"]==init and z["solver"]==method]
        assert len(candidates)==1,(run,len(candidates))
        b=candidates[0]
        best=b["best_exact_center_JSD"];final=b["final_exact_center_JSD"]
        assert finite(best) and finite(final)
        assert abs(best-b["selected_exact_center_JSD"])<1e-11
        assert (best<=TOL)==b["reached_threshold"]
        return {"run_id":run,"status":"COMPLETED","evidence_status":"ACCEPT_B01_V2",
            "best_exact_center_JSD":best,"final_exact_center_JSD":final,
            "success":bool(best<=TOL),"stop_reason":b["stop_reason"],
            "exclusive_solver_wall_s":None,
            "inclusive_solver_wall_s":b["inclusive_solver_wall_seconds"],
            "monitor_exact_forward_wall_s":b["explicit_monitor_exact_forward_wall_s"],
            "counts":b["counts_including_warmup_and_monitoring"],
            "peak_rss_mib":None,"restart_used":b["restart_used"],
            "selected_checkpoint_step":b["selected_checkpoint_step"],
            "raw_file":str(B01/"summary.json").replace(str(ROOT)+"/",""),
            "note":"B01-v2 accepted historical run; wall inclusive, peak RSS is job-level"}
    directory=RAW/row["block_id"]/run
    summaries=sorted(directory.glob("*/summary.json")) if directory.exists() else []
    failures=sorted(directory.glob("*/failure.json")) if directory.exists() else []
    if len(summaries)>1:
        return {"run_id":run,"status":"NOT_VERIFIED","reason":"MULTIPLE_COMPLETED_ATTEMPTS","attempts":len(summaries)}
    if summaries:
        p=summaries[0];d=read(p)
        try:
            assert d["run_id"]==run and d["block_id"]==row["block_id"]
            assert (d["class"],d["initialization"],d["solver"])==(row["class_index"],row["initialization"],row["solver"])
            assert d["status"]=="COMPLETED" and d["evidence_status"]=="PROVISIONAL_PENDING_FINAL_QA"
            b=d["best_exact_center_JSD"];f=d["final_exact_center_JSD"];sel=d["selected_exact_center_JSD"]
            assert all(finite(x) and x>=0 for x in (b,f,sel))
            assert abs(b-sel)<1e-11 and (b<=TOL)==d["success"]
            assert d["tol"]==TOL and d["walk"]=={"beta":.4,"lambda":.8,"H":12,"dtype":"float64"}
            assert d["witness_given_to_solver"] is False and d["test_used"] is False
            attempt=d["attempt_id"]
            assert len(attempt)==32 and p.parent.name==attempt
            expected_root=6852780179 if row["block_id"].split("-")[1] in ("B03","B05") else 6853301767
            assert d["owner_comment_id"]==expected_root,("UNEXPECTED_OWNER_ROOT",run)
            audit_ref=p.parent/"audit_ref.json";pins=p.parent/"input_pins.json"
            trace=p.parent/"trace.json";curve=p.parent/"curve.csv"
            assert all(q.is_file() for q in (audit_ref,pins,trace,curve))
            ar=read(audit_ref);ip=read(pins)
            assert ar["attempt_id"]==attempt and ip["run_id"]==run
            assert ip["input_sha256"]==d["input_sha256"]
            assert ip["owner_root_sha256"]==d["owner_root_sha256"]
            assert ip["source_sha256"] and ip["source_git_blob_sha1"]
            assert ip["owner_root_sha256"]==d["owner_root_sha256"]
            for suffix,field in (("start","start_sha256"),("terminal","terminal_sha256")):
                file=AUDIT_ROOT/(attempt+"."+suffix+".json")
                assert file.is_file()
                assert sha(file.read_bytes())==ar[field],(run,suffix)
            evidence=sha(p.read_bytes());thash=sha(trace.read_bytes());curvehash=sha(curve.read_bytes())
            audit_terminal=read(AUDIT_ROOT/(attempt+".terminal.json"))
            assert audit_terminal["status"]=="COMPLETED",run
            return {"run_id":run,"status":"COMPLETED",
                 "evidence_status":"PROVISIONAL_PENDING_FINAL_QA",
                 "best_exact_center_JSD":b,"final_exact_center_JSD":f,
                 "success":d["success"],"stop_reason":d["stop_reason"],
                 "exclusive_solver_wall_s":d.get("exclusive_solver_wall_s"),
                 "inclusive_solver_wall_s":d.get("inclusive_solver_wall_s"),
                 "monitor_exact_forward_wall_s":d.get("monitor_exact_forward_wall_s"),
                 "counts":d["counts"],"peak_rss_mib":d.get("peak_rss_mib"),
                 "restart_used":d.get("restart_used"),"selected_checkpoint_step":d.get("selected_checkpoint_step"),
                 "raw_file":str(p.relative_to(ROOT)),
                 "raw_summary_sha256":evidence,"raw_trace_sha256":thash,"raw_curve_sha256":curvehash,
                 "attempt_id":attempt,"source_pin_sha256":d.get("source_pin_sha256")}
        except (AssertionError,KeyError,ValueError,TypeError) as exc:
            return {"run_id":run,"status":"NOT_VERIFIED","reason":"DATA_AUDIT_OR_SCHEMA_FAILED","detail":str(exc)[:250],"path":str(p)}
    if failures:
        return {"run_id":run,"status":"FAILED","reason":"ATTEMPT_FAILURE",
                "failure_files":[str(x.relative_to(ROOT)) for x in failures]}
    return {"run_id":run,"status":"NOT_RUN","reason":"NO_RAW_RESULT"}
def aggregate(args):
    plan=read(PLAN)
    assert len(plan["runs"])==108 and len(set(x["run_id"] for x in plan["runs"]))==108
    assert len(plan["blocks"])==6
    bs=read(B01/"summary.json")
    assert len(bs["runs"])==18 and bs["status"]=="completed"
    centers=sources_and_targets(plan)
    records=[]
    for row in plan["runs"]:
        result=sample_raw(row,bs["runs"])
        result.update({key:row[key] for key in
                       ("block_id","class_index","initialization","initialization_seed",
                        "solver","nref","data_seed","teacher_seed")})
        records.append(result)
    by_id={r["run_id"]:r for r in records}
    assert len(by_id)==108
    status=collections.Counter(r["status"] for r in records)
    full=status["COMPLETED"]==108 and len(status)==1
    if args.complete and not full:raise RuntimeError("COVERAGE_NOT_COMPLETE "+str(status))
    out=ROOT/("analysis_v3/preview" if not args.complete else "results")
    tab=out/"tables" if not args.complete else ROOT/"tables"
    fig=out/"figures" if not args.complete else ROOT/"figures"
    report_path=out/"analysis_generated_report.txt"
    groups=[]
    for block in plan["blocks"]:
        bid=block["block_id"]
        for method in METHODS:
            rows=[r for r in records if r["block_id"]==bid and r["solver"]==method]
            assert len(rows)==6
            complete=[r for r in rows if r["status"]=="COMPLETED"]
            measured=[r for r in complete if finite(r.get("best_exact_center_JSD"))]
            err=[r["best_exact_center_JSD"] for r in measured]
            final=[r["final_exact_center_JSD"] for r in measured]
            agg={"block_id":bid,"nref":block["nref"],"data_seed":block["data_seed"],
                 "solver":method,"planned":6,"completed":len(complete),
                 "success":sum(1 for x in complete if x.get("success") is True),
                 "failed":sum(r["status"]=="FAILED" for r in rows),
                 "not_run":sum(r["status"]=="NOT_RUN" for r in rows),
                 "not_verified":sum(r["status"]=="NOT_VERIFIED" for r in rows),
                 "success_rate_denominator_six":sum(1 for x in complete if x.get("success"))/6,
                 "best_JSD_median":median(err),"best_JSD_max":max(err) if err else None,
                 "final_JSD_median":median(final),**{f"best_JSD_{k}":v for k,v in quantiles(err).items()},
                 "resource_rss_max_mib":max([x["peak_rss_mib"] for x in complete if finite(x.get("peak_rss_mib"))],default=None),
                 "exclusive_wall_sum_s":sum(x["exclusive_solver_wall_s"] for x in complete if finite(x.get("exclusive_solver_wall_s"))),
                 "inclusive_wall_sum_s":sum(x["inclusive_solver_wall_s"] for x in complete if finite(x.get("inclusive_solver_wall_s")))}
            groups.append(agg)
    geom=[]
    for block in plan["blocks"]:
        bid=block["block_id"]
        data=centers[bid]
        g=geometry(data["centers"])
        geom.append({"block_id":bid,"nref":block["nref"],"data_seed":block["data_seed"],
                     **{k:v for k,v in g.items() if k not in ("pairwise","class_weights","global_center")},
                     "pairwise":g["pairwise"],
                     "center_rows":data["centers"],"weighting":"uniform_class_1_over_3",
                     "status":"MEASURED_TARGET_CENTERS_ONLY","manifest_sha256":data["manifest_sha256"]})
    coverage={"schema":"R05_SEC410_108_COVERAGE_V3","evidence_status":"PROVISIONAL_PENDING_FINAL_QA",
              "plan_sha256":sha(PLAN.read_bytes()),"historical_B01_v2_summary_sha256":sha((B01/"summary.json").read_bytes()),
              "planned_count":108,"status_counts":dict(status),
              "B01_v2_independently_accepted":18,
              "new_run_count":sum(r["status"]=="COMPLETED" and r["nref"] in (150,300)
                                and "-B01-" not in r["block_id"] for r in records),
              "seed_generators":3,"data_seeds":[20261102,20261103,20261104],
              "nref_levels":[150,300],"complete_all":full,
              "records":records}
    # Independent exact-forward replay of selected class policies. This checks
    # sample-level densities and geometry; never infer R from center JSD alone.
    reconstructed=[]
    missing_recon=[]
    for block in plan["blocks"]:
        bid=block["block_id"]
        short=bid.split("-")[1]
        path=ROOT/"results/reconstructed"/short/"metrics.json"
        if not path.is_file():
            missing_recon.append(short)
            continue
        evidence=read(path)
        assert (evidence["status"]=="PASS_INTERNAL_EXACT_REPLAY" and
                evidence["block"]==short and evidence["replayed_policy_count"]==18 and
                len(evidence["combinations"])==6 and evidence["never_read_Test_or_Validation"] is True),short
        for value in evidence["combinations"]:
            assert len(value["matrix_of_reconstructed_class_centers"])==3
            assert len(value["per_class"])==3
            assert max(abs(c["exact_center_JSD"]-c["raw_center_JSD"])
                       for c in value["per_class"])<=1e-12
        reconstructed.append(evidence)
    if args.complete and len(reconstructed)!=6:
        raise RuntimeError("MISSING_INDEPENDENT_RECONSTRUCTED_METRICS "+str(missing_recon))
    limits={
      "classification":{"BA":"NOT_APPLICABLE_TO_CORE_INVERSE","confusion":"NOT_APPLICABLE_TO_CORE_INVERSE",
                        "decision_margin":"NOT_APPLICABLE_TO_CORE_INVERSE"},
      "density":{"R_KL":"MEASURED_REPLAYED_TRAIN_ONLY" if len(reconstructed)==6 else "PARTIALLY_MEASURED_REPLAYED_TRAIN_ONLY",
                 "R_JSD":"MEASURED_REPLAYED_TRAIN_ONLY" if len(reconstructed)==6 else "PARTIALLY_MEASURED_REPLAYED_TRAIN_ONLY",
                 "R_sqrt_JSD":"MEASURED_REPLAYED_TRAIN_ONLY" if len(reconstructed)==6 else "PARTIALLY_MEASURED_REPLAYED_TRAIN_ONLY"},
      "centers_geometry":{"I_T_BT_TNL_BTNL":"MEASURED_TARGET_CENTERS",
                          "selected_policy_I_T_BT_TNL_BTNL":"MEASURED_REPLAYED_TRAIN_ONLY" if len(reconstructed)==6 else "PARTIALLY_MEASURED_REPLAYED_TRAIN_ONLY"},
      "inverse":{"best_final_exact_JSD":"MEASURED_PER_COMPLETED_RUN",
                 "wall_resource_and_counts":"MEASURED_WHEN_RAW_AVAILABLE"},
      "surrogate_losses":{"Q_loss_per_checkpoint":"PRESENT_IN_METHOD_RAW_TRACES_NOT_AGGREGATED",
                          "sigmoid_margin":"NOT_APPLICABLE_TO_CORE_INVERSE"}}
    metrics={"schema":"R05_SEC410_METRICS_V3",
             "status":"COMPLETE_PROVISIONAL" if full else "PARTIAL_PROVISIONAL",
             "protocol":{"JSD_tol":TOL,"beta":.4,"lambda":.8,"H":12,
                         "dtype":"float64","method_order":list(METHODS),
                         "independent_data_generators":3,
                         "no_Validation_or_Test":True,
                         "B01_v2_replayed":any(r["block"]=="B01" for r in reconstructed)},
             "group_metrics":groups,"target_center_geometry":geom,
             "reconstructed_center_density_and_geometry":reconstructed,
             "missing_reconstruction_blocks":missing_recon,
             "metric_coverage":limits,
             "aggregation_note":"108 runs are within only 3 independent data-generator seeds; class and initialization rows are dependent.",
             "wall_note":"inclusive method walls overlap in Q->GN; never sum across methods as physical job wall."}
    emit(out/"coverage.json",coverage);emit(out/"metrics_full.json",metrics)
    run_columns=["block_id","run_id","nref","data_seed","class_index","initialization","solver",
                 "status","success","best_exact_center_JSD","final_exact_center_JSD",
                 "stop_reason","exclusive_solver_wall_s","inclusive_solver_wall_s","peak_rss_mib",
                 "evidence_status","raw_file","raw_summary_sha256"]
    table(tab/"run_metrics.csv",records,run_columns)
    gcols=[k for k in groups[0] if k not in ("best_JSD_median",)]+["best_JSD_median"]
    table(tab/"success_by_block_method.csv",groups,gcols)
    table(tab/"protocol_blocks.csv",plan["blocks"],
          ["block_id","nref","data_seed","reference_seed","teacher_seed",
           "class_count","planned_solver_runs"])
    geoRows=[]
    for z in geom:
        base={k:v for k,v in z.items() if k not in ("pairwise","center_rows")}
        geoRows.append(base)
    table(tab/"target_geometry.csv",geoRows,
          ["block_id","nref","data_seed","I","T","BT","TNL","BTNL","status","weighting"])
    reconRows=[]
    for block in reconstructed:
        for v in block["combinations"]:
            reconRows.append({"block":block["block"],"nref":block["nref"],
              "data_seed":block["data_seed"],"solver":v["method"],
              "initialization":v["initialization"],
              "I":v["I"],"T":v["T"],"BT":v["BT"],"TNL":v["TNL"],"BTNL":v["BTNL"],
              "R_KL":v["R_KL_uniform_class"],"R_JSD":v["R_JSD_uniform_class"],
              "R_sqrt_JSD":v["R_sqrt_JSD_uniform_class"],
              "I_target":v["target_geometry"]["I"],
              "evidence_status":v["evidence_status"]})
    table(tab/"reconstructed_geometry_density.csv",reconRows,
          ["block","nref","data_seed","solver","initialization","I","T","BT",
           "TNL","BTNL","R_KL","R_JSD","R_sqrt_JSD","I_target","evidence_status"])
    pairs=[dict(block_id=z["block_id"],**p) for z in geom for p in z["pairwise"]]
    table(tab/"target_pairwise_distances.csv",pairs,
          ["block_id","class_a","class_b","KL_a_to_b","KL_b_to_a","JSD","sqrt_JSD"])
    byseed=[]
    for seed in sorted({r["data_seed"] for r in records}):
        for n in (150,300):
            for method in METHODS:
                rr=[r for r in records if r["data_seed"]==seed and r["nref"]==n and r["solver"]==method]
                assert len(rr)==6
                byseed.append({"data_seed":seed,"nref":n,"method":method,
                    "successful":sum(r.get("success") is True for r in rr),
                    "completed":sum(r["status"]=="COMPLETED" for r in rr),
                    "planned":6})
    table(tab/"seed_n_method.csv",byseed,["data_seed","nref","method","successful","completed","planned"])
    report=[
      "# گزارش داده‌محور بازسازی معکوس مراکز دست‌یافتنی — بخش ۴.۱۰",
      "",
      "**وضعیت:** "+("همهٔ ۱۰۸ ردیف شواهد ثبت شده؛ در انتظار ممیزی مستقل نهایی" if full else "گزارش موقت، اجرای ۱۰۸تایی هنوز کامل نیست"),
      "",
      "مبدأ معیارها فقط `plan_108.json`، ۱۸ نتیجهٔ پذیرفته‌شدهٔ `B01-v2` و فایل‌های raw هر اجرای جدید است.",
      "هدف‌های هر بلوک از سیاست شاهدِ نهان و Train ساخته شده‌اند؛ موفقیت به معنای بازسازی مرکز با JSD دقیق کمتر یا مساوی `1e-8` است.",
      "برابری مرکز به معنای شناسایی یکتای سیاست نیست. مجموعه شامل تنها سه بذر مستقل داده است.",
      "",
      "## پوشش واقعی",
      "",
      "تکمیل‌شده: **"+str(status['COMPLETED'])+"/108**؛ شکست ثبت‌شده: **"+str(status['FAILED'])+
      "**؛ اجرا نشده: **"+str(status['NOT_RUN'])+"**؛ نامعتبر/نیازمند بازبینی: **"+str(status['NOT_VERIFIED'])+"**.",
      "",
      "| بلوک | n | روش | موفق/۶ | تکمیل/۶ | میانگین/میانه JSD برگزیده |",
      "|---|---:|---|---:|---:|---:|",
    ]
    for v in groups:
        report.append("| "+v["block_id"].split("-")[1]+" | "+str(v["nref"])+" | "+v["solver"]+
          " | "+str(v["success"])+"/6 | "+str(v["completed"])+"/6 | "+
          ("%.3g"%v["best_JSD_median"] if v["best_JSD_median"] is not None else "—")+" |")
    report.extend([
      "",
      "## حدود اعتبار",
      "",
      "آمار توصیفی را باید در سطح بذر مولد تحلیل کرد؛ ۱۰۸ نتیجه تکرار مستقل نیستند.",
      "آمار زمان روش Q+GN شامل آغاز گرمِ Q است؛ برای جلوگیری از دوباره‌شماری، مجموع زمان‌های روش‌ها با زمان واقعی کل کار برابر فرض نمی‌شود.",
      "معیارهای `I,T,BT,TNL,BTNL` هم برای مراکز هدف و هم برای مراکز بازپخش‌شده با وزن برابر طبقات محاسبه شده‌اند.",
      "معیار `R` با بازپخش توزیع همسایگی **هر نمونهٔ Train** پیرامون مرکز مربوط به خودش به دست آمده است؛ از JSD مراکز استنتاج نشده است.",
      "BA، confusion و margin بدون پروتکل مجاز طبقه‌بندی محاسبه نشده‌اند.",
      "",
      "دادهٔ اصلی: `results/coverage.json` و `results/metrics_full.json`؛ جدول‌ها در `tables/` هستند.",
      "تا صدور رأی ممیزی مستقل نهایی، تمام خروجی‌های جدید `PROVISIONAL_PENDING_FINAL_QA`اند.",
      "",
    ])
    report_path.write_text("\n".join(report),encoding="utf-8")
    # Pure JSON summary is the authoritative source; plots are presentation only.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig.mkdir(parents=True,exist_ok=True)
    for n in (150,300):
        figg,ax=plt.subplots(figsize=(9,4.8))
        panel=[z for z in groups if z["nref"]==n]
        bnames=sorted(set(z["block_id"].split("-")[1] for z in panel))
        for i,method in enumerate(METHODS):
            vals=[next(z for z in panel if z["block_id"].split("-")[1]==bn and z["solver"]==method)["success"]
                  for bn in bnames]
            xpos=[j+.23*(i-1) for j in range(len(bnames))]
            ax.bar(xpos,vals,width=.22,color=COLORS[method],label=method)
        ax.set_xticks(range(len(bnames)),bnames)
        ax.set_ylim(0,6.5)
        ax.set_ylabel("Successful reconstructions / 6")
        ax.set_title(f"Attainable-center reconstruction, reference n={n}")
        ax.axhline(6,ls=":",color="#777777",lw=.7)
        ax.legend(frameon=False,ncol=3,loc="upper center")
        figg.tight_layout()
        figg.savefig(fig/f"success_n{n}.pdf",bbox_inches="tight")
        figg.savefig(fig/f"success_n{n}.png",dpi=170,bbox_inches="tight")
        plt.close(figg)
    figg,ax=plt.subplots(figsize=(9,5))
    for method in METHODS:
        a=[r for r in records if r["solver"]==method and r["status"]=="COMPLETED"]
        if not a:continue
        xx=[r["nref"]+(-15 if method=="Q-only" else 0 if method=="Exact-Adam" else 15) for r in a]
        yy=[math.log10(max(r["best_exact_center_JSD"],1e-16)) for r in a]
        ax.scatter(xx,yy,color=COLORS[method],alpha=.65,label=method,s=24)
    ax.axhline(-8,color="#555555",ls="--",lw=1,label="JSD tolerance 1e-8")
    ax.set_xticks([150,300],["150","300"])
    ax.set_xlabel("Reference sample count")
    ax.set_ylabel("log10(best exact-center JSD)")
    ax.set_title("Best exact-center reconstruction error (all recorded runs)")
    ax.legend(frameon=False)
    figg.tight_layout()
    figg.savefig(fig/"best_jsd_by_n.pdf",bbox_inches="tight")
    figg.savefig(fig/"best_jsd_by_n.png",dpi=170,bbox_inches="tight")
    plt.close(figg)
    if reconstructed:
        figg,ax=plt.subplots(figsize=(8.8,4.9))
        ordmap={"B01":0,"B02":1,"B03":2,"B04":3,"B05":4,"B06":5}
        for method in METHODS:
            records_plot=[v for r in reconstructed for v in r["combinations"] if v["method"]==method]
            xx=[ordmap[z["block"]]+(-.20 if z["initialization"]=="zero" else .20)
                for z in records_plot]
            ax.scatter(xx,[z["I"] for z in records_plot],color=COLORS[method],
                       label=method,s=29,alpha=.7)
        ax.set_xticks(range(6),list(ordmap))
        ax.set_xlabel("Block (fixed seed and n)")
        ax.set_ylabel("I of reconstructed class centers (nats)")
        ax.set_title("Information separation of reconstructed centers")
        ax.legend(frameon=False,ncol=3)
        figg.tight_layout()
        figg.savefig(fig/"reconstructed_I.pdf",bbox_inches="tight")
        figg.savefig(fig/"reconstructed_I.png",dpi=170,bbox_inches="tight")
        plt.close(figg)
        figg,ax=plt.subplots(figsize=(8.7,5.1))
        for method in METHODS:
            subset=[v for r in reconstructed for v in r["combinations"] if v["method"]==method]
            ax.scatter([v["R_KL_uniform_class"] for v in subset],[v["I"] for v in subset],
                       color=COLORS[method],label=method,s=33,alpha=.74)
        ax.set_xlabel("Within-class density R (KL, nats)")
        ax.set_ylabel("Between-class information I (nats)")
        ax.set_title("Density-separation geometry; same center does not imply same R")
        ax.legend(frameon=False)
        figg.tight_layout()
        figg.savefig(fig/"density_vs_separation.pdf",bbox_inches="tight")
        figg.savefig(fig/"density_vs_separation.png",dpi=170,bbox_inches="tight")
        plt.close(figg)
    emit(out/"analysis_manifest.json",{"source_plan_sha256":sha(PLAN.read_bytes()),
         "source_b01_summary_sha256":sha((B01/"summary.json").read_bytes()),
         "total_planned":108,"coverage":dict(status),"complete":full,
         "files":["coverage.json","metrics_full.json","tables/run_metrics.csv",
                  "tables/success_by_block_method.csv","tables/protocol_blocks.csv",
                  "tables/target_geometry.csv","tables/target_pairwise_distances.csv",
                  "tables/reconstructed_geometry_density.csv","tables/seed_n_method.csv","figures/success_n150.pdf",
                  "figures/success_n300.pdf","figures/best_jsd_by_n.pdf",
                  "analysis_generated_report.txt"],
         "no_solver_executed":True,"evidence_status":"PROVISIONAL_PENDING_FINAL_QA"})
    print("R05_AGGREGATE",json.dumps({"complete":full,"statuses":dict(status),
          "output":str(out)},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--complete",action="store_true",
                   help="Fail closed unless exactly 108 verified complete runs are present")
    aggregate(p.parse_args())
