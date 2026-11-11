#!/usr/bin/env python3
"""RESEARCHER-01 B01-v2: independent 18-case paired stop-rule revision.

Uses only existing StreamGraph and P2-validated D09 update instrumentation.
Resource guards: <=840s numerical, 1430 MiB preemptive RSS (1536 MiB hard
envelope), one CI numeric job; persist every checkpoint and failed attempt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]  # .../repo/code/experiments/inverse_attainable_standard/standard_plan/b01_run.py
sys.path.insert(0,str(HERE.parent))  # P2 instrumentation, no engine changes
import p2_v2_instrumentation as p2
from generate import generate,load_config
from stream_graph import StreamGraph
from run import DT,jsd

RESULT=HERE/"results"/"B01_v2"
LEGACY=HERE/"results"/"B01"
M=RESULT/"manifest.json"
RAW=RESULT/"trace.json"
CSV=RESULT/"curve.csv"
SUMMARY=RESULT/"summary.json"
PLAN=HERE/"plan_108.json"
BID="INV01-B01-n150-data20261102"
SECONDS=840.
PEAK_SOFT_MIB=1430.


def configure_p2():
 p2.MANIFEST=M
 p2.RAW=RAW
 p2.CSV=CSV
 p2.SUMMARY=SUMMARY
 p2.state={"started":None,"finished":None,"status":"not_started",
           "stop_reason":None,"runs":[],"errors":[],
           "wall_numerical_seconds":None,"peak_rss_mib":0}
 p2.stop_signal_reason=None
 p2.MEM_SOFT_LIMIT_MIB=PEAK_SOFT_MIB


def sha_arr(arr):
 return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def target_per_class(g,y,teacher_seed):
 rng=np.random.default_rng(teacher_seed)
 logit=rng.normal(0,1.7,size=(g.k,g.d+1,g.d))
 logit-=logit.mean(axis=-1,keepdims=True)
 witness_sha=sha_arr(logit)
 centers=[]
 for c in range(g.k):
  idx=np.flatnonzero(y==c)
  with torch.no_grad():
   ctr=g.center_train(torch.as_tensor(logit[c],dtype=DT),idx).detach()
  if abs(ctr.sum().item()-1)>1e-10 or float(jsd(ctr,ctr))>1e-12:
   raise AssertionError("invalid witness at class "+str(c))
  centers.append(ctr)
 del logit
 return centers,witness_sha


def solve_context(g,y,target,cl,init,init_seed,caps,summary):
 p2.stopcheck()
 idx=np.flatnonzero(y==cl)
 key=f"B01-v2-n150-d20261102-c{cl}-{init}"
 if init=="zero":
  start=None
 else:
  rng=np.random.default_rng(init_seed)
  start=torch.as_tensor(rng.normal(0,.1,size=(g.d+1,g.d)),dtype=DT)
 if start is not None and start.shape!=(g.d+1,g.d):
  raise AssertionError("invalid initialization")
 zq,rq=p2.adam_path(key,"Q-only",g,idx,target,caps["Q_only"],
                     initial=start,init_label=init)
 za,ra=p2.adam_path(key,"Exact-Adam",g,idx,target,caps["Exact_Adam"],
                     initial=start,init_label=init)
 zgn,rgn=p2.gn_instrumented(key,g,idx,target,zq,caps["GN"],
                            rq["elapsed_solver_wall_s"],"warm",rq["counts"])
 rgn["parent_initialization"]=init
 retry=None
 if rgn["best_JSD"]>p2.TOL:
  rz,retry=p2.gn_instrumented(key,g,idx,target,torch.zeros_like(zq),
     caps["GN_neutral_restart"],
     rq["elapsed_solver_wall_s"]+rgn["elapsed_solver_wall_s"],
     "neutral_restart",p2.addcost(rq["counts"],rgn["counts"]))
  retry["parent_initialization"]=init
  retry["counts"]["retries"]=1
  retry["inclusive_cost"]=p2.addcost(retry["inclusive_cost"],{"retries":1})
  p2.persist()
 gnbest=min(rgn["best_JSD"],retry["best_JSD"] if retry else float("inf"))
 selected_gn=retry if retry is not None and retry["best_JSD"]<rgn["best_JSD"] else rgn
 core=[
   ("Q-only",rq,rq["elapsed_solver_wall_s"],rq["counts"],"Q-only",rq["stop_reason"]),
   ("Exact-Adam",ra,ra["elapsed_solver_wall_s"],ra["counts"],"Exact-Adam",ra["stop_reason"]),
   ("Q+Bootstrap-GN",selected_gn,
      rq["elapsed_solver_wall_s"]+rgn["elapsed_solver_wall_s"]+(retry["elapsed_solver_wall_s"] if retry else 0),
      retry["inclusive_cost"] if retry else rgn["inclusive_cost"],"GN_retry" if selected_gn is retry else "GN_first",
      ("tolerance" if gnbest<=p2.TOL else "restart_cap" if retry else rgn["stop_reason"]))
 ]
 for name,src,wall,cost,stage,why in core:
  best=src["best_JSD"]
  final=src["final_JSD"]
  selected=torch.as_tensor(src["selected_policy_logits"],dtype=DT)
  selected_err=p2.exact_center_loss(g,selected,idx,target)
  if abs(selected_err-best)>1e-12:raise AssertionError("best policy not selected at exact best JSD")
  summary.append({"problem":key,"class":cl,"start":init,"solver":name,
    "best_exact_center_JSD":best,"final_exact_center_JSD":final,
    "selected_exact_center_JSD":selected_err,
    "selected_policy_logits":src["selected_policy_logits"],
    "selected_checkpoint_step":src["selected_policy_step"],"selected_stage":stage,
    "final_policy_logits":src["final_policy_logits"],
    "reached_threshold":best<=p2.TOL,"inclusive_solver_wall_seconds":wall,
    "estimated_nonmonitor_solver_wall_seconds":max(0.,wall-
       (ra.get("monitor_exact_forward_wall_s",0.) if name=="Exact-Adam" else rq.get("monitor_exact_forward_wall_s",0.))),
    "explicit_monitor_exact_forward_wall_s":(rq.get("monitor_exact_forward_wall_s",0.) if name!="Exact-Adam" else ra.get("monitor_exact_forward_wall_s",0.)),
    "explicit_monitor_exact_forward_count":(rq.get("monitor_exact_forward_count",0) if name!="Exact-Adam" else ra.get("monitor_exact_forward_count",0)),
    "counts_including_warmup_and_monitoring":cost,"stop_reason":why,
    "plateau_diagnostic_steps":src.get("plateau_diagnostic_count",0),
    "first_plateau_step":src.get("first_plateau_step"),
    "restart_used":bool(retry and name=="Q+Bootstrap-GN")})
 return {"problem":key,"class":cl,"init":init,"best":{
         "Q_only":rq["best_JSD"],"Exact_Adam":ra["best_JSD"],"Q_GN":gnbest},
         "restarted":retry is not None}


def verify_random_start_light(g,y,target,cl,seed):
 """Check newly enabled random Q/Adam for a tiny two-step exact replay.

 This verifies that a different initial state reaches the same logits using
 historical Adam update algebra, without executing an unbounded grid.
 """
 rng=np.random.default_rng(seed)
 z0=torch.as_tensor(rng.normal(0,.1,size=(g.d+1,g.d)),dtype=DT)
 idx=np.flatnonzero(y==cl)
 out={}
 for method in ("Q-only","Exact-Adam"):
  param=torch.nn.Parameter(z0.detach().clone())
  opt=torch.optim.Adam([param],lr=.12)
  for _ in range(2):
   opt.zero_grad(set_to_none=True)
   loss=g.local_q_loss(param,idx,target) if method=="Q-only" else jsd(g.center_train(param,idx),target)
   loss.backward();opt.step()
  # D09 algorithm's two-step parameter update independently reimplemented.
  # After the actual call below, compare parameter and exact-JSD output.
  runname=f"internal_random_init_regression_c{cl}_{method}"
  z1,r1=p2.adam_path(runname,method,g,idx,target,2,initial=z0,init_label="independent_random")
  diff=float(torch.max(torch.abs(z1-param.detach())).item())
  if diff>2e-12:raise AssertionError(f"random {method} mismatch {diff}")
  r1["internal_regression_only"]=True
  out[method]={"two_step_max_abs_logits_diff":diff}
 p2.persist()
 return out


def run():
 configure_p2()
 torch.set_num_threads(1)
 if any(f.exists() for f in (M,RAW,SUMMARY,CSV)):
  raise FileExistsError("B01_v2 exists; never overwrite or duplicate the block")
 for name in ("manifest.json","summary.json","trace.json","curve.csv"):
  if not (LEGACY/name).is_file():raise FileNotFoundError("B01 historical evidence missing: "+name)
 p2.START=time.perf_counter();p2.DEADLINE=p2.START+SECONDS
 signal.signal(signal.SIGUSR1,p2.signal_stop)
 evt=threading.Event()
 watch=threading.Thread(target=p2.watch_resources,args=(evt,),daemon=True)
 watch.start()
 p2.state["status"]="in_progress"
 p2.state["started"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
 p2.persist()
 try:
  plan=json.loads(PLAN.read_text(encoding="utf-8"))
  assert plan["status"]=="PROPOSED_AWAITING_MAIN_RUN_GATE"
  expected=[x for x in plan["runs"] if x["block_id"]==BID]
  if len(expected)!=18:raise AssertionError("B01 must have exactly 18 planned solver entries")
  spec=load_config(ROOT/"code/data_generator/config.json")
  assert spec["seed"]==20261102
  with tempfile.TemporaryDirectory() as t:
   dat=generate(spec,t,finite_q=False)
  all_x,all_y=dat["train"][0],dat["train"][2]
  rseed=20261201
  rng=np.random.default_rng(rseed)
  selected=np.concatenate([rng.choice(np.flatnonzero(all_y==c),50,replace=False)
                           for c in range(3)])
  rng.shuffle(selected)
  x,y=all_x[selected],all_y[selected]
  if np.bincount(y,minlength=3).tolist()!=[50,50,50]:
   raise AssertionError("B01 reference not balanced")
  g=StreamGraph(x,y,spec["gamma"],None,beta=.4,lam=.8,horizon=12)
  centers,wit_hash=target_per_class(g,y,teacher_seed=20261211)
  target_meta=[{"class":c,"target":t.tolist(),"target_sha256":sha_arr(t.numpy()),
                "self_jsd":float(jsd(t,t))} for c,t in enumerate(centers)]
  # HARD pre-solve pairing gate: same source data, reference mask and witnesses.
  old=json.loads((LEGACY/"manifest.json").read_text(encoding="utf8"))
  for key,value in {"source_train_X_sha256":sha_arr(all_x),
                    "source_train_y_sha256":sha_arr(all_y),
                    "selected_reference_ids_sha256":sha_arr(selected),
                    "teacher_logits_sha256_ONLY":wit_hash,
                    "generator_config_sha256":p2.sha256_file(ROOT/"code/data_generator/config.json"),
                    "plan_sha256":p2.sha256_file(PLAN)}.items():
   if old.get(key)!=value:raise AssertionError("B01/B01_v2 pairing mismatch in "+key)
  if old["targets"]!=target_meta:raise AssertionError("B01/B01_v2 target centers/hashes mismatch")
  if old["data_seed"]!=20261102 or old["reference_selection_seed"]!=rseed or old["teacher_seed"]!=20261211:
   raise AssertionError("Historical seed mismatch")
  if old["caps"]!={"Q_only":100,"Exact_Adam":150,"GN":30,"GN_neutral_restart":30}:
   raise AssertionError("Original caps differ")
  manifest={"stage":"LOCKED_BEFORE_FIRST_SOLVER_STEP",
    "version":"B01_v2","researcher":"RESEARCHER-01",
    "historical_B01_manifest_SHA256":p2.sha256_file(LEGACY/"manifest.json"),
    "historical_B01_summary_SHA256":p2.sha256_file(LEGACY/"summary.json"),
    "historical_B01_trace_SHA256":p2.sha256_file(LEGACY/"trace.json"),
    "historical_B01_curve_SHA256":p2.sha256_file(LEGACY/"curve.csv"),
    "paired_targets_exact_match":True,
    "best_checkpoint_policy_selection":"lowest exact multi-step per-class center JSD, recorded before solve; no validation/Test",
    "monitor_wall_policy":"separate exact forward after every optimizer step from total solver wall; checkpoint serializing remains included",
    "block":BID,"source_reference_chapter4":"de02fdda91e23db177de4429b3e1a9693c046d83",
    "parent_p2_report":"05316dd81d97170f61c46cef40048c5f1d2e46f2",
    "code_sha":os.getenv("GITHUB_SHA","unknown"),"script_sha256":p2.sha256_file(__file__),
    "instrumentation_sha256":p2.sha256_file(p2.__file__),
    "streamgraph_sha256":p2.sha256_file(ROOT/"code/experiments/diag_max_full300/stream_graph.py"),
    "generator_config_sha256":p2.sha256_file(ROOT/"code/data_generator/config.json"),
    "plan_sha256":p2.sha256_file(PLAN),"data_seed":spec["seed"],
    "source_train_X_sha256":sha_arr(all_x),"source_train_y_sha256":sha_arr(all_y),
    "reference_selection_seed":rseed,"selected_reference_ids_sha256":sha_arr(selected),
    "teacher_seed":20261211,"teacher_logits_sha256_ONLY":wit_hash,
    "teacher_given_to_solver":False,
    "targets":target_meta,"no_Test":True,"no_Monte_Carlo":True,"nref":150,
    "beta":.4,"lambda":.8,"H":12,"float64":True,"JSD_success_tol":1e-8,
    "caps":{"Q_only":100,"Exact_Adam":150,"GN":30,"GN_neutral_restart":30},
    "stop":{"on_success_exact_JSD":1e-8,"on_optimizer_cap":True,
            "on_actual_numerical_failure":True,"on_resources":True,
            "plateau_diagnostic_only":True,"plateau_termination":False,
            "min_step":p2.GLOBAL_CFG["stop_min_step"],
            "patience":p2.GLOBAL_CFG["stop_patience"],
            "min_relative_improvement":p2.GLOBAL_CFG["stop_relative_improvement"]},
    "time_guard_seconds":SECONDS,"rss_soft_limit_mib":PEAK_SOFT_MIB,
    "rss_ceiling_mib":1536,"expected_solver_runs":18,
    "test_forbidden":True,"chapter4_and_LaTeX_untouched":True}
  p2.atomic_json(M,manifest)
  p2.state["manifest_sha256"]=p2.sha256_file(M)
  p2.state["target_meta"]=target_meta
  p2.state["random_start_regression"]=verify_random_start_light(g,y,centers[0],0,20261221)
  p2.persist()
  runs=[]
  for cl in range(3):
   for init in ("zero","independent_random"):
    p2.stopcheck()
    seed=None if init=="zero" else 20261221+100*cl
    runs.append(solve_context(g,y,centers[cl],cl,init,seed,manifest["caps"],p2.state.setdefault("main_results",[])))
    p2.persist()
  main=p2.state["main_results"]
  if len(main)!=18:raise AssertionError(f"Need 18 completed solver records; got {len(main)}")
  seen={(r["class"],r["start"],r["solver"]) for r in main}
  if len(seen)!=18:raise AssertionError("Duplicated run context")
  if any(x["stop_reason"]=="plateau" for x in main):
   raise AssertionError("No plateau termination is permitted in v2")
  p2.state["status"]="completed"
  p2.state["stop_reason"]="B01_V2_18_SOLVERS_COMPLETE"
 except (KeyboardInterrupt,TimeoutError,MemoryError,RuntimeError,AssertionError,ArithmeticError) as exc:
  p2.state["status"]="stopped"
  p2.state["stop_reason"]=str(exc)
  p2.state["errors"].append(traceback.format_exc(limit=7))
 finally:
  evt.set()
  p2.state["finished"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
  p2.state["wall_numerical_seconds"]=time.perf_counter()-p2.START
  p2.persist()
  main=p2.state.get("main_results",[])
  rollup={}
  for method in ("Q-only","Exact-Adam","Q+Bootstrap-GN"):
   a=[x for x in main if x["solver"]==method]
   rollup[method]={"complete":len(a),"successes":sum(q["reached_threshold"] for q in a),
     "mean_best_JSD":float(np.mean([q["best_exact_center_JSD"] for q in a])) if a else None,
     "max_best_JSD":max((q["best_exact_center_JSD"] for q in a),default=None),
     "sum_inclusive_solver_time_s":sum(q["inclusive_solver_wall_seconds"] for q in a)}
  out={"block":BID,"version":"B01_v2","researcher":"RESEARCHER-01","status":p2.state["status"],"stop_reason":p2.state["stop_reason"],
    "wall_numerical_seconds":p2.state["wall_numerical_seconds"],
    "peak_rss_mib":p2.peak_mib(),"expected_solver_runs":18,"completed_solver_runs":len(main),
    "summaries":rollup,"runs":main,"target_metadata":p2.state.get("target_meta"),
    "random_start_regression":p2.state.get("random_start_regression"),
    "manifest_sha256":p2.state.get("manifest_sha256"),
    "notes":["B01_v2 is a separate versioned repeat of the exact same 18 paired solver cases; no legacy overwrite.",
    "Plateau only diagnostic, no optimizer is terminated by plateau.",
    "Best selected policy is the exact-JSD-minimum checkpoint, and final policy is separately preserved.",
    "Not the full 108 matrix; this is the first gated 18-solver block.",
    "No independent data-generator repeats in B01: all classes share one generated train dataset.",
    "Solver method wall includes P2 checkpoint logging/monitoring, exact-forward each step.",
    "GN inclusive time and counts include Q warmup plus optional neutral restart.",
    "A class target is guaranteed attainable by a hidden witness; no inference uniqueness/accuracy claims."]}
  p2.atomic_json(SUMMARY,out)
  print("INV01_P3_B01_RESULT",json.dumps({k:out[k] for k in
    ("status","stop_reason","wall_numerical_seconds","peak_rss_mib",
      "completed_solver_runs","summaries")},ensure_ascii=False),flush=True)
 if p2.state["status"]!="completed":
  raise SystemExit(2)


def verify():
 for f in (PLAN,M,RAW,CSV,SUMMARY):
  if not f.exists():raise FileNotFoundError(f)
 m=json.loads(M.read_text());d=json.loads(RAW.read_text());s=json.loads(SUMMARY.read_text())
 if m["stage"]!="LOCKED_BEFORE_FIRST_SOLVER_STEP" or m["nref"]!=150 or not m["stop"]["plateau_diagnostic_only"] or m["stop"]["plateau_termination"]:
  raise AssertionError("Missing locked B01 manifest")
 if s["status"]!="completed" or s["completed_solver_runs"]!=18:
  raise AssertionError("B01 incomplete")
 if s["peak_rss_mib"]>1536 or s["wall_numerical_seconds"]>900:
  raise AssertionError("B01 resource bounds exceeded")
 if any(not np.isfinite(r["best_exact_center_JSD"]) for r in s["runs"]):
  raise AssertionError("NaN result")
 if any(r["stop_reason"]=="plateau" for r in s["runs"]):raise AssertionError("Plateau termination found")
 if not m["paired_targets_exact_match"]:raise AssertionError("Target pairing gate missing")
 for r in s["runs"]:
  if abs(r["selected_exact_center_JSD"]-r["best_exact_center_JSD"])>1e-12:
   raise AssertionError("Wrong best checkpoint selection")
  if len(r["selected_policy_logits"])!=6 or any(len(row)!=5 for row in r["selected_policy_logits"]):
   raise AssertionError("Selected policy shape wrong")
 if not all(v["two_step_max_abs_logits_diff"]<2e-12 for v in s["random_start_regression"].values()):
  raise AssertionError("random start not checked")
 if any(len([x for x in s["runs"] if x["solver"]==method])!=6 for method in
    ("Q-only","Exact-Adam","Q+Bootstrap-GN")):
  raise AssertionError("unbalanced solver coverage")
 print("INV01_P3_B01_V2_VERIFY_PASS",s["completed_solver_runs"],
       s["wall_numerical_seconds"],s["peak_rss_mib"])



def precheck():
 """Tiny non-persistent regression: plateau fires each iteration but NEVER terminates."""
 old=(p2.persist,p2.stopcheck,p2.plateau,p2.state,p2.START,p2.DEADLINE)
 class ToyGraph:
  d=2;k=2
  def center_train(self,z,idx):
   logits=torch.stack((z[0,0]+z[1,0]+z[2,0],z[0,1]+z[1,1]+z[2,1]))
   return torch.softmax(logits,dim=0)
  def local_q_loss(self,z,idx,target):
   return torch.sum((self.center_train(z,idx)-target)**2)+0.001*torch.sum(z*z)
 try:
  p2.persist=lambda:None
  p2.stopcheck=lambda:0
  p2.plateau=lambda hist,step:step>=1
  p2.START=time.perf_counter()
  p2.DEADLINE=p2.START+20
  p2.state={"runs":[]}
  g=ToyGraph();target=torch.tensor([.95,.05],dtype=DT);idx=np.array([0])
  for method in ("Q-only","Exact-Adam"):
   _,r=p2.adam_path("internal_forced_plateau",method,g,idx,target,3)
   assert r["stop_reason"]=="cap" and r["counts"]["q_grad" if method=="Q-only" else "exact_grad"]==3
   assert r.get("plateau_diagnostic_count")==3
   assert abs(r["best_JSD"]-r["selected_policy_JSD_verified"])<1e-12
  z0=torch.zeros((3,2),dtype=DT)
  _,r=p2.gn_instrumented("internal_forced_plateau",g,idx,target,z0,2,0.,"warm")
  assert r.get("plateau_diagnostic_count",0)>=1 and r["stop_reason"]!="plateau"
  _,r=p2.adam_path("internal_tolerance", "Exact-Adam",g,idx,
                    torch.tensor([.5,.5],dtype=DT),3)
  assert r["stop_reason"]=="tolerance" and r["counts"]["exact_grad"]==0
  print("RESEARCHER-01_B01_V2_STOP_RULE_PRECHECK_PASS")
 finally:
  p2.persist,p2.stopcheck,p2.plateau,p2.state,p2.START,p2.DEADLINE=old

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 grp=ap.add_mutually_exclusive_group(required=True)
 grp.add_argument("--run",action="store_true");grp.add_argument("--verify",action="store_true")
 grp.add_argument("--precheck",action="store_true")
 arg=ap.parse_args()
 if arg.run:run()
 elif arg.precheck:precheck()
 else:verify()
