#!/usr/bin/env python3
"""CH4-EXP-INV-01/P2: bounded convergence-budget pilot; no full D09 matrix.

Uses unmodified StreamGraph and original D09 formulas; logs each exact-JSD
checkpoint, evaluation counts, time, memory, and normal GN/fallback/restart.
One numerical CI job; 840s numerical deadline (900s authorised ceiling),
soft RSS stop at 1430 MiB beneath 1536 MiB absolute envelope.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import threading
import time
import traceback

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(ROOT/"code/experiments/multistep_policy_target_match"))
sys.path.insert(0,str(ROOT/"code/experiments/diag_max_full300"))
from generate import generate,load_config
from stream_graph import StreamGraph
from run import DT,jsd
from bootstrap_inverse import warm_q,exact_adam,damped_gauss_newton

BUDGET_S=840.0
MEM_SOFT_LIMIT_MIB=1430.0
MEM_HARD_BOUND_MIB=1536.0
TOL=1e-8
GLOBAL_CFG={
 "phase":"P2_BOUNDED_PILOT_NOT_STANDARD_D09",
 "data_seed":20261102,"reference_seed":20261117,"teacher_seed":20261118,
 "solver_initialization_seed":20261221,
 "n_primary":150,"c_primary":0,"n_secondary":300,"c_secondary":0,
 "beta":0.4,"lambda":0.8,"H":12,"float":"float64",
 "threshold_JSD":TOL,"learning_rate":0.12,
 "budget_numerical_seconds":BUDGET_S,"monitor_soft_rss_mib":MEM_SOFT_LIMIT_MIB,
 "memory_hard_ceiling_mib":MEM_HARD_BOUND_MIB,
 "checkpoint_steps":[0,1,2,5,10,25,50,100],
 "stop_patience":18,"stop_min_step":25,"stop_relative_improvement":1e-4,
 "primary_caps":{"Q_only":100,"Exact_Adam":100,"GN":30,"GN_neutral_restart":30},
 "optional_random_init_Exact_Adam_steps":50,
 "secondary_caps":{"Q_only":25,"Exact_Adam":25,"GN":10,"GN_neutral_restart":10},
 "secondary_only_after_primary_completed":True,
 "secondary_requires_any_primary_solver_tolerance":True,
 "no_Monte_Carlo":True,"no_Test":True,"no_LaTeX":True,"no_standard_108_runs":True,
}
RESULTS=HERE/"results"
MANIFEST=RESULTS/"p2_manifest.json"
RAW=RESULTS/"p2_trace.json"
CSV=RESULTS/"p2_curve.csv"
SUMMARY=RESULTS/"p2_summary.json"
REF_SHA="7f32e07119b768155bc0163e9383c7eb007cc983"
START=0.0
DEADLINE=0.0
state={"started":None,"finished":None,"status":"not_started","stop_reason":None,
       "runs":[],"errors":[],"wall_numerical_seconds":None,"peak_rss_mib":0.0}
last_task="setup"
stop_signal_reason=None


def sha256_file(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(a):
 return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def proc_rss_mib():
 for l in Path("/proc/self/status").read_text().splitlines():
  if l.startswith("VmRSS:"):return int(l.split()[1])/1024.
 raise RuntimeError("VmRSS unavailable")


def peak_mib():
 return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.


def atomic_json(path,thing):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+".partial")
 tmp.write_text(json.dumps(thing,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 os.replace(tmp,path)


def write_curve():
 rows=[]
 for run in state["runs"]:
  for p in run["trace"]:
   rows.append({"problem":run["problem"],"method":run["method"],"stage":p.get("stage",""),
    "step":p["step"],"method_wall_s":p["method_wall_s"],"inclusive_wall_s":p["inclusive_wall_s"],
    "global_compute_wall_s":p["global_compute_wall_s"],"exact_center_JSD":p["exact_center_JSD"],
    "best_JSD":p["best_JSD"],"surrogate":p.get("surrogate",""),
    **p["cost"],"accepted":p.get("accepted",""),"fallback":p.get("fallback","")})
 if not rows:return
 fields=list(rows[0])
 tmp=CSV.with_suffix(".csv.partial")
 with tmp.open("w",encoding="utf8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
  w.writeheader();w.writerows(rows)
 os.replace(tmp,CSV)


def persist():
 state["peak_rss_mib"]=max(state["peak_rss_mib"],peak_mib())
 state["wall_numerical_seconds"]=max(0.,time.perf_counter()-START) if START else 0.
 atomic_json(RAW,state)
 write_curve()


def stopcheck():
 if stop_signal_reason:raise RuntimeError("RESOURCE_"+stop_signal_reason)
 if time.perf_counter()>=DEADLINE:raise TimeoutError("NUMERICAL_BUDGET")
 p=peak_mib()
 if p>=MEM_SOFT_LIMIT_MIB:raise MemoryError(f"RSS soft limit {p:.2f} MiB")
 return p


def signal_stop(_signal,_frame):
 raise RuntimeError("RESOURCE_MONITOR_REQUEST")


def watch_resources(stop_evt):
 global stop_signal_reason
 while not stop_evt.wait(0.06):
  try:
   if peak_mib()>=MEM_SOFT_LIMIT_MIB:
    stop_signal_reason="RSS_LIMIT"
    os.kill(os.getpid(),signal.SIGUSR1)
    break
   if time.perf_counter()>=DEADLINE:
    stop_signal_reason="NUMERICAL_TIME_LIMIT"
    os.kill(os.getpid(),signal.SIGUSR1)
    break
  except (OSError,ProcessLookupError):
   break


def counts():
 return {"q_grad":0,"exact_grad":0,"exact_forward":0,"jacobian":0,"line_search":0,"retries":0}


def addcost(a,b):
 return {k:a.get(k,0)+b.get(k,0) for k in counts()}


def exact_center_loss(g,z,idx,target):
 with torch.no_grad():
  center=g.center_train(z,idx)
  if not bool(torch.isfinite(center).all()):raise ArithmeticError("nonfinite center")
  if abs(center.sum().item()-1)>1e-8:raise ArithmeticError("center mass not 1")
  val=float(jsd(center,target).item())
  if not np.isfinite(val) or val < -1e-14:raise ArithmeticError("nonfinite JSD")
  return val


def trace_point(run,kind,step,error,begin,cost,offset=0,surrogate=None,accepted=None,fallback=None):
 stopcheck()
 if not np.isfinite(error):raise ArithmeticError("nonfinite traced center error")
 t=time.perf_counter()
 best=min(error,run["best_JSD"])
 run["best_JSD"]=best
 row={"stage":kind,"step":step,"exact_center_JSD":error,"best_JSD":best,
      "method_wall_s":t-begin,"inclusive_wall_s":t-begin+offset,
      "global_compute_wall_s":t-START,"cost":copy.deepcopy(costs_normalized(cost)),
      "rss_mib":proc_rss_mib()}
 if surrogate is not None:row["surrogate"]=surrogate
 if accepted is not None:row["accepted"]=bool(accepted)
 if fallback is not None:row["fallback"]=fallback
 run["trace"].append(row)
 persist()


def costs_normalized(c):
 return {key:int(c.get(key,0)) for key in counts()}


def plateau(history,step):
 if step<GLOBAL_CFG["stop_min_step"] or len(history)<GLOBAL_CFG["stop_patience"]+1:return False
 patience=GLOBAL_CFG["stop_patience"]
 now=min(x["exact_center_JSD"] for x in history)
 earlier=min(x["exact_center_JSD"] for x in history[:-patience])
 if earlier<=TOL:return False
 return (earlier-now)/max(earlier,1e-14)<GLOBAL_CFG["stop_relative_improvement"]


def new_run(problem,method,initial):
 r={"problem":problem,"method":method,"initialization":initial,
    "status":"in_progress","stop_reason":None,"best_JSD":float("inf"),
    "final_JSD":None,"reached_tolerance":False,"counts":counts(),"trace":[]}
 state["runs"].append(r);persist()
 return r


def finish_run(r,g,z,idx,target,reason):
 err=exact_center_loss(g,z,idx,target)
 r["final_JSD"]=err
 r["best_JSD"]=min(r["best_JSD"],err)
 r["reached_tolerance"]=r["best_JSD"]<=TOL
 r["stop_reason"]=reason
 r["status"]="complete"
 persist()
 return z.detach().clone()


def adam_path(problem,method,g,idx,target,cap,initial=None,init_label="zero"):
 """Literal D09 Adam update, with exact center verification after each step."""
 stopcheck()
 r=new_run(problem,method,init_label)
 begin=time.perf_counter()
 z=torch.nn.Parameter(torch.zeros((g.d+1,g.d),dtype=DT) if initial is None else initial.detach().clone())
 opt=torch.optim.Adam([z],lr=GLOBAL_CFG["learning_rate"])
 cost=r["counts"]
 e=exact_center_loss(g,z,idx,target);cost["exact_forward"]+=1
 trace_point(r,method,0,e,begin,cost)
 reason="cap"
 for step in range(1,cap+1):
  stopcheck()
  if e<=TOL:reason="tolerance";break
  opt.zero_grad(set_to_none=True)
  loss=g.local_q_loss(z,idx,target) if method=="Q-only" else jsd(g.center_train(z,idx),target)
  if not bool(torch.isfinite(loss)):raise ArithmeticError("loss not finite")
  surrogate=float(loss.detach().item())
  loss.backward()
  if not bool(torch.isfinite(z.grad).all()):raise ArithmeticError("gradient not finite")
  opt.step()
  cost["q_grad" if method=="Q-only" else "exact_grad"]+=1
  e=exact_center_loss(g,z,idx,target);cost["exact_forward"]+=1
  trace_point(r,method,step,e,begin,cost,surrogate=surrogate)
  if e<=TOL:reason="tolerance";break
  if plateau(r["trace"],step):reason="plateau";break
 finish_run(r,g,z,idx,target,reason)
 r["elapsed_solver_wall_s"]=time.perf_counter()-begin
 return z.detach().clone(),r


def gn_instrumented(problem,g,idx,target,z_start,cap,offset,phase,cost_prior=None):
 """Exact D09 GN algebra, line-search, fallback and damping with instrumentation."""
 stopcheck()
 r=new_run(problem,"Q+Bootstrap-GN" if phase=="warm" else "Q+Bootstrap-GN-restart","from_Q" if phase=="warm" else "neutral")
 begin=time.perf_counter()
 z=z_start.detach().clone()
 cost=r["counts"]
 e=exact_center_loss(g,z,idx,target);cost["exact_forward"]+=1
 trace_point(r,phase,0,e,begin,cost,offset)
 damping=1e-4
 reason="cap"
 for iteration in range(1,cap+1):
  stopcheck()
  if e<=TOL:reason="tolerance";break
  zflat=z.reshape(-1).detach().requires_grad_(True)
  v=g.center_train(zflat.reshape(g.d+1,g.d),idx)[:-1]
  rows=[torch.autograd.grad(v[c],zflat,retain_graph=c<g.k-2)[0]
        for c in range(g.k-1)]
  J=torch.stack(rows)
  cost["jacobian"]+=1
  residual=target[:-1]-v.detach()
  try:
   change=J.T@torch.linalg.solve(J@J.T+damping*torch.eye(g.k-1,dtype=DT),residual)
  except RuntimeError:
   reason="singular_solve";break
  magnitude=torch.linalg.vector_norm(change)
  if not bool(torch.isfinite(magnitude)) or magnitude<1e-12:
   reason="small_or_nonfinite_step";break
  if magnitude>2.5:change=change*(2.5/magnitude)
  accepted=False;fallback=None
  for shrink in range(12):
   stopcheck()
   trial=z+change.reshape(z.shape)*(0.5**shrink)
   new_e=exact_center_loss(g,trial,idx,target)
   cost["exact_forward"]+=1;cost["line_search"]+=1
   if new_e<e-1e-14:
    z,e,accepted=trial.detach(),new_e,True
    damping=max(damping*.7,1e-8);break
  if not accepted:
   zg=z.detach().clone().requires_grad_(True)
   merit=jsd(g.center_train(zg,idx),target)
   grad=torch.autograd.grad(merit,zg)[0]
   cost["exact_grad"]+=1
   gn=torch.linalg.vector_norm(grad)
   if bool(torch.isfinite(gn)) and gn>1e-14:
    direction=-grad/gn*min(2.5,1.)
    for shrink in range(20):
     stopcheck()
     trial=z+direction*(.5**shrink)
     new_e=exact_center_loss(g,trial,idx,target)
     cost["exact_forward"]+=1;cost["line_search"]+=1
     if new_e<e-1e-14:
      z,e,accepted=trial.detach(),new_e,True
      fallback="exact_gradient";break
   damping*=10
  trace_point(r,phase,iteration,e,begin,cost,offset,accepted=accepted,fallback=fallback)
  if e<=TOL:reason="tolerance";break
  if not accepted and damping>1e4:reason="damping_limit";break
  if plateau(r["trace"],iteration):reason="plateau";break
 finish_run(r,g,z,idx,target,reason)
 r["elapsed_solver_wall_s"]=time.perf_counter()-begin
 r["offset_warmup_s"]=offset
 r["inclusive_cost"]=addcost(cost,cost_prior or {})
 persist()
 return z.detach().clone(),r


def norm_trace_close(a,b,name,limit=2e-9):
 if len(a)!=len(b):raise AssertionError(f"{name}: trace lengths mismatch")
 err=max(abs(x-y) for x,y in zip(a,b))
 if err>limit:raise AssertionError(f"{name}: mismatch {err}")
 return err


def tiny_solver_equivalence(spec,x,y):
 """Run instrumented update rules on 36-only and check original D09 functions."""
 g=StreamGraph(x[:36],y[:36],spec["gamma"],None,beta=.4,lam=.8,horizon=12)
 idx=np.flatnonzero(y[:36]==0)
 if len(idx)<2:raise AssertionError("tiny reference insufficient")
 teacher=np.random.default_rng(312).normal(size=(g.d+1,g.d))
 teacher-=teacher.mean(axis=-1,keepdims=True)
 target=g.center_train(torch.tensor(teacher,dtype=DT),idx).detach()
 # Avoid counting this test as primary experiment.
 tiny_name="internal_regression_n36"
 # We run original and instrumented with cap=3, record in trace and mark internal.
 zq_old,_,_=warm_q(g,idx,target,3,.12)
 zq_new,rq=adam_path(tiny_name,"Q-only",g,idx,target,3)
 old_exact=float(jsd(g.center_train(zq_old,idx),target))
 dq=float(torch.max(torch.abs(zq_old-zq_new)))
 if dq>3e-11:raise AssertionError("Q update mismatch")
 za_old,_,_,za_trace=exact_adam(g,idx,target,3,.12,1e-14)
 za_new,ra=adam_path(tiny_name,"Exact-Adam",g,idx,target,3)
 da=float(torch.max(torch.abs(za_old-za_new)))
 if da>3e-11:raise AssertionError("Adam update mismatch")
 gn_old,_,oldcost,gn_trace=damped_gauss_newton(g,idx,target,zq_old,3,1e-14)
 gn_new,rg=gn_instrumented(tiny_name,g,idx,target,zq_new,3,0.0,"warm")
 dg=float(torch.max(torch.abs(gn_old-gn_new)))
 if dg>3e-10:raise AssertionError("GN full update mismatch")
 if oldcost["jacobian"]!=rg["counts"]["jacobian"]:raise AssertionError("GN Jacobian count mismatch")
 norm_trace_close([q["error"] for q in gn_trace],[q["exact_center_JSD"] for q in rg["trace"]],"GN trace")
 # Tiny regression outputs are not counted as experimental question/independent sample.
 for r in (rq,ra,rg):
  r["internal_regression_only"]=True
 persist()
 return {"same_q_logits_max_abs":dq,"same_adam_logits_max_abs":da,
         "same_gn_logits_max_abs":dg,"GN_jacobians":oldcost["jacobian"],
         "original_GN_fallback_count":oldcost["exact_grad"],
         "first_Q_final_error":old_exact}


def make_problem(spec,full_x,full_y,n,reference_seed,teacher_seed):
 if n==150:
  rng=np.random.default_rng(reference_seed)
  selected=np.concatenate([rng.choice(np.flatnonzero(full_y==c),50,replace=False) for c in range(3)])
  rng.shuffle(selected)
 elif n==300:
  selected=np.arange(len(full_y))
 else:raise ValueError(n)
 x,y=full_x[selected],full_y[selected]
 if n==150 and np.bincount(y,minlength=3).tolist()!=[50,50,50]:
  raise AssertionError("not balanced")
 g=StreamGraph(x,y,spec["gamma"],None,beta=.4,lam=.8,horizon=12)
 idx=np.flatnonzero(y==0)
 rng=np.random.default_rng(teacher_seed)
 policy=rng.normal(0,1.7,size=(3,g.d+1,g.d))
 policy-=policy.mean(axis=-1,keepdims=True)
 witness=policy[0].copy()
 with torch.no_grad():
  target=g.center_train(torch.tensor(witness,dtype=DT),idx).detach()
 witness_error=float(jsd(target,target))
 if abs(target.sum().item()-1)>1e-10 or witness_error>1e-12:
  raise AssertionError("teacher target invalid")
 h=sha256_array(witness)
 del policy,witness
 return g,idx,target,{
  "n":n,"class_counts":np.bincount(y,minlength=3).tolist(),
  "target":target.tolist(),"target_sha256":sha256_array(target.numpy()),
  "reference_ids_sha256":sha256_array(selected),
  "witness_sha256_only":h,"witness_self_jsd":witness_error,
  "witness_given_to_solvers":False
 }


def solve_problem(g,idx,target,problem,caps,allow_extra=False):
 """A single matched problem; Q-only/Adam/GN from zero; optional one extra start."""
 t0=time.perf_counter()
 zq,rq=adam_path(problem,"Q-only",g,idx,target,caps["Q_only"])
 za,ra=adam_path(problem,"Exact-Adam",g,idx,target,caps["Exact_Adam"])
 warm_s=rq["elapsed_solver_wall_s"]
 zb,rgn=gn_instrumented(problem,g,idx,target,zq,caps["GN"],warm_s,"warm",rq["counts"])
 # Retry only after GN failure, as in historical D09.
 restart=None
 if rgn["best_JSD"]>TOL and caps["GN_neutral_restart"]>0:
  rz,rr=gn_instrumented(problem,g,idx,target,
                        torch.zeros_like(zq),caps["GN_neutral_restart"],
                        warm_s+rgn["elapsed_solver_wall_s"],"neutral_restart",
                        addcost(rq["counts"],rgn["counts"]))
  rr["counts"]["retries"]=1
  rr["inclusive_cost"]=addcost(rr["inclusive_cost"],{"retries":1})
  restart=rr
  best=min(rgn["best_JSD"],rr["best_JSD"])
 else:best=rgn["best_JSD"]
 random=None
 if allow_extra and stopcheck()<MEM_SOFT_LIMIT_MIB and DEADLINE-time.perf_counter()>90:
  rng=np.random.default_rng(GLOBAL_CFG["solver_initialization_seed"])
  initial=torch.tensor(rng.normal(0,.1,size=(g.d+1,g.d)),dtype=DT)
  _,random=adam_path(problem,"Exact-Adam",g,idx,target,
                        GLOBAL_CFG["optional_random_init_Exact_Adam_steps"],
                        initial=initial,init_label="independent_random")
 summary={"problem":problem,"n":g.n,"class":0,
          "Q_only":rq["best_JSD"],"Exact_Adam":ra["best_JSD"],
          "Q+Bootstrap-GN":best,
          "GN_first":rgn["best_JSD"],
          "GN_retry":None if restart is None else restart["best_JSD"],
          "random_Exact_Adam":None if random is None else random["best_JSD"],
          "wall_problem_s":time.perf_counter()-t0,
          "any_reached_tolerance":min(rq["best_JSD"],ra["best_JSD"],best)<=TOL}
 return summary


def max_budget_comparison():
 """Checkpoint-at-or-before comparisons. NO extrapolation, no interpolation."""
 grid=[.05,.1,.2,.5,1,2,5,10,20,40,60,90,120,180,240]
 comp={}
 for problem in sorted(set(r["problem"] for r in state["runs"] if not r.get("internal_regression_only"))):
  comp[problem]={}
  for run in state["runs"]:
   if run["problem"]!=problem or run.get("internal_regression_only"):continue
   checkpoints=[]
   for second in grid:
    available=[t for t in run["trace"] if t["inclusive_wall_s"]<=second]
    if not available:continue
    t=available[-1]
    checkpoints.append({"budget_wall_s":second,"checkpoint_wall_s":t["inclusive_wall_s"],
                        "step":t["step"],"best_JSD_at_or_before_budget":
                          min(x["exact_center_JSD"] for x in available),
                        "note":"last saved checkpoint at/before time; no interpolation"})
   comp[problem][run["method"]+"/"+run["initialization"]]=checkpoints
 return comp


def main():
 global START,DEADLINE
 torch.set_num_threads(1)
 if RAW.exists() or MANIFEST.exists():
  raise FileExistsError("P2 evidence already present: no repeat numerical job permitted")
 START=time.perf_counter();DEADLINE=START+BUDGET_S
 stop_evt=threading.Event()
 signal.signal(signal.SIGUSR1,signal_stop)
 watcher=threading.Thread(target=watch_resources,args=(stop_evt,),daemon=True)
 watcher.start()
 state["started"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
 state["status"]="in_progress";persist()
 try:
  src=ROOT/"code/data_generator/config.json"
  spec=load_config(src)
  if spec["seed"]!=GLOBAL_CFG["data_seed"] or spec["sizes"]["train"]!=300:raise AssertionError("source config changed")
  with tempfile.TemporaryDirectory() as temp:
   data=generate(spec,temp,finite_q=False)
  x,y=data["train"][0],data["train"][2]
  if np.bincount(y,minlength=3).tolist()!=[72,147,81]:raise AssertionError("source class counts changed")
  # Before any solver, lock exact source & target, objective, stop rules and limits.
  g,idx,target,problemmeta=make_problem(spec,x,y,150,GLOBAL_CFG["reference_seed"],GLOBAL_CFG["teacher_seed"])
  manifest={
   "manifest_stage":"LOCKED_BEFORE_SOLVE",
   "source_chapter4":REF_SHA,
   "code_execution_SHA":os.getenv("GITHUB_SHA","unreported"),
   "script_file_sha256":sha256_file(__file__),
   "streamgraph_sha256":sha256_file(ROOT/"code/experiments/diag_max_full300/stream_graph.py"),
   "D09_solver_sha256":sha256_file(ROOT/"code/experiments/multistep_policy_target_match/bootstrap_inverse.py"),
   "config_file_sha256":sha256_file(src),
   "train_X_sha256":sha256_array(x),"train_y_sha256":sha256_array(y),
   "fixed_settings":GLOBAL_CFG,
   "primary_problem":problemmeta,
   "one_numeric_job":True,"CI_install_excluded_from_solver_time":True,
   "Test_read":False,
   "expected_results":["p2_trace.json","p2_curve.csv","p2_summary.json"]
  }
  atomic_json(MANIFEST,manifest)
  state["manifest_sha256"]=sha256_file(MANIFEST)
  state["primary_problem_metadata"]=problemmeta
  persist()
  # Cheap compatibility check, no new Monte Carlo / Test.
  state["instrumentation_regression"]=tiny_solver_equivalence(spec,x,y)
  persist()
  state["primary_summary"]=solve_problem(g,idx,target,"n150_class0",
                           GLOBAL_CFG["primary_caps"],allow_extra=True)
  del g
  persist()
  # Only if first primary class has produced a true-threshold success, and ample wall budget.
  if (state["primary_summary"]["any_reached_tolerance"] and
      time.perf_counter()-START <BUDGET_S*.65 and
      peak_mib()<MEM_SOFT_LIMIT_MIB*.85):
   g2,idx2,target2,meta2=make_problem(spec,x,y,300,GLOBAL_CFG["reference_seed"],GLOBAL_CFG["teacher_seed"])
   state["secondary_problem_metadata"]=meta2
   state["secondary_summary"]=solve_problem(g2,idx2,target2,"n300_class0",
                   GLOBAL_CFG["secondary_caps"],allow_extra=False)
   del g2
   persist()
  else:
   state["secondary_skip_reason"]="Primary not at tolerance or insufficient budget/headroom; authorized secondary is optional"
  state["status"]="completed"
  state["stop_reason"]="bounded_protocol_completed"
 except (KeyboardInterrupt,TimeoutError,MemoryError,RuntimeError,ArithmeticError,AssertionError) as exc:
  state["status"]="stopped"
  state["stop_reason"]=type(exc).__name__+": "+str(exc)
  state["errors"].append(traceback.format_exc(limit=5))
 finally:
  stop_evt.set()
  state["finished"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
  state["peak_rss_mib"]=max(state["peak_rss_mib"],peak_mib())
  state["wall_numerical_seconds"]=time.perf_counter()-START
  persist()
  summary={"status":state["status"],"stop_reason":state["stop_reason"],
           "wall_numerical_seconds":state["wall_numerical_seconds"],
           "peak_rss_mib":state["peak_rss_mib"],
           "primary_summary":state.get("primary_summary"),
           "secondary_summary":state.get("secondary_summary"),
           "secondary_skip_reason":state.get("secondary_skip_reason"),
           "curve_budget_comparison":max_budget_comparison(),
           "retries":[{"problem":r["problem"],"method":r["method"],"counts":r["counts"],
                       "stop_reason":r["stop_reason"],"best_JSD":r["best_JSD"]}
                      for r in state["runs"] if r["method"].endswith("-restart")],
           "internal_regression":state.get("instrumentation_regression"),
           "manifest_sha256":state.get("manifest_sha256"),
           "notes":["Same teacher-generated target; the witness policy is never given to the solver.",
           "Q-only optimizes a local surrogate; success is exact multi-step JSD.",
           "GN inclusive elapsed and evaluation counts include Q warm start and actual retries.",
           "RSS peak is whole Python process; 15 min is the whole numerical mission, not every method.",
           "Any time-grid comparison uses last available checkpoint at/before, never interpolation.",
           "n300 is optional and gated; no complete n300 optimization and no 108-run matrix."]}
  atomic_json(SUMMARY,summary)
  print("P2_BOUNDED_RESULT",json.dumps({k:summary.get(k) for k in
       ("status","stop_reason","wall_numerical_seconds","peak_rss_mib",
        "primary_summary","secondary_summary","secondary_skip_reason")}),flush=True)
 if state["status"]!="completed":
  # Preserve all files and force explicit STOP/REVISE review in Issue.
  raise SystemExit(2)


def verify():
 for f in [MANIFEST,RAW,CSV,SUMMARY]:
  if not f.exists():raise FileNotFoundError(f)
 m=json.loads(MANIFEST.read_text());d=json.loads(RAW.read_text());s=json.loads(SUMMARY.read_text())
 c=m["fixed_settings"]
 if m["manifest_stage"]!="LOCKED_BEFORE_SOLVE" or c["no_Monte_Carlo"] is not True or c["no_standard_108_runs"] is not True:
  raise AssertionError("Manifest invalid")
 if c["n_primary"]!=150 or m["primary_problem"]["class_counts"]!=[50,50,50]:
  raise AssertionError("Wrong problem")
 if s["wall_numerical_seconds"]>900 or s["peak_rss_mib"]>MEM_HARD_BOUND_MIB:
  raise AssertionError("Resource budget exceeded")
 if d["status"]!="completed":raise AssertionError("P2 exited before complete; inspect preserved outputs")
 for r in d["runs"]:
  if r["status"]!="complete":raise AssertionError("Incomplete method")
  if not r["trace"]:raise AssertionError("Empty trace")
  for p in r["trace"]:
   if not np.isfinite(p["exact_center_JSD"]):raise AssertionError("Invalid JSD")
   if not np.isfinite(p["inclusive_wall_s"]):raise AssertionError("Invalid wall time")
  if r["method"].startswith("Q+Bootstrap-GN"):
   # Includes normal correction, fallback, optional retry.
   for a,b in zip(r["trace"],r["trace"][1:]):
    if b["exact_center_JSD"]>a["exact_center_JSD"]+1e-12:
     raise AssertionError("GN accepted worse JSD")
 if not d.get("instrumentation_regression"):raise AssertionError("No historical solver trace comparison")
 if max(d["instrumentation_regression"].get(k,1.) for k in
        ("same_q_logits_max_abs","same_adam_logits_max_abs","same_gn_logits_max_abs"))>3e-10:
  raise AssertionError("Instrumented solvers diverge")
 print("P2_VERIFICATION_PASS",s["status"],s["wall_numerical_seconds"],s["peak_rss_mib"])


if __name__=="__main__":
 ap=argparse.ArgumentParser()
 g=ap.add_mutually_exclusive_group(required=True)
 g.add_argument("--run",action="store_true")
 g.add_argument("--verify",action="store_true")
 args=ap.parse_args()
 if args.run:main()
 else:verify()
