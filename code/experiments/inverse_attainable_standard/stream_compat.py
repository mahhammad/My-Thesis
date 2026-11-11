#!/usr/bin/env python3
"""Phase-1 StreamGraph equivalence, full D09 solver-path differentiation & REAL cost.

Bounded: tiny legacy equivalence, one source Train, class-0 probes at n=150/300,
Q<=2, Exact-Adam<=2, GN<=2, one forced-fallback GN; no Monte Carlo or main runs.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code/data_generator"))
# The CI stages *unaltered* archived code in its original import-relative location.
sys.path.insert(0,str(ROOT/"code/experiments/multistep_policy_target_match"))
sys.path.insert(0,str(ROOT/"code/experiments/diag_max_full300"))
from generate import generate,load_config
from run import ExactGraph,DT,jsd
from stream_graph import StreamGraph
from bootstrap_inverse import warm_q,exact_adam,damped_gauss_newton,exact_error

OUT=HERE/"results"/"streamgraph_compat_cost.json"
BASE_SHA="de02fdda91e23db177de4429b3e1a9693c046d83"


def rss_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])/1024
    raise RuntimeError("Linux VmRSS required")


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024


def timed(name,fn):
    gc.collect()
    start_rss=rss_mb()
    begin=time.perf_counter()
    result=fn()
    sec=time.perf_counter()-begin
    return result,{"operation":name,"wall_s":sec,
      "rss_before_mib":start_rss,"rss_after_mib":rss_mb(),
      "process_peak_rss_mib_after":peak_mb()}


def validate_trace(trace,cost,initial,target,graph,idx):
    values=[r["error"] for r in trace]
    if not all(np.isfinite(values)) or any(b>a+1e-12 for a,b in zip(values,values[1:])):
        raise AssertionError("GN error not monotone finite")
    actual=exact_error(graph,target,idx,initial) if False else None
    if cost["jacobian"]!=len(trace)-1:
        raise AssertionError("GN count mismatch")
    if cost["exact_forward"]<cost["jacobian"]+1:
        raise AssertionError("GN forward count mismatch")
    return {"step_count":len(trace)-1,"errors":values,
            "accepted":[t.get("accepted") for t in trace[1:]],
            "fallback_count":sum(t.get("fallback")=="exact_gradient" for t in trace),
            "counts":cost}


def matrix_error(a,b):
    return float(torch.max(torch.abs(a-b)).detach().item())


def jac_rows(graph,z,idx):
    p=z.detach().clone().requires_grad_(True)
    v=graph.center_train(p,idx)[:-1]
    arr=[]
    for c in range(graph.k-1):
        arr.append(torch.autograd.grad(v[c],p,retain_graph=c<graph.k-2)[0])
    return torch.stack(arr).detach()


def derivative_check(graph,z,idx,target,coords=((0,0),(1,2),(5,0))):
    p=z.detach().clone().requires_grad_(True)
    # Check BOTH GN Jacobian and the fallback's exact-JSD gradient.
    v=graph.center_train(p,idx)
    jac=[]
    for c in range(graph.k-1):
        jac.append(torch.autograd.grad(v[c],p,retain_graph=True)[0])
    J=torch.stack(jac).detach()
    gl=torch.autograd.grad(jsd(v,target),p)[0].detach()
    h=1e-5
    max_J,max_loss=0.,0.
    for a,b in coords:
        plus=p.detach().clone();minus=p.detach().clone()
        plus[a,b]+=h;minus[a,b]-=h
        with torch.no_grad():
            vp=graph.center_train(plus,idx)
            vm=graph.center_train(minus,idx)
            fd=(vp[:-1]-vm[:-1])/(2*h)
            lg=(jsd(vp,target)-jsd(vm,target))/(2*h)
        max_J=max(max_J,float((fd-J[:,:,a,b]).abs().max())) if False else max(
                 max_J,float((fd-J[:,a,b]).abs().max()))
        max_loss=max(max_loss,float(abs(lg-gl[a,b])))
    if max_J>3e-6 or max_loss>3e-6:
        raise AssertionError(f"finite-difference mismatch GN Jacobian={max_J} JSD_gradient={max_loss}")
    return {"max_jacobian_fd_error":max_J,
            "max_exact_jsd_gradient_fd_error":max_loss,
            "coordinates":[list(q) for q in coords],"step_h":h}


def tiny_equivalence(x,y,gamma):
    n=36
    xx=x[:n]; yy=y[:n]; idx=np.flatnonzero(yy==0)
    if len(idx)<2 or len(set(yy))!=3:raise AssertionError("tiny reference invalid")
    legacy=ExactGraph(xx,yy,gamma,np.arange(n),xx[:3],horizon=12,beta=.4,lam=.8)
    stream=StreamGraph(xx,yy,gamma,xx[:3],horizon=12,beta=.4,lam=.8)
    teacher=np.random.default_rng(20261118).normal(0,1.7,size=(6,5))
    with torch.no_grad():target=stream.center_train(torch.tensor(teacher,dtype=DT),idx).detach()
    z=torch.tensor(np.random.default_rng(20261119).normal(0,.18,size=(6,5)),dtype=DT)
    errs={}
    va=legacy.center_train(z,idx,return_all=True)
    vb=stream.center_train(z,idx,return_all=True)
    errs["sample_distribution"]=matrix_error(va,vb)
    errs["center"]=matrix_error(va.mean(0),vb.mean(0))
    errs["q_objective"]=abs(float(legacy.local_q_loss(z,idx,target)-
                                   stream.local_q_loss(z,idx,target)))
    errs["center_jacobian"]=matrix_error(jac_rows(legacy,z,idx),jac_rows(stream,z,idx))
    def objective_grad(g,kind):
        p=z.clone().requires_grad_(True)
        loss=g.local_q_loss(p,idx,target) if kind=="q" else jsd(g.center_train(p,idx),target)
        return torch.autograd.grad(loss,p)[0]
    errs["q_gradient"]=matrix_error(objective_grad(legacy,"q"),objective_grad(stream,"q"))
    errs["exact_jsd_gradient"]=matrix_error(objective_grad(legacy,"exact"),objective_grad(stream,"exact"))
    if any(v>3e-10 for v in errs.values()):
        raise AssertionError(f"StreamGraph ≠ legacy in tiny graph: {errs}")
    # Call ORIGINAL D09 solver implementations, not hand-rewritten approximations.
    q0,_,cq0=warm_q(legacy,idx,target,4,.12)
    q1,_,cq1=warm_q(stream,idx,target,4,.12)
    errs["Q_only_final_logits"]=matrix_error(q0,q1)
    a0,_,ca0,tr0=exact_adam(legacy,idx,target,4,.12,1e-12)
    a1,_,ca1,tr1=exact_adam(stream,idx,target,4,.12,1e-12)
    errs["Exact_Adam_final_logits"]=matrix_error(a0,a1)
    errs["Exact_Adam_trace"]=max(abs(l["error"]-r["error"]) for l,r in zip(tr0,tr1))
    b0,_,cb0,tb0=damped_gauss_newton(legacy,idx,target,q0,2,1e-12)
    b1,_,cb1,tb1=damped_gauss_newton(stream,idx,target,q1,2,1e-12)
    errs["GN_final_logits"]=matrix_error(b0,b1)
    errs["GN_trace"]=max(abs(l["error"]-r["error"]) for l,r in zip(tb0,tb1))
    validate_trace(tb1,cb1,target,b1,stream,idx)
    # Force the alternative EXACT-JSD gradient branch through max_backtrack=0:
    # zero GN line-search slots; the original solver still performs a full
    # fallback-gradient line search. This does not change production behavior.
    f0,_,cf0,tf0=damped_gauss_newton(legacy,idx,target,q0,1,1e-12,max_backtrack=0)
    f1,_,cf1,tf1=damped_gauss_newton(stream,idx,target,q1,1,1e-12,max_backtrack=0)
    errs["GN_fallback_final_logits"]=matrix_error(f0,f1)
    errs["GN_fallback_trace"]=max(abs(l["error"]-r["error"]) for l,r in zip(tf0,tf1))
    validate_trace(tf1,cf1,target,f1,stream,idx)
    if cf1["exact_grad"]!=1:
        raise AssertionError("GN full fallback exact-gradient not exercised")
    # Check GN Jacobian and fallback gradient at *warm start and endpoint*.
    derivatives={
      "warm_start":derivative_check(stream,q1,idx,target),
      "GN_endpoint":derivative_check(stream,b1,idx,target),
      "fallback_endpoint":derivative_check(stream,f1,idx,target)
    }
    if any(v>3e-10 for v in errs.values()):
        raise AssertionError(f"D09 solver output mismatch: {errs}")
    return {"n":n,"class_count":len(idx),
       "maximum_legacy_to_stream_errors":errs,
       "Q_only_cost_legacy":cq0,"Q_only_cost_stream":cq1,
       "Exact_Adam_cost_stream":ca1,
       "GN_normal_cost_legacy":cb0,"GN_normal_cost_stream":cb1,
       "GN_fallback_cost_legacy":cf0,"GN_fallback_cost_stream":cf1,
       "GN_normal":validate_trace(tb1,cb1,target,b1,stream,idx),
       "GN_fallback":validate_trace(tf1,cf1,target,f1,stream,idx),
       "derivatives":derivatives}


def cost_one_size(x,y,spec,n,reference_seed,teacher_seed):
    if n==150:
        rng=np.random.default_rng(reference_seed)
        selected=np.concatenate([rng.choice(np.flatnonzero(y==c),50,replace=False)
                                for c in range(3)])
        rng.shuffle(selected)
    elif n==300:
        selected=np.arange(len(y))
    else:raise ValueError(n)
    xx,yy=x[selected],y[selected]
    idx=np.flatnonzero(yy==0)
    timings=[]
    (g,t)=timed("StreamGraph_construct",lambda:StreamGraph(xx,yy,spec["gamma"],None,
                                      beta=.4,lam=.8,horizon=12))
    timings.append(t)
    teacher=np.random.default_rng(teacher_seed).normal(0,1.7,size=(6,5))
    teacher-=teacher.mean(axis=-1,keepdims=True)
    with torch.no_grad():target=g.center_train(torch.tensor(teacher,dtype=DT),idx).detach()
    # Clear witness before solver entry. Fitting sees target but not generator logits.
    witness_hash=hashlib.sha256(teacher.tobytes()).hexdigest()
    del teacher
    z0=torch.zeros((6,5),dtype=DT)
    (start,t)=timed("one_exact_center_forward",lambda:exact_error(g,z0,idx,target));timings.append(t)
    ((zq,sec_q,cq),t)=timed("D09_Q_only_2_gradient_steps",lambda:warm_q(g,idx,target,2,.12));timings.append(t)
    ((za,sec_a,ca,ta),t)=timed("D09_Exact_Adam_2_gradient_steps",lambda:exact_adam(g,idx,target,2,.12,1e-12));timings.append(t)
    ((zb,sec_b,cb,tb),t)=timed("D09_GN_2_iterations_from_Q",lambda:damped_gauss_newton(g,idx,target,zq,2,1e-12));timings.append(t)
    gn_norm=validate_trace(tb,cb,target,zb,g,idx)
    ((zf,sec_f,cf,tf),t)=timed("D09_GN_forced_fallback_1_iteration",lambda:damped_gauss_newton(g,idx,target,zq,1,1e-12,max_backtrack=0));timings.append(t)
    gn_fallback=validate_trace(tf,cf,target,zf,g,idx)
    if cf["exact_grad"]!=1:raise AssertionError("GN fallback exact gradient not exercised at n="+str(n))
    # One finite-difference check each for full-center J and true-JSD gradient.
    (d_check,t)=timed("GN_Jacobian_and_fallback_JSD_gradient_finite_diff",
                       lambda:derivative_check(g,zq,idx,target,coords=((5,0),)))
    timings.append(t)
    measured={"size":n,"class_counts":np.bincount(yy,minlength=3).tolist(),
      "sampled_reference_idx_sha256":hashlib.sha256(np.ascontiguousarray(selected).tobytes()).hexdigest(),
      "teacher_policy_sha256_only":witness_hash,
      "teacher_not_passed_to_solver":True,
      "initial_exact_JS":start,"Q_only_exact_JS":exact_error(g,zq,idx,target),
      "Exact_Adam_exact_JS":exact_error(g,za,idx,target),
      "GN_exact_JS":exact_error(g,zb,idx,target),
      "GN_fallback_exact_JS":exact_error(g,zf,idx,target),
      "costs":{"Q_only":cq,"Exact_Adam":ca,"GN":cb,"GN_fallback":cf},
      "solver_trace":{"Exact_Adam":ta,"GN":gn_norm,"GN_fallback":gn_fallback},
      "derivative_finite_difference":d_check,
      "timings":timings,"peak_process_rss_mib_after":peak_mb()}
    del g
    gc.collect()
    return measured


def measure():
    torch.set_num_threads(1)
    if OUT.exists():raise FileExistsError("Cannot overwrite saved preflight evidence")
    spec=load_config(ROOT/"code/data_generator/config.json")
    if spec["seed"]!=20261102 or spec["sizes"]["train"]!=300:
        raise AssertionError("Generator drift")
    t0=time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        data=generate(spec,temp,finite_q=False)
    x,y=data["train"][0],data["train"][2]
    tiny,t=timed("tiny_solver_path_equivalence",lambda:tiny_equivalence(x,y,spec["gamma"]))
    rows=[]
    for n in (150,300):
        rows.append(cost_one_size(x,y,spec,n,20261117,20261118))
    record={"phase":"BOUNDED_STREAMGRAPH_COMPATIBILITY_AND_REAL_COST_ONLY",
       "base_sha":BASE_SHA,
       "source_stream_graph":"code/experiments/archive/diag_max_full300/stream_graph.py",
       "source_d09":["code/experiments/archive/multistep_policy_target_match/run.py",
                      "code/experiments/archive/multistep_policy_target_match/bootstrap_inverse.py"],
       "config":{"H":12,"beta":.4,"lambda":.8,"tol_future":1e-8,
          "tiny_n":36,"n_reference_sizes":[150,300],"only_class_for_cost":0,
          "solver_caps":{"q_grad":2,"exact_adam_grad":2,"GN_jacobian":2,
                          "GN_forced_fallback":1},
          "no_monte_carlo":True,"no_main_runs":True,"test_used":False,
          "same_data_generator_train_seed":20261102,"reference_seed":20261117,
          "teacher_seed":20261118},
       "source_train_class_counts":np.bincount(y,minlength=3).tolist(),
       "tiny":tiny,"tiny_timing":t,"measurements":rows,
       "environment":{"wall_total_seconds":time.perf_counter()-t0,
          "rss_peak_mib":peak_mb(),"rss_final_mib":rss_mb(),
          "cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),
          "python":sys.version.split()[0],"torch":torch.__version__,
          "numpy":np.__version__,"platform":platform.platform()},
       "no_optimization_success_or_accuracy_claim":True}
    OUT.parent.mkdir(exist_ok=True,parents=True)
    OUT.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print("INV01_STREAM_PASS",json.dumps({"sizes":[{"n":z["size"],
          "times":{t["operation"]:t["wall_s"] for t in z["timings"]},
          "peak_mib":z["peak_process_rss_mib_after"],
          "GN_cost":z["costs"]["GN"],"fallback_cost":z["costs"]["GN_fallback"]}
       for z in rows],"tiny_max_diff":max(tiny["maximum_legacy_to_stream_errors"].values()),
       "total_seconds":record["environment"]["wall_total_seconds"]}))


def verify():
    a=json.loads(OUT.read_text())
    if a["base_sha"]!=BASE_SHA or not a["config"]["no_monte_carlo"] or not a["config"]["no_main_runs"]:
        raise AssertionError("Experiment scope drift")
    if [t["size"] for t in a["measurements"]]!=[150,300]:
        raise AssertionError("Size coverage")
    if max(a["tiny"]["maximum_legacy_to_stream_errors"].values())>=3e-10:
        raise AssertionError("Original D09 solver compatibility")
    for r in a["measurements"]:
        if r["costs"]["GN_fallback"]["exact_grad"]!=1:
            raise AssertionError("Fallback derivative not validated")
        if r["derivative_finite_difference"]["max_jacobian_fd_error"]>3e-6:
            raise AssertionError("GN Jacobian mismatch")
        if r["derivative_finite_difference"]["max_exact_jsd_gradient_fd_error"]>3e-6:
            raise AssertionError("GN fallback merit gradient mismatch")
    print("INV01_STREAM_VERIFY_PASS",a["environment"]["rss_peak_mib"])


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--measure",action="store_true")
    g.add_argument("--verify",action="store_true")
    args=ap.parse_args()
    if args.measure:measure()
    else:verify()
