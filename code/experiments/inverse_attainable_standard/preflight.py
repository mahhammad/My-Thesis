#!/usr/bin/env python3
"""CH4-EXP-INV-01 phase 1 only: bounded CPU/RSS calibration and correctness checks.

Hard stops: only one 150-sample balanced graph, one class for any solver step,
<=10 Adam steps per capped pilot, <=2 GN iterations. NO full experimental
matrix, no Test access, no Monte Carlo, no modification to historical outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import gc
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(ROOT/"code/experiments/multistep_policy_target_match"))
from generate import generate,load_config
from run import ExactGraph,jsd,DT
from bootstrap_inverse import warm_q,exact_error,damped_gauss_newton

DATA_SEED=20261102
REFERENCE_SEED=20261117
TEACHER_SEED=20261118
INIT_SEED=20261119
BETA,LAMBDA,H=.4,.8,12
N=150
TOL=1e-8
LR=.12
OUT=HERE/"results"/"preflight_measurements.json"


def rss_kib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS unavailable")


def maximum_rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def timed(label, action):
    gc.collect()
    before=rss_kib()
    start=time.perf_counter()
    result=action()
    elapsed=time.perf_counter()-start
    return result,{"stage":label,"wall_seconds":float(elapsed),
                   "rss_before_kib":before,"rss_after_kib":rss_kib(),
                   "process_peak_rss_kib_after_stage":maximum_rss_kib()}


def exact_adam_probe(g,idx,target,start_logits,limit):
    """Only cap sensitivity, no claim of converged inversion."""
    z=torch.nn.Parameter(start_logits.clone().detach())
    optimizer=torch.optim.Adam([z],lr=LR)
    history=[]
    with torch.no_grad():
        initial=float(jsd(g.center_train(z,idx),target).item())
    history.append({"step":0,"exact_JSD":initial})
    for it in range(1,limit+1):
        if history[-1]["exact_JSD"]<=TOL:break
        optimizer.zero_grad(set_to_none=True)
        objective=jsd(g.center_train(z,idx),target)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            err=float(jsd(g.center_train(z,idx),target).item())
        history.append({"step":it,"exact_JSD":err})
    return history


def run():
    torch.set_num_threads(1)
    torch.manual_seed(INIT_SEED)
    np.random.seed(54321)
    if OUT.exists():
        raise FileExistsError("Existing preflight measurements: do not overwrite; use --verify")
    timings=[]
    src_config=ROOT/"code/data_generator/config.json"
    spec=load_config(src_config)
    if spec["seed"]!=DATA_SEED or spec["split_sizes"]["train"]!=300:
        raise AssertionError("Generator source seed/size changed")
    started=time.perf_counter()
    def make_graph():
        with tempfile.TemporaryDirectory() as tmp:
            data=generate(spec,tmp,finite_q=False)
        x_all,y_all=data["train"][0],data["train"][2]
        counts=np.bincount(y_all,minlength=3).tolist()
        if len(y_all)!=300 or min(counts)<50:
            raise AssertionError("Cannot choose 50 per class in source Train")
        rng=np.random.default_rng(REFERENCE_SEED)
        ref=np.concatenate([rng.choice(np.flatnonzero(y_all==c),50,replace=False)
                            for c in range(3)])
        rng.shuffle(ref)
        x,y=x_all[ref],y_all[ref]
        graph=ExactGraph(x,y,spec["gamma"],np.arange(N),data["validation"][0],
                         beta=BETA,lam=LAMBDA,horizon=H)
        return graph,y,ref,counts
    (g,y,reference,source_counts),timing=timed("generate_train_and_init_exact_graph_n150",make_graph)
    timings.append(timing)
    if np.bincount(y,minlength=3).tolist()!=[50,50,50]:
        raise AssertionError("Not 150 balanced")
    if len(set(reference.tolist()))!=N:
        raise AssertionError("Repeated reference index")
    if not bool((g.coeff>=0).all()) or abs(float(g.coeff.sum())-1)>1e-13:
        raise AssertionError("Invalid capped-horizon coefficients")
    rng=np.random.default_rng(TEACHER_SEED)
    teacher=rng.normal(0,1.7,size=(3,g.d+1,g.d))
    teacher-=teacher.mean(axis=-1,keepdims=True)
    teacher_hash=hashlib.sha256(np.ascontiguousarray(teacher).tobytes()).hexdigest()
    targets=[]
    for c in range(3):
        idx=np.flatnonzero(y==c)
        with torch.no_grad():
            t=g.center_train(torch.as_tensor(teacher[c],dtype=DT),idx)
        p=t.numpy()
        if not np.all(np.isfinite(p)) or np.any(p<0) or abs(p.sum()-1)>1e-12:
            raise AssertionError("Teacher center not a probability distribution")
        if float(jsd(t,torch.as_tensor(p,dtype=DT)))>1e-12:
            raise AssertionError("Known-feasible teacher fails JSD witness")
        targets.append(p.tolist())
    del teacher  # Teacher does not enter solver initializations or objectives.
    idx=np.flatnonzero(y==0)
    target=torch.as_tensor(targets[0],dtype=DT)
    z0=torch.zeros((g.d+1,g.d),dtype=DT)
    rng_init=np.random.default_rng(INIT_SEED)
    zrandom=torch.as_tensor(rng_init.normal(0,0.1,size=z0.shape),dtype=DT)
    (check0,timing)=timed("exact_forward_zero_logits_class0",
                    lambda: exact_error(g,z0,idx,target))
    timings.append(timing)
    (qresult,timing)=timed("Q_only_three_gradient_steps_class0",
                    lambda: warm_q(g,idx,target,3,LR))
    zq,_,qc=qresult
    timings.append(timing)
    (qerr,timing)=timed("verify_Q_warmup_on_exact_multi_step_center",
                    lambda: exact_error(g,zq,idx,target))
    timings.append(timing)
    pilot={}
    for initial,z in (("zero",z0),("independent_random",zrandom)):
        limits=(2,5,10) if initial=="zero" else (5,)
        for limit in limits:
            (trace,timing)=timed(f"exact_adam_{initial}_cap_{limit}_class0",
                                  lambda z=z,limit=limit:exact_adam_probe(g,idx,target,z,limit))
            timings.append(timing)
            pilot[f"{initial}_cap_{limit}"]={
                "initial_exact_JSD":trace[0]["exact_JSD"],
                "final_exact_JSD":trace[-1]["exact_JSD"],
                "executed_steps":trace[-1]["step"],
                "budget_cap":limit,"reached_threshold":trace[-1]["exact_JSD"]<=TOL,
                "trace":trace,"wall_seconds":timing["wall_seconds"]
            }
    def jacobian():
        z=z0.detach().clone().requires_grad_(True)
        center=g.center_train(z,idx)[:-1]
        rows=[torch.autograd.grad(center[c],z,retain_graph=(c<g.k-2))[0]
              for c in range(g.k-1)]
        result=torch.stack(rows)
        if result.shape!=(2,6,5) or not bool(torch.isfinite(result).all()):
            raise AssertionError("Invalid exact center Jacobian")
        return result.detach()
    (J,timing)=timed("one_exact_center_jacobian_zero_logits_class0",jacobian)
    timings.append(timing)
    (gn,timing)=timed("GN_two_corrections_after_three_Q_steps_class0",
                      lambda:damped_gauss_newton(g,idx,target,zq,2,TOL))
    timings.append(timing)
    zgn,gn_elapsed,gncounts,gntrace=gn
    gnerr=float(jsd(g.center_train(zgn,idx),target).item())
    if abs(gnerr-gntrace[-1]["error"])>1e-10:
        raise AssertionError("Final GN recorded JSD mismatch")
    if any(b["error"]>a["error"]+1e-12 for a,b in zip(gntrace,gntrace[1:])):
        raise AssertionError("GN correction increased exact JSD")
    # Independent scale/correctness sanity check: no full optimisation.
    results={
        "schema":"ch4-exp-inv01-preflight-v1",
        "phase":"PREFLIGHT_ONLY_NO_MAIN_RUN",
        "source_chapter4_sha":"de02fdda91e23db177de4429b3e1a9693c046d83",
        "configuration":{"beta":BETA,"lambda":LAMBDA,"H":H,"float":"float64",
          "nref":N,"ref_count_by_class":[50,50,50],
          "full_train_class_counts":source_counts,"tolerance_exact_JSD":TOL,
          "lr":LR,"max_pilot_adam_steps_per_run":10,
          "max_GN_iterations":2,"max_Q_gradient_steps":3,
          "no_Monte_Carlo":True,"historical_Test_accessed":False,
          "no_full_training":True},
        "independent_seeds":{"data":DATA_SEED,"reference_selection":REFERENCE_SEED,
          "teacher":TEACHER_SEED,"solver_initialization":INIT_SEED},
        "reference_ids_sha256":hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest(),
        "teacher_policy_sha256":teacher_hash,
        "teacher_policy_given_to_solver":False,
        "teacher_exact_centers":targets,
        "source_config_sha256":hashlib.sha256(src_config.read_bytes()).hexdigest(),
        "witness_checks":{"valid_simplex":True,"JSD_self_at_most_1e-12":True},
        "reference_class_0":{"zero_exact_error":check0,
          "Q_three_steps_exact_error":qerr,"q_gradient_count":qc["q_grad"],
          "GN_after_Q_three_steps":{"exact_error":gnerr,
            "GN_iterations":len(gntrace)-1,"cost":gncounts,"trace":gntrace}},
        "exact_adam_pilot":pilot,
        "timings":timings,
        "resource":{"wall_total_preflight_seconds":time.perf_counter()-started,
          "peak_process_rss_MiB":maximum_rss_kib()/1024.,
          "final_process_rss_MiB":rss_kib()/1024.,
          "platform":platform.platform(),"python":sys.version.split()[0],
          "torch":torch.__version__,"numpy":np.__version__,
          "cpu_count":os.cpu_count(),"pytorch_threads":torch.get_num_threads(),
          "memory_measure":"Linux /proc/self/status VmRSS and ru_maxrss process peak RSS; "
            "includes Python+Torch+graph+autograd, not isolated tensors."},
        "observed_only_not_extrapolation":True,
        "d15_reused_as_independent_complement":True,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                   encoding="utf8")
    print("INV01_PREFLIGHT_PASS",json.dumps({
        "pilot":{k:{"initial":x["initial_exact_JSD"],"final":x["final_exact_JSD"],
          "steps":x["executed_steps"],"seconds":x["wall_seconds"]} for k,x in pilot.items()},
        "measurements":[{k:v for k,v in x.items() if k in ("stage","wall_seconds",
             "process_peak_rss_kib_after_stage")} for x in timings],
        "peak_rss_MiB":results["resource"]["peak_process_rss_MiB"],
        "GN_error":gnerr,
        "no_main_run":True
    },ensure_ascii=False))


def verify():
    if not OUT.is_file():
        raise FileNotFoundError(OUT)
    d=json.loads(OUT.read_text(encoding="utf8"))
    c=d["configuration"]
    if c["nref"]!=150 or c["ref_count_by_class"]!=[50,50,50]:
        raise AssertionError("Wrong preflight reference size")
    if (c["max_pilot_adam_steps_per_run"]>10 or c["max_GN_iterations"]>2 or
        c["max_Q_gradient_steps"]>3 or not c["no_Monte_Carlo"] or
        not c["no_full_training"] or c["historical_Test_accessed"]):
        raise AssertionError("Preflight scope violated")
    if d["teacher_policy_given_to_solver"]:
        raise AssertionError("Leakage from teacher")
    for item in d["exact_adam_pilot"].values():
        if item["executed_steps"]>item["budget_cap"]:
            raise AssertionError("Adam exceeded pilot budget")
    if not d["witness_checks"]["JSD_self_at_most_1e-12"]:
        raise AssertionError("Witness validation missing")
    if d["resource"]["peak_process_rss_MiB"]<=0:
        raise AssertionError("Memory measurement absent")
    print("INV01_VERIFY_PASS",d["resource"]["peak_process_rss_MiB"])


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--measure",action="store_true")
    group.add_argument("--verify",action="store_true")
    opts=ap.parse_args()
    if opts.measure:run()
    else:verify()
