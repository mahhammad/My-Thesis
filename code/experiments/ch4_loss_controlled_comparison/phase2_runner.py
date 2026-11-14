"""E-CH4-13 Phase 2 full300/25-step paired runner.

DRY RUN by default. NOT authorized for fitting until separate preliminary technical QA,
manager's written execution authorization, verified gate artifact, passing current-SHA CI and
single-use token are present. No automatic training on Git push.
All 12 cells are one immutable fixed batch; failures charge budget and STOP,
not silently retry. Uses Train/Validation adapter only, never Test.
"""
from __future__ import annotations
import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from generate import load_config
from dataset_adapter import dataset_train_val
from engine import DT,ExactMaskedWalk,balanced
from replay_v1 import objective_loss
from train_missing_objectives import train_loss
from snapshot_pipeline import save_snapshot,load_snapshot,array_fingerprint
from phase2_report import report_from_frozen
from phase2_protocol import (
    SEEDS,FAMILIES,LOSSES,N_TRAIN,N_VALIDATION,N_UPDATES,K,D,
    BETA,LAMBDA,H,TAU,LR,MAX_FITS,grid,validate_grid,
    plan_manifest,reference_ids,expected_data_gate,evidence_gate,
    atomic_new_json
)
from phase2_frozen_data import load as load_frozen, INPUTS as FROZEN_INPUTS
from phase2_github_gate import api as github_api, validate_online as verify_github_gate, REPO as GITHUB_REPO

RESULT_ROOT=HERE/"results/phase2_300_25"
CFG=ROOT/"code/data_generator/config.json"

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def numeric_nested(x):
    if x is None: return False
    if isinstance(x,dict): return all(numeric_nested(v) for v in x.values())
    try:return bool(np.isfinite(np.asarray(x,dtype=float)).all())
    except (ValueError,TypeError):return False

def canonical_loss(graph,z,family,objective):
    if objective==LOSSES[0]:
        return objective_loss(graph,graph.train_outputs(z,family),"margin_JSD",TAU)
    if objective==LOSSES[1]:
        return train_loss(graph,z,family,"softplus_JS_tau005")
    raise ValueError("Objective not approved")

def gradients_at_policy(graph,z,family):
    """Same-policy paired gradient cosine; no optimizer step."""
    gradients=[]
    values=[]
    for obj in LOSSES:
        v=canonical_loss(graph,z,family,obj)
        g=torch.autograd.grad(v,z,retain_graph=False,create_graph=False)[0]
        if not bool(torch.isfinite(v)) or not bool(torch.isfinite(g).all()):
            raise FloatingPointError("Cross-objective gradient nonfinite")
        gradients.append(g.detach().flatten());values.append(float(v.detach()))
    norm=[torch.linalg.vector_norm(g) for g in gradients]
    cos=(float(torch.dot(gradients[0],gradients[1])/(norm[0]*norm[1]))
         if bool((norm[0]>0)&(norm[1]>0)) else None)
    return {"sigmoid_gradient_l2":float(norm[0]),"softplus_gradient_l2":float(norm[1]),
            "cosine":cos,"loss_values":values}

def make_coverage(report,contract):
    coverage={}
    for group in contract["metric_groups"]:
        gname=group["group"]
        if gname=="experiment_specific": continue
        for split in (("train",) if group.get("center_only") else ("train","validation")):
            section=report[split]
            for name in group.get("ids",[]):
                value=(section.get("center_geometry",{}).get(name,section.get(name))
                       if gname=="center_geometry" else section.get(name))
                measured=numeric_nested(value)
                coverage[f"{split}.{name}"]={
                  "status":"measured" if measured else (
                      "undefined_or_infinite" if name in
                      ("R_KL","NLL_KL_simple","NLL_KL_normalized_historical")
                      else "not_measured"),
                  "reason":"metric in canonical report" if measured else
                      "canonical reporter value unavailable/nonfinite; never filled with 0"}
    coverage["test"]={"status":"not_applicable","reason":"forbidden by Issue #81"}
    return coverage

def fingerprint_match(info,expected):
    for key,snap in (("train_x_sha256","train_reference_x_sha256"),
                     ("train_y_sha256","train_reference_y_sha256"),
                     ("validation_x_sha256","validation_x_sha256"),
                     ("config_sha256","config_sha256")):
        if info[snap]!=expected[key]:raise RuntimeError("CRITICAL_HASH_MISMATCH "+key)

def fit_once(graph,family,objective,checkpoint_dir=None):
    """Exactly 25 optimizer updates; terminal step 25 is measured, not selected."""
    shape=((D+1,D) if family=="shared" else (K,D+1,D))
    z=torch.nn.Parameter(torch.zeros(shape,dtype=DT))
    opt=torch.optim.Adam([z],lr=LR)
    trace=[]
    initial_zero=True
    started=time.perf_counter()
    for step in range(N_UPDATES+1):
        opt.zero_grad(set_to_none=True)
        v=canonical_loss(graph,z,family,objective)
        if not bool(torch.isfinite(v)): raise FloatingPointError("Nonfinite objective")
        v.backward()
        if z.grad is None or not bool(torch.isfinite(z.grad).all()):
            raise FloatingPointError("Nonfinite loss gradient")
        with torch.no_grad():
            score=graph.train_outputs(z,family)
            train_pred=score.scores.argmax(1)
            train_ba=float(balanced((train_pred==graph.labels).to(DT),graph.labels,K))
            row={"update_completed":step,"objective_loss":float(v.detach()),
                 "gradient_l2":float(torch.linalg.vector_norm(z.grad.detach())),
                 "gradient_linf":float(z.grad.abs().max()),
                 "train_ba":train_ba,
                 "train_mean_margin":float(score.margin.mean()),
                 "elapsed_s":time.perf_counter()-started}
        if step in (0,N_UPDATES):
            row["cross_objective_gradient"]=gradients_at_policy(graph,z,family)
        if step<N_UPDATES:
            old=z.detach().clone()
            opt.step()
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError("Nonfinite policy after Adam step")
            row["adam_update_l2"]=float(torch.linalg.vector_norm((z-old).detach()))
        else:
            row["adam_update_l2"]=None
        trace.append(row)
        if checkpoint_dir is not None:
            # Each completed optimizer update is independently recoverable.
            # No resume/retraining permitted; this is forensic evidence only.
            atomic_new_json(Path(checkpoint_dir)/
                f"step_{step:02d}.json",
                {"update_completed":step,"raw_logits":z.detach().tolist(),
                 "trace_entry":row,"objective":objective,"family":family,
                 "test_accessed":False})
    if len(trace)!=26 or trace[-1]["update_completed"]!=25:
        raise AssertionError("Exactly 25 updates required")
    return z.detach(),trace,time.perf_counter()-started

def execute(out,source_sha,one_time_gate_token,authorization_path=None):
    """Protected expensive code; never reached by default dry-run or CI tests."""
    if not source_sha or len(source_sha)!=40:
        raise ValueError("Canonical 40-digit Git commit SHA required")
    if os.environ.get("E_CH4_13_PHASE2_OPT_IN")!="I_ACCEPT_12_FIT_CAP":
        raise PermissionError("No explicit 12-fit opt-in")
    if os.environ.get("GITHUB_ACTIONS")!="true":
        raise PermissionError("TRAINING_ONLY_IN_MANUAL_GITHUB_ACTIONS")
    if os.environ.get("GITHUB_REF_NAME")!="experiment/ch4-loss-controlled-comparison-01":
        raise PermissionError("WRONG_BRANCH")
    if os.environ.get("GITHUB_WORKFLOW")!="E-CH4-13 phase2 authorized 12-fit batch":
        raise PermissionError("WRONG_WORKFLOW")
    if authorization_path is None:
        raise PermissionError("Missing ephemeral online-verified authorization path")
    auth=evidence_gate(execute=True,approval_file=Path(authorization_path),
                       cli_token=one_time_gate_token)
    token=os.environ.get("GH_TOKEN")
    if not token:
        raise PermissionError("Missing GitHub online-read credential")
    online=verify_github_gate(source_sha,os.environ["GITHUB_REF_NAME"],token)
    if (online["manager_comment"]!=auth["manager_authorization_url"] or
        online["auditor_comment"]!=auth["technical_audit_url"] or
        online["ci_url"]!=auth["ci_url"]):
        raise PermissionError("Remote approval no longer agrees with ephemeral gate")
    ref=auth.get("reservation_ref","")
    expected_ref="refs/tags/e-ch4-13-phase2-budget-spent-"+source_sha
    if ref!=expected_ref:
        raise PermissionError("Missing once-only budget reservation")
    ref_response=github_api("GET","/repos/"+GITHUB_REPO+
        "/git/ref/tags/e-ch4-13-phase2-budget-spent-"+source_sha,token)
    if ref_response.get("ref")!=ref or ref_response.get("object",{}).get("sha")!=source_sha:
        raise PermissionError("Remote budget marker mismatch")
    if auth.get("github_run_id")!=os.environ.get("GITHUB_RUN_ID"):
        raise PermissionError("Execution gate bound to another workflow run")
    if auth["source_sha"]!=source_sha:
        raise PermissionError("QA/CI source SHA differs from actual training code commit")
    if os.environ.get("OPENBLAS_CORETYPE")!="generic" or os.environ.get("OPENBLAS_NUM_THREADS")!="1":
        raise ValueError("BLAS fingerprint environment not pinned")
    if np.__version__!="2.4.6" or not torch.__version__.startswith("2.15.0"):
        raise ValueError("Unapproved numerical package versions")
    if Path(out).exists():
        raise FileExistsError("Immutable phase2 result root already exists; NEVER rerun")
    validate_grid(grid())
    torch.set_num_threads(1)
    # Read immutable, audited Train/Validation bytes for each generator:
    # no fresh BLAS-dependent multivariate-normal generation and no Test.
    if not (FROZEN_INPUTS/"manifest.json").is_file():
        raise FileNotFoundError("Frozen Train/Validation inputs not independently prepared")
    frozen_manifest=json.loads((FROZEN_INPUTS/"manifest.json").read_text("utf8"))
    if frozen_manifest.get("test_accessed") is not False:
        raise ValueError("Frozen data Test rule violated")
    data={"schema":"e-ch4-13-phase2-data-contract-v2",
          "frozen_manifest_sha256":sha(FROZEN_INPUTS/"manifest.json"),
          "datasets":{},"test_accessed":False}
    cached={}
    spec=load_config(CFG)
    for seed in SEEDS:
        (x,y),(vx,vy)=load_frozen(seed)
        ids=reference_ids(y)
        rec={"n_train":len(x),"n_validation":len(vx),
             "n_classes":K,"n_features":D,
             "reference_ids":ids.tolist(),
             "config_sha256":frozen_manifest["config_sha256"],
             "train_x_sha256":array_fingerprint(x[ids]),
             "train_y_sha256":array_fingerprint(y[ids].astype(np.int64)),
             "validation_x_sha256":array_fingerprint(vx),
             "test_accessed":False}
        expected=frozen_manifest["records"][str(seed)]
        for col in ("train_x_sha256","train_y_sha256","validation_x_sha256"):
            if rec[col]!=expected[col]:
                raise ValueError("Frozen data fingerprint drift: "+col)
        data["datasets"][str(seed)]=rec
        cached[seed]=(ExactMaskedWalk(x[ids],y[ids],spec["gamma"],
                      beta=BETA,lam=LAMBDA,horizon=H,query_x=vx),vx,ids,rec)
    expected_data_gate(data["datasets"])
    # Output is created only after all pre-training contract checks.
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    atomic_new_json(out/"data_fingerprints.json",data)
    atomic_new_json(out/"protocol.json",plan_manifest())
    # A sanitized authorization record is archived; NEVER persist gate secret.
    auth_public={k:v for k,v in auth.items() if k!="one_time_gate_token"}
    atomic_new_json(out/"authorization_record.json",auth_public)
    provenance={"schema":"e-ch4-13-phase2-provenance-v1",
       "source_commit":source_sha,"preliminary_technical_audit_url":auth["technical_audit_url"],
       "manager_execution_authorization_url":auth["manager_authorization_url"],
       "same_commit_ci":auth["ci_url"],"numpy":np.__version__,
       "torch":torch.__version__,"python":platform.python_version(),
       "blas_coretype":"generic","blas_threads":1,"test_accessed":False}
    atomic_new_json(out/"provenance.json",provenance)
    contract=json.loads((ROOT/"code/experiments/COMMON_METRICS_CONTRACT.json").read_text("utf8"))
    ledger={"schema":"e-ch4-13-phase2-budget-v1","cap":MAX_FITS,
            "attempt_count":0,"status":"STARTED","attempts":[]}
    summary=[]
    for job in grid():
        # Write a charge marker in an immutable file before calling optimizer.
        job_path=out/job.key()
        atomic_new_json(out/"attempt_charges"/(job.key()+".json"),
                        {"run":job.to_dict(),"charged":True,"status":"attempted",
                         "source_sha":source_sha})
        ledger["attempts"].append({"key":job.key(),"status":"charged"})
        ledger["attempt_count"]+=1
        # Durable, append-only ledger checkpoint BEFORE the optimizer. Uploaded
        # even on failure, alongside the separately immutable charge marker.
        atomic_new_json(out/"budget_checkpoints"/
                        f"attempt_{ledger['attempt_count']:02d}.json",ledger)
        graph,vx,ids,rec=cached[job.seed]
        try:
            z,trace,elapsed=fit_once(graph,job.family,job.loss,
                checkpoint_dir=out/"optimizer_checkpoints"/job.key())
            # Save finished trace immediately. Reporter or snapshot failures
            # must not erase training effort or its computed gradient trajectory.
            atomic_new_json(job_path/"trace.json",trace)
        except BaseException as exc:
            atomic_new_json(out/"failures"/
                (job.key()+".json"),
                {"run_key":job.key(),"attempt_charged":True,
                 "exception_type":type(exc).__name__,
                 "exception_message":str(exc)[:2000],
                 "test_accessed":False})
            raise
        try:
            save_snapshot(job_path,graph,z,job.family,reference_ids=ids,
                 config_path=CFG,config_reference="code/data_generator/config.json",
                 data_seed=job.seed,reference_seed=20261107,objective=job.loss,
                 tau=TAU,beta=BETA,lam=LAMBDA,horizon=H,validation_x=vx,
                 code_commit=source_sha,experiment_metadata={
                     "phase":"E-CH4-13-phase2","family":job.family,"updates":N_UPDATES,
                     "optimizer":"Adam","lr":LR,"zero_init":True,"paired":True,
                     "reference_rule":"all_300_ordered_train","test_accessed":False})
            model,arrays=load_snapshot(job_path)
            fingerprint_match(model["data"],rec)
            report=report_from_frozen(job_path,job.seed)
            if report["validation"].get("status")!="ok":
                raise ValueError("Validation unavailable, STOP")
            if report["source"]["historical_test_accessed"]:
                raise AssertionError("Test accessed")
            atomic_new_json(job_path/"metrics_full.json",report)
            atomic_new_json(job_path/"coverage.json",make_coverage(report,contract))
            values={"run_key":job.key(),**job.to_dict(),
                "train_ba":report["train"]["balanced_accuracy"],
                "validation_ba":report["validation"]["balanced_accuracy"],
                "I":report["train"]["I"],"R_KL":report["train"]["R_KL"],
                "R_JS":report["train"]["R_JS"],"R_sqrt_JS":report["train"]["R_sqrt_JS"],
                "elapsed_s":elapsed,"gradient_l2_init":trace[0]["gradient_l2"],
                "gradient_l2_final":trace[-1]["gradient_l2"],
                "margin_validation_median":report["validation"]["margin_quantiles"]["0.5"],
                "trace_sha256":sha(job_path/"trace.json"),
                "model_sha256":sha(job_path/"model.json"),
                "distributions_sha256":sha(job_path/"distributions.npz"),
                "metrics_full_sha256":sha(job_path/"metrics_full.json"),
                "coverage_sha256":sha(job_path/"coverage.json")}
            summary.append(values)
            # A STOP leaves attempt charges visible but does not auto-retry or skip.
            atomic_new_json(job_path/"run_evidence.json",values)
        except BaseException as exc:
            atomic_new_json(out/"failures"/
                (job.key()+"_report.json"),
                {"run_key":job.key(),"attempt_charged":True,
                 "stage":"snapshot_or_report","exception_type":type(exc).__name__,
                 "exception_message":str(exc)[:2000],"test_accessed":False})
            raise
        atomic_new_json(out/"success_checkpoints"/
                        f"fit_{ledger['attempt_count']:02d}.json",values)
    if len(summary)!=MAX_FITS or ledger["attempt_count"]!=MAX_FITS:
        raise RuntimeError("Incomplete grid STOP")
    ledger["status"]="COMPLETE_UNAUDITED"
    atomic_new_json(out/"fit_budget_ledger.json",ledger)
    with (out/"metrics_summary.csv").open("x",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    result={"schema":"e-ch4-13-phase2-raw-manifest-v1",
       "source_commit":source_sha,"status":"PHASE2_UNAUDITED","fit_count":MAX_FITS,
       "updates_per_fit":N_UPDATES,"test_accessed":False,
       "data_fingerprints_sha256":sha(out/"data_fingerprints.json"),
       "provenance_sha256":sha(out/"provenance.json"),"runs":summary}
    atomic_new_json(out/"manifest.json",result)
    print(json.dumps({"status":"PHASE2_UNAUDITED","fit_count":len(summary),
        "manifest_sha256":sha(out/"manifest.json"),"test_accessed":False}))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path,default=RESULT_ROOT)
    parser.add_argument("--source-sha",default="")
    parser.add_argument("--one-time-token",default="")
    parser.add_argument("--authorization-path",type=Path,default=None)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN_NO_TRAINING","plan":plan_manifest()},
                         indent=2,ensure_ascii=False))
        return
    execute(args.output,args.source_sha,args.one_time_token,args.authorization_path)

if __name__=="__main__":main()
