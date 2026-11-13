#!/usr/bin/env python3
"""R05 §4.10: one SHA-attested bounded solve, new runner, old V2 untouched."""
from __future__ import annotations
import argparse
import hashlib
import json
import importlib.metadata
import os
import platform
import signal
import stat
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from data_contract import (ROOT,EXPERIMENT,SOURCE_PATHS,Deny,read_locked,owner_root,
                           decoded_block,attested_sources,sha,gitblob,require)

TOL=1e-8
SOFT_WALL=840.0
HARD_WALL=900.0
SOFT_RSS=1430.0
HARD_RSS=1536.0
CAPS={"Q-only":100,"Exact-Adam":150,"GN":30,"GN_restart":30}
ORDER=["Q-only","Exact-Adam","Q+Bootstrap-GN"]

def canonical(o):
    return (json.dumps(o,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"),allow_nan=False)+"\n").encode()

def exclusive_record(directory:Path,name:str,record:dict):
    require(directory.is_dir() and not directory.is_symlink(),"AUDIT_DIR_UNSAFE")
    require(directory.stat().st_uid==os.getuid() and
            directory.stat().st_mode & 0o077 == 0,"AUDIT_PERMISSIONS")
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0)
    dfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|getattr(os,"O_NOFOLLOW",0))
    try:
        fd=os.open(name,flags,0o600,dir_fd=dfd)
        try:
            msg=canonical(record)
            os.write(fd,msg)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(dfd)
        return sha(msg)
    finally:
        os.close(dfd)

def mk_audit_root():
    root=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"r05-sec410-append-only-audit"
    root.mkdir(mode=0o700,parents=False,exist_ok=True)
    require(root.stat().st_uid==os.getuid() and root.stat().st_mode & 0o077 == 0,
            "AUDIT_ROOT_PERMISSIONS")
    require(not root.is_symlink() and not root.resolve().is_relative_to(ROOT.resolve()),
            "AUDIT_NOT_EXTERNAL")
    return root

def create_attempt_folder(row,attempt):
    p=EXPERIMENT/"results/raw"/row["block_id"]/row["run_id"]/attempt
    p.mkdir(parents=True,exist_ok=False)
    return p

def write_unique(path,obj):
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0)
    fd=os.open(path,flags,0o600)
    try:
        raw=canonical(obj)
        os.write(fd,raw)
        os.fsync(fd)
    finally: os.close(fd)
    return sha(raw)

def pinned_module(name,code_bytes,path):
    """Execute previously-attested bytes directly: no import resolver/pyc or reopen."""
    require(name not in sys.modules,"TARGET_MODULE_ALREADY_IMPORTED")
    module=types.ModuleType(name)
    module.__file__=str(path)
    module.__package__=""
    module.__dict__["__builtins__"]=__builtins__
    # Install before exec so 'from generate/run/...' in pinned code reuses it.
    sys.modules[name]=module
    try:
        exec(compile(code_bytes,str(path),"exec"),module.__dict__)
        require(sys.modules.get(name) is module,"MODULE_REPLACED_DURING_LOAD")
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module

def init_modules(captured):
    import numpy as np
    import torch
    # These five names are loaded ONLY via compiled, previously hashed memory.
    names=[("generate","generator"),("run","graph_run"),
           ("stream_graph","streamgraph"),("bootstrap_inverse","bootstrap"),
           ("p2_v2_instrumentation","p2_instrumentation")]
    loaded={}
    for name,key in names:
        loaded[name]=pinned_module(name,captured[key],ROOT/SOURCE_PATHS[key])
    return np,torch,loaded

def check_attested_modules(captured,loaded):
    for name,key in [("generate","generator"),("run","graph_run"),
                     ("stream_graph","streamgraph"),("bootstrap_inverse","bootstrap"),
                     ("p2_v2_instrumentation","p2_instrumentation")]:
        require(sys.modules.get(name) is loaded[name],"AT_USE_MODULE_IDENTITY")
        require(gitblob(captured[key])==gitblob(read_locked(ROOT/SOURCE_PATHS[key])),
                "AT_USE_SOURCE_DRIFT")
    # One final intentional verification before creating the first graph.
    # Worktree must remain isolated; actual executing code is in-memory captures.

def run_solve(args):
    attempt=uuid.uuid4().hex
    now=time.time()
    run_id=f"INV01-{args.block}-n{150 if args.block in ('B03','B05') else 300}-data{20261103 if args.block in ('B03','B04') else 20261104}-C{args.class_index}-{'Z' if args.initialization=='zero' else 'R'}-{'Q' if args.method=='Q-only' else 'A' if args.method=='Exact-Adam' else 'G'}"
    audit_dir=mk_audit_root()
    start_digest=exclusive_record(audit_dir,attempt+".start.json",{
        "phase":"START","attempt_id":attempt,"run_id":run_id,
        "timestamp":now,"code_commit":os.environ.get("GITHUB_SHA"),
        "graph_constructed":0,"optimizer_steps":0})
    resultdir=None
    phase="OWNER_ROOT"
    graph_constructed=0
    status="INVALID_PRECHECK"
    terminal={"status":"INTERRUPTED","phase":"START_ONLY"}
    after_digest=None
    try:
        root,root_digest=owner_root(args.owner_root_comment)
        require(root["runtime"]=={"python":platform.python_version(),
                                    "numpy":importlib.metadata.version("numpy"),
                                    "torch":importlib.metadata.version("torch")},
                "HISTORICAL_RUNTIME_MISMATCH")
        require(root["allow_numeric"] is True,"OWNER_ROOT_NOT_SOLVER_ENABLED")
        require(root["limits"]=={"soft_wall_s":840,"hard_wall_s":900,
                                  "soft_rss_mib":1430,"hard_rss_mib":1536},
                "RESOURCE_LIMITS_NOT_PINNED")
        require(args.block in root["blocks"] and
                args.method in ORDER and args.initialization in ("zero","independent_random"),
                "SCOPE_BLOCK_METHOD")
        phase="PRE_SOLVE_PINNED_INPUT"
        inputs=decoded_block(root,args.block)
        p_rows=inputs["plan_rows"]
        matches=[r for r in p_rows if r["run_id"]==run_id]
        require(len(matches)==1,"RUN_ID_NOT_PLANNED")
        row=matches[0]
        require((row["class_index"],row["initialization"],row["solver"])==
                (args.class_index,args.initialization,args.method),"RUN_DESCRIPTOR_DRIFT")
        captured=attested_sources(root)
        # Duplicate-run prevention: any prior completed result for this run ID is final.
        runfolder=EXPERIMENT/"results/raw"/row["block_id"]/run_id
        if runfolder.exists():
            for completed in runfolder.glob("*/summary.json"):
                obj=json.loads(completed.read_text())
                require(obj.get("status")!="COMPLETED","ALREADY_COMPLETED")
        resultdir=create_attempt_folder(row,attempt)
        write_unique(resultdir/"input_pins.json",{
            "attempt_id":attempt,"run_id":run_id,"root_comment_id":args.owner_root_comment,
            "owner_root_sha256":root_digest,"input_sha256":inputs["input_sha256"],
            "source_sha256":{key:sha(v) for key,v in captured.items()},
            "source_git_blob_sha1":{key:gitblob(v) for key,v in captured.items()},
            "plan_row":row,"resource_limits":root["limits"],"audited_start_sha256":start_digest})
        phase="AT_USE_EXECUTE_PINNED_BYTES"
        np,torch,modules=init_modules(captured)
        require({"python":platform.python_version(),"numpy":np.__version__,
                 "torch":torch.__version__}==root["runtime"],"RUNTIME_IMPORT_VERSION_DRIFT")
        for name in ("X","y","mask","ids","targets"):
            require(inputs[name].flags.writeable is False,"IMMUTABLE_ARRAY_REQUIRED")
        check_attested_modules(captured,modules)
        phase="FIRST_GRAPH"
        StreamGraph=modules["stream_graph"].StreamGraph
        p2=modules["p2_v2_instrumentation"]
        torch.set_num_threads(1)
        data=inputs["X"]
        labels=inputs["y"]
        config=modules["generate"].load_config(ROOT/"code/data_generator/config.json")
        gamma=config["gamma"]
        graph=StreamGraph(data,labels,gamma,validation_x=None,beta=.4,lam=.8,horizon=12)
        graph_constructed=1
        idx=np.flatnonzero(labels==args.class_index)
        require(len(idx)==inputs["manifest"]["class_counts"][args.class_index],
                "GRAPH_CLASS_SET_DRIFT")
        target=torch.as_tensor(inputs["targets"][args.class_index].copy(),dtype=torch.float64)
        rng=None
        initial=None
        if args.initialization=="independent_random":
            rng=np.random.default_rng(row["initialization_seed"])
            initial=torch.as_tensor(rng.normal(0,.1,size=(graph.d+1,graph.d)),dtype=torch.float64)
        # Test actual exact forward on the frozen Train and target before optimization.
        # This is a counted pre-solve check, not a complete solver run.
        p2.START=time.perf_counter()
        p2.DEADLINE=p2.START+SOFT_WALL
        p2.MEM_SOFT_LIMIT_MIB=SOFT_RSS
        p2.MEM_HARD_BOUND_MIB=HARD_RSS
        p2.GLOBAL_CFG["learning_rate"]=0.12
        p2.TOL=TOL
        p2.RESULTS=resultdir
        p2.RAW=resultdir/"trace.json"
        p2.CSV=resultdir/"curve.csv"
        p2.SUMMARY=resultdir/"p2_summary.json"
        p2.MANIFEST=resultdir/"p2_manifest.json"
        p2.state={"started":time.time(),"finished":None,"status":"in_progress",
                  "stop_reason":None,"runs":[],"errors":[],"wall_numerical_seconds":None,
                  "peak_rss_mib":0.0}
        signal.signal(signal.SIGUSR1,p2.signal_stop)
        stop=threading.Event()
        thread=threading.Thread(target=p2.watch_resources,args=(stop,),daemon=True)
        thread.start()
        phase="NUMERICAL"
        try:
            key=f"{row['block_id']}-C{args.class_index}-{'Z' if args.initialization=='zero' else 'R'}"
            if args.method=="Q-only":
                z,record=p2.adam_path(key,"Q-only",graph,idx,target,CAPS["Q-only"],
                                      initial=initial,init_label=args.initialization)
                stage="Q-only"
                restart=False
                cost=record["counts"]
                inclusive_wall=record["elapsed_solver_wall_s"]
            elif args.method=="Exact-Adam":
                z,record=p2.adam_path(key,"Exact-Adam",graph,idx,target,CAPS["Exact-Adam"],
                                      initial=initial,init_label=args.initialization)
                stage="Exact-Adam"
                restart=False
                cost=record["counts"]
                inclusive_wall=record["elapsed_solver_wall_s"]
            else:
                # The historical paired protocol shares Q's work in the same (class,init).
                # Use Q's FINAL logits, not its selected best checkpoint.
                qrow=next(r for r in p_rows if r["class_index"]==args.class_index
                    and r["initialization"]==args.initialization and r["solver"]=="Q-only")
                qdir=EXPERIMENT/"results/raw"/row["block_id"]/qrow["run_id"]
                prior=[]
                if qdir.exists():
                    for f in qdir.glob("*/summary.json"):
                        prior.append(json.loads(f.read_text()))
                prior=[r for r in prior if r.get("status")=="COMPLETED" and
                       r.get("input_sha256")==inputs["input_sha256"] and
                       r.get("source_pin_sha256")==sha(canonical({k:sha(v) for k,v in captured.items()}))]
                require(len(prior)==1,"MISSING_ATTESTED_Q_WARMUP")
                q=prior[0]
                zq=torch.as_tensor(q["final_policy_logits"],dtype=torch.float64)
                require(zq.shape==(graph.d+1,graph.d),"Q_WARM_SHAPE")
                require(abs(p2.exact_center_loss(graph,zq,idx,target)-q["final_exact_center_JSD"])<1e-12,"Q_WARM_EXACT_JSD_MISMATCH")
                _,first=p2.gn_instrumented(key,graph,idx,target,zq,CAPS["GN"],
                              q["exclusive_solver_wall_s"],"warm",q["counts"])
                again=None
                if first["best_JSD"]>TOL:
                    _,again=p2.gn_instrumented(key,graph,idx,target,torch.zeros_like(zq),
                              CAPS["GN_restart"],q["exclusive_solver_wall_s"]+
                              first["elapsed_solver_wall_s"],"neutral_restart",
                              p2.addcost(q["counts"],first["counts"]))
                record=(again if again is not None and again["best_JSD"]<first["best_JSD"] else first)
                stage="GN_restart" if record is again else "GN_first"
                restart=again is not None
                cost=(again["inclusive_cost"] if again is not None
                      else p2.addcost(first["counts"],q["counts"]))
                inclusive_wall=q["exclusive_solver_wall_s"]+first["elapsed_solver_wall_s"]+(again["elapsed_solver_wall_s"] if again else 0)
            selected=torch.as_tensor(record["selected_policy_logits"],dtype=torch.float64)
            selected_exact=p2.exact_center_loss(graph,selected,idx,target)
            require(abs(selected_exact-record["selected_policy_exact_JSD"])<=1e-12,
                    "SELECTED_POLICY_JSD_DRIFT")
            status="COMPLETED"
            outcome={
                "status":status,"schema":"R05_INV01_SOLVER_RAW_V3","run_id":run_id,
                "block_id":row["block_id"],"class":args.class_index,
                "initialization":args.initialization,"initialization_seed":row["initialization_seed"],
                "solver":args.method,"attempt_id":attempt,
                "best_exact_center_JSD":float(record["best_JSD"]),
                "final_exact_center_JSD":float(record["final_JSD"]),
                "selected_exact_center_JSD":float(selected_exact),
                "success":bool(record["best_JSD"]<=TOL),
                "stop_reason":record["stop_reason"],
                "selected_policy_logits":record["selected_policy_logits"],
                "final_policy_logits":record["final_policy_logits"],
                "selected_checkpoint_step":record["selected_policy_step"],
                "selected_stage":stage,"restart_used":restart,
                "counts":cost,"exclusive_solver_wall_s":time.perf_counter()-p2.START,
                "inclusive_solver_wall_s":inclusive_wall,
                "monitor_exact_forward_wall_s":record.get("monitor_exact_forward_wall_s"),
                "monitor_exact_forward_count":record.get("monitor_exact_forward_count"),
                "peak_rss_mib":p2.peak_mib(),
                "input_sha256":inputs["input_sha256"],
                "source_pin_sha256":sha(canonical({k:sha(v) for k,v in captured.items()})),
                "owner_root_sha256":root_digest,"owner_comment_id":args.owner_root_comment,
                "source_commit":os.environ.get("GITHUB_SHA"),"trusted_root_external_to_checkout":True,
                "witness_given_to_solver":False,
                "test_used":False,"tol":TOL,"walk":{"beta":.4,"lambda":.8,"H":12,"dtype":"float64"},
                "bounded_limits":root["limits"],
                "evidence_status":"PROVISIONAL_PENDING_FINAL_QA"}
            write_unique(resultdir/"summary.json",outcome)
            terminal={"status":"COMPLETED","run_id":run_id,"attempt_id":attempt,
                      "code_commit":os.environ.get("GITHUB_SHA"),
                      "start_digest":start_digest,"selected_exact_JSD":selected_exact,
                      "graph_constructed":graph_constructed,
                      "steps":cost,"rss_mib":p2.peak_mib(),
                      "source_pin_sha256":outcome["source_pin_sha256"]}
        finally:
            stop.set()
            thread.join(timeout=1)
    except BaseException as exc:
        if isinstance(exc,Deny):
            status="INVALID_PRECHECK"
        elif isinstance(exc,TimeoutError):
            status="TIMEOUT"
        elif isinstance(exc,MemoryError):
            status="RESOURCE_STOP"
        else:
            status="FAILED"
        terminal={"status":status,"phase":phase,"attempt_id":attempt,"run_id":run_id,
                  "error_type":type(exc).__name__,"error":str(exc)[:250],
                  "graph_constructed":graph_constructed,"start_digest":start_digest}
        if resultdir:
            try:
                write_unique(resultdir/"failure.json",terminal)
                write_unique(resultdir/"traceback.json",{"traceback":traceback.format_exc(limit=8)})
            except (OSError,TypeError):pass
    finally:
        try:
            after_digest=exclusive_record(audit_dir,attempt+".terminal.json",
                                          {"phase":"TERMINAL",**terminal})
        except BaseException:
            status="FAILED"
            # Missing terminal is a visible incomplete attempt and never success.
        if resultdir is not None:
            try: write_unique(resultdir/"audit_ref.json",
                              {"attempt_id":attempt,"external_audit_root":str(audit_dir),
                               "start_sha256":start_digest,"terminal_sha256":after_digest})
            except OSError: pass
        print("R05_SOLVER_ATTEMPT",json.dumps({"run_id":run_id,"attempt_id":attempt,
                             "status":status,"phase":phase,"audit_terminal":after_digest is not None},
                             sort_keys=True),flush=True)
    return 0 if status=="COMPLETED" and after_digest else 2

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--block",choices=("B02","B03","B04","B05","B06"),required=True)
    p.add_argument("--class-index",type=int,choices=(0,1,2),required=True)
    p.add_argument("--initialization",choices=("zero","independent_random"),required=True)
    p.add_argument("--method",choices=ORDER,required=True)
    p.add_argument("--owner-root-comment",type=int,required=True)
    args=p.parse_args()
    raise SystemExit(run_solve(args))

if __name__=="__main__":main()
