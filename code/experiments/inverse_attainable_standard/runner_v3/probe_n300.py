#!/usr/bin/env python3
"""R05 §4.10 n300 F0/Q1/A1/J1/M1 probe, NO optimizer steps or solver.
Read-only pinned input; all source code executed from attested bytes in memory.
"""
from __future__ import annotations
import argparse,json,os,platform,resource,sys,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from data_contract import (ROOT,EXPERIMENT,Deny,owner_root,decoded_block,attested_sources,
                           require,sha,gitblob)
from run_one import (mk_audit_root,exclusive_record,init_modules,check_attested_modules,
                     write_unique)
BLOCKS=("B02","B04","B06")
PROBE_BUDGET=120.0
PROBE_HARD_RSS=1024.0

def peakrss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--block",choices=BLOCKS,required=True)
    p.add_argument("--owner-root-comment",type=int,required=True)
    args=p.parse_args()
    attempt=uuid.uuid4().hex
    audit=mk_audit_root()
    sd=exclusive_record(audit,attempt+".start.json",
                        {"phase":"START","attempt_id":attempt,"operation":"PROBE_ONLY",
                         "block":args.block,"graph_constructed":0,"optimizer_steps":0})
    resultdir=EXPERIMENT/"results/probes"/args.block/attempt
    resultdir.mkdir(parents=True,exist_ok=False)
    phase="OWNER"
    status="INVALID_PRECHECK"
    records={}
    wallstart=time.perf_counter()
    try:
        root,trustsha=owner_root(args.owner_root_comment)
        require(root["allow_numeric"] is True,"NO_PROBE_GRANT")
        inputs=decoded_block(root,args.block)
        import numpy as np, torch
        require({"python":platform.python_version(),"numpy":np.__version__,
                 "torch":torch.__version__}==root["runtime"],"RUNTIME_MISMATCH")
        captured=attested_sources(root)
        phase="AT_USE"
        _,_,loaded=init_modules(captured)
        check_attested_modules(captured,loaded)
        cfg=loaded["generate"].load_config(ROOT/"code/data_generator/config.json")
        StreamGraph=loaded["stream_graph"].StreamGraph
        jsd=loaded["run"].jsd
        torch.set_num_threads(1)
        phase="F0"
        graph=StreamGraph(inputs["X"],inputs["y"],cfg["gamma"],validation_x=None,
                          beta=.4,lam=.8,horizon=12)
        idx=np.flatnonzero(inputs["y"]==1)  # registered worst-case c=1 in n300
        nclass=len(idx)
        require(nclass==inputs["manifest"]["class_counts"][1],"CLASS_REF_DRIFT")
        target=torch.as_tensor(inputs["targets"][1].copy(),dtype=torch.float64)
        z0=torch.zeros((graph.d+1,graph.d),dtype=torch.float64)
        count={"optimizer_steps":0,"exact_forwards":0,"q_gradient":0,
               "exact_gradient":0,"jacobian_rows":0,"monitor_forward":0}
        def guard():
            require(time.perf_counter()-wallstart<PROBE_BUDGET,"PROBE_TIMEOUT")
            require(peakrss()<PROBE_HARD_RSS,"PROBE_RSS_HARD")
        def measure(name,fun):
            guard()
            start=time.perf_counter()
            val=fun()
            stage_wall=time.perf_counter()-start
            guard()
            records[name]={"wall_s":stage_wall,"peak_rss_mib":peakrss()}
            return val
        def F0():
            with torch.no_grad():
                center=graph.center_train(z0,idx)
                err=jsd(center,target)
                require(torch.isfinite(err).item(),"F0_NONFINITE")
                return float(err)
        f0=measure("F0",F0)
        count["exact_forwards"]+=1
        phase="Q1"
        def Q1():
            z=z0.detach().clone().requires_grad_(True)
            val=graph.local_q_loss(z,idx,target)
            val.backward()
            require(bool(torch.isfinite(z.grad).all()),"Q1_NONFINITE")
            return float(val.detach())
        q1=measure("Q1",Q1)
        count["q_gradient"]+=1
        phase="A1"
        def A1():
            z=z0.detach().clone().requires_grad_(True)
            val=jsd(graph.center_train(z,idx),target)
            val.backward()
            require(bool(torch.isfinite(z.grad).all()),"A1_NONFINITE")
            return float(val.detach())
        a1=measure("A1",A1)
        count["exact_gradient"]+=1
        phase="J1"
        def J1():
            z=z0.reshape(-1).detach().requires_grad_(True)
            v=graph.center_train(z.reshape(graph.d+1,graph.d),idx)[:-1]
            grad=[torch.autograd.grad(v[c],z,retain_graph=c<graph.k-2)[0]
                  for c in range(graph.k-1)]
            j=torch.stack(grad)
            require(tuple(j.shape)==(2,30),"J1_SHAPE")
            require(bool(torch.isfinite(j).all()),"J1_NONFINITE")
            return float(torch.linalg.vector_norm(j).detach())
        j1=measure("J1",J1)
        count["jacobian_rows"]+=2
        phase="M1"
        m1=measure("M1",F0)
        count["monitor_forward"]+=1
        require(abs(m1-f0)<=1e-12,"MONITOR_FORWARD_DRIFT")
        report={"schema":"R05_INV01_N300_RESOURCE_PROBE_V1","status":"PROBE_PASS",
                "block":args.block,"attempt_id":attempt,"nref":300,
                "class_index":1,"class_count":nclass,
                "source_commit":os.environ.get("GITHUB_SHA"),
                "root_comment":args.owner_root_comment,"root_sha256":trustsha,
                "input_sha256":inputs["input_sha256"],
                "metrics":{"F0_exact_JSD":f0,"Q1_surrogate":q1,"A1_exact_JSD":a1,
                           "J1_jacobian_norm":j1,"M1_exact_JSD":m1},
                "operations":count,"stages":records,
                "wall_s":time.perf_counter()-wallstart,"peak_rss_mib":peakrss(),
                "resource_limits":{"wall_s":PROBE_BUDGET,"rss_mib":PROBE_HARD_RSS},
                "solver_runs":0,"witness_exposed":False,"test_read":False,
                "evidence_status":"PROVISIONAL_PENDING_FINAL_QA"}
        write_unique(resultdir/"probe.json",report)
        status="PROBE_PASS"
        print("R05_N300_PROBE_PASS",json.dumps({
            "block":args.block,"wall":report["wall_s"],"rss":report["peak_rss_mib"],
            "F0":f0,"J1":records["J1"]}),flush=True)
        terminal={"status":"PROBE_PASS","block":args.block,"attempt_id":attempt,
                  "operations":count,"graph_constructed":1,"solver_runs":0,
                  "start_digest":sd}
    except BaseException as exc:
        status="PROBE_FAIL"
        terminal={"status":status,"block":args.block,"attempt_id":attempt,
                  "phase":phase,"error_type":type(exc).__name__,
                  "error":str(exc)[:250],"solver_runs":0,"start_digest":sd}
        write_unique(resultdir/"failure.json",terminal)
        print("R05_N300_PROBE_FAIL",json.dumps(terminal),flush=True)
    finally:
        d=exclusive_record(audit,attempt+".terminal.json",{"phase":"TERMINAL",**terminal})
        write_unique(resultdir/"audit_ref.json",{"start_sha256":sd,"terminal_sha256":d,
                     "external_audit_dir":str(audit),"attempt_id":attempt})
    raise SystemExit(0 if status=="PROBE_PASS" else 2)

if __name__=="__main__":main()
