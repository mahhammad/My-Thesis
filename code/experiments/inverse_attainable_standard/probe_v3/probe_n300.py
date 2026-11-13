#!/usr/bin/env python3
"""R05 §4.10 n300 resource pre-gate: F0/Q1/A1/J1/M1 (NO solver steps)."""
from __future__ import annotations
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"runner_v3"))
from data_contract import EXPERIMENT,ROOT,Deny,require,owner_root,decoded_block,attested_sources,sha
from run_one import init_modules,check_attested_modules,exclusive_record,mk_audit_root,write_unique

LIMIT=110.0
RSS_LIMIT_MIB=1024.0

def rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def run(block,root_id):
    ident=uuid.uuid4().hex
    ad=mk_audit_root()
    start=exclusive_record(ad,ident+".start.json",{"event":"START","task":"N300_RESOURCE_PROBE",
        "block":block,"attempt_id":ident,"graph_constructed":0,"solver_steps":0,"at":time.time()})
    out=EXPERIMENT/"results"/"resource_probes"/block/ident
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result={"status":"INVALID_PRECHECK","task":"F0_Q1_A1_J1_M1","block":block,
            "attempt_id":ident,"start_audit_sha256":start,"solver_steps":0,
            "graph_constructed":0}
    try:
        require(sys.flags.optimize==0,"OPTIMIZED_PYTHON_DENIED")
        root,root_sha=owner_root(root_id)
        require(root["allow_numeric"] is True,"ROOT_NUMERIC_NOT_GRANTED")
        require(root["runtime"]["python"]==sys.version.split()[0],"PYTHON_RUNTIME_DRIFT")
        inputs=decoded_block(root,block)
        require(inputs["block"]["nref"]==300,"WRONG_NREF")
        captured=attested_sources(root)
        t0=time.perf_counter()
        np,torch,mods=init_modules(captured)
        require(root["runtime"]["numpy"]==np.__version__ and
                root["runtime"]["torch"]==torch.__version__,"TORCH_NUMPY_RUNTIME_DRIFT")
        check_attested_modules(captured,mods)
        require(all(not inputs[k].flags.writeable for k in ("X","y","mask","ids","targets")),
                "MUTABLE_INPUTS")
        # First graph is created ONLY after root, immutable input, source and imported-code gates.
        from generate import load_config
        cfg=mods["generate"].load_config(ROOT/"code/data_generator/config.json")
        graph=mods["stream_graph"].StreamGraph(inputs["X"],inputs["y"],cfg["gamma"],validation_x=None,
                                                beta=.4,lam=.8,horizon=12)
        result["graph_constructed"]=1
        labels=inputs["y"]
        cl=int(np.argmax(np.bincount(labels,minlength=3)))
        idx=np.flatnonzero(labels==cl)
        target=torch.as_tensor(inputs["targets"][cl].copy(),dtype=torch.float64)
        Z=torch.nn.Parameter(torch.zeros((graph.d+1,graph.d),dtype=torch.float64))
        JSD=mods["run"].jsd
        measures={}
        def stamp(stage,op):
            begin=time.perf_counter()
            outcome=op()
            elapsed=time.perf_counter()-begin
            current=rss()
            measures[stage]={"wall_s":elapsed,"rss_peak_mib":current}
            require(elapsed>=0 and elapsed<LIMIT,"PROBE_TIME_"+stage)
            require(current<RSS_LIMIT_MIB,"PROBE_MEMORY_"+stage)
            require(time.monotonic()-started<LIMIT,"PROBE_TOTAL_WALL")
            return outcome
        stamp("F0_exact_forward",lambda:graph.center_train(Z,idx).detach())
        def Q1():
            Z.grad=None
            v=graph.local_q_loss(Z,idx,target)
            require(bool(torch.isfinite(v)),"Q1_NOT_FINITE")
            v.backward()
            require(Z.grad is not None and bool(torch.isfinite(Z.grad).all()),"Q1_GRAD")
        stamp("Q1_one_gradient",Q1)
        def A1():
            Z.grad=None
            v=JSD(graph.center_train(Z,idx),target)
            require(bool(torch.isfinite(v)),"A1_NOT_FINITE")
            v.backward()
            require(Z.grad is not None and bool(torch.isfinite(Z.grad).all()),"A1_GRAD")
        stamp("A1_exact_gradient",A1)
        def J1():
            zflat=Z.detach().reshape(-1).requires_grad_(True)
            v=graph.center_train(zflat.reshape(graph.d+1,graph.d),idx)[:-1]
            rows=[torch.autograd.grad(v[i],zflat,retain_graph=i<len(v)-1)[0]
                  for i in range(len(v))]
            matrix=torch.stack(rows)
            require(matrix.shape==(2,(graph.d+1)*graph.d) and
                    bool(torch.isfinite(matrix).all()),"J1_SHAPE_OR_NAN")
        stamp("J1_jacobian_two_gradients",J1)
        def M1():
            with torch.no_grad():
                v=graph.center_train(Z,idx)
                return float(JSD(v,target).item())
        monitored=stamp("M1_exact_monitor",M1)
        require(monitored>=-1e-14,"M1_NEGATIVE")
        result.update(status="PASS_INTERNAL_RESOURCE_PROBE",class_index=cl,
                      class_count=len(idx),resource_measures=measures,
                      wall_total_s=time.monotonic()-started,peak_rss_mib=rss(),
                      root_comment_id=root_id,root_payload_sha256=root_sha,
                      snapshot_sha256=inputs["input_sha256"]["snapshot"],
                      target_sha256=inputs["input_sha256"]["targets"],
                      source_sha256={k:sha(v) for k,v in captured.items()},
                      first_graph_attested=True,
                      evidence_status="PROVISIONAL_PENDING_FINAL_QA")
    except BaseException as exc:
        result.update(status="FAIL_CLOSED",error_type=type(exc).__name__,error=str(exc)[:300],
                      wall_total_s=time.monotonic()-started,peak_rss_mib=rss())
    finally:
        end=exclusive_record(ad,ident+".terminal.json",{"event":"TERMINAL","task":"N300_RESOURCE_PROBE",
                    "attempt_id":ident,"status":result["status"],"at":time.time(),
                    "graph_constructed":result["graph_constructed"],"solver_steps":0})
        result["terminal_audit_sha256"]=end
        write_unique(out/"probe.json",result)
        print("R05_N300_RESOURCE_PROBE",json.dumps({"block":block,"status":result["status"],
             "rss":result.get("peak_rss_mib"),"wall":result.get("wall_total_s"),
             "class":result.get("class_index"),"attempt":ident}),flush=True)
    return result["status"]=="PASS_INTERNAL_RESOURCE_PROBE"

if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1] not in ("B02","B04","B06"):
        raise SystemExit("Usage: probe_n300.py B02|B04|B06 owner_root_comment_id")
    raise SystemExit(0 if run(sys.argv[1],int(sys.argv[2])) else 2)
