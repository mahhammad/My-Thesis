#!/usr/bin/env python3
"""B02 n=300 immutable Train and witness-target preflight. NO OPTIMIZERS."""
from __future__ import annotations
import hashlib, json, os, platform, resource, struct, sys, tempfile, threading, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(ROOT/"code/experiments/multistep_policy_target_match"))
sys.path.insert(0,str(ROOT/"code/experiments/diag_max_full300"))
from generate import generate,load_config
from stream_graph import StreamGraph
from run import jsd, DT

PLAN=ROOT/"code/experiments/inverse_attainable_standard/standard_plan/plan_108.json"
CFG=ROOT/"code/data_generator/config.json"
GENER=ROOT/"code/data_generator/generate.py"
STREAM=ROOT/"code/experiments/diag_max_full300/stream_graph.py"
RUN=ROOT/"code/experiments/multistep_policy_target_match/run.py"
BID="INV01-B02-n300-data20261102"
MAGIC=b"B02TRAIN_CANONICAL_V1\n"
BUDGET=120.
RSS_CEILING=1024.
START=time.monotonic()
PEAK=0.
STOP=threading.Event()
def sha(b): return hashlib.sha256(b).hexdigest()
def filehash(path): return sha(path.read_bytes())
def rss():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"): return int(line.split()[1])/1024
    except OSError: pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
def guard(stage):
    global PEAK
    PEAK=max(PEAK,rss())
    if PEAK>=RSS_CEILING: raise MemoryError(f"RSS hard limit at {stage}: {PEAK:.2f} MiB")
    if time.monotonic()-START>=BUDGET: raise TimeoutError(f"Numerical wall limit at {stage}")
def watchdog():
    while not STOP.wait(.05):
        if rss()>=RSS_CEILING or time.monotonic()-START>=BUDGET:
            (HERE/"guard_failure.log").write_text(f"FAIL_CLOSED watchdog wall={time.monotonic()-START:.4f}, rss={rss():.3f}\n")
            os._exit(124)
def put_atomic(path,payload):
    temp=path.with_name(path.name+".partial")
    temp.write_bytes(payload)
    os.replace(temp,path)
def jsonbytes(o): return (json.dumps(o,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
def canonical(x,y,mask,ids):
    if x.shape!=(300,5) or y.shape!=(300,) or mask.shape!=x.shape or ids.shape!=(300,):
        raise ValueError("unexpected B02 shape")
    if not np.array_equal(mask,np.isfinite(x)): raise ValueError("mask/X mismatch")
    if not np.array_equal(ids,np.arange(300)): raise ValueError("reference IDs or order drift")
    if np.isinf(x).any(): raise ValueError("infinite X")
    z=np.array(x,dtype="<f8",order="C",copy=True)
    z[~mask]=np.float64(np.nan)   # canonical quiet NaN, no change to observed value
    ys=np.ascontiguousarray(y,dtype="<i8")
    ms=np.ascontiguousarray(mask.astype("u1"))
    ix=np.ascontiguousarray(ids,dtype="<i8")
    return MAGIC+z.tobytes(order="C")+ys.tobytes(order="C")+ms.tobytes(order="C")+ix.tobytes(order="C")
def parse(payload):
    if not payload.startswith(MAGIC): raise ValueError("magic mismatch")
    tail=memoryview(payload)[len(MAGIC):]
    chunks=(300*5*8,300*8,300*5,300*8)
    if len(tail)!=sum(chunks): raise ValueError("payload size mismatch")
    i=0
    out=[]
    for width,dt,shape in zip(chunks,("<f8","<i8","u1","<i8"),((300,5),(300,),(300,5),(300,))):
        out.append(np.frombuffer(tail[i:i+width],dtype=dt).copy().reshape(shape))
        i+=width
    x,y,m,ids=out
    if not np.array_equal(np.isfinite(x),m.astype(bool)):raise ValueError("mask inconsistency")
    if not np.array_equal(ids,np.arange(300)):raise ValueError("ID corruption")
    if np.any(y<0) or np.any(y>=3):raise ValueError("invalid label")
    if sorted(np.bincount(y,minlength=3).tolist())==[0,0,300]:raise ValueError("class collapse")
    return x,y,m.astype(bool),ids
def verify_snapshot(snapshot,manifest,targets):
    if sha(snapshot)!=manifest["snapshot"]["file_sha256"]:raise ValueError("snapshot file SHA mismatch")
    if sha(snapshot)!=manifest["snapshot"]["payload_sha256"]:raise ValueError("snapshot payload SHA mismatch")
    xx,yy,mm,ii=parse(snapshot)
    for name,arr in (("X",xx),("y",yy),("mask",mm.astype("u1")),("ids",ii)):
        bb=np.ascontiguousarray(arr).tobytes()
        if sha(bb)!=manifest["snapshot"]["array_sha256"][name]:raise ValueError(name+" SHA mismatch")
    for name,path in (("config",CFG),("generator",GENER),("streamgraph",STREAM),("graph_run",RUN),("plan",PLAN)):
        if manifest["source_sha256"][name]!=filehash(path):raise ValueError(name+" source SHA mismatch")
    if len(targets)!=3*3*8 or sha(targets)!=manifest["targets"]["file_sha256"]:
        raise ValueError("target file SHA mismatch")
    tc=np.frombuffer(targets,dtype="<f8").reshape((3,3))
    for c in range(3):
        if sha(np.ascontiguousarray(tc[c]).tobytes())!=manifest["targets"]["per_class"][c]["sha256"]:
            raise ValueError("target SHA mismatch")
        if not np.isclose(tc[c].sum(),1.,atol=1e-10) or not np.isfinite(tc[c]).all():
            raise ValueError("invalid target")
    if np.bincount(yy,minlength=3).tolist()!=manifest["class_counts"]:
        raise ValueError("class count mismatch")
    return xx,yy,mm,ii
def test_tamper(raw,m,t):
    out={}
    for name,offset in [("byte",len(MAGIC)+4),("mask",len(MAGIC)+300*5*8+300*8+10),("id",len(raw)-8)]:
        corrupted=bytearray(raw)
        corrupted[offset]^=1
        try: verify_snapshot(bytes(corrupted),m,t)
        except (ValueError,AssertionError) as exc:out[name]={"detected":True,"reason":str(exc)}
        else:raise AssertionError("tampering was accepted: "+name)
    return out

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    for p in ("train_canonical.bin","targets_float64_le.bin","manifest.json","verification.json"):
        if (HERE/p).exists(): raise FileExistsError("immutable output already exists: "+p)
    watcher=threading.Thread(target=watchdog,daemon=True); watcher.start()
    try:
        guard("start")
        plan=json.loads(PLAN.read_text())
        block=next(b for b in plan["blocks"] if b["block_id"]==BID)
        rows=[r for r in plan["runs"] if r["block_id"]==BID]
        required={"Q-only","Exact-Adam","Q+Bootstrap-GN"}
        assert len(rows)==18 and len({r["run_id"] for r in rows})==18
        assert block["nref"]==300 and block["data_seed"]==20261102 and block["reference_seed"]==20261201 and block["teacher_seed"]==20261211
        assert not block["authorized_to_execute"] and all(r["state"]=="NOT_RUN" for r in rows)
        assert {(r["class_index"],r["initialization"],r["solver"]) for r in rows}=={
            (c,st,method) for c in range(3) for st in ("zero","independent_random") for method in required}
        assert all(r["precision"]=="float64" and r["H"]==12 and r["beta"]==.4 and r["lambda"]==.8 for r in rows)
        spec=load_config(CFG)
        assert spec["seed"]==20261102 and spec["sizes"]["train"]==300
        # Original generator writes all three historical splits into a temporary directory.
        # Only the TRAIN entry is accessed here. No Validation/Test reads or scores.
        with tempfile.TemporaryDirectory() as tmp:
            train=generate(spec,tmp,finite_q=False)["train"]
            x,mask,y=train
            x=x.copy(); y=y.copy();mask=mask.copy()
        guard("generated train")
        ids=np.arange(300,dtype=np.int64)
        assert np.array_equal(mask,np.isfinite(x)) and x.shape==(300,5)
        counts=np.bincount(y,minlength=3).tolist()
        if len(counts)!=3 or min(counts)<=0:raise ValueError("empty class")
        payload=canonical(x,y,mask,ids)
        parsed=parse(payload)
        if not np.array_equal(parsed[1],y) or not np.array_equal(parsed[3],ids) or not np.array_equal(parsed[2],mask):
            raise AssertionError("train roundtrip mismatch")
        if not np.array_equal(parsed[0][mask],x[mask]) or not np.array_equal(np.isnan(parsed[0]),np.isnan(x)):
            raise AssertionError("X roundtrip mismatch")
        guard("snapshot roundtrip")
        torch.set_num_threads(1)
        graph=StreamGraph(x,y,spec["gamma"],validation_x=None,beta=.4,lam=.8,horizon=12)
        rng=np.random.default_rng(20261211)
        witness=rng.normal(0,1.7,size=(3,graph.d+1,graph.d))
        witness-=witness.mean(axis=-1,keepdims=True)
        witness_hash=sha(np.ascontiguousarray(witness).tobytes())
        centers=[]
        for c in range(3):
            with torch.no_grad():
                ctr=graph.center_train(torch.as_tensor(witness[c],dtype=DT),np.flatnonzero(y==c)).detach().cpu().numpy()
            if not np.isfinite(ctr).all() or np.any(ctr<0) or not np.isclose(ctr.sum(),1.,atol=1e-10):
                raise AssertionError("invalid target class "+str(c))
            centers.append(np.asarray(ctr,dtype="<f8"))
            guard("target forward "+str(c))
        del graph,witness
        targets=np.stack(centers).astype("<f8",copy=False).tobytes(order="C")
        array_sha={name:sha(np.ascontiguousarray(arr).tobytes()) for name,arr in
                   (("X",parsed[0]),("y",parsed[1]),("mask",parsed[2].astype("u1")),("ids",parsed[3]))}
        manifest={
          "schema":"B02_TRAIN_AND_TARGET_V1","block":BID,"status":"LOCKED_NO_SOLVERS",
          "reference_branch_sha":"968675cde8783adf69db13efd7de0fb3aab703c5",
          "chapter4_sha":"de02fdda91e23db177de4429b3e1a9693c046d83",
          "snapshot":{"file":"train_canonical.bin","file_sha256":sha(payload),
            "payload_sha256":sha(payload),"array_sha256":array_sha,
            "format":"MAGIC + row-major little-endian float64 X[300,5] + int64 y[300] + uint8 mask[300,5] + int64 ids[300]",
            "magic":MAGIC.decode().strip(),"shape":{"X":[300,5],"y":[300],"mask":[300,5],"ids":[300]},
            "endianness":"little","order":"C","missing":"canonical IEEE754 quiet NaN float64 for absent; mask stored separately",
            "ids_semantics":"original zero-based Train row indices, unchanged order"},
          "source_sha256":{k:filehash(p) for k,p in (("config",CFG),("generator",GENER),("streamgraph",STREAM),("graph_run",RUN),("plan",PLAN))},
          "seed":{"data":20261102,"reference":20261201,"teacher":20261211,
                  "random_initializations":[r["initialization_seed"] for r in rows if r["initialization"]=="independent_random"]},
          "class_counts":counts,"observed_per_feature":mask.sum(axis=0).tolist(),
          "nref":300,"three_classes":True,"planned_run_ids":[r["run_id"] for r in rows],
          "contexts":18,"authorized_to_solve":False,
          "walk":{"dtype":"torch.float64","beta":.4,"lambda":.8,"H":12,"method":"exact_no_grad_three_forwards"},
          "targets":{"file":"targets_float64_le.bin","file_sha256":sha(targets),
            "per_class":[{"class":c,"vector":centers[c].tolist(),"sha256":sha(centers[c].tobytes()),
             "sum":float(centers[c].sum()),"self_jsd":float(jsd(torch.tensor(centers[c],dtype=DT),torch.tensor(centers[c],dtype=DT)).item())} for c in range(3)]},
          "teacher_logits_SHA256_ONLY":witness_hash,"teacher_given_to_solver":False,
          "no_optimizer":True,"no_test_read":True,"generator_creates_unused_historical_other_splits_in_tempdir":True,
          "versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},
          "resource_limits":{"numerical_wall_seconds":120,"rss_mib":1024}
        }
        guard("manifest prepared")
        verify_snapshot(payload,manifest,targets)
        tamper=test_tamper(payload,manifest,targets)
        guard("tamper tests")
        put_atomic(HERE/"train_canonical.bin",payload)
        put_atomic(HERE/"targets_float64_le.bin",targets)
        put_atomic(HERE/"manifest.json",jsonbytes(manifest))
        verification={"status":"PASS","roundtrip":"PASS","tamper":tamper,
          "wall_numerical_seconds":time.monotonic()-START,"peak_rss_mib":max(PEAK,rss()),
          "solvers_executed":0,"forwards":3,"gradients":0,"jacobians":0}
        put_atomic(HERE/"verification.json",jsonbytes(verification))
        print(json.dumps({"status":"PASS","snapshot_sha":sha(payload),"targets_sha":sha(targets),
                          "class_counts":counts,"wall_s":verification["wall_numerical_seconds"],
                          "peak_rss_mib":verification["peak_rss_mib"],"tamper":list(tamper)},sort_keys=True))
    except BaseException as exc:
        fail={"status":"STOP","reason":repr(exc),"wall_numerical_seconds":time.monotonic()-START,
              "peak_rss_mib":max(PEAK,rss()),"solvers_executed":0}
        put_atomic(HERE/"guard_failure.json",jsonbytes(fail))
        raise
    finally: STOP.set()

if __name__=="__main__":main()
