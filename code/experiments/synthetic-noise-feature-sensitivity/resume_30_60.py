"""E-CH4-14 continuation ONLY: exact Adam step 30 -> step 60, never a fresh fit.

Preflight reads original evidence and evaluates step 30 without performing updates.
Execute requires a separately reserved online-authorized one-time continuation context.
No code in this module touches a Test split. Original steps 0..30 remain immutable.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, pathlib, sys, time
from datetime import datetime, timezone
import numpy as np
import torch

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runner import (BETA,LAMBDA,H,K,LR,OBJECTIVE,UPDATES,DT,grid,shape_for,
                    prediction_counts,validation_scores)
from engine import ExactMaskedWalk
from train_missing_objectives import train_loss
from checkpoints import load_checkpoint
from advanced_analysis import policy3, verify_metadata
from data import file_sha, fingerprint

FROZEN_BASE="7bc66e4d2535fb15dfa5b569933b2d6c5a0962ac"
APPROVED_INPUT_SPEC="82917272b0ac3850cecfd7bf4698dce55cbe955f8e3c308cf2c44d30e7f1ab8a"
DEST=HERE/"results/continuation_30_60"
TRACE_KEYS=("train_softplus_loss","train_BA","train_Accuracy","validation_BA",
            "validation_Accuracy","gradient_l2","gradient_linf","logits_l2")
NFI_KEYS=("F3","E3","q1","q2","q3","F3_uniform","E3_uniform","NFI3","delta_F3",
          "p0_noise","p0_informative")

def require(cond,message):
    if not cond:raise PermissionError(message)

def source_evidence():
    m=json.loads((HERE/"inputs/derived/manifest.json").read_text())
    prov=json.loads((HERE/"results/metrics/provenance.json").read_text())
    require(m.get("test_accessed") is False and prov.get("test_accessed") is False,"Test provenance")
    require(m.get("schema")=="e-ch4-14-derived-v1" and
            m.get("spec_sha256")==APPROVED_INPUT_SPEC and
            prov.get("data_spec_sha256")==APPROVED_INPUT_SPEC,"Frozen data spec mismatch")
    require(prov.get("n_fits")==36 and prov.get("updates_per_fit")==30,"Original fit count")
    rows={x["subset_id"]:x for x in m["datasets"]}
    require(len(rows)==18 and len(grid())==36,"Wrong 18×2 job grid")
    require(set(rows)=={s for s,f in grid()},"Unknown subset / job")
    return rows,prov

def read_frozen_dataset(row):
    """No Test, no data generation; all four NPZ arrays and file byte SHA checked."""
    path=HERE/"inputs/derived"/(row["subset_id"]+".npz")
    require(file_sha(path)==row["file_sha256"],"Derived NPZ byte SHA differs")
    with np.load(path,allow_pickle=False) as f:
        require(set(f.files)=={"train_x","train_y","validation_x","validation_y"},
                "Only Train and Validation arrays allowed")
        d={key:f[key].copy() for key in f.files}
    for key,val in d.items():
        require(fingerprint(val)==row["hashes"][key+"_sha256"],"Frozen array SHA: "+key)
    d["gamma"]=np.asarray(row["gamma"],dtype=np.float64)
    require(d["train_x"].shape==(300,row["d"]) and
            d["validation_x"].shape==(90,row["d"]) and
            d["train_y"].shape==(300,) and
            d["validation_y"].shape==(90,) and
            d["gamma"].shape==(row["d"],),"Frozen array dimension mismatch")
    return d

def original_trace_and_policy(run_id):
    trace_path=HERE/"results/traces"/run_id/"trace.json"
    original=json.loads(trace_path.read_text())
    require(original.get("run_id")==run_id and original.get("test_accessed") is False,
            "Frozen trace run mismatch")
    trace=original["trace"]
    require(len(trace)==31 and [a["step"] for a in trace]==list(range(31)),
            "Frozen trace has no complete steps 0..30")
    return trace_path,trace

def restore_state(run_id,row,prov,trace_path):
    """SHA-pinned safe load; original state has Adam moments, not just the logits."""
    src=HERE/"results/checkpoints"/run_id/"step_30.pt"
    expected=src.with_suffix(".pt.sha256").read_text().strip()
    require(len(expected)==64,"Frozen checkpoint SHA malformed")
    payload=load_checkpoint(src,expected_sha256=expected)
    verify_metadata(payload,row,payload["metadata"],run_id,src,trace_path,prov["source_sha"])
    fam=run_id.split("__")[1]; d=row["d"]
    require(tuple(payload["logits"].shape)==shape_for(d,fam),"Wrong resumed policy shape")
    z=torch.nn.Parameter(payload["logits"].detach().clone().to(dtype=DT,device="cpu"))
    optimizer=torch.optim.Adam([z],lr=LR)
    optimizer.load_state_dict(payload["optimizer"])
    require(optimizer.param_groups[0]["lr"]==LR,"Optimizer lr differs from frozen run")
    require(len(optimizer.state)==1 and z in optimizer.state,"Optimizer state not mapped to policy")
    state=optimizer.state[z]
    require(int(state["step"])==30,"Adam counter must resume at 30")
    for name in ("exp_avg","exp_avg_sq"):
        require(name in state and state[name].shape==z.shape
                and state[name].dtype==DT and bool(torch.isfinite(state[name]).all()),
                "Missing or incompatible Adam moment: "+name)
    require(bool(torch.equal(z.detach(),payload["logits"])),"Loaded logits must be exact")
    return z,optimizer,expected

def snapshot(graph,validation_y,z,optimizer,run_id,step,feature_mapping,*,update_norm,
             previous_loss=None,wall_elapsed=0):
    """Observe current state; never calls optimizer.step()."""
    require(30<=step<=60,"Only cumulative steps 30..60 allowed")
    require(int(optimizer.state[z]["step"])==step,"Optimizer counter drift")
    z.grad=None
    loss=train_loss(graph,z,run_id.split("__")[1],OBJECTIVE)
    require(bool(torch.isfinite(loss)),"Nonfinite loss")
    loss.backward()
    require(z.grad is not None and bool(torch.isfinite(z.grad).all()),"Nonfinite gradient")
    with torch.no_grad():
        tr=graph.train_outputs(z,run_id.split("__")[1])
        tba,tacc=prediction_counts(tr.scores,graph.labels)
        vs=validation_scores(graph,z,run_id.split("__")[1])
        vba,vacc=prediction_counts(vs,torch.as_tensor(validation_y,dtype=torch.long))
        norm=float(torch.linalg.vector_norm(z))
        g2=float(torch.linalg.vector_norm(z.grad))
        gi=float(torch.max(torch.abs(z.grad)))
        matrix=z.detach().cpu().numpy().copy()
    policies=[("shared",matrix)] if matrix.ndim==2 else [(str(i),matrix[i]) for i in range(3)]
    by_class={}
    for name,sub in policies:
        by_class[name]=policy3(sub,feature_mapping)
    if matrix.ndim==3:
        by_class["macro"]={k:(sum(c[k] for c in
             (by_class["0"],by_class["1"],by_class["2"]))/3
             if all(c[k] is not None for c in
             (by_class["0"],by_class["1"],by_class["2"])) else None)
             for k in by_class["0"]}
    row=dict(run_id=run_id,step=step,update_completed=step,train_softplus_loss=float(loss.detach()),
        train_BA=tba,train_Accuracy=tacc,validation_BA=vba,validation_Accuracy=vacc,
        gradient_l2=g2,gradient_linf=gi,logits_l2=norm,adam_update_l2=update_norm,
        loss_delta=(None if previous_loss is None else float(loss.detach())-previous_loss),
        wall_elapsed_s=float(wall_elapsed),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        nfi3=by_class,numeric_status="finite",test_accessed=False)
    return row,matrix

def compare_step30(row,logits,orig,nfi_rows,run_id):
    """All original scalars plus published F3/E3/NFI3 must agree before update 31."""
    require(row["step"]==orig["step"]==30,"Step 30 boundary mismatch")
    for key in TRACE_KEYS:
        target=float(orig[key]); got=float(row[key])
        # BA must match an identical hard-decision count, not just approximate.
        atol=2e-11 if key not in ("gradient_l2","gradient_linf") else 5e-9
        require(np.isclose(got,target,rtol=2e-8,atol=atol),
                "Original step30 diagnostic mismatch: "+run_id+" "+key)
    require(np.isclose(row["adam_update_l2"],orig["adam_update_l2"],rtol=0,atol=1e-14),
            "Step30 update norm must originate from saved original trace")
    records={x["candidate_class"]:x for x in nfi_rows if x["run_id"]==run_id}
    require(set(records)==set(row["nfi3"]),"Incomplete published NFI30 policy rows")
    for key,values in row["nfi3"].items():
        for metric in NFI_KEYS:
            expected=records[key][metric];actual=values[metric]
            if expected is None or actual is None:
                require(expected is None and actual is None,"NFI null mismatch")
            else:require(np.isclose(actual,expected,rtol=2e-10,atol=2e-12),
                         "Step30 published NFI differs: "+key+"/"+metric)
    require(np.isclose(np.linalg.norm(logits),orig["logits_l2"],rtol=2e-8,atol=1e-11),
            "Step30 logits L2 mismatch")
    return True

def nfi_table():
    """Read the frozen after-30 analysis without recomputing the historical baseline."""
    import csv
    with (HERE/"tables/nfi3_by_policy.csv").open(newline="",encoding="utf8") as f:
        raw=list(csv.DictReader(f))
    fields=set(NFI_KEYS)
    for x in raw:
        for key in fields:
            x[key]=float(x[key]) if x[key]!="" else None
    require(len(raw)==90,"Missing frozen checkpoint-30 NFI policy table")
    return raw

def read_run(run_id,row,prov,nfi_rows):
    trace_path,trace=original_trace_and_policy(run_id)
    d=read_frozen_dataset(row)
    z,opt,digest=restore_state(run_id,row,prov,trace_path)
    graph=ExactMaskedWalk(d["train_x"],d["train_y"],d["gamma"],
                          beta=BETA,lam=LAMBDA,horizon=H,query_x=d["validation_x"])
    first,matrix=snapshot(graph,d["validation_y"],z,opt,run_id,30,row["feature_mapping"],
                           update_norm=trace[30]["adam_update_l2"],
                           previous_loss=trace[29]["train_softplus_loss"])
    compare_step30(first,matrix,trace[30],nfi_rows,run_id)
    return graph,d,z,opt,first,matrix,digest,trace_path

def preflight():
    """Verifies all 36 origins, then terminates without stepping Adam."""
    require(not DEST.exists(),"Continuation results directory exists; no retry")
    rows,prov=source_evidence()
    nfi_rows=nfi_table()
    digests={}
    for subset,family in grid():
        run_id=f"{subset}__{family}"
        _,_,_,_,_,_,sha,_=read_run(run_id,rows[subset],prov,nfi_rows)
        digests[run_id]=sha
    require(len(digests)==36,"Incomplete step30 preflight")
    print("E14_RESUME_PREFLIGHT_NO_FIT_PASS",json.dumps(
        {"runs":36,"source30":FROZEN_BASE,"additional_updates_executed":0,
         "checkpoint_sha256":digests,"test_accessed":False},sort_keys=True))
    return digests

def atomic_new(path,contents,*,binary=False):
    """Exclusive create; never overwrite a completed or partial file."""
    path=pathlib.Path(path)
    with path.open("xb" if binary else "x",**({} if binary else {"encoding":"utf8"})) as f:
        f.write(contents)
        f.flush();os.fsync(f.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()

def append_durable(stream,entry):
    stream.write(json.dumps(entry,sort_keys=True,allow_nan=False)+"\n")
    stream.flush();os.fsync(stream.fileno())

def safe_step60_checkpoint(path,z,opt,trace_sha,logits_sha,source_sha,origin_sha,run_id,auth):
    state=opt.state_dict()
    require(int(opt.state[z]["step"])==60,"Step60 Adam counter")
    meta=dict(schema="e-ch4-14-resume-checkpoint-metadata-v1",run_id=run_id,
       source_code_sha=source_sha,source_checkpoint30_sha256=origin_sha,
       policy_logits_steps_sha256=logits_sha,trace_jsonl_sha256=trace_sha,
       owner_approval_url=auth["owner_approval_url"],qa_approval_url=auth["qa_approval_url"],
       budget_ref=auth["budget_ref"],test_accessed=False)
    payload={"schema":"e-ch4-14-resume-checkpoint-v1","step":60,
             "logits":z.detach().clone().cpu(),"optimizer":state,"metadata":meta}
    buffer=io.BytesIO();torch.save(payload,buffer)
    digest=atomic_new(path,buffer.getvalue(),binary=True)
    atomic_new(path.with_suffix(".pt.sha256"),digest+"\n")
    return digest

def run_one(run_id,row,prov,nfi_rows,ledger,auth):
    """A single committed-to-budget run; no call possible without production authorization."""
    graph,d,z,opt,first,matrix,origsha,trace_path=read_run(run_id,row,prov,nfi_rows)
    output=DEST/run_id
    output.mkdir(exist_ok=False)
    append_durable(ledger,dict(event="START",run_id=run_id,origin_checkpoint_sha256=origsha))
    # Per-step JSONL is written before the next update so partial evidence survives.
    # The final canonical stacked NPZ is a secondary immutable export of this journal.
    trace_file=output/"trace_30_60.jsonl"
    logits_file=output/"logits_30_60.jsonl"
    states=[];start=time.perf_counter()
    with trace_file.open("x",encoding="utf8") as trace,logits_file.open("x",encoding="utf8") as policies:
        previous=first["train_softplus_loss"]
        for step in range(30,61):
            if step==30:
                rec,mat=first,matrix
            else:
                require(int(opt.state[z]["step"])==step-1,"Adam counter prior to step")
                before=z.detach().clone()
                opt.step()
                # Every completed optimizer.step spends budget EVEN if its result fails validation.
                append_durable(ledger,dict(event="UPDATE",run_id=run_id,
                                           new_updates_completed=step-30,cumulative_step=step))
                require(bool(torch.isfinite(z).all()),"Nonfinite updated logits")
                delta=float(torch.linalg.vector_norm(z.detach()-before))
                require(int(opt.state[z]["step"])==step,"Adam update index")
                rec,mat=snapshot(graph,d["validation_y"],z,opt,run_id,step,row["feature_mapping"],
                     update_norm=delta,previous_loss=previous,wall_elapsed=time.perf_counter()-start)
                previous=rec["train_softplus_loss"]
            states.append(mat)
            append_durable(trace,rec)
            append_durable(policies,dict(step=step,logits=mat.tolist(),
                sha256_float64_le=hashlib.sha256(np.asarray(mat,dtype="<f8").tobytes()).hexdigest()))
    trace_sha=file_sha(trace_file)
    logits_journal_sha=file_sha(logits_file)
    matrices=np.stack(states,axis=0)
    require(matrices.dtype==np.float64 and matrices.shape[0]==31,"Invalid logits sequence")
    npz=output/"logits_steps30_60.npz"
    if npz.exists():raise FileExistsError(npz)
    with npz.open("xb") as f:
        np.savez(f,steps=np.arange(30,61,dtype=np.int64),logits=matrices)
        f.flush();os.fsync(f.fileno())
    npz_sha=file_sha(npz)
    chk=output/"step_60.pt"
    new_sha=safe_step60_checkpoint(chk,z,opt,trace_sha,npz_sha,auth["approved_code_sha"],origsha,run_id,auth)
    manifest=dict(schema="e-ch4-14-resume-run-manifest-v1",run_id=run_id,
       old_checkpoint_sha256=origsha,new_checkpoint_sha256=new_sha,
       trace_sha256=trace_sha,logits_journal_sha256=logits_journal_sha,
       logits_npz_sha256=npz_sha,step_start=30,step_end=60,
       updates_added=30,logits_shape=list(matrices.shape),
       owner_approval_url=auth["owner_approval_url"],qa_approval_url=auth["qa_approval_url"],
       budget_ref=auth["budget_ref"],test_accessed=False)
    atomic_new(output/"manifest.json",json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    append_durable(ledger,dict(event="SUCCESS",run_id=run_id,updates_added=30,
                               new_checkpoint_sha256=new_sha))
    return new_sha

def authorized_execute(context):
    import resume_gate
    auth=resume_gate.verify_runtime_context(context)
    require(os.environ.get("E14_RESUME_EXECUTE_VERIFIED")=="true","No approved runtime")
    require(not DEST.exists(),"Continuation already started; no implicit retry")
    require(auth["budget_ref"]=="refs/tags/e-ch4-14-resume-30-60-budget-spent",
            "Not the independent continuation tag")
    rows,prov=source_evidence()
    nfi=nfi_table()
    # Must preflight ALL origins before using any additional training budget.
    preflight()
    DEST.mkdir(exist_ok=False)
    ledger_path=DEST/"resume_budget.jsonl"
    with ledger_path.open("x",encoding="utf8") as ledger:
        append_durable(ledger,dict(event="BUDGET_RESERVED",source_sha=auth["approved_code_sha"],
               budget_ref=auth["budget_ref"],owner_approval_url=auth["owner_approval_url"],
               qa_approval_url=auth["qa_approval_url"],updates_max=1080,
               runs=36,test_accessed=False))
        try:
            for subset,family in grid():
                run_one(subset+"__"+family,rows[subset],prov,nfi,ledger,auth)
        except BaseException as exc:
            append_durable(ledger,dict(event="FAIL_STOP",
                          error_type=type(exc).__name__,message=str(exc)[:350],
                          test_accessed=False))
            raise
    print("E14_RESUME_30_TO_60_COMPLETED",json.dumps(dict(fits=36,added_updates=1080,test_accessed=False)))

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=("preflight","execute"))
    p.add_argument("--context",type=pathlib.Path)
    args=p.parse_args(argv)
    if args.mode=="preflight":
        preflight()
    else:
        require(args.context is not None,"Verified runtime context required")
        authorized_execute(args.context)

if __name__=="__main__":
    main()
