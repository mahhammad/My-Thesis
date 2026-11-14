#!/usr/bin/env python3
"""R05 Sec410: independent Train-only selected-policy center + density replay.

Does not optimize. Uses exact archived StreamGraph + source byte pins from GitHub
owner root; reconstructs centers for every selected policy then computes the
actual Chapter-3 R/I/T/BT/TNL/BTNL diagnostics on Train reference samples.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,resource,sys,tempfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE/"runner_v3"))
sys.path.insert(0,str(BASE/"analysis_v3"))
from data_contract import owner_root,decoded_block,attested_sources,sha,require
from run_one import init_modules,check_attested_modules
from aggregate import geometry,kl,jsd,read,emit,TOL

BLOCKS={
"B01":"INV01-B01-n150-data20261102",
"B02":"INV01-B02-n300-data20261102",
"B03":"INV01-B03-n150-data20261103",
"B04":"INV01-B04-n300-data20261103",
"B05":"INV01-B05-n150-data20261104",
"B06":"INV01-B06-n300-data20261104"
}

def peakrss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
def block_inputs(block,root,captured,mods,np):
    if block!="B01":
        return decoded_block(root,block)
    # Historical B01 v2: regenerate TRAIN only, verify EXACT frozen hashes.
    manifest=read(BASE/"standard_plan/results/B01_v2/validated_run_01/manifest.json")
    cfg=mods["generate"].load_config(ROOT/"code/data_generator/config.json")
    require(cfg["seed"]==20261102,"B01_CONFIG_SEED_DRIFT")
    with tempfile.TemporaryDirectory() as tmp:
        data=mods["generate"].generate(cfg,tmp,finite_q=False)
    fullX,fully=data["train"][0],data["train"][2]
    hd=lambda x:sha(np.ascontiguousarray(x).tobytes())
    require(hd(fullX)==manifest["source_train_X_sha256"],"B01_TRAIN_X_DRIFT")
    require(hd(fully)==manifest["source_train_y_sha256"],"B01_TRAIN_Y_DRIFT")
    rng=np.random.default_rng(20261201)
    indices=np.concatenate([rng.choice(np.flatnonzero(fully==c),50,replace=False) for c in range(3)])
    rng.shuffle(indices)
    require(hd(indices)==manifest["selected_reference_ids_sha256"],"B01_REFERENCE_ID_DRIFT")
    x,y=fullX[indices],fully[indices]
    require(np.bincount(y,minlength=3).tolist()==[50,50,50],"B01_BALANCED_COUNT_DRIFT")
    tar=np.asarray([row["target"] for row in manifest["targets"]],dtype="<f8")
    require(tar.shape==(3,3) and bool(np.isfinite(tar).all()),"B01_TARGET_INVALID")
    require(all(hd(tar[c])==manifest["targets"][c]["target_sha256"] for c in range(3)),
            "B01_TARGET_ROW_DRIFT")
    return {"X":x,"y":y,"mask":np.isfinite(x),"ids":indices,
            "targets":tar,"manifest":{"class_counts":[50,50,50]},
            "block":{"block_id":BLOCKS["B01"],"nref":150,"data_seed":20261102},
            "input_sha256":{"source_train_X":hd(fullX),"selected_reference_ids":hd(indices)}}

def accepted_raw(block,plan):
    bid=BLOCKS[block]
    out={}
    if block=="B01":
        summary=read(BASE/"standard_plan/results/B01_v2/validated_run_01/summary.json")
        assert summary["status"]=="completed" and len(summary["runs"])==18
        for r in summary["runs"]:
            key=(r["class"],r["start"],r["solver"])
            assert key not in out
            out[key]={"selected_policy_logits":r["selected_policy_logits"],
                      "selected_exact_center_JSD":r["selected_exact_center_JSD"],
                      "evidence_status":"ACCEPT_B01_V2"}
    else:
        for row in plan["runs"]:
            if row["block_id"]!=bid:continue
            p=BASE/"results/raw"/bid/row["run_id"]
            files=list(p.glob("*/summary.json"))
            require(len(files)==1,"MISSING_OR_DUPLICATE_RUN:"+row["run_id"])
            r=read(files[0])
            require(r.get("status")=="COMPLETED" and r["run_id"]==row["run_id"],
                    "RAW_NOT_COMPLETED")
            key=(row["class_index"],row["initialization"],row["solver"])
            require(key not in out,"DUPLICATE_CONTEXT")
            out[key]=r
    require(len(out)==18,"EIGHTEEN_SELECTED_POLICIES_REQUIRED")
    return out

def execute(block,root_id):
    import numpy as np,torch
    start=time.perf_counter()
    root,root_sha=owner_root(root_id)
    captured=attested_sources(root)
    np,torch,mods=init_modules(captured)
    require({"python":sys.version.split()[0],"numpy":np.__version__,"torch":torch.__version__}==root["runtime"],
            "ANALYSIS_RUNTIME")
    inputs=block_inputs(block,root,captured,mods,np)
    check_attested_modules(captured,mods)
    cfg=mods["generate"].load_config(ROOT/"code/data_generator/config.json")
    x,y,targets=inputs["X"],inputs["y"],inputs["targets"]
    torch.set_num_threads(1)
    graph=mods["stream_graph"].StreamGraph(x,y,cfg["gamma"],validation_x=None,beta=.4,lam=.8,horizon=12)
    plan=read(BASE/"standard_plan/plan_108.json")
    raw=accepted_raw(block,plan)
    combinations=[]
    for method in ("Q-only","Exact-Adam","Q+Bootstrap-GN"):
        for init in ("zero","independent_random"):
            matrix=[];rkl=[];rjsd=[];rsqrt=[];class_details=[]
            for c in range(3):
                row=raw[(c,init,method)]
                logits=np.asarray(row["selected_policy_logits"],dtype=np.float64)
                require(logits.shape==(graph.d+1,graph.d),"LOGITS_SHAPE")
                idx=np.flatnonzero(y==c)
                with torch.no_grad():
                    per=graph.center_train(torch.as_tensor(logits,dtype=torch.float64),
                                           idx,return_all=True).detach().cpu().numpy()
                ctr=per.mean(axis=0)
                observed=float(mods["run"].jsd(torch.as_tensor(ctr,dtype=torch.float64),
                                               torch.as_tensor(targets[c],dtype=torch.float64)).item())
                saved=float(row["selected_exact_center_JSD"])
                require(abs(observed-saved)<=1e-12,
                        "EXACT_REPLAY_MISMATCH:%s:%s:%s"%(block,method,c))
                for p in per:
                    require(abs(float(sum(p))-1)<1e-8 and bool(np.isfinite(p).all()),"SAMPLE_SIMPLEX")
                dens_kl=[kl(p,ctr) for p in per]
                dens_js=[jsd(p,ctr) for p in per]
                dens_sq=[math.sqrt(d) for d in dens_js]
                require(all(d is not None and finite_nonneg(d) for d in dens_kl+dens_js+dens_sq),
                        "DENSITY_METRIC_UNDEFINED")
                rkl.append(sum(dens_kl)/len(dens_kl))
                rjsd.append(sum(dens_js)/len(dens_js))
                rsqrt.append(sum(dens_sq)/len(dens_sq))
                matrix.append([float(a) for a in ctr])
                class_details.append({"class":c,"n":len(per),"replayed_center":matrix[-1],
                  "exact_center_JSD":observed,"raw_center_JSD":saved,
                  "R_KL":rkl[-1],"R_JSD":rjsd[-1],"R_sqrt_JSD":rsqrt[-1]})
            stats=geometry(matrix)
            stats.update({
                "block":block,"method":method,"initialization":init,
                "matrix_of_reconstructed_class_centers":matrix,
                "R_KL_uniform_class":sum(rkl)/3,"R_JSD_uniform_class":sum(rjsd)/3,
                "R_sqrt_JSD_uniform_class":sum(rsqrt)/3,
                "R_KL_empirical":sum(rkl[c]*len(np.flatnonzero(y==c)) for c in range(3))/len(y),
                "per_class":class_details,
                "target_geometry":geometry([[float(z) for z in target] for target in targets]),
                "evidence_status":"PROVISIONAL_PENDING_FINAL_QA",
                "no_solver_optimizer_or_Test":True
            })
            combinations.append(stats)
            require(peakrss()<1024,"RESOURCE_RSS_LIMIT")
            require(time.perf_counter()-start<600,"RESOURCE_WALL_LIMIT")
    out=BASE/"results/reconstructed"/block
    out.mkdir(parents=True,exist_ok=True)
    target=out/"metrics.json"
    require(not target.exists(),"EXISTING_RECONSTRUCTED_METRICS")
    output={"schema":"R05_SEC410_RECONSTRUCTED_CENTER_METRICS_V3",
            "block":block,"nref":len(y),"data_seed":inputs["block"]["data_seed"],
            "owner_root_comment_id":root_id,"root_sha256":root_sha,
            "source_shas":{k:sha(v) for k,v in captured.items()},
            "input_shas":inputs["input_sha256"],
            "class_counts":[int(sum(y==c)) for c in range(3)],
            "status":"PASS_INTERNAL_EXACT_REPLAY",
            "combinations":combinations,"method_count":3,"init_count":2,
            "class_count":3,"replayed_policy_count":18,
            "wall_seconds":time.perf_counter()-start,"peak_rss_mib":peakrss(),
            "never_read_Test_or_Validation":True,
            "evidence_status":"PROVISIONAL_PENDING_FINAL_QA"}
    emit(target,output)
    print("R05_RECONSTRUCTED_CENTER_REPLAY",json.dumps({"block":block,
          "replayed_policies":18,"wall":output["wall_seconds"],
          "rss":output["peak_rss_mib"],"status":output["status"]}),flush=True)
def finite_nonneg(v):return v is not None and math.isfinite(v) and v>=-1e-12
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--block",choices=tuple(BLOCKS),required=True)
    ap.add_argument("--owner-root-comment",type=int,default=6853301767)
    args=ap.parse_args()
    execute(args.block,args.owner_root_comment)
