"""R05 §4.10: fail-closed, non-model data and GitHub-owner trust contract.

This module itself imports only Python's standard library and NEVER admits a
graph or solver. The owner-trust authority is an immutable-by-policy Issue #60
comment, authenticated via GitHub HTTPS API and owner metadata, OUTSIDE the
mutable experiment checkout. See README for threat-model limitations.
"""
from __future__ import annotations
import hashlib, json, os, stat, struct, urllib.request
from pathlib import Path

REPO = "mahhammad/My-Thesis"
ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT/"code/experiments/inverse_attainable_standard"
PLAN = EXPERIMENT/"standard_plan/plan_108.json"
BASE = EXPERIMENT/"snapshot_v2/results"
HIST = EXPERIMENT/"b02_preflight"
MAGIC_N = b"INV01_TRAIN_REFERENCE_V2\n"
MAGIC_B02 = b"B02TRAIN_CANONICAL_V1\n"
SNAP_NAMES = {
 "B02":"INV01-B02-n300-data20261102",
 "B03":"INV01-B03-n150-data20261103",
 "B04":"INV01-B04-n300-data20261103",
 "B05":"INV01-B05-n150-data20261104",
 "B06":"INV01-B06-n300-data20261104",
}
SOURCE_PATHS = {
 "config":"code/data_generator/config.json",
 "generator":"code/data_generator/generate.py",
 "graph_run":"code/experiments/multistep_policy_target_match/run.py",
 "streamgraph":"code/experiments/diag_max_full300/stream_graph.py",
 "bootstrap":"code/experiments/multistep_policy_target_match/bootstrap_inverse.py",
 "p2_instrumentation":"code/experiments/inverse_attainable_standard/standard_plan/p2_v2_instrumentation.py",
 "plan":"code/experiments/inverse_attainable_standard/standard_plan/plan_108.json",
 "runner":"code/experiments/inverse_attainable_standard/runner_v3/run_one.py",
 "contract":"code/experiments/inverse_attainable_standard/runner_v3/data_contract.py"
}

class Deny(RuntimeError):
    def __init__(self,code):
        self.code=code
        super().__init__(code)

def require(ok,code):
    if not ok:
        raise Deny(code)

def sha(x:bytes):
    return hashlib.sha256(x).hexdigest()

def gitblob(x:bytes):
    return hashlib.sha1(b"blob "+str(len(x)).encode()+b"\x00"+x).hexdigest()

def strictjson(raw:bytes):
    def pairs(xs):
        d={}
        for k,v in xs:
            require(k not in d,"DUPLICATE_JSON_KEY")
            d[k]=v
        return d
    def bad(x):
        raise Deny("NONFINITE_JSON")
    return json.loads(raw.decode("utf-8"),object_pairs_hook=pairs,parse_constant=bad)

def read_locked(path:Path):
    # Refuse symlink, non-regular file, path escaping repository.
    root=ROOT.resolve(strict=True)
    path=path.absolute()
    require(path.is_relative_to(root),"PATH_OUTSIDE_REPO")
    cur=path
    while cur!=root:
        require(not cur.is_symlink(),"SYMLINK_SOURCE")
        cur=cur.parent
    fd=os.open(str(path),os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0))
    try:
        st=os.fstat(fd)
        require(stat.S_ISREG(st.st_mode),"NONREGULAR_SOURCE")
        with os.fdopen(fd,"rb",closefd=False) as f:
            data=f.read()
        require(os.fstat(fd).st_ino==st.st_ino and os.fstat(fd).st_size==len(data),
                "SOURCE_CHANGED_DURING_READ")
        return bytes(data)
    finally:
        os.close(fd)

def owner_root(comment_id:int):
    require(isinstance(comment_id,int) and comment_id>0,"NO_ROOT_COMMENT")
    api=f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}"
    token=os.environ.get("GITHUB_TOKEN")
    require(bool(token),"NO_GITHUB_READ_TOKEN")
    req=urllib.request.Request(api,headers={
        "Authorization":"Bearer "+token,"Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28","User-Agent":"R05-Sec410-attested-runner"})
    with urllib.request.urlopen(req,timeout=15) as r:
        raw=r.read(100000)
    j=strictjson(raw)
    require(j.get("user",{}).get("login")=="mahhammad" and
            j.get("author_association")=="OWNER","UNTRUSTED_COMMENT_AUTHOR")
    require(j.get("created_at")==j.get("updated_at"),"EDITED_ROOT_COMMENT")
    body=j.get("body","")
    prefix="# [R05][SEC410-OWNER-TRUST-ROOT-V1]"
    require(body.startswith(prefix),"ROOT_SCHEMA_HEADER")
    require(body.count("```json")==1 and body.count("```")==2,"ROOT_FENCE")
    payload=body.split("```json",1)[1].split("```",1)[0].strip()
    root=strictjson(payload.encode())
    require(root.get("schema")=="R05_SEC410_OWNER_TRUST_V1" and root.get("status")=="FROZEN",
            "ROOT_SCHEMA")
    require(root.get("original_authorization_comment_id")==6852050071,"WRONG_MANDATE")
    require(j.get("id")==comment_id,"ROOT_ID_MISMATCH")
    return root,sha(payload.encode("utf-8"))

def validate_plan(root,block):
    p=strictjson(read_locked(PLAN))
    require(sha(read_locked(PLAN))==root["source_sha256"]["plan"],"PLAN_SHA")
    require(p.get("schema")=="CH4-EXP-INV-01-P3-proposed-plan-v1","PLAN_SCHEMA")
    require(len(p["runs"])==108 and len(p["blocks"])==6,"PLAN_CARDINALITY")
    bids=[b for b in p["blocks"] if b["block_id"]==SNAP_NAMES[block]]
    require(len(bids)==1,"BLOCK_NOT_UNIQUE")
    b=bids[0]
    require(b["authorized_to_execute"] is False,"HISTORICAL_PLAN_MUTATED")
    rows=[r for r in p["runs"] if r["block_id"]==b["block_id"]]
    require(len(rows)==18 and len({r["run_id"] for r in rows})==18,"RUN_MATRIX_18")
    combos={(c,i,m) for c in range(3)
            for i in ("zero","independent_random")
            for m in ("Q-only","Exact-Adam","Q+Bootstrap-GN")}
    require({(r["class_index"],r["initialization"],r["solver"]) for r in rows}==combos,
            "RUN_MATRIX_TRIPLETS")
    require(all(r["state"]=="NOT_RUN" for r in rows),"HISTORICAL_PLAN_RUN_STATE")
    require(all(r["reference_seed"]==(b["reference_seed"] if b["nref"]==150 else None)
                for r in rows),"G17_REFERENCE_SEMANTICS")
    require(all((r["beta"],r["lambda"],r["H"],r["precision"],r["exact_JSD_success_tol"])==
                (.4,.8,12,"float64",1e-8) for r in rows),"WALK_DRIFT")
    return b,rows

def decoded_block(root,block):
    require(block in SNAP_NAMES and block in root["blocks"],"BLOCK_NOT_AUTHORIZED")
    name=SNAP_NAMES[block]
    path=HIST if block=="B02" else BASE/name
    mbytes=read_locked(path/"manifest.json")
    snap=read_locked(path/"train_canonical.bin")
    targets=read_locked(path/"targets_float64_le.bin")
    pins=root["blocks"][block]
    require(gitblob(mbytes)==pins["manifest_blob_sha1"],"MANIFEST_GIT_BLOB_SHA")
    require(sha(snap)==pins["snapshot_sha256"],"SNAPSHOT_OWNER_SHA")
    require(sha(targets)==pins["targets_sha256"],"TARGET_OWNER_SHA")
    m=strictjson(mbytes)
    require(m.get("nref")==pins["nref"],"SNAPSHOT_NREF")
    n=pins["nref"]
    import numpy as np
    if block=="B02":
        require(m.get("schema")=="B02_TRAIN_AND_TARGET_V1","SNAPSHOT_SCHEMA_B02")
        require(snap.startswith(MAGIC_B02),"SNAPSHOT_MAGIC_B02")
        require(m["snapshot"]["file_sha256"]==sha(snap) and
                m["targets"]["file_sha256"]==sha(targets),"SNAPSHOT_META_SHA_B02")
        require(m.get("status")=="LOCKED_NO_SOLVERS","SNAPSHOT_STATUS")
        start=len(MAGIC_B02)
    else:
        require(m.get("schema")=="INV01_TRAIN_REFERENCE_V2" and
                m.get("status")=="LOCKED_NO_SOLVERS","SNAPSHOT_SCHEMA_V2")
        require(snap.startswith(MAGIC_N),"SNAPSHOT_MAGIC_NEW")
        require(m["snapshot"]["sha256"]==sha(snap) and
                m["targets"]["sha256"]==sha(targets),"SNAPSHOT_META_SHA")
        require(m["source_sha256"]==root["snapshot_source_sha256"],"SOURCE_DIGEST_DRIFT")
        require(m["reference_selection_seed"]==pins["reference_seed"],"SNAPSHOT_REF_SEED")
        start=len(MAGIC_N)
    chunks=[(n*5*8,"<f8",(n,5),"X"),(n*8,"<i8",(n,),"y"),
            (n*5,"u1",(n,5),"mask"),(n*8,"<i8",(n,),"ids")]
    require(len(snap)==start+sum(k[0] for k in chunks),"SNAPSHOT_SIZE")
    arrays={}
    offset=start
    for count,dtype,shape,key in chunks:
        raw=snap[offset:offset+count]
        arrays[key]=np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
        require(sha(raw)==(m["snapshot"]["array_sha256"][key]
                          if block=="B02" else m["snapshot"]["array_sha256"][key]),
                "ARRAY_SHA_"+key.upper())
        offset+=count
    X,y,mask,ids=(arrays[k] for k in ("X","y","mask","ids"))
    require(set(np.unique(mask).tolist()) <= {0,1},"INVALID_MASK")
    require(np.array_equal(mask.astype(bool),np.isfinite(X)),"MASK_X_DRIFT")
    require(not np.isinf(X).any(),"INFINITE_FEATURE")
    bits=X.view("<u8")[mask==0]
    require(bool((bits==0x7ff8000000000000).all()),"NONCANONICAL_NAN")
    require(bool(((y>=0)&(y<3)).all()),"INVALID_LABEL")
    require(len(set(ids.tolist()))==n and bool(((ids>=0)&(ids<300)).all()),"INVALID_IDS")
    if n==300: require(np.array_equal(ids,np.arange(300)),"FULL_TRAIN_ORDER")
    if n==150: require(np.bincount(y,minlength=3).tolist()==[50,50,50],
                       "BALANCED_REFERENCE")
    require(np.bincount(y,minlength=3).tolist()==m["class_counts"],"CLASS_COUNTS")
    require(np.array_equal(mask.sum(axis=0),m["observed_per_feature"]),"OBSERVED_COUNTS")
    require(len(targets)==72,"TARGET_LENGTH")
    tgt=np.frombuffer(targets,dtype="<f8").reshape((3,3)).copy()
    require(bool(np.isfinite(tgt).all() and (tgt>=0).all() and
                  np.all(abs(tgt.sum(axis=1)-1)<1e-10)),"INVALID_TARGET_SIMPLEX")
    perclass=m["targets"]["per_class"]
    for c in range(3):
        row=targets[c*24:(c+1)*24]
        require(sha(row)==perclass[c]["sha256"],"TARGET_ROW_SHA")
        vec=perclass[c].get("vector",perclass[c].get("values"))
        require(struct.pack("<3d",*vec)==row,"TARGET_JSON_BINARY_DRIFT")
    b,rows=validate_plan(root,block)
    require((b["nref"],b["data_seed"],b["teacher_seed"])==
            (n,pins["data_seed"],pins["teacher_seed"]),"BLOCK_PLAN_DRIFT")
    if block=="B02":
        require(m["seed"]["data"]==pins["data_seed"] and
                m["seed"]["teacher"]==pins["teacher_seed"],"B02_SEED_DRIFT")
    else:
        require(m["data_seed"]==pins["data_seed"] and
                m["teacher_seed"]==pins["teacher_seed"],"SEED_DRIFT")
        require(m["planned_run_ids"]==[r["run_id"] for r in rows],"PLAN_ORDER_DRIFT")
    for ar in (X,y,mask,ids,tgt):
        ar.flags.writeable=False
    return {"X":X,"y":y,"mask":mask,"ids":ids,"targets":tgt,
            "manifest":m,"plan_rows":rows,"block":b,
            "input_sha256":{"snapshot":sha(snap),"targets":sha(targets),"manifest":sha(mbytes)}}

def attested_sources(root):
    pins=root["source_sha256"]
    bl= root["source_blobs_sha1"]
    captured={}
    for name,p in SOURCE_PATHS.items():
        b=read_locked(ROOT/p)
        if name in pins: require(sha(b)==pins[name],"SOURCE_SHA_"+name)
        if name in bl: require(gitblob(b)==bl[name],"SOURCE_BLOB_"+name)
        captured[name]=b
    return captured
