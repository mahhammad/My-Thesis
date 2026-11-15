#!/usr/bin/env python3
"""R05 §4.10 — bounded standard-plan Train-only snapshot builder, NO SOLVERS.

New versioned artifacts only. Never overwrites B02 historical canonical input.
A snapshot is not an admission to an optimization run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / "code/experiments/inverse-attainable/standard_plan/plan_108.json"
CFG = ROOT / "code/data_generator/config.json"
SOURCE_PATHS = {
    "config": CFG,
    "generator": ROOT / "code/data_generator/generate.py",
    "streamgraph": ROOT / "code/experiments/benchmark_v1/stream_graph.py",
    "graph_run": ROOT / "code/experiments/benchmark_v1/run.py",
    "plan": PLAN,
}
BLOCKS = {
    "B03": "INV01-B03-n150-data20261103",
    "B04": "INV01-B04-n300-data20261103",
    "B05": "INV01-B05-n150-data20261104",
    "B06": "INV01-B06-n300-data20261104",
}
MAGIC = b"INV01_TRAIN_REFERENCE_V2\n"
BUDGET = 120.0
RSS_HARD_MIB = 1024.0

def digest(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()

def encode_json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")

def safe_write(path: Path, content: bytes):
    if path.exists():
        raise FileExistsError("Never overwrite evidence: " + str(path))
    temp = path.with_name(path.name + ".partial")
    with temp.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_one(short: str):
    if sys.flags.optimize:
        raise RuntimeError("python optimized mode forbidden")
    block_id = BLOCKS[short]
    out = HERE / "results" / block_id
    out.mkdir(parents=True, exist_ok=True)
    required_files = ["train_canonical.bin", "targets_float64_le.bin",
                      "manifest.json", "verification.json"]
    if any((out / n).exists() for n in required_files):
        raise FileExistsError("Existing snapshot; never regenerate " + block_id)
    clock = time.monotonic()
    stop = threading.Event()
    peak = [0.0]

    def guard(stage):
        peak[0] = max(peak[0], rss_mib())
        if peak[0] >= RSS_HARD_MIB:
            raise MemoryError("RSS cap " + stage)
        if time.monotonic() - clock >= BUDGET:
            raise TimeoutError("Wall cap " + stage)

    def watch():
        while not stop.wait(.1):
            if rss_mib() >= RSS_HARD_MIB or time.monotonic() - clock >= BUDGET:
                (out / "resource_hard_stop.txt").write_text(
                    "RESOURCE_STOP; NO SOLVE; snapshot incomplete\n", encoding="utf8")
                os._exit(124)

    watch_thread = threading.Thread(target=watch, daemon=True)
    watch_thread.start()
    try:
        import numpy as np
        import torch
        for p in SOURCE_PATHS.values():
            if not p.is_file() or p.is_symlink():
                raise RuntimeError("Source path missing or symlink: " + str(p))
        # Stage exact source roots before importing legacy modules (no optimizer imported).
        source_hash = {k: digest(p.read_bytes()) for k,p in SOURCE_PATHS.items()}
        sys.path[0:0] = [str(ROOT / "code/data_generator"),
                         str(ROOT / "code/experiments/benchmark_v1"),
                         str(ROOT / "code/experiments/benchmark_v1")]
        from generate import generate, load_config
        from stream_graph import StreamGraph
        from run import DT, jsd
        plan = json.loads(PLAN.read_text(encoding="utf8"))
        matching = [b for b in plan["blocks"] if b["block_id"] == block_id]
        if len(matching) != 1:
            raise RuntimeError("Block absent/duplicate in frozen plan")
        b = matching[0]
        nref = b["nref"]
        if nref not in (150, 300) or b["data_seed"] not in (20261103,20261104):
            raise RuntimeError("Wrong plan block")
        rows = [r for r in plan["runs"] if r["block_id"] == block_id]
        triplets = {(c, init, method) for c in range(3)
                    for init in ("zero", "independent_random")
                    for method in ("Q-only", "Exact-Adam", "Q+Bootstrap-GN")}
        if (len(rows) != 18 or len({r["run_id"] for r in rows}) != 18
                or {(r["class_index"],r["initialization"],r["solver"]) for r in rows} != triplets
                or b["authorized_to_execute"] is not False
                or any(r["state"] != "NOT_RUN" for r in rows)):
            raise RuntimeError("G17 plan coverage/authorization mismatch")
        row_ref = b["reference_seed"] if nref == 150 else None
        if any(r["reference_seed"] != row_ref for r in rows):
            raise RuntimeError("G17: n300 full Train reference_seed must be null")
        if any((r["nref"],r["data_seed"],r["teacher_seed"]) !=
               (nref,b["data_seed"],b["teacher_seed"]) for r in rows):
            raise RuntimeError("G17 seed/size drift")
        if any((r["beta"],r["lambda"],r["H"],r["precision"],
                r["exact_JSD_success_tol"]) != (.4,.8,12,"float64",1e-8)
               for r in rows):
            raise RuntimeError("Walk parameter/tolerance drift")
        guard("plan")
        spec = load_config(CFG)
        if spec["sizes"]["train"] != 300:
            raise RuntimeError("Source full Train size is no longer 300")
        spec["seed"] = b["data_seed"]  # explicit plan seed; historical config unchanged
        # Generator writes other historical splits, but we only ACCESS the Train entry.
        with tempfile.TemporaryDirectory() as tmp:
            Xfull, maskfull, yfull = generate(spec, tmp, finite_q=False)["train"]
            Xfull, maskfull, yfull = Xfull.copy(), maskfull.copy(), yfull.copy()
        guard("Train only")
        if (Xfull.shape != (300,5) or maskfull.shape != (300,5)
                or yfull.shape != (300,) or not np.array_equal(maskfull,np.isfinite(Xfull))):
            raise RuntimeError("Train shape/mask mismatch")
        if nref == 150:
            rng = np.random.default_rng(b["reference_seed"])
            ids = np.concatenate([rng.choice(np.flatnonzero(yfull==c),50,replace=False)
                                  for c in range(3)])
            rng.shuffle(ids)
        else:
            ids = np.arange(300)
        X, mask, y = Xfull[ids], maskfull[ids], yfull[ids]
        if (len(set(ids.tolist())) != nref or len(ids) != nref
                or (nref==150 and np.bincount(y,minlength=3).tolist() != [50,50,50])):
            raise RuntimeError("Invalid reference IDs or class coverage")
        z = np.array(X,dtype="<f8",order="C",copy=True)
        z[~mask] = np.float64(np.nan)
        yy = np.ascontiguousarray(y,dtype="<i8")
        mm = np.ascontiguousarray(mask.astype("u1"))
        ii = np.ascontiguousarray(ids,dtype="<i8")
        blobparts = {"X":z.tobytes(order="C"),"y":yy.tobytes(),"mask":mm.tobytes(),"ids":ii.tobytes()}
        raw = MAGIC + b"".join(blobparts[k] for k in ("X","y","mask","ids"))
        k = len(MAGIC)
        parsed = {}
        for key,dtype,shape in (("X","<f8",(nref,5)),("y","<i8",(nref,)),
                                ("mask","u1",(nref,5)),("ids","<i8",(nref,))):
            data = blobparts[key]
            parsed[key] = np.frombuffer(raw[k:k+len(data)],dtype=dtype).reshape(shape).copy()
            k += len(data)
        if (k != len(raw) or not np.array_equal(parsed["ids"],ii)
                or not np.array_equal(parsed["mask"],mm)
                or not np.array_equal(np.isnan(parsed["X"]),np.isnan(z))
                or not np.array_equal(parsed["X"][mask],z[mask])):
            raise RuntimeError("Canonical roundtrip failure")
        torch.set_num_threads(1)
        graph = StreamGraph(X,y,spec["gamma"],validation_x=None,beta=.4,lam=.8,horizon=12)
        witness = np.random.default_rng(b["teacher_seed"]).normal(
            0,1.7,size=(3,graph.d+1,graph.d))
        witness -= witness.mean(axis=-1,keepdims=True)
        witness_digest = digest(np.ascontiguousarray(witness).tobytes())
        center = []
        for c in range(3):
            with torch.no_grad():
                v = graph.center_train(torch.as_tensor(witness[c],dtype=DT),
                                       np.flatnonzero(y==c)).detach().cpu().numpy()
            if v.shape != (3,) or not np.isfinite(v).all() or (v<0).any() or abs(sum(v)-1)>1e-10:
                raise RuntimeError("Invalid attainable target")
            center.append(np.array(v,dtype="<f8"))
            guard("target forward")
        del witness,graph
        targets = np.stack(center).astype("<f8",copy=False).tobytes(order="C")
        counts = np.bincount(y,minlength=3).tolist()
        manifest = {
            "schema":"INV01_TRAIN_REFERENCE_V2","status":"LOCKED_NO_SOLVERS",
            "block_id":block_id,"nref":nref,"data_seed":b["data_seed"],
            "reference_selection_seed":row_ref,
            "reference_seed_label_in_block":b["reference_seed"],
            "teacher_seed":b["teacher_seed"],
            "source_commit":os.environ.get("GITHUB_SHA","LOCAL_UNATTESTED"),
            "source_sha256":source_hash,
            "planned_run_ids":[r["run_id"] for r in rows],
            "id_semantics":"original Train indices, order preserved",
            "class_counts":counts,
            "observed_per_feature":mask.sum(axis=0).tolist(),
            "snapshot":{"file":"train_canonical.bin","magic":MAGIC.decode().strip(),
                        "sha256":digest(raw),"shape":[nref,5],
                        "array_sha256":{k:digest(v) for k,v in blobparts.items()},
                        "canonical_little_endian":True},
            "targets":{"file":"targets_float64_le.bin","sha256":digest(targets),
                       "per_class":[{"class":c,"values":center[c].tolist(),
                                     "sha256":digest(center[c].tobytes())} for c in range(3)]},
            "witness_logits_sha256_ONLY":witness_digest,
            "witness_given_to_solver":False,
            "beta":.4,"lambda":.8,"H":12,"precision":"float64",
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},
            "resource":{"hard_wall_s":BUDGET,"hard_RSS_MiB":RSS_HARD_MIB},
            "forward_count":3,"jacobian_count":0,"solver_count":0,
        }
        guard("before write")
        safe_write(out/"train_canonical.bin",raw)
        safe_write(out/"targets_float64_le.bin",targets)
        safe_write(out/"manifest.json",encode_json(manifest))
        verify={"status":"PASS_SNAPSHOT_ONLY","block":block_id,
                "train_source_hashes_verified":True,
                "canonical_roundtrip_verified":True,
                "target_sha256":digest(targets),"snapshot_sha256":digest(raw),
                "forwards":3,"solves":0,
                "wall_seconds":time.monotonic()-clock,
                "peak_RSS_MiB":max(peak[0],rss_mib())}
        safe_write(out/"verification.json",encode_json(verify))
        print("R05_SNAPSHOT_OK",json.dumps(verify,sort_keys=True),flush=True)
    except BaseException as exc:
        # Never pretend a partially written snapshot is valid.
        failure={"status":"FAILED","block_id":block_id,"error":str(exc),
                 "wall_seconds":time.monotonic()-clock,"peak_RSS_MiB":max(peak[0],rss_mib()),
                 "solves":0}
        try: safe_write(out/"failure.json",encode_json(failure))
        except FileExistsError: pass
        raise
    finally:
        stop.set()
        watch_thread.join(timeout=2)

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in BLOCKS:
        raise SystemExit("Usage: python build_standard_snapshots.py B03|B04|B05|B06")
    run_one(sys.argv[1])
