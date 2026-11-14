"""B02 fail-closed byte/semantic gate CANDIDATE. No model or data generator imports.

This module NEVER authorizes a graph, optimization, or resource probe.  Any
future admission mechanism requires separate management authority and QA.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct
import sys
from typing import Mapping

MAGIC = b"B02TRAIN_CANONICAL_V1\n"
BLOCK = "INV01-B02-n300-data20261102"
SNAPSHOT_COMMIT = "05acde39b76b49e0b55337301251594be0ad6d92"
PLAN_COMMIT = "968675cde8783adf69db13efd7de0fb3aab703c5"
CH4_COMMIT = "de02fdda91e23db177de4429b3e1a9693c046d83"
SOURCES = {
    "config": "code/data_generator/config.json",
    "generator": "code/data_generator/generate.py",
    "graph_run": "code/experiments/archive/multistep_policy_target_match/run.py",
    "streamgraph": "code/experiments/archive/diag_max_full300/stream_graph.py",
    "plan": "code/experiments/inverse_attainable_standard/standard_plan/plan_108.json",
}
CHUNKS = {"X": (22,12000), "y": (12022,2400), "mask": (14422,1500), "ids": (15922,2400)}
METHODS = {"Q-only":"Q", "Exact-Adam":"A", "Q+Bootstrap-GN":"G"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class GateError(Exception):
    def __init__(self, code: str, stage: str, explanation: str = ""):
        self.code, self.stage, self.explanation = code, stage, explanation
        super().__init__(f"{stage}:{code}: {explanation}")


def need(ok: bool, code: str, stage: str, detail: str = "") -> None:
    if not ok:
        raise GateError(code, stage, detail)


def _json(raw: bytes, name: str):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            need(key not in obj, "E_SCHEMA", name, "duplicate JSON key")
            obj[key] = value
        return obj
    def reject_constant(value):
        raise GateError("E_SCHEMA", name, "non-finite JSON constant")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise GateError("E_SCHEMA", name, "invalid JSON encoding/structure") from exc


def _path(root: Path, relative: str, *, code: str = "E_UNSAFE_PATH") -> Path:
    need(isinstance(relative, str) and relative and not Path(relative).is_absolute(), code, "G00", "absolute/empty path")
    need(".." not in Path(relative).parts, code, "G00", "path traversal")
    root = root.resolve(strict=True)
    candidate = root / relative
    need(not any(part.is_symlink() for part in [candidate, *list(candidate.parents)[:len(Path(relative).parts)] ]), code, "G00", "symlink")
    resolved = candidate.resolve(strict=True)
    need(resolved.is_relative_to(root) and resolved.is_file(), code, "G00", "unsafe source file")
    return resolved


def read_pinned_bundle(snapshot_dir: Path, code_root: Path, *, refs: Mapping[str,str]):
    """Local loader for an external checkout already checked at the pinned SHAs.

    Actual git-object provenance must be established by the trusted CI checkout;
    refs alone are *not* proof of provenance. Content hashes below are mandatory.
    """
    need(refs.get("snapshot") == SNAPSHOT_COMMIT and refs.get("plan") == PLAN_COMMIT and
         refs.get("chapter4") == CH4_COMMIT, "E_COMMIT_PIN", "G00")
    snap = Path(snapshot_dir).resolve(strict=True)
    root = Path(code_root).resolve(strict=True)
    return Bundle(
        manifest=_path(snap,"manifest.json").read_bytes(),
        snapshot=_path(snap,"train_canonical.bin").read_bytes(),
        targets=_path(snap,"targets_float64_le.bin").read_bytes(),
        plan=_path(root,SOURCES["plan"]).read_bytes(),
        source_bytes={k: _path(root,p).read_bytes() for k,p in SOURCES.items()},
        source_paths={k: _path(root,p) for k,p in SOURCES.items()},
        refs=dict(refs),
    )


@dataclass(frozen=True)
class Bundle:
    manifest: bytes
    snapshot: bytes
    targets: bytes
    plan: bytes
    source_bytes: Mapping[str, bytes]
    source_paths: Mapping[str, Path]
    refs: Mapping[str, str]


@dataclass(frozen=True)
class TrustedSpec:
    """Must arrive from a management-controlled root independent of checkout."""
    manifest_sha256: str
    snapshot_sha256: str
    targets_sha256: str
    array_sha256: Mapping[str, str]
    source_sha256: Mapping[str, str]
    approved_runtime: Mapping[str, str]
    snapshot_commit: str = SNAPSHOT_COMMIT
    plan_commit: str = PLAN_COMMIT
    chapter4_commit: str = CH4_COMMIT


@dataclass(frozen=True)
class VerifiedB02Inputs:
    X: tuple[tuple[float, ...], ...]
    y: tuple[int, ...]
    mask: tuple[tuple[int, ...], ...]
    ids: tuple[int, ...]
    targets: tuple[tuple[float, ...], ...]
    snapshot_sha256: str
    targets_sha256: str
    # Only immutable value objects / byte payloads cross the verifier boundary.
    # Never retain mutable Bundle.mapping or live Path objects for subsequent use.
    source_pins: tuple[tuple[str, str, str, bytes], ...] = ()
    import_attestation: tuple[tuple[str, str, str], ...] = ()
    import_path_at_verify: tuple[str, ...] = ()
    meta_finder_ids_at_verify: tuple[int, ...] = ()
    trust_manifest_sha256: str = ""


def _required_manifest(m):
    need(isinstance(m,dict) and m.get("schema")=="B02_TRAIN_AND_TARGET_V1" and
         m.get("block")==BLOCK and m.get("status")=="LOCKED_NO_SOLVERS" and
         type(m.get("nref")) is int and m["nref"]==300 and
         m.get("three_classes") is True and m.get("contexts")==18,
         "E_SCHEMA","G01","schema, block or numeric shape")
    need(m.get("authorized_to_solve") is False and m.get("teacher_given_to_solver") is False,
         "E_SOLVER_API_BOUNDARY","G18","historical admission / witness flag")
    need(m.get("chapter4_sha")==CH4_COMMIT and m.get("reference_branch_sha")==PLAN_COMMIT,
         "E_COMMIT_PIN","G00")
    s=m.get("snapshot",{})
    need(isinstance(s,dict) and s.get("file")=="train_canonical.bin" and
         s.get("magic")==MAGIC.decode().strip() and s.get("shape")==
         {"X":[300,5],"y":[300],"mask":[300,5],"ids":[300]},
         "E_SCHEMA","G02","snapshot metadata")
    need(s.get("endianness")=="little" and s.get("order")=="C" and
         s.get("format")=="MAGIC + row-major little-endian float64 X[300,5] + int64 y[300] + uint8 mask[300,5] + int64 ids[300]" and
         s.get("missing")=="canonical IEEE754 quiet NaN float64 for absent; mask stored separately" and
         s.get("ids_semantics")=="original zero-based Train row indices, unchanged order",
         "E_FORMAT_META","G03")
    t=m.get("targets",{})
    need(isinstance(t,dict) and t.get("file")=="targets_float64_le.bin" and
         isinstance(t.get("per_class"),list) and len(t["per_class"])==3,
         "E_SCHEMA","G12","target metadata")


def _runtime_info():
    def dist(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "UNAVAILABLE"
    return {"python":platform.python_version(),"numpy":dist("numpy"),"torch":dist("torch")}


def _check_import_paths(source_paths, expected_hashes):
    """Observe EFFECTIVE import resolution, without importing target modules.

    Checking PathFinder alone ignores sys.meta_path and sys.modules; this check
    consults effective finder order and pins SourceFileLoader raw bytes. This is
    only a PRE-USE observation: a runner must independently bind actual imports
    and protect this state until the first graph operation.
    """
    records = []
    for name, source_name in (("run", "graph_run"), ("stream_graph", "streamgraph"), ("generate", "generator")):
        need(name not in sys.modules, "E_IMPORT_CACHED", "G15",
             "import already present before preflight: " + name)
        expected = Path(source_paths[source_name])
        need(expected.is_file() and not expected.is_symlink(),
             "E_IMPORT_RESOLUTION", "G15", "unsafe expected module path " + name)
        expected = expected.resolve(strict=True)
        # importlib.util.find_spec uses the effective sys.meta_path order. It
        # may invoke a finder but MUST NOT import the target module here.
        try:
            actual = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
            raise GateError("E_IMPORT_RESOLUTION", "G15", "finder error for " + name) from exc
        need(name not in sys.modules, "E_IMPORT_CACHED", "G15",
             "finder imported the target module " + name)
        need(actual is not None and isinstance(actual.loader, importlib.machinery.SourceFileLoader)
             and actual.origin is not None, "E_IMPORT_LOADER", "G15", name)
        origin = Path(actual.origin).resolve(strict=True)
        need(origin == expected and Path(actual.loader.path).resolve(strict=True) == expected,
             "E_IMPORT_RESOLUTION", "G15", "effective meta_path origin " + name)
        need(actual.loader.name == name, "E_IMPORT_LOADER", "G15", name)
        try:
            read_by_loader = actual.loader.get_data(str(expected))
            read_by_path = expected.read_bytes()
        except OSError as exc:
            raise GateError("E_IMPORT_LOADER", "G15", "loader cannot read pinned source") from exc
        expected_hash = expected_hashes[source_name]
        need(sha(read_by_loader) == expected_hash and sha(read_by_path) == expected_hash,
             "E_IMPORT_LOADER_HASH", "G15", "loader/source-byte mismatch for " + name)
        records.append((name, str(expected), expected_hash))
    return tuple(records)


def inspect_b02_bundle(bundle: Bundle, trusted: TrustedSpec) -> VerifiedB02Inputs:
    """Pure inspection. Does not grant access to graph operations."""
    need(sys.flags.optimize == 0,"E_RUNTIME_OPTIMIZED","G19")
    need(bundle.refs == {"snapshot":trusted.snapshot_commit,"plan":trusted.plan_commit,"chapter4":trusted.chapter4_commit} and
         trusted.snapshot_commit==SNAPSHOT_COMMIT and trusted.plan_commit==PLAN_COMMIT and trusted.chapter4_commit==CH4_COMMIT,
         "E_COMMIT_PIN","G00")
    need(len(trusted.manifest_sha256)==64 and sha(bundle.manifest)==trusted.manifest_sha256,
         "E_MANIFEST_TRUST","G00")
    m=_json(bundle.manifest,"G01")
    _required_manifest(m)
    need(len(bundle.snapshot)==18322,"E_SIZE","G02","Train file length")
    need(bundle.snapshot.startswith(MAGIC),"E_MAGIC","G02")
    need(sha(bundle.snapshot)==trusted.snapshot_sha256 and
         m["snapshot"].get("file_sha256")==trusted.snapshot_sha256 and
         m["snapshot"].get("payload_sha256")==trusted.snapshot_sha256,
         "E_SNAPSHOT_SHA","G04")
    slices={name:bundle.snapshot[off:off+length] for name,(off,length) in CHUNKS.items()}
    for name, raw in slices.items():
        need(sha(raw)==trusted.array_sha256[name] and
             m["snapshot"]["array_sha256"].get(name)==trusted.array_sha256[name],
             "E_ARRAY_SHA_"+name.upper(),"G04")
    xx=struct.unpack("<1500d", slices["X"])
    y=struct.unpack("<300q", slices["y"])
    mask=slices["mask"]
    ids=struct.unpack("<300q", slices["ids"])
    need(all(v in (0,1) for v in mask),"E_MASK_DOMAIN","G05")
    for idx,(x,flag) in enumerate(zip(xx,mask)):
        if flag:
            need(math.isfinite(x),"E_MASK_X_SEMANTIC","G06","observed missing/inf")
        else:
            need(math.isnan(x),"E_MASK_X_SEMANTIC","G06","absent is finite/inf")
            bits=struct.unpack_from("<Q",slices["X"],idx*8)[0]
            need(bits==0x7ff8000000000000,"E_NAN_CANONICAL","G07")
    need(all(type(v) is int and 0 <= v < 3 for v in y),"E_LABEL_DOMAIN","G08")
    counts=[y.count(i) for i in range(3)]
    need(counts==[72,147,81] and m.get("class_counts")==counts,"E_CLASS_COUNTS","G09")
    need(ids==tuple(range(300)),"E_IDS_ORDER","G10")
    observed=[sum(mask[i*5+j] for i in range(300)) for j in range(5)]
    need(observed==[300,283,264,281,238] and m.get("observed_per_feature")==observed,
         "E_OBSERVED_COUNTS","G11")
    need(len(bundle.targets)==72,"E_TARGET_SIZE","G12")
    need(sha(bundle.targets)==trusted.targets_sha256 and
         m["targets"].get("file_sha256")==trusted.targets_sha256,
         "E_TARGET_FILE_SHA","G12")
    target_rows=[]
    for c in range(3):
        raw=bundle.targets[c*24:c*24+24]
        t=struct.unpack("<3d",raw)
        row=m["targets"]["per_class"][c]
        need(sha(raw)==row.get("sha256"),"E_TARGET_ROW_SHA","G12",str(c))
        need(all(math.isfinite(v) for v in t),"E_TARGET_NONFINITE","G13",str(c))
        need(all(v>=0 for v in t),"E_TARGET_NEGATIVE","G13",str(c))
        need(abs(sum(t)-1.0)<=1e-10,"E_TARGET_MASS","G13",str(c))
        need(type(row.get("class")) is int and row["class"]==c and
             isinstance(row.get("vector"),list) and len(row["vector"])==3 and
             all(type(v) in (int,float) and math.isfinite(v) for v in row["vector"]) and
             struct.pack("<3d",*row["vector"])==raw,
             "E_TARGET_VECTOR_META","G14",str(c))
        need(type(row.get("sum")) in (float,int) and
             abs(row["sum"]-sum(t))<=1e-10 and row.get("self_jsd")==0.0,
             "E_TARGET_VECTOR_META","G14","sum/JSD meta")
        target_rows.append(t)
    need(set(bundle.source_bytes)==set(SOURCES) and set(bundle.source_paths)==set(SOURCES),
         "E_SOURCE_SHA","G15","source set")
    verified_source_pins=[]
    verified_source_paths={}
    for key in SOURCES:
        # Read/copy exactly once, then hash the SAME bytes handed to any future
        # consumer. Never re-read Bundle mutable mappings at return time.
        copied=bytes(bundle.source_bytes[key])
        location=Path(bundle.source_paths[key]).resolve(strict=True)
        loaded=location.read_bytes()
        digest=trusted.source_sha256[key]
        need(sha(copied)==digest and sha(loaded)==digest and
             m.get("source_sha256",{}).get(key)==digest,
             "E_SOURCE_SHA","G15",key)
        verified_source_pins.append((key,str(location),digest,copied))
        verified_source_paths[key]=location
    need(bundle.plan==next(item[3] for item in verified_source_pins if item[0]=="plan"),
         "E_SOURCE_SHA","G15","plan bytes mismatch")
    imports=_check_import_paths(verified_source_paths, trusted.source_sha256)
    need(m.get("seed",{}).get("data")==20261102 and m["seed"].get("reference")==20261201 and
         m["seed"].get("teacher")==20261211 and m.get("walk")==
         {"H":12,"beta":.4,"lambda":.8,"dtype":"torch.float64","method":"exact_no_grad_three_forwards"},
         "E_SEED_PLAN","G16")
    runtime=_runtime_info()
    need(runtime==dict(trusted.approved_runtime) and runtime==m.get("versions"),
         "E_RUNTIME_VERSION","G16",str(runtime))
    p=_json(bundle.plan,"G17")
    need(p.get("schema")=="CH4-EXP-INV-01-P3-proposed-plan-v1" and
         p.get("status")=="PROPOSED_AWAITING_MAIN_RUN_GATE" and
         p.get("source_chapter4_sha")==CH4_COMMIT,
         "E_PLAN_MATRIX","G17","plan metadata")
    blocks=[b for b in p.get("blocks",[]) if b.get("block_id")==BLOCK]
    need(len(blocks)==1 and blocks[0].get("authorized_to_execute") is False and
         all(blocks[0].get(k)==v for k,v in {"nref":300,"data_seed":20261102,"reference_seed":20261201,"teacher_seed":20261211,"class_count":3,"planned_solver_runs":18}.items()),
         "E_RUN_STATE","G17","block")
    rows=[r for r in p.get("runs",[]) if r.get("block_id")==BLOCK]
    need(len(rows)==18 and len({r.get("run_id") for r in rows})==18,
         "E_PLAN_MATRIX","G17","run IDs")
    found=set()
    expected_ids=[]
    random_seeds=[]
    for c in range(3):
        for st,abbr in (("zero","Z"),("independent_random","R")):
            for method,suffix in METHODS.items():
                expected_ids.append(f"{BLOCK}-C{c}-{abbr}-{suffix}")
    for r in rows:
        c,st,method=r.get("class_index"),r.get("initialization"),r.get("solver")
        need(type(c) is int and c in (0,1,2) and st in ("zero","independent_random") and method in METHODS,
             "E_PLAN_MATRIX","G17","invalid tuple")
        found.add((c,st,method))
        seed=None if st=="zero" else 20261221+100*c
        need(r.get("initialization_seed")==seed and (seed is None or type(r.get("initialization_seed")) is int),
             "E_INIT_SEED","G17")
        if st!="zero":random_seeds.append(seed)
        need(r.get("state")=="NOT_RUN", "E_RUN_STATE","G17")
        need(r.get("run_id")==f"{BLOCK}-C{c}-{'Z' if st=='zero' else 'R'}-{METHODS[method]}" and
             all(r.get(k)==v for k,v in {"nref":300,"data_seed":20261102,"reference_seed":20261201,
                "teacher_seed":20261211,"beta":.4,"lambda":.8,"H":12,"precision":"float64","exact_JSD_success_tol":1e-8,
                "shared_problem_id":f"n300-d20261102-c{c}"}.items()),
             "E_PLAN_MATRIX","G17","run descriptor")
    need(len(found)==18 and set(m.get("planned_run_ids",[]))==set(expected_ids) and
         m.get("planned_run_ids")==[r["run_id"] for r in rows],
         "E_PLAN_MATRIX","G17","manifest/plan order")
    need(m.get("seed",{}).get("random_initializations")==random_seeds,
         "E_INIT_SEED","G17","manifest random seed order")
    return VerifiedB02Inputs(
        X=tuple(tuple(xx[i*5:(i+1)*5]) for i in range(300)),
        y=y, mask=tuple(tuple(mask[i*5:(i+1)*5]) for i in range(300)),
        ids=ids,targets=tuple(target_rows),snapshot_sha256=sha(bundle.snapshot),targets_sha256=sha(bundle.targets),
        source_pins=tuple(verified_source_pins),
        import_attestation=imports, import_path_at_verify=tuple(sys.path),
        meta_finder_ids_at_verify=tuple(id(finder) for finder in sys.meta_path),
        trust_manifest_sha256=trusted.manifest_sha256)


def verify_b02_inputs(*, bundle:Bundle, trusted_spec:TrustedSpec, requested_scope:str):
    """The only candidate public admission API. Always denies all requested scopes.

    A successful byte/semantic inspection must NEVER unlock graph operations.
    """
    need(requested_scope in ("RESOURCE_PROBE","SOLVE"),"E_NOT_AUTHORIZED","G18","unsupported scope")
    inspect_b02_bundle(bundle,trusted_spec)
    raise GateError("DENY_NO_AUTHORIZATION","G18", "candidate gate has no grant mechanism")

def check_preuse_stub(verified: VerifiedB02Inputs, trusted_spec: TrustedSpec):
    """TOCTOU diagnostic ONLY, never a permission or runner integration.

    Future runner must consume ONLY VerifiedB02Inputs immutable X/y/mask/ids,
    targets and source bytes; it must never re-open a mutable Bundle or a source
    path. Source code imports are not bound to verified bytes in this candidate.
    Even an unchanged pre-use snapshot therefore DENIES with INTEGRATION_REQUIRED.
    """
    need(sys.flags.optimize == 0, "E_RUNTIME_OPTIMIZED", "G19")
    need(type(verified) is VerifiedB02Inputs and type(trusted_spec) is TrustedSpec,
         "E_PREUSE_TYPE", "G20")
    need(verified.trust_manifest_sha256 == trusted_spec.manifest_sha256 and
         verified.snapshot_sha256 == trusted_spec.snapshot_sha256 and
         verified.targets_sha256 == trusted_spec.targets_sha256,
         "E_PREUSE_TRUST", "G20")
    need(tuple(sys.path) == verified.import_path_at_verify and
         tuple(id(finder) for finder in sys.meta_path) == verified.meta_finder_ids_at_verify,
         "E_PREUSE_IMPORT_ENV", "G20", "sys.path or sys.meta_path drift")
    for key, name, digest, immutable_content in verified.source_pins:
        need(digest == trusted_spec.source_sha256[key] and sha(immutable_content) == digest,
             "E_PREUSE_SOURCE", "G20", "immutable source pin drift")
        source_path = Path(name)
        need(not source_path.is_symlink() and source_path.is_file(),
             "E_PREUSE_SOURCE", "G20", "symlink/missing source")
        try:
            current = source_path.read_bytes()
        except OSError as exc:
            raise GateError("E_PREUSE_SOURCE", "G20", "source changed/unreadable") from exc
        need(sha(current) == digest, "E_PREUSE_SOURCE", "G20", "disk content mutated")
    try:
        live = _check_import_paths(
            {source_key: Path(path) for source_key, path, _, _ in verified.source_pins},
            trusted_spec.source_sha256,
        )
    except GateError as exc:
        raise GateError("E_PREUSE_IMPORT", "G20", exc.code + ":" + exc.explanation) from exc
    need(live == verified.import_attestation, "E_PREUSE_IMPORT", "G20", "import origin drift")
    raise GateError("INTEGRATION_REQUIRED", "G20",
                    "pre-use observation is not an authorized import/graph operation")
