"""Synthetic Train-only byte fixtures. No generator/graph/optimizer imports or calls."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from audit_candidate import audit
from pure_gate import (BLOCK, MAGIC, SNAPSHOT_COMMIT, PLAN_COMMIT, CH4_COMMIT,
    SOURCES, Bundle, TrustedSpec, GateError, sha, inspect_b02_bundle,
    verify_b02_inputs, read_pinned_bundle)


def jb(x):return (json.dumps(x,sort_keys=True,ensure_ascii=True,indent=2)+"\n").encode()


def runtime():
    def version(n):
        try:return importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:return "UNAVAILABLE"
    return {"python":platform.python_version(),"numpy":version("numpy"),"torch":version("torch")}


def make_case(root):
    root=Path(root)
    x=[];mask=[]
    for row in range(300):
        for col,seen in enumerate([300,283,264,281,238]):
            observed=int(row<seen);mask.append(observed)
            x.append(struct.pack("<d",float(row*5+col+1) / 100) if observed else bytes.fromhex("000000000000f87f"))
    y=[0]*72+[1]*147+[2]*81
    xraw=b"".join(x);yraw=struct.pack("<300q",*y);mraw=bytes(mask);idraw=struct.pack("<300q",*range(300))
    raw=MAGIC+xraw+yraw+mraw+idraw
    target_rows=[(.3,.4,.3),(.2,.6,.2),(.1,.4,.5)]
    targets=b"".join(struct.pack("<3d",*row) for row in target_rows)
    plan={"schema":"CH4-EXP-INV-01-P3-proposed-plan-v1","status":"PROPOSED_AWAITING_MAIN_RUN_GATE",
          "source_chapter4_sha":CH4_COMMIT,"blocks":[{"block_id":BLOCK,"nref":300,"data_seed":20261102,"reference_seed":20261201,
          "teacher_seed":20261211,"class_count":3,"planned_solver_runs":18,"authorized_to_execute":False}],"runs":[]}
    suffix={"Q-only":"Q","Exact-Adam":"A","Q+Bootstrap-GN":"G"}
    for c in range(3):
        for st,letter in (("zero","Z"),("independent_random","R")):
            for solver,short in suffix.items():
                plan["runs"].append({"block_id":BLOCK,"run_id":f"{BLOCK}-C{c}-{letter}-{short}","nref":300,
                   "data_seed":20261102,"reference_seed":20261201,"teacher_seed":20261211,"class_index":c,
                   "initialization":st,"initialization_seed":None if st=="zero" else 20261221+100*c,
                   "solver":solver,"beta":.4,"lambda":.8,"H":12,"precision":"float64","exact_JSD_success_tol":1e-8,
                   "shared_problem_id":f"n300-d20261102-c{c}","state":"NOT_RUN"})
    paths={};sources={}
    for name,p in SOURCES.items():
        path=root / p
        path.parent.mkdir(parents=True,exist_ok=True)
        src=jb(plan) if name=="plan" else b"# synthetic fixture; no model imported\n"
        path.write_bytes(src)
        paths[name]=path;sources[name]=src
    m={"schema":"B02_TRAIN_AND_TARGET_V1","block":BLOCK,"status":"LOCKED_NO_SOLVERS", "nref":300,
       "contexts":18,"three_classes":True,"authorized_to_solve":False,"teacher_given_to_solver":False,
       "chapter4_sha":CH4_COMMIT,"reference_branch_sha":PLAN_COMMIT,"class_counts":[72,147,81],
       "observed_per_feature":[300,283,264,281,238],
       "planned_run_ids":[r["run_id"] for r in plan["runs"]],
       "seed":{"data":20261102,"reference":20261201,"teacher":20261211,
               "random_initializations":[r["initialization_seed"] for r in plan["runs"] if r["initialization"]=="independent_random"]},
       "walk":{"H":12,"beta":.4,"lambda":.8,"dtype":"torch.float64","method":"exact_no_grad_three_forwards"},
       "versions":runtime(),
       "snapshot":{"file":"train_canonical.bin","magic":"B02TRAIN_CANONICAL_V1","endianness":"little","order":"C",
           "shape":{"X":[300,5],"y":[300],"mask":[300,5],"ids":[300]},
           "missing":"canonical IEEE754 quiet NaN float64 for absent; mask stored separately",
           "ids_semantics":"original zero-based Train row indices, unchanged order",
           "format":"MAGIC + row-major little-endian float64 X[300,5] + int64 y[300] + uint8 mask[300,5] + int64 ids[300]"},
       "targets":{"file":"targets_float64_le.bin","per_class":[{"class":i,"vector":list(row),"sum":sum(row),"self_jsd":0.0} for i,row in enumerate(target_rows)]}}
    return (Bundle(jb(m),raw,targets,jb(plan),sources,paths,{
         "snapshot":SNAPSHOT_COMMIT,"plan":PLAN_COMMIT,"chapter4":CH4_COMMIT}),m,plan)


def resign(bundle,m,plan=None):
    """TEST-ONLY independently signed fixture anchor; never trusted in production."""
    m=json.loads(json.dumps(m))
    raw=bundle.snapshot;tar=bundle.targets
    m["snapshot"].update(file_sha256=sha(raw),payload_sha256=sha(raw))
    m["snapshot"]["array_sha256"]={k:sha(raw[s:s+n]) for k,s,n in
        (("X",22,12000),("y",12022,2400),("mask",14422,1500),("ids",15922,2400))}
    m["targets"]["file_sha256"]=sha(tar)
    for c,row in enumerate(m["targets"]["per_class"]):
        row["sha256"]=sha(tar[c*24:(c+1)*24])
    sources=dict(bundle.source_bytes)
    if plan is not None:
        sources["plan"]=jb(plan)
        bundle=replace(bundle,plan=jb(plan))
    for name, data in sources.items():
        Path(bundle.source_paths[name]).write_bytes(data) # synthetic scratch fixture only
    m["source_sha256"]={k:sha(v) for k,v in sources.items()}
    manifest=jb(m)
    bundle=replace(bundle,manifest=manifest,source_bytes=sources)
    trust=TrustedSpec(manifest_sha256=sha(manifest),snapshot_sha256=sha(raw),targets_sha256=sha(tar),
       array_sha256=dict(m["snapshot"]["array_sha256"]),source_sha256=dict(m["source_sha256"]),
       approved_runtime=runtime())
    return bundle,trust

class PureGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="b02pure_")
        self.root=Path(self.tmp.name)
        self.base,self.m,self.plan=make_case(self.root)
        self.base,self.trust=resign(self.base,self.m)
        self.old_path=sys.path[:]
        for name in ("graph_run","streamgraph","generator"):
            sys.path.insert(0,str(self.base.source_paths[name].parent))
    def tearDown(self):
        sys.path[:]=self.old_path
        self.tmp.cleanup()
    def check_code(self,expected, *, bundle=None,manifest=None,plan=None, outer_resign=True):
        bundle=bundle or self.base
        m=manifest if manifest is not None else self.m
        if outer_resign:
            bundle,trust=resign(bundle,m,plan)
        else:trust=self.trust
        with self.assertRaises(GateError) as cm:
            inspect_b02_bundle(bundle,trust)
        self.assertEqual(cm.exception.code,expected,str(cm.exception))
        # Each negative fixture is independently logged, not merely asserted.
        path,record=audit(bundle,trust,"RESOURCE_PROBE",self.root/"fixture-audit-append-only")
        self.assertEqual(record["code"],expected)
        self.assertEqual(record["trusted_manifest_sha256"],trust.manifest_sha256)
        self.assertEqual(record["graph_constructed"],0)
        self.assertEqual(record["optimizer_steps"],0)
        self.assertTrue(record["attempt_id"] and path.is_file())
        return cm.exception
    def test_valid_semantic_bundle_but_denied(self):
        v=inspect_b02_bundle(self.base,self.trust)
        self.assertEqual(len(v.X),300)
        self.assertEqual(len(v.targets),3)
        for scope in ("RESOURCE_PROBE","SOLVE"):
            with self.assertRaises(GateError) as cm:
                verify_b02_inputs(bundle=self.base,trusted_spec=self.trust,requested_scope=scope)
            self.assertEqual(cm.exception.code,"DENY_NO_AUTHORIZATION")
        # No graph/optimizer class or function exists in this fixture.
        counters={"graph_constructed":0,"optimizer_steps":0}
        def forbidden_graph_factory():
            counters["graph_constructed"]+=1
            raise AssertionError("graph factory must never be reached")
        def forbidden_optimizer_step():
            counters["optimizer_steps"]+=1
            raise AssertionError("optimizer must never be reached")
        # Stub admission invokes factories ONLY after verification returns.
        try:
            verify_b02_inputs(bundle=self.base,trusted_spec=self.trust,requested_scope="SOLVE")
        except GateError:
            pass
        else:
            forbidden_graph_factory()
            forbidden_optimizer_step()
        self.assertEqual(counters,{"graph_constructed":0,"optimizer_steps":0})
    def test_no_production_trust_resign(self):
        changed=replace(self.base,manifest=self.base.manifest+b" ")
        self.check_code("E_MANIFEST_TRUST",bundle=changed,outer_resign=False)
    def test_header_length(self):
        raw=bytearray(self.base.snapshot);raw[0]^=1
        self.check_code("E_MAGIC",bundle=replace(self.base,snapshot=bytes(raw)))
        self.check_code("E_SIZE",bundle=replace(self.base,snapshot=self.base.snapshot[:-1]))
    def test_schema(self):
        m=json.loads(json.dumps(self.m));m["schema"]="other"
        self.check_code("E_SCHEMA",manifest=m)
    def test_metadata_endianness(self):
        m=json.loads(json.dumps(self.m));m["snapshot"]["endianness"]="big"
        self.check_code("E_FORMAT_META",manifest=m)
    def test_array_hash_layer(self):
        raw=bytearray(self.base.snapshot);raw[24]^=1
        changed=replace(self.base,snapshot=bytes(raw))
        # Sign outer only (not component X) with test anchor to reach array check.
        m=json.loads(json.dumps(self.m));m["snapshot"]["file_sha256"]=sha(changed.snapshot)
        m["snapshot"]["payload_sha256"]=sha(changed.snapshot)
        changed=replace(changed,manifest=jb(m))
        anchor=replace(self.trust,manifest_sha256=sha(changed.manifest),snapshot_sha256=sha(changed.snapshot))
        with self.assertRaises(GateError) as cm:inspect_b02_bundle(changed,anchor)
        self.assertEqual(cm.exception.code,"E_ARRAY_SHA_X")
    def test_mask_nonbinary(self):
        raw=bytearray(self.base.snapshot);raw[14422]=2
        self.check_code("E_MASK_DOMAIN",bundle=replace(self.base,snapshot=bytes(raw)))
    def test_mask_semantics(self):
        raw=bytearray(self.base.snapshot);raw[14422]=0
        self.check_code("E_MASK_X_SEMANTIC",bundle=replace(self.base,snapshot=bytes(raw)))
    def test_noncanonical_nan(self):
        raw=bytearray(self.base.snapshot)
        idx=299*5+4;raw[22+idx*8:22+(idx+1)*8]=struct.pack("<Q",0x7ff8000000000001)
        self.check_code("E_NAN_CANONICAL",bundle=replace(self.base,snapshot=bytes(raw)))
    def test_labels(self):
        raw=bytearray(self.base.snapshot);struct.pack_into("<q",raw,12022,3)
        self.check_code("E_LABEL_DOMAIN",bundle=replace(self.base,snapshot=bytes(raw)))
    def test_histogram(self):
        raw=bytearray(self.base.snapshot);struct.pack_into("<q",raw,12022,1)
        self.check_code("E_CLASS_COUNTS",bundle=replace(self.base,snapshot=bytes(raw)))
    def test_reference_ids(self):
        raw=bytearray(self.base.snapshot);raw[-16:-8],raw[-8:]=raw[-8:],raw[-16:-8]
        self.check_code("E_IDS_ORDER",bundle=replace(self.base,snapshot=bytes(raw)))
    def test_observed_counts(self):
        raw=bytearray(self.base.snapshot)
        raw[14422]=0;raw[22:30]=struct.pack("<Q",0x7ff8000000000000)
        self.check_code("E_OBSERVED_COUNTS",bundle=replace(self.base,snapshot=bytes(raw)))
    def test_target_length(self):
        self.check_code("E_TARGET_SIZE",bundle=replace(self.base,targets=self.base.targets[:-1]))
    def test_target_negative_sum1(self):
        raw=bytearray(self.base.targets);raw[:24]=struct.pack("<3d",-.1,.6,.5)
        self.check_code("E_TARGET_NEGATIVE",bundle=replace(self.base,targets=bytes(raw)))
    def test_target_mass_strict(self):
        raw=bytearray(self.base.targets);raw[:24]=struct.pack("<3d",.3000001,.4,.3)
        self.check_code("E_TARGET_MASS",bundle=replace(self.base,targets=bytes(raw)))
    def test_target_json_binary_bitwise(self):
        m=json.loads(json.dumps(self.m));m["targets"]["per_class"][0]["vector"][0]=.3000000001
        self.check_code("E_TARGET_VECTOR_META",manifest=m)
    def test_source_hash_drift(self):
        b=replace(self.base,source_bytes=dict(self.base.source_bytes,config=b"changed"))
        self.check_code("E_SOURCE_SHA",bundle=b,outer_resign=False)
    def test_import_resolution_shadow(self):
        root=self.root/"shadow";root.mkdir();(root/"run.py").write_text("# shadow")
        sys.path.insert(0,str(root))
        self.check_code("E_IMPORT_RESOLUTION")
    def test_plan_duplicate_run(self):
        plan=json.loads(json.dumps(self.plan));plan["runs"][1]["run_id"]=plan["runs"][0]["run_id"]
        self.check_code("E_PLAN_MATRIX",plan=plan)
    def test_plan_state(self):
        plan=json.loads(json.dumps(self.plan));plan["runs"][1]["state"]="COMPLETE"
        self.check_code("E_RUN_STATE",plan=plan)
    def test_seed_drift(self):
        plan=json.loads(json.dumps(self.plan));plan["runs"][3]["initialization_seed"]+=1
        self.check_code("E_INIT_SEED",plan=plan)
    def test_runtime_drift(self):
        m=json.loads(json.dumps(self.m));m["versions"]["python"]="0.0.0"
        self.check_code("E_RUNTIME_VERSION",manifest=m)
    def test_python_optimized_rejects_before_io(self):
        # No model imports, no test dataset or graph usage.
        script="from pure_gate import inspect_b02_bundle,GateError;\ntry: inspect_b02_bundle(None,None)\nexcept GateError as e: print(e.code); exit(0 if e.code=='E_RUNTIME_OPTIMIZED' else 1)\nexit(2)"
        r=subprocess.run([sys.executable,"-O","-c",script],cwd=Path(__file__).parent,
                         capture_output=True,text=True,timeout=3,check=False)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn("E_RUNTIME_OPTIMIZED",r.stdout)
    def test_toctou_immutable_copy(self):
        original=inspect_b02_bundle(self.base,self.trust)
        p=self.root/"train_canonical.bin";p.write_bytes(self.base.snapshot)
        p.write_bytes(b"tampered after validation")
        self.assertEqual(original.snapshot_sha256,sha(self.base.snapshot))
        self.assertEqual(original.ids,tuple(range(300)))
        with self.assertRaises(GateError) as cm:
            inspect_b02_bundle(replace(self.base,snapshot=p.read_bytes()),self.trust)
        self.assertEqual(cm.exception.code,"E_SIZE")
    def test_audit_append_only_and_failures_have_attempt_ids(self):
        out=self.root/"audit-new"
        p1,r1=audit(self.base,self.trust,"RESOURCE_PROBE",out,protected_dirs=(self.root/"immutable-historical",))
        p2,r2=audit(self.base,self.trust,"RESOURCE_PROBE",out,protected_dirs=(self.root/"immutable-historical",))
        self.assertNotEqual(p1,p2)
        self.assertEqual(r1["code"],"DENY_NO_AUTHORIZATION")
        self.assertEqual(r2["code"],"DENY_NO_AUTHORIZATION")
        self.assertEqual((r1["graph_constructed"],r2["optimizer_steps"]),(0,0))
        self.assertNotEqual(r1["attempt_id"],r2["attempt_id"])
        m=json.loads(json.dumps(self.m));m["schema"]="tamper"
        invalid,t=resign(self.base,m)
        p3,r3=audit(invalid,t,"SOLVE",out)
        self.assertEqual(r3["code"],"E_SCHEMA")
        self.assertTrue(all(p.exists() for p in (p1,p2,p3)))
        self.assertEqual(len(list(out.glob("*.json"))),3)
        for p in (p1,p2,p3):
            record=json.loads(p.read_text())
            self.assertIn("attempt_id",record)
            self.assertIn("trusted_manifest_sha256",record)
            self.assertIn("code",record)
        with self.assertRaises(GateError) as cm:
            audit(self.base,self.trust,"SOLVE",self.root/"immutable-historical"/"audit",
                  protected_dirs=(self.root/"immutable-historical",))
        self.assertEqual(cm.exception.code,"E_UNSAFE_PATH")
    def test_unsupported_scope_deny(self):
        with self.assertRaises(GateError) as cm:
            verify_b02_inputs(bundle=self.base,trusted_spec=self.trust,requested_scope="OTHER")
        self.assertEqual(cm.exception.code,"E_NOT_AUTHORIZED")
    def test_no_assert_used_in_candidate(self):
        self.assertNotIn(b"assert ",Path(__file__).with_name("pure_gate.py").read_bytes())

if __name__=="__main__":unittest.main(verbosity=2)
