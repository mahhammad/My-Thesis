"""Synthetic Train-only byte fixtures. No generator/graph/optimizer imports or calls."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor
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

from audit_candidate import audit, audit_preflight_attempt, write_record
from pure_gate import (BLOCK, MAGIC, SNAPSHOT_COMMIT, PLAN_COMMIT, CH4_COMMIT,
    SOURCES, Bundle, TrustedSpec, GateError, sha, inspect_b02_bundle,
    verify_b02_inputs, read_pinned_bundle, check_preuse_stub)


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

    # V2 targeted adversarial suite. All paths, source texts and arrays are
    # synthetic private temporary fixtures; no graph/module/solver imported.
    def test_g15_meta_path_effective_shadow(self):
        shadow=self.root/"adversarial";shadow.mkdir()
        (shadow/"run.py").write_text("MARKER='shadow'\n")
        class ShadowFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname=="run":
                    return importlib.util.spec_from_file_location(fullname,shadow/"run.py")
                return None
        finder=ShadowFinder()
        old=list(sys.meta_path)
        try:
            sys.meta_path.insert(0,finder)
            # PathFinder alone still points at the expected path.
            legacy=importlib.machinery.PathFinder.find_spec("run",sys.path)
            self.assertEqual(Path(legacy.origin).resolve(),self.base.source_paths["graph_run"].resolve())
            with self.assertRaises(GateError) as cm:
                inspect_b02_bundle(self.base,self.trust)
            self.assertEqual(cm.exception.code,"E_IMPORT_RESOLUTION")
        finally:
            sys.meta_path[:]=old
        self.assertNotIn("run",sys.modules)

    def test_g15_sys_modules_shadow(self):
        shadow=types.ModuleType("run")
        shadow.__file__=str(self.root/"shadow_fake_run.py")
        old=sys.modules.get("run")
        sys.modules["run"]=shadow
        try:
            with self.assertRaises(GateError) as cm:
                inspect_b02_bundle(self.base,self.trust)
            self.assertEqual(cm.exception.code,"E_IMPORT_CACHED")
        finally:
            if old is None:sys.modules.pop("run",None)
            else:sys.modules["run"]=old

    def test_g15_origin_is_pinned_but_loader_is_not(self):
        expected=self.base.source_paths["graph_run"]
        class SpoofFinder(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname=="run":
                    return importlib.machinery.ModuleSpec("run",object(),origin=str(expected))
                return None
        old=list(sys.meta_path)
        try:
            sys.meta_path.insert(0,SpoofFinder())
            with self.assertRaises(GateError) as cm:
                inspect_b02_bundle(self.base,self.trust)
            self.assertEqual(cm.exception.code,"E_IMPORT_LOADER")
        finally:sys.meta_path[:]=old

    def test_g20_immutable_values_and_preuse_still_denied(self):
        candidate=inspect_b02_bundle(self.base,self.trust)
        self.assertTrue(isinstance(candidate.X,tuple) and isinstance(candidate.source_pins,tuple))
        self.assertEqual(candidate.source_pins[0][3],self.base.source_bytes["config"])
        self.base.source_bytes["config"]=b"mutated Bundle mapping after verify"
        self.assertNotEqual(candidate.source_pins[0][3],self.base.source_bytes["config"])
        with self.assertRaises(GateError) as cm:
            check_preuse_stub(candidate,self.trust)
        self.assertEqual(cm.exception.code,"INTEGRATION_REQUIRED")
        self.assertEqual(candidate.ids,tuple(range(300)))

    def test_g20_mutable_bundle_swap_during_inspection_uses_pinned_bytes(self):
        import pure_gate
        original=self.base.source_bytes["config"]
        actual_check=pure_gate._check_import_paths
        def mutate_after_hash(source_paths, expected_hashes):
            result=actual_check(source_paths,expected_hashes)
            self.base.source_bytes["config"]=b"attacker changed mutable Bundle mapping"
            return result
        with patch("pure_gate._check_import_paths", side_effect=mutate_after_hash):
            verified=inspect_b02_bundle(self.base,self.trust)
        self.assertEqual(verified.source_pins[0][3],original)
        self.assertNotEqual(verified.source_pins[0][3],self.base.source_bytes["config"])
        with self.assertRaises(GateError) as cm:
            check_preuse_stub(verified,self.trust)
        self.assertEqual(cm.exception.code,"INTEGRATION_REQUIRED")

    def test_g20_concurrent_source_swap_before_use(self):
        candidate=inspect_b02_bundle(self.base,self.trust)
        go=threading.Event();done=threading.Event()
        source=self.base.source_paths["graph_run"]
        def attacker():
            if not go.wait(2):return
            source.write_bytes(b"# SWAPPED AFTER VERIFICATION\n")
            done.set()
        th=threading.Thread(target=attacker)
        th.start()
        go.set()
        self.assertTrue(done.wait(2))
        try:
            with self.assertRaises(GateError) as cm:
                check_preuse_stub(candidate,self.trust)
            self.assertEqual(cm.exception.code,"E_PREUSE_SOURCE")
        finally:th.join(timeout=2)
        self.assertFalse(th.is_alive())

    def test_g20_sys_modules_import_appears_before_use(self):
        candidate=inspect_b02_bundle(self.base,self.trust)
        old=sys.modules.get("run")
        sys.modules["run"]=types.ModuleType("run")
        try:
            with self.assertRaises(GateError) as cm:
                check_preuse_stub(candidate,self.trust)
            self.assertEqual(cm.exception.code,"E_PREUSE_IMPORT")
        finally:
            if old is None:sys.modules.pop("run",None)
            else:sys.modules["run"]=old

    def test_g20_meta_path_changes_before_use(self):
        candidate=inspect_b02_bundle(self.base,self.trust)
        old=list(sys.meta_path)
        try:
            sys.meta_path.insert(0,object())
            with self.assertRaises(GateError) as cm:
                check_preuse_stub(candidate,self.trust)
            self.assertEqual(cm.exception.code,"E_PREUSE_IMPORT_ENV")
        finally:sys.meta_path[:]=old

    def _audited_preflight(self, trust_path, *, snap=None, root=None, scope="SOLVE", refs=None):
        out=self.root/"preflight-log"
        start,end,record=audit_preflight_attempt(
            snapshot_dir=snap or self.root/"snapshot-scratch",
            repo_root=root or self.root/"checkout-scratch",
            trusted_path=trust_path,audit_output=out,
            scope=scope,refs=refs)
        self.assertTrue(start.is_file() and end.is_file())
        self.assertEqual(json.loads(start.read_text())["status"],"STARTED")
        saved=json.loads(end.read_text())
        self.assertEqual(saved["code"],record["code"])
        self.assertEqual(saved["stage"],record["stage"])
        self.assertEqual(saved["attempt_id"],json.loads(start.read_text())["attempt_id"])
        self.assertEqual(saved["graph_constructed"],0)
        self.assertEqual(saved["optimizer_steps"],0)
        self.assertEqual(saved["phase"],"terminal")
        return record

    def test_audit_preflight_missing_trust_recorded(self):
        record=self._audited_preflight(self.root/"nonexistent-trust.json")
        self.assertEqual((record["code"],record["stage"]),("E_TRUST_MISSING","PREFLIGHT_TRUST"))
        record2=self._audited_preflight(None)
        self.assertEqual(record2["code"],"E_TRUST_MISSING")

    def test_audit_preflight_malformed_trust_recorded(self):
        p=self.root/"trust.json"
        p.write_text("{not valid")
        record=self._audited_preflight(p)
        self.assertEqual((record["code"],record["stage"]),("E_TRUST_JSON","PREFLIGHT_TRUST"))

    def test_audit_preflight_trust_inside_repo_and_symlink(self):
        repo=self.root/"checkout-scratch";repo.mkdir(exist_ok=True)
        trust=repo/"trust.json";trust.write_text("{}")
        record=self._audited_preflight(trust,root=repo)
        self.assertEqual(record["code"],"E_TRUST_UNSAFE")
        link=self.root/"trust-alias.json";link.symlink_to(trust)
        record2=self._audited_preflight(link,root=repo)
        self.assertEqual(record2["code"],"E_TRUST_UNSAFE")

    def test_audit_preflight_invalid_trust_schema(self):
        p=self.root/"trust-unexpected.json"
        p.write_text(json.dumps({"manifest_sha256":"missing others"}))
        record=self._audited_preflight(p)
        self.assertEqual(record["code"],"E_TRUST_SCHEMA")

    def test_audit_preflight_loader_missing_path_and_wrong_refs(self):
        from dataclasses import asdict
        trust=self.root/"trust-proposal-test.json"
        trust.write_text(json.dumps(asdict(self.trust)))
        missing=self._audited_preflight(trust,snap=self.root/"not-a-snapshot")
        self.assertEqual((missing["code"],missing["stage"]),
                         ("E_INPUT_MISSING","PREFLIGHT_BUNDLE"))
        drift=self._audited_preflight(trust,
                 refs={"snapshot":"wrong","plan":PLAN_COMMIT,"chapter4":CH4_COMMIT})
        self.assertEqual((drift["code"],drift["stage"]),("E_COMMIT_PIN","PREFLIGHT_BUNDLE"))

    def test_audit_preflight_snapshot_symlink_attempt_logged(self):
        from dataclasses import asdict
        trust=self.root/"trust-test.json"
        trust.write_text(json.dumps(asdict(self.trust)))
        repo=self.root/"checkout-scratch";repo.mkdir(exist_ok=True)
        snap=self.root/"snapshot-scratch";snap.mkdir(exist_ok=True)
        (snap/"manifest.json").symlink_to(self.root/"untrusted-source")
        record=self._audited_preflight(trust,snap=snap,root=repo)
        self.assertEqual(record["stage"],"PREFLIGHT_BUNDLE")
        self.assertIn(record["code"],("E_UNSAFE_PATH","E_INPUT_MISSING"))

    def test_audit_repeated_and_concurrent_exclusive_create(self):
        out=self.root/"append-root"
        records=[{"attempt_id":f"{i+1:032x}","phase":"terminal","code":"E_TEST"} for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            paths=list(ex.map(lambda rec:write_record(out,rec),records))
        self.assertEqual(len({str(p) for p in paths}),4)
        self.assertEqual(len(list(out.glob("*.json"))),4)
        original=paths[0].read_bytes()
        with self.assertRaises(GateError) as cm:
            write_record(out,records[0])
        self.assertEqual(cm.exception.code,"E_AUDIT_WRITE")
        self.assertEqual(paths[0].read_bytes(),original)

    def test_audit_unavailable_root_denies_without_claim(self):
        bad=self.root/"unsafe-absent-parent"/"audit"
        with self.assertRaises(GateError) as cm:
            audit_preflight_attempt(snapshot_dir=None,repo_root=None,
                 trusted_path=None,audit_output=bad,scope="SOLVE")
        self.assertEqual(cm.exception.code,"E_AUDIT_ROOT_UNAVAILABLE")
        self.assertFalse(bad.exists())
        real=self.root/"secure";real.mkdir()
        aliased=self.root/"symlink-audit"
        aliased.symlink_to(real,target_is_directory=True)
        with self.assertRaises(GateError) as cm2:
            write_record(aliased,{"code":"E_TEST"})
        self.assertEqual(cm2.exception.code,"E_AUDIT_ROOT_UNAVAILABLE")

    def test_audit_python_O_starts_then_rejects(self):
        from dataclasses import asdict
        out=self.root/"optimized-audit"
        p=self.root/"trust-test.json";p.write_text(json.dumps(asdict(self.trust)))
        script=("from audit_candidate import audit_preflight_attempt;"
                "from pathlib import Path;"
                "import sys;"
                "_,_,r=audit_preflight_attempt(snapshot_dir=Path(sys.argv[1]),"
                "repo_root=Path(sys.argv[2]),trusted_path=Path(sys.argv[3]),"
                "audit_output=Path(sys.argv[4]),scope='SOLVE');"
                "print(r['code']);sys.exit(0 if r['code']=='E_RUNTIME_OPTIMIZED' else 1)")
        r=subprocess.run([sys.executable,"-O","-c",script,str(self.root/"snapshot-scratch"),
                          str(self.root/"checkout-scratch"),str(p),str(out)],
                          cwd=Path(__file__).parent,capture_output=True,text=True,timeout=4)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn("E_RUNTIME_OPTIMIZED",r.stdout)
        entries=list(out.glob("*.json"))
        self.assertEqual(len(entries),2)
        self.assertEqual({json.loads(x.read_text())["phase"] for x in entries},{"start","terminal"})

    def test_audit_preflight_loader_exception_has_stage(self):
        from dataclasses import asdict
        p=self.root/"trust-loader.json";p.write_text(json.dumps(asdict(self.trust)))
        with patch("audit_candidate.read_pinned_bundle",side_effect=RuntimeError("toy loader failure")):
            record=self._audited_preflight(p)
        self.assertEqual((record["code"],record["stage"]),
                         ("E_INPUT_LOADER","PREFLIGHT_BUNDLE"))
        self.assertIn("RuntimeError",record["reason"])

    def test_audit_preflight_checkout_root_symlink(self):
        from dataclasses import asdict
        p=self.root/"trust-root.json";p.write_text(json.dumps(asdict(self.trust)))
        actual=self.root/"real-snapshot";actual.mkdir()
        alias=self.root/"aliased-snapshot";alias.symlink_to(actual,target_is_directory=True)
        record=self._audited_preflight(p,snap=alias)
        self.assertEqual((record["code"],record["stage"]),
                         ("E_UNSAFE_PATH","PREFLIGHT_BUNDLE"))

    def test_audit_preflight_duplicate_json_trust_key(self):
        p=self.root/"double-trust.json"
        p.write_text('{"manifest_sha256":"a","manifest_sha256":"b"}')
        record=self._audited_preflight(p)
        self.assertEqual((record["code"],record["stage"]),
                         ("E_TRUST_JSON","PREFLIGHT_TRUST"))

    def test_audit_nongrant_exit_and_stage(self):
        from dataclasses import asdict
        out=self.root/"cli-audit"
        trust=self.root/"missing-trust-json"
        cmd=[sys.executable,str(Path(__file__).with_name("audit_candidate.py")),
             "--audit-output",str(out),"--trusted-spec",str(trust),"--scope","SOLVE"]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=4)
        self.assertEqual(p.returncode,2,p.stderr)
        result=json.loads(p.stdout)
        self.assertEqual(result["code"],"E_TRUST_MISSING")
        self.assertEqual(len(list(out.glob("*.json"))),2)

if __name__=="__main__":unittest.main(verbosity=2)
