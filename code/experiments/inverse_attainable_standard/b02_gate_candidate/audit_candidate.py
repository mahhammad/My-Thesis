"""Append-only read-only candidate audit. Does not run actual model code.

This CLI never grants authorization. An input bundle must be copied/pinned
externally; root-of-trust JSON must be supplied from outside snapshot checkout.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import os
import sys
import time
import uuid
from pure_gate import (TrustedSpec, GateError, read_pinned_bundle,
                       verify_b02_inputs, sha, SNAPSHOT_COMMIT, PLAN_COMMIT, CH4_COMMIT)


def write_record(directory:Path, record:dict, *, protected_dirs:tuple[Path,...]=()):
    directory=directory.resolve()
    for protected in protected_dirs:
        protected=protected.resolve()
        if directory==protected or directory.is_relative_to(protected):
            raise GateError("E_UNSAFE_PATH","AUDIT","audit output must not touch historical files")
    directory.mkdir(parents=True,exist_ok=True)
    attempt=record.get("attempt_id") or uuid.uuid4().hex
    record=dict(record,attempt_id=attempt)
    body=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode()
    dest=directory/(attempt+".json")
    fd=os.open(str(dest),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"wb") as out:
            out.write(body)
            out.flush();os.fsync(out.fileno())
    except BaseException:
        # Even on interrupted writes, never overwrite an existing record.
        raise
    return dest


def audit(bundle,trust,scope,outdir,*,protected_dirs=()):
    started=time.monotonic()
    attempt=uuid.uuid4().hex
    record={"attempt_id":attempt,"scope":scope,"trusted_manifest_sha256":trust.manifest_sha256,
        "snapshot_sha256_expected":trust.snapshot_sha256,"version":"B02_GATE_CANDIDATE_V1",
        "graph_constructed":0,"optimizer_steps":0}
    try:
        verify_b02_inputs(bundle=bundle,trusted_spec=trust,requested_scope=scope)
    except GateError as exc:
        record.update(status="DENIED" if exc.code=="DENY_NO_AUTHORIZATION" else "REJECTED",
                      code=exc.code,stage=exc.stage,reason=exc.explanation)
    except BaseException as exc:
        record.update(status="REJECTED",code="E_UNEXPECTED",stage="AUDIT",reason=type(exc).__name__)
    else:
        # Fail closed even if future code accidentally returns from verifier.
        record.update(status="REJECTED",code="E_FAIL_OPEN",stage="AUDIT")
    record["wall_seconds"]=time.monotonic()-started
    path=write_record(Path(outdir),record,protected_dirs=protected_dirs)
    return path,record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir",type=Path,required=True)
    parser.add_argument("--repo-root",type=Path,required=True)
    parser.add_argument("--trusted-spec",type=Path,required=True)
    parser.add_argument("--audit-output",type=Path,required=True)
    parser.add_argument("--scope",choices=("RESOURCE_PROBE","SOLVE"),default="RESOURCE_PROBE")
    args=parser.parse_args()
    # Reject trust path in snapshot checkout; the root trust must be external.
    trusted_path=args.trusted_spec.resolve(strict=True)
    snapshot_root=args.snapshot_dir.resolve(strict=True)
    code_root=args.repo_root.resolve(strict=True)
    if trusted_path.is_relative_to(snapshot_root) or trusted_path.is_relative_to(code_root):
        raise SystemExit("DENY: trusted spec may not be stored in snapshot/repository checkout")
    raw=trusted_path.read_bytes()
    trust=TrustedSpec(**json.loads(raw))
    bundle=read_pinned_bundle(args.snapshot_dir,args.repo_root,refs={
       "snapshot":SNAPSHOT_COMMIT,"plan":PLAN_COMMIT,"chapter4":CH4_COMMIT})
    path,record=audit(bundle,trust,args.scope,args.audit_output,protected_dirs=(snapshot_root,code_root))
    print(json.dumps({"audit_file":str(path),"status":record["status"],"code":record["code"]},sort_keys=True))
    # DENY_NO_AUTHORIZATION is an expected audit outcome, never authorization.
    return 0 if record["code"]=="DENY_NO_AUTHORIZATION" else 2

if __name__=="__main__":sys.exit(main())
