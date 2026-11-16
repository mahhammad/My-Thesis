"""G4 protected stage in ONE GitHub Actions step with only its own token.

No secrets in GITHUB_ENV, job env, artifact, git commit, or issue. This module
is never invoked by ordinary NO-FIT CI. Only prevalidated workflow input plus
live owner/QA authorization can reserve one stage budget.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import execution_gate
import stage_execute
import bounded_run

STAGES=("raw_noise","noise_masks","derived","fit")


def safe_receipt(stage, outcome, phase, *, token=None, root=None):
    """Immutable, intentionally minimal evidence without API/secret material."""
    if stage not in STAGES or outcome not in ("FAIL_STOP","EXECUTED"):
        raise ValueError("Invalid receipt")
    base=Path(root) if root is not None else HERE/"results/metrics"
    if not base.is_dir():
        raise FileNotFoundError(base)
    run=os.environ.get("GITHUB_RUN_ID","")
    if not run.isdigit():
        raise PermissionError("Valid GitHub run ID required")
    path=base/f"e14_execution_receipt_{stage}_{run}.json"
    receipt={
        "schema":"e-ch4-14-execution-receipt-v1",
        "stage":stage,"outcome":outcome,"last_phase":phase,
        "workflow_run_id":run,
        "head_sha":os.environ.get("GITHUB_SHA"),
        "event":os.environ.get("GITHUB_EVENT_NAME"),
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "test_accessed":False,"secrets_recorded":False,
        "retry_authorized":False,
    }
    rendered=json.dumps(receipt,sort_keys=True,indent=2)+"\n"
    if token and token in rendered:
        raise PermissionError("Secret must never enter the recovery receipt")
    with path.open("x",encoding="utf8") as f:
        f.write(rendered)
        f.flush()
        os.fsync(f.fileno())
    return path


def run(stage, owner_id, qa_id, *, token, verify=None, preflight=None,
        reserve=None, execute=None, receipt=safe_receipt):
    if stage not in STAGES or type(owner_id) is not int or type(qa_id) is not int:
        raise PermissionError("Invalid stage or owner/QA IDs")
    if owner_id < 1 or qa_id < 1 or not isinstance(token,str) or len(token)<32:
        raise PermissionError("Missing specific per-stage Secret or machine approval")
    # Only the current stage's one token is ever read in this protected step.
    if os.environ.get("E14_STAGE_TOKEN") != token:
        raise PermissionError("Token not scoped to this step")
    context=Path(os.environ["RUNNER_TEMP"])/"e14_verified_context.json"
    command=["--stage",stage,"--context",str(context)]
    review=["--stage",stage,"--owner-comment-id",str(owner_id),
            "--qa-comment-id",str(qa_id),"--context",str(context)]
    verify=verify or execution_gate.main
    preflight=preflight or stage_execute.main
    reserve=reserve or execution_gate.main
    execute=execute or bounded_run.main
    phase="VERIFY"
    status="FAIL_STOP"
    try:
        verify(["--mode","verify",*review])
        phase="PREFLIGHT"
        preflight(["--mode","preflight",*command])
        phase="RESERVE"
        reserve(["--mode","reserve",*review])
        phase="EXECUTE"
        execute(command)
        phase="COMPLETE"
        status="EXECUTED"
        return 0
    finally:
        # Even rejected authorization leaves a non-secret status for recovery;
        # artifact upload is independent of the subsequent git push.
        receipt(stage,status,phase,token=token)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=STAGES,required=True)
    parser.add_argument("--owner-comment-id",type=int,required=True)
    parser.add_argument("--qa-comment-id",type=int,required=True)
    args=parser.parse_args(argv)
    try:
        return run(args.stage,args.owner_comment_id,args.qa_comment_id,
                   token=os.environ.get("E14_STAGE_TOKEN",""))
    except Exception as exc:
        # Do not echo exception strings which could originate from external APIs.
        print("E14_FAIL_STOP",args.stage,type(exc).__name__,file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
