"""Online-only GitHub authorization gate, for manual protected CI execution.

Never authorizes training from local gate JSON/env alone. Uses GitHub Actions
GITHUB_TOKEN to read GitHub-owned Issue #81 comments and same-SHA CI status,
then atomically creates a permanent, globally one-time Git ref *before* fitting.
The ref is never deleted, even on failure. No Test or optimizer is imported.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import secrets
import urllib.error
import urllib.request

REPO="mahhammad/My-Thesis"
BRANCH="experiment/ch4-loss-controlled-comparison-01"
OWNER="mahhammad"
CI_WORKFLOW="e-ch4-13-phase2-design.yml"
AUDIT_WORD="TECHNICAL_PREFLIGHT_FIXES_ACCEPTED"
APPROVAL_WORD="PHASE2_EXECUTION_AUTHORIZED"
FIT_CAP=12
GITHUB="https://api.github.com"
SHA_RE=re.compile(r"^[0-9a-f]{40}$")

def api(method,path,token,body=None,opener=None):
    if not token or not isinstance(token,str):
        raise PermissionError("NO_GITHUB_ACTIONS_TOKEN")
    if not path.startswith("/repos/"+REPO+"/"):
        raise ValueError("GitHub endpoint outside designated repository")
    data=None if body is None else json.dumps(body).encode("utf8")
    req=urllib.request.Request(GITHUB+path,data=data,method=method,headers={
        "Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
        "User-Agent":"e-ch4-13-phase2-auth-gate",
        "Content-Type":"application/json"})
    handle=opener or urllib.request.urlopen
    with handle(req,timeout=30) as res:
        return json.load(res)

def comment_matches(comment,word,sha):
    if comment.get("user",{}).get("login")!=OWNER:
        return False
    if comment.get("author_association") not in ("OWNER","MEMBER","COLLABORATOR"):
        return False
    body=comment.get("body") or ""
    return (re.search(r"(?<![A-Z0-9_])"+word+r"(?![A-Z0-9_])",body) is not None
            and ("candidate_sha="+sha) in body
            and "max_fits=12" in body
            and not comment.get("performed_via_github_app"))

def select_authorizations(comments,sha):
    # Reviewer may separately certify technical fixes and explicitly authorize.
    approvals=[c for c in comments if comment_matches(c,APPROVAL_WORD,sha)]
    audits=[c for c in comments if comment_matches(c,AUDIT_WORD,sha)]
    if not approvals or not audits:
        raise PermissionError("MISSING_SEPARATE_TECHNICAL_FIX_APPROVAL_OR_EXECUTION_PERMISSION")
    # The manager execution authorization must occur after technical fix acceptance.
    auth=max(approvals,key=lambda c:c["created_at"])
    aud=max((c for c in audits if c["created_at"]<=auth["created_at"]),
            key=lambda c:c["created_at"],default=None)
    if aud is None or aud["id"]==auth["id"]:
        raise PermissionError("TECHNICAL_REVIEW_AND_AUTHORIZATION_NOT_SEPARATE")
    return aud,auth

def select_same_sha_ci(runs,sha):
    matched=[r for r in runs if r.get("head_sha")==sha
        and r.get("status")=="completed" and r.get("conclusion")=="success"
        and r.get("path","").endswith("/"+CI_WORKFLOW)
        and r.get("event") in ("push","workflow_dispatch")]
    if not matched:
        raise PermissionError("SAME_SHA_DESIGN_CI_NOT_SUCCESSFUL")
    return max(matched,key=lambda r:r.get("run_number",0))

def validate_online(sha,branch,token,api_fn=api):
    if not SHA_RE.fullmatch(sha):raise PermissionError("INVALID_GIT_SHA")
    if branch!=BRANCH:raise PermissionError("UNAPPROVED_BRANCH")
    head=api_fn("GET","/repos/"+REPO+"/branches/"+BRANCH,token)
    if head["commit"]["sha"]!=sha:raise PermissionError("BRANCH_HEAD_SHA_CHANGED")
    if os.environ.get("GITHUB_REPOSITORY",REPO)!=REPO:
        raise PermissionError("WRONG_GITHUB_REPOSITORY")
    if os.environ.get("GITHUB_REF_NAME",BRANCH)!=BRANCH:
        raise PermissionError("WRONG_GITHUB_REF")
    comments=api_fn("GET","/repos/"+REPO+"/issues/81/comments?per_page=100",token)
    if not isinstance(comments,list) or len(comments)>=100:
        raise PermissionError("PAGINATION_UNVERIFIED_STOP")
    aud,auth=select_authorizations(comments,sha)
    records=api_fn("GET","/repos/"+REPO+
                   "/actions/workflows/"+CI_WORKFLOW+"/runs?head_sha="+sha+"&per_page=100",token)
    if records.get("total_count",100)>=100:
        raise PermissionError("CI_PAGINATION_UNVERIFIED_STOP")
    ci=select_same_sha_ci(records.get("workflow_runs",[]),sha)
    return {"auditor_comment":aud["html_url"],"manager_comment":auth["html_url"],
            "ci_url":ci["html_url"],"source_sha":sha,
            "ci_run_id":ci["id"],"fit_cap":FIT_CAP}

def reserve_once(sha,token,api_fn=api):
    """Atomic server-side reservation; 422 means budget already consumed.
    Check is not performed separately: concurrent callers race on same ref.
    """
    name="refs/tags/e-ch4-13-phase2-budget-spent-"+sha
    try:
        api_fn("POST","/repos/"+REPO+"/git/refs",token,{"ref":name,"sha":sha})
    except urllib.error.HTTPError as error:
        if error.code==422:
            raise PermissionError("PHASE2_12_FIT_BUDGET_ALREADY_RESERVED") from error
        raise
    return name

def issue_gate(sha,branch,token,out,*,validate=validate_online,reserve=reserve_once):
    proof=validate(sha,branch,token)
    # Claim BEFORE emitting a valid gate. Even failure after this call consumes
    # the global batch budget. Recovery is audit-only; no silent new batch.
    ref=reserve(sha,token)
    gate={"schema":"e-ch4-13-phase2-online-authorization-v2",
        "status":"PHASE2_EXECUTION_AUTHORIZED",
        "technical_audit_status":"PASS","phase2_tests":"PASS",
        "technical_audit_url":proof["auditor_comment"],
        "manager_authorization_url":proof["manager_comment"],
        "ci_url":proof["ci_url"],"source_sha":sha,
        "one_time_gate_token":secrets.token_hex(32),
        "used":False,"training_authorized":True,
        "github_run_id":os.environ.get("GITHUB_RUN_ID"),
        "reservation_ref":ref,"fit_cap":FIT_CAP}
    # Only in temporary runner filesystem; NEVER commit or upload token-bearing gate.
    out=Path(out)
    if out.exists():raise FileExistsError("Authorization gate already exists")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf8") as f:json.dump(gate,f,indent=2)
    try:out.chmod(0o600)
    except OSError:pass
    return gate

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sha",required=True)
    p.add_argument("--branch",required=True)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if os.environ.get("GITHUB_ACTIONS")!="true":
        raise PermissionError("Only GitHub Actions environment permitted")
    token=os.environ.get("GH_TOKEN")
    gate=issue_gate(a.sha,a.branch,token,a.out)
    # No secret token logged; the following URL is safe provenance.
    print(json.dumps({"status":"BUDGET_RESERVED_ONCE","reservation_ref":gate["reservation_ref"],
        "sha":a.sha,"ci":gate["ci_url"],"authorization":gate["manager_authorization_url"]}))
if __name__=="__main__":main()
