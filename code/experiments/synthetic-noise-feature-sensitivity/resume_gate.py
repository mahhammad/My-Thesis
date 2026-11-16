"""Standalone one-time online authorization for E14 continuation 30->60.

Never accepts the old fit approval/tag, and never trains. No research secrets in
request validation, dependency install or preflight. Does not record token bytes.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, re, subprocess, urllib.error, urllib.request

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REPO="mahhammad/My-Thesis"
BRANCH="experiment/ch4-14-noise-sensitivity-01"
TAG="refs/tags/e-ch4-14-resume-30-60-budget-spent"
FROZEN="7bc66e4d2535fb15dfa5b569933b2d6c5a0962ac"
SPEC="82917272b0ac3850cecfd7bf4698dce55cbe955f8e3c308cf2c44d30e7f1ab8a"
OWNER_MARKER="E14_RESUME_OWNER_APPROVAL_V1\n"
QA_MARKER="E14_RESUME_QA_APPROVAL_V1\n"
REQUEST_PATH="code/experiments/synthetic-noise-feature-sensitivity/requests/resume_30_60.json"
MESSAGE="run(e-ch4-14): request resume_30_60"
WORKFLOW="E-CH4-14 authorized resume 30-60"
SHA=re.compile(r"[0-9a-f]{40}\Z")

def expect(value,message):
    if not value:raise PermissionError(message)

def git(*args):
    return subprocess.check_output(("git",)+args,cwd=ROOT,text=True).strip()

def github(method,path,payload=None):
    token=os.environ.get("GITHUB_TOKEN")
    expect(token and os.environ.get("GITHUB_ACTIONS")=="true","Only GitHub Actions authenticated request")
    raw=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request("https://api.github.com"+path,data=raw,method=method,
         headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github+json",
                  "X-GitHub-Api-Version":"2022-11-28","User-Agent":"e14-resume-one-time-gate",
                  **({"Content-Type":"application/json"} if raw is not None else {})})
    with urllib.request.urlopen(req,timeout=30) as res:
        return json.loads(res.read().decode())

def expected_checkpoint_set_sha():
    """SHA of sorted mapping id -> verified sidecar digest, all 36 origins."""
    import sys
    sys.path.insert(0,str(HERE))
    from runner import grid
    from data import file_sha
    result={}
    for subset,family in grid():
        name=subset+"__"+family
        cp=HERE/"results/checkpoints"/name/"step_30.pt"
        sha=(cp.with_suffix(".pt.sha256")).read_text().strip()
        expect(bool(re.fullmatch(r"[0-9a-f]{64}",sha)) and file_sha(cp)==sha,
               "Missing or modified original checkpoint: "+name)
        result[name]=sha
    expect(len(result)==36,"Origin checkpoint set incomplete")
    payload=json.dumps(result,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()

def compare_approvals(owner,qa,*,code_sha,ci_run_id,checkpoint_sha,token_digest):
    """Distinct authenticated comments; roles audited, never fabricated here."""
    base={"issue":90,"stage":"resume_only","approved_code_sha":code_sha,
      "frozen_results_sha":FROZEN,"data_spec_sha256":SPEC,
      "checkpoint_set_sha256":checkpoint_sha,"fits":36,
      "from_step":30,"to_step":60,"additional_updates":1080,
      "budget_ref":TAG,"ci_run_id":ci_run_id,"test_accessed":False}
    for name,doc in (("owner",owner),("qa",qa)):
        expect(isinstance(doc,dict),"Malformed "+name+" approval")
        for key,value in base.items():
            expect(type(doc.get(key)) is type(value) and doc[key]==value,
                   "Approval mismatch "+name+"."+key)
    expect(owner.get("schema")=="e-ch4-14-resume-owner-v1" and
       owner.get("vote")=="AUTHORIZE_RESUME_ONLY" and
       owner.get("secret_sha256")==token_digest,"Owner vote/token mismatch")
    expect(qa.get("schema")=="e-ch4-14-resume-qa-v1" and
       qa.get("vote")=="ACCEPT_RESUME_PROTOCOL_AND_BUDGET" and
       type(qa.get("owner_comment_id")) is int and
       qa.get("audit")=="INDEPENDENT_CODE_AND_ORIGIN_REVIEW",
       "Independent QA vote absent")
    return True

def parse_comment(comment,marker):
    body=comment.get("body","")
    expect(body.startswith(marker),"Wrong approval marker")
    record=json.loads(body[len(marker):].strip())
    expect(isinstance(record,dict),"Not an approval document")
    return record

def validate_push(event,request,*,head,parent,entries,message):
    """Single new request commit only; no hidden files or force push."""
    expect(event.get("after")==head and event.get("before")==parent,
           "GitHub push SHA does not match one-parent HEAD")
    expect(event.get("ref")=="refs/heads/"+BRANCH and
       event.get("repository",{}).get("full_name")==REPO and
       event.get("forced") is False and event.get("created") is False and
       event.get("deleted") is False,"Wrong push envelope")
    expect(event.get("head_commit",{}).get("id")==head and
       event.get("head_commit",{}).get("message")==MESSAGE and
       event.get("commits")==[event.get("commits",[{}])[0]] and
       len(event.get("commits",[]))==1 and
       event["commits"][0].get("id")==head,"Not exactly one request push")
    expect(message==MESSAGE and entries==["A\t"+REQUEST_PATH],
           "Exactly one added request file required")
    expect(set(request)=={"schema","stage","approved_code_sha","owner_comment_id",
                         "qa_comment_id","test_accessed"},"Unexpected request fields")
    expect(request["schema"]=="e-ch4-14-resume-request-v1" and
       request["stage"]=="resume_only" and request["test_accessed"] is False,
       "Wrong request schema/stage")
    expect(request["approved_code_sha"]==parent and
       bool(SHA.fullmatch(parent)) and bool(SHA.fullmatch(head)),
       "Approved code must be exact request parent")
    expect(all(type(request[k]) is int and request[k]>0
               for k in ("owner_comment_id","qa_comment_id")) and
       request["owner_comment_id"]!=request["qa_comment_id"],
       "Two distinct positive approval IDs required")
    return True

def validate_current_push():
    expect(os.environ.get("GITHUB_ACTIONS")=="true"
       and os.environ.get("GITHUB_EVENT_NAME")=="push"
       and os.environ.get("GITHUB_REF_NAME")==BRANCH
       and os.environ.get("GITHUB_WORKFLOW")==WORKFLOW,
       "No continuation workflow push context")
    head=git("rev-parse","HEAD")
    expect(head==os.environ.get("GITHUB_SHA"),"Checkout HEAD is not Actions SHA")
    parents=git("rev-list","--parents","-n","1","HEAD").split()
    expect(len(parents)==2 and parents[0]==head,"Not single-parent request commit")
    before=parents[1]
    entries=git("diff-tree","--no-commit-id","--name-status","--no-renames","-r","HEAD").splitlines()
    msg=git("show","-s","--format=%B","HEAD")
    event=json.loads(pathlib.Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    request=json.loads((ROOT/REQUEST_PATH).read_text())
    validate_push(event,request,head=head,parent=before,entries=entries,message=msg)
    return request,head,before

def check_frozen_git():
    """Original report/checkpoint bytes match the pre-continuation Git tree."""
    for base in ("code/experiments/synthetic-noise-feature-sensitivity/inputs",
                 "code/experiments/synthetic-noise-feature-sensitivity/results"):
        # Continution has a distinct output directory; at preflight none may exist.
        diff=git("diff","--name-only",FROZEN,"HEAD","--",base)
        expect(not diff,"Frozen original inputs/results differ from historical SHA")
    expect(not (HERE/"results/continuation_30_60").exists(),"Continuation already recorded")
    return True

def check_reviews(*,request,head,approved):
    """Live GitHub proof, source-bound and budget-bound; no secret stored."""
    from resume_30_60 import source_evidence
    source_evidence()
    check_frozen_git()
    checkpoint_sha=expected_checkpoint_set_sha()
    owner_id=request["owner_comment_id"];qa_id=request["qa_comment_id"]
    owner_comment=github("GET",f"/repos/{REPO}/issues/comments/{owner_id}")
    qa_comment=github("GET",f"/repos/{REPO}/issues/comments/{qa_id}")
    expect(owner_comment.get("user",{}).get("login")=="mahhammad",
           "Owner approval must be from actual repository owner")
    expect(qa_comment.get("user",{}).get("login"),"QA needs authenticated author")
    owner=parse_comment(owner_comment,OWNER_MARKER)
    qa=parse_comment(qa_comment,QA_MARKER)
    expect(qa.get("owner_comment_id")==owner_id and
           qa_comment.get("id")==qa_id and owner_comment.get("id")==owner_id,
           "Approval comment provenance mismatch")
    ci=github("GET",f"/repos/{REPO}/actions/runs/{owner['ci_run_id']}")
    expect(ci.get("head_sha")==approved and ci.get("conclusion")=="success"
           and ci.get("name")=="E-CH4-14 branch checks (NO FIT)",
           "Approved code SHA must have successful complete read-only CI")
    secret=os.environ.get("E14_RESUME_TOKEN")
    expect(secret and len(secret)>=24,"Independent resume-only secret missing")
    token_hash=hashlib.sha256(secret.encode("utf8")).hexdigest()
    compare_approvals(owner,qa,code_sha=approved,ci_run_id=owner["ci_run_id"],
                      checkpoint_sha=checkpoint_sha,token_digest=token_hash)
    # The new secret must not reuse the earlier fit secret: separate workflow key
    # is necessary, and the authorization is bound to this new tag and source SHA.
    try:
        tag=github("GET",f"/repos/{REPO}/git/ref/tags/{TAG.removeprefix('refs/tags/')}")
    except urllib.error.HTTPError as exc:
        if exc.code!=404:raise
    else:
        raise FileExistsError("Continuation one-time budget already spent")
    return dict(schema="e-ch4-14-resume-runtime-v1",stage="resume_only",
           approved_code_sha=approved,source_sha=head,budget_ref=TAG,
           owner_approval_url=f"https://github.com/{REPO}/issues/90#issuecomment-{owner_id}",
           qa_approval_url=f"https://github.com/{REPO}/issues/90#issuecomment-{qa_id}",
           checkpoint_set_sha256=checkpoint_sha,ci_run_id=owner["ci_run_id"],
           run_id=os.environ["GITHUB_RUN_ID"],test_accessed=False)

def reserve(context):
    """Globally atomic GitHub Git-tag reservation; never replay/retry."""
    result=github("POST",f"/repos/{REPO}/git/refs",
        dict(ref=TAG,sha=context["source_sha"]))
    expect(result.get("ref")==TAG and
           result.get("object",{}).get("sha")==context["source_sha"],
           "One-time tag creation not confirmed")
    return TAG

def verify_runtime_context(path):
    """Checks at actual execute entry AFTER the tag was successfully reserved."""
    temp=pathlib.Path(os.environ.get("RUNNER_TEMP","/missing-runner-temp")).resolve()
    target=pathlib.Path(path).resolve()
    expect(target.parent==temp and target.name=="e14_resume_verified.json",
           "Context must be in current Actions runner temp")
    ctx=json.loads(target.read_text())
    expect(ctx["schema"]=="e-ch4-14-resume-runtime-v1"
           and ctx["stage"]=="resume_only" and ctx["budget_ref"]==TAG
           and ctx["test_accessed"] is False
           and ctx["run_id"]==os.environ.get("GITHUB_RUN_ID")
           and ctx["source_sha"]==os.environ.get("GITHUB_SHA")
           and ctx["approved_code_sha"]==git("rev-parse","HEAD^"),
           "No verified context bound to exact request/run")
    ref=github("GET",f"/repos/{REPO}/git/ref/tags/{TAG.removeprefix('refs/tags/')}")
    expect(ref.get("ref")==TAG and ref.get("object",{}).get("sha")==ctx["source_sha"],
           "Budget tag not confirmed on current request SHA")
    return ctx

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("validate-request","reserve"),required=True)
    p.add_argument("--context",type=pathlib.Path)
    args=p.parse_args(argv)
    request,head,parent=validate_current_push()
    if args.mode=="validate-request":
        print("E14_RESUME_REQUEST_VALID",head,"No secret accessed; no fit")
        return
    expect(args.context is not None,"Explicit ephemeral context required")
    verified=check_reviews(request=request,head=head,approved=parent)
    from resume_30_60 import preflight
    preflight()  # All 36 origins checked before reservation.
    reserve(verified)
    target=pathlib.Path(args.context).resolve()
    require_dir=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve()
    expect(target.parent==require_dir and target.name=="e14_resume_verified.json" and not target.exists(),
           "Context path must be unique and private")
    with target.open("x",encoding="utf8") as f:
        json.dump(verified,f,indent=2,sort_keys=True)
        f.flush();os.fsync(f.fileno())
    os.chmod(target,0o600)
    print("E14_RESUME_ONCE_ONLY_RESERVED","Source verified, not yet trained")

if __name__=="__main__":
    main()
