"""Fail-closed validation of GitHub Actions push and single-file launch commit.

No secret ever enters this module. In particular no GitHub webhook file-change
arrays are assumed; all mutations and commit ancestry come from local Git.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BRANCH = "experiment/ch4-14-noise-sensitivity-01"
REPO = "mahhammad/My-Thesis"
DATA_STAGES = ("raw_noise", "noise_masks", "derived")
STAGES = DATA_STAGES + ("fit",)
SCHEMA = "e-ch4-14-branch-launch-request-v1"
REQUEST_PREFIX = "code/experiments/synthetic-noise-feature-sensitivity/requests/"
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


def _sha(s):
    return isinstance(s, str) and SHA_RE.fullmatch(s) is not None


def _stage_kind(stage, kind):
    if stage not in STAGES or kind not in ("data", "fit"):
        raise PermissionError("Unknown G4 stage or workflow")
    if (kind == "fit") != (stage == "fit"):
        raise PermissionError("Stage cannot execute via another workflow")


def validate_request(stage, request, *, kind, changed_entries, event,
                     checkout_sha, parents_line, commit_message):
    """Validate the real push envelope against independently collected Git facts.

    name-status is a list like ['A\\tpath'], NOT event.head_commit.added.
    """
    _stage_kind(stage, kind)
    path = f"{REQUEST_PREFIX}{stage}.json"
    if changed_entries != [f"A\t{path}"]:
        raise PermissionError("Push must add exactly one new request; no M/D/extra files")
    if not _sha(checkout_sha) or event.get("after") != checkout_sha:
        raise PermissionError("GitHub event after and checked-out HEAD disagree")
    pieces = parents_line.strip().split()
    if len(pieces) != 2 or pieces[0] != checkout_sha or not _sha(pieces[1]):
        raise PermissionError("Commit must have exactly one valid parent")
    parent = pieces[1]
    if (event.get("before") != parent or
            event.get("ref") != f"refs/heads/{BRANCH}" or
            event.get("repository", {}).get("full_name") != REPO or
            event.get("forced") is not False or
            event.get("deleted") is not False or
            event.get("created") is not False):
        raise PermissionError("Push ancestry/branch/force information mismatch")
    commits = event.get("commits")
    if (not isinstance(commits, list) or len(commits) != 1
            or commits[0].get("id") != checkout_sha):
        raise PermissionError("Exactly one SHA-matching pushed commit required")
    head = event.get("head_commit")
    if (not isinstance(head, dict) or head.get("id") != checkout_sha):
        raise PermissionError("Head commit SHA missing/mismatched")
    if (commit_message != f"run(e-ch4-14): request {stage}"
            or head.get("message") != commit_message):
        raise PermissionError("Incorrect manual request commit message")
    if not isinstance(request, dict) or set(request) != {
        "schema", "stage", "owner_comment_id", "qa_comment_id",
        "approved_code_sha", "test_accessed"
    }:
        raise PermissionError("Unexpected request fields")
    if (request["schema"] != SCHEMA or request["stage"] != stage
            or request["test_accessed"] is not False or
            not _sha(request["approved_code_sha"])
            or any(type(request[k]) is not int or request[k] < 1
                   for k in ("owner_comment_id", "qa_comment_id"))):
        raise PermissionError("Malformed stage/owner/QA/source request")
    return request


def validate_dispatch(stage, owner_id, qa_id, *, kind):
    _stage_kind(stage, kind)
    if any(type(v) is not int or v < 1 for v in (owner_id, qa_id)):
        raise PermissionError("Explicit owner/QA comment IDs required")
    return {"stage":stage, "owner_comment_id":owner_id,
            "qa_comment_id":qa_id, "approved_code_sha": ""}


def _git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,
                          capture_output=True,text=True).stdout


def _safe_int(raw):
    if not isinstance(raw, str) or not raw.isdecimal():
        raise PermissionError("Invalid numeric approval comment ID")
    return int(raw)


def _outputs(request, *, verified):
    """Only non-secret metadata can cross steps; never write GITHUB_ENV."""
    out = Path(os.environ["GITHUB_OUTPUT"])
    values = {
        "stage": request["stage"],
        "owner_comment_id": request["owner_comment_id"],
        "qa_comment_id": request["qa_comment_id"],
        "approved_code_sha": request["approved_code_sha"],
        "push_verified": "true" if verified else "false",
    }
    with out.open("a",encoding="utf8") as f:
        for k,v in values.items():
            val = str(v)
            if any(c in val for c in ("\r","\n")):
                raise PermissionError("Unsafe output metadata")
            f.write(f"{k}={val}\n")


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--kind",choices=("data","fit"),required=True)
    args=parser.parse_args(argv)
    if os.environ.get("GITHUB_REF_NAME") != BRANCH:
        raise PermissionError("Wrong E14 branch")
    event_name=os.environ.get("GITHUB_EVENT_NAME")
    if event_name == "push":
        event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text("utf8"))
        checkout_sha=_git("rev-parse","HEAD").strip()
        if checkout_sha != os.environ.get("GITHUB_SHA"):
            raise PermissionError("Workflow SHA does not match checked-out HEAD")
        parents=_git("rev-list","--parents","-n","1","HEAD")
        entries=_git("diff-tree","--no-commit-id","--name-status",
                     "--no-renames","-r","HEAD").splitlines()
        message=_git("show","-s","--format=%B","HEAD").rstrip("\n")
        if len(entries) != 1:
            raise PermissionError("Only one new Git path may be changed")
        match=re.fullmatch(re.escape(REQUEST_PREFIX)+
                           r"(raw_noise|noise_masks|derived|fit)\.json",
                           entries[0].split("\t",1)[-1])
        if match is None:
            raise PermissionError("Exact E14 request filename required")
        stage=match.group(1)
        path=ROOT / (REQUEST_PREFIX+stage+".json")
        request=validate_request(stage,json.loads(path.read_text("utf8")),
                      kind=args.kind,changed_entries=entries,event=event,
                      checkout_sha=checkout_sha,parents_line=parents,
                      commit_message=message)
        verified=True
    elif event_name == "workflow_dispatch":
        stage=os.environ.get("E14_INPUT_STAGE","")
        request=validate_dispatch(stage,
                 _safe_int(os.environ.get("E14_INPUT_OWNER","")),
                 _safe_int(os.environ.get("E14_INPUT_QA","")),kind=args.kind)
        verified=False
    else:
        raise PermissionError("Only explicit branch request or manual dispatch")
    _outputs(request,verified=verified)
    print("E14_REQUEST_METADATA_VALID",request["stage"],
          "No secret read; no research stage started")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
