"""Manual-by-commit, branch-local E14 launch request validation.

This is the alternative to workflow_dispatch when the workflow exists only in
a research branch: a single, new requests/<stage>.json commit triggers a
branch-limited push job. Ordinary pushes never run research stages.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
BRANCH = "experiment/ch4-14-noise-sensitivity-01"
DATA_STAGES = ("raw_noise", "noise_masks", "derived")
STAGES = DATA_STAGES + ("fit",)
SECRETS = {
    "raw_noise": "E14_RAW_NOISE_TOKEN",
    "noise_masks": "E14_NOISE_MASKS_TOKEN",
    "derived": "E14_DERIVED_TOKEN",
    "fit": "E14_FIT_TOKEN",
}
SCHEMA = "e-ch4-14-branch-launch-request-v1"


def validate_request(stage, request, *, kind, changed_paths, event):
    """No source, Test, data, or other file change is accepted in a launch push."""
    if stage not in STAGES or kind not in ("data", "fit"):
        raise PermissionError("Unknown E14 launch stage")
    if (kind == "data" and stage not in DATA_STAGES
            or kind == "fit" and stage != "fit"):
        raise PermissionError("Stage not allowed for this workflow")
    expected = f"code/experiments/synthetic-noise-feature-sensitivity/requests/{stage}.json"
    if changed_paths != [expected]:
        raise PermissionError("Launch must be a single new request file; no other changes")
    if not isinstance(request, dict) or set(request) != {
        "schema", "stage", "owner_comment_id", "qa_comment_id",
        "approved_code_sha", "test_accessed"
    }:
        raise PermissionError("Malformed branch launch request")
    if request["schema"] != SCHEMA or request["stage"] != stage or request["test_accessed"] is not False:
        raise PermissionError("Request schema/stage/Test mismatch")
    if (type(request["owner_comment_id"]) is not int or
            type(request["qa_comment_id"]) is not int or
            request["owner_comment_id"] < 1 or request["qa_comment_id"] < 1):
        raise PermissionError("Valid exact Issue #90 owner and QA comment IDs required")
    if not isinstance(request["approved_code_sha"],str) or not re.fullmatch("[0-9a-f]{40}",request["approved_code_sha"]):
        raise PermissionError("Full approved source SHA required")
    if (event.get("deleted") is True or event.get("forced") is True or
            len(event.get("commits", [])) != 1 or
            event.get("head_commit",{}).get("added") != [expected] or
            event.get("head_commit",{}).get("modified") or
            event.get("head_commit",{}).get("removed") or
            event.get("head_commit",{}).get("message") !=
                 f"run(e-ch4-14): request {stage}"):
        raise PermissionError("Only an explicit one-file launch commit is permitted")
    return request


def validate_dispatch(stage, owner_id, qa_id, *, kind):
    if stage not in STAGES or (kind == "fit") != (stage == "fit"):
        raise PermissionError("Wrong workflow for requested stage")
    if (type(owner_id) is not int or type(qa_id) is not int
            or owner_id < 1 or qa_id < 1):
        raise PermissionError("Owner and independent QA comment IDs required")
    return {"stage": stage, "owner_comment_id": owner_id, "qa_comment_id": qa_id}


def _env_file(values):
    target = Path(os.environ["GITHUB_ENV"])
    with target.open("a",encoding="utf8") as out:
        for name, value in values.items():
            value = str(value)
            if any(c in value for c in ("\n", "\r")):
                raise PermissionError("Unsafe workflow environment value")
            out.write(name + "=" + value + "\n")


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--kind",choices=("data","fit"),required=True)
    args=p.parse_args(argv)
    if os.environ.get("GITHUB_REF_NAME") != BRANCH:
        raise PermissionError("No branch-local launch off E14 branch")
    event_name=os.environ.get("GITHUB_EVENT_NAME")
    if event_name=="workflow_dispatch":
        stage=os.environ.get("E14_INPUT_STAGE","")
        owner=int(os.environ.get("E14_INPUT_OWNER","0"))
        qa=int(os.environ.get("E14_INPUT_QA","0"))
        request=validate_dispatch(stage,owner,qa,kind=args.kind)
        verified="false"
    elif event_name=="push":
        event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text("utf8"))
        root=HERE.parents[2]
        sha=os.environ.get("GITHUB_SHA")
        if event.get("after") != sha or not isinstance(sha,str):
            raise PermissionError("GitHub push payload/head SHA mismatch")
        diff=subprocess.run(["git","diff-tree","--no-commit-id","--name-only","-r",sha],
                            cwd=root,text=True,capture_output=True,check=True)
        changed=sorted(diff.stdout.splitlines())
        if len(changed)!=1:
            raise PermissionError("Push must introduce exactly one request file")
        file=changed[0]
        match=re.fullmatch(
            r"code/experiments/synthetic-noise-feature-sensitivity/requests/(raw_noise|noise_masks|derived|fit)\.json",
            file)
        if match is None:
            raise PermissionError("No request file in exact E14 path")
        stage=match.group(1)
        body=json.loads((root/file).read_text("utf8"))
        request=validate_request(stage,body,kind=args.kind,
                                 changed_paths=changed,event=event)
        verified="true"
    else:
        raise PermissionError("Only manual dispatch or one-file manual push request allowed")
    stage=request["stage"]
    token=os.environ.get(SECRETS[stage],"")
    if len(token)<32 or any(c in token for c in ("\n","\r")):
        raise PermissionError("Missing or invalid per-stage GitHub secret; never launch")
    output={"E14_STAGE":stage,
            "E14_OWNER_COMMENT_ID":request["owner_comment_id"],
            "E14_QA_COMMENT_ID":request["qa_comment_id"],
            "E14_STAGE_TOKEN":token,
            "E14_PUSH_REQUEST_VERIFIED":verified}
    _env_file(output)
    print("E14_EXPLICIT_STAGE_REQUEST_VALID",stage,"event",event_name,
          "no data / fit performed in this step")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
