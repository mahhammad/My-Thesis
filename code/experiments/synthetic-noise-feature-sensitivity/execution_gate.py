"""E-CH4-14 online one-time authorization gate (Issue #90).

Only GitHub's live REST API is authoritative; URLs printed in a JSON document
are NOT evidence. This module performs no scientific work. The 'verify' command
is read-only and 'reserve' exclusively creates an immutable-by-policy Git ref
BEFORE ANY research write/fit; retries must use a new explicit research budget.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from data import NoiseSpec, load_mother, spec_sha

REPO = "mahhammad/My-Thesis"
ISSUE = 90
BRANCH = "experiment/ch4-14-noise-sensitivity-01"
STAGES = ("raw_noise", "noise_masks", "derived", "fit")
OWNER_MARKER = "E14_OWNER_APPROVAL_V1\n"
QA_MARKER = "E14_QA_APPROVAL_V1\n"
OWNER_SCHEMA = "e-ch4-14-owner-approval-v1"
QA_SCHEMA = "e-ch4-14-independent-qa-v1"
EXPECTED_WORKFLOWS = {
    "raw_noise": "E-CH4-14 authorized data generation",
    "noise_masks": "E-CH4-14 authorized data generation",
    "derived": "E-CH4-14 authorized data generation",
    "fit": "E-CH4-14 authorized 36-fit run",
}
TAG_NAMES = {stage: "e-ch4-14-" + stage.replace("_", "-") + "-budget-spent"
             for stage in STAGES}
RECOVERY = {
    "request": "raw_noise_recovery_01",
    "failed_run_id": 43836997795,
    "failed_request_sha": "5a21ef3cb4a248e46141ab4fca366162ede95a33",
    "budget_tag": "e-ch4-14-raw-noise-recovery-01-budget-spent",
}
def _budget_tag(stage, owner, qa, *, fetch):
    """Independent once-only budget for witnessed pre-generation failure only."""
    recovery = owner.get("raw_noise_recovery")
    if recovery is None:
        return TAG_NAMES[stage], None
    if stage != "raw_noise" or recovery != RECOVERY or qa.get("raw_noise_recovery") != RECOVERY:
        raise PermissionError("Unapproved recovery or mismatched QA decision")
    if (os.environ.get("E14_REQUEST_RECOVERY") != RECOVERY["request"]
            or os.environ.get("GITHUB_EVENT_NAME") != "push"):
        raise PermissionError("Recovery requires its own new one-file request")
    tag = fetch(f"/repos/{REPO}/git/ref/tags/{TAG_NAMES['raw_noise']}")
    if (tag.get("ref") != "refs/tags/" + TAG_NAMES["raw_noise"] or
            tag.get("object", {}).get("sha") != RECOVERY["failed_request_sha"]):
        raise PermissionError("Previous immutable spent tag does not match failed attempt")
    run = fetch(f"/repos/{REPO}/actions/runs/{RECOVERY['failed_run_id']}")
    if (run.get("head_sha") != RECOVERY["failed_request_sha"] or
            run.get("conclusion") != "failure" or
            run.get("name") != EXPECTED_WORKFLOWS["raw_noise"]):
        raise PermissionError("Historical failed attempt not proven online")
    return RECOVERY["budget_tag"], RECOVERY["request"]


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def github_get(path, *, token=None, transport=None):
    """Read authenticated GitHub JSON; raise on 403, malformed JSON and failures."""
    if transport is not None:
        return transport("GET", path, None)
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise PermissionError("GITHUB_TOKEN required for authenticated live issue verification")
    req = Request("https://api.github.com" + path, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer " + token,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "E14-verified-scientific-gate",
    })
    with urlopen(req, timeout=20) as response:
        return json.load(response)


def github_post(path, payload, *, token=None, transport=None):
    if transport is not None:
        return transport("POST", path, payload)
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise PermissionError("No authenticated GitHub token")
    req = Request("https://api.github.com" + path, data=json.dumps(payload).encode(),
                  method="POST", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "E14-verified-scientific-gate",
    })
    with urlopen(req, timeout=20) as response:
        return json.load(response)


def _comment(comment_id, marker, *, fetch):
    if type(comment_id) is not int or comment_id <= 0:
        raise PermissionError("Explicit numeric Issue #90 comment ID is required")
    row = fetch(f"/repos/{REPO}/issues/comments/{comment_id}")
    if (row.get("issue_url") != f"https://api.github.com/repos/{REPO}/issues/{ISSUE}"
            or row.get("id") != comment_id or
            row.get("created_at") != row.get("updated_at")):
        raise PermissionError("Comment is wrong Issue, edited or unverified")
    body = row.get("body", "")
    if not isinstance(body, str) or not body.startswith(marker):
        raise PermissionError("Missing exact machine-readable signed decision marker")
    try:
        payload = json.loads(body[len(marker):])
    except (ValueError, TypeError) as e:
        raise PermissionError("Decision payload is not a single JSON object") from e
    if not isinstance(payload, dict):
        raise PermissionError("Decision must be a JSON object")
    return row, payload


def _sha(value):
    return (isinstance(value, str) and
            re.fullmatch("[0-9a-f]{40}", value) is not None)


def _verify_code_ancestry(approved, current, *, git_check=True, diff_paths=None):
    if not _sha(approved) or not _sha(current):
        raise PermissionError("Valid 40-character code and current SHAs required")
    if approved == current:
        return
    if git_check:
        proc = subprocess.run(["git", "merge-base", "--is-ancestor", approved, current],
                              cwd=ROOT, check=False)
        if proc.returncode != 0:
            raise PermissionError("Running SHA not descendant of approved source")
        result = subprocess.run(["git", "diff", "--name-only", approved, current],
                                cwd=ROOT, check=True, text=True, capture_output=True)
        changed = result.stdout.splitlines()
    elif diff_paths is not None:
        changed = diff_paths
    else:
        raise PermissionError("Unverified ancestry cannot be accepted")
    prefix = "code/experiments/synthetic-noise-feature-sensitivity/"
    allowed = (prefix + "inputs/noise_raw/", prefix + "inputs/noise_masks/",
               prefix + "inputs/derived/", prefix + "requests/", prefix + "results/",
               prefix + "tables/", prefix + "figures/")
    if any(not name.startswith(allowed) for name in changed):
        raise PermissionError("Unapproved code/workflow change after approved SHA")


def verify(stage, owner_comment_id, qa_comment_id, *, fetch, token,
           current_sha, run_id, workflow, branch, event, git_check=True,
           diff_paths=None):
    """Verify live owner, independent auditor, code, CI, exact budget, split and token."""
    if stage not in STAGES:
        raise PermissionError("Unknown E14 execution stage")
    if (workflow != EXPECTED_WORKFLOWS[stage] or branch != BRANCH
            or event not in ("workflow_dispatch", "push") or not str(run_id).isdigit()):
        raise PermissionError("Only the exact manual E14 stage workflow is authorized")
    if not isinstance(token, str) or len(token) < 32:
        raise PermissionError("High-entropy per-stage one-time token is missing")
    repository = fetch(f"/repos/{REPO}")
    if repository.get("full_name") != REPO or repository.get("owner", {}).get("login") != "mahhammad":
        raise PermissionError("Repository owner identity mismatch")
    owner_row, approval = _comment(owner_comment_id, OWNER_MARKER, fetch=fetch)
    qa_row, qa = _comment(qa_comment_id, QA_MARKER, fetch=fetch)
    owner_login = repository["owner"]["login"]
    qa_login = approval.get("independent_auditor_login")
    # Independent review means a distinct evidence-based audit and decision,
    # not necessarily a separate GitHub account. Research roles in this
    # project share the repository owner's login.
    if (owner_comment_id == qa_comment_id or
            owner_row.get("user", {}).get("login") != owner_login or
            not isinstance(qa_login, str) or not qa_login or
            qa_row.get("user", {}).get("login") != qa_login):
        raise PermissionError("Owner and QA decisions must be distinct verified comments")
    if event == "push" and (
            os.environ.get("E14_PUSH_REQUEST_VERIFIED") != "true"
            or os.environ.get("E14_STAGE") != stage
            or os.environ.get("E14_REQUEST_APPROVED_CODE_SHA") != approval.get("approved_code_sha")):
        raise PermissionError("Push event requires validated, SHA-bound manual request")
    code_sha = approval.get("approved_code_sha")
    _verify_code_ancestry(code_sha, current_sha, git_check=git_check,
                          diff_paths=diff_paths)
    if (approval.get("schema") != OWNER_SCHEMA or approval.get("issue") != ISSUE
            or approval.get("repo") != REPO or approval.get("branch") != BRANCH
            or approval.get("approved_stages") != list(STAGES)
            or approval.get("stop_on_failure") is not True
            or approval.get("no_auto_retry") is not True
            or approval.get("sequential_fits") is not True
            or approval.get("test_accessed") is not False
            or approval.get("n_train") != 300 or approval.get("n_validation") != 90
            or approval.get("fit_count") != 36 or approval.get("updates_per_fit") != 30
            or approval.get("total_update_budget") != 1080):
        raise PermissionError("Incomplete or mismatched owner full-protocol decision")
    try:
        spec = NoiseSpec(**approval["noise_spec"]).validate()
    except (TypeError, KeyError, ValueError) as e:
        raise PermissionError("Owner noise distribution/params not approved") from e
    digest = spec_sha(spec)
    if approval.get("noise_spec_sha256") != digest:
        raise PermissionError("Owner spec fingerprint mismatch")
    wall = approval.get("wall_minutes")
    rss = approval.get("max_rss_mib")
    if (type(wall) is not int or not 1 <= wall <= 330 or
            type(rss) is not int or not 512 <= rss <= 6144):
        raise PermissionError("Unapproved or unsafe execution wall/RSS limits")
    tokens = approval.get("stage_token_sha256")
    if (not isinstance(tokens, dict) or set(tokens) != set(STAGES) or
            tokens[stage] != _hash(stage + ":" + token)):
        raise PermissionError("Incorrect stage-specific one-time token")
    if (qa.get("schema") != QA_SCHEMA or qa.get("issue") != ISSUE
            or qa.get("owner_comment_id") != owner_comment_id
            or qa.get("approved_code_sha") != code_sha
            or qa.get("noise_spec_sha256") != digest
            or qa.get("vote") != "ACCEPT_G4_AND_PROTOCOL"
            or qa.get("approved_stages") != list(STAGES)
            or qa.get("max_rss_mib") != rss
            or qa.get("wall_minutes") != wall):
        raise PermissionError("No matching independent QA on exact protocol/source")
    ci_id = qa.get("ci_run_id")
    if type(ci_id) is not int or ci_id <= 0:
        raise PermissionError("An independently verified CI run ID is required")
    ci = fetch(f"/repos/{REPO}/actions/runs/{ci_id}")
    if (ci.get("head_sha") != code_sha or ci.get("conclusion") != "success"
            or ci.get("name") != "E-CH4-14 branch checks (NO FIT)"
            or ci.get("head_branch") != BRANCH):
        raise PermissionError("Online CI not successful for approved code SHA")
    budget_tag, recovery_request = _budget_tag(stage, approval, qa, fetch=fetch)
    mother = load_mother()  # immutable read-only checksum; never generates a split.
    ctx = {
        "issue": ISSUE, "stage": stage, "noise_spec": asdict(spec),
        "source_sha": current_sha, "approved_code_sha": code_sha,
        "approved_spec_sha256": digest, "data_spec_sha256": digest,
        "owner_approved_data_sha256": mother.source_sha,
        "owner_approval_url": owner_row["html_url"],
        "independent_qa_url": qa_row["html_url"],
        "authorized_fits": 36, "updates_per_fit": 30,
        "test_accessed": False, "workflow_run_id": str(run_id),
        "one_time_token_sha256": _hash(token),
        "max_wall_minutes": wall, "max_rss_mib": rss,
        "budget_reservation": "refs/tags/" + budget_tag,
        "recovery_request": recovery_request,
        "owner_comment_id": owner_comment_id, "qa_comment_id": qa_comment_id,
    }
    # Canonical URL is checked against actual returned GitHub records.
    for field, row in (("owner_approval_url", owner_row),
                       ("independent_qa_url", qa_row)):
        expected = f"https://github.com/{REPO}/issues/{ISSUE}#issuecomment-{row['id']}"
        if ctx[field] != expected:
            raise PermissionError("GitHub comment URL does not match confirmed ID")
    return ctx


def verify_unspent(stage, *, fetch, budget_ref=None):
    if stage not in STAGES:
        raise PermissionError("Unknown stage")
    budget_ref = budget_ref or ("refs/tags/" + TAG_NAMES[stage])
    if budget_ref not in ("refs/tags/" + TAG_NAMES[stage],
                          "refs/tags/" + RECOVERY["budget_tag"]):
        raise PermissionError("Unrecognized once-only budget reference")
    if stage != "raw_noise" and budget_ref.endswith(RECOVERY["budget_tag"]):
        raise PermissionError("Recovery budget cannot pay for another stage")
    try:
        fetch(f"/repos/{REPO}/git/ref/tags/{budget_ref.removeprefix('refs/tags/')}")
    except HTTPError as e:
        if e.code == 404:
            return True
        raise
    raise FileExistsError("E14 stage budget already reserved: retry forbidden")


def reserve(stage, current_sha, *, fetch, post, budget_ref=None):
    """Perform globally exclusive once-only reservation; never reset an old tag."""
    ref = budget_ref or ("refs/tags/" + TAG_NAMES[stage])
    verify_unspent(stage, fetch=fetch, budget_ref=ref)
    result = post(f"/repos/{REPO}/git/refs", {"ref": ref, "sha": current_sha})
    if result.get("ref") != ref or result.get("object", {}).get("sha") != current_sha:
        raise RuntimeError("GitHub did not confirm stage reservation")
    return ref


def _env_input(name):
    value = os.environ.get(name)
    if not value:
        raise PermissionError("Missing workflow-only environment " + name)
    return value


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("verify", "reserve"), required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--owner-comment-id", type=int, required=True)
    p.add_argument("--qa-comment-id", type=int, required=True)
    p.add_argument("--context", type=Path, required=True)
    args = p.parse_args(argv)
    temp = Path(_env_input("RUNNER_TEMP")).resolve()
    ctx_path = args.context.resolve()
    if ctx_path.parent != temp or ctx_path.name != "e14_verified_context.json":
        raise PermissionError("Verified approval context may only reside in runner temp")
    if _env_input("GITHUB_SHA") != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip():
        raise PermissionError("Checkout HEAD is not workflow SHA")
    fetch = lambda path: github_get(path)
    ctx = verify(args.stage, args.owner_comment_id, args.qa_comment_id,
                 fetch=fetch, token=_env_input("E14_STAGE_TOKEN"),
                 current_sha=_env_input("GITHUB_SHA"), run_id=_env_input("GITHUB_RUN_ID"),
                 workflow=_env_input("GITHUB_WORKFLOW"),
                 branch=_env_input("GITHUB_REF_NAME"),
                 event=_env_input("GITHUB_EVENT_NAME"))
    if args.mode == "verify":
        verify_unspent(args.stage, fetch=fetch,
                       budget_ref=ctx["budget_reservation"])
        if ctx_path.exists():
            raise FileExistsError("Refusing stale approval context")
        ctx_path.write_text(json.dumps(ctx, sort_keys=True, indent=2) + "\n", encoding="utf8")
        os.chmod(ctx_path, 0o600)
        print("E14_ONLINE_VERIFICATION_PASS", args.stage, ctx["approved_code_sha"],
              "G4 only; no research stage executed")
    else:
        if not ctx_path.is_file():
            raise PermissionError("No previous verification context in current run")
        previous = json.loads(ctx_path.read_text(encoding="utf8"))
        if previous != ctx:
            raise PermissionError("Owner/QA/CI decision drift since verification")
        ref = reserve(args.stage, ctx["source_sha"], fetch=fetch,
                      post=lambda path, payload: github_post(path, payload),
                      budget_ref=ctx["budget_reservation"])
        ctx["budget_reservation"] = ref
        ctx_path.write_text(json.dumps(ctx, sort_keys=True, indent=2) + "\n",
                            encoding="utf8")
        print("E14_ONCE_ONLY_RESERVATION_CONFIRMED", ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
