"""Resource-capped, irreversibly budgeted E-CH4-14 stage launcher.

Only called AFTER two separate online verifies + GitHub reservation. This file
never runs in branch NO-FIT CI; it starts no work on import. The RSS monitor
covers the process tree and FAIL_STOP is preserved on forced termination.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from execution_gate import TAG_NAMES, github_get
from stage_execute import load_context, preflight

MAX_JOB_SECONDS = 20 * 60
POLL_SECONDS = 1.0


def validate_reservation(stage, context, *, fetch=github_get):
    """Require the once-only tag to exist, at the current full Git SHA."""
    if stage not in TAG_NAMES:
        raise PermissionError("Unknown stage")
    approved_ref = context.get("budget_reservation","")
    if not approved_ref.startswith("refs/tags/"):
        raise PermissionError("No approved stage reservation in context")
    ref = fetch(f"/repos/mahhammad/My-Thesis/git/ref/tags/{approved_ref.removeprefix('refs/tags/')}")
    expected = context.get("budget_reservation")
    allowed = {"refs/tags/" + TAG_NAMES[stage]}
    if stage == "raw_noise" and context.get("recovery_request") == "raw_noise_recovery_01":
        allowed.add("refs/tags/e-ch4-14-raw-noise-recovery-01-budget-spent")
    if expected not in allowed:
        raise PermissionError("Unapproved budget reference")
    if (ref.get("ref") != expected
            or ref.get("object", {}).get("sha") != context.get("source_sha")
            or context.get("budget_reservation") != expected):
        raise PermissionError("No matching GitHub-once stage reservation")
    return True


def _rss_tree_bytes(proc):
    import psutil
    parent = psutil.Process(proc.pid)
    members = [parent]
    try:
        members += parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _kill_tree(proc):
    import psutil
    try:
        p = psutil.Process(proc.pid)
        children = p.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    except psutil.NoSuchProcess:
        pass
    proc.wait(timeout=30)


def _append_fail_stop(stage, reason):
    """Record a watchdog-origin FAIL_STOP in the immutable spent ledger."""
    if stage != "fit":
        return
    path = HERE / "results/metrics/e14_spent_budget.jsonl"
    if not path.exists():
        return
    message = json.dumps({"event": "FAIL_STOP", "stage": stage,
                          "error_type": "RESOURCE_GUARD", "message": reason})
    with path.open("a", encoding="utf8") as out:
        out.write(message + "\n")
        out.flush()
        os.fsync(out.fileno())


def _fit_active_start(ledger_path, seen_offset, current_job_start):
    """Observed ledger START/SUCCESS transitions; never attempts a retry."""
    if not ledger_path.exists():
        return seen_offset, current_job_start
    with ledger_path.open("r", encoding="utf8") as f:
        f.seek(seen_offset)
        for line in f:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("event") == "START":
                current_job_start = time.monotonic()
            elif entry.get("event") in ("SUCCESS", "FAIL_STOP"):
                current_job_start = None
        seen_offset = f.tell()
    return seen_offset, current_job_start


def launch(stage, context_path, *, runner=subprocess.Popen):
    ctx, spec = load_context(context_path, stage)
    validate_reservation(stage, ctx)
    preflight(stage, ctx, spec)
    event = os.environ.get("GITHUB_EVENT_NAME")
    if event not in ("workflow_dispatch", "push"):
        raise PermissionError("Only a manual dispatch/explicit branch request can run")
    if event == "push" and (
            os.environ.get("E14_PUSH_REQUEST_VERIFIED") != "true"
            or os.environ.get("E14_STAGE") != stage):
        raise PermissionError("Push without exact verified launch request rejected")
    max_min = ctx["max_wall_minutes"]
    max_rss = ctx["max_rss_mib"]
    if type(max_min) is not int or not 1 <= max_min <= 330:
        raise PermissionError("Invalid owner/QA wall-clock budget")
    if type(max_rss) is not int or not 512 <= max_rss <= 6144:
        raise PermissionError("Invalid owner/QA RSS budget")
    if stage == "fit":
        command = [sys.executable, str(HERE/"runner.py"), "--execute",
                   "--authorization", str(context_path), "--source-sha", ctx["source_sha"]]
    else:
        command = [sys.executable, str(HERE/"stage_execute.py"), "--mode", "execute",
                   "--stage", stage, "--context", str(context_path)]
    begin = time.monotonic()
    proc = runner(command, cwd=ROOT, env=os.environ.copy())
    ledger = HERE / "results/metrics/e14_spent_budget.jsonl"
    ledger_offset, job_started = 0, None
    rss_max = 0
    try:
        while proc.poll() is None:
            rss_max = max(rss_max, _rss_tree_bytes(proc))
            elapsed = time.monotonic() - begin
            if elapsed > max_min*60:
                raise TimeoutError("Approved stage wall-clock budget exceeded")
            if rss_max > max_rss*1024*1024:
                raise MemoryError("Approved RSS budget exceeded")
            if stage == "fit":
                ledger_offset, job_started = _fit_active_start(ledger, ledger_offset, job_started)
                if job_started is not None and time.monotonic() - job_started > MAX_JOB_SECONDS:
                    raise TimeoutError("Approved 20-minute per-fit budget exceeded")
            time.sleep(POLL_SECONDS)
        if proc.returncode != 0:
            raise RuntimeError("Stage exited nonzero; automatic retry forbidden")
        print(json.dumps({"stage": stage, "status": "completed",
                          "elapsed_seconds": time.monotonic()-begin,
                          "peak_process_tree_rss_bytes": rss_max,
                          "test_accessed": False}))
        return 0
    except BaseException as exc:
        if proc.poll() is None:
            _kill_tree(proc)
        _append_fail_stop(stage, str(exc))
        raise


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("raw_noise","noise_masks","derived","fit"), required=True)
    p.add_argument("--context", type=Path, required=True)
    a = p.parse_args(argv)
    return launch(a.stage, a.context)


if __name__ == "__main__":
    raise SystemExit(main())
