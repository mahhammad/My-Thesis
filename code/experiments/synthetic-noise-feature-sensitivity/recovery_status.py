"""Emergency receipt emitted on every workflow completion path; never accesses secrets."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def safe_status(env=None):
    env=env or os.environ
    statuses = {name:env.get(name,"") for name in
                ("E14_STATUS_STAGE","E14_STATUS_REQUEST","E14_STATUS_RAW",
                 "E14_STATUS_MASK","E14_STATUS_DERIVED","E14_STATUS_FIT")}
    allowed={"","success","failure","cancelled","skipped",
             "raw_noise","noise_masks","derived","fit"}
    if any(value not in allowed for value in statuses.values()):
        raise PermissionError("Workflow stage/result status invalid")
    run_id=env.get("GITHUB_RUN_ID","")
    if not run_id.isdecimal():
        raise PermissionError("GitHub run ID missing")
    return {"schema":"e-ch4-14-recovery-status-v1","statuses":statuses,
            "workflow_run_id":run_id,"head_sha":env.get("GITHUB_SHA",""),
            "attempt":env.get("GITHUB_RUN_ATTEMPT",""),
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "test_accessed":False, "secret_values_included":False,
            "persistence_confirmed":False,
            "hard_job_timeout_could_prevent_this_step":True}


def main():
    target=Path(os.environ["RUNNER_TEMP"])/"e14_job_status.json"
    with target.open("x",encoding="utf8") as f:
        json.dump(safe_status(),f,indent=2,sort_keys=True)
        f.write("\n")
    print("E14_NON_SECRET_EMERGENCY_RECEIPT_READY")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
