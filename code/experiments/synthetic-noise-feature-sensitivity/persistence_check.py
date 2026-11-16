"""Post-success E14 evidence gate. Must pass AFTER artifact upload and git push.

This is a cheap structural and hash check, not a substitute for independent
scientific QA. The remote branch MUST contain this run's immutable evidence.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BRANCH="experiment/ch4-14-noise-sensitivity-01"
STAGES=("raw_noise","noise_masks","derived","fit")


def sha_file(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def required_files(stage,root=HERE):
    root=Path(root)
    inputs=root/"inputs"
    results=root/"results"
    if stage=="raw_noise":
        return [inputs/"noise_raw"/x for x in
                ("new_noise_seed20261102.npz","manifest.json")]
    if stage=="noise_masks":
        return [inputs/"noise_masks"/x for x in
                ("new_noise_masks.npz","manifest.json")]
    if stage=="derived":
        from data import SUBSETS
        return [inputs/"derived"/"manifest.json"] + [
            inputs/"derived"/(sid+".npz") for sid,_ in SUBSETS]
    if stage=="fit":
        from runner import grid
        files=[results/"metrics"/file for file in (
            "e14_spent_budget.jsonl","metrics_full.json","coverage.json",
            "metrics_summary.csv","metrics_grouped.csv","metrics_paired.csv",
            "metrics_paired_by_k.csv","metrics_analysis.json","provenance.json")]
        for subset,family in grid():
            job=f"{subset}__{family}"
            files.extend([results/"traces"/job/"trace.json",
                          results/"traces"/job/"trace.csv",
                          results/"checkpoints"/job/"step_30.pt",
                          results/"checkpoints"/job/"step_30.pt.sha256",
                          results/"metrics"/(job+".json")])
        return files
    raise ValueError("Invalid stage")


def check_evidence(stage,root=HERE):
    root=Path(root)
    files=required_files(stage,root)
    missing=[str(f) for f in files if not f.is_file() or f.stat().st_size==0]
    if missing:
        raise FileNotFoundError("Incomplete E14 evidence: "+", ".join(missing[:5]))
    if stage in ("raw_noise","noise_masks"):
        name="noise_raw" if stage=="raw_noise" else "noise_masks"
        item="new_noise_seed20261102.npz" if stage=="raw_noise" else "new_noise_masks.npz"
        manifest=json.loads((root/"inputs"/name/"manifest.json").read_text("utf8"))
        if sha_file(root/"inputs"/name/item)!=manifest.get("file_sha256"):
            raise ValueError("Archived stage asset checksum differs from manifest")
    elif stage=="derived":
        man=json.loads((root/"inputs"/"derived"/"manifest.json").read_text("utf8"))
        if len(man.get("datasets",[]))!=18:
            raise ValueError("Not exactly 18 derived datasets")
        for row in man["datasets"]:
            file=root/"inputs"/"derived"/(row["subset_id"]+".npz")
            if sha_file(file)!=row.get("file_sha256"):
                raise ValueError("Derived file SHA mismatch")
    else:
        from runner import grid
        ledger=root/"results"/"metrics"/"e14_spent_budget.jsonl"
        rows=[json.loads(line) for line in ledger.read_text("utf8").splitlines()]
        if (len([r for r in rows if r.get("event")=="BUDGET_RESERVED"])!=1
                or len([r for r in rows if r.get("event")=="UPDATE"])!=1080
                or len([r for r in rows if r.get("event")=="SUCCESS"])!=36
                or any(r.get("event")=="FAIL_STOP" for r in rows)):
            raise ValueError("The 1080-update 36-job budget ledger is incomplete")
        for sid,fam in grid():
            job=f"{sid}__{fam}"
            updates=[r["updates_completed"] for r in rows
                     if r.get("event")=="UPDATE" and r.get("run_id")==job]
            if updates!=list(range(1,31)):
                raise ValueError("Missing or out-of-order per-job update")
            trace=json.loads((root/"results"/"traces"/job/"trace.json").read_text("utf8"))["trace"]
            if len(trace)!=31 or [r["step"] for r in trace]!=list(range(31)):
                raise ValueError("Trace steps invalid")
            pt=root/"results"/"checkpoints"/job/"step_30.pt"
            digest=pt.with_suffix(".pt.sha256").read_text("utf8").split()[0]
            if sha_file(pt)!=digest:
                raise ValueError("Checkpoint digest mismatch")
    return len(files)


def remote_exact(local_sha, *, runner=subprocess.check_output):
    remote=runner(["git","ls-remote","--exit-code","origin",
                   f"refs/heads/{BRANCH}"],cwd=ROOT,text=True)
    tokens=remote.strip().split()
    if len(tokens)!=2 or tokens[0]!=local_sha or tokens[1]!=f"refs/heads/{BRANCH}":
        raise RuntimeError("Remote research branch does not contain the preserved evidence")
    return True


def check_complete(stage,artifact_id,*,root=HERE,local_sha=None,runner=subprocess.check_output):
    if stage not in STAGES or not isinstance(artifact_id,str) or not artifact_id.isdecimal() or int(artifact_id)<1:
        raise PermissionError("Independent uploaded artifact confirmation missing")
    count=check_evidence(stage,root)
    if local_sha is None:
        local_sha=runner(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if len(local_sha)!=40 or any(c not in "0123456789abcdef" for c in local_sha):
        raise PermissionError("Local preserved evidence SHA missing")
    remote_exact(local_sha,runner=runner)
    return {"stage":stage,"artifact_id":artifact_id,"required_files_checked":count,
            "committed_sha":local_sha,"persistence_confirmed":True}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=STAGES,required=True)
    p.add_argument("--kind",choices=("data","fit"),required=True)
    args=p.parse_args(argv)
    if (args.kind=="data")!=(args.stage!="fit"):
        raise PermissionError("Wrong workflow for E14 persistence check")
    result=check_complete(args.stage,os.environ.get("E14_ARTIFACT_ID",""))
    print("E14_PERSISTENCE_CONFIRMED",json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
