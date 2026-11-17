"""Bounded no-retry monitor for the independently authorized E14 resume process.

The monitor cannot reserve budget or authorize a run. It only limits an already
authorized child and preserves FAIL_STOP evidence if memory/time is exceeded.
"""
from __future__ import annotations
import argparse,json,os,pathlib,signal,subprocess,sys,time
import psutil

HERE=pathlib.Path(__file__).resolve().parent
OUTPUT=HERE/"results/continuation_30_60"
MAX_TOTAL_S=300*60
MAX_PER_JOB_S=20*60
MAX_RSS=6144*1024**2
POLL_S=2

def require(value,message):
    if not value:raise PermissionError(message)

def observed(ledger):
    if not ledger.is_file():return None
    try:
        records=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    except (ValueError,OSError):
        return None
    starts=[x for x in records if x.get("event")=="START"]
    done={x.get("run_id") for x in records if x.get("event")=="SUCCESS"}
    return starts[-1].get("run_id") if starts and starts[-1].get("run_id") not in done else None

def current_rss(pid):
    try:
        p=psutil.Process(pid)
        descendants=[p]+p.children(recursive=True)
        return sum(q.memory_info().rss for q in descendants if q.is_running())
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        return 0

def terminate_group(proc):
    if proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait(timeout=10)

def fail_stop(reason):
    ledger=OUTPUT/"resume_budget.jsonl"
    if not ledger.exists():return
    try:
        events=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
        if any(x.get("event")=="FAIL_STOP" for x in events):return
        with ledger.open("a",encoding="utf8") as out:
            out.write(json.dumps({"event":"FAIL_STOP","error_type":"ResourceWatchdog",
                                  "message":reason,"test_accessed":False})+"\n")
            out.flush();os.fsync(out.fileno())
    except (OSError,ValueError):
        # Never hide the original watchdog error.
        pass

def run(context):
    require(os.environ.get("E14_RESUME_EXECUTE_VERIFIED")=="true",
            "Resume watchdog not authorized")
    require(os.environ.get("GITHUB_ACTIONS")=="true","Only real GitHub Actions")
    args=[sys.executable,str(HERE/"resume_30_60.py"),"--mode","execute",
          "--context",str(context)]
    proc=subprocess.Popen(args,cwd=HERE.parents[2],start_new_session=True)
    start=time.monotonic();last_job=None;job_start=start;maxrss=0
    try:
        while proc.poll() is None:
            elapsed=time.monotonic()-start
            rss=current_rss(proc.pid);maxrss=max(maxrss,rss)
            job=observed(OUTPUT/"resume_budget.jsonl")
            if job!=last_job:
                last_job=job;job_start=time.monotonic()
            if rss>MAX_RSS:raise RuntimeError("RSS_LIMIT_EXCEEDED")
            if elapsed>MAX_TOTAL_S:raise RuntimeError("TOTAL_WALLTIME_EXCEEDED")
            if job is not None and time.monotonic()-job_start>MAX_PER_JOB_S:
                raise RuntimeError("PER_FIT_WALLTIME_EXCEEDED:"+job)
            time.sleep(POLL_S)
        if proc.returncode!=0:
            raise RuntimeError("RESUME_CHILD_NONZERO_EXIT:"+str(proc.returncode))
        print("E14_RESUME_RESOURCE_PASS",json.dumps(
          {"max_aggregate_rss_mib":round(maxrss/1048576,2),
           "wall_seconds":round(time.monotonic()-start,2),
           "test_accessed":False}),flush=True)
    except BaseException as e:
        terminate_group(proc);fail_stop(str(e)[:200]);raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--context",type=pathlib.Path,required=True)
    a=p.parse_args();run(a.context)

if __name__=="__main__":
    main()
