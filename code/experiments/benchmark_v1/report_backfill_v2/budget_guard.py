#!/usr/bin/env python3
"""RESEARCHER-03: process-tree RSS/time guard; audit log even on failure."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,os
from pathlib import Path
import signal,subprocess,sys,time

def read_rss_kib(pid:int)->int:
    try:
        with open(f"/proc/{pid}/status",encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        pass
    return 0

def descendants(pid:int):
    all_pids=set()
    queue=[pid]
    while queue:
        cur=queue.pop()
        if cur in all_pids: continue
        all_pids.add(cur)
        try:
            children=Path(f"/proc/{cur}/task/{cur}/children").read_text().split()
            queue.extend(int(x) for x in children)
        except (OSError,ValueError):
            pass
    return all_pids

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--seconds",type=float,required=True)
    a.add_argument("--rss-mib",type=float,default=1024)
    a.add_argument("--log",type=Path,required=True)
    a.add_argument("command",nargs=argparse.REMAINDER)
    x=a.parse_args()
    cmd=x.command[1:] if x.command and x.command[0]=="--" else x.command
    if not cmd or x.seconds<=0 or x.seconds>180 or x.rss_mib>1024:
        raise SystemExit("Invalid budget/command")
    start=time.monotonic(); maxrss=0; status=None; reason="unknown"
    child=subprocess.Popen(cmd,start_new_session=True)
    try:
        while True:
            now=time.monotonic()
            rss=sum(read_rss_kib(pid) for pid in descendants(child.pid))
            maxrss=max(maxrss,rss)
            status=child.poll()
            if rss > x.rss_mib*1024:
                reason="RSS_BUDGET_EXCEEDED";break
            if now-start >= x.seconds:
                reason="TIME_BUDGET_EXCEEDED";break
            if status is not None:
                reason="OK" if status==0 else "CHILD_EXIT_NONZERO"
                break
            time.sleep(.1)
        if reason in ("RSS_BUDGET_EXCEEDED","TIME_BUDGET_EXCEEDED"):
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        else:
            child.wait()
            status=child.returncode
    finally:
        report={"schema":"researcher03-run-budget-v1","utc":datetime.now(timezone.utc).isoformat(),
            "command":cmd,"seconds_budget":x.seconds,"rss_mib_budget":x.rss_mib,
            "elapsed_seconds":time.monotonic()-start,"peak_tree_rss_mib":maxrss/1024,
            "child_exit_status":status,"status":reason,
            "test_accessed":False,"optimizer_steps":0,"inference_replays":0}
        x.log.parent.mkdir(parents=True,exist_ok=True)
        x.log.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
        print("R03_BUDGET_AUDIT "+json.dumps(report),flush=True)
    if reason!="OK":
        raise SystemExit(137 if reason=="RSS_BUDGET_EXCEEDED" else 124 if reason=="TIME_BUDGET_EXCEEDED" else 1)

if __name__=="__main__":
    main()
