#!/usr/bin/env python3
"""RESEARCHER-03: fail-closed one-shot orchestration, no walk, Test, or training.

The outer budget_guard.py covers process tree time/RSS INCLUDING subprocess.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys, traceback

HERE=Path(__file__).resolve().parent
OUT=HERE/"full_76_candidate"

def run():
    if OUT.exists():
        raise RuntimeError("FULL_76_CANDIDATE_DIR_EXISTS: refuse accidental overwrite")
    import preflight_recovery
    preflight=preflight_recovery.run_preflight(OUT)
    if preflight["run_count"]!=76:
        raise RuntimeError("PREFLIGHT DID NOT VERIFY 76")
    # Invoke exactly the historical center-only rebuild implementation,
    # not a replacement formula engine or inference replay.
    subprocess.run([sys.executable,str(HERE/"center_only_rebuild.py"),
                    "--execute","--out-dir",str(OUT)],check=True)
    import verify_recovery
    verify_recovery.main()
    print("RESEARCHER-03: PRECHECK+REBUILD+36-REGRESSION+76-COVERAGE PASS",flush=True)

if __name__=="__main__":
    try:run()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"FAILED_GATE.json").write_text(
            json.dumps({"result":"STOP","reason":type(e).__name__+":"+str(e),
                "test_accessed":False,"optimizer_steps":0,"inference_replays":0},
                ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        traceback.print_exc()
        raise SystemExit(1)
