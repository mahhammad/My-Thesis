#!/usr/bin/env python3
"""RESEARCHER-03: all-archive checksum identity preflight (no center calculation)."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from center_only_rebuild import BASE, EXPECTED_ARCHIVES, archive_records, read_index

HERE=Path(__file__).resolve().parent

def run_preflight(out_dir:Path):
    index=read_index()
    checked=[]
    seen=set()
    for archive,(expected_sha,expected_count) in EXPECTED_ARCHIVES.items():
        records=archive_records(archive,index)  # checks SHA256 ZIP/member/master/manifest AND metadata
        members=[]
        for row,raw,member,member_sha,sha in records:
            if row["file"] in seen:
                raise ValueError("DUPLICATE_RUN_ID "+row["file"])
            seen.add(row["file"])
            centers=raw.get("final",{}).get("train",{}).get("centers")
            if centers is None:
                raise ValueError("NO_SAVED_TRAIN_CENTER "+member)
            if len(centers)!=3 or any(len(v)!=3 for v in centers):
                raise ValueError("BAD_CENTER_SHAPE "+member)
            members.append({"run_id":row["file"],"member":member,"sha256":member_sha,
                "family":row["family"],"objective":row["objective"],
                "data_seed":int(row["data_seed"]),"steps":int(row["steps"]),"n_ref":int(row["n"]),
                "validation_defined":bool(row["val_BA"])})
        if len(records)!=expected_count:
            raise ValueError("ARCHIVE_COUNT_MISMATCH "+archive)
        checked.append({"source_archive":archive,"sha256":expected_sha,"sha256_verified":True,
            "n":len(records),"members":members})
    if seen!=set(index):
        raise ValueError(f"RUN_COVERAGE_MISMATCH missing={set(index)-seen} extra={seen-set(index)}")
    report={"schema":"researcher03-archive-recovery-preflight-v1",
        "source_ref":"chapter4@de02fdda91e23db177de4429b3e1a9693c046d83",
        "sources_checkout":str(BASE),"archive_count":len(checked),
        "run_count":len(seen),"family_counts":{f:sum(row["family"]==f for row in index.values()) for f in
        ("shared","class","node","node_class")},
        "validation_numeric":sum(bool(row["val_BA"]) for row in index.values()),
        "train_center_only":True,"test_accessed":False,"optimizer_steps":0,
        "inference_replay_count":0,"archives":checked,
        "checked_utc":datetime.now(timezone.utc).isoformat()}
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"archive_preflight.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return report

if __name__=="__main__":
    print(json.dumps(run_preflight(HERE/"full_76_candidate"),indent=2,ensure_ascii=False))
