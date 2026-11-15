#!/usr/bin/env python3
"""READ-ONLY B02 snapshot audit; no generator, graph forward, or solver calls."""
import copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build_snapshot import MAGIC, verify_snapshot, sha, jsonbytes, put_atomic, rss
START=time.monotonic()
def must_reject(label,payload,manifest,targets,expected):
    try: verify_snapshot(payload,manifest,targets)
    except (ValueError,AssertionError) as exc:
        if expected not in str(exc):raise AssertionError(label+" unexpected rejection: "+str(exc))
        return {"PASS":True,"reason":str(exc)}
    raise AssertionError("FAIL_OPEN "+label)
def main():
    raw=(HERE/"train_canonical.bin").read_bytes()
    targets=(HERE/"targets_float64_le.bin").read_bytes()
    man=json.loads((HERE/"manifest.json").read_text())
    x,y,mask,ids=verify_snapshot(raw,man,targets)
    assert len(raw)==len(MAGIC)+300*5*8+300*8+300*5+300*8
    assert np.bincount(y,minlength=3).tolist()==[72,147,81]
    assert man["contexts"]==18 and man["authorized_to_solve"] is False
    assert man["teacher_given_to_solver"] is False
    assert man["targets"]["per_class"][0]["self_jsd"]==0
    out={}
    # A numeric payload byte changed: outer checksum is deliberately re-signed,
    # forcing the independent X-array checksum to reject it.
    changed=bytearray(raw)
    changed[len(MAGIC)+4]^=1
    cman=copy.deepcopy(man);cman["snapshot"]["file_sha256"]=sha(changed)
    cman["snapshot"]["payload_sha256"]=sha(changed)
    out["numeric_byte"]=must_reject("numeric_byte",bytes(changed),cman,targets,"X SHA mismatch")
    # Flip a missing-mask bit and update all outer and mask checksums.
    # Semantic mask-vs-isfinite invariant must fail despite matching hashes.
    changed=bytearray(raw)
    idx=len(MAGIC)+300*5*8+300*8+10
    changed[idx]^=1
    cman=copy.deepcopy(man)
    cman["snapshot"]["file_sha256"]=sha(changed)
    cman["snapshot"]["payload_sha256"]=sha(changed)
    offset=len(MAGIC)+300*5*8+300*8
    cman["snapshot"]["array_sha256"]["mask"]=sha(bytes(changed[offset:offset+300*5]))
    out["mask_bit"]=must_reject("mask_bit",bytes(changed),cman,targets,"mask inconsistency")
    # Swap two complete reference IDs, then re-sign both file and ID checksum.
    # Semantic exact-order check must reject it even with matching hashes.
    changed=bytearray(raw)
    offset=len(raw)-300*8
    first=changed[offset:offset+8]
    second=changed[offset+8:offset+16]
    changed[offset:offset+8]=second
    changed[offset+8:offset+16]=first
    cman=copy.deepcopy(man)
    cman["snapshot"]["file_sha256"]=sha(changed)
    cman["snapshot"]["payload_sha256"]=sha(changed)
    cman["snapshot"]["array_sha256"]["ids"]=sha(bytes(changed[offset:]))
    out["id_swap"]=must_reject("id_swap",bytes(changed),cman,targets,"ID corruption")
    # Target is tampered, with outer target checksum re-signed.
    changed=bytearray(targets)
    changed[7]^=1
    cman=copy.deepcopy(man)
    cman["targets"]["file_sha256"]=sha(changed)
    out["target_byte"]=must_reject("target_byte",raw,cman,bytes(changed),"target SHA mismatch")
    result={"status":"PASS","mode":"read_only_no_forward","tamper_semantic":out,
            "wall_seconds":time.monotonic()-START,"rss_mib":rss(),
            "numeric_steps":0,"forwards":0,"optimizers":0,
            "snapshot_file_sha256":sha(raw),"target_file_sha256":sha(targets)}
    if result["wall_seconds"]>=120 or result["rss_mib"]>=1024:raise RuntimeError("resource cap")
    report=HERE/"tamper_deep_verification.json"
    if report.exists():raise FileExistsError("never overwrite deep verification")
    put_atomic(report,jsonbytes(result))
    print("RESEARCHER04_B02_DEEP_TAMPER_PASS",json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
