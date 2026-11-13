#!/usr/bin/env python3
"""Light consistency checks for published SEC411 derived tables; no heavy runs."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))
def check():
    c = load("results/certified_summary.json")
    assert len(c["seed_graphs"]) == 6
    assert len(c["D11"]) == 2
    assert c["nonbaseline"]["count"] == 48
    t1 = (ROOT/"tables/certified_bayes.tex").read_text(encoding="utf-8")
    for rec in c["seed_graphs"]:
        assert str(rec["seed"]) in t1
        assert rec["certified_balanced_mean_jsd_lower_nats_floor"] in t1
    t2 = (ROOT/"tables/d11_fixed_targets.tex").read_text(encoding="utf-8")
    for rec in c["D11"]:
        assert rec["target"] in t2
        assert rec["certified_mean_jsd_lower_nats_floor"] in t2
    m = load("results/metric_seed_summary.json")["rows"]
    assert len(m) == 4
    t3 = (ROOT/"tables/balanced_accuracy.tex").read_text(encoding="utf-8")
    for rec in m:
        for key in ("train", "val", "test"):
            assert f'{rec[key]*100:.2f}' in t3
    rows = (ROOT/"results/bounds_54_rounded.csv").read_text().strip().splitlines()
    assert len(rows) == 55
    print("PASS SEC411 tables: 6 certificates, 2 D11, 4 BA means, 54 sensitivity rows")
if __name__ == "__main__":
    check()
