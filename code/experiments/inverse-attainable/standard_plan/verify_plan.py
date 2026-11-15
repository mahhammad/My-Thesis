#!/usr/bin/env python3
"""Read-only sanity checks for proposed INV01 108-case run matrix: no solver runs."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json
root=Path(__file__).resolve().parent
j=json.loads((root/"plan_108.json").read_text(encoding="utf-8"))
r=j["runs"]
assert j["status"]=="PROPOSED_AWAITING_MAIN_RUN_GATE"
assert j["safeguards"]["do_not_run_until_gate"]
assert len(r)==108 and len(set(x["run_id"] for x in r))==108
assert len(j["blocks"])==6 and set(x["state"] for x in r)=={"NOT_RUN"}
assert len(set(x["shared_problem_id"] for x in r))==18
assert all(x["nref"] in (150,300) for x in r)
assert all(x["data_seed"] in (20261102,20261103,20261104) for x in r)
assert all(x["solver"] in ("Q-only","Exact-Adam","Q+Bootstrap-GN") for x in r)
assert len(set((x["shared_problem_id"],x["initialization"]) for x in r))==36
for k,v in Counter(x["block_id"] for x in r).items():assert v==18,(k,v)
for k,rows in defaultdict(list,((k,[x for x in r if x["shared_problem_id"]==k]) for k in set(x["shared_problem_id"] for x in r))).items():
 assert len(rows)==6
 assert len(set(x["teacher_seed"] for x in rows))==1
 assert {x["solver"] for x in rows}=={"Q-only","Exact-Adam","Q+Bootstrap-GN"}
 assert {x["initialization"] for x in rows}=={"zero","independent_random"}
with (root/"plan_108.csv").open(encoding="utf8",newline="") as f:
 assert len(list(csv.DictReader(f)))==108
print("INV01_STANDARD_108_PLAN_PASS — NO RUNS EXECUTED")
