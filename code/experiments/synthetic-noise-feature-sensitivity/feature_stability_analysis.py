"""E-CH4-14 frozen feature preference stability in horizons 1,2,3.

Reuses independently replayed exact §3.8 per-feature counts at step 60.
The 3 informative feature IDs are original indices 0,1,3, not local
matrix columns. For class, feature shares are descriptive class-policy means.
No fit, no Test, no masked-walk claim, no ranking-is-causal-importance claim.
"""
from __future__ import annotations
import argparse,csv,io
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TABLE=HERE/"tables"
GOOD=(0,1,3)
PAIRS=((0,1),(0,3),(1,3))


def csvbytes(rows):
    s=io.StringIO(newline="")
    w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows)
    return s.getvalue().encode("utf-8")


def build():
    with (TABLE/"noise_horizon_by_feature_60.csv").open(encoding="utf-8",newline="") as f:
        source=list(csv.DictReader(f))
    per=defaultdict(dict)
    for r in source:
        h=int(r["h"])
        if h>3:continue
        key=(r["run_id"],r["family"],int(r["noise_count"]),h)
        feat=int(r["feature"])
        assert feat not in per[key],f"Duplicate feature {key} {feat}"
        per[key][feat]=float(r["selection_share"])
    assert len(per)==36*3
    baseline={(fam,h):per[(f"N0-01__{fam}",fam,0,h)]
              for fam in ("shared","class") for h in (1,2,3)}
    rows=[]
    for (run,fam,k,h),probs in sorted(per.items()):
        assert len(probs)==k+3
        assert set(GOOD).issubset(probs)
        assert np.isclose(sum(probs.values()),1,atol=1e-11)
        orig=baseline[(fam,h)]
        total=sum(probs[a] for a in GOOD)
        assert total>0
        w={a:probs[a]/total for a in GOOD}
        old={a:orig[a] for a in GOOD}
        reversals=0
        near_ties=0
        details=[]
        for a,b in PAIRS:
            x=np.sign(old[a]-old[b])
            y=np.sign(w[a]-w[b])
            if abs(old[a]-old[b])<1e-10 or abs(w[a]-w[b])<1e-10:
                near_ties+=1
                state="tie"
            elif x*y<0:
                reversals+=1
                state="reversal"
            else:
                state="preserved"
            details.append(state)
        r={"run_id":run,"family":fam,"noise_count":k,"h":h,
           "main_share":total,"noise_share":1-total,
           "main_share_delta":total-1,
           "main_conditional_TV":.5*sum(abs(w[a]-old[a]) for a in GOOD),
           "rank_reversal_count":reversals,"rank_tie_pair_count":near_ties,
           "rank_preserved_pairs":3-reversals-near_ties}
        for a in GOOD:
            r[f"feature_{a}_share"]=probs[a]
            r[f"feature_{a}_share_delta"]=probs[a]-orig[a]
            r[f"feature_{a}_conditional_share"]=w[a]
            r[f"feature_{a}_conditional_delta"]=w[a]-old[a]
        for (a,b),state in zip(PAIRS,details):
            r[f"order_{a}_{b}"]=state
        rows.append(r)
    assert len(rows)==108
    groups=[]
    for family in ("shared","class"):
        for k in range(6):
            for h in (1,2,3):
                arr=[r for r in rows if r["family"]==family and
                     r["noise_count"]==k and r["h"]==h]
                assert len(arr)==(1 if k in (0,5) else 4)
                item={"family":family,"noise_count":k,"h":h,
                      "n_dependent_subsets":len(arr)}
                for name in ("main_share","main_share_delta","main_conditional_TV",
                             "rank_reversal_count",
                             *[f"feature_{a}_share" for a in GOOD],
                             *[f"feature_{a}_conditional_share" for a in GOOD]):
                    x=np.asarray([r[name] for r in arr])
                    item[name+"_mean"]=float(x.mean())
                    item[name+"_min"]=float(x.min())
                    item[name+"_max"]=float(x.max())
                groups.append(item)
    return rows,groups


def artifacts():
    rows,groups=build()
    return {TABLE/"feature_preference_by_run_60.csv":csvbytes(rows),
            TABLE/"feature_preference_by_noise_level_60.csv":csvbytes(groups)}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("write","check"),default="check")
    args=p.parse_args()
    for path,data in artifacts().items():
        if args.mode=="write":
            path.write_bytes(data)
        else:
            assert path.is_file() and path.read_bytes()==data,path
    print("E14_FEATURE_PREFERENCE_PASS")
