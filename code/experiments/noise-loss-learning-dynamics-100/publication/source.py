"""Immutable E-CH4-14 output reader and official Train/Validation aggregation.

Source: Run 44419173443 (32 artifacts, 152 trajectories), standard data
manifest from Run 44417037644. No fit, no Test and no PyTorch.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

RUN_ID = 44419173443
FAMILIES = ("shared", "class")
OBJECTIVES = ("sigmoid_JS_margin_tau005", "softplus_JS_tau005")
COMPARABLE = range(1, 7)
PRIMARY = (0, 1, 3)
NOISE = (2, 4, 5, 6, 7, 8, 9)
TRAIN = (
    "loss", "gradient_l2", "accuracy", "balanced_accuracy",
    "precision_macro", "recall_macro", "density_r_kl",
    "density_r_js", "density_r_sqrt_js", "separation_i",
    "separation_t", "separation_bt", "separation_tnl",
    "separation_btnl", "margin_mean"
)
VALIDATION = ("accuracy", "balanced_accuracy",
              "precision_macro", "recall_macro")


def mapping(dataset_dir):
    parent = Path(dataset_dir)
    manifest_path = parent / "derived" / "manifest.json"
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert man["schema"] == "E-CH4-14-100-derived-standard-seed-v1"
    assert man["standard_data_seed"] == 20261102 and man["test_accessed"] is False
    assert len(man["datasets"]) == 38
    result = {}
    for ds in man["datasets"]:
        assert ds["dataset_id"] not in result
        name = parent / "derived" / (ds["dataset_id"] + ".npz")
        assert hashlib.sha256(name.read_bytes()).hexdigest() == ds["sha256"]
        assert set(ds["feature_indices"]) == set(PRIMARY).union(ds["noise_indices"])
        assert len(ds["feature_indices"]) == 3 + ds["noise_count"]
        result[ds["dataset_id"]] = ds
    return result


def load(artifacts_dir, dataset_dir):
    """SHA+101-state audit for all 152 trajectories; return dict of full rows."""
    map_ = mapping(dataset_dir)
    root = Path(artifacts_dir)
    found = {}
    paths = sorted(root.rglob("trace.jsonl"))
    assert len(paths) == 152, ("not 152 traces", len(paths))
    for trace in paths:
        run_id = trace.parent.name
        dataset_id, objective, family = run_id.split("__")
        assert dataset_id in map_ and objective in OBJECTIVES and family in FAMILIES
        assert run_id not in found, ("duplicate trajectory", run_id)
        proof = json.loads((trace.parent / "complete.json").read_text("utf-8"))
        raw = trace.read_bytes()
        assert proof["run_id"] == run_id and proof["updates"] == 100
        assert proof["test_accessed"] is False
        assert hashlib.sha256(raw).hexdigest() == proof["trace_sha256"]
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == 101 and [r["step"] for r in rows] == list(range(101))
        assert all(r["run_id"] == run_id for r in rows)
        assert all(set(r["validation"]) == set(VALIDATION).union({
            "precision_by_class", "recall_by_class"}) for r in rows)
        expected = 3 if family == "class" else 1
        for row in rows:
            assert len(row["policy"]["unmasked"]) == expected
            assert all("test" not in key.lower() for key in row)
        found[run_id] = dict(
            dataset_id=dataset_id, objective=objective, family=family,
            noise_count=map_[dataset_id]["noise_count"],
            feature_indices=map_[dataset_id]["feature_indices"],
            rows=rows,
        )
    assert len(found) == 152
    for count in COMPARABLE:
        for fam in FAMILIES:
            for obj in OBJECTIVES:
                group = group_runs(found, fam, obj, count)
                assert len(group) == 6, (fam, obj, count, len(group))
    return found


def group_runs(data, family, objective, noise_count):
    return [r for r in data.values() if r["family"] == family
            and r["objective"] == objective and r["noise_count"] == noise_count]


def get_series(run, split, field):
    return np.array([
        np.nan if row[split][field] is None else float(row[split][field])
        for row in run["rows"]
    ], dtype=float)


def write_table(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(data, out):
    fields = ("family", "objective", "noise_count", "step", "split", "metric",
              "mean", "descriptive_sd", "minimum", "maximum", "subset_count")
    result = []
    for family in FAMILIES:
        for objective in OBJECTIVES:
            for count in COMPARABLE:
                group = group_runs(data, family, objective, count)
                for split, metrics in (("train", TRAIN), ("validation", VALIDATION)):
                    for metric in metrics:
                        arr = np.stack([get_series(r, split, metric) for r in group])
                        if np.isnan(arr).all():
                            continue
                        for t in range(101):
                            samples = arr[:, t]
                            finite = samples[np.isfinite(samples)]
                            if len(finite) != 6:
                                continue
                            result.append(dict(family=family, objective=objective,
                               noise_count=count, step=t, split=split, metric=metric,
                               mean=float(finite.mean()),
                               descriptive_sd=float(finite.std(ddof=1)),
                               minimum=float(finite.min()), maximum=float(finite.max()),
                               subset_count=6))
    write_table(out, result, fields)
    return len(result)


def policy_metrics(p0, P, feature_indices, h=3):
    """Exact feature-only *unmasked* policy process (§3.8)."""
    d = len(feature_indices)
    pp = [i for i, old in enumerate(feature_indices) if old in PRIMARY]
    nn = [i for i, old in enumerate(feature_indices) if old in NOISE]
    assert len(pp) == 3 and len(nn) == d-3
    p0 = np.asarray(p0, dtype=float)
    P = np.asarray(P, dtype=float)
    assert p0.shape == (d,) and P.shape == (d, d)
    assert np.isclose(p0.sum(), 1) and np.allclose(P.sum(axis=1), 1)
    states = [p0]
    for _ in range(3):
        states.append(states[-1] @ P)
    q = [float(x[nn].sum()) for x in states[:3]]
    p_reduced = P[np.ix_(pp, pp)]
    hit2 = 1. - float(p0[pp] @ p_reduced @ np.ones(3))
    hit3 = 1. - float(p0[pp] @ p_reduced @ p_reduced @ np.ones(3))
    s2 = (q[0] + q[1])/2
    s3 = sum(q)/3
    uniform = len(nn)/d
    result = dict(q1=q[0], q2=q[1], q3=q[2], hit2=hit2, hit3=hit3,
                  nbar3=sum(q), s2=s2, s3=s3, sp3=1-s3,
                  NFI3=hit3/(1-(3/d)**3), NSR2=s2/uniform, NSR3=s3/uniform)
    return result, (states[0]+states[1]+states[2])/3


def policy_at(run, row):
    """Macro average across class-specific hypotheses, never mix their matrices."""
    vals = []
    features = run["feature_indices"]
    for item in row["policy"]["unmasked"]:
        metrics, shares = policy_metrics(item["p0"], item["P"], features)
        assert abs(metrics["hit3"]-item["hit_at_least_once"]) < 1e-8
        assert abs(metrics["nbar3"]-item["expected_noise_visits"]) < 1e-8
        assert abs(metrics["NFI3"]-item["relative_hit_to_uniform"]) < 1e-8
        vals.append((metrics, shares))
    keys = vals[0][0]
    avg = {key: float(np.mean([x[0][key] for x in vals])) for key in keys}
    shares = np.stack([x[1] for x in vals]).mean(axis=0)
    assert abs(shares.sum()-1) < 1e-10
    return avg, shares


def noise_table(data, out):
    fields = ("family", "objective", "noise_count", "step", "metric",
              "mean", "descriptive_sd", "min", "max", "subset_count")
    records = []
    metrics = ("q1","q2","q3","hit2","hit3","nbar3","s2","s3",
               "sp3","NFI3","NSR2","NSR3")
    for family in FAMILIES:
        for objective in OBJECTIVES:
            for count in COMPARABLE:
                rr = group_runs(data,family,objective,count)
                trajectories = [[policy_at(run,row)[0] for row in run["rows"]]
                                for run in rr]
                for key in metrics:
                    arr = np.array([[state[key] for state in traj]
                                    for traj in trajectories])
                    for t in range(101):
                        v = arr[:,t]
                        records.append(dict(family=family,objective=objective,
                            noise_count=count,step=t,metric=key,
                            mean=float(v.mean()),descriptive_sd=float(v.std(ddof=1)),
                            min=float(v.min()),max=float(v.max()),subset_count=6))
    write_table(out,records,fields)
    return len(records)
