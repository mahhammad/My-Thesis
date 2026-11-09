#!/usr/bin/env python3
"""Finite-reference vs analytic population Q, independent of policy learning.

Run from repository root:
  python code/experiments/q_population_validation/compare.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/data_generator"))
from generate import load_config, finite_one_step, population_one_step

CONFIG = ROOT / "code/data_generator/config.json"
OUT = Path(__file__).resolve().parent / "results"
SIZES = (300, 1000, 3000, 10000)
REPLICATES = 12


def sample(spec, n, rng):
    """Same conditional Gaussian, class sampling, and MCAR as data_generator.generate."""
    k, d = spec["mu"].shape
    y = rng.choice(k, size=n, p=spec["pi"])
    x = np.empty((n, d))
    for c in range(k):
        ix = np.flatnonzero(y == c)
        if len(ix):
            x[ix] = rng.multivariate_normal(spec["mu"][c], spec["cov"][c], size=len(ix))
    observed = rng.random(x.shape) >= spec["missing"][None, :]
    x[~observed] = np.nan
    return x, y


def metrics(q, population, queries):
    mask = np.isfinite(queries)
    if not np.array_equal(np.isfinite(q).all(axis=-1), mask):
        raise ValueError("finite Q does not respect query missingness")
    if not np.array_equal(np.isfinite(population).all(axis=-1), mask):
        raise ValueError("population Q does not respect query missingness")
    a, b = q[mask], population[mask]
    if not (np.allclose(a.sum(axis=1), 1) and np.allclose(b.sum(axis=1), 1)):
        raise ValueError("invalid Q probability vectors")
    tv = 0.5 * np.abs(a - b).sum(axis=1)
    mid = (a + b) / 2
    jsd = np.zeros(len(tv))
    for p in (a, b):
        jsd += 0.5 * np.sum(np.where(p > 0,
                           p * np.log(np.maximum(p, 1e-300) / np.maximum(mid, 1e-300)), 0), axis=1)
    return dict(n_observed_feature_queries=len(tv), mean_TV=float(tv.mean()),
                median_TV=float(np.median(tv)), p95_TV=float(np.quantile(tv, .95)),
                mean_JSD=float(jsd.mean()), mean_sqrt_JSD=float(np.sqrt(jsd).mean()),
                mean_L1=float((2 * tv).mean()),
                mean_TV_per_feature=[
                    float((0.5 * np.abs(q[:, a, :] - population[:, a, :])
                           .sum(axis=-1))[mask[:, a]].mean())
                    for a in range(queries.shape[1])])


def analyze():
    spec = load_config(CONFIG)
    children = np.random.SeedSequence(spec["seed"]).spawn(3)
    queries, _ = sample(spec, 90, np.random.default_rng(children[1]))
    truth = population_one_step(spec, queries)
    raw = []
    for trial in range(REPLICATES):
        ref, labels = sample(spec, max(SIZES), np.random.default_rng(11900 + trial))
        for n in SIZES:
            empirical = finite_one_step(spec, queries, ref[:n], labels[:n])
            raw.append(dict(trial=trial, n_reference=n,
                            **metrics(empirical, truth, queries)))
    grouped = []
    for n in SIZES:
        rows = [r for r in raw if r["n_reference"] == n]
        grouped.append(dict(
            n_reference=n, replicates=REPLICATES,
            avg_mean_TV=float(np.mean([r["mean_TV"] for r in rows])),
            sd_mean_TV=float(np.std([r["mean_TV"] for r in rows], ddof=1)),
            avg_mean_sqrt_JSD=float(np.mean([r["mean_sqrt_JSD"] for r in rows])),
            avg_p95_TV=float(np.mean([r["p95_TV"] for r in rows])),
            avg_mean_TV_per_feature=np.mean(
                [r["mean_TV_per_feature"] for r in rows], axis=0).tolist()))
    original, y = sample(spec, 300, np.random.default_rng(children[0]))
    baseline = metrics(finite_one_step(spec, queries, original, y), truth, queries)
    return dict(date="2026-11-07", source_commit="1a951d578125baf71a53cec3c1c467b67b6218f2",
                reference_sizes=list(SIZES), replicates_per_size=REPLICATES,
                n_fixed_validation_queries=90, fixed_query_seed=spec["seed"],
                exact_default_300_reference=baseline, comparison=grouped,
                replicate_data=raw)


def save(results):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "q_population_validation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "q_population_replicates.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["trial", "n_reference", "mean_TV", "mean_sqrt_JSD", "p95_TV"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results["replicate_data"]:
            writer.writerow({key: r[key] for key in columns})


if __name__ == "__main__":
    result = analyze()
    save(result)
    for r in result["comparison"]:
        print(f"{r['n_reference']:5d} | TV={r['avg_mean_TV']:.5f} "
              f"(SD={r['sd_mean_TV']:.5f}) | sqrt-JSD={r['avg_mean_sqrt_JSD']:.5f}")
