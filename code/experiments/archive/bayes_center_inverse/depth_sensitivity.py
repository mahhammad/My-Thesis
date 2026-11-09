#!/usr/bin/env python3
"""H / beta / lambda sensitivity of target reachability and effective depth.

Exact finite-horizon step weights follow the *normalized-per-walk* rule used
by the project. Target centers are frozen: classwise average of Bayes
posteriors of all 300 labeled train samples. Reachability tests are
coordinatewise OUTER bounds, not a policy-recovery optimization.

The tight bound on a selected reference subset excludes the origin at every
subsequent step; it permits a time-, origin-, and state-dependent policy.
The full-reference bound further allows revisiting the origin; both are
relaxations of shared-Z policies. Failure of either proves impossibility,
passing either is inconclusive. No Monte Carlo or classifier score is used.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from run_bayes import (ROOT, ExactGraph, bayes_class_centers, generate,
                       jsdiv, load_config, coefficients)
from feasibility_bound import center_bounds, origin_return_relaxed_bounds

HERE = Path(__file__).resolve().parent


def _binary_jsd_lower_bound(target, lower, upper):
    """Coarsening K categories to one-vs-rest gives valid JSD lower bound."""
    target, lower, upper = map(np.asarray, (target, lower, upper))
    nearest = np.clip(target, lower, upper)
    return np.stack([
        [max(float(jsdiv(np.array([target[c, j], 1-target[c, j]]),
                         np.array([nearest[c, j], 1-nearest[c, j]])))
             for j in range(target.shape[1]))]
        for c in range(target.shape[0])
    ])


def summarize_bounds(bounds, target):
    lo = np.array(bounds['lower_center_probabilities'] if 'lower_center_probabilities' in bounds else bounds['lower'])
    hi = np.array(bounds['upper_center_probabilities'] if 'upper_center_probabilities' in bounds else bounds['upper'])
    target = np.asarray(target)
    missed = (target < lo - 1e-12) | (target > hi + 1e-12)
    binary = _binary_jsd_lower_bound(target, lo, hi)
    return {
        'diagonal_upper': np.diag(hi).tolist(),
        'diagonal_target_minus_upper': (np.diag(target) - np.diag(hi)).tolist(),
        'proved_unattainable_class_rows': np.flatnonzero(missed.any(axis=1)).tolist(),
        'JSD_lower_bound_per_class': binary[:, 0].tolist(),
        'JSD_lower_bound_class_average': float(binary.mean()),
        'lower_center_coordinatewise': lo.tolist(),
        'upper_center_coordinatewise': hi.tolist(),
    }


def calculate(seed=20261107, subset=72, betas=(.2, .4, .7),
              lambdas=(.4, .8, 1.), horizons=(1, 2, 3, 4, 12),
              tight_subset=True):
    spec = load_config(ROOT / 'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as tmp:
        splits = generate(spec, tmp, finite_q=False)
    x, y = splits['train'][0], splits['train'][2]
    target, posterior = bayes_class_centers(spec, x, y)
    k = len(spec['pi'])
    if subset % k or subset > len(x):
        raise ValueError('subset must be divisible by K and within training data')
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(y == c), subset//k,
                                         replace=False) for c in range(k)])
    rng.shuffle(chosen)
    xs, ys = x[chosen], y[chosen]
    counts = np.isfinite(x).sum(axis=0)
    subset_counts = np.isfinite(xs).sum(axis=0)
    records = []
    for H in horizons:
        for beta in ((.4,) if H == 1 else betas):
            for lam in ((.8,) if H == 1 else lambdas):
                t0 = time.perf_counter()
                alpha = coefficients(beta, lam, H)
                assert np.isclose(alpha.sum(), 1.)
                relaxed = origin_return_relaxed_bounds(x, y, spec['gamma'], alpha)
                time_full = time.perf_counter()-t0
                tight = None
                time_subset = None
                if tight_subset:
                    t0 = time.perf_counter()
                    g = ExactGraph(xs, ys, spec['gamma'], np.arange(subset),
                                   splits['validation'][0][:1],
                                   beta=beta, lam=lam, horizon=H)
                    tight = summarize_bounds(center_bounds(g, ys), target)
                    time_subset = time.perf_counter() - t0
                records.append({
                    'H': int(H), 'beta': float(beta), 'lambda': float(lam),
                    'alpha': alpha.tolist(),
                    'mean_sampled_length': float((1-(1-beta)**H)/beta),
                    'mean_weighted_hop': float(alpha @ np.arange(1,H+1)),
                    'alpha_cumulative_1_to_2': float(alpha[:2].sum()),
                    'alpha_cumulative_1_to_3': float(alpha[:3].sum()),
                    'alpha_cumulative_1_to_4': float(alpha[:4].sum()),
                    'full300_return_relaxed': summarize_bounds(relaxed, target),
                    'subset_origin_excluded_time_dependent': tight,
                    'full_bound_wall_s': time_full,
                    'subset_bound_wall_s': time_subset,
                })
    best = {
        'max_full_upper_diagonal_over_grid': np.stack(
            [np.array(r['full300_return_relaxed']['diagonal_upper']) for r in records]).max(axis=0).tolist(),
        'all_grid_points_prove_Bayes_unattainable_on_full300': all(
            len(r['full300_return_relaxed']['proved_unattainable_class_rows']) == k for r in records),
        'any_grid_point_without_proved_unattainable_class': any(
            len(r['full300_return_relaxed']['proved_unattainable_class_rows']) == 0 for r in records),
        'best_full_lower_bound_on_mean_JSD': float(min(
            r['full300_return_relaxed']['JSD_lower_bound_class_average'] for r in records)),
    }
    return {'meta': {'goal': 'sensitivity of Bayes target reachability to beta, lambda and depth H',
                     'data_seed': int(spec['seed']), 'reference_seed': seed,
                     'reference_size_full': len(x), 'reference_size_subset': subset,
                     'K': k, 'd': len(spec['gamma']),
                     'sample_feature_nodes_full': int(np.isfinite(x).sum()),
                     'sample_feature_nodes_subset': int(np.isfinite(xs).sum()),
                     'sample_transition_edges_full': int((counts*(counts-1)).sum()),
                     'sample_transition_edges_subset': int((subset_counts*(subset_counts-1)).sum()),
                     'feature_switch_edges_full': int((np.isfinite(x).sum(axis=1)**2).sum()),
                     'feature_switch_edges_subset': int((np.isfinite(xs).sum(axis=1)**2).sum()),
                     'grid': {'H': list(horizons), 'beta': list(betas), 'lambda': list(lambdas)},
                     'H1_independent_of_beta_lambda': True,
                     'bounds_are_outer_relaxations_not_exact_optima': True,
                     'target_averages_all_300_training_posteriors': True},
            'bayes_target_centers': target.tolist(),
            'selected_reference_indices': chosen.tolist(),
            'summary': best, 'records': records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--subset', type=int, default=72)
    p.add_argument('--out', type=Path,
                   default=HERE/'results'/'depth_sensitivity_full300_and_subset72.json')
    a = p.parse_args()
    result = calculate(seed=a.seed, subset=a.subset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print('Bayes diag', np.diag(result['bayes_target_centers']))
    print('Max attainable upper bound from relaxed full grid',
          result['summary']['max_full_upper_diagonal_over_grid'])
    print('All settings prove unattainable for all three classes',
          result['summary']['all_grid_points_prove_Bayes_unattainable_on_full300'])
    print('Rows',len(result['records']),'saved',a.out)

if __name__ == '__main__':
    main()
