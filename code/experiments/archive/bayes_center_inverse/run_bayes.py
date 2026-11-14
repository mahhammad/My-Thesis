#!/usr/bin/env python3
"""Recover a class-hypothesis policy from classwise mean Bayes label posteriors.

Laboratory-only diagnostic, NOT a model training objective.  Bayes targets use
*all 300 training samples*, never validation/test. All outputs of the walk
are deterministic matrix propagation (geometric stopped walks, lambda-weighted,
original source excluded after the first step). Policies are class-hypothesis
conditioned Z_c; training origin labels are ONLY used for inverse fitting.

Run from any directory: python code/experiments/bayes_center_inverse/run_bayes.py --seed 20261107 --nref 72
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/data_generator'))
sys.path.insert(0, str(ROOT / 'code/experiments/multistep_policy_target_match'))
from generate import generate, load_config  # noqa: E402
from run import ExactGraph, DT, jsd, coefficients  # noqa: E402
from bootstrap_inverse import warm_q, exact_adam, damped_gauss_newton, add_counts  # noqa: E402
from feasibility_bound import center_bounds  # noqa: E402


def bayes_posterior(spec: dict, x: np.ndarray) -> np.ndarray:
    """P(class|observed x) with correlated Gaussian classes and MCAR masks.

    Duplicates the original generator's Bayes posterior on the training data;
    missing coordinates are integrated out (not imputed). The missingness
    mechanism is independent of class and thus cancels from Bayes' rule.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != spec['mu'].shape[1]:
        raise ValueError('wrong feature shape')
    n, k = len(x), len(spec['pi'])
    logp = np.broadcast_to(np.log(spec['pi']), (n, k)).copy()
    observed = np.isfinite(x)
    for pattern in np.unique(observed, axis=0):
        rows = np.flatnonzero((observed == pattern).all(axis=1))
        idx = np.flatnonzero(pattern)
        if not len(idx):
            continue
        for c in range(k):
            cov = spec['cov'][c][np.ix_(idx, idx)]
            chol = np.linalg.cholesky(cov)
            residual = (x[np.ix_(rows, idx)] - spec['mu'][c, idx]).T
            standardized = np.linalg.solve(chol, residual)
            logp[rows, c] -= (0.5 * np.sum(standardized ** 2, axis=0)
                              + np.log(np.diag(chol)).sum()
                              + 0.5 * len(idx) * np.log(2.0 * np.pi))
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def bayes_class_centers(spec, train_x, train_y):
    posterior = bayes_posterior(spec, train_x)
    k = len(spec['pi'])
    targets = np.stack([posterior[train_y == c].mean(axis=0) for c in range(k)])
    return targets, posterior


def entropy(p):
    p = np.asarray(p, float)
    return -np.where(p > 0, p * np.log(np.clip(p, 1e-300, None)), 0.0).sum(axis=-1)


def jsdiv(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    return np.maximum(0., entropy(m) - 0.5 * (entropy(p) + entropy(q)))


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.sum(np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))), 0.0), axis=-1)


def center_geometry(centers):
    centers = np.asarray(centers, float)
    k = len(centers)
    off = np.array([jsdiv(centers[i], centers[j]) for i in range(k) for j in range(i + 1, k)])
    return {'I': float(entropy(centers.mean(axis=0)) - entropy(centers).mean()),
            'mean_pairwise_JSD': float(off.mean()) if len(off) else 0.,
            'min_pairwise_sqrt_JSD': float(np.sqrt(off.min())) if len(off) else 0.,
            'center_entropy_by_class': entropy(centers).tolist()}


def summarize_training(graph, y, policies, target, bayes_ref, *, origin_subset=None):
    """Training diagnostics under true-label-conditioned policies (oracle, NOT prediction).

    These diagnostics are intentionally separated from heldout candidate-conditional
    measurements; no oracle-conditioned training accuracy is reported.
    """
    k = graph.k
    centers, representations = [], []
    for c in range(k):
        ids = np.flatnonzero(y == c)
        with torch.no_grad():
            r = graph.center_train(policies[c], ids, return_all=True).cpu().numpy()
        representations.append(r)
        centers.append(r.mean(axis=0))
    centers = np.stack(centers)
    Rkl = float(np.mean([kl(representations[c], centers[c]).mean() for c in range(k)]))
    Rjs = float(np.mean([jsdiv(representations[c], centers[c]).mean() for c in range(k)]))
    Rbayes = float(np.mean([jsdiv(representations[c], bayes_ref[y == c]).mean() for c in range(k)]))
    residual_jsd = jsdiv(centers, target)
    return {'centers': centers.tolist(),
            'per_class_target_JSD': residual_jsd.tolist(),
            'mean_target_JSD': float(residual_jsd.mean()),
            'max_target_JSD': float(residual_jsd.max()),
            'train_within_KL_R': Rkl, 'train_within_JSD': Rjs,
            'train_JSD_to_individual_Bayes_posteriors': Rbayes,
            'center_geometry': center_geometry(centers)}


def heldout_candidate_metrics(graph, policies, realized_centers, val_y, val_posterior):
    """All candidate-hypothesis walks run without validation labels.

    Distance to frozen realized *training* center, not to Bayes targets.
    Candidate policy is chosen by hypothesized class, not true query label.
    """
    k = graph.k
    with torch.no_grad():
        candidates = [graph.evaluate_val(policies[c]).cpu().numpy() for c in range(k)]
    cent = np.asarray(realized_centers, float)
    distances = np.stack([jsdiv(candidates[c], cent[c]) for c in range(k)], axis=1)
    own = distances[np.arange(len(val_y)), val_y]
    rivals = np.min(np.where(np.eye(k, dtype=bool)[val_y], np.inf, distances), axis=1)
    margin = rivals - own
    selected = np.stack([candidates[val_y[i]][i] for i in range(len(val_y))])
    return {'validation_mean_JS_distance_to_true_candidate_center': float(own.mean()),
            'validation_mean_JS_margin': float(margin.mean()),
            'validation_margin_nonpositive_fraction': float(np.mean(margin <= 0)),
            'validation_mean_JSD_candidate_true_to_Bayes_posterior': float(jsdiv(selected, val_posterior).mean()),
            'validation_candidate_hypothesis_balanced_accuracy_diagnostic': float(np.mean([(distances[val_y == c].argmin(axis=1) == c).mean() for c in range(k)]))}



def shared_exact_adam(graph, labels, targets, *, steps, lr, tol):
    """Single Z for all class centers, unlike Z_c conditioned on hypothesis.

    This is a stricter, class-independent policy family; failure to attain
    Bayes centers here cannot be conflated with conditional inverse fitting.
    """
    z = torch.nn.Parameter(torch.zeros((graph.d + 1, graph.d), dtype=DT))
    opt = torch.optim.Adam([z], lr=lr)
    class_origins = [np.flatnonzero(labels == c) for c in range(graph.k)]
    targets_t = torch.tensor(targets, dtype=DT)
    count = {'exact_grad': 0, 'exact_forward': 0}
    trace = []
    t0 = time.perf_counter()
    def all_centers():
        return torch.stack([graph.center_train(z, ids) for ids in class_origins])
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = jsd(all_centers(), targets_t).mean()
        loss.backward()
        opt.step()
        count['exact_grad'] += 1
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            with torch.no_grad():
                err = float(jsd(all_centers(), targets_t).mean().item())
            count['exact_forward'] += 1
            trace.append({'step': step + 1, 'mean_JSD': err})
            if err <= tol:
                break
    return z.detach(), time.perf_counter() - t0, count, trace

def run_experiment(seed=20261107, nref=72, q_steps=45, exact_steps=90,
                   bootstrap_steps=30, tol=1e-7, lr=0.12, beta=.4, lam=.8,
                   horizon=12, include_validation=True):
    torch.set_num_threads(1)
    spec = load_config(ROOT / 'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as tmp:
        data = generate(spec, tmp, finite_q=False)
    full_x, full_y = data['train'][0], data['train'][2]
    target, posterior = bayes_class_centers(spec, full_x, full_y)
    k = len(spec['pi'])
    if nref > len(full_y) or nref % k:
        raise ValueError('nref must be a multiple of K and <= full training size')
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(full_y == c), nref // k, replace=False) for c in range(k)])
    rng.shuffle(chosen)
    x, y = full_x[chosen], full_y[chosen]
    vx, vy = data['validation'][0], data['validation'][2]
    val_posterior = bayes_posterior(spec, vx)
    tsetup = time.perf_counter()
    graph = ExactGraph(x, y, spec['gamma'], np.arange(nref), vx,
                       beta=beta, lam=lam, horizon=horizon)
    setup_sec = time.perf_counter() - tsetup
    tbound = time.perf_counter()
    feasibility = center_bounds(graph, y)
    bound_time_s = time.perf_counter() - tbound
    lower = np.asarray(feasibility['lower_center_probabilities'])
    upper = np.asarray(feasibility['upper_center_probabilities'])
    clamp = np.minimum(np.maximum(target, lower), upper)
    lower_bound_jsd = []
    for c in range(k):
        # JSD contracts when K categories are collapsed to one versus others.
        # This lower bound need NOT be tight, but a positive bound proves
        # unattainability even with arbitrary origin/time-varying policies.
        binary_js = [jsdiv(np.array([target[c, j], 1 - target[c, j]]),
                           np.array([clamp[c, j], 1 - clamp[c, j]]))
                     for j in range(k)]
        lower_bound_jsd.append(float(max(binary_js)))
    feasibility.update({'target_inside_coordinate_intervals': ((lower - 1e-12 <= target) & (target <= upper + 1e-12)).tolist(),
                        'proved_infeasible_center_rows': [c for c in range(k) if np.any((target[c] > upper[c] + 1e-12) | (target[c] < lower[c] - 1e-12))],
                        'JSD_lower_bound_by_binary_coarsening': lower_bound_jsd,
                        'calculation_time_s': bound_time_s})
    methods = ('q_only', 'exact_adam', 'q_bootstrap_gn')
    policies = {method: [] for method in ('uniform',) + methods}
    records = []
    for c in range(k):
        idx = np.flatnonzero(y == c)
        u = torch.tensor(target[c], dtype=DT)
        neutral = torch.zeros((graph.d + 1, graph.d), dtype=DT)
        zq, tq, cq = warm_q(graph, idx, u, q_steps, lr)
        t0 = time.perf_counter()
        eq = float(jsd(graph.center_train(zq, idx), u).item())
        tq += time.perf_counter() - t0
        cq['exact_forward'] += 1
        ze, te, ce, etrace = exact_adam(graph, idx, u, exact_steps, lr, tol)
        zb, tb, cb, btrace = damped_gauss_newton(graph, idx, u, zq, bootstrap_steps, tol)
        neutral_restart = None
        if btrace[-1]['error'] > tol:
            z_alt, t_alt, c_alt, t_alt_trace = damped_gauss_newton(graph, idx, u, neutral, bootstrap_steps, tol)
            neutral_restart = {'elapsed_s': t_alt, 'cost': c_alt,
                               'error': t_alt_trace[-1]['error'],
                               'accepted_as_best': t_alt_trace[-1]['error'] < btrace[-1]['error'],
                               'trace': t_alt_trace}
            tb += t_alt
            cb = add_counts(cb, c_alt)
            if neutral_restart['accepted_as_best']:
                zb = z_alt
        t0 = time.perf_counter()
        eb = float(jsd(graph.center_train(zb, idx), u).item())
        tb += time.perf_counter() - t0
        cb['exact_forward'] += 1
        policies['uniform'].append(neutral)
        policies['q_only'].append(zq)
        policies['exact_adam'].append(ze)
        policies['q_bootstrap_gn'].append(zb)
        records.append({
            'class': c, 'target': target[c].tolist(),
            'q_only': {'error': eq, 'elapsed_s': tq, 'cost': cq, 'Z': zq.numpy().tolist()},
            'exact_adam': {'error': float(jsd(graph.center_train(ze, idx), u).item()),
                           'elapsed_s': te, 'cost': ce, 'Z': ze.numpy().tolist(), 'trace': etrace},
            'q_bootstrap_gn': {'error': eb, 'elapsed_s': tq + tb,
                               'cost': add_counts(cq, cb), 'Z': zb.numpy().tolist(),
                               'q_warmup_s': tq, 'gn_and_restart_s': tb,
                               'trace': btrace, 'neutral_restart': neutral_restart},
        })
    zshared, tshared, cshared, trshared = shared_exact_adam(
        graph, y, target, steps=exact_steps, lr=lr, tol=tol)
    policies['shared_exact_adam'] = [zshared] * k
    comparisons = {}
    for method, pp in policies.items():
        t0 = time.perf_counter()
        tr = summarize_training(graph, y, pp, target, posterior[chosen])
        if include_validation:
            v = heldout_candidate_metrics(graph, pp, np.asarray(tr['centers']), vy, val_posterior)
        else:
            v = None
        eval_sec = time.perf_counter() - t0
        if method == 'uniform':
            costs = None
        elif method == 'shared_exact_adam':
            costs = {'elapsed_s_total': tshared, 'counts_total': cshared, 'trace': trshared}
        else:
            costs = {'elapsed_s_total': float(sum(row[method]['elapsed_s'] for row in records)),
                     'counts_total': {key: int(sum(row[method]['cost'].get(key, 0) for row in records))
                                      for key in records[0][method]['cost']}}
        comparisons[method] = {**tr, 'validation': v, 'timing': costs,
                               'diagnostic_evaluation_s': float(eval_sec)}
    # Baseline Bayes is a posterior reference; it does NOT arise from a graph policy.
    oracle_within_JS = float(np.mean([jsdiv(posterior[full_y == c], target[c]).mean() for c in range(k)]))
    return {'meta': {'seed': seed, 'nref': nref, 'n_train_full': len(full_y),
                     'n_validation': len(vx), 'K': k, 'd': graph.d,
                     'observed_sample_feature_nodes': int(np.isfinite(x).sum()),
                     'edges_feature_transition_total': int(sum(int(v)*(int(v)-1) for v in np.isfinite(x).sum(axis=0))),
                     'policy_switch_pairs_total': int((np.isfinite(x).sum(axis=1)**2).sum()),
                     'nref_indices_in_original_training': chosen.tolist(),
                     'targets_training_only': True,
                     'targets': 'per_true_label average of exact Bayes posterior over all 300 training samples',
                     'policy_family': 'three per-class hypothesis Z_c baselines; plus a strict single shared Z for all classes',
                     'label_usage': 'true class only for fitting class-specific policies and oracle training diagnostic',
                     'beta': beta, 'lambda': lam, 'horizon': horizon, 'step_coefficients': coefficients(beta, lam, horizon).tolist(),
                     'q_steps': q_steps, 'exact_steps': exact_steps, 'bootstrap_steps': bootstrap_steps,
                     'tol': tol, 'lr': lr, 'max_wall_time_notes': 'CPU one thread; setup shared; method timers do not include external graph construction'},
            'Bayes': {'centers': target.tolist(),
                      'center_geometry': center_geometry(target),
                      'train_oracle_within_JSD': oracle_within_JS,
                      'train_mean_posterior_entropy': float(entropy(posterior).mean()),
                      'validation_mean_posterior_entropy': float(entropy(val_posterior).mean())},
            'setup_time_s': setup_sec, 'relaxed_feasibility_bounds': feasibility,
            'methods': comparisons, 'per_class': records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--nref', type=int, default=72)
    p.add_argument('--q-steps', type=int, default=45)
    p.add_argument('--exact-steps', type=int, default=90)
    p.add_argument('--bootstrap-steps', type=int, default=30)
    p.add_argument('--tol', type=float, default=1e-7)
    p.add_argument('--lr', type=float, default=.12)
    p.add_argument('--no-validation', action='store_true')
    p.add_argument('--out', type=Path, default=HERE / 'results/bayes_seed_20261107.json')
    args = p.parse_args()
    result = run_experiment(seed=args.seed, nref=args.nref,
                            q_steps=args.q_steps, exact_steps=args.exact_steps,
                            bootstrap_steps=args.bootstrap_steps, tol=args.tol,
                            lr=args.lr, include_validation=not args.no_validation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Bayes class-mean targets:', np.round(result['Bayes']['centers'], 6).tolist(), flush=True)
    for name, v in result['methods'].items():
        print(f'{name:17s} meanJSD={v["mean_target_JSD"]:.10f} '
              f'I={v["center_geometry"]["I"]:.7f} R_JS={v["train_within_JSD"]:.7f} '
              f'time={v["timing"]["elapsed_s_total"]:.3f}s' if v['timing'] else
              f'{name:17s} meanJSD={v["mean_target_JSD"]:.10f} '
              f'I={v["center_geometry"]["I"]:.7f} R_JS={v["train_within_JSD"]:.7f}', flush=True)
    print('Saved:', args.out, flush=True)

if __name__ == '__main__':
    main()
