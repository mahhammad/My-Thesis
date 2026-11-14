#!/usr/bin/env python3
"""Inverse center -> feature policy: feasible targets, Q baseline, exact Adam, Q-warm GN.

Laboratory tool, NOT a proposed policy-learning model. All methods use the
same class-conditional (d+1,d) policy family and frozen attainable targets.
The cost of Q warm-start is included in the hybrid/bootstrap timer.
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
from generate import generate, load_config  # noqa: E402
from run import DT, ExactGraph, jsd  # noqa: E402


def make_problem(seed: int, nref: int, teacher_scale: float = 1.7,
                 beta: float = .4, lam: float = .8, horizon: int = 12):
    spec = load_config(ROOT / 'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as tmp:
        data = generate(spec, tmp, finite_q=False)
    x_all, y_all = data['train'][0], data['train'][2]
    k = len(spec['pi'])
    if nref % k:
        raise ValueError('nref must be divisible by number of classes')
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(y_all == c), nref // k,
                                        replace=False) for c in range(k)])
    rng.shuffle(chosen)
    x, y = x_all[chosen], y_all[chosen]
    graph = ExactGraph(x, y, spec['gamma'], np.arange(nref),
                       data['validation'][0], beta=beta, lam=lam, horizon=horizon)
    # Fix a known feasible target, using a hidden witness policy for each class.
    teachers = rng.normal(0, teacher_scale, size=(k, graph.d + 1, graph.d))
    # Avoid softmax row-shift redundancy in stored logits for reproducible inspection.
    teachers -= teachers.mean(axis=-1, keepdims=True)
    with torch.no_grad():
        target = np.stack([graph.center_train(torch.tensor(teachers[c], dtype=DT),
                                             np.flatnonzero(y == c)).numpy()
                           for c in range(k)])
    return graph, y, target, teachers, chosen


def exact_error(graph, z, indices, u):
    return float(jsd(graph.center_train(z, indices), u).item())


def warm_q(graph, indices, u, steps: int, lr: float):
    """Previous local Q approximation, now stopped without exact multi-step gradients."""
    z = torch.nn.Parameter(torch.zeros((graph.d + 1, graph.d), dtype=DT))
    opt = torch.optim.Adam([z], lr=lr)
    start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = graph.local_q_loss(z, indices, u)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    return z.detach(), elapsed, {'q_grad': steps, 'exact_forward': 0, 'exact_grad': 0,
                                  'jacobian': 0, 'line_search': 0}


def exact_adam(graph, indices, u, steps: int, lr: float, tol: float):
    z = torch.nn.Parameter(torch.zeros((graph.d + 1, graph.d), dtype=DT))
    opt = torch.optim.Adam([z], lr=lr)
    count = {'q_grad': 0, 'exact_grad': 0, 'exact_forward': 0,
             'jacobian': 0, 'line_search': 0}
    trace = []
    start = time.perf_counter()
    # Initial/final measurements and periodic acceptance cost all included.
    with torch.no_grad():
        e = exact_error(graph, z, indices, u)
    count['exact_forward'] += 1
    trace.append({'step': 0, 'error': e, 'elapsed_s': time.perf_counter() - start})
    for t in range(steps):
        if e <= tol:
            break
        opt.zero_grad(set_to_none=True)
        loss = jsd(graph.center_train(z, indices), u)
        loss.backward()
        opt.step()
        count['exact_grad'] += 1
        with torch.no_grad():
            e = exact_error(graph, z, indices, u)
        count['exact_forward'] += 1
        trace.append({'step': t + 1, 'error': e, 'elapsed_s': time.perf_counter() - start})
    return z.detach(), time.perf_counter() - start, count, trace


def _center_coordinates(graph, zflat, indices):
    z = zflat.reshape(graph.d + 1, graph.d)
    # The omitted last coordinate is redundant (sum of center entries is one).
    return graph.center_train(z, indices)[:-1]


def damped_gauss_newton(graph, indices, u, z_init, max_steps: int, tol: float,
                        initial_damping: float = 1e-4, trust_radius: float = 2.5,
                        max_backtrack: int = 12):
    """Jacobian-based bootstrapped center correction; accepts only actual improvements.

    Solve min ||J delta - (target-current)||^2 + mu ||delta||^2 via
    delta = J^T (JJ^T + mu I)^-1 residual, using a small (K-1)x(K-1) solve.
    Backtracking uses the full exact JSD objective, not linearized residual.
    """
    z = z_init.detach().clone()
    count = {'q_grad': 0, 'exact_grad': 0, 'exact_forward': 0,
             'jacobian': 0, 'line_search': 0}
    t0 = time.perf_counter()
    with torch.no_grad():
        e = exact_error(graph, z, indices, u)
    count['exact_forward'] += 1
    trace = [{'step': 0, 'error': e, 'elapsed_s': time.perf_counter() - t0,
              'accepted': True}]
    damping = initial_damping
    for iteration in range(max_steps):
        if e <= tol:
            break
        zflat = z.reshape(-1).detach().requires_grad_(True)
        v = _center_coordinates(graph, zflat, indices)
        # Reuse graph from the single forward pass across K-1 reverse passes.
        rows = []
        for c in range(graph.k - 1):
            rows.append(torch.autograd.grad(v[c], zflat, retain_graph=c < graph.k - 2)[0])
        J = torch.stack(rows)  # [k-1, (d+1)*d]
        count['jacobian'] += 1
        residual = u[:-1] - v.detach()
        try:
            change = J.T @ torch.linalg.solve(J @ J.T + damping *
                                               torch.eye(graph.k - 1, dtype=DT), residual)
        except RuntimeError:
            break
        magnitude = torch.linalg.vector_norm(change)
        if not torch.isfinite(magnitude) or magnitude < 1e-12:
            break
        if magnitude > trust_radius:
            change = change * (trust_radius / magnitude)
        # Strict monotone acceptance; retry with smaller steps first.
        accepted = False
        for shrink in range(max_backtrack):
            trial = z + change.reshape(z.shape) * (0.5 ** shrink)
            with torch.no_grad():
                new_e = exact_error(graph, trial, indices, u)
            count['exact_forward'] += 1
            count['line_search'] += 1
            if new_e < e - 1e-14:
                z, e, accepted = trial.detach(), new_e, True
                damping = max(damping * 0.7, 1e-8)
                break
        trace.append({'step': iteration + 1, 'error': e,
                      'elapsed_s': time.perf_counter() - t0, 'accepted': accepted,
                      'damping': damping})
        if not accepted:
            # A linearized Euclidean residual direction need not descend in JSD.
            # Fall back to the exact merit-function gradient (still not a global
            # optimization guarantee), with the same strict acceptance test.
            zg = z.detach().clone().requires_grad_(True)
            merit = jsd(graph.center_train(zg, indices), u)
            grad = torch.autograd.grad(merit, zg)[0]
            count['exact_grad'] += 1
            gn = torch.linalg.vector_norm(grad)
            if torch.isfinite(gn) and gn > 1e-14:
                direction = -grad / gn * min(trust_radius, 1.0)
                for shrink in range(max_backtrack + 8):
                    trial = z + direction * (0.5 ** shrink)
                    with torch.no_grad():
                        new_e = exact_error(graph, trial, indices, u)
                    count['exact_forward'] += 1
                    count['line_search'] += 1
                    if new_e < e - 1e-14:
                        z, e, accepted = trial.detach(), new_e, True
                        trace[-1].update({'error': e, 'accepted': True,
                                          'fallback': 'exact_gradient'})
                        break
            damping *= 10
            if not accepted and damping > 1e4:
                break
    return z, time.perf_counter() - t0, count, trace


def add_counts(a, b):
    return {key: a.get(key, 0) + b.get(key, 0) for key in a.keys() | b.keys()}


def benchmark(seed=20261107, nref=36, q_steps=45, exact_steps=80,
              bootstrap_steps=12, lr=.12, tol=1e-7, teacher_scale=1.7):
    torch.set_num_threads(1)
    graph, labels, targets, teachers, chosen = make_problem(seed, nref, teacher_scale)
    records = []
    for c in range(graph.k):
        idx = np.flatnonzero(labels == c)
        u = torch.tensor(targets[c], dtype=DT)
        t0 = time.perf_counter()
        initial = torch.zeros((graph.d + 1, graph.d), dtype=DT)
        init_error = exact_error(graph, initial, idx, u)
        initial_time = time.perf_counter() - t0
        # Q-only baseline: its precise terminal error is measured, not optimized.
        z_q, tq, cq = warm_q(graph, idx, u, q_steps, lr)
        t0 = time.perf_counter()
        eq = exact_error(graph, z_q, idx, u)
        tq += time.perf_counter() - t0
        cq['exact_forward'] += 1
        # Direct exact optimizer, starting from same neutral zero logits.
        z_exact, te, ce, trace_exact = exact_adam(graph, idx, u, exact_steps, lr, tol)
        t0 = time.perf_counter()
        ee = exact_error(graph, z_exact, idx, u)
        te += time.perf_counter() - t0
        # Include this final verification in cost.
        ce['exact_forward'] += 1
        # Bootstrap includes the full cost of generating the Q policy.
        z_boot, tb, cb, trace_boot = damped_gauss_newton(graph, idx, u, z_q, bootstrap_steps, tol)
        bootstrap_restart = None
        # The Q approximation can land in a different basin. A neutral
        # restart is allowed only if the first bootstrap failed. Both the
        # extra Jacobians and the extra wall time count toward its cost.
        if trace_boot[-1]['error'] > tol:
            z_alt, t_alt, c_alt, trace_alt = damped_gauss_newton(
                graph, idx, u, initial, bootstrap_steps, tol)
            bootstrap_restart = {'trace': trace_alt, 'elapsed_s': t_alt,
                                 'cost': c_alt, 'accepted_as_best':
                                 trace_alt[-1]['error'] < trace_boot[-1]['error']}
            tb += t_alt
            cb = add_counts(cb, c_alt)
            if bootstrap_restart['accepted_as_best']:
                z_boot = z_alt
        t0 = time.perf_counter()
        eb = exact_error(graph, z_boot, idx, u)
        tb += time.perf_counter() - t0
        cb['exact_forward'] += 1
        records.append({
            'class': c, 'target': targets[c].tolist(), 'witness_policy': teachers[c].tolist(),
            'initial_error': init_error, 'initial_measurement_s': initial_time,
            'q_only': {'error': eq, 'time_s': tq, 'cost': cq, 'policy': z_q.numpy().tolist()},
            'exact_adam': {'error': ee, 'time_s': te, 'cost': ce,
                           'policy': z_exact.numpy().tolist(), 'trace': trace_exact},
            'q_bootstrap_gn': {'error': eb, 'time_s': tq + tb, 'cost': add_counts(cq, cb),
                               'policy': z_boot.numpy().tolist(),
                               'q_warmup_time_s': tq, 'gn_time_s': tb,
                               'q_initial_trace': trace_boot, 'neutral_restart': bootstrap_restart},
        })
    methods = ('q_only', 'exact_adam', 'q_bootstrap_gn')
    summary = {}
    for key in methods:
        errors = [row[key]['error'] for row in records]
        summary[key] = {
            'mean_JSD': float(np.mean(errors)), 'max_JSD': float(max(errors)),
            'wall_s_total': sum(row[key]['time_s'] for row in records),
            'classes_reached_tol': sum(e <= tol for e in errors),
            'cost_total': {v: sum(row[key]['cost'][v] for row in records) for v in records[0][key]['cost']},
        }
    return {
        'meta': {'seed': seed, 'data_seed': 20261102, 'nref': nref,
                 'nclasses': graph.k, 'nfeatures': graph.d, 'beta': .4,
                 'lambda': .8, 'horizon': 12, 'q_steps': q_steps,
                 'exact_steps': exact_steps, 'bootstrap_steps': bootstrap_steps,
                 'lr': lr, 'tol': tol, 'teacher_scale': teacher_scale,
                 'policy_family': 'class-conditional shared masked Z_c, (d+1)*d logits',
                 'targets': 'attainable exact multi-step centers from hidden teacher policies',
                 'cost_measurement': 'CPU wall time includes q warm-start for bootstrap;'
                                     ' verification exact passes included; single CPU thread',
                 'reference_indices': chosen.tolist()},
        'summary': summary, 'per_class': records,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--nref', type=int, default=36)
    p.add_argument('--q-steps', type=int, default=45)
    p.add_argument('--exact-steps', type=int, default=80)
    p.add_argument('--bootstrap-steps', type=int, default=12)
    p.add_argument('--lr', type=float, default=.12)
    p.add_argument('--tol', type=float, default=1e-7)
    p.add_argument('--teacher-scale', type=float, default=1.7)
    p.add_argument('--out', type=Path, default=HERE / 'results/bootstrap_seed_20261107.json')
    a = p.parse_args()
    output = benchmark(seed=a.seed, nref=a.nref, q_steps=a.q_steps,
                       exact_steps=a.exact_steps, bootstrap_steps=a.bootstrap_steps,
                       lr=a.lr, tol=a.tol, teacher_scale=a.teacher_scale)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('seed', a.seed, 'nref', a.nref, 'threshold', a.tol, flush=True)
    for k, v in output['summary'].items():
        print(f'{k:17s} JSD={v["mean_JSD"]:.9f} max={v["max_JSD"]:.9f} '
              f'classes_at_tol={v["classes_reached_tol"]}/{output["meta"]["nclasses"]} '
              f'time={v["wall_s_total"]:.2f}s', flush=True)
    print('saved', a.out, flush=True)


if __name__ == '__main__':
    main()
