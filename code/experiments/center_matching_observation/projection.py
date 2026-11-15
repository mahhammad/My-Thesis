"""Batched convex projection of target class distributions onto attainable Q mixtures.

No learning, no class parameter fitting.  NumPy-only conditional-gradient solver.
"""
from __future__ import annotations
import numpy as np


def entropy(p):
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.), axis=-1)


def jsd(p, q):
    m = 0.5 * (p + q)
    return np.maximum(0., entropy(m) - 0.5 * (entropy(p) + entropy(q)))


def _gradient(p, v):
    # Numerical stabilization ONLY for the optimization derivative at simplex boundaries.
    # The objective and measured JSD always use the exact p and v.
    eps = 1e-15
    return 0.5 * np.log((2. * np.maximum(p, eps)) /
                        np.maximum(p + v, eps))


def optimize_mixtures(q, targets, *, steps=70, tol=1e-9, bisect=18):
    """Return constrained feature policy, residual JSD and FW dual-gap.

    q: (M,D,K), with NaN for forbidden features; targets: (M,K) or (K,).
    Uses exact line search up to bisection tolerance for the JSD objective.
    Rows with no permitted feature raise ValueError. This is an *approximate*
    optimization algorithm: use returned dual gaps to assess convergence.
    """
    q = np.asarray(q, float)
    if q.ndim != 3 or q.shape[-1] < 2:
        raise ValueError('Q must have shape (M,D,K)')
    m, d, k = q.shape
    v = np.broadcast_to(np.asarray(targets, float), (m, k))
    if (v < 0).any() or not np.allclose(v.sum(axis=1), 1, atol=1e-8):
        raise ValueError('targets must be probability distributions')
    active = np.isfinite(q).all(axis=-1)
    if not active.any(axis=1).all():
        raise ValueError('at least one state has no admissible feature')
    if np.any(q[active] < -1e-12) or not np.allclose(q[active].sum(axis=-1), 1., atol=1e-8):
        raise ValueError('Q rows must be probability distributions')
    rows = np.where(active[..., None], q, 0.)
    pi = active.astype(float) / active.sum(axis=-1, keepdims=True)
    vals = np.einsum('md,mdk->mk', pi, rows)
    gap = np.full(m, np.inf)
    for _ in range(steps):
        g = _gradient(vals, v)
        cost = np.einsum('mdk,mk->md', rows, g)
        cost[~active] = np.inf
        best = np.argmin(cost, axis=1)
        # Dual gap bounds suboptimality for convex objective.
        gap = np.einsum('md,md->m', pi, np.where(active,cost,0.)) - cost[np.arange(m),best]
        todo = gap > tol
        if not todo.any():
            break
        idx = np.flatnonzero(todo)
        q_best = rows[idx, best[idx], :]
        delta = q_best - vals[idx]
        # Exact line search on a convex line segment by derivative bisection.
        deriv_at_one = np.sum(delta * _gradient(q_best, v[idx]), axis=1)
        high = np.ones(len(idx))
        low = np.zeros(len(idx))
        for _ in range(bisect):
            middle = 0.5 * (low + high)
            p_mid = vals[idx] + middle[:,None] * delta
            derivative = np.sum(delta * _gradient(p_mid, v[idx]), axis=1)
            high = np.where(derivative >= 0, middle, high)
            low = np.where(derivative < 0, middle, low)
        alpha = np.where(deriv_at_one <= 0, 1., 0.5*(high+low))
        vals[idx] += alpha[:,None] * delta
        pi[idx] *= (1-alpha)[:,None]
        pi[idx,best[idx]] += alpha
    return pi, jsd(vals, v), np.maximum(0.,gap)
