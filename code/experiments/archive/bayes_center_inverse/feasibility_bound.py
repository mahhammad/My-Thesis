"""Rigorous per-coordinate reachable bounds via finite-horizon Bellman optimization.

For each origin s and label c, maximize/minimize the *expected normalized*
walk label-mass in c over ALL origin-conditional, time-dependent policies.
The action at each intermediate (sample, entry-feature, time) may choose any
legal output feature. This is a SUPERSET of stationary shared-Z policies, so
violations of a bound prove a Bayes target coordinate is unattainable.

No Monte Carlo; uses the exact per-feature, origin-excluded transition kernels.
"""
import numpy as np


def bounds_for_all_origins(graph, labels):
    n, k, d = graph.n, graph.k, graph.d
    alpha = graph.coeff.detach().numpy()
    H = len(alpha)
    trans = graph.transitions.detach().numpy()  # [origin,b,i,j]
    start = graph.start.detach().numpy()  # [origin,j,b]
    good = graph.valid.detach().numpy()  # [origin,i,b]
    goodstart = graph.valid_start.detach().numpy()
    lo = np.zeros((n, k))
    hi = np.zeros((n, k))
    if len(graph.origins) != n or not np.array_equal(graph.origins, np.arange(n)):
        raise ValueError('bounds require every reference node to be an origin in index order')
    for s in range(n):
        for c in range(k):
            reward = (labels == c).astype(float)
            low = np.broadcast_to(alpha[-1] * reward[:, None], (n, d)).copy()
            high = low.copy()
            for t in range(H - 2, -1, -1):
                # continuation for each available choice of feature b
                low_by_feature = np.einsum('bij,jb->ib', trans[s], low, optimize=True)
                high_by_feature = np.einsum('bij,jb->ib', trans[s], high, optimize=True)
                min_choice = np.where(good[s], low_by_feature, np.inf).min(axis=1)
                max_choice = np.where(good[s], high_by_feature, -np.inf).max(axis=1)
                # At the forbidden origin, value is never used after the first step.
                min_choice[s] = 0.
                max_choice[s] = 0.
                low = np.broadcast_to((alpha[t] * reward + min_choice)[:, None], (n, d)).copy()
                high = np.broadcast_to((alpha[t] * reward + max_choice)[:, None], (n, d)).copy()
            first_lo = np.einsum('jb,jb->b', start[s], low)
            first_hi = np.einsum('jb,jb->b', start[s], high)
            lo[s, c] = np.min(first_lo[goodstart[s]])
            hi[s, c] = np.max(first_hi[goodstart[s]])
    return lo, hi


def center_bounds(graph, labels):
    per_origin_lo, per_origin_hi = bounds_for_all_origins(graph, labels)
    k = graph.k
    lb = np.stack([per_origin_lo[labels == c].mean(axis=0) for c in range(k)])
    ub = np.stack([per_origin_hi[labels == c].mean(axis=0) for c in range(k)])
    return {'lower_center_probabilities': lb.tolist(),
            'upper_center_probabilities': ub.tolist(),
            'diagonal_lower': np.diag(lb).tolist(),
            'diagonal_upper': np.diag(ub).tolist(),
            'meaning': 'coordinate-wise bounds under relaxed origin/time-specific arbitrary feature policies; outside interval proves impossibility for the restricted shared-Z policies'}


def origin_return_relaxed_bounds(x, labels, gamma, alpha):
    """Cheap O(K*H*E) valid bound allowing returns to origin.

    No origin-specific matrices are formed. The controller is even allowed to
    vary by time and state. If the Bayes target is outside these intervals,
    it is certainly unattainable by the original source-excluding model.
    Otherwise inconclusive, because the relaxation can be very loose.
    """
    from run import wgraph
    x = np.asarray(x)
    labels = np.asarray(labels)
    base = wgraph(x, gamma)  # [feature,current,dest]
    n, d = x.shape
    k = int(labels.max()) + 1
    good = base.sum(axis=2).T > 1e-12  # [current,feature]
    bound_l = np.empty((k, k))
    bound_u = np.empty((k, k))
    for c in range(k):
        reward = (labels == c).astype(float)
        low = np.broadcast_to(alpha[-1] * reward[:, None], (n,d)).copy()
        high = low.copy()
        for t in range(len(alpha)-2, -1, -1):
            low_choice = np.einsum('bij,jb->ib', base, low, optimize=True)
            high_choice = np.einsum('bij,jb->ib', base, high, optimize=True)
            lowbest = np.where(good,low_choice,np.inf).min(axis=1)
            highbest = np.where(good,high_choice,-np.inf).max(axis=1)
            low = np.broadcast_to((alpha[t]*reward+lowbest)[:,None],(n,d)).copy()
            high = np.broadcast_to((alpha[t]*reward+highbest)[:,None],(n,d)).copy()
        at_start_low = np.einsum('bij,jb->ib', base, low, optimize=True)
        at_start_high = np.einsum('bij,jb->ib', base, high, optimize=True)
        start_min = np.where(good, at_start_low, np.inf).min(axis=1)
        start_max = np.where(good, at_start_high, -np.inf).max(axis=1)
        for yc in range(k):
            bound_l[yc,c] = start_min[labels == yc].mean()
            bound_u[yc,c] = start_max[labels == yc].mean()
    return {'lower':bound_l.tolist(), 'upper':bound_u.tolist()}
