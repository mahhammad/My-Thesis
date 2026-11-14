"""Exact and provably controlled truncated deterministic propagation of random walks.

This is NOT a new random-walk policy. For a capped geometric length L and per
walk lambda-normalized steps, the expectation is a convex combination of hop
label distributions with coefficient alpha_t. Truncating after h hops and
renormalizing the first h alpha coefficients incurs total-variation error
at most sum_{t>h} alpha_t, for ANY graph and admissible stationary policy.

No Monte Carlo and no powers of a dense (nd)x(nd) matrix are materialized.
"""
import numpy as np
import torch


def deterministic_representations(graph, z, origins, max_hops=None):
    idx = torch.as_tensor(origins, dtype=torch.long)
    if max_hops is None:
        max_hops = graph.H
    if not 1 <= max_hops <= graph.H:
        raise ValueError('max_hops must be between 1 and graph.H')
    policy = graph.local_policy(z)[idx]
    first = graph.masked_softmax(z[graph.d][None, :].expand(len(idx), -1), graph.valid_start[idx])
    state = graph.start[idx] * first[:, None, :]
    alpha = graph.coeff[:max_hops]
    alpha = alpha / alpha.sum()
    vec = torch.zeros((len(idx), graph.k), dtype=state.dtype)
    for t in range(max_hops):
        vec = vec + alpha[t] * torch.einsum('mjb,jk->mk', state, graph.yhot)
        if t == max_hops - 1:
            break
        selected = torch.einsum('mja,mjab->mjb', state, policy)
        step = selected.permute(0, 2, 1).reshape(-1, 1, graph.n)
        trans = graph.transitions[idx].reshape(-1, graph.n, graph.n)
        state = torch.bmm(step, trans).reshape(len(idx), graph.d, graph.n).transpose(1, 2)
    return vec


def approximation_diagnostic(graph, z_by_class, labels, cutoff=(1, 2, 3, 4, 6)):
    from run_bayes import jsdiv
    results = {}
    with torch.no_grad():
        for c, z in enumerate(z_by_class):
            ids = np.flatnonzero(labels == c)
            exact = deterministic_representations(graph, z, ids, graph.H).cpu().numpy()
            approx = {}
            for h in cutoff:
                if h > graph.H:
                    continue
                pred = deterministic_representations(graph, z, ids, h).cpu().numpy()
                tv = .5 * np.abs(exact - pred).sum(axis=1)
                tail = float(graph.coeff[h:].sum())
                if np.max(tv) > tail + 1e-9:
                    raise AssertionError('Tail bound violated')
                approx[str(h)] = {'tail_mass_TV_bound': tail,
                                  'empirical_mean_TV': float(tv.mean()),
                                  'empirical_max_TV': float(tv.max()),
                                  'empirical_mean_JSD': float(jsdiv(exact, pred).mean())}
            results[str(c)] = approx
    return results
