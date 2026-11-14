"""Exact finite-reference one-step label distributions, without Monte Carlo.

Q[query, feature, class] = P(label of next node | query, chosen feature).
Destinations exclude the current reference node and, optionally, walk origin.
This is different from the Gaussian population oracle in data_generator.
"""
from __future__ import annotations

import numpy as np
from walks import ReferenceGraph, _weights


def one_step_q(graph: ReferenceGraph, queries: np.ndarray | None = None,
               train_indices: np.ndarray | None = None,
               origin_indices: np.ndarray | None = None,
               batch_size: int = 64) -> np.ndarray:
    """Return Q with shape (N_query, D, K), NaN for absent/invalid features.

    queries=None computes Q for every training node as the first step, excluding
    itself. train_indices identifies queries which are nodes in the reference.
    origin_indices optionally bans an *additional* original start node, to
    compute Q for intermediate positions in walks where returning is forbidden.
    Neither policy logits nor query labels are used.
    """
    if queries is None:
        queries = graph.x
        if train_indices is None:
            train_indices = np.arange(len(graph.x))
    queries = np.asarray(queries, float)
    if queries.ndim != 2 or queries.shape[1] != graph.x.shape[1]:
        raise ValueError('queries must have shape (N,D)')
    n, d = queries.shape
    nr, k = len(graph.x), graph.nclasses
    ids = np.full(n, -1, int) if train_indices is None else np.asarray(train_indices, int)
    origins = np.full(n, -1, int) if origin_indices is None else np.asarray(origin_indices, int)
    if ids.shape != (n,) or origins.shape != (n,):
        raise ValueError('train_indices/origin_indices must have N entries')
    if ((ids < -1) | (ids >= nr)).any() or ((origins < -1) | (origins >= nr)).any():
        raise ValueError('invalid reference index')
    valid_ids = ids >= 0
    if valid_ids.any() and not np.array_equal(np.isnan(queries[valid_ids]),
                                               np.isnan(graph.x[ids[valid_ids]])):
        raise ValueError('train_indices mask does not match the reference')
    if valid_ids.any() and not np.array_equal(np.nan_to_num(queries[valid_ids]),
                                               np.nan_to_num(graph.x[ids[valid_ids]])):
        raise ValueError('train_indices values do not match the reference')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be positive')
    out = np.full((n, d, k), np.nan)
    one_hot = np.eye(k)[graph.y]
    for first in range(0, n, batch_size):
        last = min(n, first+batch_size)
        rows = np.arange(first, last)
        # Cached reference-reference weights include the self exclusion.
        # For unseen queries or mixed subsets, compute feature kernels afresh.
        if valid_ids[rows].all():
            w = graph.weights[ids[rows]].copy()
        else:
            w = _weights(queries[rows], graph.x, graph.gamma, ids[rows])
        banned = origins[rows] >= 0
        if banned.any():
            w[np.flatnonzero(banned), :, origins[rows][banned]] = 0.
        totals = w.sum(axis=-1)
        numerators = w @ one_hot
        # Advanced indexing created a copy; assign the completed batch back.
        batch = np.full((len(rows), d, k), np.nan)
        np.divide(numerators, totals[..., None], out=batch,
                  where=(totals[..., None] > 0))
        out[rows] = batch
    return out


def main():
    """CLI to materialize the exact finite-reference Q matrices."""
    import argparse
    from pathlib import Path
    from walks import load_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True,
                        help='Read kernel_gamma; policy logits are not used')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-queries', action='store_true',
                        help='Query rows exactly correspond to reference rows')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    from walks import load_policy
    rx, ry = load_data(args.reference)
    qx, _ = load_data(args.query, labels_required=False)
    policy = load_policy(args.policy, rx.shape[1])
    graph = ReferenceGraph(rx, ry, policy['kernel_gamma'])
    ids = np.arange(len(rx)) if args.train_queries else None
    if args.train_queries and (qx.shape != rx.shape or
                               not np.array_equal(np.nan_to_num(qx, nan=0), np.nan_to_num(rx, nan=0)) or
                               not np.array_equal(np.isnan(qx), np.isnan(rx))):
        raise ValueError('--train-queries requires the same ordered reference data')
    q = one_step_q(graph, qx, train_indices=ids, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, q_finite=q)
    print(f'Saved one-step Q, shape={q.shape}, to {args.out}')


if __name__ == '__main__':
    main()
