"""Evaluate a fixed policy with PRECOMPUTED centers; NEVER update centers.

Usage: python evaluate.py --reference train.npz --query test.npz \\
  --centers centers.npz --policy policy.json --out test_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from walks import (ReferenceGraph, load_data, load_policy, policy_fingerprint,
                   reference_fingerprint)
from metrics import report, METRICS


def run(reference_x: np.ndarray, reference_y: np.ndarray,
        query_x: np.ndarray, centers: np.ndarray, policy: dict,
        query_y: np.ndarray | None = None, *, metric: str = 'js_similarity',
        class_weights: np.ndarray | None = None):
    """No center updates; query labels used only in report(), never representation()."""
    graph = ReferenceGraph(reference_x, reference_y, policy['kernel_gamma'])
    if centers.shape != (graph.nclasses, graph.nclasses):
        raise ValueError('center shape does not match reference class set')
    p, no_walk = graph.representation(query_x, policy)
    if no_walk.any():
        raise ValueError('query has no admissible transitions')
    result, preds = report(p, centers, query_y, metric=metric,
                           class_weights=class_weights)
    return result, {'neighborhoods': p, **preds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--query', type=Path, required=True)
    ap.add_argument('--centers', type=Path, required=True)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--metric', choices=METRICS, default='js_similarity',
                    help='Distance/similarity used for nearest-center prediction')
    ap.add_argument('--class-weights', nargs='+', type=float, default=None,
                    help='Class weights (default: uniform across classes)')
    ap.add_argument('--out', type=Path, required=True,
                    help='prefix; creates <out>.json and <out>.npz')
    args = ap.parse_args()
    ref_x, ref_y = load_data(args.reference)
    query_x, query_y = load_data(args.query, labels_required=False)
    policy = load_policy(args.policy, ref_x.shape[1])
    with np.load(args.centers, allow_pickle=False) as z:
        centers = z['centers'].copy()
        if str(z['policy_fingerprint'].item()) != policy_fingerprint(policy):
            raise ValueError('center policy differs from evaluation policy')
        if str(z['reference_fingerprint'].item()) != reference_fingerprint(ref_x, ref_y):
            raise ValueError('reference data differs from the center-building reference')
    result, details = run(ref_x, ref_y, query_x, centers, policy, query_y,
                          metric=args.metric, class_weights=args.class_weights)
    result.update(stage='heldout_evaluation', policy_name=policy['name'],
                  centers_source=str(args.centers),
                  reference_source=str(args.reference),
                  query_source=str(args.query),
                  centers_updated=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(args.out.with_suffix('.npz'), **details)
    print(f'Report: {args.out.with_suffix(".json")}')
    if 'accuracy' in result:
        print(f'Accuracy: {result["accuracy"]:.3f}; balanced: {result["balanced_accuracy"]:.3f}')
    else:
        print('Prediction-only: no query labels were supplied.')


if __name__ == '__main__':
    main()
