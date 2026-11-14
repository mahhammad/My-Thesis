"""Build fixed class centers for a given policy; training labels used here only.

Usage: python centers.py --train train.npz --policy policy.json --out centers.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from walks import (ReferenceGraph, load_data, load_policy, policy_fingerprint,
                   reference_fingerprint)
from metrics import report, class_centers, METRICS


def build_centers(reference_x: np.ndarray, reference_y: np.ndarray,
                  policy: dict, sample_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute policy-dependent neighborhoods and classwise mean centers."""
    graph = ReferenceGraph(reference_x, reference_y, policy['kernel_gamma'])
    neighborhoods, no_walk = graph.representation(reference_x, policy,
                                                  train_indices=np.arange(len(reference_x)))
    if no_walk.any():
        raise ValueError('training source has no admissible transitions')
    centers = class_centers(neighborhoods, reference_y, sample_weights=sample_weights)
    if not np.allclose(centers.sum(axis=1), 1, atol=1e-10):
        raise RuntimeError('centers are not normalized class label distributions')
    return centers, neighborhoods


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--metric', choices=METRICS, default='js_similarity')
    ap.add_argument('--class-weights', nargs='+', type=float, default=None,
                    help='Class weights for training report, not center fitting')
    args = ap.parse_args()
    x, y = load_data(args.train)
    policy = load_policy(args.policy, x.shape[1])
    centers, neighborhoods = build_centers(x, y, policy)
    fingerprint = reference_fingerprint(x, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, centers=centers,
                        classes=np.arange(len(centers)),
                        policy_fingerprint=policy_fingerprint(policy),
                        reference_fingerprint=fingerprint,
                        training_neighborhoods=neighborhoods)
    report_data, predictions = report(neighborhoods, centers, y,
                                      metric=args.metric, class_weights=args.class_weights)
    report_data['policy_name'] = policy['name']
    report_data['stage'] = 'training_fit_centers'
    report_data['training_metrics_are_not_holdout'] = True
    report_data['walks_per_sample'] = policy['walks_per_sample']
    report_data['max_steps'] = policy['max_steps']
    (args.out.parent/(args.out.stem+'_train_report.json')).write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {args.out} with {len(centers)} policy-dependent centers.')
    print(f'Training accuracy (not holdout): {report_data["accuracy"]:.3f}')


if __name__ == '__main__':
    main()
