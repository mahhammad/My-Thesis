"""Classification and geometry metrics for fixed class centers."""
from __future__ import annotations

import numpy as np


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=-1)


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Broadcast-compatible Jensen-Shannon divergence; log base e."""
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    m = 0.5 * (p + q)
    return np.clip(_entropy(m) - 0.5 * (_entropy(p) + _entropy(q)), 0, np.log(2))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(P||Q), with +inf for an unsupported positive component."""
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    invalid = np.any((p > 0) & (q <= 0), axis=-1)
    parts = np.where(p > 0, p * (np.log(np.maximum(p, 1e-300))
                                - np.log(np.maximum(q, 1e-300))), 0)
    return np.where(invalid, np.inf, parts.sum(axis=-1))


def classify(neighborhoods: np.ndarray, centers: np.ndarray) -> dict:
    """Predictions require only neighborhoods and centers, never truth labels."""
    js = jsd(neighborhoods[:, None, :], centers[None, :, :])
    similarity = 1 - js / np.log(2)
    if len(centers) > 1:
        order = np.argsort(-similarity, axis=1, kind='stable')
        top = order[:, 0]
        margin = similarity[np.arange(len(top)), top] - similarity[np.arange(len(top)), order[:, 1]]
    else:
        top = np.zeros(len(neighborhoods), dtype=int)
        margin = np.ones(len(neighborhoods))
    return {'prediction': top, 'similarity': similarity, 'jsd': js,
            'kl': kl_divergence(neighborhoods[:, None, :], centers[None, :, :]),
            'winning_margin': margin}


def report(neighborhoods: np.ndarray, centers: np.ndarray, y: np.ndarray | None) -> tuple[dict, dict]:
    from collections import OrderedDict
    n, k = neighborhoods.shape
    pred = classify(neighborhoods, centers)
    geometry = jsd(centers[:, None, :], centers[None, :, :])
    offdiag = geometry[~np.eye(k, dtype=bool)]
    result = OrderedDict(n_samples=int(n), n_classes=int(k),
                         mean_winning_margin=float(np.mean(pred['winning_margin'])),
                         class_center_jsd=geometry.tolist(),
                         min_center_js_distance=float(np.sqrt(offdiag.min())) if offdiag.size else 0,
                         center_similarity=(1-geometry/np.log(2)).tolist())
    if y is not None:
        y = np.asarray(y, dtype=int)
        if y.shape != (n,) or ((y < 0) | (y >= k)).any():
            raise ValueError('evaluation labels must be in 0..K-1')
        confusion = np.zeros((k, k), dtype=int)
        np.add.at(confusion, (y, pred['prediction']), 1)
        counts = confusion.sum(axis=1)
        recalls = np.divide(np.diag(confusion), counts, out=np.full(k, np.nan), where=counts!=0)
        correct_margin = pred['similarity'][np.arange(n), y] - np.max(
            np.where(np.eye(k, dtype=bool)[y], -np.inf, pred['similarity']), axis=1)
        js_own = pred['jsd'][np.arange(n), y]
        kl_own = pred['kl'][np.arange(n), y]
        priors = counts / n
        center_mean = priors @ centers
        mutual_information = _entropy(center_mean) - np.sum(priors*_entropy(centers))
        result.update(accuracy=float(np.trace(confusion)/n),
                      error_rate=float(1-np.trace(confusion)/n),
                      balanced_accuracy=float(np.nanmean(recalls)),
                      balanced_error=float(1-np.nanmean(recalls)),
                      confusion=confusion.tolist(),
                      per_class_recall=[float(v) if np.isfinite(v) else None for v in recalls],
                      per_class_counts=counts.tolist(),
                      mean_true_margin=float(correct_margin.mean()),
                      fraction_nonpositive_true_margin=float(np.mean(correct_margin <= 0)),
                      within_class_jsd=float(js_own.mean()),
                      within_class_kl=float(kl_own.mean()) if np.isfinite(kl_own).all() else None,
                      center_information=float(mutual_information))
        pred['true_margin'] = correct_margin
    return result, pred
