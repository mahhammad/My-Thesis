"""Probability, classification and Chapter-3 representation metrics (NumPy only).

The score used to classify a query is configurable. Class centers, class weights,
and decision scores are explicit inputs; evaluation never refits centers.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

METRICS = ('kl', 'sqrt_jsd', 'js_similarity')
LOG2 = float(np.log(2.0))


def _probabilities(p, name='probabilities'):
    p = np.asarray(p, dtype=float)
    if p.ndim < 1 or p.shape[-1] < 2 or not np.isfinite(p).all():
        raise ValueError(f'{name} must contain finite probability vectors')
    if (p < 0).any() or not np.allclose(p.sum(axis=-1), 1., atol=1e-8):
        raise ValueError(f'{name} must be nonnegative and sum to one')
    return p


def entropy(p):
    p = _probabilities(p)
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.), axis=-1)


def _entropy(p):
    """Backward-compatible alias."""
    return entropy(p)


def jsd(p, q):
    """Jensen-Shannon divergence, with natural logarithms."""
    p, q = np.broadcast_arrays(_probabilities(p, 'p'), _probabilities(q, 'q'))
    m = (p + q) / 2
    return np.clip(entropy(m) - (entropy(p) + entropy(q)) / 2, 0., LOG2)


def sqrt_jsd(p, q):
    return np.sqrt(jsd(p, q))


def js_similarity(p, q):
    """1 - JSD/log(2), the Hilbert-cosine similarity used in Chapter 3."""
    return 1. - jsd(p, q) / LOG2


def kl_divergence(p, q):
    """Exact KL(P || Q). Unsupported positive mass gives +inf, not clipping."""
    p, q = np.broadcast_arrays(_probabilities(p, 'p'), _probabilities(q, 'q'))
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.)
    return terms.sum(axis=-1)


def pairwise_metric(p, centers, metric='js_similarity'):
    """Returns shape (n_queries, K); KL and sqrt-JSD are minimized."""
    if metric not in METRICS:
        raise ValueError(f'metric must be one of {METRICS}')
    p, centers = _probabilities(p, 'neighborhoods'), _probabilities(centers, 'centers')
    if p.ndim != 2 or centers.ndim != 2 or p.shape[1] != centers.shape[1] or centers.shape[0] != centers.shape[1]:
        raise ValueError('neighborhoods and centers must have shape (N,K) and (K,K)')
    a, b = p[:, None, :], centers[None, :, :]
    return {'kl': kl_divergence, 'sqrt_jsd': sqrt_jsd,
            'js_similarity': js_similarity}[metric](a, b)


def class_weights_vector(k, weights=None):
    """Normalize nonnegative class weights; uniform when unspecified."""
    w = np.ones(k) if weights is None else np.asarray(weights, float)
    if w.shape != (k,) or not np.isfinite(w).all() or (w < 0).any() or not w.sum() > 0:
        raise ValueError('class weights must be finite, nonnegative, and nonzero')
    return w / w.sum()


def sample_weights_vector(labels, k, class_weights=None, sample_weights=None):
    """Balanced by class; optional within-class weights and class weights."""
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1 or (y < 0).any() or (y >= k).any():
        raise ValueError('invalid class labels')
    cls = class_weights_vector(k, class_weights)
    w = np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights, float)
    if w.shape != (len(y),) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('sample weights must be nonnegative and finite')
    sums = np.bincount(y, weights=w, minlength=k)
    if np.any((cls > 0) & (sums == 0)):
        raise ValueError('every positively weighted class must have samples')
    out = np.zeros(len(y), dtype=float)
    valid = sums[y] > 0
    out[valid] = cls[y[valid]] * w[valid] / sums[y[valid]]
    return out


def class_centers(neighborhoods, labels, sample_weights=None, nclasses=None):
    """Mean neighborhood per training class, optionally weighted within class."""
    p = _probabilities(neighborhoods, 'neighborhoods')
    y = np.asarray(labels, dtype=int)
    if p.ndim != 2 or y.shape != (len(p),):
        raise ValueError('neighborhoods/labels shape mismatch')
    k = p.shape[1] if nclasses is None else nclasses
    if k != p.shape[1] or (y < 0).any() or (y >= k).any():
        raise ValueError('labels or class count mismatch')
    w = np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights, float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('invalid sample weights')
    counts = np.bincount(y, weights=w, minlength=k)
    if (counts == 0).any():
        raise ValueError('every class must have positive total weight')
    centers = np.zeros((k, k))
    np.add.at(centers, y, p * w[:, None])
    return centers / counts[:, None]


def _optimal_assignment(cost):
    """Minimum-cost bijection, exact subset DP; no SciPy dependency."""
    k = len(cost)
    if k > 16:
        raise ValueError('exact permutation metrics are limited to K<=16')
    @lru_cache(None)
    def solve(mask):
        r = mask.bit_count()
        if r == k:
            return 0.
        return min(cost[r, c] + solve(mask | (1 << c))
                   for c in range(k) if not (mask >> c) & 1)
    return float(solve(0))


def center_metrics(centers, class_weights=None):
    """Center-only T/TNL/BT/BTNL/I and JS geometry (Chapter 3)."""
    c = _probabilities(centers, 'centers')
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError('center matrix must be K x K')
    k = len(c)
    omega = class_weights_vector(k, class_weights)
    diag = np.diag(c)
    j = jsd(c[:, None, :], c[None, :, :])
    sep = np.sqrt(j[~np.eye(k, dtype=bool)].min()) if k > 1 else 0.
    mean = omega @ c
    # Define zero-probability logarithmic costs as +inf.
    with np.errstate(divide='ignore'):
        costs = -np.log(c)
    costs = np.where(omega[:, None] > 0, costs, 0.)
    return dict(T=float(omega @ diag), TNL=float(omega @ -np.log(np.where(diag > 0, diag, 1.))) if np.all((diag > 0) | (omega == 0)) else float('inf'),
                BT=-_optimal_assignment(-omega[:, None] * c),
                BTNL=_optimal_assignment(omega[:, None] * costs),
                I=float(entropy(mean) - omega @ entropy(c)),
                center_jsd=j, center_similarity=1. - j / LOG2,
                min_center_js_distance=float(sep), class_weights=omega)


def classify(neighborhoods, centers, metric='js_similarity'):
    """Predict without using labels. Tie-break: lowest class index."""
    values = pairwise_metric(neighborhoods, centers, metric)
    scores = values if metric == 'js_similarity' else -values
    order = np.argsort(-scores, axis=1, kind='stable')
    pred = order[:, 0]
    margin = scores[np.arange(len(pred)), pred] - scores[np.arange(len(pred)), order[:, 1]]
    p = _probabilities(neighborhoods)
    c = _probabilities(centers)
    js = jsd(p[:, None, :], c[None, :, :])
    return dict(prediction=pred, decision_metric=metric,
                decision_values=values, decision_scores=scores,
                similarity=1-js/LOG2, jsd=js, sqrt_jsd=np.sqrt(js),
                kl=kl_divergence(p[:, None, :], c[None, :, :]),
                winning_margin=margin)


def normalized_nll(kl_distances, labels, sample_weights=None):
    """Normalized KL-distance Softmax NLL; returns nan if distances are infinite."""
    d = np.asarray(kl_distances, dtype=float)
    y = np.asarray(labels, dtype=int)
    if d.ndim != 2 or y.shape != (len(d),):
        raise ValueError('distance/label dimensions mismatch')
    if not np.isfinite(d).all():
        return float('nan')
    std = d.std(axis=1)
    scores = np.zeros_like(d)
    np.divide(d.mean(axis=1, keepdims=True) - d, std[:, None],
              out=scores, where=std[:, None] > 1e-14)
    loss = np.logaddexp.reduce(scores, axis=1) - scores[np.arange(len(y)), y]
    w = np.ones(len(y))/len(y) if sample_weights is None else sample_weights
    return float(np.asarray(w) @ loss)


def soft_margin_loss(scores, labels, temperature, sample_weights=None):
    """tau*log(1+sum_{r!=y} exp((s_r-s_y)/tau)); higher scores are better."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(labels, int)
    if not temperature > 0 or scores.ndim != 2 or y.shape != (len(scores),):
        raise ValueError('invalid soft-margin inputs')
    delta = (scores - scores[np.arange(len(y)), y, None]) / temperature
    delta[np.arange(len(y)), y] = -np.inf
    loss = temperature * np.logaddexp(0., np.logaddexp.reduce(delta, axis=1))
    w = np.ones(len(y))/len(y) if sample_weights is None else sample_weights
    return float(np.asarray(w) @ loss)


def margin_criterion(neighborhoods, centers, labels, class_weights=None,
                     sample_weights=None):
    """J_margin = sum_c omega_c sum_{r!=c} var(M_cr)/(var+mean²).

    Returns nan if any center has zero entries (KL log margins need positive centers).
    """
    p, c = _probabilities(neighborhoods), _probabilities(centers)
    y = np.asarray(labels, int)
    if (c <= 0).any():
        return float('nan')
    within = sample_weights_vector(y, len(c), class_weights, sample_weights)
    ans = 0.
    omega = class_weights_vector(len(c), class_weights)
    for real in range(len(c)):
        if omega[real] == 0:
            continue
        ind = np.flatnonzero(y == real)
        if not len(ind):
            continue
        w = within[ind] / within[ind].sum()
        for rival in range(len(c)):
            if rival == real:
                continue
            coeff = np.log(c[real] / c[rival])
            margins = p[ind] @ coeff
            mean = float(w @ margins)
            var = float(w @ (margins - mean)**2)
            ans += omega[real] * (var / (var + mean*mean) if var+mean*mean > 1e-15 else 1.)
    return float(ans)


def report(neighborhoods, centers, y=None, metric='js_similarity',
           class_weights=None, sample_weights=None):
    """Metrics from provided representations/centers/labels/weights; no fitting.

    Default omega is UNIFORM, including when evaluating a held-out set.
    """
    p, c = _probabilities(neighborhoods), _probabilities(centers)
    pred = classify(p, c, metric)
    geom = center_metrics(c, class_weights)
    k = len(c)
    result = dict(n_samples=int(len(p)), n_classes=int(k), decision_metric=metric,
                  mean_winning_margin=float(np.mean(pred['winning_margin'])),
                  class_center_jsd=geom['center_jsd'].tolist(),
                  min_center_js_distance=geom['min_center_js_distance'],
                  center_similarity=geom['center_similarity'].tolist(),
                  center_information=geom['I'], T=geom['T'], TNL=geom['TNL'] if np.isfinite(geom['TNL']) else None,
                  BT=geom['BT'], BTNL=geom['BTNL'] if np.isfinite(geom['BTNL']) else None)
    if y is None:
        return result, pred
    y = np.asarray(y, int)
    w = sample_weights_vector(y, k, class_weights, sample_weights)
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (y, pred['prediction']), 1)
    recalls = np.divide(np.diag(confusion), confusion.sum(axis=1),
                        out=np.full(k, np.nan), where=confusion.sum(axis=1) != 0)
    true_scores = pred['decision_scores'][np.arange(len(y)), y]
    scores = pred['decision_scores'].copy()
    scores[np.arange(len(y)), y] = -np.inf
    true_margin = true_scores - np.max(scores, axis=1)
    d = pred['kl']
    own = d[np.arange(len(y)), y]
    negd = -d
    nll = np.logaddexp.reduce(negd, axis=1) - negd[np.arange(len(y)), y]
    norm_loss = normalized_nll(d, y, w)
    jm = margin_criterion(p, c, y, class_weights, sample_weights)
    result.update(accuracy=float(np.mean(pred['prediction'] == y)),
                  error_rate=float(np.mean(pred['prediction'] != y)),
                  balanced_accuracy=float(sum(class_weights_vector(k, class_weights)[v]*recalls[v] for v in range(k) if np.isfinite(recalls[v]))),
                  balanced_error=float(w @ (pred['prediction'] != y)),
                  confusion=confusion.tolist(),
                  per_class_recall=[float(r) if np.isfinite(r) else None for r in recalls],
                  per_class_counts=confusion.sum(axis=1).tolist(),
                  mean_true_margin=float(w @ true_margin),
                  fraction_nonpositive_true_margin=float(w @ (true_margin <= 0)),
                  within_class_jsd=float(w @ pred['jsd'][np.arange(len(y)), y]),
                  within_class_kl=float(w @ own) if np.isfinite(own).all() else None,
                  R=float(w @ own) if np.isfinite(own).all() else None,
                  NLL=float(w @ nll) if np.isfinite(nll).all() else None,
                  NLL_norm=norm_loss if np.isfinite(norm_loss) else None,
                  J_margin=jm if np.isfinite(jm) else None,
                  margin_softplus_0_03=soft_margin_loss(pred['similarity'], y, 0.03, w),
                  margin_softplus_0_08=soft_margin_loss(pred['similarity'], y, 0.08, w),
                  margin_softplus_0_15=soft_margin_loss(pred['similarity'], y, 0.15, w))
    pred['true_margin'] = true_margin
    return result, pred
