"""NumPy-only objectives and complete Train/Validation metric calculations.

Exact score 1-JSD/ln(2); Sigmoid-JS and Softplus-JS match §4.13.
Class centers always use Train; Validation never influences gradients.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code/policy_evaluator"))
from metrics import center_metrics
from numpy_engine import LOG2, jsd, js_score_and_partials

TAU = .005
OBJECTIVES = ("sigmoid_JS_margin_tau005", "softplus_JS_tau005")


def balanced_mean(values, labels, k=3):
    return float(np.mean([np.mean(values[labels == c]) for c in range(k)]))


def classification(scores, labels, k=3):
    """Predictions are argmax JS similarity. Undefined per-class precision = 0."""
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(scores).argmax(axis=1)
    c = np.zeros((k, k), dtype=np.int64)
    np.add.at(c, (y, pred), 1)
    correct = np.diag(c).astype(np.float64)
    recall = correct / np.maximum(c.sum(axis=1), 1)
    precision = correct / np.maximum(c.sum(axis=0), 1)
    return {"accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(recall.mean()),
            "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()),
            "precision_by_class": precision.tolist(),
            "recall_by_class": recall.tolist()}


def forward_model(graph, logits, family, *, record=False):
    k, n = graph.k, graph.n
    if family == "shared":
        rep, context = graph.forward_train(logits, family, record=record)
        per_hypothesis = np.broadcast_to(rep[:, None, :], (n, k, k))
        contexts = [context]
    elif family == "class":
        results = [graph.forward_train(logits, family, c, record=record)
                   for c in range(k)]
        per_hypothesis = np.stack([r[0] for r in results], axis=1)
        contexts = [r[1] for r in results]
    else:
        raise ValueError("Unknown policy family")
    labels = graph.labels
    centers = np.stack([per_hypothesis[labels == c, c, :].mean(axis=0)
                        for c in range(k)])
    scores, ds_dr, ds_dc = js_score_and_partials(
        per_hypothesis, centers[None, :, :])
    return per_hypothesis, centers, scores, ds_dr, ds_dc, contexts


def objective_and_score_derivative(scores, y, objective, tau=TAU):
    if objective not in OBJECTIVES:
        raise ValueError("Unknown Chapter 4.13 objective")
    n, k = scores.shape
    counts = np.bincount(y, minlength=k)
    if np.any(counts == 0):
        raise ValueError("Every Train class required")
    weight = 1. / (k * counts[y])
    index = np.arange(n)
    true = scores[index, y]
    score_gradient = np.zeros_like(scores)
    if objective.startswith("sigmoid"):
        rival = np.array(scores, copy=True)
        rival[index, y] = -np.inf
        closest = rival.argmax(axis=1)
        margin = true - rival[index, closest]
        sigmoid_loss = np.exp(-np.logaddexp(0., margin / tau))
        objective_value = float(weight @ sigmoid_loss)
        dmargin = -(weight / tau) * sigmoid_loss * (1. - sigmoid_loss)
        score_gradient[index, y] += dmargin
        score_gradient[index, closest] -= dmargin
    else:
        scaled = (scores - true[:, None]) / tau
        largest = scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled - largest)
        probs = exp / exp.sum(axis=1, keepdims=True)
        objective_value = float(weight @ (
            tau * (largest[:, 0] + np.log(exp.sum(axis=1)))))
        score_gradient[:] = weight[:, None] * probs
        score_gradient[index, y] -= weight
    return objective_value, score_gradient


def train_loss_and_grad(graph, logits, family, objective):
    by, centers, scores, d_rep, d_center, contexts = forward_model(
        graph, logits, family, record=True)
    loss, ds = objective_and_score_derivative(scores, graph.labels, objective)
    y, k = graph.labels, graph.k
    target_by_class = []
    for c in range(k):
        upstream = ds[:, c, None] * d_rep[:, c, :]
        cgrad = (ds[:, c, None] * d_center[:, c, :]).sum(axis=0)
        upstream[y == c] += cgrad / np.count_nonzero(y == c)
        target_by_class.append(upstream)
    if family == "shared":
        grad = graph.backward_train(
            contexts[0], np.sum(target_by_class, axis=0))
    else:
        grad = np.stack([graph.backward_train(contexts[c], target_by_class[c])
                         for c in range(k)])
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise FloatingPointError("Nonfinite exact-walk objective/gradient")
    return loss, grad, by, centers, scores


def _safe(x):
    if x is None:
        return None
    y = float(x)
    return y if np.isfinite(y) else None


def _kl(p, q):
    bad = np.any((p > 0) & (q <= 0), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(p > 0, p * (
            np.log(np.maximum(p, 1e-300)) -
            np.log(np.maximum(q, 1e-300))), 0).sum(axis=-1)
    return np.where(bad, np.inf, val)


def train_diagnostics(graph, reps, centers, scores):
    y = graph.labels
    n, k = scores.shape
    own = reps[np.arange(n), y, :]
    dist_js = jsd(own, centers[y])
    dist_sqrt = np.sqrt(dist_js)
    dist_kl = _kl(own, centers[y])
    geo = center_metrics(centers)
    result = classification(scores, y, k)
    true = scores[np.arange(n), y]
    competitor = np.array(scores, copy=True)
    competitor[np.arange(n), y] = -np.inf
    result.update({
        "density_r_js": balanced_mean(dist_js, y, k),
        "density_r_sqrt_js": balanced_mean(dist_sqrt, y, k),
        "density_r_kl": _safe(balanced_mean(dist_kl, y, k)),
        "separation_i": _safe(geo["I"]),
        "separation_t": _safe(geo["T"]),
        "separation_bt": _safe(geo["BT"]),
        "separation_tnl": _safe(geo["TNL"]),
        "separation_btnl": _safe(geo["BTNL"]),
        "margin_mean": float(np.mean(true - competitor.max(axis=1))),
        "r_by_class": [{
            "R_JS": float(np.mean(dist_js[y == c])),
            "R_sqrt_JS": float(np.mean(dist_sqrt[y == c])),
            "R_KL": _safe(np.mean(dist_kl[y == c]))
        } for c in range(k)]
    })
    return result


def validation_diagnostics(graph, logits, family, centers, validation_labels):
    if graph.qx is None:
        raise ValueError("Query features required")
    k, m = graph.k, len(graph.qx)
    if family == "shared":
        rep = graph.forward_query(logits, family)
        by = np.broadcast_to(rep[:, None, :], (m, k, k))
    else:
        by = np.stack([graph.forward_query(logits, family, c)
                       for c in range(k)], axis=1)
    scores = 1. - jsd(by, centers[None, :, :]) / LOG2
    return classification(scores, validation_labels, k)
