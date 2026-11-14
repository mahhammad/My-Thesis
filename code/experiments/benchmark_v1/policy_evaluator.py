"""Unified nonparametric policy evaluation (never trains, never reads Test).

Public API: evaluate_policy(dataset, policy, gamma, beta=.4, lam=.8, horizon=12)
Input dataset: {"reference_x", "reference_y", optional "query_x"}
Input policy: {"family": shared|class|node|node_class|class_tied_node,
               "logits": tensor or array}
For class_tied_node, logits is {"start_by_class": (K,d),
                                 "intermediate_by_node_class": (K,n,d,d)}.
For free node/node_class, optional "query_start_logits" supplies the explicit
new-query policy; without it Validation remains UNDEFINED.

Output contains distributions for ALL class hypotheses, reference centers
and explicit Validation status; no ground-truth query labels are required.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from engine import DT, ExactMaskedWalk

FAMILIES = ("shared", "class", "node", "node_class", "class_tied_node")


@dataclass
class PolicyEvaluation:
    train_by_hypothesis: np.ndarray
    validation_by_hypothesis: np.ndarray
    centers: np.ndarray
    validation_status: str
    effective_family: str
    effective_logits: torch.Tensor
    query_start_logits: torch.Tensor | None


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=DT)
    return torch.as_tensor(value, dtype=DT)


def normalize_policy(policy, graph):
    """Normalize policy representation, preserve class-tied starts exactly."""
    family = policy["family"]
    if family not in FAMILIES:
        raise ValueError(f"Unknown policy family: {family}")
    start = None
    if family == "class_tied_node":
        blocks = policy["logits"]
        a = _tensor(blocks["start_by_class"])
        b = _tensor(blocks["intermediate_by_node_class"])
        if tuple(a.shape) != (graph.k, graph.d):
            raise ValueError("start_by_class must be (K,d)")
        if tuple(b.shape) != (graph.k, graph.n, graph.d, graph.d):
            raise ValueError("intermediate_by_node_class must be (K,n,d,d)")
        logits = torch.cat(
            [b, a[:, None, None, :].expand(graph.k, graph.n, 1, graph.d)],
            dim=2
        )
        effective_family = "node_class"
        if graph.qx is not None:
            start = a[:, None, :].expand(graph.k, len(graph.qx), graph.d)
    else:
        logits = _tensor(policy["logits"])
        effective_family = family
        if policy.get("query_start_logits") is not None:
            start = _tensor(policy["query_start_logits"])
    graph.check_logits(logits, effective_family)
    if start is not None and graph.qx is not None:
        count = len(graph.qx)
        shape = ((count, graph.d) if effective_family == "node" else
                 (graph.k, count, graph.d))
        if tuple(start.shape) != shape and effective_family in ("node", "node_class"):
            raise ValueError(f"query_start_logits must be {shape}")
    return effective_family, logits, start


def _validate_probabilities(name, a, expected):
    a = np.asarray(a, np.float64)
    if a.shape != tuple(expected):
        raise ValueError(f"{name}: expected {expected}, got {a.shape}")
    if not np.isfinite(a).all() or np.any(a < -1e-9):
        raise ValueError(f"{name}: invalid probability")
    if not np.allclose(a.sum(-1), 1, atol=1e-8, rtol=0):
        raise ValueError(f"{name}: mass is not one")
    return a


def evaluate_graph(graph, policy):
    """Compute all hypotheses without using true query labels."""
    family, logits, starts = normalize_policy(policy, graph)
    n, k = graph.n, graph.k
    with torch.no_grad():
        if family in ("shared", "node"):
            common = graph.represent_train(logits, family).detach().cpu().numpy()
            train = np.broadcast_to(common[:, None, :], (n, k, k)).copy()
        else:
            train = np.stack([
                graph.represent_train(logits, family, c).detach().cpu().numpy()
                for c in range(k)], axis=1)
        train = _validate_probabilities("train_by_hypothesis", train, (n, k, k))
        labels = graph.labels.cpu().numpy()
        if not np.array_equal(np.unique(labels), np.arange(k)):
            raise ValueError("All classes must appear in reference data")
        centers = np.stack([train[labels == c, c].mean(0) for c in range(k)])
        centers = _validate_probabilities("centers", centers, (k, k))
        m = 0 if graph.qx is None else len(graph.qx)
        defined = m > 0 and (family in ("shared", "class") or starts is not None)
        if defined:
            if family in ("shared", "node"):
                v = graph.represent_query(
                    logits, family, query_start_logits=starts
                ).detach().cpu().numpy()
                val = np.broadcast_to(v[:, None, :], (m, k, k)).copy()
            else:
                val = np.stack([
                    graph.represent_query(
                        logits, family, c, query_start_logits=starts
                    ).detach().cpu().numpy()
                    for c in range(k)], axis=1)
            val = _validate_probabilities("validation_by_hypothesis", val, (m, k, k))
            status = "available"
        else:
            val = np.empty((0, k, k), dtype=np.float64)
            status = "no_queries" if not m else "not_defined_without_query_start"
    return PolicyEvaluation(train, val, centers, status, family, logits, starts)


def evaluate_policy(dataset, policy, *, gamma, beta=.4, lam=.8, horizon=12):
    """General policy function: dataset + family/parameters -> label distributions.

    The dataset's reference_y is used only to construct the class centers and
    label the destination nodes; a query's true label is never an input.
    """
    graph = ExactMaskedWalk(
        np.asarray(dataset["reference_x"], dtype=float),
        np.asarray(dataset["reference_y"], dtype=int),
        gamma, beta=beta, lam=lam, horizon=horizon,
        query_x=(None if dataset.get("query_x") is None else
                 np.asarray(dataset["query_x"], dtype=float))
    )
    return evaluate_graph(graph, policy)
