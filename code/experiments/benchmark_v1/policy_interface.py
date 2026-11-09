"""One public policy interface shared by all IMC experimental families.

The experiment supplies: dataset, policy family, policy parameters, walk
parameters. This module evaluates stored policies, never trains them.
All K candidate-class hypotheses are evaluated without using a query label.

Supported:
  shared, class, node, node_class (legacy tensors)
  class_tied_node (one class-specific start row, node/class-specific middles)
A new family only needs an adapter here, not changes to the metric reporter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from engine import DT, ExactMaskedWalk


@dataclass(frozen=True)
class PolicyAdapter:
    """Resolved effective policy; not a fitted model or a predictor of labels."""
    family: str
    variant: str
    logits: torch.Tensor
    query_start_logits: torch.Tensor | None = None
    query_start_source: str | None = None


def create_policy(family, parameters, *, n, k, d, n_query=0):
    """Normalize diverse family parameters to one engine-compatible policy.

    For ordinary families parameters can be a raw tensor or a mapping
    with logits and optional query_start_logits. Free node query starts are
    not inferred silently. Hybrid class_tied_node expects distinct tensors.
    """
    if family == "class_tied_node":
        if not isinstance(parameters, dict):
            raise TypeError("class_tied_node requires a parameter dictionary")
        start = torch.as_tensor(parameters["start_by_class"], dtype=DT)
        middle = torch.as_tensor(parameters["intermediate_by_node_class"], dtype=DT)
        if tuple(start.shape) != (k, d) or tuple(middle.shape) != (k, n, d, d):
            raise ValueError("Hybrid start/middle parameter shapes mismatch")
        effective = torch.cat(
            (middle, start[:, None, None, :].expand(k, n, 1, d)), dim=2
        )
        qstart = (start[:, None, :].expand(k, n_query, d).contiguous()
                  if n_query else None)
        return PolicyAdapter(
            "node_class", family, effective, qstart,
            "class_tied_start_by_candidate_hypothesis"
        )

    shape = {
        "shared": (d + 1, d),
        "class": (k, d + 1, d),
        "node": (n, d + 1, d),
        "node_class": (k, n, d + 1, d)
    }.get(family)
    if shape is None:
        raise ValueError(f"Unknown policy family: {family}")
    raw = parameters.get("logits") if isinstance(parameters, dict) else parameters
    if raw is None:
        raise ValueError("Missing policy logits")
    logits = torch.as_tensor(raw, dtype=DT)
    if tuple(logits.shape) != shape or not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{family}: invalid logits shape/values, expected {shape}")
    query_starts = None
    query_source = None
    if isinstance(parameters, dict) and parameters.get("query_start_logits") is not None:
        if family not in ("node", "node_class"):
            raise ValueError("External query starts only apply to free node policies")
        query_starts = torch.as_tensor(parameters["query_start_logits"], dtype=DT)
        expected = (n_query, d) if family == "node" else (k, n_query, d)
        if tuple(query_starts.shape) != expected:
            raise ValueError(f"query_start_logits must have shape {expected}")
        query_source = parameters.get("query_start_source")
        if not query_source:
            raise ValueError("Explicit query starts must declare their provenance")
    return PolicyAdapter(family, family, logits, query_starts, query_source)


def evaluate_policy(dataset, policy_family, policy_parameters, *,
                    gamma, beta=.4, lam=.8, horizon=12):
    """Generic model evaluation: policy + dataset -> distributions, no metrics.

    dataset = {
      ref_x: (n,d), ref_y: (n,), query_x: optional (m,d)
    }
    ref_y is used only to form the training class centers, never to select a
    hypothesis for scoring a query. No Test split is implicitly accessed.
    """
    x = np.asarray(dataset["ref_x"], dtype=float)
    y = np.asarray(dataset["ref_y"], dtype=int)
    q = (None if dataset.get("query_x") is None else
         np.asarray(dataset["query_x"], dtype=float))
    graph = ExactMaskedWalk(
        x, y, gamma, beta=beta, lam=lam, horizon=horizon, query_x=q
    )
    policy = create_policy(
        policy_family, policy_parameters, n=graph.n, k=graph.k, d=graph.d,
        n_query=0 if q is None else len(q)
    )
    # Local import avoids binding the data/reporting layers into engine.py.
    from snapshot_pipeline import collect_representations
    train, centers, queries, status = collect_representations(
        graph, policy.logits, policy.family,
        query_starts=policy.query_start_logits
    )
    return {
        "graph": graph,
        "policy": policy,
        "train_by_hypothesis": train,
        "validation_by_hypothesis": queries,
        "centers": centers,
        "validation_status": status
    }
