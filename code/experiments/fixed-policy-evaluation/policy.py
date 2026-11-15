"""Exact probability-space Guided/Uniform mixing for the Chapter 4 E-CH4-05 design.

This module computes masked transition probabilities, NOT logits, and does
not modify the existing ExactMaskedWalk engine. Engine integration and
Train/Validation evaluation must be done as a separately registered run.
Only the registered 5-feature generator is supported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "guided_policy.json"


def load_policy():
    with CONFIG_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_over_allowed(weights, valid):
    """Return a probability vector over valid feature choices only.

    valid must reflect BOTH observed and currently traversable features.
    """
    w = tuple(float(v) for v in weights)
    m = tuple(bool(v) for v in valid)
    if len(w) != 5 or len(m) != 5:
        raise ValueError("Expected five feature weights and five valid flags")
    if any(not math.isfinite(v) or v < 0 for v in w):
        raise ValueError("Weights must be finite and nonnegative")
    den = sum(value for value, allowed in zip(w, m) if allowed)
    if den <= 0:
        raise ValueError("No positive-weight allowed feature; cannot normalize")
    return tuple(value / den if allowed else 0.0 for value, allowed in zip(w, m))


def guided_probabilities(valid, previous_feature=None):
    """Guided transition on the current sample's valid feature support.

    previous_feature=None means initial policy (last row of W).
    """
    cfg = load_policy()
    row = cfg["start_row_index"] if previous_feature is None else previous_feature
    if not isinstance(row, int) or not 0 <= row <= 5:
        raise ValueError("Previous feature must be 0..4, or None for start")
    if row == 5 and previous_feature is not None:
        raise ValueError("Start row must be selected with previous_feature=None")
    return normalize_over_allowed(cfg["transition_weights"][row], valid)


def uniform_probabilities(valid):
    """Uniform over the currently allowed, observed/traversable features."""
    return normalize_over_allowed((1.0,) * 5, valid)


def mixed_probabilities(valid, alpha, previous_feature=None):
    """(1-alpha) Uniform(valid) + alpha Guided(valid), after each masks.

    When alpha=0, only the uniform policy is required. For alpha>0,
    at least one of features {0,1,3} must be a valid destination.
    """
    a = float(alpha)
    if not math.isfinite(a) or not 0 <= a <= 1:
        raise ValueError("alpha must be between 0 and 1 inclusive")
    u = uniform_probabilities(valid)
    if a == 0:
        return u
    g = guided_probabilities(valid, previous_feature)
    if a == 1:
        return g
    return tuple((1-a)*p_u + a*p_g for p_u,p_g in zip(u,g))


def transition_matrix(valid, alpha=1.0):
    """Six masked rows: previous-feature rows 0..4 and the start row at 5."""
    rows = [mixed_probabilities(valid, alpha, a) for a in range(5)]
    rows.append(mixed_probabilities(valid, alpha, None))
    return tuple(rows)


if __name__ == "__main__":
    cfg = load_policy()
    print(json.dumps({
        "policy_id": cfg["policy_id"],
        "weights": cfg["transition_weights"],
        "all_features_present_guided": guided_probabilities((1,1,1,1,1)),
        "alpha_grid": {
            str(a): mixed_probabilities((1,1,1,1,1), a)
            for a in cfg["alpha_grid"]
        },
        "evaluated_accuracy": False,
    }, indent=2))
