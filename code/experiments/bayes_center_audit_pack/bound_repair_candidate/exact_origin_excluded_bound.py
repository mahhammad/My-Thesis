"""Candidate Bellman outer bound for the *original origin-excluded* walk.

This file deliberately does not use ``origin_return_relaxed_bounds`` and never
substitutes the origin-return-allowed base kernel for the true kernel.

Source contracts (immutable):
  code/experiments/archive/multistep_policy_target_match/run.py::wgraph,
      ExactGraph.__init__, ExactGraph.center_train
  code/experiments/archive/bayes_center_inverse/feasibility_bound.py::
      bounds_for_all_origins, center_bounds
  chapter3@b60bab9f30453e9b5634793135e7b0251bfcf9d0

No data generation, no optimizer, no model evaluation, no CLI that can launch
full300 or a parameter sweep. Intended for small fixtures and independent QA.
"""
from __future__ import annotations

import numpy as np

BOUND_KERNEL_VERSION = "exact_origin_excluded_low_memory_candidate_v2_direct_numerator"
_MASK_TOLERANCE = 1e-12  # matches the historical ExactGraph.valid implementation


def shared_feature_kernel(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Construct only B[b,i,j] -- O(d*n*n) storage, identical to wgraph.

    B excludes the *current* sample from the destination. Excluding the
    *initial origin* is applied per-origin later, not by modifying B.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or gamma.shape != (x.shape[1],):
        raise ValueError("expected n>=3 samples, d features, and d gamma values")
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
        raise ValueError("gamma must be finite and strictly positive")
    n, d = x.shape
    present = np.isfinite(x)
    if np.any(present.sum(axis=0) < 3):
        raise ValueError("Chapter-3 non-dead-end precondition: >=3 present per feature")
    if not np.all(present.any(axis=1)):
        raise ValueError("each origin must have at least one observed feature")
    base = np.empty((d, n, n), dtype=np.float64)
    for b in range(d):
        good = present[:, b]
        values = np.where(good, x[:, b], 0.0)
        weights = np.exp(-gamma[b] * (values[:, None] - values[None, :]) ** 2)
        weights *= good[:, None] & good[None, :]
        np.fill_diagonal(weights, 0.0)
        denom = weights.sum(axis=1, keepdims=True)
        np.divide(weights, denom, out=weights, where=denom > 0)
        if np.any(good & (weights.sum(axis=1) <= _MASK_TOLERANCE)):
            raise ValueError("numerically degenerate base row (underflow / no destinations)")
        base[b] = weights
    return base


def _excluded_denominator(base: np.ndarray, origin: int) -> np.ndarray:
    """Return sum_{j != origin} B[b,i,j] in [i,b] layout.

    Two slices avoid catastrophic cancellation when B[b,i,origin] is near 1.
    They are views; neither a fourth-order transition tensor nor a per-origin
    full feature kernel is allocated.
    """
    d, n, _ = base.shape
    denom = np.empty((n, d), dtype=np.float64)
    for b in range(d):
        denom[:, b] = (base[b, :, :origin].sum(axis=1)
                        + base[b, :, origin + 1:].sum(axis=1))
    return denom


def _continuation(base: np.ndarray, value: np.ndarray, origin: int,
                  den: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """E_T[value(j,b) | i,b], with actual renormalized q_b^(origin).

    Numerator: sum_{j != origin} B[b,i,j] f[j,b] -- computed DIRECTLY.
    Divisor: sum_{j != origin} B[b,i,j], also directly slice-summed.

    Do not form the full dot product and subtract the origin contribution:
    that calculation catastrophically cancels when B[b,i,origin] is near 1.
    A temporary n*d value copy adds O(nd) space, not a 4-D kernel.
    """
    if not np.all(np.isfinite(value)):
        raise ArithmeticError("nonfinite Bellman continuation input")
    if not np.all(np.isfinite(den)) or np.any(den[legal] <= _MASK_TOLERANCE):
        raise ArithmeticError("invalid origin-excluded denominator or mask")
    value_excluded = value.copy()
    value_excluded[origin, :] = 0.0
    numerator = np.einsum("bij,jb->ib", base, value_excluded, optimize=True)
    if not np.all(np.isfinite(numerator)):
        raise ArithmeticError("nonfinite origin-excluded numerator")
    result = np.zeros_like(numerator)
    np.divide(numerator, den, out=result, where=legal)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("nonfinite origin-excluded continuation")
    return result


def exact_origin_excluded_center_bounds(
    x: np.ndarray,
    labels: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    *,
    return_per_origin: bool = False,
) -> dict:
    """Coordinatewise valid outer bounds for fixed masked Z policies.

    Each first hop uses B[b,origin,j]. Each later hop uses the original
    q_b^(origin): no self destination, no origin destination, and the *other*
    destinations renormalized. For each (origin,class-coordinate,time,state)
    the Bellman controller can choose any legal b; thus it contains the fixed
    class-conditional Z policy, whose masked action is a convex combination.

    Shape: shared kernel O(d*n^2), origin local work O(n*d), returned output
    O(K^2) or optional O(n*K). Runtime O(K*H*d*n^3) dense operations in the
    n-origin full reference case. No per-origin 4-D transition is materialized.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    alpha = np.asarray(alpha, dtype=np.float64)
    if labels.shape != (len(x),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a length-n integer vector")
    if np.any(labels < 0) or not np.array_equal(np.unique(labels), np.arange(labels.max() + 1)):
        raise ValueError("nonempty contiguous classes 0..K-1 are required")
    if alpha.ndim != 1 or len(alpha) < 1 or not np.all(np.isfinite(alpha)):
        raise ValueError("invalid finite-horizon alpha coefficients")
    if np.any(alpha < 0) or not np.isclose(alpha.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("alpha must be nonnegative and sum to 1")

    n, d = x.shape
    k = int(labels.max()) + 1
    base = shared_feature_kernel(x, gamma)
    present = np.isfinite(x)
    low_per_origin = np.empty((n, k), dtype=np.float64)
    high_per_origin = np.empty((n, k), dtype=np.float64)
    H = len(alpha)

    for origin in range(n):
        den = _excluded_denominator(base, origin)
        legal = den > _MASK_TOLERANCE
        # Full3+ feature assumption ensures this for every observed state.
        # If violated due to numerical underflow, refuse to certify a bound.
        if H > 1 and np.any(~legal.any(axis=1)):
            raise ValueError("state without a legal post-first-hop feature")
        first_allowed = present[origin] & (base[:, origin, :].sum(axis=1) > _MASK_TOLERANCE)
        if not first_allowed.any() or np.any(present[origin] & ~first_allowed):
            raise ValueError("initial policy can choose a feature without a legal destination")

        for coordinate in range(k):
            reward = (labels == coordinate).astype(np.float64)
            value_low = np.broadcast_to(alpha[-1] * reward[:, None], (n, d)).copy()
            value_high = value_low.copy()
            for t in range(H - 2, -1, -1):
                cont_low = _continuation(base, value_low, origin, den, legal)
                cont_high = _continuation(base, value_high, origin, den, legal)
                min_action = np.where(legal, cont_low, np.inf).min(axis=1)
                max_action = np.where(legal, cont_high, -np.inf).max(axis=1)
                # The original model never visits the initial origin after
                # the first hop, so the value of this unreachable state is
                # immaterial. Keep it finite as in bounds_for_all_origins.
                if H > 1:
                    min_action[origin] = 0.0
                    max_action[origin] = 0.0
                value_low = np.broadcast_to((alpha[t] * reward + min_action)[:, None], (n, d)).copy()
                value_high = np.broadcast_to((alpha[t] * reward + max_action)[:, None], (n, d)).copy()
            # B is correct for the first hop: current=origin, diagonal zero.
            first_low = np.einsum("bj,jb->b", base[:, origin, :], value_low, optimize=True)
            first_high = np.einsum("bj,jb->b", base[:, origin, :], value_high, optimize=True)
            low_per_origin[origin, coordinate] = first_low[first_allowed].min()
            high_per_origin[origin, coordinate] = first_high[first_allowed].max()

    lower = np.stack([low_per_origin[labels == c].mean(axis=0) for c in range(k)])
    upper = np.stack([high_per_origin[labels == c].mean(axis=0) for c in range(k)])
    # Guard both per-origin and aggregated probabilities: do not silently clip.
    # A small tolerance accommodates roundoff; it is NOT a certified error envelope.
    for name, lo, hi in (("per-origin", low_per_origin, high_per_origin),
                         ("class-center", lower, upper)):
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ArithmeticError(f"nonfinite {name} Bellman interval")
        if np.any(lo > hi + 1e-10):
            raise ArithmeticError(f"invalid {name} Bellman outer interval")
        if np.any(lo < -1e-10) or np.any(hi < -1e-10) or np.any(lo > 1 + 1e-10) or np.any(hi > 1 + 1e-10):
            raise ArithmeticError(f"out-of-range {name} Bellman probability")
    result = {
        "kernel_version": BOUND_KERNEL_VERSION,
        "kernel_contract": "exact_origin_excluded_and_renormalized_after_first_hop",
        "origin_return_within_walk": "forbidden",
        "first_hop": "original_B_with_zero_self_transition",
        "policy_relaxation": "origin_time_state_entry_feature_and_target_coordinate_specific",
        "H": H,
        "alpha": alpha.tolist(),
        "n_reference": n,
        "n_features": d,
        "n_classes": k,
        "lower_center_probabilities": lower.tolist(),
        "upper_center_probabilities": upper.tolist(),
        "diagonal_lower": np.diag(lower).tolist(),
        "diagonal_upper": np.diag(upper).tolist(),
        "old_return_allowed_bounds_reused": False,
    }
    if return_per_origin:
        result["lower_per_origin"] = low_per_origin.tolist()
        result["upper_per_origin"] = high_per_origin.tolist()
    return result
