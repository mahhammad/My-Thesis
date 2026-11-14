"""Optional specialized report owned by the center-recovery experiment.

Use from the experiment's own run script, after save_snapshot(...):
    from experiment_center_target import save_center_target_report
    save_center_target_report(run_dir)

This module is NOT imported or called by the universal score reporter.
It does not recompute walks, retrain policy, or use any labels/Test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from snapshot_pipeline import load_snapshot, _atomic_bytes


def center_target_metrics(centers, target_centers):
    observed = np.asarray(centers, dtype=np.float64)
    target = np.asarray(target_centers, dtype=np.float64)
    if observed.shape != target.shape or target.ndim != 2:
        raise ValueError("Observed and target center shapes differ")
    if not np.isfinite(target).all() or np.min(target) < -1e-9 or not np.allclose(
        target.sum(axis=1), 1, rtol=0, atol=1e-8
    ):
        raise ValueError("Target centers must be valid distributions")
    m = 0.5 * (observed + target)
    def entropy(v):
        return -np.sum(np.where(v > 0, v*np.log(np.maximum(v, 1e-300)), 0.0), axis=-1)
    distances = np.maximum(0.0, entropy(m) - .5*(entropy(observed)+entropy(target)))
    diff = observed - target
    return {
        "per_class_JS": distances.tolist(),
        "mean_center_JS": float(distances.mean()),
        "max_center_JS": float(distances.max()),
        "mean_center_sqrt_JS": float(np.sqrt(distances).mean()),
        "frobenius_distance": float(np.linalg.norm(diff)),
        "max_abs_residual": float(np.max(np.abs(diff))),
        "mean_abs_residual": float(np.mean(np.abs(diff)))
    }


def save_center_target_report(run_dir):
    """Specialized experiment artifact: only target is read from model metadata."""
    run_dir = Path(run_dir)
    model, arrays = load_snapshot(run_dir)
    target = model.get("target", {})
    if target.get("centers") is None:
        raise ValueError("This experiment has no target centers in model.json")
    summary = {
        "schema": "imc-experiment-center-target-v1",
        "experiment": "center_recovery",
        "target_source": target.get("source"),
        "result": center_target_metrics(arrays["centers"], target["centers"])
    }
    _atomic_bytes(run_dir / "experiment_report.json",
                  (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
                   + "\n").encode("utf8"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    res = save_center_target_report(args.run_dir)
    print("Mean center JSD:", res["result"]["mean_center_JS"])
    print("Saved:", args.run_dir / "experiment_report.json")
